"""
process_manager.py — Launch, stop, pause, resume, and inspect QEMU VMs.

The :class:`VMProcess` façade keeps QEMU alive after SSH disconnect by
auto-detecting the best detachment method (``tmux`` > ``screen`` > POSIX
double-fork daemon).  All mutable state is persisted to a JSON *state file*
so that a later CLI invocation can reconnect to a running VM.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import VMConfig, check_port_available
from qemu_builder import build_qemu_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums / data structures
# ---------------------------------------------------------------------------

class VMState(str, Enum):
    """Possible states of a managed VM."""

    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    UNKNOWN = "unknown"


@dataclass
class VMStatus:
    """Snapshot of a VM's runtime status.

    Serialised to / from the JSON state file so subsequent CLI invocations
    can reconnect.
    """

    name: str = ""
    state: VMState = VMState.STOPPED
    pid: int = 0
    start_time: str = ""
    config_path: str = ""
    cpu_percent: float = 0.0
    ram_mb_used: float = 0.0
    uptime_seconds: float = 0.0
    detach_method: str = ""
    session_name: str = ""
    serial_socket: str = ""
    monitor_socket: str = ""
    qmp_socket: str = ""


# ---------------------------------------------------------------------------
# State-file helpers (atomic write)
# ---------------------------------------------------------------------------

def _state_file(state_dir: Path) -> Path:
    """Return the canonical path to the state JSON file.

    Args:
        state_dir: Base directory (e.g. ``~/.vms``).

    Returns:
        ``state_dir / "state.json"``.
    """
    return state_dir / "state.json"


def _read_state(state_dir: Path) -> Dict[str, Any]:
    """Read the full state dict from disk.

    Args:
        state_dir: Base directory.

    Returns:
        A dict keyed by VM name.
    """
    sf = _state_file(state_dir)
    if not sf.exists():
        return {}
    with sf.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_state(state_dir: Path, data: Dict[str, Any]) -> None:
    """Atomically write the state dict to disk.

    Writes to a temporary file first, then renames — guaranteeing that a
    crash mid-write never leaves a corrupted file.

    Args:
        state_dir: Base directory.
        data:      Full state dict.
    """
    sf = _state_file(state_dir)
    sf.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp, sf)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_vm_status(state_dir: Path, status: VMStatus) -> None:
    """Upsert a single VM status entry in the state file.

    Args:
        state_dir: Base directory.
        status:    The VM status to save.
    """
    data = _read_state(state_dir)
    data[status.name] = asdict(status)
    _write_state(state_dir, data)


def _load_vm_status(state_dir: Path, name: str) -> Optional[VMStatus]:
    """Load one VM's status from the state file.

    Args:
        state_dir: Base directory.
        name:      VM name.

    Returns:
        A ``VMStatus`` or ``None`` if the VM is not tracked.
    """
    data = _read_state(state_dir)
    entry = data.get(name)
    if entry is None:
        return None
    entry["state"] = VMState(entry.get("state", "stopped"))
    return VMStatus(**entry)


def _remove_vm_status(state_dir: Path, name: str) -> None:
    """Remove a VM from the state file.

    Args:
        state_dir: Base directory.
        name:      VM name.
    """
    data = _read_state(state_dir)
    data.pop(name, None)
    _write_state(state_dir, data)


# ---------------------------------------------------------------------------
# PID / proc helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return ``True`` if *pid* refers to a living process.

    Args:
        pid: Process ID.

    Returns:
        Whether the process is alive.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but we lack permission


def _read_proc_stat(pid: int) -> Dict[str, float]:
    """Read CPU and RSS from ``/proc/<pid>/stat`` and ``/proc/<pid>/statm``.

    Args:
        pid: Process ID.

    Returns:
        Dict with ``cpu_percent`` and ``ram_mb`` (best-effort; 0 on error).
    """
    result: Dict[str, float] = {"cpu_percent": 0.0, "ram_mb": 0.0}
    try:
        statm = Path(f"/proc/{pid}/statm").read_text()
        pages = int(statm.split()[1])  # resident pages
        result["ram_mb"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024, 1)
    except Exception:
        pass

    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.split()
        utime = int(fields[13])
        stime = int(fields[14])
        ticks = os.sysconf("SC_CLK_TCK")
        total_sec = (utime + stime) / ticks
        # Uptime of the process.
        start_ticks = int(fields[21])
        with open("/proc/uptime") as f:
            sys_up = float(f.read().split()[0])
        proc_up = sys_up - (start_ticks / ticks)
        if proc_up > 0:
            result["cpu_percent"] = round(100.0 * total_sec / proc_up, 1)
    except Exception:
        pass

    return result


# ---------------------------------------------------------------------------
# Detachment strategies
# ---------------------------------------------------------------------------

def _detect_detach_method() -> str:
    """Return the best available detach mechanism.

    Returns:
        ``"tmux"``, ``"screen"``, or ``"daemon"`` (POSIX double-fork).
    """
    if shutil.which("tmux"):
        return "tmux"
    if shutil.which("screen"):
        return "screen"
    return "daemon"


def _start_via_tmux(
    session: str, cmd: List[str], log_path: Path,
) -> int:
    """Launch *cmd* inside a detached ``tmux`` session.

    Args:
        session:  tmux session name.
        cmd:      The QEMU command list.
        log_path: Path to redirect output.

    Returns:
        PID of the QEMU process.
    """
    # Create a detached tmux session running the command.
    shell_safe = " ".join(_quote(c) for c in cmd)
    tmux_cmd = [
        "tmux", "new-session", "-d", "-s", session,
        "bash", "-c", f"{shell_safe} > {log_path} 2>&1",
    ]
    subprocess.Popen(
        tmux_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).wait()
    # Give QEMU a moment to start.
    time.sleep(0.5)
    return _get_tmux_pid(session)


def _get_tmux_pid(session: str) -> int:
    """Retrieve the PID of the process running inside a tmux session.

    Args:
        session: tmux session name.

    Returns:
        PID, or 0 on failure.
    """
    try:
        out = subprocess.check_output(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return 0


def _start_via_screen(
    session: str, cmd: List[str], log_path: Path,
) -> int:
    """Launch *cmd* inside a detached ``screen`` session.

    Args:
        session:  screen session name.
        cmd:      The QEMU command list.
        log_path: Path to redirect output.

    Returns:
        PID of the QEMU process (best-effort).
    """
    shell_safe = " ".join(_quote(c) for c in cmd)
    screen_cmd = [
        "screen", "-dmS", session,
        "bash", "-c", f"{shell_safe} > {log_path} 2>&1",
    ]
    subprocess.Popen(
        screen_cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).wait()
    time.sleep(0.5)
    return _get_screen_pid(session)


def _get_screen_pid(session: str) -> int:
    """Best-effort PID lookup for a screen session.

    Args:
        session: screen session name.

    Returns:
        PID, or 0.
    """
    try:
        out = subprocess.check_output(
            ["screen", "-ls"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        for line in out.splitlines():
            if session in line:
                pid_str = line.strip().split(".")[0]
                return int(pid_str)
    except Exception:
        pass
    return 0


def _start_via_daemon(
    cmd: List[str], log_path: Path,
) -> int:
    """Launch *cmd* via POSIX double-fork to detach from the terminal.

    Args:
        cmd:      The QEMU command list.
        log_path: Path for stdout/stderr.

    Returns:
        PID of the child QEMU process.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("a")

    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log_fh.close()
    return proc.pid


def _quote(s: str) -> str:
    """Shell-quote a string for embedding in a tmux/screen command.

    Args:
        s: Raw string.

    Returns:
        Quoted string.
    """
    if " " in s or "," in s or ";" in s or "'" in s:
        return "'" + s.replace("'", "'\\''") + "'"
    return s


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class VMProcess:
    """High-level façade for the full VM lifecycle.

    All methods are static / class-level — they operate on the shared state
    file rather than holding long-lived process handles.
    """

    # ---- Start -----------------------------------------------------------

    @staticmethod
    def start(
        config: VMConfig,
        state_dir: Path,
        *,
        timeout: float = 10.0,
    ) -> VMStatus:
        """Launch a QEMU VM in the background.

        The best available detach method is auto-detected.  Port conflicts
        are checked before launch.

        Args:
            config:    Validated VM configuration.
            state_dir: Directory for state file, logs, sockets.
            timeout:   Seconds to wait for QEMU to start.

        Returns:
            Updated ``VMStatus``.

        Raises:
            RuntimeError: If the VM is already running or launch fails.
        """
        existing = _load_vm_status(state_dir, config.name)
        if existing and existing.state in (VMState.RUNNING, VMState.PAUSED):
            if _pid_alive(existing.pid):
                raise RuntimeError(
                    f"VM '{config.name}' is already {existing.state.value} "
                    f"(pid {existing.pid})."
                )

        # Check port availability.
        for rule in config.network.hostfwd:
            try:
                port = int(rule.split(":")[2].split("-")[0])
                if not check_port_available(port):
                    logger.warning("Port %d is already in use (rule: %s)", port, rule)
            except (IndexError, ValueError):
                pass

        # Build command.
        qemu_cmd = build_qemu_command(config, state_dir=state_dir)

        # Log file.
        log_dir = state_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{config.name}.log"

        # Detach.
        method = _detect_detach_method()
        session_name = f"vm-{config.name}"

        if method == "tmux":
            pid = _start_via_tmux(session_name, qemu_cmd, log_path)
        elif method == "screen":
            pid = _start_via_screen(session_name, qemu_cmd, log_path)
        else:
            pid = _start_via_daemon(qemu_cmd, log_path)

        if pid <= 0:
            raise RuntimeError(f"Failed to start VM '{config.name}' (pid=0).")

        # Wait for the process to become alive.
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if _pid_alive(pid):
                break
            time.sleep(0.2)
        else:
            raise RuntimeError(
                f"VM '{config.name}' did not start within {timeout}s."
            )

        # Sockets.
        sock_dir = state_dir / "sockets"
        serial_sock = str(sock_dir / f"{config.name}-serial.sock")
        monitor_sock = str(sock_dir / f"{config.name}-monitor.sock")
        qmp_sock = str(sock_dir / f"{config.name}-qmp.sock")

        status = VMStatus(
            name=config.name,
            state=VMState.RUNNING,
            pid=pid,
            start_time=datetime.now(timezone.utc).isoformat(),
            config_path=str(config.config_path or ""),
            detach_method=method,
            session_name=session_name,
            serial_socket=serial_sock,
            monitor_socket=monitor_sock,
            qmp_socket=qmp_sock,
        )
        _save_vm_status(state_dir, status)
        logger.info("Started VM '%s' (pid=%d, method=%s)", config.name, pid, method)
        return status

    # ---- Stop ------------------------------------------------------------

    @staticmethod
    def stop(
        name: str,
        state_dir: Path,
        *,
        graceful: bool = True,
        timeout: float = 30.0,
    ) -> None:
        """Stop a running VM.

        If *graceful* is ``True``, send ``SIGTERM`` first and wait up to
        *timeout* seconds before falling back to ``SIGKILL``.

        Args:
            name:      VM name.
            state_dir: State directory.
            graceful:  Whether to try SIGTERM first.
            timeout:   Seconds to wait for graceful shutdown.

        Raises:
            RuntimeError: If the VM is not tracked.
        """
        status = _load_vm_status(state_dir, name)
        if status is None:
            raise RuntimeError(f"VM '{name}' is not tracked.")

        pid = status.pid
        if not _pid_alive(pid):
            status.state = VMState.STOPPED
            _save_vm_status(state_dir, status)
            logger.info("VM '%s' was already dead.", name)
            return

        if graceful:
            os.kill(pid, signal.SIGTERM)
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                if not _pid_alive(pid):
                    break
                time.sleep(0.5)
            else:
                logger.warning(
                    "VM '%s' did not stop gracefully; sending SIGKILL.", name)
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
        else:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)

        status.state = VMState.STOPPED
        _save_vm_status(state_dir, status)

        # Clean tmux/screen session.
        _cleanup_session(status)

        logger.info("Stopped VM '%s'.", name)

    # ---- Pause / Resume --------------------------------------------------

    @staticmethod
    def pause(name: str, state_dir: Path) -> None:
        """Pause (suspend) a running VM via ``SIGSTOP``.

        Args:
            name:      VM name.
            state_dir: State directory.

        Raises:
            RuntimeError: If the VM is not running.
        """
        status = _load_vm_status(state_dir, name)
        if status is None or not _pid_alive(status.pid):
            raise RuntimeError(f"VM '{name}' is not running.")
        os.kill(status.pid, signal.SIGSTOP)
        status.state = VMState.PAUSED
        _save_vm_status(state_dir, status)
        logger.info("Paused VM '%s'.", name)

    @staticmethod
    def resume(name: str, state_dir: Path) -> None:
        """Resume a paused VM via ``SIGCONT``.

        Args:
            name:      VM name.
            state_dir: State directory.

        Raises:
            RuntimeError: If the VM is not paused.
        """
        status = _load_vm_status(state_dir, name)
        if status is None or not _pid_alive(status.pid):
            raise RuntimeError(f"VM '{name}' is not tracked or not alive.")
        os.kill(status.pid, signal.SIGCONT)
        status.state = VMState.RUNNING
        _save_vm_status(state_dir, status)
        logger.info("Resumed VM '%s'.", name)

    # ---- Restart ---------------------------------------------------------

    @staticmethod
    def restart(
        config: VMConfig,
        state_dir: Path,
        *,
        timeout: float = 30.0,
    ) -> VMStatus:
        """Stop then re-start a VM.

        Args:
            config:    VM configuration (must have the same *name*).
            state_dir: State directory.
            timeout:   Shutdown timeout.

        Returns:
            New ``VMStatus`` after restart.
        """
        try:
            VMProcess.stop(config.name, state_dir, timeout=timeout)
        except RuntimeError:
            pass
        return VMProcess.start(config, state_dir)

    # ---- Status ----------------------------------------------------------

    @staticmethod
    def get_status(name: str, state_dir: Path) -> VMStatus:
        """Return the current status of a VM, refreshing live metrics.

        Args:
            name:      VM name.
            state_dir: State directory.

        Returns:
            A ``VMStatus`` with up-to-date CPU/RAM/uptime.
        """
        status = _load_vm_status(state_dir, name)
        if status is None:
            return VMStatus(name=name, state=VMState.UNKNOWN)

        if _pid_alive(status.pid):
            proc_info = _read_proc_stat(status.pid)
            status.cpu_percent = proc_info["cpu_percent"]
            status.ram_mb_used = proc_info["ram_mb"]
            # Compute uptime.
            try:
                started = datetime.fromisoformat(status.start_time)
                status.uptime_seconds = (
                    datetime.now(timezone.utc) - started
                ).total_seconds()
            except Exception:
                pass
            # State may still be PAUSED if it was paused before.
            if status.state not in (VMState.RUNNING, VMState.PAUSED):
                status.state = VMState.RUNNING
        else:
            status.state = VMState.STOPPED

        _save_vm_status(state_dir, status)
        return status

    # ---- List ------------------------------------------------------------

    @staticmethod
    def list_vms(state_dir: Path) -> List[VMStatus]:
        """Return a list of all tracked VMs with refreshed status.

        Args:
            state_dir: State directory.

        Returns:
            List of ``VMStatus`` objects.
        """
        data = _read_state(state_dir)
        result: List[VMStatus] = []
        for name in data:
            result.append(VMProcess.get_status(name, state_dir))
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cleanup_session(status: VMStatus) -> None:
    """Kill leftover tmux / screen session after a VM stop.

    Args:
        status: The VM status with session info.
    """
    if status.detach_method == "tmux" and status.session_name:
        try:
            subprocess.Popen(
                ["tmux", "kill-session", "-t", status.session_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).wait()
        except Exception:
            pass
    elif status.detach_method == "screen" and status.session_name:
        try:
            subprocess.Popen(
                ["screen", "-S", status.session_name, "-X", "quit"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).wait()
        except Exception:
            pass
