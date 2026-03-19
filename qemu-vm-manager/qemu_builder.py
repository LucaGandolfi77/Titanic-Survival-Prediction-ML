"""
qemu_builder.py — Assemble a QEMU CLI command from a VMConfig.

The single public function :func:`build_qemu_command` translates the
high-level :class:`~config.VMConfig` dataclass into an argument list
suitable for :class:`subprocess.Popen`.  No paths are hard-coded: the
``qemu-system-<arch>`` binary is located via :func:`shutil.which`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from config import VMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QEMU binary lookup
# ---------------------------------------------------------------------------

class QemuNotFoundError(Exception):
    """Raised when the required ``qemu-system-*`` binary is not on PATH."""


_INSTALL_HINTS = {
    "debian": "sudo apt-get install qemu-system",
    "ubuntu": "sudo apt-get install qemu-system",
    "arch": "sudo pacman -S qemu-full",
    "alpine": "sudo apk add qemu-system-x86_64   # (adjust for your arch)",
}


def find_qemu_binary(arch: str) -> str:
    """Locate the ``qemu-system-<arch>`` executable.

    Args:
        arch: QEMU architecture string, e.g. ``"x86_64"``.

    Returns:
        Absolute path to the binary.

    Raises:
        QemuNotFoundError: If the binary is not found on ``$PATH``.
    """
    name = f"qemu-system-{arch}"
    path = shutil.which(name)
    if path is None:
        lines = [f"'{name}' not found on PATH.  Install QEMU:"]
        for distro, cmd in _INSTALL_HINTS.items():
            lines.append(f"  {distro:8s}: {cmd}")
        raise QemuNotFoundError("\n".join(lines))
    return path


# ---------------------------------------------------------------------------
# Command builder
# ---------------------------------------------------------------------------

def build_qemu_command(
    config: VMConfig,
    *,
    state_dir: Path | None = None,
) -> List[str]:
    """Build the full ``qemu-system-*`` argument vector.

    Args:
        config:    A validated :class:`VMConfig`.
        state_dir: If provided, QEMU monitor/serial sockets may be placed
                   here.  Otherwise the ``config.monitor`` value is used
                   verbatim.

    Returns:
        A list of strings ready for :class:`subprocess.Popen`.

    Raises:
        QemuNotFoundError: If the QEMU binary for *config.arch* is missing.
    """
    binary = find_qemu_binary(config.arch)
    cmd: List[str] = [binary]

    # Machine / CPU --------------------------------------------------------
    cmd += ["-cpu", config.cpu]
    cmd += ["-smp", str(config.cores)]
    cmd += ["-m", str(config.ram_mb)]

    # Display — headless ---------------------------------------------------
    cmd += ["-nographic"]

    # Drive ----------------------------------------------------------------
    disk = Path(config.disk_image)
    # Detect format from extension.
    fmt = "qcow2" if disk.suffix in (".qcow2", ".qcow") else "raw"
    cmd += [
        "-drive",
        f"file={disk},format={fmt},if=virtio",
    ]

    # Kernel / initrd (optional) -------------------------------------------
    if config.kernel:
        cmd += ["-kernel", config.kernel]
    if config.initrd:
        cmd += ["-initrd", config.initrd]
    if config.kernel_args and config.kernel:
        cmd += ["-append", config.kernel_args]

    # Serial ---------------------------------------------------------------
    serial_target = config.serial
    if serial_target == "stdio":
        # In background mode we redirect serial to a pty so the process
        # doesn't block on a missing tty.  The caller (process_manager)
        # overrides this when running interactively.
        if state_dir is not None:
            socket_path = state_dir / "sockets" / f"{config.name}-serial.sock"
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            serial_target = f"unix:{socket_path},server,nowait"
        else:
            serial_target = "stdio"
    cmd += ["-serial", serial_target]

    # Monitor --------------------------------------------------------------
    monitor_target = config.monitor
    if state_dir is not None and not config.monitor_is_unix:
        # Prefer a Unix socket for the monitor when running managed.
        sock = state_dir / "sockets" / f"{config.name}-monitor.sock"
        sock.parent.mkdir(parents=True, exist_ok=True)
        monitor_target = f"unix:{sock},server,nowait"
    cmd += ["-monitor", monitor_target]

    # QMP (always via Unix socket when state_dir available) ----------------
    if state_dir is not None:
        qmp_sock = state_dir / "sockets" / f"{config.name}-qmp.sock"
        qmp_sock.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["-qmp", f"unix:{qmp_sock},server,nowait"]

    # Network --------------------------------------------------------------
    net_type = config.network.type
    if net_type == "none":
        cmd += ["-nic", "none"]
    elif net_type == "tap":
        cmd += ["-nic", "tap"]
    else:
        # User-mode networking.
        net_str = "user"
        for fwd in config.network.hostfwd:
            net_str += f",hostfwd={fwd}"
        cmd += ["-nic", net_str]

    # Extra raw arguments --------------------------------------------------
    cmd += config.extra_args

    logger.debug("Built QEMU command: %s", cmd)
    return cmd
