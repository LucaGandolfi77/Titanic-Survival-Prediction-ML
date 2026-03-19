"""
monitor.py — QMP client, human-monitor client, and console attachment.

Provides three ways to interact with a running QEMU instance:

1. :class:`QMPClient` — send JSON commands via the QEMU Machine Protocol.
2. :class:`HumanMonitorClient` — interactive text-based QEMU monitor.
3. :func:`attach_console` — connect to the VM's serial console for direct
   terminal interaction.
"""

from __future__ import annotations

import json
import logging
import os
import readline  # enables line-editing in interactive_monitor
import select
import shutil
import socket
import struct
import subprocess
import sys
import termios
import time
import tty
from pathlib import Path
from typing import Any, Dict, Optional

from process_manager import VMStatus, VMProcess, _load_vm_status

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# QMP Client
# ---------------------------------------------------------------------------

class QMPError(Exception):
    """Raised on QMP protocol-level errors."""


class QMPClient:
    """Minimal QEMU Machine Protocol client (Unix-socket or TCP).

    Usage::

        with QMPClient("/path/to/qmp.sock") as qmp:
            info = qmp.execute("query-status")
            print(info)

    Args:
        address: Either a Unix socket path (``str`` / ``Path``) or a
                 ``(host, port)`` tuple for TCP.
        timeout: Socket timeout in seconds.
    """

    def __init__(
        self,
        address: str | Path | tuple[str, int],
        timeout: float = 10.0,
    ) -> None:
        self._address = address
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None

    # Context-manager interface.

    def __enter__(self) -> "QMPClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Connection.

    def connect(self) -> None:
        """Open the socket and negotiate QMP capabilities.

        Raises:
            QMPError: If the greeting or negotiation fails.
        """
        if isinstance(self._address, (str, Path)):
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        self._sock.settimeout(self._timeout)
        addr = str(self._address) if isinstance(self._address, Path) else self._address
        self._sock.connect(addr)

        # Read greeting.
        greeting = self._recv_json()
        if "QMP" not in greeting:
            raise QMPError(f"Unexpected QMP greeting: {greeting}")
        logger.debug("QMP greeting: %s", greeting)

        # Negotiate capabilities (empty list = accept defaults).
        self._send_json({"execute": "qmp_capabilities"})
        resp = self._recv_json()
        if "error" in resp:
            raise QMPError(f"qmp_capabilities error: {resp['error']}")

    def close(self) -> None:
        """Close the underlying socket."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def execute(self, command: str, **arguments: Any) -> Dict[str, Any]:
        """Send a QMP command and return the response dict.

        Args:
            command:    QMP command name (e.g. ``"query-status"``).
            **arguments: Optional keyword arguments passed in the
                         ``"arguments"`` envelope.

        Returns:
            The ``"return"`` value from QMP.

        Raises:
            QMPError: On protocol errors or if the VM returns an error.
        """
        msg: Dict[str, Any] = {"execute": command}
        if arguments:
            msg["arguments"] = arguments
        self._send_json(msg)

        # Read responses; skip asynchronous events.
        while True:
            resp = self._recv_json()
            if "event" in resp:
                logger.debug("QMP event: %s", resp)
                continue
            if "error" in resp:
                raise QMPError(f"QMP error: {resp['error']}")
            return resp.get("return", resp)

    # Internal I/O.

    def _send_json(self, obj: Any) -> None:
        """Serialise and send a JSON object.

        Args:
            obj: JSON-serialisable Python object.
        """
        raw = json.dumps(obj).encode("utf-8") + b"\n"
        self._sock.sendall(raw)

    def _recv_json(self) -> Dict[str, Any]:
        """Read a single JSON object from the socket.

        Returns:
            Parsed dict.
        """
        buf = b""
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise QMPError("QMP socket closed unexpectedly.")
            buf += chunk
            try:
                return json.loads(buf.decode("utf-8"))
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Human Monitor Client (telnet-style TCP / Unix socket)
# ---------------------------------------------------------------------------

class HumanMonitorClient:
    """Interactive REPL for the QEMU human monitor.

    Connect to the monitor socket, send raw text commands, and receive
    text responses.

    Args:
        address: Unix socket path or ``(host, port)`` tuple.
        timeout: Socket timeout in seconds.
    """

    def __init__(
        self,
        address: str | Path | tuple[str, int],
        timeout: float = 5.0,
    ) -> None:
        self._address = address
        self._timeout = timeout
        self._sock: Optional[socket.socket] = None

    def __enter__(self) -> "HumanMonitorClient":
        self.connect()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def connect(self) -> None:
        """Open the socket and consume the initial QEMU banner.

        Raises:
            OSError: If connection fails.
        """
        if isinstance(self._address, (str, Path)):
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        else:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self._timeout)
        addr = str(self._address) if isinstance(self._address, Path) else self._address
        self._sock.connect(addr)
        # Consume banner.
        self._recv_text()

    def close(self) -> None:
        """Close the monitor socket."""
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    def send_command(self, cmd: str) -> str:
        """Send a text command and return the response.

        Args:
            cmd: Monitor command, e.g. ``"info status"``.

        Returns:
            Response text from QEMU.
        """
        self._sock.sendall((cmd + "\n").encode("utf-8"))
        time.sleep(0.3)
        return self._recv_text()

    def _recv_text(self) -> str:
        """Read all available data from the socket.

        Returns:
            Decoded text.
        """
        chunks: list[bytes] = []
        self._sock.setblocking(False)
        try:
            while True:
                ready, _, _ = select.select([self._sock], [], [], 0.5)
                if not ready:
                    break
                data = self._sock.recv(4096)
                if not data:
                    break
                chunks.append(data)
        except BlockingIOError:
            pass
        finally:
            self._sock.setblocking(True)
            self._sock.settimeout(self._timeout)
        return b"".join(chunks).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Console attachment
# ---------------------------------------------------------------------------

def attach_console(name: str, state_dir: Path) -> None:
    """Attach to a VM's serial console interactively.

    If ``socat`` is installed it is used for clean terminal pass-through;
    otherwise a pure-Python raw-socket loop is used.

    Detach with **Ctrl+]**.

    Args:
        name:      VM name.
        state_dir: State directory.

    Raises:
        RuntimeError: If the VM is not running or has no serial socket.
    """
    status = _load_vm_status(state_dir, name)
    if status is None:
        raise RuntimeError(f"VM '{name}' is not tracked.")
    if not status.serial_socket:
        raise RuntimeError(f"VM '{name}' has no serial socket configured.")
    sock_path = Path(status.serial_socket)
    if not sock_path.exists():
        raise RuntimeError(f"Serial socket does not exist: {sock_path}")

    socat = shutil.which("socat")
    if socat:
        _attach_socat(str(sock_path))
    else:
        _attach_raw(str(sock_path))


def _attach_socat(sock_path: str) -> None:
    """Use ``socat`` for clean serial console passthrough.

    Args:
        sock_path: Unix socket path.
    """
    print(f"Attaching to serial console via socat (Ctrl+] to detach)...")
    proc = subprocess.Popen(
        ["socat", f"UNIX-CONNECT:{sock_path}", "STDIO,raw,echo=0,escape=0x1d"],
        stdin=sys.stdin,
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def _attach_raw(sock_path: str) -> None:
    """Pure-Python raw-socket serial console attachment.

    Args:
        sock_path: Unix socket path.
    """
    DETACH_BYTE = 0x1D  # Ctrl+]

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(sock_path)
    sock.setblocking(False)

    old_tty = termios.tcgetattr(sys.stdin)
    print("Attached to serial console (Ctrl+] to detach).")
    try:
        tty.setraw(sys.stdin.fileno())
        while True:
            rlist, _, _ = select.select([sock, sys.stdin], [], [], 0.1)
            if sock in rlist:
                data = sock.recv(4096)
                if not data:
                    break
                os.write(sys.stdout.fileno(), data)
            if sys.stdin in rlist:
                ch = os.read(sys.stdin.fileno(), 1)
                if not ch or ch[0] == DETACH_BYTE:
                    break
                sock.sendall(ch)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        sock.close()
        print("\nDetached from console.")


# ---------------------------------------------------------------------------
# Interactive monitor REPL
# ---------------------------------------------------------------------------

def interactive_monitor(name: str, state_dir: Path) -> None:
    """Open an interactive REPL session with the QEMU monitor.

    Args:
        name:      VM name.
        state_dir: State directory.

    Raises:
        RuntimeError: If no monitor socket is available.
    """
    status = _load_vm_status(state_dir, name)
    if status is None:
        raise RuntimeError(f"VM '{name}' is not tracked.")
    if not status.monitor_socket:
        raise RuntimeError(f"VM '{name}' has no monitor socket configured.")
    sock_path = Path(status.monitor_socket)
    if not sock_path.exists():
        raise RuntimeError(f"Monitor socket does not exist: {sock_path}")

    with HumanMonitorClient(str(sock_path)) as client:
        print(f"QEMU monitor for '{name}' (type 'quit' to exit REPL, "
              f"'help' for commands).")
        while True:
            try:
                cmd = input("(qemu) ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if cmd.strip().lower() in ("quit", "exit", "q"):
                break
            resp = client.send_command(cmd)
            if resp:
                print(resp, end="")


# ---------------------------------------------------------------------------
# QMP single-command dispatch
# ---------------------------------------------------------------------------

def qmp_command(
    name: str,
    state_dir: Path,
    command: str,
    args_json: str = "{}",
) -> Dict[str, Any]:
    """Send a single QMP command and return the response.

    Args:
        name:      VM name.
        state_dir: State directory.
        command:   QMP command string.
        args_json: JSON-encoded arguments.

    Returns:
        Parsed QMP response dict.

    Raises:
        RuntimeError: If no QMP socket is available.
    """
    status = _load_vm_status(state_dir, name)
    if status is None:
        raise RuntimeError(f"VM '{name}' is not tracked.")
    if not status.qmp_socket:
        raise RuntimeError(f"VM '{name}' has no QMP socket configured.")
    sock_path = Path(status.qmp_socket)
    if not sock_path.exists():
        raise RuntimeError(f"QMP socket does not exist: {sock_path}")

    parsed_args = json.loads(args_json)

    with QMPClient(str(sock_path)) as qmp:
        return qmp.execute(command, **parsed_args)
