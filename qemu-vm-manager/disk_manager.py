"""
disk_manager.py — Wrappers for ``qemu-img`` disk-image operations.

Every function locates ``qemu-img`` via :func:`shutil.which` and executes
it through :class:`subprocess.Popen` with explicit ``stdout`` / ``stderr``
redirection (never ``shell=True``).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Binary lookup
# ---------------------------------------------------------------------------

class QemuImgNotFoundError(Exception):
    """Raised when ``qemu-img`` is not on PATH."""


def _find_qemu_img() -> str:
    """Return the path to ``qemu-img`` or raise.

    Returns:
        Absolute path to the binary.

    Raises:
        QemuImgNotFoundError: If not found.
    """
    path = shutil.which("qemu-img")
    if path is None:
        raise QemuImgNotFoundError(
            "'qemu-img' not found on PATH.  Install QEMU utilities:\n"
            "  Debian/Ubuntu: sudo apt-get install qemu-utils\n"
            "  Arch Linux   : sudo pacman -S qemu-img\n"
            "  Alpine Linux : sudo apk add qemu-img"
        )
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_disk(
    path: str | Path,
    size_gb: int,
    fmt: str = "qcow2",
) -> Path:
    """Create a new disk image.

    Args:
        path:    Destination file path.
        size_gb: Size in gigabytes.
        fmt:     Image format (default ``qcow2``).

    Returns:
        The resolved :class:`Path` of the created image.

    Raises:
        QemuImgNotFoundError: If ``qemu-img`` is missing.
        subprocess.CalledProcessError: If the command fails.
    """
    qemu_img = _find_qemu_img()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [qemu_img, "create", "-f", fmt, str(p), f"{size_gb}G"]
    logger.info("Creating disk: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr,
        )
    logger.info("Disk created: %s (%dG, %s)", p, size_gb, fmt)
    return p.resolve()


def disk_info(path: str | Path) -> Dict[str, Any]:
    """Return metadata about a disk image.

    Args:
        path: Path to the disk image.

    Returns:
        Parsed JSON dict from ``qemu-img info --output=json``.

    Raises:
        QemuImgNotFoundError: If ``qemu-img`` is missing.
        FileNotFoundError: If *path* does not exist.
        subprocess.CalledProcessError: On command failure.
    """
    qemu_img = _find_qemu_img()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Disk image not found: {p}")

    cmd = [qemu_img, "info", "--output=json", str(p)]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr,
        )
    return json.loads(stdout.decode("utf-8"))


def create_snapshot(
    name: str,
    snapshot_tag: str,
    state_dir: Path,
) -> str:
    """Create an internal QEMU snapshot of a running VM's disk.

    This uses ``qemu-img snapshot -c <tag> <image>``.  The VM should
    ideally be paused first for consistency.

    Args:
        name:         VM name (used to look up the disk image from state).
        snapshot_tag: A label for the snapshot.
        state_dir:    State directory (to resolve config).

    Returns:
        Human-readable success message.

    Raises:
        QemuImgNotFoundError: If ``qemu-img`` is missing.
        RuntimeError: If the VM config cannot be found.
    """
    qemu_img = _find_qemu_img()

    # Resolve disk image path from config.
    from config import load_config
    from process_manager import _load_vm_status

    status = _load_vm_status(state_dir, name)
    if status is None or not status.config_path:
        raise RuntimeError(f"Cannot find config for VM '{name}'.")

    cfg = load_config(status.config_path)
    disk = cfg.disk_image

    cmd = [qemu_img, "snapshot", "-c", snapshot_tag, disk]
    logger.info("Creating snapshot: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr,
        )
    return f"Snapshot '{snapshot_tag}' created for '{name}'."


def resize_disk(
    path: str | Path,
    new_size: str,
) -> str:
    """Resize a disk image.

    Args:
        path:     Path to the disk image.
        new_size: New size string understood by ``qemu-img``, e.g.
                  ``"20G"`` or ``"+5G"``.

    Returns:
        Human-readable success message.

    Raises:
        QemuImgNotFoundError: If ``qemu-img`` is missing.
        FileNotFoundError: If *path* does not exist.
        subprocess.CalledProcessError: On failure.
    """
    qemu_img = _find_qemu_img()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Disk image not found: {p}")

    cmd = [qemu_img, "resize", str(p), new_size]
    logger.info("Resizing disk: %s", " ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, stderr,
        )
    return f"Disk '{p}' resized to {new_size}."
