"""
tests/test_qemu_builder.py — Unit tests for QEMU command assembly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import NetworkConfig, VMConfig
from qemu_builder import QemuNotFoundError, build_qemu_command, find_qemu_binary


# ---------------------------------------------------------------------------
# find_qemu_binary
# ---------------------------------------------------------------------------

class TestFindQemuBinary:
    """Tests for :func:`find_qemu_binary`."""

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_found(self, mock_which: object) -> None:
        assert find_qemu_binary("x86_64") == "/usr/bin/qemu-system-x86_64"

    @patch("qemu_builder.shutil.which", return_value=None)
    def test_not_found_raises(self, mock_which: object) -> None:
        with pytest.raises(QemuNotFoundError, match="not found"):
            find_qemu_binary("x86_64")


# ---------------------------------------------------------------------------
# build_qemu_command
# ---------------------------------------------------------------------------

class TestBuildQemuCommand:
    """Tests for :func:`build_qemu_command`."""

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_minimal_command(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            arch="x86_64",
            cpu="host",
            cores=2,
            ram_mb=512,
            disk_image="/tmp/test.qcow2",
            network=NetworkConfig(type="user", hostfwd=["tcp::2222-:22"]),
            serial="stdio",
            monitor="telnet::4444,server,nowait",
        )
        cmd = build_qemu_command(cfg)
        assert cmd[0] == "/usr/bin/qemu-system-x86_64"
        assert "-nographic" in cmd
        assert "-cpu" in cmd
        assert "host" in cmd
        assert "-smp" in cmd
        assert "2" in cmd
        assert "-m" in cmd
        assert "512" in cmd

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_drive_flag(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            disk_image="/tmp/test.qcow2",
        )
        cmd = build_qemu_command(cfg)
        drive_args = [a for a in cmd if a.startswith("file=")]
        assert any("/tmp/test.qcow2" in a for a in cmd)

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_kernel_and_initrd(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            disk_image="/tmp/d.qcow2",
            kernel="/boot/vmlinuz",
            initrd="/boot/initrd.img",
            kernel_args="console=ttyS0 root=/dev/vda1",
        )
        cmd = build_qemu_command(cfg)
        assert "-kernel" in cmd
        assert "/boot/vmlinuz" in cmd
        assert "-initrd" in cmd
        assert "/boot/initrd.img" in cmd
        assert "-append" in cmd

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_no_network(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            disk_image="/tmp/d.qcow2",
            network=NetworkConfig(type="none"),
        )
        cmd = build_qemu_command(cfg)
        assert "none" in cmd

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_hostfwd_in_nic(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            disk_image="/tmp/d.qcow2",
            network=NetworkConfig(
                type="user",
                hostfwd=["tcp::2222-:22", "tcp::8080-:80"],
            ),
        )
        cmd = build_qemu_command(cfg)
        nic_idx = cmd.index("-nic")
        nic_val = cmd[nic_idx + 1]
        assert "hostfwd=tcp::2222-:22" in nic_val
        assert "hostfwd=tcp::8080-:80" in nic_val

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_extra_args(self, _mock: object) -> None:
        cfg = VMConfig(
            name="test",
            disk_image="/tmp/d.qcow2",
            extra_args=["-enable-kvm", "-boot", "d"],
        )
        cmd = build_qemu_command(cfg)
        assert "-enable-kvm" in cmd
        assert "-boot" in cmd

    @patch("qemu_builder.shutil.which", return_value="/usr/bin/qemu-system-x86_64")
    def test_state_dir_creates_sockets(self, _mock: object, tmp_path: Path) -> None:
        cfg = VMConfig(
            name="sock-test",
            disk_image="/tmp/d.qcow2",
        )
        cmd = build_qemu_command(cfg, state_dir=tmp_path)
        # Should contain unix: socket paths for serial, monitor, QMP.
        joined = " ".join(cmd)
        assert "unix:" in joined
        assert "sock-test-serial.sock" in joined
        assert "sock-test-monitor.sock" in joined
        assert "sock-test-qmp.sock" in joined
        # Sockets directory should have been created.
        assert (tmp_path / "sockets").is_dir()
