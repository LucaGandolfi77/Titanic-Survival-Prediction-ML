"""
tests/test_process_manager.py — Unit tests for state-file I/O and get_status.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from process_manager import (
    VMState,
    VMStatus,
    _load_vm_status,
    _pid_alive,
    _read_state,
    _save_vm_status,
    _write_state,
    VMProcess,
)


# ---------------------------------------------------------------------------
# State-file I/O
# ---------------------------------------------------------------------------

class TestStateFile:
    """Tests for atomic state-file read/write."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        data = {"vm1": {"name": "vm1", "state": "running", "pid": 1234}}
        _write_state(tmp_path, data)
        loaded = _read_state(tmp_path)
        assert loaded == data

    def test_read_missing_returns_empty(self, tmp_path: Path) -> None:
        assert _read_state(tmp_path) == {}

    def test_atomicity(self, tmp_path: Path) -> None:
        """State file must not be corrupted even on double-write."""
        _write_state(tmp_path, {"a": 1})
        _write_state(tmp_path, {"b": 2})
        assert _read_state(tmp_path) == {"b": 2}


class TestSaveLoadVMStatus:
    """Tests for per-VM save/load helpers."""

    def test_round_trip(self, tmp_path: Path) -> None:
        status = VMStatus(
            name="test-vm",
            state=VMState.RUNNING,
            pid=9999,
            start_time="2025-01-01T00:00:00+00:00",
            config_path="/etc/vm.yaml",
        )
        _save_vm_status(tmp_path, status)
        loaded = _load_vm_status(tmp_path, "test-vm")
        assert loaded is not None
        assert loaded.name == "test-vm"
        assert loaded.state == VMState.RUNNING
        assert loaded.pid == 9999

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        assert _load_vm_status(tmp_path, "no-such-vm") is None


# ---------------------------------------------------------------------------
# PID helpers
# ---------------------------------------------------------------------------

class TestPidAlive:
    """Tests for :func:`_pid_alive`."""

    def test_own_pid_is_alive(self) -> None:
        assert _pid_alive(os.getpid()) is True

    def test_zero_is_not_alive(self) -> None:
        assert _pid_alive(0) is False

    def test_negative_is_not_alive(self) -> None:
        assert _pid_alive(-1) is False


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    """Tests for :meth:`VMProcess.get_status`."""

    def test_unknown_vm(self, tmp_path: Path) -> None:
        st = VMProcess.get_status("ghost", tmp_path)
        assert st.state == VMState.UNKNOWN

    def test_dead_pid_marks_stopped(self, tmp_path: Path) -> None:
        status = VMStatus(
            name="dead",
            state=VMState.RUNNING,
            pid=999999999,  # very unlikely to be alive
        )
        _save_vm_status(tmp_path, status)
        st = VMProcess.get_status("dead", tmp_path)
        assert st.state == VMState.STOPPED


# ---------------------------------------------------------------------------
# list_vms
# ---------------------------------------------------------------------------

class TestListVMs:
    """Tests for :meth:`VMProcess.list_vms`."""

    def test_empty_state(self, tmp_path: Path) -> None:
        assert VMProcess.list_vms(tmp_path) == []

    def test_returns_all(self, tmp_path: Path) -> None:
        for name in ("vm-a", "vm-b"):
            _save_vm_status(
                tmp_path,
                VMStatus(name=name, state=VMState.STOPPED),
            )
        vms = VMProcess.list_vms(tmp_path)
        assert len(vms) == 2
        names = {v.name for v in vms}
        assert names == {"vm-a", "vm-b"}
