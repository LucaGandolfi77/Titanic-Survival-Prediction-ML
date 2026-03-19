"""
tests/test_config.py — Unit tests for config loading, validation, and parsing.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (
    ConfigValidationError,
    NetworkConfig,
    VMConfig,
    check_port_available,
    load_config,
    load_all_configs,
    validate_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid YAML config and return its path."""
    # Also create a fake disk image so validation passes.
    disk = tmp_path / "test.qcow2"
    disk.write_bytes(b"\x00")
    cfg = tmp_path / "test-vm.yaml"
    cfg.write_text(textwrap.dedent(f"""\
        name: "test-vm"
        arch: "x86_64"
        cpu: "host"
        cores: 2
        ram_mb: 256
        disk_image: "{disk}"
        network:
          type: "user"
          hostfwd:
            - "tcp::2222-:22"
        serial: "stdio"
        monitor: "telnet::4444,server,nowait"
    """))
    return cfg


@pytest.fixture()
def bad_yaml(tmp_path: Path) -> Path:
    """Write a YAML config with a missing disk image."""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(textwrap.dedent("""\
        name: "bad-vm"
        disk_image: "/nonexistent/disk.qcow2"
        ram_mb: 16
        cores: 0
    """))
    return cfg


# ---------------------------------------------------------------------------
# Tests — load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for :func:`load_config`."""

    def test_loads_valid_yaml(self, sample_yaml: Path) -> None:
        cfg = load_config(sample_yaml)
        assert cfg.name == "test-vm"
        assert cfg.arch == "x86_64"
        assert cfg.ram_mb == 256
        assert cfg.cores == 2
        assert cfg.network.type == "user"
        assert cfg.network.hostfwd == ["tcp::2222-:22"]
        assert cfg.config_path == sample_yaml.resolve()

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_defaults_applied(self, tmp_path: Path) -> None:
        disk = tmp_path / "d.qcow2"
        disk.write_bytes(b"\x00")
        cfg_path = tmp_path / "minimal.yaml"
        cfg_path.write_text(f'name: "mini"\ndisk_image: "{disk}"\n')
        cfg = load_config(cfg_path)
        assert cfg.cpu == "host"
        assert cfg.cores == 2
        assert cfg.serial == "stdio"


# ---------------------------------------------------------------------------
# Tests — load_all_configs
# ---------------------------------------------------------------------------

class TestLoadAllConfigs:
    """Tests for :func:`load_all_configs`."""

    def test_scans_directory(self, tmp_path: Path) -> None:
        for i in range(3):
            disk = tmp_path / f"d{i}.qcow2"
            disk.write_bytes(b"\x00")
            (tmp_path / f"vm{i}.yaml").write_text(
                f'name: "vm{i}"\ndisk_image: "{disk}"\n'
            )
        # A non-YAML file should be ignored.
        (tmp_path / "readme.txt").write_text("ignore me")
        configs = load_all_configs(tmp_path)
        assert len(configs) == 3

    def test_nonexistent_dir_returns_empty(self) -> None:
        configs = load_all_configs("/nonexistent/dir")
        assert configs == []


# ---------------------------------------------------------------------------
# Tests — validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for :func:`validate_config`."""

    def test_valid_config_no_errors(self, sample_yaml: Path) -> None:
        cfg = load_config(sample_yaml)
        errors = validate_config(cfg)
        assert errors == []

    def test_missing_name(self) -> None:
        cfg = VMConfig(name="", disk_image="/tmp/x")
        errors = validate_config(cfg)
        assert any("name" in e.lower() for e in errors)

    def test_missing_disk_image(self) -> None:
        cfg = VMConfig(name="x", disk_image="")
        errors = validate_config(cfg)
        assert any("disk_image" in e for e in errors)

    def test_nonexistent_disk(self) -> None:
        cfg = VMConfig(name="x", disk_image="/no/such/file.qcow2")
        errors = validate_config(cfg)
        assert any("does not exist" in e for e in errors)

    def test_low_ram(self) -> None:
        cfg = VMConfig(name="x", disk_image="/tmp/x", ram_mb=1)
        errors = validate_config(cfg)
        assert any("ram_mb" in e for e in errors)

    def test_zero_cores(self) -> None:
        cfg = VMConfig(name="x", disk_image="/tmp/x", cores=0)
        errors = validate_config(cfg)
        assert any("cores" in e for e in errors)

    def test_bad_hostfwd_rule(self) -> None:
        cfg = VMConfig(
            name="x",
            disk_image="/tmp/x",
            network=NetworkConfig(hostfwd=["badformat"]),
        )
        errors = validate_config(cfg)
        assert any("hostfwd" in e.lower() or "Invalid" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests — VMConfig properties
# ---------------------------------------------------------------------------

class TestVMConfigProperties:
    """Tests for derived properties on :class:`VMConfig`."""

    def test_monitor_port_telnet(self) -> None:
        cfg = VMConfig(monitor="telnet::4444,server,nowait")
        assert cfg.monitor_port == 4444

    def test_monitor_port_unparseable(self) -> None:
        cfg = VMConfig(monitor="unix:/tmp/mon.sock")
        assert cfg.monitor_port is None

    def test_monitor_is_unix(self) -> None:
        cfg = VMConfig(monitor="unix:/tmp/mon.sock")
        assert cfg.monitor_is_unix is True
        cfg2 = VMConfig(monitor="telnet::4444,server,nowait")
        assert cfg2.monitor_is_unix is False


# ---------------------------------------------------------------------------
# Tests — check_port_available
# ---------------------------------------------------------------------------

class TestCheckPort:
    """Tests for :func:`check_port_available`."""

    def test_high_port_likely_free(self) -> None:
        # Port 0 is special — the OS assigns an ephemeral port, so binding
        # to an explicit high port is a reasonable proxy.
        assert isinstance(check_port_available(59123), bool)
