"""
config.py — VM configuration dataclass, YAML loading, and validation.

Provides :class:`VMConfig`, a fully-typed dataclass that maps one-to-one to
the YAML schema expected by the rest of the project, plus helpers to load a
single config or scan a directory for all ``*.yaml`` / ``*.yml`` files.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-structures
# ---------------------------------------------------------------------------

@dataclass
class NetworkConfig:
    """Network subsystem configuration for a single VM.

    Attributes:
        type:    QEMU network backend — ``"user"``, ``"tap"``, or ``"none"``.
        hostfwd: List of host-forward rules, e.g. ``"tcp::2222-:22"``.
    """

    type: str = "user"
    hostfwd: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main config
# ---------------------------------------------------------------------------

@dataclass
class VMConfig:
    """Complete description of a single virtual machine.

    Every field maps directly to the YAML schema documented in the README.
    Optional fields default to ``None`` or sensible values so that a minimal
    config with just *name*, *arch*, *ram_mb* and *disk_image* is enough.
    """

    name: str = ""
    arch: str = "x86_64"
    cpu: str = "host"
    cores: int = 2
    ram_mb: int = 512
    disk_image: str = ""
    kernel: Optional[str] = None
    initrd: Optional[str] = None
    kernel_args: str = "console=ttyS0"
    network: NetworkConfig = field(default_factory=NetworkConfig)
    serial: str = "stdio"
    monitor: str = "telnet::4444,server,nowait"
    extra_args: List[str] = field(default_factory=list)

    # Derived at load time — not from YAML.
    config_path: Optional[Path] = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def monitor_port(self) -> Optional[int]:
        """Extract the TCP port from *monitor* if it is a telnet string.

        Returns:
            The port number, or ``None`` if parsing fails.
        """
        try:
            # Format: "telnet::<port>,server,nowait"
            parts = self.monitor.split(",")[0]  # "telnet::<port>"
            port_str = parts.split(":")[2]
            return int(port_str)
        except (IndexError, ValueError):
            return None

    @property
    def monitor_is_unix(self) -> bool:
        """Return ``True`` if the monitor is a Unix socket path."""
        return self.monitor.startswith("unix:")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ConfigValidationError(Exception):
    """Raised when a VM config file is invalid or incomplete."""


def validate_config(cfg: VMConfig) -> List[str]:
    """Check a :class:`VMConfig` for missing / invalid fields.

    Returns a list of human-readable error strings; an empty list means
    the config is valid.

    Args:
        cfg: The config to validate.

    Returns:
        List of validation error messages.
    """
    errors: List[str] = []

    if not cfg.name:
        errors.append("'name' is required and cannot be empty.")
    if not cfg.disk_image:
        errors.append("'disk_image' is required.")
    elif not Path(cfg.disk_image).exists():
        errors.append(f"disk_image '{cfg.disk_image}' does not exist.")
    if cfg.ram_mb < 32:
        errors.append(f"ram_mb ({cfg.ram_mb}) must be >= 32.")
    if cfg.cores < 1:
        errors.append(f"cores ({cfg.cores}) must be >= 1.")
    if cfg.kernel and not Path(cfg.kernel).exists():
        errors.append(f"kernel '{cfg.kernel}' does not exist.")
    if cfg.initrd and not Path(cfg.initrd).exists():
        errors.append(f"initrd '{cfg.initrd}' does not exist.")

    # Check host-forward ports.
    for rule in cfg.network.hostfwd:
        _validate_hostfwd_rule(rule, errors)

    return errors


def _validate_hostfwd_rule(rule: str, errors: List[str]) -> None:
    """Validate a single host-forward rule and append errors if invalid.

    Args:
        rule:   e.g. ``"tcp::2222-:22"``.
        errors: Accumulator list.
    """
    parts = rule.split(":")
    if len(parts) < 3:
        errors.append(f"Invalid hostfwd rule '{rule}': expected <proto>::<hostport>-:<guestport>")
        return
    try:
        host_port_str = parts[2].split("-")[0]
        host_port = int(host_port_str)
        if not 1 <= host_port <= 65535:
            errors.append(f"Host port {host_port} in rule '{rule}' is out of range.")
    except (ValueError, IndexError):
        errors.append(f"Cannot parse host port from rule '{rule}'.")


def check_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Return ``True`` if *port* on *host* is free (not already bound).

    Args:
        port: TCP port number.
        host: Bind address to check.

    Returns:
        ``True`` if the port is available.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _parse_network(raw: Any) -> NetworkConfig:
    """Convert a raw YAML dict (or None) into a :class:`NetworkConfig`.

    Args:
        raw: The ``network:`` section from YAML.

    Returns:
        A ``NetworkConfig`` instance.
    """
    if raw is None:
        return NetworkConfig()
    if isinstance(raw, dict):
        return NetworkConfig(
            type=raw.get("type", "user"),
            hostfwd=raw.get("hostfwd", []),
        )
    return NetworkConfig()


def load_config(path: str | Path) -> VMConfig:
    """Load and parse a single YAML VM config file.

    Args:
        path: Filesystem path to the ``.yaml`` file.

    Returns:
        A populated ``VMConfig``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ConfigValidationError: If YAML parsing fails.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        data: Dict[str, Any] = yaml.safe_load(fh) or {}

    try:
        cfg = VMConfig(
            name=data.get("name", p.stem),
            arch=data.get("arch", "x86_64"),
            cpu=data.get("cpu", "host"),
            cores=int(data.get("cores", 2)),
            ram_mb=int(data.get("ram_mb", 512)),
            disk_image=data.get("disk_image", ""),
            kernel=data.get("kernel") or None,
            initrd=data.get("initrd") or None,
            kernel_args=data.get("kernel_args", "console=ttyS0"),
            network=_parse_network(data.get("network")),
            serial=data.get("serial", "stdio"),
            monitor=data.get("monitor", "telnet::4444,server,nowait"),
            extra_args=data.get("extra_args", []),
            config_path=p.resolve(),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigValidationError(f"Error parsing {p}: {exc}") from exc

    logger.debug("Loaded config '%s' from %s", cfg.name, p)
    return cfg


def load_all_configs(directory: str | Path) -> List[VMConfig]:
    """Scan *directory* for ``*.yaml`` / ``*.yml`` files and load them all.

    Args:
        directory: Path to scan.

    Returns:
        List of ``VMConfig`` instances (only successfully parsed ones).
    """
    d = Path(directory)
    configs: List[VMConfig] = []
    if not d.is_dir():
        logger.warning("Config directory does not exist: %s", d)
        return configs

    for ext in ("*.yaml", "*.yml"):
        for p in sorted(d.glob(ext)):
            try:
                configs.append(load_config(p))
            except Exception as exc:
                logger.warning("Skipping %s: %s", p, exc)

    logger.info("Loaded %d VM configs from %s", len(configs), d)
    return configs
