"""
cli.py — Command-line interface built with ``argparse``.

Defines the ``vm`` top-level command with sub-commands for every
lifecycle action, disk management, network diagnostics, and the TUI
dashboard.  Each handler converts internal log messages to user-friendly
terminal output.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from config import (
    VMConfig,
    check_port_available,
    load_all_configs,
    load_config,
    validate_config,
)
from disk_manager import create_disk, create_snapshot, disk_info, resize_disk
from logging_utils import log_event, setup_logging, tail_log
from monitor import attach_console, interactive_monitor, qmp_command
from process_manager import VMProcess, VMState
from tui import run_dashboard

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_STATE_DIR = Path.home() / ".vms"
_DEFAULT_CONFIG_DIR = Path.home() / ".vms"


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the full ``argparse`` parser tree.

    Returns:
        The root parser.
    """
    root = argparse.ArgumentParser(
        prog="vm",
        description="QEMU VM Manager — headless VM lifecycle tool for SSH environments.",
    )
    root.add_argument(
        "--state-dir",
        type=Path,
        default=_DEFAULT_STATE_DIR,
        help="Directory for state, logs, and sockets (default: ~/.vms).",
    )
    root.add_argument(
        "--config-dir",
        type=Path,
        default=_DEFAULT_CONFIG_DIR,
        help="Directory containing VM YAML configs (default: ~/.vms).",
    )
    root.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )

    sub = root.add_subparsers(dest="command", required=True)

    # -- start -------------------------------------------------------------
    p = sub.add_parser("start", help="Start a VM.")
    p.add_argument(
        "name_or_config",
        help="VM name (looked up in config-dir) or path to a .yaml file.",
    )

    # -- stop --------------------------------------------------------------
    p = sub.add_parser("stop", help="Stop a running VM.")
    p.add_argument("name", help="VM name.")
    p.add_argument(
        "-f", "--force",
        action="store_true",
        help="Force kill (SIGKILL) without graceful shutdown.",
    )

    # -- pause -------------------------------------------------------------
    p = sub.add_parser("pause", help="Pause a running VM.")
    p.add_argument("name", help="VM name.")

    # -- resume ------------------------------------------------------------
    p = sub.add_parser("resume", help="Resume a paused VM.")
    p.add_argument("name", help="VM name.")

    # -- restart -----------------------------------------------------------
    p = sub.add_parser("restart", help="Restart a VM (stop + start).")
    p.add_argument(
        "name_or_config",
        help="VM name or path to config file.",
    )

    # -- status ------------------------------------------------------------
    p = sub.add_parser("status", help="Show status of a VM.")
    p.add_argument("name", nargs="?", help="VM name (omit for all).")

    # -- list --------------------------------------------------------------
    sub.add_parser("list", help="List all managed VMs.")

    # -- console -----------------------------------------------------------
    p = sub.add_parser("console", help="Attach to a VM's serial console.")
    p.add_argument("name", help="VM name.")

    # -- monitor -----------------------------------------------------------
    p = sub.add_parser("monitor", help="Open QEMU monitor REPL.")
    p.add_argument("name", help="VM name.")

    # -- qmp ---------------------------------------------------------------
    p = sub.add_parser("qmp", help="Send a single QMP command.")
    p.add_argument("name", help="VM name.")
    p.add_argument("qmp_command", help="QMP command name.")
    p.add_argument("args_json", nargs="?", default="{}", help="JSON arguments.")

    # -- disk --------------------------------------------------------------
    disk_p = sub.add_parser("disk", help="Disk image management.")
    disk_sub = disk_p.add_subparsers(dest="disk_action", required=True)

    dc = disk_sub.add_parser("create", help="Create a new qcow2 disk.")
    dc.add_argument("path", help="Output path for the new image.")
    dc.add_argument("size_gb", type=int, help="Size in GB.")

    di = disk_sub.add_parser("info", help="Show disk image info.")
    di.add_argument("image_path", help="Path to the image.")

    ds = disk_sub.add_parser("snapshot", help="Create a QEMU snapshot.")
    ds.add_argument("name", help="VM name.")
    ds.add_argument("snapshot_tag", help="Snapshot label.")

    dr = disk_sub.add_parser("resize", help="Resize a disk image.")
    dr.add_argument("image_path", help="Path to the image.")
    dr.add_argument("new_size", help="New size (e.g. 20G or +5G).")

    # -- network -----------------------------------------------------------
    p = sub.add_parser("network", help="Show network / port-forward info.")
    p.add_argument("name", help="VM name.")

    # -- dashboard ---------------------------------------------------------
    sub.add_parser("dashboard", help="Launch the live TUI dashboard.")

    # -- config validate ---------------------------------------------------
    cv = sub.add_parser("config", help="Config utilities.")
    cv_sub = cv.add_subparsers(dest="config_action", required=True)
    cv_val = cv_sub.add_parser("validate", help="Validate a config file.")
    cv_val.add_argument("path", help="Path to the YAML config.")

    return root


# ---------------------------------------------------------------------------
# Resolve a VM config from a name-or-path argument
# ---------------------------------------------------------------------------

def _resolve_config(
    name_or_path: str,
    config_dir: Path,
) -> VMConfig:
    """Find a config by name (scanning *config_dir*) or by file path.

    Args:
        name_or_path: Either a bare name or a ``*.yaml`` path.
        config_dir:   Directory to scan for YAML files.

    Returns:
        Loaded ``VMConfig``.

    Raises:
        SystemExit: If the config cannot be found.
    """
    p = Path(name_or_path)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return load_config(p)

    # Look up by name in config_dir.
    for ext in ("yaml", "yml"):
        candidate = config_dir / f"{name_or_path}.{ext}"
        if candidate.exists():
            return load_config(candidate)

    # Try loading all and matching by name.
    for cfg in load_all_configs(config_dir):
        if cfg.name == name_or_path:
            return cfg

    _error(f"Cannot find config for '{name_or_path}' in {config_dir}.")
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _info(msg: str) -> None:
    """Print an informational message to stdout.

    Args:
        msg: Message text.
    """
    print(f"[+] {msg}")


def _error(msg: str) -> None:
    """Print an error message to stderr.

    Args:
        msg: Error text.
    """
    print(f"[!] {msg}", file=sys.stderr)


def _format_uptime(seconds: float) -> str:
    """Format seconds as ``Xd Xh Xm Xs``.

    Args:
        seconds: Elapsed seconds.

    Returns:
        Human-readable duration.
    """
    if seconds <= 0:
        return "-"
    d = int(seconds) // 86400
    h = (int(seconds) % 86400) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_start(args: argparse.Namespace) -> None:
    """Handle ``vm start``.

    Args:
        args: Parsed CLI arguments.
    """
    cfg = _resolve_config(args.name_or_config, args.config_dir)
    errors = validate_config(cfg)
    if errors:
        for e in errors:
            _error(e)
        raise SystemExit(1)

    # Port-availability warnings.
    for rule in cfg.network.hostfwd:
        try:
            port = int(rule.split(":")[2].split("-")[0])
            if not check_port_available(port):
                _error(f"WARNING: port {port} is already in use (rule: {rule}).")
        except (IndexError, ValueError):
            pass

    try:
        status = VMProcess.start(cfg, args.state_dir)
        log_event(args.state_dir, cfg.name, "START", f"pid={status.pid}")
        _info(f"VM '{cfg.name}' started (pid={status.pid}, method={status.detach_method}).")
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_stop(args: argparse.Namespace) -> None:
    """Handle ``vm stop``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        VMProcess.stop(args.name, args.state_dir, graceful=not args.force)
        log_event(args.state_dir, args.name, "STOP")
        _info(f"VM '{args.name}' stopped.")
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_pause(args: argparse.Namespace) -> None:
    """Handle ``vm pause``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        VMProcess.pause(args.name, args.state_dir)
        log_event(args.state_dir, args.name, "PAUSE")
        _info(f"VM '{args.name}' paused.")
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_resume(args: argparse.Namespace) -> None:
    """Handle ``vm resume``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        VMProcess.resume(args.name, args.state_dir)
        log_event(args.state_dir, args.name, "RESUME")
        _info(f"VM '{args.name}' resumed.")
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_restart(args: argparse.Namespace) -> None:
    """Handle ``vm restart``.

    Args:
        args: Parsed CLI arguments.
    """
    cfg = _resolve_config(args.name_or_config, args.config_dir)
    try:
        status = VMProcess.restart(cfg, args.state_dir)
        log_event(args.state_dir, cfg.name, "RESTART", f"pid={status.pid}")
        _info(f"VM '{cfg.name}' restarted (pid={status.pid}).")
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Handle ``vm status [name]``.

    Args:
        args: Parsed CLI arguments.
    """
    if args.name:
        st = VMProcess.get_status(args.name, args.state_dir)
        _print_status_table([st])
    else:
        vms = VMProcess.list_vms(args.state_dir)
        if not vms:
            _info("No VMs tracked.")
            return
        _print_status_table(vms)


def cmd_list(args: argparse.Namespace) -> None:
    """Handle ``vm list``.

    Args:
        args: Parsed CLI arguments.
    """
    vms = VMProcess.list_vms(args.state_dir)
    configs = load_all_configs(args.config_dir)
    tracked = {v.name for v in vms}
    for c in configs:
        if c.name not in tracked:
            vms.append(
                VMProcess.get_status(c.name, args.state_dir)
            )
    if not vms:
        _info("No VMs found.")
        return
    _print_status_table(vms)


def _print_status_table(vms: list) -> None:
    """Render a table of VM statuses.

    Args:
        vms: List of ``VMStatus`` objects.
    """
    header = f"{'NAME':<20s} {'STATE':<9s} {'PID':>7s} {'CPU%':>6s} {'RAM MB':>8s} {'UPTIME':>12s}"
    print(header)
    print("-" * len(header))
    for vm in vms:
        print(
            f"{vm.name:<20s} "
            f"{vm.state.value:<9s} "
            f"{vm.pid if vm.pid else '-':>7} "
            f"{vm.cpu_percent:>6.1f} "
            f"{vm.ram_mb_used:>8.1f} "
            f"{_format_uptime(vm.uptime_seconds):>12s}"
        )


def cmd_console(args: argparse.Namespace) -> None:
    """Handle ``vm console``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        attach_console(args.name, args.state_dir)
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_monitor(args: argparse.Namespace) -> None:
    """Handle ``vm monitor``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        interactive_monitor(args.name, args.state_dir)
    except RuntimeError as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_qmp(args: argparse.Namespace) -> None:
    """Handle ``vm qmp``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        result = qmp_command(
            args.name, args.state_dir, args.qmp_command, args.args_json,
        )
        print(json.dumps(result, indent=2))
    except Exception as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_disk(args: argparse.Namespace) -> None:
    """Handle ``vm disk <action>``.

    Args:
        args: Parsed CLI arguments.
    """
    action = args.disk_action
    try:
        if action == "create":
            path = create_disk(args.path, args.size_gb)
            _info(f"Disk created: {path}")
        elif action == "info":
            info = disk_info(args.image_path)
            print(json.dumps(info, indent=2))
        elif action == "snapshot":
            msg = create_snapshot(args.name, args.snapshot_tag, args.state_dir)
            _info(msg)
        elif action == "resize":
            msg = resize_disk(args.image_path, args.new_size)
            _info(msg)
    except Exception as exc:
        _error(str(exc))
        raise SystemExit(1)


def cmd_network(args: argparse.Namespace) -> None:
    """Handle ``vm network``.

    Shows the port-forwarding rules and whether each host port is free.

    Args:
        args: Parsed CLI arguments.
    """
    cfg = _resolve_config(args.name, args.config_dir)
    if not cfg.network.hostfwd:
        _info(f"VM '{args.name}' has no port-forwarding rules.")
        return

    print(f"Port forwarding for '{args.name}':")
    print(f"  {'RULE':<30s} {'HOST PORT':>10s} {'STATUS':>10s}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for rule in cfg.network.hostfwd:
        try:
            port = int(rule.split(":")[2].split("-")[0])
            ok = check_port_available(port)
            status_str = "free" if ok else "IN USE"
        except (IndexError, ValueError):
            port = 0
            status_str = "?"
        print(f"  {rule:<30s} {port:>10d} {status_str:>10s}")


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Handle ``vm dashboard``.

    Args:
        args: Parsed CLI arguments.
    """
    run_dashboard(args.state_dir, args.config_dir)


def cmd_config_validate(args: argparse.Namespace) -> None:
    """Handle ``vm config validate``.

    Args:
        args: Parsed CLI arguments.
    """
    try:
        cfg = load_config(args.path)
    except Exception as exc:
        _error(str(exc))
        raise SystemExit(1)

    errors = validate_config(cfg)
    if errors:
        _error(f"Config '{args.path}' has {len(errors)} error(s):")
        for e in errors:
            _error(f"  - {e}")
        raise SystemExit(1)
    _info(f"Config '{args.path}' is valid ({cfg.name}, {cfg.arch}, {cfg.ram_mb}MB).")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

_DISPATCH = {
    "start": cmd_start,
    "stop": cmd_stop,
    "pause": cmd_pause,
    "resume": cmd_resume,
    "restart": cmd_restart,
    "status": cmd_status,
    "list": cmd_list,
    "console": cmd_console,
    "monitor": cmd_monitor,
    "qmp": cmd_qmp,
    "disk": cmd_disk,
    "network": cmd_network,
    "dashboard": cmd_dashboard,
    "config": cmd_config_validate,
}


def run_cli(argv: Sequence[str] | None = None) -> None:
    """Parse arguments and dispatch to the appropriate handler.

    Args:
        argv: Explicit argument list (uses ``sys.argv`` if ``None``).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.verbose)

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        raise SystemExit(1)
    handler(args)
