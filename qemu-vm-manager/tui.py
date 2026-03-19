"""
tui.py — Curses-based live TUI dashboard for VM management.

Displays a table of all VMs with status, CPU, RAM, uptime, and a
scrollable log tail for the selected VM.  Keyboard shortcuts allow
starting, stopping, pausing, and connecting to VMs without leaving
the dashboard.
"""

from __future__ import annotations

import curses
import logging
import time
from pathlib import Path
from typing import List, Optional

from config import VMConfig, load_all_configs, load_config
from logging_utils import tail_log
from process_manager import VMProcess, VMState, VMStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_uptime(seconds: float) -> str:
    """Convert seconds to a ``HH:MM:SS`` string.

    Args:
        seconds: Total elapsed seconds.

    Returns:
        Formatted string.
    """
    if seconds <= 0:
        return "--:--:--"
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _state_label(state: VMState) -> str:
    """Return a fixed-width, capitalised state label.

    Args:
        state: The VM state enum value.

    Returns:
        A 7-character-wide label.
    """
    return state.value.upper().center(7)


def _color_for_state(state: VMState) -> int:
    """Return a curses colour pair index for a VM state.

    Args:
        state: VM state.

    Returns:
        Colour pair number.
    """
    if state == VMState.RUNNING:
        return 1  # green
    if state == VMState.PAUSED:
        return 2  # yellow
    return 3      # red


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def run_dashboard(
    state_dir: Path,
    config_dir: Path,
    refresh_interval: float = 2.0,
) -> None:
    """Launch the curses TUI dashboard.

    The dashboard auto-refreshes every *refresh_interval* seconds.

    Keyboard shortcuts:

    =========  ===========================================
    Key        Action
    =========  ===========================================
    ↑ / ↓      Select VM
    ``s``      Start selected VM
    ``k``      Stop (kill) selected VM
    ``p``      Pause / resume selected VM
    ``c``      Attach to console (exits TUI temporarily)
    ``m``      Open QEMU monitor REPL
    ``q``      Quit dashboard
    =========  ===========================================

    Args:
        state_dir:        Path to the state directory.
        config_dir:       Path to VM config YAML directory.
        refresh_interval: Seconds between status refreshes.
    """
    curses.wrapper(
        _dashboard_main, state_dir, config_dir, refresh_interval,
    )


def _dashboard_main(
    stdscr: curses.window,
    state_dir: Path,
    config_dir: Path,
    refresh_interval: float,
) -> None:
    """Inner curses loop (called by ``curses.wrapper``).

    Args:
        stdscr:           The root curses window.
        state_dir:        State directory.
        config_dir:       Config directory.
        refresh_interval: Seconds between refreshes.
    """
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(int(refresh_interval * 1000))

    # Colours.
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)   # running
    curses.init_pair(2, curses.COLOR_YELLOW, -1)  # paused
    curses.init_pair(3, curses.COLOR_RED, -1)     # stopped
    curses.init_pair(4, curses.COLOR_CYAN, -1)    # header
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)  # selected

    selected = 0
    message: str = ""
    configs = load_all_configs(config_dir)
    config_map = {c.name: c for c in configs}

    while True:
        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()

        # Refresh VM list.
        vms: List[VMStatus] = VMProcess.list_vms(state_dir)
        # Merge configs that aren't tracked yet.
        tracked_names = {v.name for v in vms}
        for c in configs:
            if c.name not in tracked_names:
                vms.append(VMStatus(name=c.name, state=VMState.STOPPED))

        if selected >= len(vms):
            selected = max(0, len(vms) - 1)

        # ---- Title bar -----------------------------------------------------
        title = " QEMU VM Manager Dashboard "
        stdscr.addstr(0, max(0, (max_x - len(title)) // 2), title,
                      curses.A_BOLD | curses.color_pair(4))

        # ---- Column headers ------------------------------------------------
        header_y = 2
        header = f"{'NAME':<20s} {'STATE':^9s} {'CPU%':>5s} {'RAM MB':>7s} {'UPTIME':>10s}"
        stdscr.addstr(header_y, 1, header[:max_x - 2],
                      curses.A_BOLD | curses.A_UNDERLINE)

        # ---- VM rows -------------------------------------------------------
        table_height = max(3, (max_y - 10) // 2)
        for idx, vm in enumerate(vms[:table_height]):
            y = header_y + 1 + idx
            if y >= max_y - 1:
                break
            line = (
                f"{vm.name:<20s} "
                f"{_state_label(vm.state)} "
                f"{vm.cpu_percent:5.1f} "
                f"{vm.ram_mb_used:7.1f} "
                f"{_format_uptime(vm.uptime_seconds):>10s}"
            )
            attr = curses.color_pair(_color_for_state(vm.state))
            if idx == selected:
                attr = curses.color_pair(5) | curses.A_BOLD
            stdscr.addstr(y, 1, line[:max_x - 2], attr)

        # ---- Log tail for selected VM --------------------------------------
        log_y = header_y + 2 + table_height
        sel_name = vms[selected].name if vms else ""
        log_title = f" Log: {sel_name} "
        if log_y < max_y - 1:
            stdscr.addstr(log_y, 1, log_title[:max_x - 2],
                          curses.A_BOLD | curses.color_pair(4))
        log_lines = tail_log(sel_name, state_dir, lines=min(15, max_y - log_y - 3))
        for i, l in enumerate(log_lines):
            ly = log_y + 1 + i
            if ly >= max_y - 1:
                break
            stdscr.addstr(ly, 2, l[:max_x - 3])

        # ---- Status bar ----------------------------------------------------
        bar_y = max_y - 1
        shortcuts = " [s]tart [k]ill [p]ause/resume [c]onsole [m]onitor [q]uit "
        if message:
            shortcuts = f" {message} | {shortcuts}"
        stdscr.addstr(bar_y, 0, shortcuts[:max_x],
                      curses.A_REVERSE)

        stdscr.refresh()

        # ---- Input ---------------------------------------------------------
        ch = stdscr.getch()
        message = ""

        if ch == ord("q"):
            break
        elif ch == curses.KEY_UP:
            selected = max(0, selected - 1)
        elif ch == curses.KEY_DOWN:
            selected = min(len(vms) - 1, selected + 1)
        elif ch == ord("s") and vms:
            message = _action_start(vms[selected].name, config_map, state_dir)
        elif ch == ord("k") and vms:
            message = _action_stop(vms[selected].name, state_dir)
        elif ch == ord("p") and vms:
            message = _action_pause_resume(vms[selected], state_dir)
        elif ch == ord("c") and vms:
            curses.endwin()
            _action_console(vms[selected].name, state_dir)
            stdscr.refresh()
        elif ch == ord("m") and vms:
            curses.endwin()
            _action_monitor(vms[selected].name, state_dir)
            stdscr.refresh()


# ---------------------------------------------------------------------------
# Action wrappers (return status message strings)
# ---------------------------------------------------------------------------

def _action_start(
    name: str,
    config_map: dict[str, VMConfig],
    state_dir: Path,
) -> str:
    """Attempt to start a VM and return a user message.

    Args:
        name:       VM name.
        config_map: Loaded configs keyed by name.
        state_dir:  State directory.

    Returns:
        Status message.
    """
    cfg = config_map.get(name)
    if cfg is None:
        return f"No config found for '{name}'."
    try:
        VMProcess.start(cfg, state_dir)
        return f"Started '{name}'."
    except Exception as exc:
        return f"Start failed: {exc}"


def _action_stop(name: str, state_dir: Path) -> str:
    """Attempt to stop a VM and return a user message.

    Args:
        name:      VM name.
        state_dir: State directory.

    Returns:
        Status message.
    """
    try:
        VMProcess.stop(name, state_dir)
        return f"Stopped '{name}'."
    except Exception as exc:
        return f"Stop failed: {exc}"


def _action_pause_resume(vm: VMStatus, state_dir: Path) -> str:
    """Toggle pause/resume for a VM.

    Args:
        vm:        Current VM status.
        state_dir: State directory.

    Returns:
        Status message.
    """
    try:
        if vm.state == VMState.RUNNING:
            VMProcess.pause(vm.name, state_dir)
            return f"Paused '{vm.name}'."
        elif vm.state == VMState.PAUSED:
            VMProcess.resume(vm.name, state_dir)
            return f"Resumed '{vm.name}'."
        else:
            return f"'{vm.name}' is not running."
    except Exception as exc:
        return f"Pause/resume failed: {exc}"


def _action_console(name: str, state_dir: Path) -> None:
    """Temporarily exit curses and attach to a VM's serial console.

    Args:
        name:      VM name.
        state_dir: State directory.
    """
    from monitor import attach_console
    try:
        attach_console(name, state_dir)
    except Exception as exc:
        print(f"Console error: {exc}")
        input("Press Enter to return to dashboard...")


def _action_monitor(name: str, state_dir: Path) -> None:
    """Temporarily exit curses and open the QEMU monitor REPL.

    Args:
        name:      VM name.
        state_dir: State directory.
    """
    from monitor import interactive_monitor
    try:
        interactive_monitor(name, state_dir)
    except Exception as exc:
        print(f"Monitor error: {exc}")
        input("Press Enter to return to dashboard...")
