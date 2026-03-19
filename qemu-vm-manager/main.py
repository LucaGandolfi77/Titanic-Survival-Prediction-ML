#!/usr/bin/env python3
"""
main.py — Entry point for the QEMU VM Manager CLI.

Dispatches to :mod:`cli` and handles top-level exceptions gracefully
(e.g. ``KeyboardInterrupt`` detaches from a console without killing
the VM).
"""

from __future__ import annotations

import sys

from cli import run_cli


def main() -> None:
    """Run the CLI, translating exceptions to clean exit codes."""
    try:
        run_cli()
    except KeyboardInterrupt:
        # The user pressed Ctrl+C — exit cleanly without a traceback.
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
    except BrokenPipeError:
        # Piping output to head/less that closes early.
        raise SystemExit(0)


if __name__ == "__main__":
    main()
