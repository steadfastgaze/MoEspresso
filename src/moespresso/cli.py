"""The user-facing MoEspresso command dispatcher."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from moespresso import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="moespresso",
        description="Serve, generate from, or verify a MoEspresso package.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"MoEspresso {__version__}",
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    commands.add_parser(
        "serve",
        add_help=False,
        help="Run the OpenAI-compatible HTTP server.",
    )
    commands.add_parser(
        "generate",
        add_help=False,
        help="Generate once from a package.",
    )
    commands.add_parser(
        "verify",
        add_help=False,
        help="Verify package integrity and optional external drafter bytes.",
    )
    commands.add_parser(
        "speed",
        add_help=False,
        help="Print decode speed from an existing MoEspresso server.",
    )
    commands.add_parser(
        "completions-api-timing",
        add_help=False,
        help="Independently count tokens and time an existing chat-completions API.",
    )
    return parser


def _command(command: str) -> tuple[Callable, str]:
    if command == "serve":
        from moespresso.serve_supervisor import main

        return main, "moespresso serve"
    if command == "generate":
        from moespresso.runtime.serve import main

        return main, "moespresso generate"
    if command == "verify":
        from moespresso.runtime.serve import verify_main

        return verify_main, "moespresso verify"
    if command == "speed":
        from moespresso.runtime.diagnostics import main

        return main, "moespresso speed"
    if command == "completions-api-timing":
        from moespresso.runtime.completions_api_timing import main

        return main, "moespresso completions-api-timing"
    raise AssertionError(f"unknown command {command!r}")


def main(argv: list[str] | None = None) -> int:
    """Dispatch a spaced command to the same parser as its legacy alias."""
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not args:
        parser.print_help()
        return 0
    if args[0] in {"-h", "--help", "--version"}:
        parser.parse_args(args)
        return 0

    command = args.pop(0)
    if command not in {"serve", "generate", "verify", "speed", "completions-api-timing"}:
        parser.error(
            f"argument COMMAND: invalid choice: {command!r} "
            "(choose from 'serve', 'generate', 'verify', 'speed', 'completions-api-timing')"
        )
    command_main, prog = _command(command)
    return command_main(args, prog=prog)


if __name__ == "__main__":
    raise SystemExit(main())
