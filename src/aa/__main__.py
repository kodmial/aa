"""Command-line entrypoint for the AA Telegram worker."""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from aa import __version__
from aa.app import create_application
from aa.config import Settings
from aa.logging import configure_logging
from aa.opencode.errors import OpenCodeRateLimitError


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="aa-worker",
        description="Minimal AA Telegram worker foundation (no live calls yet).",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Boot configuration and components, then exit without running.",
    )
    parser.add_argument(
        "--require-token",
        action="store_true",
        help="Fail --check when TELEGRAM_BOT_TOKEN is missing.",
    )
    return parser


async def _run_worker(settings: Settings) -> int:
    app = create_application(settings)
    loop = asyncio.get_running_loop()

    def _request_stop() -> None:
        asyncio.ensure_future(app.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except (NotImplementedError, RuntimeError):
            continue
    await app.run()
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI main (sync wrapper around the asyncio worker)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        settings = Settings.from_env()
        settings.validate(require_bot_token=args.require_token)
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level)

    if args.check:
        # Boot path: construct the app, start/stop once without blocking.
        async def _check() -> None:
            app = create_application(settings)
            await app.start()
            await app.stop()

        asyncio.run(_check())
        print("ok")
        return 0

    if not settings.has_bot_token:
        print(
            "TELEGRAM_BOT_TOKEN is not set; refusing to run the worker.",
            file=sys.stderr,
        )
        return 2
    try:
        return asyncio.run(_run_worker(settings))
    except OpenCodeRateLimitError:
        print("OpenCode 429: runner restart required", file=sys.stderr)
        return 75


if __name__ == "__main__":
    raise SystemExit(main())
