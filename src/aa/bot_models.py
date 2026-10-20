"""AA-owned Telegram bot model policy; independent of Continuum coding routes."""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Mapping
from pathlib import Path

from aa.config import DEFAULT_PRIMARY_MODEL

_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._/-]{0,120}")


def resolve_bot_models(source: Mapping[str, str] | None = None) -> tuple[str, str]:
    """Only AA_BOT_* selects bot models; OPENCODE_MODELS is a coding route."""
    values = os.environ if source is None else source
    primary = (
        (values.get("AA_BOT_PRIMARY_MODEL", "") or "").strip() or DEFAULT_PRIMARY_MODEL
    )
    fallback = (values.get("AA_BOT_FALLBACK_MODEL", "") or "").strip() or primary
    for role, value in (("primary", primary), ("fallback", fallback)):
        if not _MODEL.fullmatch(value) or ".." in value or "//" in value:
            raise ValueError(f"invalid AA bot {role} provider/model identifier")
    return primary, fallback


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve AA-owned production model pointers"
    )
    parser.add_argument("--github-output", required=True, type=Path)
    args = parser.parse_args()
    primary, fallback = resolve_bot_models()
    with args.github_output.open("a", encoding="utf-8") as stream:
        stream.write(f"primary={primary}\nfallback={fallback}\n")
    print("AA bot primary/fallback model identifiers validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
