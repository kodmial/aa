#!/usr/bin/env python3
"""Shared OpenCode book-tool entrypoint (issue #18).

All four narrow tools (``book_search``, ``book_read``, ``book_expand``,
``book_section``) delegate to the single shared implementation in
``aa.retrieval.book_tools`` over the RU-first hybrid index. The
TypeScript wrappers under ``.opencode/tools/`` invoke this script; it
is also useful for smoke checks. Logs carry only IDs, counts, sizes
and digests, never user queries or corpus text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.retrieval.book_tools import (  # noqa: E402
    book_expand,
    book_read,
    book_search,
    book_section,
)
from aa.retrieval.index import open_hybrid_index  # noqa: E402

DEFAULT_INDEX_DIR = ROOT / "corpus" / "generated" / "retrieval"


def _fail(message: str) -> int:
    print(f"book tool failed: {message}", file=sys.stderr)
    return 1


def _load_input(raw: str) -> object:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"input is not valid JSON: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: dispatch one book tool call and print JSON."""
    parser = argparse.ArgumentParser(description="Run one RU-first book tool.")
    parser.add_argument("tool", choices=("book_search", "book_read", "book_expand", "book_section"))
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--input-json", default="", help="Tool input as a JSON string.")
    args = parser.parse_args(argv)

    try:
        payload = _load_input(args.input_json) if args.input_json else {}
    except ValueError as exc:
        return _fail(str(exc))
    if not isinstance(args.index_dir, Path):
        return _fail("index dir is invalid")
    try:
        index = open_hybrid_index(args.index_dir)
    except ValueError as exc:
        return _fail(str(exc))

    try:
        if args.tool == "book_search":
            result = book_search(index, payload)
        elif args.tool == "book_read":
            if not isinstance(payload, dict):
                raise ValueError("book_read input must be a JSON object")
            result = book_read(
                index,
                payload.get("chunk_id"),
                expected_ru_version=payload.get("expected_ru_version"),
            )
        elif args.tool == "book_expand":
            if not isinstance(payload, dict):
                raise ValueError("book_expand input must be a JSON object")
            result = book_expand(
                index,
                payload.get("chunk_id"),
                before=payload.get("before", 1),
                after=payload.get("after", 1),
                expected_ru_version=payload.get("expected_ru_version"),
            )
        else:
            if not isinstance(payload, dict):
                raise ValueError("book_section input must be a JSON object")
            result = book_section(
                index,
                payload.get("section_id"),
                chunk_offset=payload.get("chunk_offset", 0),
                chunk_limit=payload.get("chunk_limit", 8),
                expected_ru_version=payload.get("expected_ru_version"),
            )
    except ValueError as exc:
        return _fail(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
