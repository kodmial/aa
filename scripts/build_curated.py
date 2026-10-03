#!/usr/bin/env python3
"""Build a reduced corpus only by copying exact byte ranges from a source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.read_bytes()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    expected = manifest.get("source_sha256")
    actual = digest(source)
    if not expected or expected != actual:
        raise SystemExit(f"source checksum mismatch: expected={expected!r} actual={actual!r}")

    ranges = manifest.get("keep_ranges")
    if not isinstance(ranges, list) or not ranges:
        raise SystemExit("manifest must contain non-empty keep_ranges")

    chunks: list[bytes] = []
    previous_end = 0
    for index, item in enumerate(ranges):
        start = item.get("start")
        end = item.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            raise SystemExit(f"range {index}: start/end must be integers")
        if start < 0 or end <= start or end > len(source):
            raise SystemExit(f"range {index}: invalid bounds {start}:{end}")
        if start < previous_end:
            raise SystemExit(f"range {index}: ranges overlap or are out of order")

        chunk = source[start:end]
        if chunk != source[start:end]:
            raise AssertionError("exact-copy invariant violated")
        chunks.append(chunk)
        previous_end = end

    output = b"".join(chunks)
    if not output:
        raise SystemExit("refusing to write an empty curated corpus")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)

    # Strong postcondition: output is exactly the concatenation of selected
    # source byte ranges. There is no decode/encode, normalization, or rewrite.
    expected_output = b"".join(source[r["start"] : r["end"]] for r in ranges)
    if args.output.read_bytes() != expected_output:
        raise AssertionError("written output differs from selected source bytes")

    print(
        json.dumps(
            {
                "source_sha256": actual,
                "output_sha256": digest(output),
                "source_bytes": len(source),
                "output_bytes": len(output),
                "kept_ranges": len(ranges),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
