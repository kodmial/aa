#!/usr/bin/env python3
"""Fetch immutable AA source artifacts without modifying their bytes."""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "corpus" / "source.lock.json"
RAW_DIR = ROOT / "corpus" / "source" / "raw"
STATE = ROOT / "corpus" / "source" / "fetch-state.json"

USER_AGENT = "kodmial-aa-corpus-fetch/1.0"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response:
        return response.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    config = json.loads(LOCK.read_text(encoding="utf-8"))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    fetched = []
    for source in config["sources"]:
        data = fetch(source["url"])
        target = ROOT / source["raw_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

        if target.read_bytes() != data:
            raise RuntimeError(f"raw source changed while writing: {target}")

        fetched.append(
            {
                "id": source["id"],
                "url": source["url"],
                "path": source["raw_path"],
                "bytes": len(data),
                "sha256": sha256(data),
            }
        )

    core = (RAW_DIR / "AA.txt").read_bytes()
    for marker in (b"Chapter 1", b"Chapter 11", b"BILL'S STORY", b"A VISION FOR YOU"):
        if marker not in core:
            raise RuntimeError(f"required marker missing from AA.txt: {marker!r}")

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps({"version": 1, "sources": fetched}, indent=2) + "\n",
        encoding="utf-8",
    )

    for item in fetched:
        print(f"{item['id']}: {item['bytes']} bytes sha256={item['sha256']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"source fetch failed: {exc}", file=sys.stderr)
        raise
