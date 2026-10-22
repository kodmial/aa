#!/usr/bin/env python3
"""Stage the promoted persistent derived-index L2 pin for PR review (issue #429).

Only the trusted index publisher workflow may invoke ``--apply`` after the
encrypted E5+FAISS bundle was published (idempotently) to the persistent
GHCR OCI artifact and validated. Promotion updates only the immutable
digest pin ``docker/aa-index.digest``; consumers read that pin at restore
time and fail closed when it is a placeholder (bounded emergency rebuild).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGEST_FILE = ROOT / "docker" / "aa-index.digest"
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ZERO = "0" * 64
REFERENCE_RE = re.compile(r"^ghcr\.io/kodmial/aa-index@sha256:[0-9a-f]{64}$")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a validated AA index L2 digest pin.")
    parser.add_argument("--digest", required=True, help="sha256:<64 lowercase hex>")
    parser.add_argument("--fingerprint", required=True, help="16-hex cache fingerprint")
    parser.add_argument("--apply", action="store_true", help="Write the reviewed pin update")
    args = parser.parse_args(argv)
    digest = args.digest.strip()
    fingerprint = args.fingerprint.strip()
    if not DIGEST_RE.fullmatch(digest) or digest == "sha256:" + ZERO:
        print("refusing invalid or unpublished placeholder digest", file=sys.stderr)
        return 1
    if not re.fullmatch(r"[0-9a-f]{16}", fingerprint):
        print("refusing malformed cache fingerprint", file=sys.stderr)
        return 1
    old_reference = DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
    if not REFERENCE_RE.fullmatch(old_reference):
        print("existing pin file must carry the canonical aa-index reference", file=sys.stderr)
        return 1
    new_reference = f"ghcr.io/kodmial/aa-index@{digest}"
    print(f"previous: {old_reference}")
    print(f"validated: {new_reference} (cache fingerprint {fingerprint})")
    if new_reference == old_reference:
        print("L2 pin already active; no promotion PR needed.")
        return 0
    if not args.apply:
        print("dry-run pass; use --apply only after the trusted publisher succeeded")
        return 0
    DIGEST_FILE.write_text(new_reference + "\n", encoding="utf-8")
    print("L2 pin updated; open a PR, run CI/review and merge to activate it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
