#!/usr/bin/env python3
"""Promote a validated GHCR runtime image digest (issue #298).

Copies one immutable digest into BOTH the digest record and the bot runtime
container pin so they can never drift apart:

- docker/aa-runtime.digest (single authoritative line)
- .github/workflows/aa-runtime.yml `container.image` pin

Only run after .github/workflows/aa-runtime-image.yml publishes from exact
main AND its fresh-runner canary passes. Refuses the zero placeholder and
any malformed digest. Prints the one-commit rollback command.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIGEST_FILE = ROOT / "docker" / "aa-runtime.digest"
RUNTIME_WORKFLOW = ROOT / ".github" / "workflows" / "aa-runtime.yml"

DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IMAGE_RE = re.compile(r"ghcr\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}")
ZERO = "0" * 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a validated runtime image digest.")
    parser.add_argument("--digest", required=True, help="sha256:<64 hex> from the image workflow")
    parser.add_argument("--apply", action="store_true", help="Write files (default is check only)")
    args = parser.parse_args(argv)
    digest = args.digest.strip()
    if not DIGEST_RE.match(digest):
        print(f"refusing malformed digest: {digest!r}", file=sys.stderr)
        return 1
    if digest == "sha256:" + ZERO:
        print("refusing placeholder zero digest", file=sys.stderr)
        return 1
    image_ref = f"ghcr.io/kodmial/aa-runtime@{digest}"
    workflow_text = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    if IMAGE_RE.search(workflow_text) is None:
        print("aa-runtime.yml has no GHCR digest pin to promote", file=sys.stderr)
        return 1
    new_workflow = IMAGE_RE.sub(image_ref, workflow_text)
    old_digest = DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
    print(f"current: {old_digest}")
    print(f"promote: {image_ref}")
    if old_digest == image_ref and IMAGE_RE.search(workflow_text) is not None:
        print("already promoted")
        return 0
    if not args.apply:
        print("check ok (re-run with --apply to write both files)")
        return 0
    DIGEST_FILE.write_text(image_ref + "\n", encoding="utf-8")
    RUNTIME_WORKFLOW.write_text(new_workflow, encoding="utf-8")
    print("promoted. Rollback: git revert this commit (removes container pin).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
