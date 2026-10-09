#!/usr/bin/env python3
"""Stage an immutable, already-canary-verified GHCR runtime digest for PR review.

Only the trusted image release workflow may invoke --apply after the new
container passed a fresh-hosted-runner, no-Telegram-poller canary. Normal
AA runtime remains unchanged until the generated promotion PR merges.
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
IMAGE_RE = re.compile(r"ghcr\\.io/kodmial/aa-runtime@sha256:[0-9a-f]{64}")
ZERO = "0" * 64


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Promote a validated AA environment image.")
    parser.add_argument("--digest", required=True, help="sha256:<64 lowercase hex>")
    parser.add_argument("--apply", action="store_true", help="Write reviewed activation files")
    args = parser.parse_args(argv)
    digest = args.digest.strip()
    if not DIGEST_RE.fullmatch(digest) or digest == "sha256:" + ZERO:
        print("refusing invalid or unpublished placeholder digest", file=sys.stderr)
        return 1

    image_ref = f"ghcr.io/kodmial/aa-runtime@{digest}"
    workflow = RUNTIME_WORKFLOW.read_text(encoding="utf-8")
    if "group: aa-bot-runtime" not in workflow or "Reject duplicate runtime" not in workflow:
        print("refusing to promote runtime without single-poller guards", file=sys.stderr)
        return 1
    if "packages: read" not in workflow:
        print("runtime missing GHCR read permission", file=sys.stderr)
        return 1

    container_anchor = "    timeout-minutes: 330\n"
    checkout_anchor = (
        "      - name: Checkout repository\n"
        "        uses: actions/checkout@11d5960a326750d5838078e36cf38b85af677262 # v4\n"
    )
    runtime_anchor = "      - name: Build and validate AA knowledge runtime\n"
    setup_start = "      - name: Set up Python 3.12\n"
    setup_end = "      - name: Compute public cache keys\n"
    image_block = (
        "    container:\n"
        f"      image: {image_ref}\n"
        "      credentials:\n"
        "        username: ${{ github.actor }}\n"
        "        password: ${{ secrets.GITHUB_TOKEN }}\n"
    )
    preflight = (
        "      - name: Validate promoted dependency-image provenance\n"
        "        shell: bash\n"
        "        run: |\n"
        "          set -euo pipefail\n"
        "          python scripts/verify_runtime_image.py --pins-only\n"
        "          test \\"$(git rev-parse HEAD)\\" = \\"${{ github.sha }}\\"\n"
    )
    hot_gate = (
        "      - name: Verify image hot path and external model cache\n"
        "        shell: bash\n"
        "        run: python scripts/verify_runtime_image.py\n\n"
    )

    if "\n    container:\n" in workflow:
        match = IMAGE_RE.search(workflow)
        if match is None:
            print("existing container has no valid digest", file=sys.stderr)
            return 1
        new_workflow = IMAGE_RE.sub(image_ref, workflow)
    else:
        for anchor in (container_anchor, checkout_anchor, runtime_anchor, setup_start, setup_end):
            if workflow.count(anchor) != 1:
                print(f"cannot identify safe promotion anchor {anchor.strip()}", file=sys.stderr)
                return 1
        new_workflow = workflow.replace(container_anchor, container_anchor + image_block, 1)
        new_workflow = new_workflow.replace(checkout_anchor, checkout_anchor + preflight, 1)
        new_workflow = new_workflow.replace(runtime_anchor, hot_gate + runtime_anchor, 1)
        start = new_workflow.index(setup_start)
        end = new_workflow.index(setup_end, start)
        new_workflow = new_workflow[:start] + new_workflow[end:]
    if new_workflow.count(image_ref) != 1:
        print("promotion must pin exactly one job container", file=sys.stderr)
        return 1
    old_digest = DIGEST_FILE.read_text(encoding="utf-8").strip().splitlines()[-1].strip()
    print(f"previous: {old_digest}")
    print(f"validated: {image_ref}")
    if not args.apply:
        print("dry-run pass; use --apply only after the image canary passed")
        return 0
    DIGEST_FILE.write_text(image_ref + "\n", encoding="utf-8")
    RUNTIME_WORKFLOW.write_text(new_workflow, encoding="utf-8")
    print("activation files updated; open a PR, run CI/review and merge before /run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
