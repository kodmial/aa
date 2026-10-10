#!/usr/bin/env python3
"""Admission validation for the mandatory exact-main qualification policy.

A product capability whose Definition of Done specifies an exact-main
live gate must carry the explicit typed machine relation
``<!-- automation-qualification: #7 -->``. A missing marker is reported
as a setup error before the first issue snapshot rather than silently
closed (issue #336).

Usage:
    python scripts/verify_task_qualification_policy.py --body-file PATH
    python scripts/verify_task_qualification_policy.py --body "<issue body>"

Exit codes: 0 admission ok, 2 setup error (missing/invalid marker).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.mandatory_lifecycle import (  # noqa: E402
    pr_reference_kind,
    validate_pr_body_for_capability,
    validate_task_admission,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--body-file", type=str, default="")
    group.add_argument("--body", type=str, default="")
    parser.add_argument("--capability", type=int, default=0)
    parser.add_argument("--pr-body-file", type=str, default="")
    args = parser.parse_args()

    if args.body_file:
        body = Path(args.body_file).read_text(encoding="utf-8")
    else:
        body = args.body

    ok, reason = validate_task_admission(body)
    print(reason)
    print(f"required PR reference: {pr_reference_kind(body)}")
    if not ok:
        return 2
    if args.pr_body_file:
        pr_body = Path(args.pr_body_file).read_text(encoding="utf-8")
        pr_ok, pr_reason = validate_pr_body_for_capability(body, pr_body, args.capability)
        print(pr_reason)
        if not pr_ok:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
