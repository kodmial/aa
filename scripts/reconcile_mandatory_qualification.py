#!/usr/bin/env python3
"""Reconcile mandatory exact-main qualification dispatch (issue #336).

Observable idempotent post-merge trigger for the immutable tuple
(capability issue, qualification issue #7, exact merged main SHA,
contract version). Uses only the pure decisions in
``aa.qualification.mandatory_lifecycle``:

- no duplicate qualification when the same SHA already has a dispatch
  record or trustworthy final evidence;
- no acceptance of an old SHA as new;
- orphaned #7 ``automation:in-progress`` leases recover through a
  deterministic re-dispatch plan (never blind label strips or repeated
  ``/oc``).

Live GitHub access is isolated to small ``gh`` CLI calls so unit tests
exercise the pure path with ``--dispatches-file``/``--results-file``.
Privacy-safe: SHAs, issue numbers and verdicts only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.mandatory_lifecycle import (  # noqa: E402
    MandatoryQualification,
    TrustedResult,
    build_dispatch_tuple,
    classify_tracker_lease,
    format_dispatch_marker,
    orphaned_lease_recovery,
    parse_dispatch_markers,
    parse_trusted_results,
    should_dispatch_qualification,
)


def _run_gh(args: list[str]) -> str:
    proc = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} failed: {proc.stderr.strip()[:200]}")
    return proc.stdout


def _current_main_sha() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    if proc.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed")
    return proc.stdout.strip()


def _load_json_list(path: str) -> list[str]:
    raw: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"{path} must hold a JSON list")
    return [str(item) for item in raw]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability", type=int, default=0)
    parser.add_argument("--sha", type=str, default="")
    parser.add_argument("--dispatches-file", type=str, default="")
    parser.add_argument("--results-file", type=str, default="")
    parser.add_argument("--comments-file", type=str, default="")
    parser.add_argument("--has-in-progress-label", action="store_true")
    parser.add_argument("--has-active-run", action="store_true")
    parser.add_argument("--lease-age-s", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()

    sha = (args.sha or _current_main_sha()).strip().lower()
    capability = args.capability

    if args.comments_file:
        raw = json.loads(Path(args.comments_file).read_text(encoding="utf-8"))
        bodies = [str(item.get("body", "")) for item in raw]
        dispatches = parse_dispatch_markers(bodies)
        trusted = parse_trusted_results(
            [
                (
                    str(item.get("body", "")),
                    str(item.get("association", "")),
                    str(item.get("login", "")),
                )
                for item in raw
            ]
        )
    else:
        dispatch_bodies = _load_json_list(args.dispatches_file) if args.dispatches_file else []
        dispatches = parse_dispatch_markers(dispatch_bodies)
        if args.results_file:
            result_raw = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
            trusted = [
                TrustedResult(sha=str(item["sha"]).lower(), result=str(item["result"]).lower())
                for item in result_raw
            ]
        else:
            trusted = []

    lease = classify_tracker_lease(
        has_in_progress_label=args.has_in_progress_label,
        has_active_run=args.has_active_run,
        age_s=args.lease_age_s,
    )
    recovery = orphaned_lease_recovery(lease)
    print(
        json.dumps({"lease": lease.state, "recovery": recovery.action, "detail": recovery.reason})
    )

    if capability <= 0:
        print("no capability given: lease classified only, no dispatch decision")
        return 0

    item: MandatoryQualification = build_dispatch_tuple(capability, sha)
    decision = should_dispatch_qualification(
        capability=capability,
        current_main_sha=item.sha,
        dispatches=dispatches,
        results=trusted,
    )
    print(json.dumps({"dispatch": decision.dispatch, "reason": decision.reason}))
    print(
        f"immutable tuple: capability={item.capability} qualification=7 sha={item.sha} "
        f"contract={item.contract}"
    )
    if decision.dispatch:
        print(f"dispatch marker: {format_dispatch_marker(item)}")
        if args.post and not args.dry_run:
            _run_gh(["issue", "comment", "7", "--body", format_dispatch_marker(item)])
            _run_gh(
                [
                    "workflow",
                    "run",
                    "aa-self-proving-qualification.yml",
                    "--ref",
                    "main",
                    "-f",
                    f"required_sha={item.sha}",
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
