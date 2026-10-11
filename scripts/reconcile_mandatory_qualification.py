#!/usr/bin/env python3
"""Reconcile mandatory exact-main qualification dispatch (issues #336/#345).

Observable idempotent post-merge trigger for the immutable tuple
(capability issue, qualification issue #7, exact merged main SHA,
contract version). Uses only the pure decisions in
``aa.qualification.mandatory_lifecycle``:

- no duplicate qualification when the same SHA already has a dispatch
  record or trustworthy final evidence;
- exactly one shared A-F run per SHA/config generation across all waiting
  capabilities (duplicate events never mint a second run);
- no acceptance of an old SHA as new;
- orphaned #7 ``automation:in-progress`` leases recover through a
  deterministic re-dispatch plan (never blind label strips or repeated
  ``/oc``).

Issue #345 hardening: this script never takes large GitHub JSON through
CLI argv or environment variables. The workflow paginates REST reads into
ephemeral chmod-600 files; this script loads inventory and tracker history
from those files with bounded streaming, validates schema, and fails
closed with a typed BLOCKED record (exit 2, no dispatch) on corrupt or
missing pages. Only small scalar flags travel on the command line.

Live GitHub access is isolated to small ``gh`` CLI calls so unit tests
exercise the pure path with ``--dispatches-file``/``--results-file``.
Privacy-safe: SHAs, issue numbers and verdicts only.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.mandatory_lifecycle import (  # noqa: E402
    MandatoryLifecycleError,
    MandatoryQualification,
    TrackerLease,
    TrustedResult,
    build_blocked_record,
    build_dispatch_record,
    build_dispatch_tuple,
    classify_runner_state,
    classify_tracker_lease,
    format_blocked_reason,
    format_dispatch_marker,
    format_shared_dispatch_marker,
    is_genuinely_active_run,
    load_capability_inventory_file,
    load_tracker_comments_file,
    orphaned_lease_recovery,
    parse_dispatch_markers,
    parse_shared_dispatch_markers,
    parse_trusted_results,
    plan_shared_qualification_dispatch,
    serialize_record,
    should_dispatch_qualification,
    tracker_comments_to_triples,
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


def _parse_capability_list(text: str) -> list[int]:
    out: list[int] = []
    for token in text.replace(",", " ").split():
        token = token.strip()
        if token.isdigit() and int(token) > 0:
            out.append(int(token))
    return sorted(set(out))


def _load_capabilities_file(path: str) -> list[int]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        decoded: Any = json.loads(text)
        if not isinstance(decoded, list):
            raise MandatoryLifecycleError("capabilities file must hold a JSON list")
        out: list[int] = []
        for entry in decoded:
            if isinstance(entry, int) and entry > 0:
                out.append(entry)
            elif isinstance(entry, str) and entry.strip().isdigit() and int(entry) > 0:
                out.append(int(entry))
        return sorted(set(out))
    return _parse_capability_list(text)


def _write_record(path: str, record: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path)
    target.write_text(serialize_record(record) + "\n", encoding="utf-8")
    try:
        target.chmod(0o600)
    except OSError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capability", type=int, default=0)
    parser.add_argument("--capabilities", type=str, default="")
    parser.add_argument("--capabilities-file", type=str, default="")
    parser.add_argument("--inventory-file", type=str, default="")
    parser.add_argument("--sha", type=str, default="")
    parser.add_argument("--dispatches-file", type=str, default="")
    parser.add_argument("--results-file", type=str, default="")
    parser.add_argument("--comments-file", type=str, default="")
    parser.add_argument("--has-in-progress-label", action="store_true")
    parser.add_argument("--has-active-run", action="store_true")
    parser.add_argument("--run-sha", type=str, default="")
    parser.add_argument("--run-conclusion", type=str, default="")
    parser.add_argument("--run-status", type=str, default="")
    parser.add_argument("--http-status", type=int, default=0)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--no-sha-match", action="store_true")
    parser.add_argument("--lease-age-s", type=float, default=0.0)
    parser.add_argument("--actor", type=str, default="")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--run-url", type=str, default="")
    parser.add_argument("--record-file", type=str, default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--post", action="store_true")
    args = parser.parse_args()

    try:
        sha = (args.sha or _current_main_sha()).strip().lower()
    except RuntimeError as exc:
        print(json.dumps({"blocked": True, "reason": str(exc)[:200]}))
        return 2

    try:
        capabilities: list[int] = []
        if args.inventory_file:
            capabilities = load_capability_inventory_file(args.inventory_file)
        if args.capabilities_file:
            extra_caps = _load_capabilities_file(args.capabilities_file)
            capabilities = sorted(set(capabilities) | set(extra_caps))
        if args.capabilities:
            extra_listed = _parse_capability_list(args.capabilities)
            capabilities = sorted(set(capabilities) | set(extra_listed))
        if args.capability and args.capability > 0:
            capabilities = sorted(set(capabilities) | {int(args.capability)})

        if args.comments_file:
            comments = load_tracker_comments_file(args.comments_file)
            bodies = [item.body for item in comments]
            dispatches = parse_dispatch_markers(bodies)
            shared = parse_shared_dispatch_markers(bodies)
            trusted = parse_trusted_results(tracker_comments_to_triples(comments))
        else:
            dispatch_bodies = _load_json_list(args.dispatches_file) if args.dispatches_file else []
            dispatches = parse_dispatch_markers(dispatch_bodies)
            shared = parse_shared_dispatch_markers(dispatch_bodies)
            if args.results_file:
                result_raw = json.loads(Path(args.results_file).read_text(encoding="utf-8"))
                trusted = [
                    TrustedResult(sha=str(item["sha"]).lower(), result=str(item["result"]).lower())
                    for item in result_raw
                ]
            else:
                trusted = []

        genuinely_active = is_genuinely_active_run(
            has_active_run=bool(args.has_active_run),
            run_sha=(args.run_sha or ""),
            current_sha=sha,
            run_conclusion=(args.run_conclusion or ""),
        )
        lease: TrackerLease = classify_tracker_lease(
            has_in_progress_label=bool(args.has_in_progress_label),
            has_active_run=genuinely_active,
            age_s=float(args.lease_age_s),
        )
        recovery = orphaned_lease_recovery(lease)
        print(
            json.dumps(
                {"lease": lease.state, "recovery": recovery.action, "detail": recovery.reason}
            )
        )

        if args.run_status or args.http_status:
            outcome = classify_runner_state(
                status=(args.run_status or "completed"),
                conclusion=(args.run_conclusion or None),
                http_status=(int(args.http_status) or None),
                sha_matches_head=not bool(args.no_sha_match),
                attempt=int(args.attempt),
                max_attempts=int(args.max_attempts),
            )
            print(
                json.dumps(
                    {
                        "runner": outcome.action,
                        "retry_allowed": outcome.retry_allowed,
                        "detail": outcome.reason,
                    }
                )
            )
            if outcome.action in ("retain", "blocked-terminal") and not capabilities:
                blocked = build_blocked_record(
                    capabilities=[],
                    sha=sha,
                    blocked_code="BLOCKED_RUNNER_TERMINAL",
                    detail=outcome.reason[:200],
                )
                _write_record(args.record_file, blocked)
                print(json.dumps({"blocked": True, "reason": outcome.reason}))
                return 0

        if not capabilities:
            print("no capability given: lease classified only, no dispatch decision")
            empty_record = build_blocked_record(
                capabilities=[],
                sha=sha,
                blocked_code="BLOCKED_NO_WAITING_CAPABILITIES",
                detail="lease classified only",
            )
            _write_record(args.record_file, empty_record)
            print(serialize_record(empty_record))
            return 0

        plan = plan_shared_qualification_dispatch(
            capabilities=capabilities,
            current_main_sha=sha,
            dispatches=dispatches,
            results=trusted,
            lease=lease,
            shared_markers=shared,
        )
        print(
            json.dumps(
                {
                    "dispatch": plan.should_dispatch,
                    "reason": plan.reason,
                    "owner": plan.owner_capability,
                    "waiting": list(plan.waiting_capabilities),
                }
            )
        )
        if plan.should_dispatch:
            owner_item: MandatoryQualification = build_dispatch_tuple(
                int(plan.owner_capability or plan.waiting_capabilities[0]), sha
            )
            shared_marker = format_shared_dispatch_marker(
                list(plan.waiting_capabilities), owner_item.sha, contract=owner_item.contract
            )
            print(
                f"immutable shared tuple: qualification=7 sha={owner_item.sha} "
                f"contract={owner_item.contract} "
                f"waiting={','.join(str(c) for c in plan.waiting_capabilities)} "
                f"owner={plan.owner_capability}"
            )
            print(f"dispatch marker: {format_dispatch_marker(owner_item)}")
            print(f"shared marker: {shared_marker}")
            record = build_dispatch_record(
                capabilities=list(plan.waiting_capabilities),
                sha=owner_item.sha,
                contract=owner_item.contract,
                actor=args.actor,
                run_id=args.run_id,
                run_url=args.run_url,
            )
            _write_record(args.record_file, record)
            print(serialize_record(record))
            # Backward-compatible per-capability verdict for single-cap callers.
            if len(plan.waiting_capabilities) == 1:
                single = should_dispatch_qualification(
                    capability=int(plan.waiting_capabilities[0]),
                    current_main_sha=owner_item.sha,
                    dispatches=dispatches,
                    results=trusted,
                )
                print(json.dumps({"single_dispatch": single.dispatch, "reason": single.reason}))
            if args.post and not args.dry_run:
                _run_gh(["issue", "comment", "7", "--body", shared_marker])
                _run_gh(
                    [
                        "workflow",
                        "run",
                        "aa-self-proving-qualification.yml",
                        "--ref",
                        "main",
                        "-f",
                        f"required_sha={owner_item.sha}",
                    ]
                )
        else:
            # Healthy idempotent reuse (trusted result present, already
            # dispatched, shared marker present, active lease retain) is
            # not a runner failure: preserve the typed code embedded in
            # the plan reason instead of mislabeling it as terminal.
            _code_match = re.search(r"BLOCKED_[A-Z_]+", plan.reason)
            _reuse_code = _code_match.group(0) if _code_match else "BLOCKED_NO_WAITING_CAPABILITIES"
            blocked_record = build_blocked_record(
                capabilities=list(plan.waiting_capabilities),
                sha=sha,
                blocked_code=_reuse_code,
                detail=plan.reason[:200],
            )
            # Reuse/no-dispatch is informational, not an error: persist the
            # machine-reconstructable state and exit successfully.
            _write_record(args.record_file, blocked_record)
            print(serialize_record(blocked_record))
        return 0
    except MandatoryLifecycleError as exc:
        reason = format_blocked_reason("BLOCKED_TRACKER_HISTORY_INCOMPLETE", detail=str(exc)[:200])
        print(json.dumps({"blocked": True, "reason": reason}))
        try:
            fallback = build_blocked_record(
                capabilities=[],
                sha=sha,
                blocked_code="BLOCKED_TRACKER_HISTORY_INCOMPLETE",
                detail=str(exc)[:200],
            )
            _write_record(args.record_file, fallback)
            print(serialize_record(fallback))
        except MandatoryLifecycleError:
            pass
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
