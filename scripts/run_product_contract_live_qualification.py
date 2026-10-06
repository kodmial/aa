#!/usr/bin/env python3
"""Live Product Contract qualification runner for authoritative issue #7.

Executes all live lanes (scenarios 1-41 plus folded voice 1-16 with
deterministic resource gates) against the exact production boundary on
the exact required main SHA and writes privacy-safe evidence:

- ``result.json`` (deterministic PASS/FAIL/INCOMPLETE/STALE + marker);
- ``product-contract-live-summary.json`` (ids/digests/counts/latencies).

Exit codes: 0 PASS, 1 FAIL, 2 INCOMPLETE, 3 STALE.
Logs carry ids, digests, counts and latencies only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.opencode.errors import OpenCodeRateLimitError  # noqa: E402
from aa.qualification.product_contract_live import (  # noqa: E402
    EXIT_BY_STATUS,
    ProductContractLiveError,
    build_result_marker,
    checked_out_sha,
    evaluate_live,
    validate_exact_sha,
    working_tree_clean,
)


def _fail_incomplete(out_dir: Path, *, reason: str, main_sha: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": "INCOMPLETE", "reason": reason, "main_sha": main_sha}
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"product-contract live qualification INCOMPLETE: {reason}", file=sys.stderr)
    return EXIT_BY_STATUS["INCOMPLETE"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Live Product Contract qualification runner (issue #7 lanes)."
    )
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval-product-contract-live-out")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args(argv)

    try:
        expected_sha = validate_exact_sha(args.main_sha)
    except ProductContractLiveError as exc:
        print(f"product-contract live qualification failed: {exc}", file=sys.stderr)
        return EXIT_BY_STATUS["INCOMPLETE"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        checked = checked_out_sha(ROOT)
    except ProductContractLiveError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    if checked != expected_sha:
        payload = {
            "result": "STALE",
            "reason": "checked-out SHA is not the expected exact SHA",
            "main_sha": expected_sha,
        }
        (out_dir / "result.json").write_text(
            json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        print("product-contract live qualification STALE: SHA mismatch", file=sys.stderr)
        return EXIT_BY_STATUS["STALE"]
    if not working_tree_clean(ROOT):
        return _fail_incomplete(out_dir, reason="working tree is not clean", main_sha=expected_sha)

    try:
        summary = evaluate_live(expected_sha, repo_root=ROOT, run_id=str(args.run_id))
    except OpenCodeRateLimitError:
        (out_dir / "restart-required.json").write_text(
            json.dumps(
                {
                    "reason_code": "OPENCODE_429_RESTART_REQUIRED",
                    "main_sha": expected_sha,
                    "run_id": str(args.run_id),
                },
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            "product-contract live qualification: provider 429, runner restart required",
            file=sys.stderr,
        )
        return 75
    except ProductContractLiveError as exc:
        return _fail_incomplete(out_dir, reason=str(exc), main_sha=expected_sha)
    except Exception as exc:
        # Fail-closed machine-readable INCOMPLETE (issue #150): an unhandled
        # harness/import failure (for example a missing third-party module on
        # a minimal runner) must still emit result.json instead of crashing
        # with no evidence. Only the exception type travels outward.
        return _fail_incomplete(
            out_dir,
            reason=f"live-harness-error:{type(exc).__name__}",
            main_sha=expected_sha,
        )

    (out_dir / "product-contract-live-summary.json").write_text(
        json.dumps(summary.to_dict(), sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = build_result_marker(sha=expected_sha, status=summary.status, run=str(args.run_id))
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "result": summary.status,
                "main_sha": expected_sha,
                "run_id": str(args.run_id),
                "lanes": [
                    {
                        "lane": lane.lane,
                        "status": lane.status,
                        "passed": len(lane.passed),
                        "failed": len(lane.failed),
                        "incomplete": len(lane.incomplete),
                    }
                    for lane in summary.lanes
                ],
                "marker": marker,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "result": summary.status,
                "lanes": [{"lane": lane.lane, "status": lane.status} for lane in summary.lanes],
            }
        )
    )
    return EXIT_BY_STATUS[summary.status]


if __name__ == "__main__":
    raise SystemExit(main())
