#!/usr/bin/env python3
"""Bounded Gate C diagnostic canary for exact-current AA main (kodmial/aa#514).

Runs the provider-free observer/server self-test and one or two existing core
Gate C scenarios on the already-prepared runtime image, before the unchanged
18-case campaign. It never issues a product PASS/FAIL and never replaces the
authoritative Gate C/E/F verdict.

Exit codes: 0 READY, 2 BLOCKED, 3 INCONCLUSIVE, 4 FAIL.
Only bounded, privacy-safe diagnostic fields are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.gate_c_canary import (  # noqa: E402
    DEFAULT_LIVE_PROBE_BUDGET_S,
    DEFAULT_PREFLIGHT_BUDGET_S,
    DEFAULT_SCENARIO_LIMIT,
    CanaryResult,
    RealPathEvidence,
    SelfTestEvidence,
    evaluate_canary,
    run_canary,
    write_canary_result,
)
from aa.qualification.product_contract_live import (  # noqa: E402
    ProductContractLiveError,
    checked_out_sha,
    validate_exact_sha,
    working_tree_clean,
)


def _log(message: str) -> None:
    print(f"gate-c-canary: {message}", flush=True)


def _blocked_before_run(
    *,
    out_dir: Path,
    main_sha: str,
    run_id: str,
    reason: str,
    preflight_budget_s: float,
    live_probe_budget_s: float,
) -> int:
    """Persist a typed BLOCKED artifact when no probe is allowed to start."""
    result = evaluate_canary(
        main_sha=main_sha,
        run_id=run_id,
        self_test=SelfTestEvidence(passed=False, missing_fields=(reason,)),
        real_path=RealPathEvidence(error=reason),
        preflight_budget_s=preflight_budget_s,
        live_probe_budget_s=live_probe_budget_s,
        elapsed_s=0.0,
    )
    result = replace(result, reason=str(reason)[:64])
    path = write_canary_result(out_dir, result)
    _print_summary(result, path)
    return result.exit_code


def _print_summary(result: CanaryResult, path: Path) -> None:
    payload = result.to_dict()
    coverage = payload.get("evidence_coverage", {})
    _log(
        "status={status} reason={reason} exit={exit_code} real_turns={turns} "
        "budget={budget} elapsed_s={elapsed}".format(
            status=payload["status"],
            reason=payload["reason"],
            exit_code=result.exit_code,
            turns=payload.get("real_turn_count", 0),
            budget=payload.get("budget", {}).get("category", ""),
            elapsed=payload.get("budget", {}).get("elapsed_s", 0),
        )
    )
    _log(
        "coverage stage={stage} attempt={attempt} causality={causality} "
        "server_events={server_events}".format(
            stage=coverage.get("stage", "missing"),
            attempt=coverage.get("attempt", "missing"),
            causality=coverage.get("causality", "missing"),
            server_events=coverage.get("server_events", "unobserved"),
        )
    )
    missing = payload.get("missing_fields") or []
    if missing:
        _log("missing_fields=" + ",".join(str(item)[:64] for item in missing))
    _log(f"evidence={path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded Gate C diagnostic canary.")
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "live-out" / "gate-c-canary",
        help="Privacy-safe canary artifact directory",
    )
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--preflight-budget-s",
        type=float,
        default=DEFAULT_PREFLIGHT_BUDGET_S,
        help="Total canary budget (default 600s)",
    )
    parser.add_argument(
        "--live-probe-budget-s",
        type=float,
        default=DEFAULT_LIVE_PROBE_BUDGET_S,
        help="Real-path probe budget (default 300s)",
    )
    parser.add_argument(
        "--scenario-limit",
        type=int,
        default=DEFAULT_SCENARIO_LIMIT,
        help="Existing core scenarios to probe (1 or 2)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # A stale READY file must never authorize this run's Gate C if the
    # process is killed before the new artifact is written.
    (out_dir / "gate-c-canary-result.json").unlink(missing_ok=True)

    try:
        main_sha = validate_exact_sha(args.main_sha)
    except ProductContractLiveError as exc:
        _log(f"invalid main SHA: {exc}")
        return _blocked_before_run(
            out_dir=out_dir,
            main_sha=str(args.main_sha or "")[:64],
            run_id=str(args.run_id),
            reason="invalid-main-sha",
            preflight_budget_s=float(args.preflight_budget_s),
            live_probe_budget_s=float(args.live_probe_budget_s),
        )

    try:
        checked = checked_out_sha(ROOT)
    except ProductContractLiveError:
        checked = ""
    if checked != main_sha:
        _log("checked-out SHA is not the exact requested main SHA")
        return _blocked_before_run(
            out_dir=out_dir,
            main_sha=main_sha,
            run_id=str(args.run_id),
            reason="main-sha-mismatch",
            preflight_budget_s=float(args.preflight_budget_s),
            live_probe_budget_s=float(args.live_probe_budget_s),
        )
    if not working_tree_clean(ROOT):
        _log("working tree is not clean")
        return _blocked_before_run(
            out_dir=out_dir,
            main_sha=main_sha,
            run_id=str(args.run_id),
            reason="working-tree-dirty",
            preflight_budget_s=float(args.preflight_budget_s),
            live_probe_budget_s=float(args.live_probe_budget_s),
        )

    scenario_limit = max(1, min(int(args.scenario_limit), DEFAULT_SCENARIO_LIMIT))
    try:
        result = asyncio.run(
            run_canary(
                main_sha=main_sha,
                run_id=str(args.run_id),
                out_dir=out_dir,
                preflight_budget_s=float(args.preflight_budget_s),
                live_probe_budget_s=float(args.live_probe_budget_s),
                scenario_limit=scenario_limit,
            )
        )
    except KeyboardInterrupt:
        _log("interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed with a typed artifact
        _log(f"canary surface error: {type(exc).__name__}")
        return _blocked_before_run(
            out_dir=out_dir,
            main_sha=main_sha,
            run_id=str(args.run_id),
            reason=f"surface-error-{type(exc).__name__}"[:64],
            preflight_budget_s=float(args.preflight_budget_s),
            live_probe_budget_s=float(args.live_probe_budget_s),
        )

    _print_summary(result, out_dir / "gate-c-canary-result.json")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
