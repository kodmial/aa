#!/usr/bin/env python3
"""Bounded Gate C canary preflight (kodmial/aa#454).

Runs the deterministic observer/server self-test and at most two real
production turns through the existing Gate C lane before the unchanged
18-scenario campaign. ``READY`` is the only outcome that authorizes the
campaign; every other typed status keeps it closed.

Exit codes: 0 READY, 1 FAIL, 2 BLOCKED, 3 INCONCLUSIVE, 75 provider 429
restart required. Artifacts contain fixed-vocabulary diagnostics only:
no prompt, reply, book passage, credential or session id.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.opencode.errors import OpenCodeRateLimitError  # noqa: E402
from aa.qualification.gate_c_canary import (  # noqa: E402
    CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
    CATEGORY_OPENCODE_UNAVAILABLE,
    CATEGORY_PRECONDITION_FAILED,
    CATEGORY_PRIVACY_VIOLATION,
    CATEGORY_REAL_PATH_INCOMPLETE,
    CATEGORY_SELF_TEST_FAILED,
    DEFAULT_LIVE_BUDGET_S,
    DEFAULT_PREFLIGHT_BUDGET_S,
    EXIT_BY_STATUS,
    MIN_BUDGET_S,
    STATUS_BLOCKED,
    STATUS_INCONCLUSIVE,
    CanaryDecision,
    build_canary_result,
    build_real_path_report,
    evaluate_canary,
    privacy_markers,
    privacy_scan,
    run_bounded_real_path,
    run_observer_self_test,
)
from aa.qualification.live_partial_evidence import (  # noqa: E402
    validated_partial_evidence,
)
from aa.qualification.live_watchdog import (  # noqa: E402
    ENV_PROGRESS_PATH,
    read_live_progress,
)
from aa.qualification.product_contract_live import (  # noqa: E402
    ProductContractLiveError,
    checked_out_sha,
    gate_c_canary_scenarios,
    run_live_telegram_evidence_lane,
    validate_exact_sha,
    working_tree_clean,
)

EXIT_RESTART_REQUIRED = 75
CANARY_RESULT_NAME = "gate-c-canary-result.json"
SELF_TEST_NAME = "observer-self-test.json"
PROGRESS_NAME = "canary-progress.json"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _pinned_opencode_version() -> str:
    try:
        return (ROOT / "docker" / "opencode.version").read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _installed_opencode_version() -> str:
    try:
        proc = subprocess.run(
            ["opencode", "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()[:128]


def _version_matches(pinned: str, installed: str) -> bool:
    if not pinned or not installed:
        return False
    safe = re.escape(pinned)
    return bool(re.search(rf"(^|[^0-9.]){safe}([^0-9.]|$)", installed))


def _artifact_paths(out_dir: Path) -> list[str]:
    names: list[str] = []
    for name in (SELF_TEST_NAME, "self-test-progress.json", PROGRESS_NAME):
        path = out_dir / name
        if path.is_file():
            try:
                names.append(str(path.relative_to(out_dir)))
            except ValueError:
                names.append(path.name)
    for directory in ("attempt-observer", "canary-attempt-observer"):
        path = out_dir / directory
        if path.is_dir():
            names.append(f"{directory}/")
    # The result file is written immediately after this list is assembled.
    names.append(CANARY_RESULT_NAME)
    return names


def _finish(
    *,
    out_dir: Path,
    main_sha: str,
    run_id: str,
    self_test: dict[str, Any],
    real_path: dict[str, Any] | None,
    decision: CanaryDecision,
    preflight_budget_s: float,
    live_budget_s: float,
    started: float,
    extra: dict[str, Any] | None = None,
) -> int:
    elapsed = max(0.0, time.monotonic() - started)
    # Final defense in depth: the canary-owned artifacts are about to be
    # uploaded. Re-scan them for prompt fragments, artificial secrets, auth
    # headers and raw SSE text; any hit downgrades the whole preflight to a
    # typed privacy BLOCK before the result is persisted.
    privacy_ok, leaked_name = privacy_scan(
        [
            out_dir / SELF_TEST_NAME,
            out_dir / "self-test-progress.json",
            out_dir / "attempt-observer",
            out_dir / PROGRESS_NAME,
            out_dir / "canary-attempt-observer",
        ],
        privacy_markers(),
    )
    if not privacy_ok:
        self_test = dict(self_test)
        self_test["privacy_safe"] = False
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRIVACY_VIOLATION,
            "artifacts",
            f"privacy scan found a marker in {str(leaked_name)[:96]}",
        )
    result = build_canary_result(
        main_sha=main_sha,
        run_id=run_id,
        self_test=self_test,
        real_path=real_path,
        decision=decision,
        preflight_budget_s=preflight_budget_s,
        live_budget_s=live_budget_s,
        elapsed_s=elapsed,
        privacy_safe_paths=_artifact_paths(out_dir),
        extra=extra,
    )
    _write_json(out_dir / CANARY_RESULT_NAME, result)
    print(
        json.dumps(
            {
                "status": decision.status,
                "category": decision.category,
                "missing_field": decision.missing_field,
                "real_turns": int((real_path or {}).get("real_turns", 0) or 0),
                "budget_expiration_category": str(
                    (real_path or {}).get("budget_expiration_category", "none")
                ),
                "elapsed_s": round(elapsed, 3),
                "product_qualified": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return EXIT_BY_STATUS[decision.status]


def _write_restart_marker(out_dir: Path, *, main_sha: str, run_id: str) -> None:
    _write_json(
        out_dir / "restart-required.json",
        {
            "reason_code": "OPENCODE_429_RESTART_REQUIRED",
            "main_sha": main_sha,
            "run_id": str(run_id)[:64],
            "stage": "gate-c-canary",
        },
    )


async def _run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preflight_budget_s = max(MIN_BUDGET_S, float(args.preflight_budget_s))
    live_budget_s = max(MIN_BUDGET_S, float(args.live_budget_s))
    run_id = str(args.run_id)
    main_sha = str(args.main_sha or "").strip().lower()
    scenario_families = [family for family, _, _ in gate_c_canary_scenarios()]
    extra = {
        "canary_mode": True,
        "scenario_families": scenario_families,
        "gate_scenarios_executed": 0,
    }

    try:
        expected_sha = validate_exact_sha(main_sha)
    except ProductContractLiveError:
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRECONDITION_FAILED,
            "main_sha",
            "main_sha must be an exact 40-hex SHA",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=main_sha,
            run_id=run_id,
            self_test={},
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )

    try:
        checked = checked_out_sha(ROOT)
    except ProductContractLiveError:
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRECONDITION_FAILED,
            "checked_out_sha",
            "cannot determine checked-out SHA",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test={},
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )
    if checked != expected_sha:
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRECONDITION_FAILED,
            "checked_out_sha",
            "checked-out SHA is not the expected exact SHA",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test={},
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )
    if not working_tree_clean(ROOT):
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_PRECONDITION_FAILED,
            "working_tree",
            "working tree is not clean",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test={},
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )

    # 1. Deterministic observer/server self-test. It uses the real client,
    # observer, server probe and watchdog writer against a controlled local
    # HTTP server; no LLM or pinned binary is required for this step.
    try:
        self_test = await run_observer_self_test(out_dir)
    except Exception as exc:  # noqa: BLE001 - typed failure only
        self_test = {
            "schema_version": "",
            "status": "FAIL",
            "checks": {"self-test-exception": False},
            "failed_checks": [f"self-test-error-{type(exc).__name__}"[:96]],
            "server_telemetry": {"state": "unobserved"},
            "privacy_safe": False,
            "no_orphan": False,
            "llm_calls": 0,
        }
        _write_json(out_dir / SELF_TEST_NAME, self_test)

    # Fail fast: the observer/server self-test is the cheap proof that the
    # diagnostic chain is wired. When it fails or is incomplete, spending the
    # live allowance on production turns cannot repair the missing evidence.
    if str(self_test.get("status", "")).upper() != "PASS":
        failed = [str(name) for name in self_test.get("failed_checks", []) if str(name)]
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_SELF_TEST_FAILED,
            failed[0] if failed else "observer_self_test.status",
            "observer/server self-test failed; live canary not started",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test=self_test,
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )

    # 2. Pinned OpenCode binary precondition.
    pinned_version = _pinned_opencode_version()
    installed_version = _installed_opencode_version()
    extra["pinned_opencode_version"] = pinned_version[:64]
    extra["installed_opencode_version"] = installed_version[:64]
    if not _version_matches(pinned_version, installed_version):
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_OPENCODE_UNAVAILABLE,
            "opencode.version",
            "pinned OpenCode CLI is unavailable or mismatched",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test=self_test,
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )

    # 3. Bounded real-path canary. Keep the hard wrapper inside both the
    # live allowance and the remaining preflight allowance.
    remaining = preflight_budget_s - (time.monotonic() - started)
    if remaining <= MIN_BUDGET_S:
        decision = CanaryDecision(
            STATUS_INCONCLUSIVE,
            CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
            "budget.preflight_budget_s",
            "preflight budget expired before the real-path canary",
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test=self_test,
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )
    effective_live_budget_s = min(live_budget_s, remaining)
    progress_path = out_dir / PROGRESS_NAME
    os.environ[ENV_PROGRESS_PATH] = str(progress_path)
    try:
        progress_path.unlink()
    except FileNotFoundError:
        pass

    async def _lane() -> Any:
        return await run_live_telegram_evidence_lane(
            repo_root=ROOT,
            run_id=run_id,
            canary_scenarios=gate_c_canary_scenarios(),
            canary_budget_s=effective_live_budget_s,
            attempt_observer_dir=out_dir / "canary-attempt-observer",
        )

    try:
        lane_result = await run_bounded_real_path(_lane, budget_s=effective_live_budget_s)
    except OpenCodeRateLimitError:
        decision = CanaryDecision(
            STATUS_INCONCLUSIVE,
            CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
            "provider",
            "provider rate-limited the canary; external restart required",
        )
        _write_restart_marker(out_dir, main_sha=expected_sha, run_id=run_id)
        _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test=self_test,
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )
        return EXIT_RESTART_REQUIRED
    except Exception as exc:  # noqa: BLE001 - typed harness failure only
        decision = CanaryDecision(
            STATUS_BLOCKED,
            CATEGORY_REAL_PATH_INCOMPLETE,
            "real_path",
            f"canary harness error: {type(exc).__name__}"[:96],
        )
        return _finish(
            out_dir=out_dir,
            main_sha=expected_sha,
            run_id=run_id,
            self_test=self_test,
            real_path=None,
            decision=decision,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            started=started,
            extra=extra,
        )

    # Merge the watchdog's last validated partial evidence when the lane
    # itself did not carry it (for example an outer hard-budget cancellation).
    progress = read_live_progress(progress_path)
    partial = validated_partial_evidence(progress)
    metrics = dict(getattr(lane_result, "metrics", {}) or {})
    if partial is not None and not isinstance(metrics.get("partial_evidence"), dict):
        metrics["partial_evidence"] = partial
    if partial is not None and not metrics.get("attempt_audit"):
        metrics["attempt_audit"] = list(partial.get("attempt_audit", []))
        counts = partial.get("attempt_counts", {})
        if isinstance(counts, dict):
            metrics["attempt_in_flight_count"] = int(counts.get("in_flight", 0) or 0)
    lane_result = replace(lane_result, metrics=metrics)
    real_path = build_real_path_report(lane_result)

    elapsed = time.monotonic() - started
    if elapsed > preflight_budget_s:
        decision = CanaryDecision(
            STATUS_INCONCLUSIVE,
            CATEGORY_EXTERNAL_PROGRESS_UNOBSERVED,
            "budget.preflight_budget_s",
            "preflight budget expired during the real-path canary",
        )
    else:
        decision = evaluate_canary(
            self_test=self_test,
            real_path=real_path,
            preflight_budget_s=preflight_budget_s,
            live_budget_s=live_budget_s,
            elapsed_s=elapsed,
        )
    return _finish(
        out_dir=out_dir,
        main_sha=expected_sha,
        run_id=run_id,
        self_test=self_test,
        real_path=real_path,
        decision=decision,
        preflight_budget_s=preflight_budget_s,
        live_budget_s=live_budget_s,
        started=started,
        extra=extra,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bounded Gate C canary preflight (kodmial/aa#454)."
    )
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "self-proving-out")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    parser.add_argument(
        "--preflight-budget-s",
        type=float,
        default=DEFAULT_PREFLIGHT_BUDGET_S,
        help="Total preflight allowance (default 600s)",
    )
    parser.add_argument(
        "--live-budget-s",
        type=float,
        default=DEFAULT_LIVE_BUDGET_S,
        help="Real-path observation allowance (default 300s)",
    )
    args = parser.parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
