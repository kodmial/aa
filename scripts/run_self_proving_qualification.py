#!/usr/bin/env python3
"""Self-proving layered qualification runner (issue #146, Gates A-F).

Evaluates Gates A-E for the exact required main SHA and emits the Gate F
exact-main final verdict. Writes privacy-safe evidence only (ids, SHAs,
digests, counts, latencies; never user/corpus text):

- ``result.json`` (PASS/FAIL/BLOCKED/STALE + marker + blocking gate);
- ``self-proving-summary.json`` (per-gate evidence + failure reports).

Exit codes: 0 PASS, 1 FAIL, 2 BLOCKED/INCOMPLETE, 3 STALE.

Fail-closed contract:
- stale SHA, dirty tree, missing secret/model, mocked-only evidence, or
  unknown state all yield BLOCKED/STALE, never a warning and never PASS.
- infrastructure failure (provider 429, missing secrets) is reported as
  BLOCKED with a machine-readable category, never as product PASS.
- provider 429 writes a restart marker and exits 75 so the workflow can
  retire the runner and resume from the last completed gate checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aa.qualification.self_proving import (  # noqa: E402
    GateEvidence,
    SelfProvingError,
    build_result_marker,
    decide_final_verdict,
    failure_report_for_gate,
    repair_fingerprint,
    validate_exact_sha,
)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_BLOCKED = 2
EXIT_STALE = 3
EXIT_429_RESTART = 75


def _git(args: list[str]) -> str:
    proc = subprocess.run(
        ["git", *args], capture_output=True, text=True, cwd=str(ROOT), check=False
    )
    if proc.returncode != 0:
        raise SelfProvingError(f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _write(out_dir: Path, verdict: object, result: str, **extra: object) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"result": result, **extra}
    (out_dir / "result.json").write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )


def _runtime_fingerprint() -> str:
    import hashlib

    from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL, Settings

    settings = Settings.from_env({})
    material = "\x00".join(
        (
            "aa-conversation-runtime/1",
            settings.opencode_agent,
            settings.opencode_model or DEFAULT_PRIMARY_MODEL,
            settings.opencode_fallback_model or DEFAULT_FALLBACK_MODEL,
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _product_fingerprint() -> str:
    from aa.qualification.product_fingerprint import compute_product_fingerprint

    return compute_product_fingerprint(ROOT)


def _gate_a(expected_sha: str, run_id: str) -> GateEvidence:
    try:
        checked = _git(["rev-parse", "HEAD"])
    except SelfProvingError as exc:
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    if checked != expected_sha:
        return GateEvidence(
            gate="A",
            status="STALE",
            sha=expected_sha,
            failure_category="stale-sha",
            run_id=run_id,
        )
    try:
        dirty = _git(["status", "--porcelain"])
    except SelfProvingError as exc:
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="git-unavailable",
            run_id=run_id,
            detail=str(exc)[:160],
        )
    if dirty.strip():
        return GateEvidence(
            gate="A",
            status="BLOCKED",
            sha=expected_sha,
            failure_category="dirty-tree",
            run_id=run_id,
        )
    try:
        product = _product_fingerprint()
        runtime = _runtime_fingerprint()
    except Exception as exc:
        return GateEvidence(
            gate="A",
            status="FAIL",
            sha=expected_sha,
            failure_category="fingerprint-error",
            component="fingerprint-consistency",
            run_id=run_id,
            detail=type(exc).__name__[:64],
        )
    if len(product) != 64 or len(runtime) != 64:
        return GateEvidence(
            gate="A",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="fingerprint-mismatch",
            component="fingerprint-consistency",
            run_id=run_id,
        )
    # Static/build trust: exact SHA + clean tree + computable fingerprints.
    # Unit/type/lint themselves run in the workflow before this script; this
    # gate records their trust boundary (workflow must fail closed first).
    return GateEvidence(
        gate="A",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        run_id=run_id,
        live_trusted=True,
    )


def _gate_b(expected_sha: str, run_id: str, product: str, runtime: str) -> GateEvidence:
    # Deterministic component integration through real production modules
    # (no network, no secrets). Any component failure names its component.
    try:
        from aa.conversation.planner_schema import QueryPlan, validate_query_plan
        from aa.retrieval.evidence import RetrievalConfig
    except Exception as exc:
        return GateEvidence(
            gate="B",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="component-fail",
            component="planner-shape",
            run_id=run_id,
            detail=type(exc).__name__[:64],
        )
    try:
        plan = validate_query_plan(QueryPlan(queries=[f"запрос {i}" for i in range(12)]))
        if not 10 <= len(plan.queries) <= 16:
            raise ValueError("cardinality out of bounds")
        try:
            validate_query_plan(QueryPlan(queries=["q0"]))
            return GateEvidence(
                gate="B",
                status="FAIL",
                sha=expected_sha,
                product_fingerprint=product,
                runtime_fingerprint=runtime,
                failure_category="component-fail",
                component="planner-cardinality",
                run_id=run_id,
            )
        except ValueError:
            pass
        config = RetrievalConfig()
        if config.branch_top_k <= 0 or config.pool_cap <= 0:
            raise ValueError("retrieval config not production")
        import aa.retrieval.evidence as evidence_mod

        source = Path(evidence_mod.__file__).read_text(encoding="utf-8").lower()
        if "cross_encoder" in source or "bge-rerank" in source:
            raise ValueError("second-stage reranker present")
    except ValueError as exc:
        return GateEvidence(
            gate="B",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="component-fail",
            component="retrieval-rrf",
            run_id=run_id,
            detail=str(exc)[:96],
        )
    return GateEvidence(
        gate="B",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component="all-components",
        run_id=run_id,
        live_trusted=True,
    )


def _live_prerequisites() -> tuple[bool, str]:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    identity = (os.environ.get("AA_BOOK_AGE_IDENTITY", "") or "").strip()
    live = (os.environ.get("SELF_PROVING_LIVE", "") or "").strip() == "1"
    if not live:
        return False, "live-execution-not-enabled"
    if not token:
        return False, "missing-secret-telegram-token"
    if not identity:
        return False, "missing-secret-corpus-identity"
    return True, ""


def _gate_c(
    expected_sha: str, run_id: str, product: str, runtime: str
) -> tuple[GateEvidence, list[float]]:
    ready, reason = _live_prerequisites()
    if not ready:
        return GateEvidence(
            gate="C",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category=reason,
            component="live-production-path",
            run_id=run_id,
            mocked_only=True,
        ), []
    # Live execution path runs in the repository-owned workflow with real
    # OpenCode/provider/corpus/index. Locally without those assets this gate
    # stays BLOCKED (fail-closed); it never reports mocked PASS.
    return GateEvidence(
        gate="C",
        status="BLOCKED",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        failure_category="live-evidence-required",
        component="live-production-path",
        run_id=run_id,
        mocked_only=True,
    ), []


def _gate_d(expected_sha: str, run_id: str, product: str, runtime: str) -> GateEvidence:
    token = (os.environ.get("TELEGRAM_BOT_TOKEN", "") or "").strip()
    live = (os.environ.get("SELF_PROVING_LIVE", "") or "").strip() == "1"
    if not live or not token:
        return GateEvidence(
            gate="D",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category=(
                "missing-secret-telegram-token" if not token else "live-evidence-required"
            ),
            component="telegram-readiness",
            run_id=run_id,
        )
    return GateEvidence(
        gate="D",
        status="BLOCKED",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        failure_category="live-evidence-required",
        component="telegram-readiness",
        run_id=run_id,
    )


def _gate_e(
    expected_sha: str,
    run_id: str,
    product: str,
    runtime: str,
    latencies_ms: list[float],
    gate_c_live: bool,
) -> GateEvidence:
    if not gate_c_live or not latencies_ms:
        return GateEvidence(
            gate="E",
            status="BLOCKED",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="missing-live-latency",
            component="slo",
            run_id=run_id,
        )
    from aa.qualification.self_proving import percentile_ms

    p50 = percentile_ms(latencies_ms, 50)
    p95 = percentile_ms(latencies_ms, 95)
    maximum = max(latencies_ms)
    if p95 > 15_000 or maximum >= 30_000:
        return GateEvidence(
            gate="E",
            status="FAIL",
            sha=expected_sha,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            failure_category="latency-budget-exceeded",
            component="slo",
            run_id=run_id,
            live_trusted=True,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
            max_turn_ms=maximum,
        )
    return GateEvidence(
        gate="E",
        status="PASS",
        sha=expected_sha,
        product_fingerprint=product,
        runtime_fingerprint=runtime,
        component="slo",
        run_id=run_id,
        live_trusted=True,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        max_turn_ms=maximum,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-proving qualification runner (Gates A-F).")
    parser.add_argument("--main-sha", required=True, help="Exact main SHA under test")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval-self-proving-out")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    args = parser.parse_args(argv)

    try:
        expected = validate_exact_sha(args.main_sha)
    except SelfProvingError as exc:
        print(f"self-proving qualification BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(args.run_id)

    if os.environ.get("OPENCODE_429_RESTART_REQUIRED", "") == "1":
        marker = out_dir / "restart-required.json"
        marker.write_text(
            json.dumps(
                {"reason_code": "OPENCODE_429_RESTART_REQUIRED", "sha": expected},
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("self-proving qualification: provider 429, runner restart required", file=sys.stderr)
        return EXIT_429_RESTART

    gate_a = _gate_a(expected, run_id)
    if gate_a.status in ("STALE",):
        _write(out_dir, None, "STALE", main_sha=expected, run_id=run_id)
        print("self-proving qualification STALE: SHA mismatch", file=sys.stderr)
        return EXIT_STALE
    product = gate_a.product_fingerprint or _product_fingerprint()
    runtime = gate_a.runtime_fingerprint or _runtime_fingerprint()

    gate_b = _gate_b(expected, run_id, product, runtime)
    gate_c, latencies = _gate_c(expected, run_id, product, runtime)
    gate_d = _gate_d(expected, run_id, product, runtime)
    gate_e = _gate_e(expected, run_id, product, runtime, latencies, gate_c_live=(gate_c.status == "PASS" and gate_c.live_trusted))

    evidences = [gate_a, gate_b, gate_c, gate_d, gate_e]
    try:
        verdict = decide_final_verdict(
            evidences,
            current_sha=expected,
            product_fingerprint=product,
            runtime_fingerprint=runtime,
            run_id=run_id,
        )
    except SelfProvingError as exc:
        print(f"self-proving qualification BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED

    failures = []
    for evidence in verdict.gates:
        if evidence.status == "PASS":
            continue
        try:
            report = failure_report_for_gate(evidence)
        except SelfProvingError:
            continue
        payload = report.to_dict()
        payload["repair_fingerprint"] = repair_fingerprint(report)
        failures.append(payload)

    summary = verdict.to_dict()
    summary["failures"] = failures
    summary["marker"] = build_result_marker(
        sha=expected,
        status=verdict.status,
        run=run_id,
    )
    (out_dir / "self-proving-summary.json").write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "result": verdict.status,
                "main_sha": expected,
                "run_id": run_id,
                "blocking_gate": verdict.blocking_gate,
                "marker": summary["marker"],
                "gates": [{"gate": item.gate, "status": item.status} for item in verdict.gates],
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
                "result": verdict.status,
                "blocking_gate": verdict.blocking_gate,
                "gates": [{item.gate: item.status} for item in verdict.gates],
            }
        )
    )
    if verdict.status == "PASS":
        return EXIT_PASS
    if verdict.status == "FAIL":
        return EXIT_FAIL
    return EXIT_BLOCKED


if __name__ == "__main__":
    raise SystemExit(main())
