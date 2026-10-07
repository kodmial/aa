"""Self-proving qualification contracts (issue #146, Gates A-F).

Proves the non-negotiable invariant: owner ``/run`` for manual testing stays
available regardless of qualification state. Qualification state controls only
the QUALIFIED/UNQUALIFIED label, never the launch. INCOMPLETE, missing
secret/model, stale SHA, mocked-only evidence, or unknown state all mean
FAIL/BLOCKED for qualification, never a restriction on owner-initiated manual
testing.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aa.control.runtime_status import (
    RuntimeMarker,
    format_marker,
    parse_marker,
    status_report,
    transition_allowed,
)
from aa.conversation.stage_telemetry import (
    TurnTelemetry,
    evaluate_gate_c_telemetry,
    evaluate_gate_e_telemetry,
    record_stage,
    reply_signature,
)
from aa.qualification.self_proving import (
    CONTROL_ISSUE_NUMBER,
    GATE_B_COMPONENTS,
    GATE_C_STAGES,
    ISSUE_NUMBER,
    MANDATORY_GATES,
    SCENARIO_FAMILIES,
    FinalVerdict,
    GateEvidence,
    SelfProvingError,
    assert_no_text_leak,
    build_result_marker,
    canary_scope,
    decide_final_verdict,
    diversity_passes,
    effective_gate_status,
    failure_report_for_gate,
    find_reusable_repair,
    heartbeat_continuity_ok,
    is_429_restart,
    is_run_allowed,
    refusal_explanation,
    repair_fingerprint,
    rerun_plan,
    retry_delay_403,
    slo_guards,
    validate_exact_sha,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
OTHER_SHA = "b" * 40
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _pass_gate(gate: str, *, live: bool = False) -> GateEvidence:
    return GateEvidence(
        gate=gate,
        status="PASS",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-1",
        live_trusted=True
        if gate in ("C", "D", "E") and live
        else (gate not in ("C", "D", "E") or live),
        mocked_only=False,
    )


def _verdict(statuses: dict[str, str]) -> FinalVerdict:
    evidences = []
    for gate in MANDATORY_GATES:
        status = statuses.get(gate, "PASS")
        live = gate in ("C", "D", "E")
        evidences.append(
            GateEvidence(
                gate=gate,
                status=status,
                sha=SHA,
                product_fingerprint=PRODUCT,
                runtime_fingerprint=RUNTIME,
                run_id="run-1",
                failure_category="" if status == "PASS" else "test-failure",
                component="test-component" if status != "PASS" else "",
                live_trusted=live if status == "PASS" else False,
                mocked_only=status != "PASS",
            )
        )
    return decide_final_verdict(
        evidences,
        current_sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-1",
    )


def test_invariant_owner_run_always_allowed_qualification_labels_only() -> None:
    verdict = _verdict({})
    assert verdict.status == "PASS"
    # Owner /run stays available regardless of qualification state; the
    # verdict only controls the QUALIFIED/UNQUALIFIED label.
    assert is_run_allowed(verdict, current_sha=SHA) is True
    # Stale SHA still allows /run (labeled UNQUALIFIED).
    assert is_run_allowed(verdict, current_sha=OTHER_SHA) is True
    # Missing verdict still allows /run (labeled UNQUALIFIED).
    assert is_run_allowed(None, current_sha=SHA) is True


def test_incomplete_missing_stale_mocked_all_block_qualification_not_run() -> None:
    for bad in ("INCOMPLETE", "STALE", "BLOCKED"):
        verdict = _verdict({"C": bad})
        assert verdict.status in ("FAIL", "BLOCKED")
        # Qualification is BLOCKED, but owner /run stays available.
        assert is_run_allowed(verdict, current_sha=SHA) is True
    verdict = _verdict({"D": "FAIL"})
    assert verdict.status == "FAIL"
    assert is_run_allowed(verdict, current_sha=SHA) is True
    # Mocked-only PASS evidence blocks.
    mocked = GateEvidence(gate="C", status="PASS", sha=SHA, run_id="r", mocked_only=True)
    assert effective_gate_status(mocked, current_sha=SHA) == "BLOCKED"
    # Live gates without live trust block.
    untrusted = GateEvidence(gate="E", status="PASS", sha=SHA, run_id="r")
    assert effective_gate_status(untrusted, current_sha=SHA) == "BLOCKED"
    # Stale SHA blocks.
    stale = GateEvidence(gate="A", status="PASS", sha=OTHER_SHA, run_id="r")
    assert effective_gate_status(stale, current_sha=SHA) == "STALE"


def test_unknown_status_never_warns() -> None:
    with pytest.raises(SelfProvingError):
        effective_gate_status(
            GateEvidence(gate="A", status="MAYBE", sha=SHA, run_id="r"),
            current_sha=SHA,
        )
    with pytest.raises(SelfProvingError):
        validate_exact_sha("short")


def test_gate_b_components_are_attributable() -> None:
    assert len(GATE_B_COMPONENTS) >= 10
    for component in (
        "corpus-restore",
        "retrieval-rrf",
        "grounding",
        "verifier",
        "output-envelope",
        "memory-fifo",
        "safety",
    ):
        assert component in GATE_B_COMPONENTS


def test_gate_c_stages_and_families_no_whitelist() -> None:
    assert tuple(GATE_C_STAGES) == (
        "application",
        "dispatcher",
        "langgraph",
        "planner",
        "retrieval",
        "answer",
        "verifier",
        "delivery",
    )
    assert len(SCENARIO_FAMILIES) == 8
    for family in (
        "meta-capability",
        "substantive-drinking",
        "family-relationship",
        "followup-ellipsis",
        "topic-shift",
        "unsupported-out-of-book",
        "emergency",
        "long-conversation",
    ):
        assert family in SCENARIO_FAMILIES
    # Families are behavioral, not exact questions: no family id may look
    # like a hardcoded user utterance.
    for family in SCENARIO_FAMILIES:
        assert "?" not in family
        assert len(family) < 40


def test_gate_c_diversity_catches_generic_fallback_collapse() -> None:
    collapsed = [reply_signature("same generic clarification") for _ in range(3)]
    ok, _ = diversity_passes(collapsed)
    assert ok is False
    diverse = [reply_signature(f"natural reply variant {i}") for i in range(3)]
    ok, detail = diversity_passes(diverse)
    assert ok is True
    assert "distinct" in detail


def _telemetry(family: str, signature: str, total_ms: float = 1200.0) -> TurnTelemetry:
    turn = TurnTelemetry(family=family, reply_signature=signature, reply_len=42)
    per = total_ms / float(len(GATE_C_STAGES))
    for stage in GATE_C_STAGES:
        record_stage(turn, stage=stage, ok=True, latency_ms=per)
    return turn


def test_gate_c_telemetry_requires_stages_and_diversity() -> None:
    turns = [_telemetry(f"family-{i}", reply_signature(f"reply {i}")) for i in range(4)]
    ok, _, metrics = evaluate_gate_c_telemetry(turns, required_families=4)
    assert ok is True
    assert metrics["turns"] == 4
    collapsed = [_telemetry(f"family-{i}", reply_signature("same fallback")) for i in range(4)]
    ok, detail, _ = evaluate_gate_c_telemetry(collapsed, required_families=4)
    assert ok is False
    assert "fallback" in detail


def test_gate_e_slo_guard_rejects_pathological_latency() -> None:
    # The runtime #37422302821 failure (~30-60s ordinary turns) must fail.
    slow = [
        _telemetry(f"family-{i}", reply_signature(f"reply {i}"), total_ms=45000.0) for i in range(2)
    ]
    ok, detail, metrics = evaluate_gate_e_telemetry(
        slow, heartbeat_sends=12, heartbeat_interval_ms=4000.0
    )
    assert ok is False
    assert "30000" in detail or "budget" in detail
    assert metrics["max_ms"] >= 30000.0
    fast = [
        _telemetry(f"family-{i}", reply_signature(f"reply {i}"), total_ms=1200.0) for i in range(4)
    ]
    ok, _, metrics = evaluate_gate_e_telemetry(
        fast, heartbeat_sends=4, heartbeat_interval_ms=4000.0
    )
    assert ok is True
    assert metrics["p95_ms"] < 15000.0


def test_gate_e_slo_helpers() -> None:
    ok, _, metrics = slo_guards([1000.0, 1200.0, 1400.0])
    assert ok is True
    assert metrics["p50_ms"] > 0
    ok, _, _ = slo_guards([])
    assert ok is False
    ok, _, _ = slo_guards([1000.0, 45000.0])
    assert ok is False
    ok, _ = heartbeat_continuity_ok(sends=3, duration_ms=9000.0, interval_ms=4000.0)
    assert ok is True
    ok, _ = heartbeat_continuity_ok(sends=0, duration_ms=9000.0, interval_ms=4000.0)
    assert ok is False


def test_gate_f_exact_main_no_stale_reuse() -> None:
    verdict = _verdict({})
    assert verdict.status == "PASS"
    assert verdict.blocking_gate == ""
    # Fingerprint mismatch blocks.
    evidences = [
        GateEvidence(
            gate=gate,
            status="PASS",
            sha=SHA,
            product_fingerprint=("f" * 64 if gate == "B" else PRODUCT),
            runtime_fingerprint=RUNTIME,
            run_id="run-1",
            live_trusted=True,
        )
        for gate in MANDATORY_GATES
    ]
    verdict = decide_final_verdict(
        evidences,
        current_sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-1",
    )
    assert verdict.status == "BLOCKED"
    assert verdict.blocking_gate == "B"
    # Missing gate blocks.
    verdict = decide_final_verdict(
        [_pass_gate("A")],
        current_sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-1",
    )
    assert verdict.status == "BLOCKED"


def test_refusal_names_blocking_gate() -> None:
    verdict = _verdict({"C": "FAIL"})
    explanation = refusal_explanation(verdict, current_sha=SHA)
    assert "UNQUALIFIED" in explanation
    assert "Gate C" in explanation
    # Owner /run is never refused; the label is advisory.
    assert is_run_allowed(verdict, current_sha=SHA) is True
    ready = refusal_explanation(_verdict({}), current_sha=SHA)
    assert "QUALIFIED" in ready


def test_repair_loop_single_issue_and_rerun_order() -> None:
    evidence = GateEvidence(
        gate="C",
        status="FAIL",
        sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        failure_category="fallback-collapse",
        component="answer",
        run_id="run-9",
        latency_p50_ms=1200.0,
        latency_p95_ms=2500.0,
        max_turn_ms=4000.0,
    )
    report = failure_report_for_gate(evidence)
    assert report.gate == "C"
    assert report.category == "fallback-collapse"
    fingerprint = repair_fingerprint(report)
    assert len(fingerprint) == 16
    same_defect_new_sha = failure_report_for_gate(
        GateEvidence(
            gate="C",
            status="FAIL",
            sha=OTHER_SHA,
            failure_category="fallback-collapse",
            component="answer",
            run_id="run-10",
        )
    )
    assert repair_fingerprint(same_defect_new_sha) == fingerprint
    assert find_reusable_repair([(12, fingerprint), (13, "other")], fingerprint=fingerprint) == 12
    assert find_reusable_repair([(12, "other")], fingerprint=fingerprint) is None
    assert rerun_plan("C") == ["C", "D", "E", "F"]
    assert rerun_plan("A") == ["A", "B", "C", "D", "E", "F"]
    with pytest.raises(SelfProvingError):
        rerun_plan("Z")


def test_provider_429_and_403_policy() -> None:
    assert is_429_restart("OPENCODE_429_RESTART_REQUIRED") is True
    assert is_429_restart("ok") is False
    assert retry_delay_403(0) >= 5.0
    assert retry_delay_403(1) >= retry_delay_403(0)
    assert retry_delay_403(100) >= 5.0


def test_runtime_marker_lifecycle_and_status() -> None:
    assert transition_allowed(None, "STARTING") is True
    assert transition_allowed(None, "READY") is False
    assert transition_allowed("STARTING", "READY") is True
    assert transition_allowed("READY", "STOPPED") is True
    assert transition_allowed("READY", "FAILED") is True
    assert transition_allowed("STOPPED", "READY") is False
    marker = RuntimeMarker(run_id="123", sha=SHA, phase="READY", timestamp_s=1.0)
    text = format_marker(marker)
    assert "aa-runtime-status" in text
    assert "READY" in text
    parsed = parse_marker(text)
    assert parsed is not None
    assert parsed.phase == "READY"
    report = status_report(marker)
    assert "READY" in report
    assert "in_progress" not in report
    assert "STARTING" in status_report(
        RuntimeMarker(run_id="1", sha=SHA, phase="STARTING", timestamp_s=0.0)
    )


def test_canary_never_blocks_owner_run() -> None:
    scope = canary_scope()
    assert scope["schedule_hours"] == 4
    assert "C-smoke" in scope["gates"]
    assert "D-readiness" in scope["gates"]
    assert "E-latency-guard" in scope["gates"]
    assert scope["blocks_new_run_until_green"] is False
    assert scope["never_blocks_owner_run"] is True
    assert scope["auto_repair"] is True


def test_privacy_no_user_or_corpus_text() -> None:
    with pytest.raises(SelfProvingError):
        assert_no_text_leak({"answer": "secret"})
    with pytest.raises(SelfProvingError):
        build_result_marker(sha="short", status="PASS", run="1")
    marker = build_result_marker(sha=SHA, status="PASS", run="1")
    assert SHA in marker
    turn = _telemetry("meta-capability", reply_signature("hello"))
    payload = turn.to_dict()
    assert "hello" not in json.dumps(payload)
    assert_no_text_leak(payload)


def test_workflows_own_secrets_assets_and_repair() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    for needle in (
        "Gate A - static/build",
        "Gate B - deterministic component integration",
        "Gate C - LIVE exact production conversation path",
        "Gate D - real Telegram network/runtime readiness",
        "Gate E - live performance/SLO",
        "Gate F - exact-main final verdict",
        "TELEGRAM_BOT_TOKEN",
        "AA_BOOK_AGE_IDENTITY",
        "OPENCODE_MODEL",
        "aa-self-proving-repair:v1",
        "priority:p0",
        "OPENCODE_429_RESTART_REQUIRED",
        "retry",
    ):
        assert needle in workflow
    control = (REPO_ROOT / ".github" / "workflows" / "aa-runtime-control.yml").read_text(
        encoding="utf-8"
    )
    assert "aa-self-proving-qualification.yml" in control
    assert "UNQUALIFIED" in control
    assert "/run" in control
    assert "aa-runtime-status" in control
    assert "STARTING -> READY -> STOPPED|FAILED" in control
    runtime = (REPO_ROOT / ".github" / "workflows" / "aa-runtime.yml").read_text(encoding="utf-8")
    assert "aa-runtime-status" in runtime
    assert "STARTING" in runtime
    assert "READY" in runtime
    canary = (REPO_ROOT / ".github" / "workflows" / "aa-canary.yml").read_text(encoding="utf-8")
    assert 'cron: "17 */4 * * *"' in canary
    assert "Gate C smoke" in canary
    assert "Gate D readiness" in canary
    assert "Gate E latency guard" in canary
    assert "group: aa-self-proving-qualification" in canary
    assert "Activate bounded current-main canary" in canary
    assert "Re-enter authoritative convergence after canary failure" in canary
    assert CONTROL_ISSUE_NUMBER == 31
    assert ISSUE_NUMBER == 146

    # Systemic convergence guard: a failed qualification must create a
    # scheduler-admissible repair and AA must never drift back to terminal
    # pause semantics or fewer than the required three implementation slots.
    assert "automation:ready" in workflow
    assert "aa-self-proving-restart" in workflow
    assert "aa-self-proving-transient" in workflow
    assert "transient-blocked.json" in workflow
    assert "Stage timings:" in workflow
    assert "Verifier availability:" in workflow
    assert "OpenCode request timings:" in workflow
    assert "Repair recurrence:" in workflow
    assert "async function apiRetry" in workflow
    assert "retrying in" in workflow
    assert "const stableFailures = productFailures.length" in workflow
    assert "const fingerprint = stableFailures.join('|')" in workflow
    recovery = (REPO_ROOT / ".github" / "workflows" / "aa-self-proving-429-recovery.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_run" in recovery
    assert "AA self-proving qualification" in recovery
    assert "aa-self-proving-restart" in recovery
    assert "aa-self-proving-transient" in recovery
    assert "External/transient qualification blocker" in recovery
    assert "reRunWorkflowFailedJobs" in recovery
    assert "createWorkflowDispatch" in recovery
    scheduler = (REPO_ROOT / ".github" / "workflows" / "continuum-issue-scheduler.yml").read_text(
        encoding="utf-8"
    )
    assert 'ready_label: "automation:ready"' in scheduler
    assert 'pause_on_failure: "false"' in scheduler
    assert "wip_limit: \"${{ inputs.wip_limit || '3' }}\"" in scheduler

    # Gate E and terminal Gate F must preserve the same live prerequisites as
    # Gate C. Otherwise the final verdict can falsely downgrade valid live
    # evidence to C:live-execution-not-enabled.
    gate_d = workflow.split("Gate D - real Telegram network/runtime readiness", 1)[1].split(
        "Gate E - live performance/SLO", 1
    )[0]
    gate_e = workflow.split("Gate E - live performance/SLO", 1)[1].split(
        "Gate F - exact-main final verdict", 1
    )[0]
    gate_f = workflow.split("Gate F - exact-main final verdict", 1)[1].split(
        "Stage gate checkpoints for resume", 1
    )[0]
    for independent in (gate_d, gate_e):
        assert "steps.gate-c.outcome == 'success'" in independent
        assert "steps.gate-c.outcome == 'failure'" in independent
        assert "restart-required.json" in independent
    for block in (gate_e, gate_f):
        assert 'SELF_PROVING_LIVE: "1"' in block
        assert "TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}" in block
        assert "AA_BOOK_AGE_IDENTITY: ${{ secrets.AA_BOOK_AGE_IDENTITY }}" in block


def test_gate_c_preserves_live_failure_without_live_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gate F re-evaluation must preserve the Gate-C-owned live lane only."""
    import scripts.run_self_proving_qualification as runner

    monkeypatch.delenv("SELF_PROVING_LIVE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(
            {
                "main_sha": SHA,
                "run_id": "run-1",
                "status": "FAIL",
                "lanes": [
                    {
                        "lane": "live-telegram-evidence",
                        "status": "FAIL",
                        "passed": [],
                        "failed": ["live-answer-no-generic-collapse"],
                        "incomplete": [],
                        "metrics": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence, _, _ = runner._gate_c(SHA, "run-1", PRODUCT, RUNTIME)
    assert evidence.status == "FAIL"
    assert evidence.failure_category == "live-answer-no-generic-collapse"


def test_gate_c_ignores_foreign_lane_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-C lane must never be reclassified as a Gate C repair."""
    import scripts.run_self_proving_qualification as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    live_dir = tmp_path / "live-out"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "product-contract-live-summary.json").write_text(
        json.dumps(
            {
                "main_sha": SHA,
                "run_id": "run-1",
                "status": "FAIL",
                "lanes": [
                    {
                        "lane": "runtime-control-33-41",
                        "status": "FAIL",
                        "failed": ["runtime-control-failure"],
                        "passed": [],
                        "incomplete": [],
                        "metrics": {},
                    },
                    {
                        "lane": "live-telegram-evidence",
                        "status": "PASS",
                        "passed": [
                            "live-raw-telegram-transport-boundary",
                            "live-answer-no-generic-collapse",
                            "live-answer-diversity",
                            "live-actual-served-model-identity",
                            "live-planner-retrieval-answer-verifier-telemetry",
                        ],
                        "failed": [],
                        "incomplete": [],
                        "metrics": {
                            "turns_executed": 4,
                            "latency_p50_s": 20.0,
                            "latency_p95_s": 25.0,
                            "latency_max_s": 29.0,
                            "turn_latencies_ms": [18000.0, 20000.0, 24000.0, 29000.0],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence, latencies, aggregate = runner._gate_c(SHA, "run-1", PRODUCT, RUNTIME)
    assert evidence.status == "PASS"
    assert evidence.live_trusted is True
    assert latencies
    assert aggregate is not None


def test_gate_c_still_blocked_without_evidence_or_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without live evidence or live env, Gate C stays fail-closed BLOCKED."""
    import scripts.run_self_proving_qualification as runner

    monkeypatch.delenv("SELF_PROVING_LIVE", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    evidence, _, _ = runner._gate_c(SHA, "run-1", PRODUCT, RUNTIME)
    assert evidence.status == "BLOCKED"
    assert evidence.failure_category == "live-execution-not-enabled"


def test_gate_e_uses_trusted_live_latency_even_when_gate_c_functionally_fails() -> None:
    runner = (REPO_ROOT / "scripts" / "run_self_proving_qualification.py").read_text(
        encoding="utf-8"
    )
    assert "gate_c_live=gate_c.live_trusted" in runner
    assert 'gate_c_live=(gate_c.status == "PASS" and gate_c.live_trusted)' not in runner


def test_gate_e_f_preserve_live_execution_context() -> None:
    """Gate E/F re-evaluation must carry the Gate C live context (issue #153)."""
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    # Gate C plus the Gate E and Gate F re-evaluations must all set the live
    # execution context so a real live-path-failed is never reclassified as
    # live-execution-not-enabled.
    assert workflow.count('SELF_PROVING_LIVE: "1"') >= 3


def test_runner_stale_and_blocked_contract(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            "scripts/run_self_proving_qualification.py",
            "--main-sha",
            OTHER_SHA,
            "--out-dir",
            str(tmp_path / "stale"),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == 3
    payload = json.loads((tmp_path / "stale" / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] == "STALE"
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    ).stdout.strip()
    proc = subprocess.run(
        [
            "python3",
            "scripts/run_self_proving_qualification.py",
            "--main-sha",
            head,
            "--out-dir",
            str(tmp_path / "live"),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # Offline working tree carries the new repair files (dirty) or lacks live
    # secrets: either way the runner must stay fail-closed BLOCKED, never PASS.
    assert proc.returncode in (1, 2)
    payload = json.loads((tmp_path / "live" / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] in ("FAIL", "BLOCKED")
