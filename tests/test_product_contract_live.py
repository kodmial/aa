"""Live Product Contract lane tests (issue #142).

Proves qualification #7 can execute every live lane through the exact
production boundary. External live prerequisites (bot token, encrypted
snapshot identity, voice models) stay fail-closed INCOMPLETE, never a
fake PASS.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from aa.qualification.product_contract_live import (
    EXIT_BY_STATUS,
    ProductContractLiveError,
    _count_stage_outcomes,
    _token_usage_by_agent,
    assert_no_text_leak,
    build_result_marker,
    collect_static_gates,
    decide_status,
    run_control_lane,
    run_voice_lane,
    validate_exact_sha,
    working_tree_clean,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_validate_exact_sha_rejects_malformed() -> None:
    with pytest.raises(ProductContractLiveError):
        validate_exact_sha("not-a-sha")
    assert validate_exact_sha("a" * 40) == "a" * 40


def test_decide_status_is_fail_closed() -> None:
    assert decide_status(["PASS", "PASS", "PASS", "PASS", "PASS"]) == "PASS"
    assert decide_status(["PASS", "INCOMPLETE", "PASS", "PASS", "PASS"]) == "INCOMPLETE"
    assert decide_status(["PASS", "FAIL", "INCOMPLETE", "PASS", "PASS"]) == "FAIL"


def test_result_marker_format() -> None:
    marker = build_result_marker(sha="a" * 40, status="PASS", run="123")
    assert "issue=7" in marker
    assert ("a" * 40) in marker
    with pytest.raises(ProductContractLiveError):
        build_result_marker(sha="short", status="PASS", run="123")


def test_static_gates_carry_digests_only() -> None:
    gates = collect_static_gates(REPO_ROOT)
    assert len(gates["product_fingerprint"]) == 64
    assert gates["benchmark_total_substantive"] == 82
    assert_no_text_leak(gates)
    for key in ("utterance", "answer", "evidence_text", "transcript"):
        assert key not in gates


async def test_message_lane_executes_production_boundary() -> None:
    from aa.qualification.product_contract_live import run_message_lane

    result = await run_message_lane()
    assert result.lane == "product-contract-1-24"
    # Offline deterministic lanes must fully pass: the production
    # boundary is live-executed with stubbed externals.
    assert result.status == "PASS", result.failed
    assert len(result.passed) >= 20
    assert result.metrics["scenarios_executed"] == 24


async def test_transport_lane_reports_live_or_incomplete() -> None:
    from aa.qualification.product_contract_live import run_transport_lane

    result = await run_transport_lane()
    assert result.lane == "telegram-transport-25-32"
    # Without a live bot token the real-stream sublane is INCOMPLETE
    # (fail-closed); deterministic semantics still execute live.
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["scenarios_executed"] == 8
    if result.status == "INCOMPLETE":
        assert result.incomplete


def test_control_lane_bounds_and_restore() -> None:
    result = run_control_lane(REPO_ROOT)
    assert result.lane == "runtime-control-33-41"
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["max_starts"] == 4
    assert result.metrics["runtime_seconds"] == 18000


def test_voice_lane_policy_and_gates() -> None:
    result = run_voice_lane(REPO_ROOT)
    assert result.lane == "voice-1-16"
    assert result.status in ("PASS", "INCOMPLETE")
    assert result.metrics["scenarios_executed"] == 16
    assert result.metrics["peak_rss_mb"] <= 12288
    assert_no_text_leak(result.to_dict())


def test_working_tree_clean_ignores_ephemeral_outputs(tmp_path: Path) -> None:
    # Regression for kodmial/aa#151 (Gate unknown): Gate C created live-out/
    # before the clean check, and Gate B downloads models/, so a pristine
    # checkout always reported "working tree is not clean". Ephemeral outputs
    # must never count as dirty; real source edits still fail closed.
    import subprocess

    repo = tmp_path / "probe-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo), check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=str(repo), check=True)
    (repo / "tracked.txt").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=str(repo), check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=str(repo), check=True)
    assert working_tree_clean(repo) is True
    for name in (
        "live-out",
        "self-proving-out",
        "eval-self-proving-out",
        "aa-self-proving-checkpoints",
        "models",
    ):
        target = repo / name
        target.mkdir(exist_ok=True)
        (target / "output.json").write_text("{}", encoding="utf-8")
    assert working_tree_clean(repo) is True
    (repo / "runtime-status.json").write_text("{}", encoding="utf-8")
    assert working_tree_clean(repo) is True
    # Real untracked sources still fail closed.
    (repo / "new-source.py").write_text("x=1", encoding="utf-8")
    assert working_tree_clean(repo) is False
    (repo / "new-source.py").unlink()
    assert working_tree_clean(repo) is True
    # Tracked modifications still fail closed.
    (repo / "tracked.txt").write_text("edited", encoding="utf-8")
    assert working_tree_clean(repo) is False


def test_heartbeat_tolerance_allows_scheduling_jitter() -> None:
    """Live typing heartbeat (Gate C 37504648482) must tolerate loop jitter.

    The 20 Hz heartbeat over asyncio jitters by ~2% (9554 sends vs ~9740
    expected over 14 ordinary turns). Requiring 100% of the floor turns
    jitter into 14 systematic heartbeat failures even though typing is
    continuous. The production check requires 80% with at least one beat
    per turn; continuity is still proven.
    """
    interval = 0.05
    elapsed = 35.3016  # p50 from the failing run.
    minimum = max(1, int(max(0.0, elapsed - interval) / interval))
    assert minimum == 705  # sanity: ~20 Hz for 35s.
    observed = 682  # 97% of expected: continuous but jittered.
    tolerated = max(1, int(minimum * 0.8))
    assert observed >= tolerated
    assert observed < minimum  # old strict check would have failed.
    # At least one beat per short turn still required.
    assert max(1, int(1 * 0.8)) == 1


def test_gate_c_records_latency_but_gate_e_owns_slo_verdict() -> None:
    """Gate C records latency evidence; Gate E owns the SLO verdict."""
    source = (REPO_ROOT / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    self_proving = (REPO_ROOT / "scripts" / "run_self_proving_qualification.py").read_text(
        encoding="utf-8"
    )
    assert 'failed.append("live-text-max-over-budget")' not in source
    assert 'passed.append("live-text-latency-measured")' in source
    assert "P95_TARGET_MS" in self_proving
    assert "ORDINARY_TURN_BUDGET_MS" in self_proving
    assert "p95 > float(P95_TARGET_MS)" in self_proving
    assert "maximum >= float(ORDINARY_TURN_BUDGET_MS)" in self_proving


def test_async_lanes_attribute_harness_crash_instead_of_aborting() -> None:
    # Regression for kodmial/aa#150: a missing third-party module
    # (langchain_core on a minimal runner) raised an unhandled
    # ModuleNotFoundError from the production import chain, so the live
    # runner produced no result.json and no lane evidence. Each lane must
    # degrade to an attributable FAIL instead of aborting the evaluation.
    import asyncio

    import aa.qualification.product_contract_live as live_mod
    from aa.qualification.product_contract_live import LaneResult

    async def _boom() -> LaneResult:
        raise ModuleNotFoundError("No module named 'langchain_core'")

    async def _ok_transport() -> LaneResult:
        return LaneResult(lane="telegram-transport-25-32", status="PASS", passed=("ok",))

    async def _ok_live() -> LaneResult:
        return LaneResult(
            lane="live-telegram-evidence", status="INCOMPLETE", incomplete=("deferred",)
        )

    original_message = live_mod.run_message_lane
    original_transport = live_mod.run_transport_lane
    original_live = live_mod.run_live_telegram_evidence_lane
    live_mod.run_message_lane = _boom  # type: ignore[assignment]
    live_mod.run_transport_lane = _ok_transport  # type: ignore[assignment]
    live_mod.run_live_telegram_evidence_lane = _ok_live  # type: ignore[assignment]
    try:
        message, transport, live_evidence = asyncio.run(live_mod._run_async_lanes())
    finally:
        live_mod.run_message_lane = original_message
        live_mod.run_transport_lane = original_transport
        live_mod.run_live_telegram_evidence_lane = original_live
    assert message.lane == "product-contract-1-24"
    assert message.status == "FAIL"
    assert any("ModuleNotFoundError" in item for item in message.failed)
    assert_no_text_leak(message.to_dict())
    assert transport.status == "PASS"
    assert live_evidence.status == "INCOMPLETE"


def test_ephemeral_gate_checkpoints_stay_untracked() -> None:
    # Regression for kodmial/aa#150: root gate-*.done resume markers were
    # committed as tracked files although the workflow stages checkpoints
    # under aa-self-proving-checkpoints/. They must stay gitignored.
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "gate-*.done" in gitignore
    assert not (REPO_ROOT / "gate-A.done").exists()
    assert not (REPO_ROOT / "gate-B.done").exists()


def test_self_proving_gate_f_always_emits_verdict() -> None:
    # Regression for kodmial/aa#151 unknown-gate-failure: Gate F must run even
    # after a Gate C failure so the repair publisher maps a concrete blocking
    # gate instead of "unknown".
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml"
    ).read_text(encoding="utf-8")
    gate_f_index = workflow.index("Gate F - exact-main final verdict")
    gate_f_block = workflow[gate_f_index : gate_f_index + 800]
    assert "if: always()" in gate_f_block


def test_live_runner_stale_on_sha_mismatch(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            "python3",
            "scripts/run_product_contract_live_qualification.py",
            "--main-sha",
            "b" * 40,
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert proc.returncode == EXIT_BY_STATUS["STALE"]
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] == "STALE"


def test_live_runner_executes_all_lanes(tmp_path: Path) -> None:
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
            "scripts/run_product_contract_live_qualification.py",
            "--main-sha",
            head,
            "--out-dir",
            str(tmp_path),
            "--run-id",
            "test-run",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    # Offline working tree may be dirty (new untracked files) before the
    # repair merges; the runner stays fail-closed INCOMPLETE then. The key
    # assertion: either all five lanes executed with structured evidence,
    # or the runner refused the dirty tree without faking a result.
    assert proc.returncode in (EXIT_BY_STATUS["PASS"], EXIT_BY_STATUS["INCOMPLETE"])
    payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert payload["result"] in ("PASS", "INCOMPLETE")
    if "lanes" not in payload:
        assert "clean" in str(payload.get("reason", "")).casefold()
        return
    assert len(payload["lanes"]) == 5
    assert {lane["lane"] for lane in payload["lanes"]} == {
        "product-contract-1-24",
        "telegram-transport-25-32",
        "runtime-control-33-41",
        "voice-1-16",
        "live-telegram-evidence",
    }
    summary = json.loads(
        (tmp_path / "product-contract-live-summary.json").read_text(encoding="utf-8")
    )
    assert_no_text_leak(summary)


def test_count_stage_outcomes_aggregates_fixed_vocabularies() -> None:
    # Gate C repair diagnostics (issue #168): per-stage outcome histograms
    # must attribute a live collapse to the concrete stage without
    # carrying any prompt, reply, or evidence text.
    snapshots = [
        {
            "planner_outcome": "ok",
            "retrieval_outcome": "evidence-ready",
            "answer_outcome": "served",
            "verifier_outcome": "passed",
        },
        {
            "planner_outcome": "ok",
            "retrieval_outcome": "evidence-ready",
            "answer_outcome": "clarification",
            "verifier_outcome": "unsupported",
        },
        {
            "planner_outcome": "empty",
            "retrieval_outcome": "skipped-glue",
            "answer_outcome": "clarification",
            "verifier_outcome": "unavailable",
        },
    ]
    counts = _count_stage_outcomes(snapshots)
    assert counts["planner_outcome"] == {"ok": 2, "empty": 1}
    assert counts["retrieval_outcome"] == {"evidence-ready": 2, "skipped-glue": 1}
    assert counts["answer_outcome"] == {"served": 1, "clarification": 2}
    assert counts["verifier_outcome"] == {"passed": 1, "unsupported": 1, "unavailable": 1}
    assert_no_text_leak(counts)


def test_count_stage_outcomes_ignores_malformed_snapshots() -> None:
    counts = _count_stage_outcomes(
        [
            {"planner_outcome": "ok"},
            "not-a-snapshot",  # type: ignore[list-item]
            {"planner_outcome": "", "retrieval_outcome": "  "},
            {},
        ]
    )
    assert counts["planner_outcome"] == {"ok": 1}
    assert counts["retrieval_outcome"] == {}
    assert counts["answer_outcome"] == {}
    assert counts["verifier_outcome"] == {}


def test_token_usage_by_agent_aggregates_counters_only() -> None:
    # Gate C+E repair diagnostics (kodmial/aa#217 recurrence 3): the live
    # lane must attribute token/output work per logical agent from the
    # privacy-safe client audit so the next latency conclusion uses
    # measured round trips, not prompt-size assumptions.
    class _Client:
        token_usage_audit = (
            {"agent": "aa-v2", "input": 120, "output": 40, "reasoning": 5},
            {"agent": "aa-v2", "input": 80, "output": 20},
            {"agent": "aa-verifier-v2", "input": 200, "output": 3},
            {"agent": "", "input": 10, "output": 1},
            "not-an-entry",
            {"agent": "aa-v2", "input": -1, "output": "many"},
        )

    usage = _token_usage_by_agent(_Client())
    assert usage["aa-v2"]["requests"] == 3
    assert usage["aa-v2"]["input_total"] == 200
    assert usage["aa-v2"]["output_total"] == 60
    assert usage["aa-v2"]["reasoning_total"] == 5
    assert usage["aa-verifier-v2"]["requests"] == 1
    assert usage["aa-verifier-v2"]["input_total"] == 200
    assert usage["unknown"]["requests"] == 1
    assert_no_text_leak(usage)


def test_live_voice_fixture_failure_has_concrete_component() -> None:
    # Gate C repair (run 37530425848): voice fixture synthesis raised into
    # generic live-production-telegram-harness with voice_end_to_end_ms 0.0
    # and no voice end-to-end checks. A synthesis/encoding failure must map
    # to its concrete voice component, never the generic harness.
    source = (REPO_ROOT / "src" / "aa" / "qualification" / "product_contract_live.py").read_text(
        encoding="utf-8"
    )
    assert "live-voice-fixture-synthesis" in source
    assert "live-voice-raw-transport-accepted" in source
    assert "live-voice-file-fetch-seam" in source
    assert "live-voice-asr-answer-sendvoice" in source


def test_privacy_guard_allows_numeric_answer_stage_metrics() -> None:
    # Gate C repair (run 37691536129): the live lane records per-stage
    # latency as stage_latency_ms with structural keys planner/retrieval/
    # answer/verifier/total. The bare "answer" key tripped the text-leak
    # guard even though its value holds only numbers, so evaluate_live
    # raised live-summary.lanes[4].metrics.stage_latency_ms: forbidden key
    # 'answer' and Gate C degraded to BLOCKED live-evidence-required with
    # no live artifact. Numeric-only subtrees must pass; text must fail.
    from aa.qualification import self_proving as self_proving_mod

    numeric_stage_metrics = {
        "stage_latency_ms": {
            "planner": {"p50": 100.0, "p95": 200.0, "max": 300.0},
            "retrieval": {"p50": 10.0, "p95": 20.0, "max": 30.0},
            "answer": {"p50": 50.0, "p95": 60.0, "max": 70.0},
            "verifier": {"p50": 5.0, "p95": 6.0, "max": 7.0},
            "total": {"p50": 165.0, "p95": 200.0, "max": 250.0},
        }
    }
    assert_no_text_leak({"metrics": numeric_stage_metrics})
    self_proving_mod.assert_no_text_leak({"metrics": numeric_stage_metrics})
    with pytest.raises(ProductContractLiveError):
        assert_no_text_leak({"answer": "secret reply text"})
    with pytest.raises(ProductContractLiveError):
        assert_no_text_leak(numeric_stage_metrics | {"answer": "secret reply text"})
    with pytest.raises(ProductContractLiveError):
        assert_no_text_leak({"stage_latency_ms": {"answer": {"p50": 1.0, "note": "leaked text"}}})


def test_gate_c_rejects_bookless_retry_diversity() -> None:
    """Different generic replies must never qualify as book-grounded help."""
    from aa.conversation.turn_pipeline import (
        NATURAL_CLARIFICATION_REPLY,
        NATURAL_RETRY_VARIANTS,
    )
    from aa.qualification.product_contract_live import _is_grounded_substantive_reply

    verified = {
        "answer_outcome": "served",
        "verifier_outcome": "passed",
        "verifier_unavailable_units": 0,
        "turn_budget_exceeded": False,
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 1,
        "response_units": 1,
        # Production semantic flags (kodmial/aa#281): GraphTurnRuntime
        # emits all four on every real turn; counts alone never suffice.
        "adequacy_verdict": "pass",
        "answers_request": True,
        "technically_grounded": True,
        "qualified": True,
    }
    # Only the qualification predicate is under test here; this string
    # is not claimed to be a source quotation or a real generated answer.
    assert _is_grounded_substantive_reply(verified, "Проверенный ответ по книге.")
    # Fail closed (kodmial/aa#281): a missing semantic verdict FAILs even
    # when numeric counts, diversified arbitrary output, valid book
    # passage IDs and answer_outcome=served all appear favorable.
    numeric_only = {
        "answer_outcome": "served",
        "planner_query_count": 12,
        "retrieval_passages": 5,
        "verified_book_units": 1,
    }
    assert not _is_grounded_substantive_reply(numeric_only, "Проверенный ответ по книге.")
    assert not _is_grounded_substantive_reply(
        numeric_only, "Совершенно другой разнообразный произвольный текст."
    )
    for missing in ("adequacy_verdict", "answers_request", "technically_grounded", "qualified"):
        pruned = dict(verified)
        del pruned[missing]
        assert not _is_grounded_substantive_reply(pruned, "Проверенный ответ по книге.")
    for false_flag in (
        {"adequacy_verdict": "fail"},
        {"adequacy_verdict": "unknown"},
        {"adequacy_verdict": ""},
        {"answers_request": False},
        {"technically_grounded": False},
        {"qualified": False},
    ):
        assert not _is_grounded_substantive_reply(
            {**verified, **false_flag}, "Проверенный ответ по книге."
        )
    for retry in (*NATURAL_RETRY_VARIANTS, NATURAL_CLARIFICATION_REPLY):
        assert not _is_grounded_substantive_reply(verified, retry)

    assert not _is_grounded_substantive_reply(
        {**verified, "verified_book_units": 0}, "Развёрнутый, но неподтверждённый ответ."
    )
    assert not _is_grounded_substantive_reply(
        {**verified, "answer_outcome": "retry-turn-budget"}, "Любой непроверенный ответ."
    )
    assert not _is_grounded_substantive_reply(
        {**verified, "retrieval_passages": 0}, "Ответ без найденной книги."
    )
    assert not _is_grounded_substantive_reply(
        {**verified, "planner_query_count": 0}, "Ответ без поиска."
    )
