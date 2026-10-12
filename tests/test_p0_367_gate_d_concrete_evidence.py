"""P0 kodmial/aa#367: Gate D must publish concrete typed evidence on every exit path.

Proven failure on exact main 4c86dacffe7d0cc5ea56908177b6ec8256a1bf38
(run 38250583756, attempt 2):

- the Gate D workflow step failed with EXTERNAL_TEST_BOT_UNAVAILABLE before
  ``--emit-gate D`` ran, so no gate-D-evidence.json existed and Gate F
  synthesized the generic ``D:evidence-invalid:unknown``;
- the repair publisher mapped workflow failures only against summary
  ``status === 'FAIL'``, so the BLOCKED Gate D flattened to the generic
  ``D:workflow-step-failed:workflow-step`` product fingerprint;
- that generic fingerprint minted a product-code P0 for a non-product
  external/transient state (recovery-owned), recurring independently of the
  real C/E product failures.

Repair (generic, no exact-question special cases; Product Contract #110
unchanged): Gate D emits typed evidence before every early exit (concrete
telegram-readiness category preserved, original 78/75 exit code kept for
recovery); the publisher preserves any non-PASS summary verdict instead of
the generic fallback; a transient marker with no summary FAIL routes to the
recovery controller instead of a product repair issue.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aa.qualification.gate_evidence import (
    failed_gates_for_verdict,
    root_components_for_verdict,
)
from aa.qualification.self_proving import (
    GateEvidence,
    decide_final_verdict,
    failure_report_for_gate,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SHA = "a" * 40
PRODUCT = "c" * 64
RUNTIME = "d" * 64


def _workflow() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "aa-self-proving-qualification.yml").read_text(
        encoding="utf-8"
    )


def _gate_d_block(workflow: str) -> str:
    return workflow.split("Gate D - real Telegram network/runtime readiness", 1)[1].split(
        "Gate E - live performance/SLO", 1
    )[0]


def _publisher_block(workflow: str) -> str:
    return workflow.split("Publish machine-readable failure and repair marker", 1)[1].split(
        "Publish authoritative Product Contract result", 1
    )[0]


def test_gate_d_emits_before_every_early_exit() -> None:
    block = _gate_d_block(_workflow())
    # Single helper publishes the ONE typed evidence artifact; it never
    # starts a second poller and never masks the blocker exit code.
    assert "emit_gate_d() {" in block
    assert "--emit-gate D" in block
    assert block.count("emit_gate_d") >= 5
    # Every early blocker exit still publishes evidence first.
    assert "emit_gate_d\n            exit 78" in block
    # Python probe/getMe failures are captured, published, then re-raised
    # with the original code (75 runner-retire, 78 transient, else real).
    assert "getme_rc=$?" in block
    assert 'exit "$getme_rc"' in block
    assert "probe_rc=$?" in block
    assert 'exit "$probe_rc"' in block
    # Original blocker signalling is preserved for recovery routing.
    assert "exit 78" in block
    assert "transient-blocked.json" in block
    assert "restart-required.json" in block


def test_publisher_preserves_concrete_blocked_verdict() -> None:
    block = _publisher_block(_workflow())
    # Workflow failures map against any non-PASS summary verdict, so a
    # concrete BLOCKED (e.g. Gate D test-bot-unavailable) survives instead
    # of flattening to the generic workflow-step-failed fallback.
    assert "String(item.status || '') !== 'PASS'" in block
    assert "String(g.status || '') !== 'PASS'" in block
    assert "item.status === 'FAIL'" not in block
    assert "g.status === 'FAIL'" not in block
    assert "workflow-step-failed" in block


def test_publisher_routes_transient_without_fail_to_recovery() -> None:
    block = _publisher_block(_workflow())
    # A transient marker with no summary FAIL is recovery-owned: the
    # recovery controller reruns failed jobs from checkpoints instead of
    # minting a product-code P0 for an external state.
    assert "let hasProductFail = false;" in block
    assert "hasProductFail = productFailures.some" in block
    assert "const hasProductFail" not in block
    assert "!hasProductFail" in block
    assert "String(item.status || '') === 'FAIL'" in block
    assert "recovery will retry qualification" in block
    # Runs with at least one real product FAIL still reach product repair.
    assert "stableFailures" in block
    assert "Repair fingerprint:" in block


def test_gate_d_missing_identity_is_concrete_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.run_self_proving_qualification as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    for key in (
        "SELF_PROVING_LIVE",
        "TELEGRAM_BOT_TOKEN",
        "AA_TEST_TELEGRAM_BOT_TOKEN",
        "AA_TEST_TELEGRAM_BOT_ID",
        "AA_TEST_TELEGRAM_USE_PRODUCTION_LEASE",
    ):
        monkeypatch.delenv(key, raising=False)
    evidence = runner._gate_d(SHA, "run-9", PRODUCT, RUNTIME)
    assert evidence.status == "BLOCKED"
    assert evidence.component == "telegram-readiness"
    assert evidence.failure_category not in (
        "",
        "evidence-invalid",
        "workflow-step-failed",
    )
    assert "EXTERNAL_TEST_BOT_UNAVAILABLE" in evidence.detail


def test_gate_f_preserves_concrete_d_blocked() -> None:
    def _evidence(
        gate: str,
        status: str,
        category: str,
        component: str,
        *,
        live: bool,
    ) -> GateEvidence:
        return GateEvidence(
            gate=gate,
            status=status,
            sha=SHA,
            product_fingerprint=PRODUCT,
            runtime_fingerprint=RUNTIME,
            failure_category=category,
            component=component,
            run_id="run-9",
            live_trusted=live,
        )

    verdict = decide_final_verdict(
        [
            _evidence("A", "PASS", "", "", live=True),
            _evidence("B", "PASS", "all-components", "all-components", live=True),
            _evidence(
                "C",
                "FAIL",
                "live-book-grounding-substantive-drinking-2",
                "live-production-path",
                live=True,
            ),
            _evidence(
                "D",
                "BLOCKED",
                "test-bot-unavailable",
                "telegram-readiness",
                live=False,
            ),
            _evidence("E", "FAIL", "latency-budget-exceeded", "slo", live=True),
        ],
        current_sha=SHA,
        product_fingerprint=PRODUCT,
        runtime_fingerprint=RUNTIME,
        run_id="run-9",
    )
    assert verdict.status == "BLOCKED"
    assert verdict.blocking_gate == "C"
    assert failed_gates_for_verdict(verdict) == ["C", "D", "E"]
    roots = root_components_for_verdict(verdict)
    assert "D:test-bot-unavailable:telegram-readiness" in roots
    assert "D:evidence-invalid:unknown" not in roots
    assert "C:live-book-grounding-substantive-drinking-2:live-production-path" in roots
    assert "E:latency-budget-exceeded:slo" in roots
    report = failure_report_for_gate(next(item for item in verdict.gates if item.gate == "D"))
    assert report.category == "test-bot-unavailable"
    assert report.component == "telegram-readiness"


def test_no_exact_question_special_cases() -> None:
    block = _gate_d_block(_workflow()) + _publisher_block(_workflow())
    for fragment in (
        "К вечеру очень тянет выпить",
        "Под вечер опять тянеет выпить",
        "Дома снова ссора",
        "Поругались дома из-за бухла",
        "Чем ты вообще можешь быть полезен",
        "Какой телефон сейчас выгоднее купить",
    ):
        assert fragment not in block
