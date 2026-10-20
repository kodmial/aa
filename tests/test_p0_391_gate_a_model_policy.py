"""Regression: Gate A validates the AA chatbot model policy, not coder routes.

AA owns its chatbot route through the repository variables
``AA_BOT_PRIMARY_MODEL`` / ``AA_BOT_FALLBACK_MODEL`` (Work Lock #66).
Gate A validates the effective runtime aliases against that resolved policy
instead of requiring the retired Muse Spark / Space Bunny pair or a Continuum
coding-pool selection. The same verified model may serve as both primary and
fallback because the adapter suppresses same-model fallback; the current
distinct DeepSeek/Beam route is valid too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_self_proving_qualification as runner
from aa.config import DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL
from aa.qualification.self_proving import GateEvidence, runtime_model_policy_ok

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
PRODUCTION_WORKFLOWS = (
    "aa-self-proving-qualification.yml",
    "aa-gate-c-heldout-bootstrap.yml",
)
PRIMARY = "opencode/deepseek-v4.1-flash-free"
FALLBACK = "opencode/beam-free"
OTHER_APPROVED_MODEL = "opencode/other-approved-free"
DISTINCT_FALLBACK_MODEL = FALLBACK


def test_aa_owned_model_policy_accepts_same_and_distinct_routes() -> None:
    # The owner may select any approved primary/fallback pair from AA_BOT_*.
    assert runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        expected_primary_model=PRIMARY,
        expected_fallback_model=FALLBACK,
        model=PRIMARY,
        fallback_model=FALLBACK,
    )
    assert runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        expected_primary_model=PRIMARY,
        expected_fallback_model=PRIMARY,
        model=PRIMARY,
        fallback_model=PRIMARY,
    )
    # The workflow correctly rejects drift from the frozen AA_BOT_* route.
    assert not runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        expected_primary_model=PRIMARY,
        expected_fallback_model=FALLBACK,
        model=OTHER_APPROVED_MODEL,
        fallback_model=DISTINCT_FALLBACK_MODEL,
    )


@pytest.mark.parametrize(
    ("agent", "expected_primary", "expected_fallback", "model", "fallback"),
    [
        ("other-agent", "", "", PRIMARY, FALLBACK),
        ("other-agent", PRIMARY, FALLBACK, PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, "", "", PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, "", "", "opencode/foo/", FALLBACK),
        (DEFAULT_AA_AGENT, "", "", "opencode/muse-spark-1.3-contributor-free", FALLBACK),
        (DEFAULT_AA_AGENT, "", "", PRIMARY, "opencode/space-bunny-free"),
        (DEFAULT_AA_AGENT, "", "", "", FALLBACK),
        (DEFAULT_AA_AGENT, "", "", PRIMARY, ""),
        (DEFAULT_AA_AGENT, "", "", "no-slash", FALLBACK),
        (DEFAULT_AA_AGENT, "", "", "opencode/../escape", FALLBACK),
        (DEFAULT_AA_AGENT, "", "", PRIMARY, "open//code"),
        (DEFAULT_AA_AGENT, OTHER_APPROVED_MODEL, DISTINCT_FALLBACK_MODEL, PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, PRIMARY, OTHER_APPROVED_MODEL, PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, PRIMARY, FALLBACK, "opencode/other", FALLBACK),
        (DEFAULT_AA_AGENT, PRIMARY, FALLBACK, PRIMARY, "opencode/other"),
        (DEFAULT_AA_AGENT, "", FALLBACK, PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, PRIMARY, "", PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, "bad-id", FALLBACK, PRIMARY, FALLBACK),
        (DEFAULT_AA_AGENT, PRIMARY, "bad-id", PRIMARY, FALLBACK),
    ],
)
def test_model_policy_drift_is_rejected(
    agent: str,
    expected_primary: str,
    expected_fallback: str,
    model: str,
    fallback: str,
) -> None:
    assert not runtime_model_policy_ok(
        agent=agent,
        expected_primary_model=expected_primary,
        expected_fallback_model=expected_fallback,
        model=model,
        fallback_model=fallback,
    )


def test_gate_a_accepts_frozen_aa_policy_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "f" * 40

    def _fake_git(args: list[str]) -> str:
        return sha if args and args[0] == "rev-parse" else ""

    monkeypatch.setattr(runner, "_git", _fake_git)
    monkeypatch.setenv("OPENCODE_AGENT", DEFAULT_AA_AGENT)
    monkeypatch.setenv("AA_BOT_PRIMARY_MODEL", OTHER_APPROVED_MODEL)
    monkeypatch.setenv("AA_BOT_FALLBACK_MODEL", DISTINCT_FALLBACK_MODEL)
    monkeypatch.setenv("OPENCODE_MODEL", OTHER_APPROVED_MODEL)
    monkeypatch.setenv("OPENCODE_FALLBACK_MODEL", DISTINCT_FALLBACK_MODEL)

    accepted = runner._gate_a(sha, "run-391")
    assert accepted.status == "PASS"
    assert accepted.failure_category == ""

    # A stale Continuum coding route must not influence the AA chatbot policy.
    monkeypatch.setenv("OPENCODE_MODELS", "opencode/unrelated-coder")
    monkeypatch.setenv("CONTINUUM_SELECTED_MODEL", "opencode/unrelated-coder")
    unaffected = runner._gate_a(sha, "run-391")
    assert unaffected.status == "PASS"

    monkeypatch.setenv("OPENCODE_FALLBACK_MODEL", "opencode/space-bunny-free")
    retired = runner._gate_a(sha, "run-391")
    assert retired.status == "FAIL"
    assert retired.failure_category == "model-policy-mismatch"
    assert retired.component == "model-policy"

    monkeypatch.setenv("OPENCODE_FALLBACK_MODEL", "opencode/other")
    drifted = runner._gate_a(sha, "run-391")
    assert drifted.status == "FAIL"
    assert drifted.failure_category == "model-policy-mismatch"
    assert drifted.component == "model-policy"


def test_gate_a_subchecks_attribute_model_policy_failure() -> None:
    failed = runner._gate_a_subchecks(
        GateEvidence(
            gate="A",
            status="FAIL",
            sha="f" * 40,
            failure_category="model-policy-mismatch",
            component="model-policy",
            run_id="run-391",
        ),
        "pass",
        "",
    )
    by_id = {item.check_id: item.status for item in failed}
    assert by_id["a-model-policy"] == "FAIL"
    assert by_id["a-exact-sha"] == "PASS"


def test_live_workflows_use_aa_owned_policy_without_retired_models() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "OPENCODE_MODEL: opencode/" not in text, path.name
        assert "OPENCODE_FALLBACK_MODEL: opencode/" not in text, path.name

    for name in PRODUCTION_WORKFLOWS:
        workflow = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert "AA_BOT_PRIMARY_MODEL: ${{ vars.AA_BOT_PRIMARY_MODEL }}" in workflow, name
        assert "AA_BOT_FALLBACK_MODEL: ${{ vars.AA_BOT_FALLBACK_MODEL }}" in workflow, name
        assert "python -m aa.bot_models --github-output" in workflow, name
        assert "AA_BOT_PRIMARY_MODEL=%s" in workflow, name
        assert "AA_BOT_FALLBACK_MODEL=%s" in workflow, name
        assert "OPENCODE_MODEL=$VERIFIED_MODEL" not in workflow, name
        assert "CONTINUUM_SELECTED_MODEL=$VERIFIED_MODEL" not in workflow, name
        assert "model-route@" not in workflow, name
        assert "task-class: coder" not in workflow, name

    text = (WORKFLOWS / "aa-self-proving-qualification.yml").read_text(encoding="utf-8")
    assert "CONTINUUM_SELECTED_MODEL" not in text
    assert f"OPENCODE_MODEL: {DEFAULT_PRIMARY_MODEL}" not in text
