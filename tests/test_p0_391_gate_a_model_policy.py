"""Regression for the Gate A runtime model-policy repair (kodmial/aa#391).

AA owns its chatbot route through the repository variables
``AA_BOT_PRIMARY_MODEL`` / ``AA_BOT_FALLBACK_MODEL`` (Work Lock #66).
Gate A validates the effective runtime aliases against that resolved
policy instead of requiring the retired Muse Spark / Space Bunny pair or a
Continuum coding-pool selection. The same verified model may serve as both
primary and fallback because the adapter suppresses same-model fallback; the
current distinct DeepSeek/Beam route is valid too.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_self_proving_qualification as runner
from aa.config import DEFAULT_AA_AGENT, DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL
from aa.qualification.self_proving import GateEvidence, runtime_model_policy_ok

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
OTHER_APPROVED_MODEL = "opencode/other-approved-free"
DISTINCT_FALLBACK_MODEL = "opencode/beam-free"


def test_aa_owned_model_policy_accepts_same_and_distinct_routes() -> None:
    assert runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        model=DEFAULT_PRIMARY_MODEL,
        fallback_model=DEFAULT_FALLBACK_MODEL,
    )
    # The owner may select any approved primary/fallback pair from AA_BOT_*.
    assert runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        model=OTHER_APPROVED_MODEL,
        fallback_model=DISTINCT_FALLBACK_MODEL,
    )
    # The workflow correctly rejects drift from the frozen AA_BOT_* route.
    assert not runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        model=OTHER_APPROVED_MODEL,
        fallback_model=DISTINCT_FALLBACK_MODEL,
        expected_model=DEFAULT_PRIMARY_MODEL,
        expected_fallback_model=DEFAULT_FALLBACK_MODEL,
    )


@pytest.mark.parametrize(
    ("agent", "model", "fallback", "expected_primary", "expected_fallback"),
    [
        ("other-agent", DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL, "", ""),
        (
            DEFAULT_AA_AGENT,
            "opencode/muse-spark-1.3-contributor-free",
            DEFAULT_FALLBACK_MODEL,
            "",
            "",
        ),
        (DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL, "opencode/space-bunny-free", "", ""),
        (DEFAULT_AA_AGENT, "", DEFAULT_FALLBACK_MODEL, "", ""),
        (DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL, "", "", ""),
        (DEFAULT_AA_AGENT, "no-slash", DEFAULT_FALLBACK_MODEL, "", ""),
        (DEFAULT_AA_AGENT, "opencode/../escape", DEFAULT_FALLBACK_MODEL, "", ""),
        (DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL, "open//code", "", ""),
        (
            DEFAULT_AA_AGENT,
            DEFAULT_PRIMARY_MODEL,
            DISTINCT_FALLBACK_MODEL,
            OTHER_APPROVED_MODEL,
            DISTINCT_FALLBACK_MODEL,
        ),
        (
            DEFAULT_AA_AGENT,
            DEFAULT_PRIMARY_MODEL,
            DISTINCT_FALLBACK_MODEL,
            DEFAULT_PRIMARY_MODEL,
            OTHER_APPROVED_MODEL,
        ),
    ],
)
def test_model_policy_drift_is_rejected(
    agent: str,
    model: str,
    fallback: str,
    expected_primary: str,
    expected_fallback: str,
) -> None:
    assert not runtime_model_policy_ok(
        agent=agent,
        model=model,
        fallback_model=fallback,
        expected_model=expected_primary,
        expected_fallback_model=expected_fallback,
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
    monkeypatch.setenv("OPENCODE_MODELS", "opencode/dev-only")
    monkeypatch.setenv("CONTINUUM_SELECTED_MODEL", "opencode/dev-only")
    unaffected = runner._gate_a(sha, "run-391")
    assert unaffected.status == "PASS"

    monkeypatch.setenv("OPENCODE_FALLBACK_MODEL", "opencode/space-bunny-free")
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

    text = (WORKFLOWS / "aa-self-proving-qualification.yml").read_text(encoding="utf-8")
    assert "AA_BOT_PRIMARY_MODEL" in text
    assert "AA_BOT_FALLBACK_MODEL" in text
    assert "python -m aa.bot_models --github-output" in text
    assert "AA_BOT_PRIMARY_MODEL=%s" in text
    assert "AA_BOT_FALLBACK_MODEL=%s" in text
    assert "model-route@" not in text
    assert "CONTINUUM_SELECTED_MODEL" not in text
    assert f"OPENCODE_MODEL: {DEFAULT_PRIMARY_MODEL}" not in text
