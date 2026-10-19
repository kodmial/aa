"""Regression for the Gate A runtime model-policy repair (kodmial/aa#391).

The single live-validated free model serves as both the primary and the
fallback. The model adapter suppresses same-model fallback
(``fallback_model != primary_model``), so a distinct second provider is
neither required nor permitted. The Gate A policy check must therefore
validate the pinned agent/model/fallback defaults instead of demanding a
distinct retired provider, and every live workflow must pin that same
model rather than the retired Muse Spark / Space Bunny pair.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import scripts.run_self_proving_qualification as runner
from aa.config import DEFAULT_AA_AGENT, DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL
from aa.qualification.self_proving import GateEvidence, runtime_model_policy_ok

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"


def test_pinned_same_model_fallback_policy_is_accepted() -> None:
    assert DEFAULT_PRIMARY_MODEL == DEFAULT_FALLBACK_MODEL
    assert runtime_model_policy_ok(
        agent=DEFAULT_AA_AGENT,
        model=DEFAULT_PRIMARY_MODEL,
        fallback_model=DEFAULT_FALLBACK_MODEL,
    )


@pytest.mark.parametrize(
    ("agent", "model", "fallback"),
    [
        ("other-agent", DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL),
        (DEFAULT_AA_AGENT, "opencode/muse-spark-1.3-contributor-free", DEFAULT_FALLBACK_MODEL),
        (DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL, "opencode/space-bunny-free"),
        (DEFAULT_AA_AGENT, "", DEFAULT_FALLBACK_MODEL),
        (DEFAULT_AA_AGENT, DEFAULT_PRIMARY_MODEL, ""),
    ],
)
def test_model_policy_drift_is_rejected(agent: str, model: str, fallback: str) -> None:
    assert not runtime_model_policy_ok(agent=agent, model=model, fallback_model=fallback)


def test_gate_a_accepts_pinned_policy_and_rejects_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sha = "f" * 40

    def _fake_git(args: list[str]) -> str:
        return sha if args and args[0] == "rev-parse" else ""

    monkeypatch.setattr(runner, "_git", _fake_git)
    for name in ("OPENCODE_AGENT", "OPENCODE_MODEL", "OPENCODE_FALLBACK_MODEL"):
        monkeypatch.delenv(name, raising=False)

    accepted = runner._gate_a(sha, "run-391")
    assert accepted.status == "PASS"
    assert accepted.failure_category == ""

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


def test_live_workflows_pin_the_live_validated_model() -> None:
    for path in sorted(WORKFLOWS.glob("*.yml")):
        text = path.read_text(encoding="utf-8")
        assert "OPENCODE_MODEL: opencode/muse-spark" not in text, path.name
        assert "OPENCODE_FALLBACK_MODEL: opencode/space-bunny" not in text, path.name
    self_proving = (WORKFLOWS / "aa-self-proving-qualification.yml").read_text(encoding="utf-8")
    assert f"OPENCODE_MODEL: {DEFAULT_PRIMARY_MODEL}" in self_proving
    assert f"OPENCODE_FALLBACK_MODEL: {DEFAULT_FALLBACK_MODEL}" in self_proving
