"""Regression: AA chat model choices never depend on Continuum coding routes."""

import json
from pathlib import Path

import pytest

from aa.bot_models import resolve_bot_models
from aa.config import DEFAULT_PRIMARY_MODEL

ROOT = Path(__file__).resolve().parents[1]


def test_bot_models_ignore_coding_routes() -> None:
    env = {
        "OPENCODE_MODELS": "opencode/dev-only",
        "CONTINUUM_SELECTED_MODEL": "opencode/dev-only",
        "OPENCODE_MODEL": "opencode/dev-only",
    }
    assert resolve_bot_models(env) == (DEFAULT_PRIMARY_MODEL, DEFAULT_PRIMARY_MODEL)


def test_aa_primary_and_fallback_are_independent() -> None:
    env = {
        "AA_BOT_PRIMARY_MODEL": "opencode/bot-primary",
        "AA_BOT_FALLBACK_MODEL": "opencode/bot-fallback",
        "CONTINUUM_SELECTED_MODEL": "opencode/dev-only",
    }
    assert resolve_bot_models(env) == ("opencode/bot-primary", "opencode/bot-fallback")


@pytest.mark.parametrize("value", ["no-slash", "opencode/../escape", "open//code"])
def test_bad_bot_model_is_rejected(value: str) -> None:
    with pytest.raises(ValueError):
        resolve_bot_models({"AA_BOT_PRIMARY_MODEL": value})


def test_runtime_and_canary_own_their_models() -> None:
    for filename in ("aa-runtime.yml", "aa-canary.yml"):
        workflow = (ROOT / ".github/workflows" / filename).read_text()
        assert "model-route@" not in workflow
        assert "CONTINUUM_SELECTED_MODEL" not in workflow
        assert "OPENCODE_MODELS" not in workflow
        assert "AA_BOT_PRIMARY_MODEL" in workflow
        assert "AA_BOT_FALLBACK_MODEL" in workflow
        assert "steps.bot_model.outputs.primary" in workflow
        assert "steps.bot_model.outputs.fallback" in workflow


def test_agents_advertise_open_code_tools() -> None:
    config = json.loads((ROOT / "opencode.json").read_text())
    for name, agent in config["agent"].items():
        permission = agent["permission"]
        assert permission["*"] == "ask", name
        assert "deny" not in permission.values(), name
    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        assert config["agent"]["aa"]["permission"][tool] == "allow"


def test_provider_failures_not_used_as_code_repair_authority() -> None:
    workflow = (ROOT / ".github/workflows/aa-self-proving-qualification.yml").read_text()
    admission = workflow.index("const providerBlocked = productFailures.some")
    mutation = workflow.index("const marker = '<!-- aa-self-proving-repair:v1 -->';")
    assert admission < mutation
    assert "OpenCodeProviderAccessError" in workflow[admission:mutation]
    assert "if (providerBlocked)" in workflow[admission:mutation]
