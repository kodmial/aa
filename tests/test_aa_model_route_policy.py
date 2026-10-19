"""AA Gate A must trust Continuum's selected model, not old hardcoded Muse."""

from __future__ import annotations

import pytest

from aa.config import Settings
from scripts.run_self_proving_qualification import _aa_model_policy_valid


@pytest.mark.parametrize(
    ("pool", "selected", "primary", "fallback", "expected"),
    [
        ("opencode/a,opencode/b,opencode/c", "opencode/a", "opencode/a", "opencode/a", True),
        ("opencode/a,opencode/b,opencode/c", "opencode/b", "opencode/b", "opencode/b", True),
        ("opencode/a,opencode/b", "opencode/c", "opencode/c", "opencode/c", False),
        ("opencode/a,opencode/b", "", "opencode/a", "opencode/a", False),
        ("", "opencode/a", "opencode/a", "opencode/a", False),
        ("opencode/a,opencode/a", "opencode/a", "opencode/a", "opencode/a", False),
        ("opencode/a,,opencode/b", "opencode/a", "opencode/a", "opencode/a", False),
        ("opencode/a,opencode/b", "opencode/a", "opencode/b", "opencode/b", False),
        ("opencode/a,opencode/b", "opencode/a", "opencode/a", "opencode/b", False),
    ],
)
def test_policy_requires_single_frozen_approved_model(
    pool: str, selected: str, primary: str, fallback: str, expected: bool
) -> None:
    env = {"OPENCODE_MODELS": pool, "CONTINUUM_SELECTED_MODEL": selected}
    cfg = Settings.from_env(
        {"OPENCODE_AGENT": "aa", "OPENCODE_MODEL": primary, "OPENCODE_FALLBACK_MODEL": fallback}
    )
    assert _aa_model_policy_valid(cfg, env) is expected


def test_non_aa_agent_is_not_accepted() -> None:
    cfg = Settings.from_env(
        {
            "OPENCODE_AGENT": "test",
            "OPENCODE_MODEL": "opencode/a",
            "OPENCODE_FALLBACK_MODEL": "opencode/a",
        }
    )
    assert not _aa_model_policy_valid(
        cfg, {"OPENCODE_MODELS": "opencode/a", "CONTINUUM_SELECTED_MODEL": "opencode/a"}
    )
