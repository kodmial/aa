"""AA Gate A must trust the AA-owned chatbot route, not stale coder pools."""

from __future__ import annotations

import pytest

from aa.config import Settings
from scripts.run_self_proving_qualification import _aa_model_policy_valid


@pytest.mark.parametrize(
    ("bot_primary", "bot_fallback", "primary", "fallback", "expected"),
    [
        ("opencode/a", "opencode/a", "opencode/a", "opencode/a", True),
        ("opencode/a", "opencode/b", "opencode/a", "opencode/b", True),
        ("opencode/a", "opencode/b", "opencode/b", "opencode/b", False),
        ("opencode/a", "opencode/b", "opencode/a", "opencode/b ", True),
        ("", "", "opencode/a", "opencode/b", False),
        ("", "", "", "opencode/b", False),
        ("opencode/a", "opencode/b", "opencode/a", "opencode/c", False),
        ("opencode/a", "", "opencode/a", "opencode/a", True),
        ("opencode/a", "", "opencode/a", "opencode/b", False),
        ("opencode/a", "opencode/b", "opencode/a/", "opencode/b", False),
    ],
)
def test_policy_requires_effective_route_to_match_frozen_aa_policy(
    bot_primary: str,
    bot_fallback: str,
    primary: str,
    fallback: str,
    expected: bool,
) -> None:
    env = {}
    if bot_primary or bot_fallback:
        env = {
            "AA_BOT_PRIMARY_MODEL": bot_primary,
            "AA_BOT_FALLBACK_MODEL": bot_fallback,
        }
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
        cfg, {"AA_BOT_PRIMARY_MODEL": "opencode/a", "AA_BOT_FALLBACK_MODEL": "opencode/a"}
    )


def test_retired_routes_are_never_accepted() -> None:
    cfg = Settings.from_env(
        {
            "OPENCODE_AGENT": "aa",
            "OPENCODE_MODEL": "opencode/muse-spark-1.3-contributor-free",
            "OPENCODE_FALLBACK_MODEL": "opencode/space-bunny-free",
        }
    )
    assert not _aa_model_policy_valid(
        cfg,
        {
            "AA_BOT_PRIMARY_MODEL": "opencode/muse-spark-1.3-contributor-free",
            "AA_BOT_FALLBACK_MODEL": "opencode/space-bunny-free",
        },
    )
