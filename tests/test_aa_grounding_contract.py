"""Grounding contract tests (issue #26).

Locks the Definition of Done: the authoritative system prompt is committed
verbatim, the retrieval/orchestration contract is committed, and the
whole-book coverage loop, stop/saturation rule, sponsor-style behavior
without false identity claims, agent binding, and downstream references are
all explicit.
"""

from __future__ import annotations

import json
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_PATH = REPO_ROOT / "prompts" / "aa-agent-system.md"
CONTRACT_PATH = REPO_ROOT / "docs" / "aa-agent-grounding-contract.md"
OPENCODE_CONFIG_PATH = REPO_ROOT / "opencode.json"


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def test_authoritative_prompt_is_committed_verbatim() -> None:
    text = _read(PROMPT_PATH)
    assert "grounded in the core text of Alcoholics Anonymous" in text
    assert "Never claim that you are a human" in text
    assert "an AA member, the user's actual sponsor, a clinician" in text
    assert "personal sobriety/lived experience" in " ".join(text.split())
    assert "The always-loaded book map is navigation only" in text
    assert "For every substantive user message" in text
    assert "WHOLE-BOOK GROUNDING LOOP" in text
    assert "Check coverage" in text
    assert "If yes, run another retrieval pass" in text
    assert "STOP RULE" in text
    assert "no materially new support" in text
    assert "configured source/context budget is reached" in text
    assert "Never fill missing source support from model memory" in text
    assert "Every substantive claim must be traceable to exact canonical source" in text
    assert "does not establish that answer" in text
    assert "Never complete missing facts from outside\nthe book" in text
    assert "Russian and English are first-class" in text
    assert "deterministic safety layer is authoritative" in text
    assert "Do not provide medication dosing" in text
    assert "sponsor-style" in text
    assert "not role-play deception" in text
    assert "only the read-only AA book tools" in text


def test_grounding_contract_covers_orchestration() -> None:
    text = _read(CONTRACT_PATH)
    # Turn state machine stages 0-8.
    for stage in (
        "Safety/operations gate",
        "Retrieval planning",
        "Whole-corpus hybrid retrieval",
        "Coverage/diversity selection",
        "Exact source loading",
        "Coverage check",
        "Evidence-pack construction",
        "Answer synthesis",
        "Grounding gate",
    ):
        assert stage in text, f"missing stage: {stage}"
    # Whole-book coverage loop and stop rule.
    assert "3-6 retrieval aspects" in text
    assert "Maximum retrieval rounds for MVP: 2" in text
    assert "no materially new support" in text
    assert "source/context budget is reached" in text
    assert "Never truncate a source passage silently" in text
    # Fixed retrieval engine contract.
    assert "SQLite FTS5 BM25" in text
    assert "intfloat/multilingual-e5-base" in text
    assert "IndexFlatIP" in text
    assert "RRF" in text and "k=60" in text
    assert "at most 12 compact candidates per aspect" in text
    # Fixed runtime model policy.
    assert "opencode/muse-spark-1.3-contributor-free" in text
    assert "opencode/space-bunny-free" in text
    assert "no implicit third fallback" in text.lower()
    # Agent binding and tool policy.
    assert "deny-by-default" in text.lower() or "deny by default" in text.lower()
    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        assert tool in text, f"missing tool: {tool}"
    # Downstream references.
    for issue in ("#8", "#17", "#18", "#19", "#9"):
        assert issue in text, f"missing downstream reference: {issue}"
    # Safety handoff and languages.
    assert "#21" in text
    assert "Russian and English are first-class" in text


def test_opencode_config_binds_named_aa_agent() -> None:
    payload = json.loads(_read(OPENCODE_CONFIG_PATH))
    agent = payload["agent"]["aa"]
    assert agent["mode"] == "primary"
    assert "aa-agent-system.md" in str(agent["prompt"])
    permission = agent["permission"]
    assert permission["*"] == "deny"
    for tool in ("book_search", "book_read", "book_expand", "book_section"):
        assert permission[tool] == "allow"
    assert {k for k, v in permission.items() if v == "allow"} == {
        "book_search",
        "book_read",
        "book_expand",
        "book_section",
    }


def test_runtime_defaults_match_model_policy() -> None:
    from aa.config import DEFAULT_FALLBACK_MODEL, DEFAULT_PRIMARY_MODEL, Settings

    assert DEFAULT_PRIMARY_MODEL == "opencode/muse-spark-1.3-contributor-free"
    assert DEFAULT_FALLBACK_MODEL == "opencode/space-bunny-free"
    settings = Settings.from_env({})
    assert settings.opencode_agent == "aa"
    assert settings.opencode_model == DEFAULT_PRIMARY_MODEL
    assert settings.opencode_fallback_model == DEFAULT_FALLBACK_MODEL
