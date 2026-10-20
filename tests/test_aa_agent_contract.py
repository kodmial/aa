"""Authoritative Russian-first AA agent contract tests (issue #26).

Verifies the Definition of Done against #44/#46/#48 without
duplicating retrieval orchestration mechanics in assertions.
"""

from __future__ import annotations

import json
import pathlib

from aa.grounding import TRANSLATION_MARKER_RU

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "aa-agent-system.md"
CONFIG_PATH = ROOT / "opencode.json"
CONTRACT_PATH = ROOT / "docs" / "aa-agent-grounding-contract.md"

ALLOWED_TOOLS = {"book_search", "book_read", "book_expand", "book_section"}


def _prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def test_named_aa_primary_agent_is_explicit() -> None:
    config = _config()
    agent = config["agent"]
    assert isinstance(agent, dict)
    aa = agent["aa"]
    assert isinstance(aa, dict)
    assert aa["mode"] == "primary"
    prompt_ref = str(aa["prompt"])
    assert "prompts/aa-agent-system.md" in prompt_ref


def test_only_four_read_only_aa_tools_are_allowed() -> None:
    config = _config()
    agent = config["agent"]
    assert isinstance(agent, dict)
    aa = agent["aa"]
    assert isinstance(aa, dict)
    permission = aa["permission"]
    assert isinstance(permission, dict)
    assert permission.get("*") == "deny"
    allowed = {name for name, value in permission.items() if value == "allow"}
    assert allowed == ALLOWED_TOOLS


def test_system_prompt_is_english_and_concise() -> None:
    prompt = _prompt()
    without_marker = prompt.replace(TRANSLATION_MARKER_RU, "")
    without_marker.encode("ascii")
    assert len(prompt.splitlines()) <= 150


def test_russian_is_primary_user_facing_language() -> None:
    prompt = _prompt()
    assert "Russian is the primary product language" in prompt
    assert "Russian conversation is the default" in prompt


def test_exact_russian_quote_rule_is_explicit() -> None:
    prompt = _prompt()
    assert "never authorizes a Russian direct quotation" in prompt
    assert TRANSLATION_MARKER_RU in prompt
    assert "Never present a generated" in prompt


def test_english_is_not_canonical_for_russian_production() -> None:
    prompt = _prompt()
    assert "separately versioned reference/control corpus" in prompt


def test_memory_map_rewrites_are_non_evidentiary() -> None:
    prompt = _prompt()
    lowered = prompt.lower()
    assert "model memory" in lowered
    assert "book map" in lowered
    assert "rewritten queries" in lowered or "rewrites" in lowered
    assert "search previews" in lowered


def test_full_orchestration_is_not_embedded_in_prompt() -> None:
    prompt = _prompt()
    assert "planning -> retrieval -> diversity" not in prompt
    assert "evidence pack -> synthesis -> grounding" not in prompt
    assert "aa.grounding" in prompt


def test_identity_boundaries_and_safety_handoff_are_explicit() -> None:
    prompt = _prompt()
    assert "never claim to be" in prompt.lower()
    assert "actual sponsor" in prompt.lower()
    assert "deterministic" in prompt.lower()
    assert "unsupervised alcohol withdrawal" in prompt.lower()


def test_grounding_contract_matches_fixed_architecture() -> None:
    text = CONTRACT_PATH.read_text(encoding="utf-8")
    for marker in ("#44", "#46", "#48", "aa.grounding", "book_search"):
        assert marker in text
    assert "Russian is" in text
    assert "TRANSLATION_MARKER_RU" in text
    assert "prompt" in text.lower() and "orchestrat" in text.lower()
