"""Versioned English prompt artifacts for the v2 conversation graph.

The system prompts below are the approved baselines, stored as
repository prompt artifacts under ``prompts/`` so they stay versioned,
stable (prompt-cache friendly) and evaluable. This module loads them from
disk; the loaders never log prompt or user content.

The v2 prompts are English only and describe product behavior, never
FAISS/SQLite/RRF/planner mechanics. They are the production prompts used
by the LangGraph runtime after the #118 cutover.
"""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"

AA_AGENT_SYSTEM_V2_VERSION = "aa-agent-system-v2/1"
AA_PLANNER_SYSTEM_V2_VERSION = "aa-planner-system-v2/1"
AA_SUMMARIZER_SYSTEM_V2_VERSION = "aa-summarizer-system-v2/1"
AA_VERIFIER_SYSTEM_V2_VERSION = "aa-verifier-system-v2/1"

AA_AGENT_SYSTEM_V2_PATH = PROMPTS_DIR / "aa-agent-system-v2.md"
AA_PLANNER_SYSTEM_V2_PATH = PROMPTS_DIR / "aa-planner-system-v2.md"
AA_SUMMARIZER_SYSTEM_V2_PATH = PROMPTS_DIR / "aa-summarizer-system-v2.md"
AA_VERIFIER_SYSTEM_V2_PATH = PROMPTS_DIR / "aa-verifier-system-v2.md"


def _load(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"v2 prompt artifact is empty: {path}")
    return text


def load_aa_agent_system_v2() -> str:
    """Return the versioned English production AA Agent system prompt."""
    return _load(AA_AGENT_SYSTEM_V2_PATH)


def load_planner_system_v2() -> str:
    """Return the versioned English Query Planner system prompt (#112 verbatim)."""
    return _load(AA_PLANNER_SYSTEM_V2_PATH)


def load_summarizer_system_v2() -> str:
    """Return the versioned English summarization prompt (#112 verbatim)."""
    return _load(AA_SUMMARIZER_SYSTEM_V2_PATH)


def load_verifier_system_v2() -> str:
    """Return the versioned English claim-level verifier system prompt."""
    return _load(AA_VERIFIER_SYSTEM_V2_PATH)


__all__ = [
    "AA_AGENT_SYSTEM_V2_PATH",
    "AA_AGENT_SYSTEM_V2_VERSION",
    "AA_PLANNER_SYSTEM_V2_PATH",
    "AA_PLANNER_SYSTEM_V2_VERSION",
    "AA_SUMMARIZER_SYSTEM_V2_PATH",
    "AA_SUMMARIZER_SYSTEM_V2_VERSION",
    "AA_VERIFIER_SYSTEM_V2_PATH",
    "AA_VERIFIER_SYSTEM_V2_VERSION",
    "load_aa_agent_system_v2",
    "load_planner_system_v2",
    "load_summarizer_system_v2",
    "load_verifier_system_v2",
]
