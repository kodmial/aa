"""Retired conversational meta/capability turn boundary (issues #105/#106).

Retired by the issue #118 production cutover: ordinary turns no longer
route through deterministic meta/punctuation heuristics in
production. The LangGraph planner owns conversational continuity and
meta questions are answered as generative model turns (issue #301: no
hardcoded conversational replies). This module remains for offline
qualification history only; production code must not import it.
"""

from __future__ import annotations


def is_meta_capability_request(text: str) -> bool:
    """Return whether ``text`` is a meta/capability/identity question.

    Issue #301: deterministic lexical capability routing is retired. The
    model resolves intent from dialogue state, so this always returns
    ``False``. Retained only for backwards import compatibility.
    """
    _ = text
    return False


__all__ = [
    "is_meta_capability_request",
]
