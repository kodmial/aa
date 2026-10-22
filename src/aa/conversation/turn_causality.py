"""End-to-end per-turn causality attribution (kodmial/aa#417).

The qualification and production diagnostics need to know, for one
specific stalled turn, whether the OpenCode request actually started,
whether it completed, failed, or sat in socket silence. This module
carries a privacy-safe causality identity through the turn execution
context: the harness sets ``set_turn_causality`` before dispatching the
update, the Telegram dispatcher applies it inside the turn's own worker
task (contextvars do not cross task boundaries), and the OpenCode HTTP
client stamps every local request-latency audit entry with it.

Only fixed-vocabulary identifiers travel here: a causality id, a stage
label and a bounded turn trace. Never prompts, user text, replies,
passages, session ids or secrets.
"""

from __future__ import annotations

import contextvars
from typing import Any

_CausalityState = dict[str, str]

_EMPTY_STATE: _CausalityState = {}
_STATE: contextvars.ContextVar[_CausalityState] = contextvars.ContextVar(
    "aa-turn-causality", default=_EMPTY_STATE
)

MAX_CAUSALITY_ID_LEN = 64
MAX_STAGE_LEN = 64


def set_turn_causality(causality_id: str, stage: str = "") -> None:
    """Bind the current task context to one turn's causality identity."""
    _STATE.set(
        {
            "causality_id": (causality_id or "").strip()[:MAX_CAUSALITY_ID_LEN],
            "stage": (stage or "").strip()[:MAX_STAGE_LEN],
        }
    )


def clear_turn_causality() -> None:
    """Remove the causality binding from the current context."""
    _STATE.set({})


def current_causality_id() -> str:
    """Return the causality id bound to the current context (may be empty)."""
    state = _STATE.get()
    return str(state.get("causality_id", ""))[:MAX_CAUSALITY_ID_LEN]


def stamp(payload: dict[str, Any]) -> dict[str, Any]:
    """Merge the current causality binding into a privacy-safe audit entry.

    Entries with no bound causality stay untouched: global probes (health,
    shared runtime calls) never claim a turn identity.
    """
    state = _STATE.get()
    causality = str(state.get("causality_id", ""))
    if not causality:
        return dict(payload)
    merged = dict(payload)
    merged["causality_id"] = causality[:MAX_CAUSALITY_ID_LEN]
    stage = str(state.get("stage", ""))
    if stage:
        merged.setdefault("stage", stage[:MAX_STAGE_LEN])
    return merged


__all__ = [
    "MAX_CAUSALITY_ID_LEN",
    "MAX_STAGE_LEN",
    "clear_turn_causality",
    "current_causality_id",
    "set_turn_causality",
    "stamp",
]
