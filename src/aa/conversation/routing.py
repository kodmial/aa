"""Deterministic turn-routing contract (issue #105).

Explicit pre-retrieval routing for every Telegram text turn. The contract
separates conversational/meta turns from corpus-grounded AA turns so that
ordinary conversation never requires book retrieval merely because it
contains ``?`` or an interrogative such as ``зачем/что/как``.

Routing order (all deterministic, offline, no LLM):

1. safety/emergency gate (owned by :mod:`aa.safety.router`);
2. control/command turns (``/start``, ``/new``, other ``/`` commands);
3. conversational/meta turns (greeting, identity, capabilities, purpose,
   ordinary small-talk follow-up) -> direct named ``aa`` agent path;
4. substantive AA/recovery/book turns -> full retrieval/evidence pipeline;
5. true retrieval/provider/grounding failure -> bounded user-safe failure
   (owned by the orchestrator/app boundary, never invented here).

Invariants enforced by :class:`TurnDecision`:

- ``requires_grounding`` is True iff ``route`` is SUBSTANTIVE;
- conversational/meta turns never require book retrieval;
- substantive recovery/book turns always require grounding;
- emergency/blocked/command turns never require grounding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from aa.retrieval.normalize import normalize_ru, ru_tokens
from aa.safety.router import SafetyDecision, SafetyResult

logger = logging.getLogger("aa.conversation.routing")

AGENT_NAME = "aa"


class TurnRoute(Enum):
    """Authoritative route for one inbound text turn."""

    EMERGENCY = "emergency"
    BLOCKED = "blocked"
    COMMAND = "command"
    CONVERSATIONAL = "conversational"
    SUBSTANTIVE = "substantive"


@dataclass(frozen=True)
class TurnDecision:
    """Typed route decision for one turn.

    ``requires_grounding`` is the single authoritative flag: True only
    for SUBSTANTIVE turns that must pass the full planner/retrieval/
    evidence/synthesis/grounding pipeline. Every other route takes a
    bounded path without book-evidence requirements.
    """

    route: TurnRoute
    reason: str = ""
    requires_grounding: bool = False
    agent: str = AGENT_NAME

    def __post_init__(self) -> None:
        expected = self.route is TurnRoute.SUBSTANTIVE
        if self.requires_grounding != expected:
            raise ValueError("requires_grounding must be True iff route is SUBSTANTIVE")
        if self.route in (TurnRoute.CONVERSATIONAL, TurnRoute.SUBSTANTIVE):
            if self.agent != AGENT_NAME:
                raise ValueError("conversational/substantive turns use the named aa agent")
        if not self.reason:
            raise ValueError("route reason must be non-empty")


# ---------------------------------------------------------------------------
# Deterministic lexical markers (substrings over normalized text).
# ---------------------------------------------------------------------------

# Markers that force the substantive grounded path. Recovery/book
# vocabulary takes precedence over any meta phrasing so grounding is
# never skipped for AA content.
_SUBSTANTIVE_MARKERS: tuple[str, ...] = (
    "книг",
    "книж",
    "страх",
    "боюсь",
    "боязнь",
    "тревог",
    "опасен",
    "шаг",
    "инвентар",
    "обид",
    "молитв",
    "медитац",
    "срыв",
    "сорвал",
    "рецидив",
    "тяга",
    "алког",
    "трезв",
    "пьян",
    "пить",
    "бух",
    "выпив",
    "запой",
    "похмел",
    "семь",
    "жен",
    "жинк",
    "женушк",
    "муж",
    "работ",
    "начальник",
    "содруж",
    "сообществ",
    "спонсор",
    "программ",
    "высш",
    "сил",
    "вера",
    "бог",
    "анонимн",
    "больш",
    "биг",
    "book",
    "fear",
    "step",
    "drink",
    "sober",
    "relapse",
    "craving",
    "family",
)

_GREETING_NORMALIZED: frozenset[str] = frozenset(
    {
        "привет",
        "здравствуйте",
        "здравствуй",
        "добрый день",
        "добрый вечер",
        "доброе утро",
        "добрый утро",
        "спасибо",
        "благодарю",
        "пока",
        "до свидания",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "start",
    }
)

# Explicit meta phrasing about the assistant itself (identity,
# capabilities, purpose). Substrings over normalized text; each entry
# requires a self-reference so world-factual questions (e.g. about
# planets) never match.
_META_SELF_SUBSTRINGS: tuple[str, ...] = (
    "кто ты",
    "ты кто",
    "что ты",
    "ты что",
    "ты бот",
    "ты робот",
    "ты помощник",
    "ты ассистент",
    "ты человек",
    "ты ии",
    "ты нейро",
    "как тебя зовут",
    "как тебя называть",
    "твое имя",
    "твои имя",
    "твое название",
    "что ты можешь",
    "что ты умеешь",
    "что умеешь",
    "что можешь",
    "ты можешь",
    "ты умеешь",
    "ты можешь помочь",
    "чем ты",
    "чем можешь",
    "твои возможности",
    "твои функции",
    "твои функци",
    "твои возможност",
    "зачем ты",
    "для чего ты",
    "почему ты",
    "тогда зачем",
    "а зачем ты",
    "а зачем",
    "смысл тебя",
    "польза от тебя",
    "расскажи о себе",
    "расскажи про себя",
    "о себе",
    "про себя",
    "who are you",
    "what are you",
    "what can you",
    "you can",
    "your name",
)

_SMALLTALK_SUBSTRINGS: tuple[str, ...] = (
    "как дела",
    "как ты",
    "как жизнь",
    "как настроение",
    "что нового",
    "how are you",
)

# Bare discourse follow-ups without recovery content: ordinary
# conversational continuation, never a retrieval trigger on its own.
_DISCOURSE_SUBSTRINGS: tuple[str, ...] = (
    "а дальше",
    "а еще",
    "а ещ",
    "а потом",
    "и дальше",
    "и потом",
    "что дальше",
    "ну и",
    "и что",
    "расскажи",
    "продолж",
    "поясни",
    "объясни",
    "уточни",
    "поподробн",
)

# Factual interrogatives that, without self-reference, mark a world
# question. Such questions default to the substantive path so they fail
# closed safely instead of being hallucinated conversationally.
_FACTUAL_INTERROGATIVES: tuple[str, ...] = (
    "сколько",
    "где",
    "когда",
    "какой",
    "какая",
    "какое",
    "какие",
    "какого",
    "который",
    "чья",
    "чье",
    "сколько спутников",
    "how many",
    "where",
    "when",
    "which",
)


def is_command_turn(text: str) -> bool:
    """Return whether ``text`` is a control/command turn."""
    stripped = text.strip()
    return stripped.startswith("/") and len(stripped) > 1


def _contains_any(normalized: str, markers: tuple[str, ...]) -> bool:
    return any(marker in normalized for marker in markers)


def _has_substantive_markers(normalized: str) -> bool:
    """Return whether recovery/book vocabulary is present.

    Token-aware so short stems (``жен``, ``муж``, ``сил``) do not
    misfire inside unrelated words such as ``нужен``: length-3 stems
    must match a token prefix, longer stems may appear anywhere inside
    a token.
    """
    for token in ru_tokens(normalized):
        for marker in _SUBSTANTIVE_MARKERS:
            if len(marker) <= 3:
                if token == marker or token.startswith(marker):
                    return True
            elif marker in token:
                return True
    return False


def is_conversational_turn(text: str) -> bool:
    """Return whether ``text`` is a conversational/meta turn.

    Deterministic and testable: greetings, assistant-identity /
    capability / purpose phrasing, small talk, and bare discourse
    follow-ups are conversational when they carry no substantive
    recovery/book markers. Anything else (including ``?`` alone or a
    bare factual interrogative) is not conversational.
    """
    stripped = text.strip()
    if not stripped:
        return False
    if is_command_turn(stripped):
        return False
    normalized = normalize_ru(stripped)
    if normalized in _GREETING_NORMALIZED:
        return True
    if _has_substantive_markers(normalized):
        return False
    if _contains_any(normalized, _META_SELF_SUBSTRINGS):
        return True
    if _contains_any(normalized, _SMALLTALK_SUBSTRINGS):
        return True
    if _contains_any(normalized, _DISCOURSE_SUBSTRINGS):
        return True
    if _contains_any(normalized, _FACTUAL_INTERROGATIVES):
        return False
    if "?" in stripped or "？" in stripped:
        return False
    if len(normalized) <= 24:
        return True
    return False


def classify_text_kind(text: str) -> TurnRoute:
    """Classify ``text`` without the safety gate (pure text routing)."""
    stripped = text.strip()
    if not stripped:
        return TurnRoute.BLOCKED
    if is_command_turn(stripped):
        return TurnRoute.COMMAND
    if is_conversational_turn(stripped):
        return TurnRoute.CONVERSATIONAL
    return TurnRoute.SUBSTANTIVE


def route_turn(text: str, safety_result: SafetyResult | None = None) -> TurnDecision:
    """Route one turn through the explicit contract.

    When ``safety_result`` carries EMERGENCY/BLOCK, that verdict takes
    precedence before any ordinary routing and no OpenCode work may be
    scheduled. Otherwise the deterministic text classifier decides
    between COMMAND, CONVERSATIONAL, and SUBSTANTIVE.
    """
    if safety_result is not None:
        if safety_result.decision is SafetyDecision.EMERGENCY:
            return TurnDecision(
                route=TurnRoute.EMERGENCY,
                reason="safety-emergency",
                requires_grounding=False,
            )
        if safety_result.decision is SafetyDecision.BLOCK:
            return TurnDecision(
                route=TurnRoute.BLOCKED,
                reason="safety-block",
                requires_grounding=False,
            )
    else:
        if not text.strip():
            return TurnDecision(
                route=TurnRoute.BLOCKED,
                reason="empty-message",
                requires_grounding=False,
            )
    kind = classify_text_kind(text)
    if kind is TurnRoute.BLOCKED:
        return TurnDecision(
            route=TurnRoute.BLOCKED,
            reason="empty-message",
            requires_grounding=False,
        )
    if kind is TurnRoute.COMMAND:
        return TurnDecision(
            route=TurnRoute.COMMAND,
            reason="control-command",
            requires_grounding=False,
        )
    if kind is TurnRoute.CONVERSATIONAL:
        return TurnDecision(
            route=TurnRoute.CONVERSATIONAL,
            reason="conversational-meta",
            requires_grounding=False,
        )
    return TurnDecision(
        route=TurnRoute.SUBSTANTIVE,
        reason="substantive-grounded",
        requires_grounding=True,
    )


__all__ = [
    "AGENT_NAME",
    "TurnDecision",
    "TurnRoute",
    "classify_text_kind",
    "is_command_turn",
    "is_conversational_turn",
    "route_turn",
]
