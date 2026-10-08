"""Whole-turn answer adequacy gate for the v2 conversation path (kodmial/aa#251).

The per-unit verifier proves that individual substantive claims are
supported by cited passages. It does not prove that the whole delivered
turn answers the user's actual request: an unlimited sequence of true
generic empathy/capability/clarification units passes per-unit checks
while delivering zero practical help for an answerable recovery question.

This module owns the production whole-turn invariant:

- a planner error is never construed as legitimate glue;
- a substantive personal recovery/support request requires at least one
  practical, relevant, verifier-supported book unit answering that request;
- validated source identifiers alone never prove topical relevance;
- pure greetings, honest self-identity and genuinely contentless turns
  keep natural non-book glue;
- without book evidence for a substantive request the turn serves explicit
  honest unavailability, never plausible generic help.

Privacy: only counts, outcome tokens and digests travel outward. No user
text, evidence text, prompts or identifiers are logged or stored here.
Callers must never persist inputs alongside the returned assessment.

The deterministic assessment below is a generic structural/semantic
signal, never a textual exact-question whitelist: it uses message shape,
generic interrogative structure and evidence-grounded token overlap, not
memorized user questions. A future model-based semantic judge may replace
the overlap core; the invariant and the telemetry contract stay stable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("aa.conversation.answer_adequacy")

PLANNER_REASON_LEGITIMATE_GLUE = "legitimate-glue"
PLANNER_REASON_SUBSTANTIVE_WITH_QUERIES = "substantive-with-queries"
PLANNER_REASON_PROVIDER_ERROR = "provider-error"
PLANNER_REASON_TIMEOUT = "timeout"
PLANNER_REASON_INVALID = "invalid"
PLANNER_REASON_UNKNOWN = "unknown"

ADEQUACY_PASS = "pass"
ADEQUACY_FAIL = "fail"

FAILURE_OK_GLUE = ""
FAILURE_ALL_GLUE_FOR_SUBSTANTIVE = "all-glue-for-substantive"
FAILURE_IRRELEVANT_CITATION = "irrelevant-citation"
FAILURE_UNSUPPORTED_CLAIM = "unsupported-claim"
FAILURE_NO_EVIDENCE_SUBSTANTIVE = "no-evidence-substantive"
FAILURE_UNAVAILABLE_VERIFIER = "unavailable-verifier"
FAILURE_BUDGET_EXCEEDED = "budget-exceeded"
FAILURE_REPAIR_FAILED = "adequacy-repair-failed"

RECOVERY_MAX_QUERIES = 6

_WORD_RE = re.compile(r"[A-Za-z\u0400-\u04ff]+")

_GENERIC_STOPWORDS = frozenset(
    {
        "это",
        "этот",
        "эта",
        "эти",
        "как",
        "что",
        "для",
        "или",
        "при",
        "про",
        "без",
        "над",
        "под",
        "между",
        "через",
        "после",
        "перед",
        "только",
        "уже",
        "еще",
        "ещё",
        "очень",
        "можно",
        "нужно",
        "надо",
        "меня",
        "тебя",
        "себя",
        "нас",
        "вас",
        "они",
        "оно",
        "также",
        "так",
        "там",
        "тут",
        "здесь",
        "тогда",
        "когда",
        "где",
        "кто",
        "все",
        "всё",
        "весь",
        "сам",
        "сама",
        "мой",
        "моя",
        "твой",
        "твоя",
        "наш",
        "ваш",
        "его",
        "ее",
        "её",
        "их",
        "был",
        "была",
        "было",
        "были",
        "есть",
        "будет",
        "будут",
        "тебе",
        "мне",
        "вам",
        "нам",
        "and",
        "the",
        "with",
        "from",
        "that",
        "this",
        "have",
        "will",
    }
)

_GENERIC_INTERROGATIVE_MARKERS = (
    "как",
    "что",
    "почему",
    "зачем",
    "когда",
    "где",
    "куда",
    "сколько",
    "можно ли",
    "помоги",
    "помогите",
    "подскажи",
    "подскажите",
    "расскажи",
    "расскажите",
    "посоветуй",
    "посоветуйте",
    "что делать",
    "как быть",
)

# Generic capability/identity category (never an exact-question list):
# questions about what the assistant is or can do keep natural non-book
# glue. Recovery/support requests about the user's situation stay on the
# substantive path even when they share interrogative words.
_GENERIC_META_MARKERS = (
    "ты кто",
    "кто ты",
    "что ты",
    "чем ты",
    "зачем ты",
    "ты умеешь",
    "ты можешь",
    "что умеешь",
    "что можешь",
    "твоя роль",
    "твои возможности",
    "что вы можете",
    "что вы умеете",
    "вы можете",
    "вы умеете",
    "ваши возможности",
    "ваша роль",
)

# Generic structural capability vocabulary (individual tokens only, never
# an exact-question phrase): second-person references plus capability
# words plus neutral filler scaffolding around a short capability probe.
# Natural paraphrases split a capability request across non-contiguous
# tokens, so token-level structure is required; contiguous marker
# substrings alone under-recognize ordinary probes and misroute them to
# the substantive book path.
_SECOND_PERSON_TOKENS = frozenset(
    {
        "ты",
        "вы",
        "твой",
        "твоя",
        "твои",
        "твоих",
        "ваш",
        "ваша",
        "ваши",
        "тебя",
        "вас",
        "тобой",
    }
)

_CAPABILITY_TOKENS = frozenset(
    {
        "можешь",
        "можете",
        "умеешь",
        "умеете",
        "помочь",
        "помогать",
        "полезен",
        "полезна",
        "польза",
        "возможности",
        "роль",
    }
)

_META_FILLER_TOKENS = frozenset(
    {
        "вообще",
        "здесь",
        "тут",
        "просто",
        "именно",
        "пожалуйста",
        "слушай",
        "а",
        "же",
        "то",
        "ли",
    }
)


def is_meta_request(user_message: str) -> bool:
    """Whether a turn is a pure identity/capability probe.

    Fail-closed to substantive: a combined request (capability prefix
    plus a substantive help request) is never meta, even when it
    contains a meta marker substring. Only a short single-segment probe
    with no recovery-domain vocabulary and no substantive remainder
    counts as meta.
    """
    cleaned = " ".join((user_message or "").split()).strip()
    if not cleaned:
        return False
    lowered = cleaned.casefold()
    has_marker = any(marker in lowered for marker in _GENERIC_META_MARKERS)
    if not has_marker and not _has_structural_capability_shape(cleaned):
        return False
    # Any recovery-domain vocabulary makes the turn substantive, never
    # pure meta (for example a capability prefix plus an evening-craving
    # request).
    if _has_recovery_domain(lowered):
        return False
    # A combined greeting/meta prefix plus continuation is substantive.
    segments = [part.strip() for part in re.split(r"[.!?…?]+", cleaned) if part.strip()]
    if len(segments) > 1:
        return False
    if len(cleaned) > 60:
        return False
    # Strip meta markers; if a meaningful remainder remains, the turn is
    # combined rather than a pure probe.
    remainder = lowered
    for marker in _GENERIC_META_MARKERS:
        if marker in remainder:
            remainder = remainder.replace(marker, " ")
    remainder = " ".join(remainder.split())
    if not remainder:
        return True
    if any(marker in remainder for marker in ("что делать", "как быть")):
        return False
    if _has_recovery_domain(remainder):
        return False
    # Generic interrogative/request verbs alone (for example asking
    # about capabilities) stay meta; any other substantive token in the
    # remainder proves a combined request. Tokens that only restate the
    # meta probe itself (overlapping marker inflections) are ignored.
    # Neutral filler scaffolding around a short probe (general adverbs
    # of place/manner and politeness particles) likewise stays meta:
    # only a genuine topic remainder proves a combined request.
    interrogative_words: set[str] = set()
    for marker in _GENERIC_INTERROGATIVE_MARKERS:
        interrogative_words.update(_WORD_RE.findall(marker.casefold()))
    meta_words: set[str] = set()
    for marker in _GENERIC_META_MARKERS:
        meta_words.update(_WORD_RE.findall(marker.casefold()))
    remainder_tokens = _content_tokens(remainder) - interrogative_words - meta_words
    # "помочь"/"подсказать" inflections are request verbs, not proof of a
    # substantive topic on their own; require another topic token.
    remainder_tokens -= {"помочь", "подсказать", "подскажите", "помогите"}
    remainder_tokens -= _CAPABILITY_TOKENS
    remainder_tokens -= _META_FILLER_TOKENS
    if remainder_tokens:
        return False
    return True


def _has_structural_capability_shape(cleaned: str) -> bool:
    """Whether a short probe has generic capability-question structure.

    Token-level only, never an exact-question phrase: a single short
    interrogative segment addressing the assistant in the second person
    with capability vocabulary. This recognizes ordinary paraphrases
    whose capability words are split across non-contiguous tokens.
    """
    lowered = (cleaned or "").casefold()
    if not lowered or "?" not in cleaned:
        return False
    words = set(_WORD_RE.findall(lowered))
    if not (words & _SECOND_PERSON_TOKENS):
        return False
    return bool(words & _CAPABILITY_TOKENS)


# Generic alcohol-recovery domain stems for topical relevance (never an
# exact-question list): lets colloquial, slang and typo paraphrases of the
# same recovery topic count as relevant while a verified but unrelated
# book fact (for example finance) still fails. Only the domain is
# recognized here; the exact request is never matched. Short inflected
# forms (for example first-person drinking disclosures) are covered via
# short stems, never via memorized user sentences.
_RECOVERY_DOMAIN_STEMS = (
    "тяг",
    "выпи",
    "выпь",
    "буха",
    "бухл",
    "пить",
    "пью",
    "пьет",
    "пьешь",
    "пьем",
    "пьете",
    "пьют",
    "пьян",
    "пья",
    "трезв",
    "срыв",
    "запо",
    "алког",
    "алко",
    "пив",
    "похмел",
)

# Whole-word drinking past-tense forms: bare "пил" as a substring matches
# "пилить"/"пилот", so these are matched as exact tokens only. "выпил" and
# longer drinking forms are already covered by the "выпи"/"выпь" prefixes.
_RECOVERY_EXACT_TOKENS = frozenset({"пил", "пила", "пило", "пили"})


def _has_recovery_domain(text: str) -> bool:
    """Whether text contains generic alcohol-recovery domain vocabulary."""
    lowered = (text or "").casefold().replace("ё", "е")
    if not lowered:
        return False
    # Token-anchored matching only: bare-substring checks let "вып" match
    # "выполнить"/"выпуск", "пье" match "пьеса", and mid-word "пив"/"алко"
    # match unrelated words. Bare "вып"/"пье" are removed above in favor of
    # drinking-specific prefixes/inflections; the rest match at token start.
    tokens = _WORD_RE.findall(lowered)
    for token in tokens:
        if token in _RECOVERY_EXACT_TOKENS:
            return True
        for stem in _RECOVERY_DOMAIN_STEMS:
            if token.startswith(stem):
                return True
    return False


# Step-referent vocabulary for multi-turn topic fidelity (generic, never an
# exact-question list): any numbered-step mention keeps the turn
# substantive and participates in relevance alignment. Ordinal words and
# digits are both recognized so paraphrases still align.
_STEP_WORD_RE = re.compile(
    r"шаг\w*|step\s*\d+|\bперв\w*\s+шаг\w*|\bвтор\w*\s+шаг\w*"
    r"|\bтрет\w*\s+шаг\w*|\bчетверт\w*\s+шаг\w*"
    r"|\bпят\w*\s+шаг\w*|\bшест\w*\s+шаг\w*"
)
_STEP_DIGIT_RE = re.compile(r"шаг\w*\s*(?:№\s*)?(\d{1,2})", re.IGNORECASE)
_STEP_ORDINAL_TO_NUMBER = (
    ("перв", 1),
    ("втор", 2),
    ("трет", 3),
    ("четверт", 4),
    ("пят", 5),
    ("шест", 6),
    ("седьм", 7),
    ("восьм", 8),
    ("девят", 9),
    ("десят", 10),
    ("одиннадцат", 11),
    ("двенадцат", 12),
)

# Positively proven greeting vocabulary (generic, never an exact-question
# list): only messages composed entirely of conversational greeting,
# thanks or farewell words count as proven glue. Anything else, including
# short first-person disclosures, stays substantive.
_GREETING_VOCABULARY = frozenset(
    {
        "привет",
        "приветствую",
        "здравствуй",
        "здравствуйте",
        "добрый",
        "добрая",
        "доброе",
        "день",
        "вечер",
        "утро",
        "спасибо",
        "благодарю",
        "пока",
        "свидания",
        "до",
    }
)

# First-person disclosure signals (generic, never an exact-question list):
# short personal statements default to substantive/support-requiring and
# are never proven glue from length alone.
_PERSONAL_DISCLOSURE_RE = re.compile(
    r"\bя\b|\bмне\b|\bменя\b|\bмной\b|\bмною\b|\bмой\b|\bмоя\b|\bмое\b"
    r"|\bмою\b|\bмои\b|\bпью\b|\bпил\w*|\bвыпил\w*|\bне\s+могу\b"
    r"|\bкаждый\s+день\b",
    re.IGNORECASE,
)

# Demonstrative/elliptical follow-up cues: the live request alone carries
# no referent and must be resolved from conversation context.
_FOLLOWUP_REFERENCE_RE = re.compile(
    r"\bэто\b|\bэтот\b|\bэта\b|\bэти\b|\bэтом\b|\bэтого\b|\bэтим\b|\bэту\b"
    r"|\bтакой\b|\bтаком\b|\bтам\b|\bтогда\b|\bдальше\b|\bпотом\b",
    re.IGNORECASE,
)


def _has_step_reference(text: str) -> bool:
    """Whether text carries a generic numbered-step referent."""
    lowered = (text or "").casefold()
    if not lowered:
        return False
    if _STEP_WORD_RE.search(lowered) is not None:
        return True
    return _STEP_DIGIT_RE.search(lowered) is not None


def extract_step_numbers(text: str) -> set[int]:
    """Extract generic numbered-step referents from text (digits/ordinals)."""
    lowered = (text or "").casefold()
    found: set[int] = set()
    if not lowered:
        return found
    for match in _STEP_DIGIT_RE.finditer(lowered):
        try:
            number = int(match.group(1))
        except (TypeError, ValueError):
            continue
        if 1 <= number <= 12:
            found.add(number)
    for stem, number in _STEP_ORDINAL_TO_NUMBER:
        if (
            re.search(rf"\b{stem}\w*\s+шаг\w*\b", lowered) is not None
            or re.search(rf"\bшаг\w*\s+{stem}\w*\b", lowered) is not None
        ):
            found.add(number)
    return found


def _has_personal_disclosure(text: str) -> bool:
    """Whether text carries a generic first-person disclosure signal."""
    if not text:
        return False
    if _PERSONAL_DISCLOSURE_RE.search(text) is not None:
        return True
    return _has_recovery_domain(text)


def _is_positive_greeting(cleaned: str) -> bool:
    """Whether a short message is positively a greeting/thanks/farewell."""
    lowered = (cleaned or "").casefold().strip()
    if not lowered:
        return False
    stripped = re.sub(r"[!.,…?;:]+", " ", lowered)
    words = [item for item in _WORD_RE.findall(stripped) if len(item) >= 2]
    if not words:
        return False
    if any(len(item) < 2 for item in words):
        return False
    # Bare time fragments ("день"/"вечер"/"утро") and the preposition "до"
    # are not standalone greetings: a single-word "вечер" is an elliptical
    # time reference, not contentless glue. They count only inside a
    # multi-word greeting such as "добрый вечер" or "до свидания".
    if len(words) == 1 and words[0] in frozenset({"день", "вечер", "утро", "до"}):
        return False
    return all(item in _GREETING_VOCABULARY for item in words)


def _has_substantive_context(summary: str, recent: Sequence[str] | None) -> bool:
    """Whether conversation context already carries a substantive topic."""
    if _has_recovery_domain(summary) or _has_step_reference(summary):
        return True
    for item in list(recent or [])[:4]:
        text = str(item or "")
        if _has_recovery_domain(text) or _has_step_reference(text):
            return True
    return False


def _is_context_dependent_followup(text: str) -> bool:
    """Whether the live turn structurally depends on prior dialogue.

    Context rescue is deliberately narrow: explicit demonstratives,
    non-numbered step references, or very short generic follow-up words.
    Short length alone is never enough, so a terse but explicit topic
    pivot cannot inherit stale recovery context.
    """
    cleaned = " ".join((text or "").split()).strip()
    if not cleaned:
        return False
    if extract_step_numbers(cleaned):
        return False
    if _FOLLOWUP_REFERENCE_RE.search(cleaned) is not None:
        return True
    if _has_step_reference(cleaned):
        return True
    words = [item.casefold() for item in _WORD_RE.findall(cleaned) if len(item) >= 2]
    if len(words) > 3:
        return False
    content = _content_tokens(cleaned)
    return bool(content) and content.issubset(
        {"почему", "зачем", "дальше", "теперь", "потом", "значит", "делать"}
    )


def resolve_effective_request(
    user_message: str,
    summary: str = "",
    recent: Sequence[str] | None = None,
) -> str:
    """Resolve the live request against conversation context.

    Short elliptical follow-ups (for example references to a previously
    discussed step) carry no standalone referent; the resolved text joins
    the live message with the running summary and recent turns so the
    planner, retrieval, answer and adequacy stages share one referent.
    Generic structural resolution only: no exact user question is matched.
    """
    cleaned = " ".join((user_message or "").split()).strip()
    if not cleaned:
        return ""
    # Step-switch fidelity: an explicit numbered step in the live turn is
    # self-contained and wins over conversation history. Unioning context
    # here would merge the previous step with the newly requested one, so a
    # stale-step reply could intersect the unioned set and survive a switch.
    if extract_step_numbers(cleaned):
        return cleaned
    summary_cleaned = " ".join((summary or "").split()).strip()
    recent_texts = [
        " ".join(str(item).split()).strip() for item in list(recent or [])[:2] if str(item).strip()
    ]
    needs_context = _is_context_dependent_followup(cleaned)
    if not needs_context:
        return cleaned
    parts = [cleaned]
    if summary_cleaned:
        parts.append(f"Контекст: {summary_cleaned[:400]}")
    for item in recent_texts:
        if item and item.casefold() not in cleaned.casefold():
            parts.append(f"Ранее: {item[:240]}")
    return " ".join(parts).strip()


def planner_reason_for(query_count: int, raw_outcome: str) -> str:
    """Map a stored planner outcome to an explicit causal reason.

    A planner error is never construed as legitimate glue. Legacy outcome
    tokens (``ok``/``empty``/``invoked``/``glue``) are preserved for
    backward compatibility and mapped to the explicit reason vocabulary;
    error tokens (``timeout``/``invalid``/``failed``/``provider-error``)
    always map to error reasons.
    """
    normalized = (raw_outcome or "").strip().casefold()
    try:
        count = int(query_count)
    except (TypeError, ValueError):
        count = 0
    if normalized in ("timeout",):
        return PLANNER_REASON_TIMEOUT
    if normalized in ("invalid",):
        return PLANNER_REASON_INVALID
    if normalized in ("failed", "provider-error", "error"):
        return PLANNER_REASON_PROVIDER_ERROR
    if count and count > 0:
        return PLANNER_REASON_SUBSTANTIVE_WITH_QUERIES
    if normalized in ("ok", "empty", "legitimate-glue", "glue", "invoked", "skipped-glue", ""):
        # Zero queries with a non-error outcome is the only legitimate
        # glue signal. Errors are handled above and never reach here.
        return PLANNER_REASON_LEGITIMATE_GLUE
    return PLANNER_REASON_UNKNOWN


def _content_tokens(text: str) -> set[str]:
    """Return generic substantive word tokens for overlap checks."""
    tokens: set[str] = set()
    for raw in _WORD_RE.findall((text or "").casefold()):
        if len(raw) < 4:
            continue
        if raw in _GENERIC_STOPWORDS:
            continue
        tokens.add(raw)
    return tokens


def _token_prefixes(tokens: set[str], width: int = 4) -> set[str]:
    """Return fixed-width prefixes so paraphrase inflections still overlap."""
    return {token[:width] for token in tokens if len(token) >= width}


def is_proven_glue_message(
    user_message: str,
    *,
    summary: str = "",
    recent_count: int = 0,
    recent: Sequence[str] | None = None,
) -> bool:
    """Whether a turn is positively proven contentless conversational glue.

    Conservative fail-closed default: every uncertain turn is treated as
    potentially substantive, never as proven glue. A message is glue only
    when it is positively a greeting/thanks/farewell composed entirely of
    conversational vocabulary, with no question mark, no generic
    interrogative/request structure, no recovery-domain or step referent,
    no first-person disclosure, and no substantive continuation. Length or
    the absence of a question mark alone never proves glue. Conversation
    context participates: an active substantive topic in the summary or
    recent turns keeps borderline short text substantive.

    Generic structural signal only: no exact user question is matched.
    """
    cleaned = " ".join((user_message or "").split()).strip()
    if not cleaned:
        return False
    if len(cleaned) > 60:
        return False
    if "?" in cleaned:
        return False
    lowered = cleaned.casefold()
    if any(marker in lowered for marker in _GENERIC_INTERROGATIVE_MARKERS):
        return False
    words = [item for item in _WORD_RE.findall(lowered) if len(item) >= 2]
    if len(words) > 6:
        return False
    # More than one sentence-shaped segment suggests a greeting plus a
    # substantive continuation rather than a pure greeting.
    segments = [part.strip() for part in re.split(r"[.!?…]+", cleaned) if part.strip()]
    if len(segments) > 1:
        return False
    if _has_recovery_domain(lowered):
        return False
    if _has_step_reference(lowered):
        return False
    if _has_personal_disclosure(cleaned):
        return False
    # Conversational context is load-bearing: a short disclosure after a
    # recovery discussion is a continuation, never empty glue.
    recent_texts = list(recent or [])
    substantive_context = _has_substantive_context(summary, recent_texts)
    if not substantive_context and recent_count > 0 and summary.strip():
        substantive_context = True
    if substantive_context and not _is_positive_greeting(cleaned):
        return False
    return _is_positive_greeting(cleaned)


def build_recovery_queries(
    user_message: str,
    *,
    summary: str = "",
    recent: Sequence[str] | None = None,
    max_queries: int = RECOVERY_MAX_QUERIES,
) -> list[str]:
    """Build one bounded recovery query set from the actual turn context.

    Uses only the live user turn plus conversation context against the
    canonical RU book. Never returns canned generic queries and never
    matches an exact-question whitelist: the live request text is the
    primary query and context only resolves it.
    """
    cleaned = " ".join((user_message or "").split()).strip()
    if not cleaned:
        return []
    queries: list[str] = [cleaned]
    summary_cleaned = " ".join((summary or "").split()).strip()
    if summary_cleaned:
        candidate = f"{cleaned} Контекст: {summary_cleaned[:240]}"
        candidate = " ".join(candidate.split()).strip()
        if candidate and candidate.casefold() not in {item.casefold() for item in queries}:
            queries.append(candidate)
    for item in list(recent or [])[:2]:
        text = " ".join(str(item).split()).strip()
        if not text or len(text) < 8:
            continue
        candidate = f"{cleaned} Ранее: {text[:200]}"
        candidate = " ".join(candidate.split()).strip()
        if candidate.casefold() in {entry.casefold() for entry in queries}:
            continue
        queries.append(candidate)
        if len(queries) >= max_queries:
            break
    return queries[: max(1, max_queries)]


def new_turn_trace_id() -> str:
    """Return a privacy-safe random per-message trace identifier."""
    return uuid.uuid4().hex[:16]


def runtime_sha(*, repo_root: Path | None = None) -> str:
    """Return the actual runtime SHA for causal telemetry (never secrets)."""
    try:
        root = repo_root
        if root is None:
            here = Path(__file__).resolve()
            for parent in (here, *here.parents):
                if (parent / ".git").exists():
                    root = parent
                    break
            if root is None:
                return "unknown"
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=str(root),
            check=False,
        )
        sha = (proc.stdout or "").strip().lower()
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    except Exception:
        pass
    return "unknown"


@dataclass(frozen=True)
class AdequacyAssessment:
    """Whole-turn adequacy verdict for one delivered candidate."""

    substantive_request: bool
    technically_grounded: bool
    answers_request: bool
    verdict: str
    failure_category: str
    verified_book_units: int
    evidence_passages: int


def _supported_book_units(grounding_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(grounding_result, dict):
        return []
    units = grounding_result.get("units", [])
    if not isinstance(units, list):
        return []
    supported: list[dict[str, Any]] = []
    for unit in units:
        if not isinstance(unit, dict):
            continue
        if unit.get("scope") != "book":
            continue
        if unit.get("supported") is not True:
            continue
        if not unit.get("evidence_passage_ids"):
            continue
        supported.append(unit)
    return supported


def assess_turn_adequacy(
    *,
    user_message: str,
    reply: str,
    evidence_pack: Sequence[dict[str, Any]],
    grounding_result: dict[str, Any] | None,
    planner_reason: str = PLANNER_REASON_UNKNOWN,
    verifier_outcome: str = "unknown",
    unavailable_units: int = 0,
    turn_budget_exceeded: bool = False,
    summary: str = "",
    recent: Sequence[str] | None = None,
    resolved_request: str = "",
    prior_user_messages: Sequence[str] = (),
) -> AdequacyAssessment:
    """Judge the whole turn: task, context, candidate and relevant evidence.

    Invariant for any substantive personal recovery/support request: at
    least one practical, relevant, book-supported explanation/action
    answering that request must be present and verified. Validated source
    identifiers alone never prove topical relevance: the supported unit
    must share substantive content with both the cited exact passages and
    the resolved user intent, including the active numbered-step referent
    carried across turns. A cited passage from another step or irrelevant
    book material fails even with token overlap. Pure greetings, honest
    self-identity and genuinely contentless turns keep a passing glue
    verdict without book evidence.

    ``prior_user_messages`` carries the immediately preceding user turns
    in the same conversation (oldest first, generic context only). A
    terse contextual follow-up carries little topical content on its own;
    the planner already resolves such follow-ups against history, so
    relevance here also resolves against that history instead of failing
    a continuous grounded answer for not echoing a generic follow-up.
    The substantive/glue determination itself stays on the current turn
    only, so context never turns a substantive request into glue.
    """
    _ = planner_reason
    cleaned_reply = (reply or "").strip()
    pack = [item for item in (evidence_pack or []) if isinstance(item, dict)]
    supported_units = _supported_book_units(grounding_result)
    verified_count = len(supported_units)
    recent_texts = [str(item) for item in (recent or []) if str(item).strip()]
    try:
        recent_count = len(recent_texts)
    except Exception:
        recent_count = 0
    effective_request = (resolved_request or "").strip() or resolve_effective_request(
        user_message, summary=summary, recent=recent_texts
    )
    substantive = not is_proven_glue_message(
        user_message, summary=summary, recent_count=recent_count, recent=recent_texts
    )
    if substantive and is_meta_request(user_message):
        substantive = False
    if turn_budget_exceeded:
        return AdequacyAssessment(
            substantive_request=substantive,
            technically_grounded=False,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_BUDGET_EXCEEDED,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    try:
        unavailable = int(unavailable_units or 0)
    except (TypeError, ValueError):
        unavailable = 0
    if unavailable != 0 or verifier_outcome.strip() in (
        "unavailable",
        "partial-unavailable",
        "skipped-turn-budget",
    ):
        return AdequacyAssessment(
            substantive_request=substantive,
            technically_grounded=False,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_UNAVAILABLE_VERIFIER,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    if not substantive:
        # Pure glue path: no book evidence is required and no relevance
        # check applies. An unsupported substantive claim still fails.
        if grounding_result is not None:
            try:
                all_supported = bool(grounding_result.get("all_required_supported", True))
            except Exception:
                all_supported = True
            if not all_supported and verified_count == 0:
                # Glue turn with an unsupported claim is not adequate.
                return AdequacyAssessment(
                    substantive_request=False,
                    technically_grounded=False,
                    answers_request=False,
                    verdict=ADEQUACY_FAIL,
                    failure_category=FAILURE_UNSUPPORTED_CLAIM,
                    verified_book_units=verified_count,
                    evidence_passages=len(pack),
                )
        return AdequacyAssessment(
            substantive_request=False,
            technically_grounded=verified_count > 0,
            answers_request=True,
            verdict=ADEQUACY_PASS,
            failure_category=FAILURE_OK_GLUE,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    # Substantive path from here.
    if not cleaned_reply:
        return AdequacyAssessment(
            substantive_request=True,
            technically_grounded=False,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_NO_EVIDENCE_SUBSTANTIVE,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    if not pack or verified_count == 0:
        return AdequacyAssessment(
            substantive_request=True,
            technically_grounded=False,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_ALL_GLUE_FOR_SUBSTANTIVE
            if cleaned_reply
            else FAILURE_NO_EVIDENCE_SUBSTANTIVE,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    # Relevance: at least one supported book unit must share substantive
    # content with its cited exact passages, and the request must share
    # substantive content with the evidence-backed answer. This catches an
    # all-glue reply served with an available pack, a verified but
    # unrelated book fact, a citation present but without a useful step,
    # and an unsupported paraphrase that merely names the book.
    pack_by_id: dict[str, str] = {}
    for item in pack:
        passage_id = item.get("passage_id")
        text = item.get("text")
        if isinstance(passage_id, str) and passage_id and isinstance(text, str) and text:
            pack_by_id[passage_id] = text
    request_tokens = _content_tokens(effective_request)
    request_prefixes = _token_prefixes(request_tokens)
    # Step-switch fidelity: the live turn's explicit numbered step wins over
    # context. The resolved text unions history for elliptical follow-ups,
    # so extracting steps from it alone would yield {old, new} after a
    # switch and let a stale-step citation intersect the union. When the
    # live message names a step, require the citation to name that step.
    live_steps = extract_step_numbers(user_message)
    if live_steps:
        request_steps = live_steps
    else:
        request_steps = extract_step_numbers(effective_request)
    request_domain = _has_recovery_domain(effective_request) or _has_recovery_domain(user_message)
    # Contextual follow-up resolution (generic, turn-independent): a terse
    # follow-up carries little topical content on its own, so relevance
    # also resolves against the immediately preceding user turns. The
    # current-turn check stays primary; context only rescues continuity.
    context_text = " ".join(
        str(item).strip() for item in (prior_user_messages or []) if str(item).strip()
    )
    context_prefixes = _token_prefixes(_content_tokens(context_text)) if context_text else set()
    relevant_unit_found = False
    for unit in supported_units:
        cited = unit.get("evidence_passage_ids", [])
        if not isinstance(cited, list) or not cited:
            continue
        cited_texts = [
            pack_by_id[item] for item in cited if isinstance(item, str) and item in pack_by_id
        ]
        if not cited_texts:
            continue
        unit_tokens = _content_tokens(str(unit.get("text", "") or reply))
        if not unit_tokens:
            continue
        evidence_tokens: set[str] = set()
        for passage_text in cited_texts:
            evidence_tokens |= _content_tokens(passage_text)
        if not evidence_tokens:
            continue
        unit_prefixes = _token_prefixes(unit_tokens)
        evidence_prefixes = _token_prefixes(evidence_tokens)
        if len(unit_prefixes & evidence_prefixes) < 1:
            continue
        # Practical step: the supported unit itself must carry declarative
        # guidance beyond a bare question or generic offer. The overlap
        # above already ties it to exact evidence; here require that the
        # unit is not only a question.
        unit_text = str(unit.get("text", "") or reply).strip()
        if len(unit_text) < 20:
            continue
        if unit_text.rstrip().endswith("?") and len(unit_text) < 60:
            continue
        # Numbered-step fidelity: when the resolved request names a step,
        # only an explicit step mismatch fails. A verified passage from
        # another step (for example Step Three decisions served for a
        # Step One question) fails even when generic vocabulary overlaps
        # on words such as step or decision. Evidence that paraphrases
        # the requested step without literally naming a step number
        # must not fail here; topical relevance below still applies.
        if request_steps:
            cited_steps: set[int] = set()
            for passage_text in cited_texts:
                cited_steps |= extract_step_numbers(passage_text)
            cited_steps |= extract_step_numbers(unit_text)
            if cited_steps and not (request_steps & cited_steps):
                continue
        # Topical relevance to the resolved intent, with same-conversation
        # context rescue: the evidence-backed unit must share substantive
        # content with the resolved request. At least two distinct prefix
        # overlaps prove semantic alignment; a single overlap proves
        # relevance once the numbered-step referent already aligns above
        # or when a direct lexical tie exists. With no direct overlap only
        # a shared recovery domain (short disclosures, slang, typos) still
        # counts while an unrelated book fact fails. A terse contextual
        # follow-up additionally resolves against the immediately
        # preceding user turns instead of failing a continuous grounded
        # answer for not echoing a generic follow-up. Stale-context guard:
        # context rescues only a generic/terse current turn (at most two
        # substantive tokens); an explicit new-topic pivot must match on
        # its own merits, otherwise a prior craving turn would rescue a
        # craving-only reply to a current finance question.
        overlap_count = len(request_prefixes & (unit_prefixes | evidence_prefixes))
        _direct_overlap = bool(
            request_prefixes and (request_prefixes & (unit_prefixes | evidence_prefixes))
        )
        # Stale-context guard uses the raw live turn (not the unioned
        # resolved text) so elliptical follow-ups stay rescuable while
        # explicit pivots do not inherit history.
        _current_allows_context_rescue = _is_context_dependent_followup(user_message)
        if not _direct_overlap and context_prefixes and _current_allows_context_rescue:
            # Contextual follow-up resolution (generic, turn-independent):
            # a terse follow-up in an ongoing conversation is relevant
            # when the evidence-backed unit shares content with the
            # immediately preceding user turns. The current-turn check
            # above stays primary; context only rescues continuity.
            _direct_overlap = bool(context_prefixes & (unit_prefixes | evidence_prefixes))
            if _direct_overlap:
                overlap_count = max(overlap_count, 1)
        if overlap_count >= 2:
            pass
        elif overlap_count == 1 and (request_steps or _direct_overlap):
            # Single content overlap proves topical relevance once the
            # numbered-step referent already aligns above or when a direct
            # lexical tie exists; a wrong step fails on step alignment
            # even when generic vocabulary such as decisions overlaps.
            # Paraphrases without an explicit step number also reach here
            # when the request names a step.
            pass
        else:
            # Same stale-context guard for the domain fallback: context
            # domain supplements only a generic/terse current turn. An
            # explicit current pivot must itself carry recovery-domain
            # vocabulary to match recovery evidence.
            _request_domain = bool(request_domain)
            if not _request_domain and _current_allows_context_rescue and bool(context_text):
                _request_domain = _has_recovery_domain(context_text)
            _evidence_domain = any(
                _has_recovery_domain(passage_text) for passage_text in cited_texts
            ) or _has_recovery_domain(str(unit.get("text", "") or reply))
            if not (_request_domain and _evidence_domain):
                continue
        # Actionable usefulness: the unit must carry declarative substance
        # beyond sympathy or a bare offer, tied to the cited evidence above.
        if len(unit_tokens) < 2 and len(unit_text) < 40:
            continue
        relevant_unit_found = True
        break
    if not relevant_unit_found:
        # Distinguish an unrelated citation from a plain all-glue reply:
        # both fail adequacy, but the category attributes the mechanism.
        return AdequacyAssessment(
            substantive_request=True,
            technically_grounded=True,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_IRRELEVANT_CITATION,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    return AdequacyAssessment(
        substantive_request=True,
        technically_grounded=True,
        answers_request=True,
        verdict=ADEQUACY_PASS,
        failure_category=FAILURE_OK_GLUE,
        verified_book_units=verified_count,
        evidence_passages=len(pack),
    )


def check_substantive_delivery_invariant(snapshot: dict[str, Any]) -> tuple[bool, str]:
    """Check the hard delivery invariant on a privacy-safe snapshot.

    Returns ``(ok, failure_category)``. A substantive turn (positive
    query count, retrieved passages, or a substantive planner reason)
    succeeds only when book evidence was retrieved, at least one relevant
    supported book unit is present, and the whole-turn adequacy verdict
    passes. Split statuses (technically grounded, answers request,
    delivered, qualified) are evaluated separately by callers; only all
    together yield a successful substantive answer.
    """
    try:
        query_count = int(snapshot.get("planner_query_count", 0) or 0)
    except (TypeError, ValueError):
        query_count = 0
    try:
        passages = int(snapshot.get("retrieval_passages", 0) or 0)
    except (TypeError, ValueError):
        passages = 0
    try:
        verified = int(snapshot.get("verified_book_units", 0) or 0)
    except (TypeError, ValueError):
        verified = 0
    reason = str(snapshot.get("planner_reason", "") or "").strip()
    adequacy = str(snapshot.get("adequacy_verdict", "") or "").strip()
    failure = str(snapshot.get("failure_category", "") or "").strip()
    answer_outcome = str(snapshot.get("answer_outcome", "") or "").strip()
    outcome_hint = (
        answer_outcome
        in (
            "served",
            "narrowed-supported",
            "narrowed-compacted",
        )
        and reason != PLANNER_REASON_LEGITIMATE_GLUE
    )
    substantive_hint = (
        query_count > 0
        or passages > 0
        or reason == PLANNER_REASON_SUBSTANTIVE_WITH_QUERIES
        or outcome_hint
    )
    if not substantive_hint:
        return True, ""
    if passages <= 0:
        return False, FAILURE_NO_EVIDENCE_SUBSTANTIVE
    if verified <= 0:
        return False, FAILURE_ALL_GLUE_FOR_SUBSTANTIVE
    # When the production adequacy gate ran, its verdict is authoritative:
    # identifiers alone never prove relevance.
    if adequacy and adequacy != ADEQUACY_PASS:
        return False, failure or FAILURE_IRRELEVANT_CITATION
    answers = snapshot.get("answers_request", None)
    if answers is False:
        return False, failure or FAILURE_IRRELEVANT_CITATION
    return True, ""


def snapshot_statuses(snapshot: dict[str, Any], *, delivered: bool) -> dict[str, bool]:
    """Split the four regression-prevention statuses for one turn.

    Only ``technically grounded`` + ``answers request`` + ``delivered`` +
    ``qualified`` together yield a successful substantive answer. The
    scheduler must track exact-main live product acceptance separately
    and never close a P0 on a merged commit alone.
    """
    try:
        verified = int(snapshot.get("verified_book_units", 0) or 0)
    except (TypeError, ValueError):
        verified = 0
    technically_grounded = verified > 0 and not bool(snapshot.get("turn_budget_exceeded", False))
    try:
        unavailable = int(snapshot.get("verifier_unavailable_units", 0) or 0)
    except (TypeError, ValueError):
        unavailable = 1
    if unavailable != 0:
        technically_grounded = False
    answers_request = bool(snapshot.get("answers_request", technically_grounded))
    if str(snapshot.get("adequacy_verdict", "") or "").strip() == ADEQUACY_FAIL:
        answers_request = False
    qualified = bool(technically_grounded and answers_request and delivered)
    return {
        "technically_grounded": bool(technically_grounded),
        "answers_request": bool(answers_request),
        "delivered": bool(delivered),
        "qualified": bool(qualified),
    }


def trace_id_for_snapshot(snapshot: dict[str, Any] | None) -> str:
    """Return the existing trace id or a fresh privacy-safe one."""
    if isinstance(snapshot, dict):
        existing = snapshot.get("turn_trace_id")
        if isinstance(existing, str) and len(existing.strip()) >= 8:
            return existing.strip()[:32]
    return new_turn_trace_id()


def digest_text(value: str) -> str:
    """Return a privacy-safe digest prefix for correlation without text."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ADEQUACY_FAIL",
    "ADEQUACY_PASS",
    "FAILURE_ALL_GLUE_FOR_SUBSTANTIVE",
    "FAILURE_BUDGET_EXCEEDED",
    "FAILURE_IRRELEVANT_CITATION",
    "FAILURE_NO_EVIDENCE_SUBSTANTIVE",
    "FAILURE_OK_GLUE",
    "FAILURE_REPAIR_FAILED",
    "FAILURE_UNAVAILABLE_VERIFIER",
    "FAILURE_UNSUPPORTED_CLAIM",
    "PLANNER_REASON_INVALID",
    "PLANNER_REASON_LEGITIMATE_GLUE",
    "PLANNER_REASON_PROVIDER_ERROR",
    "PLANNER_REASON_SUBSTANTIVE_WITH_QUERIES",
    "PLANNER_REASON_TIMEOUT",
    "PLANNER_REASON_UNKNOWN",
    "RECOVERY_MAX_QUERIES",
    "AdequacyAssessment",
    "assess_turn_adequacy",
    "build_recovery_queries",
    "check_substantive_delivery_invariant",
    "digest_text",
    "extract_step_numbers",
    "is_meta_request",
    "is_proven_glue_message",
    "new_turn_trace_id",
    "planner_reason_for",
    "resolve_effective_request",
    "runtime_sha",
    "snapshot_statuses",
    "trace_id_for_snapshot",
]
