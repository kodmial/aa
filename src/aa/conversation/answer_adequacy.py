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
    if not any(marker in lowered for marker in _GENERIC_META_MARKERS):
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
    if remainder_tokens:
        return False
    return True


# Generic alcohol-recovery domain stems for topical relevance (never an
# exact-question list): lets colloquial, slang and typo paraphrases of the
# same recovery topic count as relevant while a verified but unrelated
# book fact (for example finance) still fails. Only the domain is
# recognized here; the exact request is never matched.
_RECOVERY_DOMAIN_STEMS = (
    "тяг",
    "выпи",
    "выпь",
    "буха",
    "бухл",
    "пить",
    "пьян",
    "трезв",
    "срыв",
    "запо",
    "алког",
    "похмел",
)


def _has_recovery_domain(text: str) -> bool:
    """Whether text contains generic alcohol-recovery domain vocabulary."""
    lowered = (text or "").casefold()
    if not lowered:
        return False
    return any(stem in lowered for stem in _RECOVERY_DOMAIN_STEMS)


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


def is_proven_glue_message(user_message: str, *, summary: str = "", recent_count: int = 0) -> bool:
    """Whether a turn is positively proven contentless conversational glue.

    Conservative fail-closed default: every uncertain turn is treated as
    potentially substantive, never as proven glue. Only a very short
    greeting/thanks-style message with no question mark, no generic
    interrogative/request structure and no substantive continuation
    counts as proven glue. A combined greeting plus request for recovery
    steps is therefore never proven glue here.

    Generic structural signal only: no exact user question is matched and
    no recovery-topic keyword list decides the outcome.
    """
    _ = (summary, recent_count)
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
    return True


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
) -> AdequacyAssessment:
    """Judge the whole turn: task, context, candidate and relevant evidence.

    Invariant for any substantive personal recovery/support request: at
    least one practical, relevant, book-supported explanation/action
    answering that request must be present and verified. Validated source
    identifiers alone never prove topical relevance: the supported unit
    must share substantive content with both the cited exact passages and
    the user's request. Pure greetings, honest self-identity and genuinely
    contentless turns keep a passing glue verdict without book evidence.
    """
    _ = planner_reason
    cleaned_reply = (reply or "").strip()
    pack = [item for item in (evidence_pack or []) if isinstance(item, dict)]
    supported_units = _supported_book_units(grounding_result)
    verified_count = len(supported_units)
    substantive = not is_proven_glue_message(user_message)
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
    request_tokens = _content_tokens(user_message)
    request_prefixes = _token_prefixes(request_tokens)
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
        # Topical relevance to this request: the evidence-backed unit must
        # share substantive content with the request, or both sides must
        # carry generic alcohol-recovery domain vocabulary so colloquial,
        # slang and typo paraphrases of the same topic still count while a
        # verified but unrelated book fact still fails. Without this,
        # identifiers alone would prove relevance.
        _direct_overlap = bool(
            request_prefixes and (request_prefixes & (unit_prefixes | evidence_prefixes))
        )
        if not _direct_overlap:
            _request_domain = _has_recovery_domain(user_message)
            _evidence_domain = any(
                _has_recovery_domain(passage_text) for passage_text in cited_texts
            ) or _has_recovery_domain(str(unit.get("text", "") or reply))
            if not (_request_domain and _evidence_domain):
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
    "is_meta_request",
    "is_proven_glue_message",
    "new_turn_trace_id",
    "planner_reason_for",
    "runtime_sha",
    "snapshot_statuses",
    "trace_id_for_snapshot",
]
