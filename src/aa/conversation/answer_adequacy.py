"""Whole-turn answer adequacy gate for the v2 conversation path (kodmial/aa#268).

Model-driven architecture: every ordinary turn enters the same hidden
planner, which returns ``mode`` + ``resolved_intent`` + ``queries``.
Application code validates schema/cardinality only and never infers
meaning from words, punctuation, step numbers, greeting lists, recovery
stems, first-person markers or curated intents.

The final product boundary asks two independent semantic questions of
the same structured verifier invocation:

1. Groundedness: are all substantive claims supported by exact current
   Evidence Pack passages?
2. Answer relevance: does the final answer address the context-resolved
   intent?

Deterministic code remains only for genuinely syntactic/mechanical
contracts: command parsing, JSON/schema validation, exact source
IDs/checksums, output-size/privacy guards, transport parsing, and
safety rules where deterministic policy is explicitly required.
"""

from __future__ import annotations

import hashlib
import logging
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


def planner_reason_for(query_count: int, raw_outcome: str) -> str:
    """Map a stored planner outcome to an explicit causal reason.

    Structural mapping only; never infers conversational meaning from
    text. Legacy outcome tokens are preserved for backward
    compatibility; error tokens always map to error reasons.
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
        return PLANNER_REASON_LEGITIMATE_GLUE
    return PLANNER_REASON_UNKNOWN


def is_conversational_plan(*, mode: str, query_count: int) -> bool:
    """Whether a planner decision is conversational glue (schema only).

    True only when the model returned ``mode == "conversational"`` with
    zero queries. No text is inspected.
    """
    try:
        count = int(query_count)
    except (TypeError, ValueError):
        count = 0
    return str(mode or "").strip() == "conversational" and count == 0


def effective_request(*, resolved_intent: str, user_message: str) -> str:
    """Return the model-resolved intent, falling back to the raw turn.

    No regex, stemming or follow-up heuristics are applied; the planner
    already resolved pronouns, ellipsis and topic shifts.
    """
    intent = " ".join(str(resolved_intent or "").split()).strip()
    if intent:
        return intent
    return " ".join(str(user_message or "").split()).strip()


def build_recovery_queries(
    user_message: str,
    *,
    summary: str = "",
    recent: Sequence[str] | None = None,
    max_queries: int = RECOVERY_MAX_QUERIES,
) -> list[str]:
    """Build a generic semantic retrieval fallback without interpreting meaning.

    The raw current turn plus bounded recent conversation supply retrieval
    inputs. No domain vocabulary, greeting lists, step parsing or
    follow-up regexes are consulted.
    """
    cleaned = " ".join((user_message or "").split()).strip()
    if not cleaned:
        return []
    queries: list[str] = [cleaned]
    summary_cleaned = " ".join((summary or "").split()).strip()
    if summary_cleaned:
        candidate = " ".join(f"{cleaned} {summary_cleaned[:240]}".split()).strip()
        if candidate and candidate.casefold() not in {item.casefold() for item in queries}:
            queries.append(candidate)
    for item in list(recent or [])[:4]:
        text = " ".join(str(item).split()).strip()
        if not text or len(text) < 2:
            continue
        candidate = " ".join(f"{cleaned} {text[:200]}".split()).strip()
        if candidate.casefold() in {entry.casefold() for entry in queries}:
            continue
        queries.append(candidate)
        if len(queries) >= max_queries:
            break
    return queries[: max(1, max_queries)]


def build_generic_fallback_queries(
    user_message: str,
    *,
    summary: str = "",
    recent: Sequence[str] | None = None,
    max_queries: int = RECOVERY_MAX_QUERIES,
) -> list[str]:
    """Alias for the generic retrieval fallback."""
    return build_recovery_queries(
        user_message, summary=summary, recent=recent, max_queries=max_queries
    )


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


def _turn_is_relevant(grounding_result: dict[str, Any] | None) -> bool:
    """Whether the verifier judged the turn relevant to the resolved intent.

    Deterministic aggregation of per-unit model verdicts from the same
    verifier invocation: explicit ``answer_relevant`` wins when present,
    otherwise at least one supported book unit with ``addresses_intent``
    proves relevance, and a turn with no book requirement is relevant
    by construction. No keyword, step-number or token-overlap heuristics.
    Fail closed until the unified relevance field is present: a supported
    book unit without an explicit ``addresses_intent`` verdict never
    proves relevance.
    """
    if not isinstance(grounding_result, dict):
        return False
    explicit = grounding_result.get("answer_relevant", None)
    if isinstance(explicit, bool):
        return explicit
    units = grounding_result.get("units", [])
    if not isinstance(units, list) or not units:
        return True
    needs_book = any(isinstance(item, dict) and item.get("scope") == "book" for item in units)
    if not needs_book:
        return True
    for item in units:
        if not isinstance(item, dict):
            continue
        if (
            item.get("scope") == "book"
            and item.get("supported") is True
            and bool(item.get("evidence_passage_ids"))
        ):
            if item.get("addresses_intent") is True:
                return True
    return False


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
    planner_mode: str = "",
    resolved_intent: str = "",
    summary: str = "",
    recent: Sequence[str] | None = None,
    resolved_request: str = "",
    prior_user_messages: Sequence[str] = (),
    planner_query_count: int | None = None,
) -> AdequacyAssessment:
    """Judge the whole turn from model verdicts only.

    ``planner_mode`` decides substantive vs conversational (model-driven,
    never text heuristics). Groundedness and answer relevance come from
    the unified structured verifier verdicts for the same Evidence Pack.
    ``summary``/``recent``/``resolved_request``/``prior_user_messages``
    are accepted for backward compatibility and ignored: context was
    already resolved by the planner into ``resolved_intent``.
    """
    _ = (summary, recent, resolved_request, prior_user_messages)
    _ = effective_request(resolved_intent=resolved_intent, user_message=user_message)
    cleaned_reply = (reply or "").strip()
    pack = [item for item in (evidence_pack or []) if isinstance(item, dict)]
    supported_units = _supported_book_units(grounding_result)
    verified_count = len(supported_units)
    try:
        stored_query_count: int | None = (
            int(planner_query_count) if planner_query_count is not None else None
        )
    except (TypeError, ValueError):
        stored_query_count = 0
    if stored_query_count is None:
        # No stored planner count supplied (backward-compatible direct
        # calls): fall back to zero so a true zero-query conversational
        # turn with an empty pack still reads as glue, while retrieval
        # stays substantive regardless. Callers with a stale non-empty
        # pack must supply the stored planner count explicitly; pack
        # length is evidence passages, never planner queries.
        stored_query_count = 0
    substantive = not is_conversational_plan(
        mode=planner_mode
        if planner_mode
        else (
            PLANNER_REASON_LEGITIMATE_GLUE
            if planner_reason == PLANNER_REASON_LEGITIMATE_GLUE
            else "retrieval"
        ),
        query_count=stored_query_count,
    ) or bool(pack)
    # When the caller supplies only a legacy planner reason without an
    # explicit mode, treat legitimate-glue as conversational and every
    # other reason as substantive. Provider errors therefore never count
    # as glue. A stale non-empty evidence pack with zero stored queries
    # still fails closed to substantive: true glue never retrieves.
    if not planner_mode:
        substantive = (
            planner_reason != PLANNER_REASON_LEGITIMATE_GLUE
            or stored_query_count != 0
            or bool(pack)
        )
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
        if grounding_result is not None:
            try:
                all_supported = bool(grounding_result.get("all_required_supported", True))
            except Exception:
                all_supported = True
            if not all_supported and verified_count == 0:
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
    # Groundedness: every required unit must be supported.
    try:
        all_supported = bool(
            grounding_result.get("all_required_supported", False)
            if isinstance(grounding_result, dict)
            else False
        )
    except Exception:
        all_supported = False
    if not substantive:
        return AdequacyAssessment(
            substantive_request=False,
            technically_grounded=True,
            answers_request=True,
            verdict=ADEQUACY_PASS,
            failure_category="",
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    if not all_supported:
        return AdequacyAssessment(
            substantive_request=substantive,
            technically_grounded=False,
            answers_request=False,
            verdict=ADEQUACY_FAIL,
            failure_category=FAILURE_UNSUPPORTED_CLAIM,
            verified_book_units=verified_count,
            evidence_passages=len(pack),
        )
    # Answer relevance: the same verifier invocation judged whether the
    # supported material addresses the resolved intent. An irrelevant but
    # perfectly grounded answer fails here.
    if not _turn_is_relevant(grounding_result):
        return AdequacyAssessment(
            substantive_request=substantive,
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
    succeeds only when book evidence was retrieved, at least one
    supported book unit is present, and the whole-turn adequacy verdict
    passes. A generic fallback never counts as success.
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
    if adequacy and adequacy != ADEQUACY_PASS:
        return False, failure or FAILURE_IRRELEVANT_CITATION
    answers = snapshot.get("answers_request", None)
    if answers is False:
        return False, failure or FAILURE_IRRELEVANT_CITATION
    return True, ""


def snapshot_statuses(snapshot: dict[str, Any], *, delivered: bool) -> dict[str, bool]:
    """Split the four regression-prevention statuses for one turn.

    Only ``technically grounded`` + ``answers request`` + ``delivered`` +
    ``qualified`` together yield a successful substantive answer.
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
    "build_generic_fallback_queries",
    "build_recovery_queries",
    "check_substantive_delivery_invariant",
    "digest_text",
    "effective_request",
    "is_conversational_plan",
    "new_turn_trace_id",
    "planner_reason_for",
    "runtime_sha",
    "snapshot_statuses",
    "trace_id_for_snapshot",
]
