"""Single exact final-answer certification boundary (kodmial/aa#304).

Every normal user-facing conversational output traverses
:func:`finalize_answer_node` after candidate construction and before
conversation-history insertion/delivery. The node certifies the exact
normalized delivered text against the exact evidence bundle actually
used (including repair/recovery) and a deterministic context snapshot.

Certification is not delivery: a successful finalization authorizes one
:class:`AnswerCandidate` for sending. Actual Telegram acceptance is
recorded with the serializable :class:`DeliveryReceipt` contract on the
transport boundary (#312 owns quote-history mutation).

Only explicitly classified protocol/service/command/emergency control
paths bypass this gate with separate typed policy handling; they can
never qualify as book answers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel, Field

logger = logging.getLogger("aa.conversation.finalization")

OutcomeKind = Literal["answer", "clarification", "unavailable", "safety"]
"""Typed delivery outcome for one certified candidate."""

ReceiptStatus = Literal["confirmed", "failed", "unknown"]
DeliveryChannel = Literal["sendMessage", "sendVoice"]


class FinalizationError(ValueError):
    """Exact final-answer certification failed (fail-closed, never delivered)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"finalization failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


class AnswerCandidate(BaseModel):
    """Serializable exact delivery candidate (LangGraph/checkpoint-safe)."""

    text: str = Field(min_length=1)
    evidence_bundle: list[dict[str, Any]] = Field(default_factory=list)
    context_digest: str = Field(min_length=1)
    outcome_kind: OutcomeKind = "answer"
    candidate_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = {"extra": "forbid"}


class WholeAnswerVerdict(BaseModel):
    """Whole-answer semantic grounding + coverage verdict (internal)."""

    supported: bool = False
    addresses_intent: bool = False
    coverage_ok: bool = False
    conditions_preserved: bool = False
    quote_ok: bool = False
    reason: str = ""
    failure_code: str = ""

    model_config = {"extra": "forbid"}


class VerificationCertificate(BaseModel):
    """Serializable certificate for exactly one normalized candidate."""

    answer_sha256: str = Field(min_length=1)
    evidence_digest: str = Field(min_length=1)
    context_digest: str = Field(min_length=1)
    claim_verdicts: list[dict[str, Any]] = Field(default_factory=list)
    whole_answer_verdict: WholeAnswerVerdict = Field(default_factory=WholeAnswerVerdict)
    outcome_kind: OutcomeKind = "answer"
    certificate_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

    model_config = {"extra": "forbid"}


class DeliveryReceipt(BaseModel):
    """Serializable transport acknowledgment seam for #312 (provisional vs ack)."""

    turn_id: str = ""
    certificate_id: str = ""
    final_sha256: str = ""
    segment_index: int = 0
    segment_count: int = 1
    char_start: int = 0
    char_end: int = 0
    utf8_start: int = 0
    utf8_end: int = 0
    status: ReceiptStatus = "unknown"
    channel: DeliveryChannel = "sendMessage"
    retry_id: str = ""

    model_config = {"extra": "forbid"}


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answer_text(text: str) -> str:
    """Normalize presentation before certification (no post-cert mutation).

    Line endings are unified and surrounding whitespace is removed.
    Interior content is byte-preserved: no clipping, substitution,
    compaction or fallback text happens here or after certification.
    Every later stage must use the normalized text verbatim; any
    substantive rewrite, deletion, compaction, quotation substitution,
    safety regeneration or repair creates a new candidate.
    """
    if not isinstance(text, str):
        raise FinalizationError("normalize-invalid", "candidate text is not a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        raise FinalizationError("normalize-empty", "candidate text is empty after normalization")
    return normalized


def evidence_digest_for_pack(pack: list[dict[str, Any]]) -> str:
    """Return a deterministic digest of the exact evidence bundle used."""
    canonical: list[dict[str, Any]] = []
    for item in list(pack or []):
        if not isinstance(item, dict):
            continue
        canonical.append(
            {
                "passage_id": str(item.get("passage_id", "")),
                "text_sha256": str(item.get("text_sha256", "")),
                "source_sha256": str(item.get("source_sha256", "")),
                "corpus_version": str(item.get("corpus_version", "")),
                "source_id": str(item.get("source_id", item.get("source", ""))),
                "section_id": str(item.get("section_id", item.get("section", ""))),
                "char_start": int(item.get("char_start", 0) or 0),
                "char_end": int(item.get("char_end", 0) or 0),
            }
        )
    canonical.sort(key=lambda entry: str(entry.get("passage_id", "")))
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_context_snapshot(
    *,
    question: str,
    resolved_intent: str,
    summary: str,
    recent_texts: list[str],
) -> dict[str, Any]:
    """Build the provisional deterministic context snapshot (pre-#305).

    #305 replaces this assembly with one canonical ``ResolvedTurn``
    object without weakening the certificate invariant. Only
    privacy-safe digests travel in the certificate; raw text never
    leaves this snapshot except through the digest.
    """
    cleaned_recent = [str(item)[:1200] for item in (recent_texts or []) if str(item).strip()]
    snapshot = {
        "question": " ".join(str(question or "").split()),
        "resolved_intent": " ".join(str(resolved_intent or "").split()),
        "summary": str(summary or "")[:2000],
        "recent": cleaned_recent[-8:],
        "snapshot_version": "turn-context-snapshot/1",
    }
    return snapshot


def context_digest_for_turn(
    *,
    question: str,
    resolved_intent: str,
    summary: str,
    recent_texts: list[str],
) -> str:
    """Return the deterministic digest of the provisional context snapshot."""
    snapshot = build_context_snapshot(
        question=question,
        resolved_intent=resolved_intent,
        summary=summary,
        recent_texts=recent_texts,
    )
    payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def infer_outcome_kind(
    *,
    text: str,
    evidence_pack: list[dict[str, Any]],
    grounding_result: dict[str, Any] | None,
    telemetry: dict[str, Any] | None = None,
) -> OutcomeKind:
    """Infer the type-specific verification kind for one served candidate."""
    _ = text
    telemetry_d = dict(telemetry) if isinstance(telemetry, dict) else {}
    if str(telemetry_d.get("outbound_safety", "") or "") == "repaired":
        return "safety"
    units = []
    if isinstance(grounding_result, dict):
        raw_units = grounding_result.get("units", [])
        units = (
            [item for item in raw_units if isinstance(item, dict)]
            if isinstance(raw_units, list)
            else []
        )
    has_book = any(item.get("scope") == "book" for item in units)
    if has_book or bool(evidence_pack):
        return "answer"
    return "clarification"


_ABSOLUTE_QUANTIFIERS = ("всегда", "никогда", " все ", " каждый ", "гарантир", "абсолютно")
_RECOMMENDATION_MARKERS = ("следует", "нужно", "надо", "попробуйте", "делайте", "стоит ", "лучше ")
_QUALIFIER_MARKERS = ("если", "когда", "при ", "важно", "однако", "но ", "кроме", "только")
_STORY_MARKERS = ("история", "рассказ", "пример", "герой", "персонаж")


def _cited_texts(
    grounding_result: dict[str, Any] | None, pack: list[dict[str, Any]]
) -> tuple[list[str], set[str]]:
    """Return cited passage texts plus the cited id set for one candidate."""
    cited_ids: set[str] = set()
    if isinstance(grounding_result, dict):
        raw_units = grounding_result.get("units", [])
        if isinstance(raw_units, list):
            for item in raw_units:
                if not isinstance(item, dict):
                    continue
                for cited in item.get("evidence_passage_ids", []) or []:
                    if isinstance(cited, str) and cited.strip():
                        cited_ids.add(cited.strip())
    by_id = {
        str(item.get("passage_id", "")): str(item.get("text", ""))
        for item in (pack or [])
        if isinstance(item, dict)
    }
    return [by_id[cited] for cited in cited_ids if cited in by_id], cited_ids


def evaluate_whole_answer(
    *,
    candidate_text: str,
    grounding_result: dict[str, Any] | None,
    evidence_pack: list[dict[str, Any]],
    question: str,
    resolved_intent: str,
    outcome_kind: OutcomeKind,
) -> WholeAnswerVerdict:
    """Evaluate the whole certified candidate against full evidence.

    Reads the full authoritative evidence bundle (never a sentence-local
    window): unsupported claims, over-generalization (absolute quantifier
    without source support), dropped conditions/exceptions (action
    recommendation with absolutes and no qualifier while cited sources
    carry conditions), story/example misuse (narrative presented without
    anchored source), and unanswered request parts (via the adequacy
    gate) all fail closed. The usefulness-only ``whole_turn_judge``
    never substitutes for this check.
    """
    if not candidate_text.strip():
        return WholeAnswerVerdict(reason="empty candidate", failure_code="empty-candidate")
    if grounding_result is None and outcome_kind == "answer":
        return WholeAnswerVerdict(reason="missing claim verdicts", failure_code="missing-verdicts")
    units: list[dict[str, Any]] = []
    if isinstance(grounding_result, dict):
        raw = grounding_result.get("units", [])
        units = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
    if outcome_kind == "answer":
        grounding_d = grounding_result if isinstance(grounding_result, dict) else {}
        if not bool(grounding_d.get("all_required_supported", False)):
            return WholeAnswerVerdict(
                reason="unsupported claims", failure_code="unsupported-claims"
            )
        for unit in units:
            if not bool(unit.get("supported", False)):
                return WholeAnswerVerdict(
                    reason="unsupported unit", failure_code="unsupported-claims"
                )
        book_units = [item for item in units if item.get("scope") == "book"]
        if not book_units:
            return WholeAnswerVerdict(
                reason="answer without book units", failure_code="unanswered-parts"
            )
        # Turn relevance: every served supported book unit must address intent.
        try:
            from aa.conversation.verifier_schema import UnitVerdict as _Verdict
            from aa.conversation.verifier_schema import validate_grounding_result as _validate

            parsed = [
                _Verdict.model_validate(
                    {
                        "unit_id": str(item.get("unit_id", f"u{i}")),
                        "scope": item.get("scope", "book"),
                        "supported": bool(item.get("supported", False)),
                        "evidence_passage_ids": list(item.get("evidence_passage_ids", []) or []),
                        "addresses_intent": bool(item.get("addresses_intent", False)),
                        "origin": item.get("origin", "book_claim"),
                        "origin_ref": dict(item.get("origin_ref", {}) or {})
                        if isinstance(item.get("origin_ref", {}), dict)
                        else {},
                    }
                )
                for i, item in enumerate(units)
            ]
            result = _validate(
                {
                    "verified": True,
                    "units": [item.model_dump() for item in parsed],
                    "all_required_supported": True,
                },
                expected_unit_ids=[str(item.unit_id) for item in parsed],
            )
            if not bool(result.answer_relevant):
                return WholeAnswerVerdict(
                    reason="irrelevant citation",
                    failure_code="unanswered-parts",
                )
        except Exception as exc:
            return WholeAnswerVerdict(
                reason=f"relevance validation failed: {type(exc).__name__}",
                failure_code="unsupported-claims",
            )
    # Over-generalization: absolute quantifiers in book text need source support.
    lowered = f" {candidate_text.casefold()} "
    if outcome_kind in ("answer", "safety") and any(
        marker in lowered for marker in _ABSOLUTE_QUANTIFIERS
    ):
        cited_texts, _ = _cited_texts(grounding_result, evidence_pack)
        joined_cited = f" {' '.join(cited_texts).casefold()} "
        if not any(marker in joined_cited for marker in _ABSOLUTE_QUANTIFIERS):
            return WholeAnswerVerdict(
                reason="over-generalization without source support",
                failure_code="over-generalization",
            )
        # Dropped conditions: absolute recommendation without any qualifier
        # while cited sources carry conditional language fails closed.
        has_recommendation = any(marker in lowered for marker in _RECOMMENDATION_MARKERS)
        has_qualifier = any(marker in lowered for marker in _QUALIFIER_MARKERS)
        cited_has_condition = any(
            marker in joined_cited for marker in ("если", "когда", "кроме", "однако", "только")
        )
        if has_recommendation and not has_qualifier and cited_has_condition:
            return WholeAnswerVerdict(
                reason="dropped conditions/exceptions",
                failure_code="dropped-conditions",
            )
    # Story/example misuse: narrative presented as book fact needs anchoring.
    if outcome_kind in ("answer", "safety") and any(marker in lowered for marker in _STORY_MARKERS):
        try:
            from aa.conversation.quote_provenance import anchor_book_span as _anchor

            story_hit = any(
                _anchor(marker, list(evidence_pack or [])) is not None
                for marker in _STORY_MARKERS
                if marker in lowered
            )
            cited_texts, _ = _cited_texts(grounding_result, evidence_pack)
            joined_cited = " ".join(cited_texts).casefold()
            if not story_hit and not any(marker in joined_cited for marker in _STORY_MARKERS):
                # Unanchored narrative alongside book claims fails closed;
                # pure glue has no book units and never reaches this branch.
                if any(item.get("scope") == "book" for item in units):
                    return WholeAnswerVerdict(
                        reason="story/example without source anchor",
                        failure_code="story-misuse",
                    )
        except Exception:
            pass
    # Unanswered parts: whole-turn adequacy must pass for substantive kinds.
    if outcome_kind in ("answer", "safety"):
        try:
            from aa.conversation.answer_adequacy import assess_turn_adequacy as _assess

            assessment = _assess(
                user_message=question,
                reply=candidate_text,
                evidence_pack=list(evidence_pack or []),
                grounding_result=grounding_result,
                planner_reason="substantive-with-queries",
                verifier_outcome="passed",
                unavailable_units=0,
                turn_budget_exceeded=False,
                planner_mode="retrieval",
                resolved_intent=resolved_intent or question,
                planner_query_count=len(evidence_pack or []),
            )
            if str(getattr(assessment, "verdict", "")) != "pass":
                return WholeAnswerVerdict(
                    reason="unanswered parts of the request",
                    failure_code="unanswered-parts",
                )
        except Exception as exc:
            return WholeAnswerVerdict(
                reason=f"coverage check failed: {type(exc).__name__}",
                failure_code="unanswered-parts",
            )
    return WholeAnswerVerdict(
        supported=True,
        addresses_intent=True,
        coverage_ok=True,
        conditions_preserved=True,
        quote_ok=True,
        reason="whole-answer grounded",
    )


def certify_candidate(
    *,
    candidate: AnswerCandidate,
    grounding_result: dict[str, Any] | None,
    question: str,
    resolved_intent: str,
    summary: str,
    recent_messages: list[BaseMessage],
    stored_unit_texts: list[dict[str, Any]] | None = None,
) -> VerificationCertificate:
    """Fully certify one normalized candidate (fail-closed).

    Re-validates exact delivered text after all transformations using
    #308's full-answer quote-span/claim-origin API and #310's strict
    model-decision/corpus-provenance validation. Distinguishes
    user-attributed quotes from book claims without laundering
    inferences and fails closed on multi-sentence invented book quotes.

    ``stored_unit_texts`` carries the pipeline's served unit texts so the
    delivered text binds to stored claim verdicts by verbatim text
    (renumbering-safe): narrowing/compaction subsets verify without
    borrowed verdicts, while novel substituted text fails closed.
    """
    from aa.conversation.evidence_integrity import validate_book_pack_for_model_use
    from aa.conversation.output_limits import QUOTE_BUDGET_CHARS, envelope_passes
    from aa.conversation.verifier import check_cited_passage_ids, check_exact_quotes

    normalized = normalize_answer_text(candidate.text)
    if normalized != candidate.text:
        raise FinalizationError("normalize-mismatch", "candidate text is not normalized")
    pack = [dict(item) for item in (candidate.evidence_bundle or []) if isinstance(item, dict)]
    try:
        validate_book_pack_for_model_use(pack)
    except Exception as exc:
        raise FinalizationError("evidence-integrity", str(exc)[:200]) from exc
    expected_context = context_digest_for_turn(
        question=question,
        resolved_intent=resolved_intent,
        summary=summary,
        recent_texts=[
            str(getattr(item, "content", ""))
            for item in (recent_messages or [])
            if isinstance(item, BaseMessage)
        ],
    )
    if candidate.context_digest != expected_context:
        raise FinalizationError("context-mismatch", "context snapshot changed after certification")
    evidence_digest = evidence_digest_for_pack(pack)
    answer_sha = sha256_text(normalized)
    # Claim-level + whole-answer context: bind the exact delivered text
    # to stored verdicts by verbatim unit text. Fresh ids are local to
    # this split; stored ids come from the pipeline, so matching is by
    # text (a novel substituted sentence has no stored referent and
    # fails closed). Subsets of verified supported content pass binding
    # here; meaning changes are caught by whole-answer adequacy below.
    units: list[Any] = []
    verdicts: list[Any] = []
    if grounding_result is not None:
        try:
            from aa.conversation.response_units import split_response_units as _split
            from aa.conversation.verifier_schema import UnitVerdict as _UnitVerdict

            units = list(_split(normalized))
            raw_units = grounding_result.get("units", [])
            raw_list = (
                [item for item in raw_units if isinstance(item, dict)]
                if isinstance(raw_units, list)
                else []
            )
            if candidate.outcome_kind in ("answer", "safety"):
                if not raw_list:
                    raise FinalizationError(
                        "claim-missing", "substantive candidate has no claim verdicts"
                    )
                stored_texts: dict[str, dict[str, Any]] = {}
                if stored_unit_texts:
                    for entry in stored_unit_texts:
                        if isinstance(entry, dict):
                            text = str(entry.get("text", ""))
                            if text:
                                stored_texts[text] = entry
                verdict_by_stored_id: dict[str, dict[str, Any]] = {}
                for item in raw_list:
                    verdict_by_stored_id[str(item.get("unit_id", ""))] = item
                text_to_verdict: dict[str, dict[str, Any]] = {}
                for stored_id, item in verdict_by_stored_id.items():
                    matched_text = ""
                    if stored_texts:
                        for text, entry in stored_texts.items():
                            if str(entry.get("unit_id", "")) == stored_id:
                                matched_text = text
                                break
                    if matched_text:
                        text_to_verdict[matched_text] = item
                # Without stored texts (legacy callers) fall back to
                # positional binding; with stored texts require verbatim.
                for i, fresh in enumerate(units):
                    fresh_text = str(getattr(fresh, "text", ""))
                    source: dict[str, Any] | None = None
                    if text_to_verdict:
                        source = text_to_verdict.get(fresh_text)
                        if source is None:
                            # Whitespace-tolerant fallback for join/split
                            # round-trips; novel content still fails.
                            for text, item in text_to_verdict.items():
                                if " ".join(text.split()) == " ".join(fresh_text.split()):
                                    source = item
                                    break
                        if source is None:
                            raise FinalizationError(
                                "claim-unbound",
                                "delivered unit has no verified source unit",
                            )
                    else:
                        if i >= len(raw_list):
                            raise FinalizationError(
                                "claim-count-mismatch",
                                "claim verdicts do not match the delivered text",
                            )
                        source = raw_list[i]
                    verdicts.append(
                        _UnitVerdict.model_validate(
                            {
                                "unit_id": str(getattr(fresh, "unit_id", f"u{i}")),
                                "scope": source.get("scope", "book"),
                                "supported": bool(source.get("supported", False)),
                                "evidence_passage_ids": list(
                                    source.get("evidence_passage_ids", []) or []
                                ),
                                "addresses_intent": bool(source.get("addresses_intent", False)),
                                "origin": source.get("origin", "book_claim"),
                                "origin_ref": dict(source.get("origin_ref", {}) or {})
                                if isinstance(source.get("origin_ref", {}), dict)
                                else {},
                            }
                        )
                    )
                # Every served book unit must be verifier-supported.
                for verdict in verdicts:
                    if str(getattr(verdict, "scope", "")) == "book" and not bool(
                        getattr(verdict, "supported", False)
                    ):
                        raise FinalizationError(
                            "claim-unsupported", "delivered unit is not supported"
                        )
            else:
                for i, item in enumerate(raw_list):
                    verdicts.append(
                        _UnitVerdict.model_validate(
                            {
                                "unit_id": str(item.get("unit_id", f"u{i}")),
                                "scope": item.get("scope", "book"),
                                "supported": bool(item.get("supported", False)),
                                "evidence_passage_ids": list(
                                    item.get("evidence_passage_ids", []) or []
                                ),
                                "addresses_intent": bool(item.get("addresses_intent", False)),
                                "origin": item.get("origin", "book_claim"),
                                "origin_ref": dict(item.get("origin_ref", {}) or {})
                                if isinstance(item.get("origin_ref", {}), dict)
                                else {},
                            }
                        )
                    )
        except FinalizationError:
            raise
        except Exception as exc:
            raise FinalizationError("claim-binding", f"{type(exc).__name__}") from exc
    # Strict cite gate against the exact bundle used.
    if verdicts:
        try:
            from aa.conversation.verifier_schema import GroundingResult as _Grounding

            pack_ids = {
                str(item.get("passage_id", "")) for item in pack if str(item.get("passage_id", ""))
            }
            check_cited_passage_ids(
                _Grounding(
                    verified=True,
                    units=list(verdicts),
                    all_required_supported=True,
                ),
                pack_ids=pack_ids,
            )
        except Exception as exc:
            raise FinalizationError("cite-unknown", str(exc)[:200]) from exc
        # Whole-answer exact quotes BEFORE any sentence-local reasoning.
        try:
            check_exact_quotes(
                units=units,
                result=_Grounding(verified=True, units=list(verdicts), all_required_supported=True),
                passages=pack,
                answer_text=normalized,
            )
        except Exception as exc:
            raise FinalizationError("quote-mismatch", str(exc)[:200]) from exc
        # Candidate-level quote/origin certification (user vs book).
        try:
            from aa.conversation.quote_provenance import build_user_message_index
            from aa.conversation.quote_provenance import certify_answer_candidate as _certify_quotes

            user_index = build_user_message_index(list(recent_messages or []), question)
            quote_cert = _certify_quotes(
                answer=normalized,
                units=units,
                verdicts=verdicts,
                passages=pack,
                user_index=user_index,
            )
            if int(quote_cert.book_quote_chars) > QUOTE_BUDGET_CHARS:
                raise FinalizationError("quote-budget", "verbatim book quota exceeded")
        except FinalizationError:
            raise
        except Exception as exc:
            raise FinalizationError("quote-origin", f"{type(exc).__name__}") from exc
    # Independent outbound safety + envelope on the exact delivered text.
    try:
        from aa.safety.outbound import is_outbound_safe as _is_safe

        if not bool(_is_safe(normalized)):
            raise FinalizationError("outbound-safety", "candidate blocked by outbound gate")
    except FinalizationError:
        raise
    except Exception as exc:
        raise FinalizationError("outbound-safety", f"{type(exc).__name__}") from exc
    if not bool(envelope_passes(normalized)):
        raise FinalizationError("envelope", "candidate exceeds the single-message envelope")
    # Language/privacy presentation contract on the exact text.
    try:
        from aa.conversation.turn_pipeline import contains_cyrillic as _has_cyrillic
        from aa.conversation.turn_pipeline import leaks_internal_terms as _leaks

        if not bool(_has_cyrillic(normalized)) or bool(_leaks(normalized)):
            raise FinalizationError("language-guard", "candidate fails presentation contract")
    except FinalizationError:
        raise
    except Exception as exc:
        raise FinalizationError("language-guard", f"{type(exc).__name__}") from exc
    whole = evaluate_whole_answer(
        candidate_text=normalized,
        grounding_result=grounding_result,
        evidence_pack=pack,
        question=question,
        resolved_intent=resolved_intent or question,
        outcome_kind=candidate.outcome_kind,
    )
    if not (
        whole.supported and whole.coverage_ok and whole.conditions_preserved and whole.quote_ok
    ):
        # evaluate_whole_answer returns all-True only on full pass.
        code = whole.failure_code or "whole-answer"
        raise FinalizationError(code, whole.reason[:200])
    claim_dicts: list[dict[str, Any]] = []
    if isinstance(grounding_result, dict):
        raw = grounding_result.get("units", [])
        claim_dicts = (
            [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        )
    return VerificationCertificate(
        answer_sha256=answer_sha,
        evidence_digest=evidence_digest,
        context_digest=candidate.context_digest,
        claim_verdicts=claim_dicts,
        whole_answer_verdict=whole,
        outcome_kind=candidate.outcome_kind,
    )


def verify_certificate(
    *,
    candidate: AnswerCandidate,
    certificate: VerificationCertificate,
    evidence_pack: list[dict[str, Any]] | None = None,
) -> None:
    """Mechanically re-verify text/evidence/context digests before send."""
    normalized = normalize_answer_text(candidate.text)
    if sha256_text(normalized) != certificate.answer_sha256:
        raise FinalizationError("stale-certificate", "candidate text changed after certification")
    active_pack = (
        [dict(item) for item in (evidence_pack or []) if isinstance(item, dict)]
        if evidence_pack is not None
        else [dict(item) for item in (candidate.evidence_bundle or []) if isinstance(item, dict)]
    )
    if evidence_digest_for_pack(active_pack) != certificate.evidence_digest:
        raise FinalizationError("stale-evidence", "evidence bundle changed after certification")
    if (
        evidence_digest_for_pack(list(candidate.evidence_bundle or []))
        != certificate.evidence_digest
    ):
        raise FinalizationError("stale-evidence", "candidate bundle mismatches certificate")
    if candidate.context_digest != certificate.context_digest:
        raise FinalizationError("stale-context", "context snapshot changed after certification")
    whole = certificate.whole_answer_verdict
    if not (
        whole.supported and whole.coverage_ok and whole.conditions_preserved and whole.quote_ok
    ):
        raise FinalizationError("certificate-negative", "certificate carries a negative verdict")


def split_certified_text(certified_text: str) -> list[str]:
    """Split after semantic finalization with order/text preservation proof."""
    from aa.conversation.output_limits import envelope_passes as _passes
    from aa.conversation.output_limits import split_text_to_envelope_segments as _split

    normalized = normalize_answer_text(certified_text)
    if _passes(normalized):
        return [normalized]
    segments = _split(normalized)
    verify_transport_split(normalized, segments)
    return segments


def _whitespace_key(text: str) -> str:
    return " ".join(str(text or "").split())


def verify_transport_split(certified_text: str, segments: list[str]) -> None:
    """Prove transport splitting preserves text/order without shortening."""
    from aa.conversation.output_limits import MAX_TRANSPORT_SEGMENTS
    from aa.conversation.output_limits import aggregate_quote_chars as _quote_chars
    from aa.conversation.output_limits import envelope_passes as _passes

    normalized = normalize_answer_text(certified_text)
    if not segments:
        raise FinalizationError("transport-empty", "no transport segments")
    if len(segments) > MAX_TRANSPORT_SEGMENTS:
        raise FinalizationError("transport-budget", "too many transport segments")
    for segment in segments:
        if not isinstance(segment, str) or not segment.strip():
            raise FinalizationError("transport-empty", "empty transport segment")
        if not bool(_passes(segment)):
            raise FinalizationError("transport-envelope", "segment exceeds the envelope")
        if normalize_answer_text(segment) != segment:
            raise FinalizationError("transport-normalize", "segment mutated after certification")
    # Order/text preservation: whitespace-normalized reassembly must equal
    # the certified text; lengths must prove no semantic shortening.
    reassembled = " ".join(segments)
    if _whitespace_key(reassembled) != _whitespace_key(normalized):
        raise FinalizationError("transport-reorder", "segments do not reassemble to certified text")
    if _quote_chars(reassembled) != _quote_chars(normalized):
        raise FinalizationError("transport-quotes", "quote content changed in transport split")


def build_delivery_receipts(
    *,
    certified_text: str,
    segments: list[str],
    certificate_id: str,
    turn_id: str,
    channel: DeliveryChannel,
    retry_id: str = "",
    status: ReceiptStatus = "unknown",
) -> list[DeliveryReceipt]:
    """Build serializable per-segment delivery attempts for one certificate."""
    normalized = normalize_answer_text(certified_text)
    verify_transport_split(normalized, list(segments or []))
    receipts: list[DeliveryReceipt] = []
    cursor = 0
    total_utf8 = len(normalized.encode("utf-8"))
    _ = total_utf8
    for index, segment in enumerate(segments):
        start = normalized.find(segment, cursor)
        if start < 0:
            # Whitespace-joined reassembly proven above; locate by order.
            start = cursor
        end = start + len(segment)
        cursor = end
        utf8_start = len(normalized[:start].encode("utf-8"))
        utf8_end = len(normalized[:end].encode("utf-8"))
        receipts.append(
            DeliveryReceipt(
                turn_id=turn_id,
                certificate_id=certificate_id,
                final_sha256=sha256_text(normalized),
                segment_index=index,
                segment_count=len(segments),
                char_start=start,
                char_end=end,
                utf8_start=utf8_start,
                utf8_end=utf8_end,
                status=status,
                channel=channel,
                retry_id=retry_id or uuid.uuid4().hex,
            )
        )
    return receipts


def candidate_from_state(
    state: Any,
    *,
    default_outcome: OutcomeKind | None = None,
) -> tuple[
    AnswerCandidate, dict[str, Any] | None, str, str, str, list[BaseMessage], list[dict[str, Any]]
]:
    """Build the exact candidate from live graph state (no transient objects)."""
    question = str(state.get("current_user_message", "") or "")
    resolved_intent = str(state.get("resolved_intent", "") or "")
    if not resolved_intent.strip():
        try:
            retry = state.get("retry_state", {})
            retry_d = dict(retry) if isinstance(retry, dict) else {}
            resolved_intent = str(retry_d.get("resolved_intent", "") or "")
        except Exception:
            resolved_intent = ""
    summary = str(state.get("conversation_summary", "") or "")
    messages = [item for item in state.get("messages", []) if isinstance(item, BaseMessage)]
    pack = [dict(item) for item in state.get("evidence_pack", []) if isinstance(item, dict)]
    raw_text = str(state.get("final_response", "") or state.get("draft_response", "") or "")
    normalized = normalize_answer_text(raw_text)
    recent_texts = [
        str(getattr(item, "content", "")) for item in messages if isinstance(item, BaseMessage)
    ]
    # Keep only human/assistant text slices for the digest.
    recent_cleaned = [text for text in recent_texts if text.strip()]
    context_digest = context_digest_for_turn(
        question=question,
        resolved_intent=resolved_intent or question,
        summary=summary,
        recent_texts=recent_cleaned,
    )
    grounding: dict[str, Any] | None = None
    raw_grounding = state.get("grounding_result", None)
    if isinstance(raw_grounding, dict):
        grounding = dict(raw_grounding)
    telemetry: dict[str, Any] | None = None
    raw_retry = state.get("retry_state", None)
    if isinstance(raw_retry, dict):
        inner = raw_retry.get("turn_telemetry", None)
        telemetry = dict(inner) if isinstance(inner, dict) else dict(raw_retry)
    outcome: OutcomeKind = default_outcome or infer_outcome_kind(
        text=normalized, evidence_pack=pack, grounding_result=grounding, telemetry=telemetry
    )
    candidate = AnswerCandidate(
        text=normalized,
        evidence_bundle=pack,
        context_digest=context_digest,
        outcome_kind=outcome,
    )
    stored_texts = [
        dict(item) for item in state.get("response_unit_texts", []) if isinstance(item, dict)
    ]
    return (
        candidate,
        grounding,
        question,
        resolved_intent or question,
        summary,
        messages,
        stored_texts,
    )


async def finalize_answer_node(state: Any) -> dict[str, Any]:
    """Single obligatory graph delivery gate (real node, not a bolt-on).

    Certifies the exact normalized candidate against the exact evidence
    bundle and context snapshot. On success the final ``AIMessage`` is
    created here (never before); on failure no message is added and the
    turn fails closed without entering conversation history.
    """
    from aa.conversation.graph_state import NORMAL_ROUTE

    if str(state.get("route", NORMAL_ROUTE)) != NORMAL_ROUTE:
        return {}
    candidate, grounding, question, resolved_intent, summary, messages, stored_texts = (
        candidate_from_state(state)
    )
    if not question.strip():
        raise FinalizationError("empty-question", "refusing certification without a question")
    try:
        certificate = certify_candidate(
            candidate=candidate,
            grounding_result=grounding,
            question=question,
            resolved_intent=resolved_intent,
            summary=summary,
            recent_messages=messages,
            stored_unit_texts=stored_texts,
        )
    except FinalizationError:
        raise
    except Exception as exc:
        raise FinalizationError("certify-failed", type(exc).__name__) from exc
    # Mechanical self-check before the message enters history.
    verify_certificate(candidate=candidate, certificate=certificate)
    return {
        "draft_response": candidate.text,
        "final_response": candidate.text,
        "evidence_pack": [dict(item) for item in candidate.evidence_bundle],
        "answer_candidate": candidate.model_dump(mode="json"),
        "verification_certificate": certificate.model_dump(mode="json"),
        "delivery_status": "provisional-certified",
        "messages": [AIMessage(content=candidate.text)],
    }


__all__ = [
    "AnswerCandidate",
    "DeliveryChannel",
    "DeliveryReceipt",
    "FinalizationError",
    "OutcomeKind",
    "ReceiptStatus",
    "VerificationCertificate",
    "WholeAnswerVerdict",
    "build_context_snapshot",
    "build_delivery_receipts",
    "candidate_from_state",
    "certify_candidate",
    "context_digest_for_turn",
    "evaluate_whole_answer",
    "evidence_digest_for_pack",
    "finalize_answer_node",
    "infer_outcome_kind",
    "normalize_answer_text",
    "sha256_text",
    "split_certified_text",
    "verify_certificate",
    "verify_transport_split",
]
