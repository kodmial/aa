"""Independent end-to-end book-fidelity qualification (kodmial/aa#307).

This module upgrades qualification from "the text traversed the pipeline
and telemetry looks positive" to independent proof that the final
delivered answer is faithful to the authoritative book, complete for the
user intent, and preserves material conditions and limitations.

Evaluation path per held-out case (all recorded)::

    user turn(s) -> resolved intent/information needs
      -> discovered/read evidence -> final AnswerCandidate
      -> VerificationCertificate -> delivered text (+ receipts)

The independent evaluator below reads the relevant authoritative book
source material (``source_texts``) itself. Production verifier booleans,
usefulness-judge verdicts, and telemetry flags (``semantic_selection_applied``,
``model_selection`` routes, positive counts) are never accepted as the
independent oracle; helpers in this module explicitly reject forged or
coercible signals.

Status contract (fail-closed): ``PASS`` / ``FAIL`` / ``INCOMPLETE`` /
``STALE``. Offline deterministic controls in this module prove the
machinery with invented fixture text only (no canonical book text, no
real user messages). Live product success additionally requires the
compiled production graph with the real canonical RU corpus/index and
providers over the live-equivalent Telegram boundary, plus an actually
completed expert human review of a representative sample. Without a
completed expert review the qualification is ``INCOMPLETE`` with reason
``expert-review-pending``, never ``PASS``. Focused unit PASS results and
synthetic reproductions are machinery evidence only and are never
counted as live product proof.

Privacy contract: public summaries carry ids, digests, counts, ranks,
latencies and failure codes only. Exact book text, user turns, answers,
and chain-of-thought never leave the trusted job except inside the
compressed + age-encrypted bundle.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = "aa-book-fidelity-307/1"
PUBLIC_SUMMARY_VERSION = "aa-book-fidelity-307-summary/1"
PROTECTED_ARTIFACT_VERSION = "aa-book-fidelity-307-protected/1"
RESULT_ISSUE = 307
TARGET_ISSUE = 7
VALID_STATUSES = ("PASS", "FAIL", "INCOMPLETE", "STALE")
EXIT_BY_STATUS = {"PASS": 0, "FAIL": 1, "INCOMPLETE": 2, "STALE": 3}

# Independence limitation surfaced in telemetry: the independent
# source-reading evaluator is procedurally separate (own rubric, own
# source reads, own verdict) but any model-backed live re-check shares
# the configured provider family with generator/verifier.
EVALUATOR_INDEPENDENCE_LIMITATION = (
    "procedurally-independent-rubric-source-read-verdict; "
    "model-backed-live-recheck-shares-configured-provider-family"
)

# Dimensions scored separately per held-out case.
DIMENSIONS = (
    "completeness",
    "support",
    "constraints",
    "no_external_advice",
    "identity",
)

# Machine-readable failure codes (confusion counting by failure type).
FAILURE_TYPES = (
    "ok",
    "incomplete-answer",
    "unsupported-claim",
    "missing-exception",
    "external-advice",
    "off-intent",
    "stale-certificate",
    "identity-mismatch",
    "verbatim-mismatch",
    "quote-budget",
    "incomplete-delivery",
    "forged-telemetry",
    "preview-starvation",
    "missing-source",
)

FORBIDDEN_SUMMARY_KEYS = frozenset(
    {
        "text",
        "exact_text",
        "utterance",
        "answer",
        "generated_answer",
        "evidence_text",
        "source_text",
        "passage_text",
        "book_text",
        "content",
        "synthetic_input",
        "transcript",
        "summary",
        "chain_of_thought",
        "hidden_reasoning",
        "secret",
        "identity",
        "private_key",
        "token",
    }
)

_SHA40_RE = re.compile(r"[0-9a-f]{40}")
_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+|\n+")


class BookFidelity307Error(ValueError):
    """Raised when #307 qualification invariants fail (fails closed)."""


def validate_exact_sha(value: str) -> str:
    """Validate a 40-hex git SHA (fail-closed)."""
    text = str(value or "").strip().lower()
    if not _SHA40_RE.fullmatch(text):
        raise BookFidelity307Error("main SHA must be 40 lowercase hex chars")
    return text


def assert_public_summary_safe(payload: Any) -> None:
    """Reject public summaries carrying private text signals."""

    def _walk(node: Any, key: str = "") -> None:
        if isinstance(node, dict):
            for sub_key, sub_value in node.items():
                name = str(sub_key)
                if name in FORBIDDEN_SUMMARY_KEYS:
                    if isinstance(sub_value, str) and len(sub_value.strip()) >= 8:
                        raise BookFidelity307Error(f"public summary leaks text under key {name!r}")
                    if isinstance(sub_value, (dict, list)) and sub_value:
                        raise BookFidelity307Error(
                            f"public summary leaks structure under key {name!r}"
                        )
                _walk(sub_value, name)
        elif isinstance(node, list):
            for item in node:
                _walk(item, key)

    _walk(payload)


def _casefold(text: str) -> str:
    return " ".join(str(text or "").casefold().split())


@dataclass(frozen=True)
class HeldOutAnnotation:
    """Independent hidden annotation for one held-out case.

    These annotations are never sent to the production
    planner/generator/verifier. ``required_meanings`` entries carry
    ``must_contain`` token alternatives plus the source idea they must
    come from; ``material_constraints`` entries carry exception/condition
    phrases that must be preserved; ``prohibited_conclusions`` entries
    carry phrases that must never appear as conclusions.
    """

    required_meanings: tuple[dict[str, Any], ...] = ()
    material_constraints: tuple[dict[str, Any], ...] = ()
    prohibited_conclusions: tuple[dict[str, Any], ...] = ()
    required_subquestions: int = 1


@dataclass(frozen=True)
class HeldOutCase:
    """One held-out evaluation case (metadata only, no oracle leak)."""

    case_id: str
    turns: int = 1
    paraphrase_group: str = ""
    annotation: HeldOutAnnotation = field(default_factory=HeldOutAnnotation)


@dataclass(frozen=True)
class ChainTrace:
    """Privacy-safe record of the complete evaluated chain.

    ``source_span_ids`` are the actually read source spans (passage/chunk
    ids with offsets, never text). ``delivered_quote_ranges`` are the
    actual delivered book-quote source ranges with the model-vs-lexical
    selection path. ``query_need_map`` links explicit query IDs to need
    IDs; ``preview_prompt_digest`` binds the actual selector preview
    prompt so preview starvation is detectable.
    """

    case_id: str
    question_digest: str = ""
    context_digest: str = ""
    retrieved_span_ids: tuple[str, ...] = ()
    read_span_ids: tuple[str, ...] = ()
    candidate_sha256: str = ""
    certificate_id: str = ""
    certificate_sha256: str = ""
    evidence_digest: str = ""
    delivered_sha256: str = ""
    delivered_quote_ranges: tuple[dict[str, Any], ...] = ()
    selection_path: str = ""
    query_need_map: tuple[dict[str, Any], ...] = ()
    preview_prompt_digest: str = ""
    delivery_receipts: tuple[dict[str, Any], ...] = ()
    latency_ms: float = 0.0

    def public_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question_digest": self.question_digest,
            "context_digest": self.context_digest,
            "retrieved_spans": len(self.retrieved_span_ids),
            "read_spans": len(self.read_span_ids),
            "candidate_sha256": self.candidate_sha256,
            "certificate_id": self.certificate_id,
            "certificate_sha256": self.certificate_sha256,
            "evidence_digest": self.evidence_digest,
            "delivered_sha256": self.delivered_sha256,
            "delivered_quote_ranges": [dict(item) for item in self.delivered_quote_ranges],
            "selection_path": self.selection_path,
            "query_need_map": [dict(item) for item in self.query_need_map],
            "preview_prompt_digest": self.preview_prompt_digest,
            "delivery_receipts": [dict(item) for item in self.delivery_receipts],
            "latency_ms": round(float(self.latency_ms), 3),
        }


@dataclass(frozen=True)
class FidelityScores:
    """Per-dimension independent verdict for one case."""

    case_id: str
    completeness: bool
    support: bool
    constraints: bool
    no_external_advice: bool
    identity: bool
    failure_code: str = "ok"
    detail: str = ""

    @property
    def passed(self) -> bool:
        return bool(
            self.completeness
            and self.support
            and self.constraints
            and self.no_external_advice
            and self.identity
        )

    def public_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "completeness": self.completeness,
            "support": self.support,
            "constraints": self.constraints,
            "no_external_advice": self.no_external_advice,
            "identity": self.identity,
            "passed": self.passed,
            "failure_code": self.failure_code,
        }


def _meaning_supported(meaning_text: str, source_joined: str) -> bool:
    """Check one required meaning against actually read source text."""
    from aa.grounding.gate import default_entails

    if not meaning_text.strip() or not source_joined.strip():
        return False
    try:
        return bool(default_entails(meaning_text, source_joined))
    except Exception:
        return False


def _sentence_claims(text: str) -> list[str]:
    parts = [part.strip() for part in _SENTENCE_SPLIT_RE.split(text or "") if part.strip()]
    return [part for part in parts if len(part) >= 8]


def evaluate_case_fidelity(
    *,
    case: HeldOutCase,
    candidate_text: str,
    certificate: dict[str, Any] | Any,
    evidence_pack: list[dict[str, Any]],
    source_texts: list[str],
    delivered_text: str,
    delivery_receipts: list[dict[str, Any]] | None = None,
    answered_subquestions: int = 1,
) -> FidelityScores:
    """Independently evaluate one held-out case against read source text.

    This evaluator reads ``source_texts`` itself and never trusts
    production verifier booleans, usefulness-judge verdicts, or telemetry
    flags. Any missing source material fails closed with
    ``missing-source``: an independent verdict without reading the book
    is not a verdict.
    """
    from aa.conversation.finalization import (
        evidence_digest_for_pack,
        normalize_answer_text,
        sha256_text,
    )
    from aa.conversation.output_limits import QUOTE_BUDGET_CHARS, aggregate_quote_chars
    from aa.conversation.quote_provenance import extract_answer_quotes

    annotation = case.annotation
    if not source_texts or not any(str(item or "").strip() for item in source_texts):
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code="missing-source",
            detail="independent evaluator has no read source material",
        )
    source_joined = "\n".join(str(item) for item in source_texts if str(item or "").strip())

    # ---- identity: final-text/certificate/delivery binding ----
    identity_ok = True
    identity_code = "ok"
    try:
        normalized_candidate = normalize_answer_text(candidate_text)
        normalized_delivered = normalize_answer_text(delivered_text)
    except Exception:
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code="identity-mismatch",
            detail="candidate or delivered text is not normalizable",
        )
    if normalized_candidate != normalized_delivered:
        identity_ok = False
        identity_code = "identity-mismatch"
    cert_answer_sha = ""
    cert_evidence_digest = ""
    try:
        if isinstance(certificate, dict):
            cert_answer_sha = str(certificate.get("answer_sha256", ""))
            cert_evidence_digest = str(certificate.get("evidence_digest", ""))
        else:
            cert_answer_sha = str(getattr(certificate, "answer_sha256", ""))
            cert_evidence_digest = str(getattr(certificate, "evidence_digest", ""))
    except Exception:
        cert_answer_sha = ""
        cert_evidence_digest = ""
    if cert_answer_sha != sha256_text(normalized_candidate):
        # A prior positive certificate reused for unrelated final text is
        # stale, not merely a transport mismatch: it must fail even when
        # the new text looks superficially credible.
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code="stale-certificate",
            detail="certificate answer hash does not match the final text",
        )
    try:
        expected_evidence_digest = evidence_digest_for_pack(list(evidence_pack or []))
    except Exception:
        expected_evidence_digest = ""
    if not cert_evidence_digest or cert_evidence_digest != expected_evidence_digest:
        # A positive certificate bound to different evidence is stale and
        # must fail even when the final text looks credible.
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code="stale-certificate",
            detail="certificate evidence digest does not match read evidence",
        )
    if not identity_ok:
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code=identity_code,
            detail="final text and certificate identity do not match",
        )

    # ---- quotes: deterministic exactness + budget/anti-export gate ----
    try:
        extraction = extract_answer_quotes(normalized_candidate)
    except Exception:
        extraction = None
    if extraction is not None and tuple(getattr(extraction, "dangling", ()) or ()):
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=False,
            failure_code="verbatim-mismatch",
            detail="unbalanced or over-long quoted span",
        )
    spans: tuple[Any, ...] = ()
    if extraction is not None:
        spans = tuple(getattr(extraction, "spans", ()) or ())
    for span in spans:
        span_text = str(getattr(span, "span_text", ""))
        if span_text and span_text not in source_joined:
            return FidelityScores(
                case_id=case.case_id,
                completeness=False,
                support=True,
                constraints=True,
                no_external_advice=True,
                identity=True,
                failure_code="verbatim-mismatch",
                detail="quoted span is not an exact substring of read source",
            )
    try:
        if aggregate_quote_chars(normalized_candidate) > QUOTE_BUDGET_CHARS:
            return FidelityScores(
                case_id=case.case_id,
                completeness=False,
                support=True,
                constraints=True,
                no_external_advice=True,
                identity=True,
                failure_code="quote-budget",
                detail="verbatim quotation exceeds the anti-export budget",
            )
    except Exception:
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=False,
            constraints=False,
            no_external_advice=False,
            identity=True,
            failure_code="quote-budget",
            detail="quote budget could not be verified",
        )

    folded_candidate = _casefold(normalized_candidate)
    # ---- prohibited/unsupported conclusions (fail even when cited) ----
    for entry in list(annotation.prohibited_conclusions or ()):
        phrase = _casefold(str(entry.get("phrase", "")))
        if phrase and phrase in folded_candidate:
            return FidelityScores(
                case_id=case.case_id,
                completeness=False,
                support=False,
                constraints=False,
                no_external_advice=False,
                identity=True,
                failure_code="unsupported-claim",
                detail="prohibited conclusion is present",
            )
    # ---- completeness: required meanings + requested parts ----
    for entry in list(annotation.required_meanings or ()):
        alternatives = [str(item) for item in list(entry.get("must_contain", []) or [])]
        if not alternatives:
            continue
        if not any(_casefold(token) in folded_candidate for token in alternatives if token):
            return FidelityScores(
                case_id=case.case_id,
                completeness=False,
                support=True,
                constraints=True,
                no_external_advice=True,
                identity=True,
                failure_code="incomplete-answer",
                detail="required meaning is missing from the delivered answer",
            )
        meaning_text = str(entry.get("meaning", "") or alternatives[0])
        if not _meaning_supported(meaning_text, source_joined):
            return FidelityScores(
                case_id=case.case_id,
                completeness=False,
                support=False,
                constraints=True,
                no_external_advice=True,
                identity=True,
                failure_code="unsupported-claim",
                detail="required meaning is not supported by read source",
            )
    if int(answered_subquestions) < int(annotation.required_subquestions or 1):
        return FidelityScores(
            case_id=case.case_id,
            completeness=False,
            support=True,
            constraints=True,
            no_external_advice=True,
            identity=True,
            failure_code="incomplete-answer",
            detail="a requested subquestion has no answer",
        )
    # ---- constraints/exceptions preserved ----
    for entry in list(annotation.material_constraints or ()):
        alternatives = [str(item) for item in list(entry.get("must_contain", []) or [])]
        if not alternatives:
            continue
        if not any(_casefold(token) in folded_candidate for token in alternatives if token):
            return FidelityScores(
                case_id=case.case_id,
                completeness=True,
                support=True,
                constraints=False,
                no_external_advice=True,
                identity=True,
                failure_code="missing-exception",
                detail="material condition or exception is omitted",
            )
    # ---- support + external advice: every substantive claim needs source ----
    for claim in _sentence_claims(normalized_candidate):
        # Quoted spans are checked for exactness above; the surrounding
        # claim still needs semantic support unless it is pure glue.
        if not _meaning_supported(claim, source_joined):
            # Short conversational glue without substantive content is
            # exempt; anything claim-like without support is external
            # advice or an unsupported interpretation.
            tokens = [token for token in re.findall(r"[а-яa-z0-9]+", claim.casefold())]
            significant = [token for token in tokens if len(token) >= 4]
            if len(significant) < 2:
                continue
            return FidelityScores(
                case_id=case.case_id,
                completeness=True,
                support=False,
                constraints=True,
                no_external_advice=False,
                identity=True,
                failure_code="external-advice",
                detail="substantive claim lacks book support",
            )
    # ---- off-intent: supported text that answers something else ----
    # At least one required meaning must be present (checked above); an
    # answer with no required meaning at all but with supported generic
    # text is off-intent. The completeness check above already covers
    # this deterministically, so reaching here means on-intent.
    _ = time.perf_counter
    return FidelityScores(
        case_id=case.case_id,
        completeness=True,
        support=True,
        constraints=True,
        no_external_advice=True,
        identity=True,
        failure_code="ok",
    )


def check_delivery_receipts(
    *,
    certificate_text: str,
    receipts: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Require confirmed delivery of the complete certified text.

    Split, partial, voice/text-fragment, or unknown receipts never prove
    that the complete text reached the user. Returns ``(ok, code)``.
    """
    from aa.conversation.finalization import normalize_answer_text, sha256_text

    if not receipts:
        return False, "incomplete-delivery"
    try:
        normalized = normalize_answer_text(certificate_text)
    except Exception:
        return False, "incomplete-delivery"
    expected = sha256_text(normalized)
    confirmed = [item for item in receipts if str(item.get("status", "")) == "confirmed"]
    if not confirmed:
        return False, "incomplete-delivery"
    for item in confirmed:
        if str(item.get("final_sha256", "")) != expected:
            return False, "identity-mismatch"
    # The confirmed intervals must jointly cover the whole text.
    try:
        total = len(normalized)
        intervals: list[tuple[int, int]] = []
        for item in sorted(confirmed, key=lambda entry: int(entry.get("char_start", 0) or 0)):
            start = int(item.get("char_start", 0) or 0)
            end = int(item.get("char_end", 0) or 0)
            start = max(0, start)
            end = min(total, end)
            if end > start:
                intervals.append((start, end))
        covered = 0
        cursor_start = -1
        cursor_end = -1
        for start, end in intervals:
            if start > cursor_end:
                if cursor_end > cursor_start:
                    covered += cursor_end - cursor_start
                cursor_start, cursor_end = start, end
            else:
                cursor_end = max(cursor_end, end)
        if cursor_end > cursor_start:
            covered += cursor_end - cursor_start
        if covered < total:
            return False, "incomplete-delivery"
    except Exception:
        return False, "incomplete-delivery"
    return True, "ok"


def reject_forged_telemetry(telemetry: dict[str, Any]) -> tuple[bool, str]:
    """Reject convenient positive flags that prove nothing about fidelity.

    Returns ``(forged, code)`` where ``forged=True`` means the telemetry
    must not be trusted. Covers: false ``semantic_selection_applied``,
    false ``model_selection`` stage/route, contradictory provider-output
    types or origin refs, forged XML candidate blocks, false
    read-as-quoted history, and coercible verifier values.
    """
    applied = telemetry.get("semantic_selection_applied", False)
    route = str(telemetry.get("selection_route", "") or "")
    order = str(telemetry.get("pack_order", "") or "")
    if applied is True and route not in ("model_selection", ""):
        # ``applied`` without the model route is incoherent unless the
        # route field is simply absent (legacy); an explicit lexical
        # route with applied=True is forged.
        if route and route != "model_selection":
            return True, "forged-telemetry"
    if order == "model_selection" and route != "model_selection":
        return True, "forged-telemetry"
    if route == "model_selection" and applied is not True:
        return True, "forged-telemetry"
    provider_type = str(telemetry.get("provider_output_type", "") or "")
    origin_ref = str(telemetry.get("origin_ref", "") or "")
    if provider_type and origin_ref and provider_type != origin_ref:
        return True, "forged-telemetry"
    candidate_block = str(telemetry.get("candidate_block", "") or "")
    if "<candidate" in candidate_block or "<AnswerCandidate" in candidate_block:
        # Raw XML candidate blocks are never trusted telemetry: they are
        # an injection surface, not provenance.
        return True, "forged-telemetry"
    read_as_quoted = telemetry.get("read_as_quoted", None)
    quoted_ranges = telemetry.get("quoted_ranges", None)
    if read_as_quoted is True and not quoted_ranges:
        return True, "forged-telemetry"
    verifier_value = telemetry.get("verifier_supported", None)
    if isinstance(verifier_value, str) and verifier_value.strip().lower() in (
        "true",
        "1",
        "yes",
        "supported",
    ):
        # Coercible string verdicts are never accepted as booleans.
        return True, "forged-telemetry"
    if "injected_source_delimiter" in telemetry:
        return True, "forged-telemetry"
    user_report = telemetry.get("user_report_claim", None)
    if isinstance(user_report, str) and user_report.strip():
        # Misattributed ``user_report`` claims never count as book proof.
        return True, "forged-telemetry"
    return False, "ok"


def check_query_need_links(
    query_need_map: list[dict[str, Any]],
    need_ids: list[str],
) -> tuple[bool, str]:
    """Require explicit query-ID to need-ID links for every need."""
    linked: set[str] = set()
    for entry in query_need_map or []:
        if not isinstance(entry, dict):
            continue
        query_id = str(entry.get("query_id", "") or "").strip()
        if not query_id:
            return False, "forged-telemetry"
        for need_id in list(entry.get("need_ids", []) or []):
            if str(need_id or "").strip():
                linked.add(str(need_id))
    for need_id in need_ids or []:
        if need_id and need_id not in linked:
            return False, "preview-starvation"
    return True, "ok"


def check_preview_starvation(
    *,
    preview_prompt_digest: str,
    per_need_preview_counts: dict[str, int],
    need_ids: list[str],
) -> tuple[bool, str]:
    """Detect per-information-need preview starvation.

    One underserved need must remain discoverable despite repeated
    high-ranked paraphrases of another: every need needs at least one
    preview, and the actual selector preview prompt must be bound via
    its digest (an empty digest proves nothing).
    """
    if not preview_prompt_digest.strip():
        return False, "preview-starvation"
    for need_id in need_ids or []:
        if int(per_need_preview_counts.get(need_id, 0) or 0) < 1:
            return False, "preview-starvation"
    return True, "ok"


def build_offline_controls() -> list[dict[str, Any]]:
    """Build deterministic offline controls (invented fixture text only).

    Each entry carries the case, candidate, certificate inputs, read
    source texts, delivery data, and the expected evaluator outcome.
    Positive controls must pass; every negative control must fail with
    the documented failure code. No canonical book text is embedded.
    """
    import hashlib as _hashlib

    from aa.conversation.finalization import (
        evidence_digest_for_pack,
        normalize_answer_text,
        sha256_text,
    )

    # Invented fixture book source: two contiguous chunks of one passage.
    chunk_a = (
        "Invented fixture morning support: steady companionship nearby helps "
        "meet the early craving calmly with patient daily review."
    )
    chunk_b = (
        "Invented fixture morning support continued: the same steady review "
        "applies only together with honest conversation, never alone, and "
        "asks a second question to be answered separately."
    )
    source_texts = [chunk_a, chunk_b]

    def _pack() -> list[dict[str, Any]]:
        first = chunk_a
        second = chunk_b
        return [
            {
                "passage_id": "chapter-3#exp0000",
                "text": first + " " + second,
                "source_id": "ru-fourth-edition-txt",
                "section_id": "chapter-3",
                "char_start": 0,
                "char_end": len(first + " " + second),
                "text_sha256": _hashlib.sha256((first + " " + second).encode("utf-8")).hexdigest(),
                "source_sha256": "s" * 64,
                "corpus_version": "r" * 64,
            }
        ]

    def _cert(text: str, pack: list[dict[str, Any]]) -> dict[str, Any]:
        normalized = normalize_answer_text(text)
        return {
            "answer_sha256": sha256_text(normalized),
            "evidence_digest": evidence_digest_for_pack(pack),
            "context_digest": "c" * 64,
            "certificate_id": "cert-" + sha256_text(normalized)[:8],
        }

    annotation = HeldOutAnnotation(
        required_meanings=(
            {"id": "m1", "must_contain": ["steady companionship"], "meaning": chunk_a},
            {"id": "m2", "must_contain": ["honest conversation"], "meaning": chunk_b},
        ),
        material_constraints=({"id": "c1", "must_contain": ["never alone"]},),
        prohibited_conclusions=({"phrase": "universal rule for every situation"},),
        required_subquestions=2,
    )
    controls: list[dict[str, Any]] = []

    def _add(
        suffix: str,
        candidate: str,
        *,
        expected_pass: bool,
        failure_code: str,
        certificate: dict[str, Any] | None = None,
        delivered: str | None = None,
        answered: int = 2,
        sources: list[str] | None = None,
        pack_override: list[dict[str, Any]] | None = None,
        annotation_override: HeldOutAnnotation | None = None,
    ) -> None:
        pack = pack_override if pack_override is not None else _pack()
        cert = certificate if certificate is not None else _cert(candidate, pack)
        controls.append(
            {
                "control_id": suffix,
                "case": HeldOutCase(
                    case_id=f"307-{suffix}",
                    annotation=annotation_override
                    if annotation_override is not None
                    else annotation,
                ),
                "candidate_text": candidate,
                "certificate": cert,
                "evidence_pack": pack,
                "source_texts": sources if sources is not None else list(source_texts),
                "delivered_text": candidate if delivered is None else delivered,
                "answered_subquestions": answered,
                "expected_pass": expected_pass,
                "expected_failure": failure_code,
            }
        )

    # Positive: faithful complete answer with an accurate contiguous
    # cross-chunk quote (two independent quoted ranges allowed).
    good = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review; second answer "
        "covers the honest conversation separately. "
        '"steady companionship nearby" and "honest conversation, never alone"'
    )
    _add("positive-faithful", good, expected_pass=True, failure_code="ok")

    # Mandatory negative 1: unrelated final text with a prior positive cert.
    unrelated = (
        "Unrelated evening city walking advice with fresh air and street "
        "maps for tourists downtown."
    )
    _add(
        "negative-unrelated-with-stale-cert",
        unrelated,
        expected_pass=False,
        failure_code="stale-certificate",
        certificate=_cert(good, _pack()),
    )
    # Mandatory negative 2: book-supported but off-intent paragraph (only
    # the first meaning, second subquestion missing).
    off_intent = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, never alone."
    )
    _add(
        "negative-off-intent-partial",
        off_intent,
        expected_pass=False,
        failure_code="incomplete-answer",
        answered=1,
    )
    # Mandatory negative 3: relevant principle with missing exception.
    missing_exception = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review together with honest conversation. "
        "First answer covers the morning review; second answer covers the "
        "honest conversation separately."
    )
    _add(
        "negative-missing-exception",
        missing_exception,
        expected_pass=False,
        failure_code="missing-exception",
    )
    # Mandatory negative 4: useful general advice absent from the book.
    external = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review; second answer "
        "covers the honest conversation separately. "
        "Drink energizing herbal tonics every hour for vitality."
    )
    _add(
        "negative-external-advice",
        external,
        expected_pass=False,
        failure_code="external-advice",
    )
    # Mandatory negative 5: incomplete two-part answer.
    incomplete = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review."
    )
    _add(
        "negative-incomplete-two-part",
        incomplete,
        expected_pass=False,
        failure_code="incomplete-answer",
        answered=1,
    )
    # Mandatory negative 6: citation points to source but semantics differ.
    wrong_semantics = (
        "Steady companionship nearby proves the universal rule for every "
        "situation without exception, with patient daily review and honest "
        "conversation, never alone. First and second answers both restate "
        "this universal rule."
    )
    _add(
        "negative-cited-but-unsupported",
        wrong_semantics,
        expected_pass=False,
        failure_code="unsupported-claim",
    )
    # Integration: invented multi-sentence book quotation.
    fabricated_quote = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review; second answer "
        "covers the honest conversation separately. "
        '"Invented fabricated morning doctrine across several sentences '
        'that never appears in the source at all."'
    )
    _add(
        "negative-fabricated-quote",
        fabricated_quote,
        expected_pass=False,
        failure_code="verbatim-mismatch",
    )
    # Correct principle applied to the wrong situation: the extra tourist
    # sentence shares no vocabulary with the read source, so the
    # independent support check fails it as external advice.
    wrong_situation = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review; second answer "
        "covers the honest conversation. "
        "Evening city tourists should follow downtown street maps at sunset."
    )
    _add(
        "negative-wrong-situation",
        wrong_situation,
        expected_pass=False,
        failure_code="external-advice",
    )
    # Story generalized into a universal rule.
    generalized = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone, and this universal rule for every situation always "
        "holds. First answer covers the morning review; second answer "
        "covers the honest conversation."
    )
    _add(
        "negative-story-as-rule",
        generalized,
        expected_pass=False,
        failure_code="unsupported-claim",
    )
    # Quote-budget/anti-export gate: a fabricated cross-line quotation
    # must fail deterministic exactness, and an accurate contiguous
    # quotation that exceeds the aggregate budget must fail the budget
    # gate even though it is verbatim.
    _add(
        "negative-quote-budget",
        good + ' "' + ("quoted invented filler sentence. " * 30) + '"',
        expected_pass=False,
        failure_code="verbatim-mismatch",
    )
    over_budget_verbatim = (
        "Steady companionship nearby helps meet the early craving calmly "
        "with patient daily review, only together with honest conversation, "
        "never alone. First answer covers the morning review; second answer "
        "covers the honest conversation separately. "
        '"' + chunk_a + " " + chunk_b + '"'
    )
    # The concatenated cross-chunk span above joins chunks with a space
    # while the independent source join uses a newline, so it exercises
    # exactness; the dedicated over-budget control below uses one long
    # contiguous source span that is verbatim and exceeds the budget.
    long_source = "Invented fixture long morning support passage. " * 20
    long_slice = long_source[:360]
    long_pack = [
        {
            "passage_id": "chapter-3#exp0000",
            "text": long_source,
            "source_id": "ru-fourth-edition-txt",
            "section_id": "chapter-3",
            "char_start": 0,
            "char_end": len(long_source),
            "text_sha256": _hashlib.sha256(long_source.encode("utf-8")).hexdigest(),
            "source_sha256": "s" * 64,
            "corpus_version": "r" * 64,
        }
    ]
    long_annotation = HeldOutAnnotation(
        required_meanings=(
            {
                "id": "m1",
                "must_contain": ["long morning support"],
                "meaning": "Invented fixture long morning support passage.",
            },
        ),
        material_constraints=(),
        prohibited_conclusions=(),
        required_subquestions=1,
    )
    long_candidate = (
        'Invented fixture long morning support passage with steady review. "' + long_slice + '"'
    )
    _add(
        "negative-accurate-quote-over-budget",
        long_candidate,
        expected_pass=False,
        failure_code="quote-budget",
        sources=[long_source],
        pack_override=long_pack,
        annotation_override=long_annotation,
        answered=1,
    )
    _ = over_budget_verbatim
    return controls


def decide_status_307(
    *,
    stale: bool,
    incomplete: bool,
    failures: int,
    case_results: list[FidelityScores],
    expert_review_completed: bool,
    live_decisive: bool,
) -> str:
    """Decide the deterministic #307 status.

    ``PASS`` requires every positive control to pass, every negative
    control to fail for the right reason, a decisive live run through
    the compiled production graph and Telegram boundary, and a completed
    expert human review. Anything weaker is ``FAIL`` (wrong verdict),
    ``INCOMPLETE`` (missing evidence or pending review), or ``STALE``.
    """
    if stale:
        return "STALE"
    if incomplete:
        return "INCOMPLETE"
    if failures < 0:
        raise BookFidelity307Error("failure count must be >= 0")
    if failures > 0:
        return "FAIL"
    if any(not isinstance(item, FidelityScores) for item in case_results):
        return "FAIL"
    # Synthetic offline machinery alone never qualifies the product.
    if not live_decisive:
        return "INCOMPLETE"
    if not expert_review_completed:
        return "INCOMPLETE"
    if not case_results:
        return "INCOMPLETE"
    positives = [item for item in case_results if item.case_id.endswith("positive-faithful")]
    if not positives:
        return "INCOMPLETE"
    if any(not item.passed for item in positives):
        return "FAIL"
    return "PASS"


def summarize_public_307(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    config_sha: str,
    model: str,
    traces: list[ChainTrace],
    scores: list[FidelityScores],
    status: str,
    run_id: str,
    expert_review_completed: bool,
    live_decisive: bool,
) -> dict[str, Any]:
    """Build the privacy-safe public summary (metrics only)."""
    if status not in VALID_STATUSES:
        raise BookFidelity307Error(f"invalid status {status!r}")
    latencies = sorted(float(item.latency_ms) for item in traces)
    by_failure: dict[str, int] = {}
    for item in scores:
        by_failure[item.failure_code] = by_failure.get(item.failure_code, 0) + 1

    def _pct(value: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(len(latencies) - 1, int(value * len(latencies)))], 3)

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_SUMMARY_VERSION,
        "issue": RESULT_ISSUE,
        "target_issue": TARGET_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "config_sha256": config_sha,
        "model": model,
        "result": status,
        "run_id": run_id,
        "expert_review_completed": bool(expert_review_completed),
        "live_decisive": bool(live_decisive),
        "independence_limitation": EVALUATOR_INDEPENDENCE_LIMITATION,
        "synthetic_only": not bool(live_decisive),
        "turn_count": len(traces),
        "case_count": len(scores),
        "pass_count": sum(1 for item in scores if item.passed),
        "fail_count": sum(1 for item in scores if not item.passed),
        "confusion_by_failure": dict(sorted(by_failure.items())),
        "latency_p50_ms": _pct(0.5),
        "latency_p95_ms": _pct(0.95),
        "traces": [item.public_row() for item in traces],
        "scores": [item.public_row() for item in scores],
    }
    assert_public_summary_safe(payload)
    return payload


def build_protected_payload_307(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    config_sha: str,
    model: str,
    traces: list[ChainTrace],
    scores: list[FidelityScores],
    status: str,
    run_id: str,
) -> dict[str, Any]:
    """Build the encrypted protected payload (counts/digests only)."""
    return {
        "schema_version": PROTECTED_ARTIFACT_VERSION,
        "issue": RESULT_ISSUE,
        "target_issue": TARGET_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "config_sha256": config_sha,
        "model": model,
        "result": status,
        "run_id": run_id,
        "traces": [item.public_row() for item in traces],
        "scores": [item.public_row() for item in scores],
    }


def live_graph_telegram_wiring_present() -> tuple[bool, str]:
    """Check the compiled production graph + Telegram boundary are wired.

    Static import-level proof used by offline tests: the exact-main
    qualification path must run through ``build_turn_graph`` /
    ``GraphTurnRuntime`` and ``Application`` delivery, never a mock-only
    declaration of product success.
    """
    try:
        from aa.app import Application  # noqa: F401
        from aa.conversation.graph import build_turn_graph, turn_input  # noqa: F401
        from aa.conversation.graph_runtime import GraphTurnRuntime  # noqa: F401
    except Exception as exc:
        return False, f"production-graph-telegram-missing: {type(exc).__name__}"
    return True, "ok"


__all__ = [
    "DIMENSIONS",
    "EVALUATOR_INDEPENDENCE_LIMITATION",
    "EXIT_BY_STATUS",
    "FAILURE_TYPES",
    "FORBIDDEN_SUMMARY_KEYS",
    "PROTECTED_ARTIFACT_VERSION",
    "PUBLIC_SUMMARY_VERSION",
    "RESULT_ISSUE",
    "SCHEMA_VERSION",
    "TARGET_ISSUE",
    "VALID_STATUSES",
    "BookFidelity307Error",
    "ChainTrace",
    "FidelityScores",
    "HeldOutAnnotation",
    "HeldOutCase",
    "assert_public_summary_safe",
    "build_offline_controls",
    "build_protected_payload_307",
    "check_delivery_receipts",
    "check_preview_starvation",
    "check_query_need_links",
    "decide_status_307",
    "evaluate_case_fidelity",
    "live_graph_telegram_wiring_present",
    "reject_forged_telemetry",
    "summarize_public_307",
    "validate_exact_sha",
]
