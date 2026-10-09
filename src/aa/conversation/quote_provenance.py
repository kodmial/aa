"""Authoritative whole-answer quote extraction and claim-origin attribution.

Issue kodmial/aa#308 (AUDIT P0): closes two reproducible failure modes:

- invented multi-sentence quoted book text bypassing exact-quote validation
  because quote detection happened after sentence splitting;
- faithfully quoted user words incorrectly required to be book citations.

Contract:

- :func:`extract_answer_quotes` parses the complete answer candidate BEFORE
  sentence segmentation, including quotes crossing sentence boundaries,
  paragraphs and newlines. It is the single canonical full-text quote
  parser used by deterministic exact-quotation checking, quote-budget
  enforcement, citation/provenance handling and delivered-quote history.
- :func:`certify_answer_candidate` validates every extracted span with
  exact answer/source offsets and durable provenance under a typed
  claim/quote provenance contract (``origin`` / ``origin_ref``).
- A quote mark alone never sets its origin: model-led semantic
  classification (``claim_origin`` on the verifier transport decision)
  and deterministic provenance/offset/role validation must both hold,
  failing closed on ambiguity.

Offset convention: ``answer_char_start`` / ``answer_char_end`` are Python
``str`` (code-point) offsets into the exact answer candidate, end
exclusive, with ``answer_utf8_start`` / ``answer_utf8_end`` as the
corresponding UTF-8 byte offsets. Source offsets use the same convention
against the canonical contiguous source range. Both are recorded and
tested so #304 can re-verify the exact final text and #312 can consume
identical spans after delivery.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

QUOTE_PARSER_VERSION = "answer-quote-parser/1"

MAX_QUOTED_SPAN_CHARS = 2000

OriginName = Literal[
    "book_claim",
    "user_report",
    "assistant_capability",
    "conversation_glue",
    "safety_override",
]

ORIGINS: tuple[str, ...] = (
    "book_claim",
    "user_report",
    "assistant_capability",
    "conversation_glue",
    "safety_override",
)

# Deterministic delimiter pairs. ASCII single quotes are deliberately NOT
# quotable: they collide with apostrophes (e.g. "don't") and would turn
# ordinary prose into phantom quotations. Supported families: «», "" (low
# German open with curly/ascii close), curly double "", ASCII double "",
# single-pointing <>, curly single ''.
_OPEN_TO_CLOSE: dict[str, str] = {
    "«": "»",
    "„": "\u201c",
    "“": "”",
    '"': '"',
    "‹": "›",
    "‘": "’",
}

# Extra closers that may terminate a „-opened span when the author uses an
# ASCII close instead of the curly one.
_ALTERNATE_CLOSE: dict[str, tuple[str, ...]] = {
    "„": ('"',),
}

_SYMMETRIC_OPENS = {'"'}


def _is_escaped(text: str, pos: int) -> bool:
    """Whether ``text[pos]`` is backslash-escaped (odd run of backslashes)."""
    backslashes = 0
    cursor = pos - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 1


def _utf8_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Return UTF-8 byte offsets for the Python-str slice ``[start:end)``."""
    return len(text[:start].encode("utf-8")), len(text[:end].encode("utf-8"))


@dataclass(frozen=True)
class ExtractedQuote:
    """One balanced quoted span from the full answer candidate."""

    span_text: str
    answer_char_start: int
    answer_char_end: int
    answer_utf8_start: int
    answer_utf8_end: int
    quote_char_start: int
    quote_char_end: int
    open_quote: str
    close_quote: str


@dataclass(frozen=True)
class DanglingQuote:
    """An unmatched/stray delimiter that must not silently evade validation."""

    answer_char_start: int
    answer_char_end: int
    answer_utf8_start: int
    answer_utf8_end: int
    detail: str


@dataclass(frozen=True)
class AnswerQuoteExtraction:
    """Outcome of parsing one full answer candidate."""

    parser_version: str
    answer_sha256: str
    answer_chars: int
    answer_utf8_len: int
    spans: tuple[ExtractedQuote, ...] = ()
    dangling: tuple[DanglingQuote, ...] = ()


def extract_answer_quotes(answer: str) -> AnswerQuoteExtraction:
    """Extract complete balanced quoted spans from the full answer text.

    Runs BEFORE sentence segmentation: newlines never terminate a span,
    so multi-sentence and multi-paragraph quotations are one span. Uses
    an explicit delimiter stack, so nesting across families is
    deterministic; unmatched openers, stray closers and over-long spans
    are reported as dangling (fail-closed downstream), never skipped
    silently. Backslash-escaped delimiters are literal text.
    """
    text = answer if isinstance(answer, str) else ""
    spans: list[ExtractedQuote] = []
    dangling: list[DanglingQuote] = []
    # Stack entries: (open_char, expected_close, content_start, quote_start).
    stack: list[tuple[str, str, int, int]] = []
    closes = set(_OPEN_TO_CLOSE.values()) | {'"'}
    idx = 0
    while idx < len(text):
        char = text[idx]
        if char == "\\":
            idx += 2 if idx + 1 < len(text) else 1
            continue
        if _is_escaped(text, idx):
            idx += 1
            continue
        if stack and char == stack[-1][1]:
            open_q, _, content_start, quote_start = stack.pop()
            content = text[content_start:idx]
            if len(content) > MAX_QUOTED_SPAN_CHARS:
                utf_start, utf_end = _utf8_span(text, quote_start, idx + 1)
                dangling.append(
                    DanglingQuote(
                        answer_char_start=quote_start,
                        answer_char_end=idx + 1,
                        answer_utf8_start=utf_start,
                        answer_utf8_end=utf_end,
                        detail="quoted-span-exceeds-maximum",
                    )
                )
            else:
                utf_cs, utf_ce = _utf8_span(text, content_start, idx)
                spans.append(
                    ExtractedQuote(
                        span_text=content,
                        answer_char_start=content_start,
                        answer_char_end=idx,
                        answer_utf8_start=utf_cs,
                        answer_utf8_end=utf_ce,
                        quote_char_start=quote_start,
                        quote_char_end=idx + 1,
                        open_quote=open_q,
                        close_quote=char,
                    )
                )
            idx += 1
            continue
        # Alternate close for „-opened spans using an ASCII close quote.
        if stack:
            expected_open = stack[-1][0]
            alternates = _ALTERNATE_CLOSE.get(expected_open, ())
            if char in alternates:
                _, _, content_start, quote_start = stack.pop()
                content = text[content_start:idx]
                if len(content) > MAX_QUOTED_SPAN_CHARS:
                    utf_start, utf_end = _utf8_span(text, quote_start, idx + 1)
                    dangling.append(
                        DanglingQuote(
                            answer_char_start=quote_start,
                            answer_char_end=idx + 1,
                            answer_utf8_start=utf_start,
                            answer_utf8_end=utf_end,
                            detail="quoted-span-exceeds-maximum",
                        )
                    )
                else:
                    utf_cs, utf_ce = _utf8_span(text, content_start, idx)
                    spans.append(
                        ExtractedQuote(
                            span_text=content,
                            answer_char_start=content_start,
                            answer_char_end=idx,
                            answer_utf8_start=utf_cs,
                            answer_utf8_end=utf_ce,
                            quote_char_start=quote_start,
                            quote_char_end=idx + 1,
                            open_quote=expected_open,
                            close_quote=char,
                        )
                    )
                idx += 1
                continue
        if char in _OPEN_TO_CLOSE and char not in _SYMMETRIC_OPENS:
            stack.append((char, _OPEN_TO_CLOSE[char], idx + 1, idx))
            idx += 1
            continue
        if char in _SYMMETRIC_OPENS:
            # Symmetric delimiters toggle: open when the top is not ours.
            stack.append((char, char, idx + 1, idx))
            idx += 1
            continue
        if char in closes:
            utf_s, utf_e = _utf8_span(text, idx, idx + 1)
            dangling.append(
                DanglingQuote(
                    answer_char_start=idx,
                    answer_char_end=idx + 1,
                    answer_utf8_start=utf_s,
                    answer_utf8_end=utf_e,
                    detail="stray-closing-quote",
                )
            )
            idx += 1
            continue
        idx += 1
    for open_q, _, content_start, quote_start in stack:
        del open_q
        utf_s, utf_e = _utf8_span(text, quote_start, len(text))
        dangling.append(
            DanglingQuote(
                answer_char_start=quote_start,
                answer_char_end=len(text),
                answer_utf8_start=utf_s,
                answer_utf8_end=utf_e,
                detail="unclosed-quote",
            )
        )
        _ = content_start
    ordered = tuple(sorted(spans, key=lambda item: (item.answer_char_start, item.answer_char_end)))
    dangling_ordered = tuple(
        sorted(dangling, key=lambda item: (item.answer_char_start, item.answer_char_end))
    )
    raw = text.encode("utf-8")
    return AnswerQuoteExtraction(
        parser_version=QUOTE_PARSER_VERSION,
        answer_sha256=hashlib.sha256(raw).hexdigest(),
        answer_chars=len(text),
        answer_utf8_len=len(raw),
        spans=ordered,
        dangling=dangling_ordered,
    )


def extract_quoted_span_texts(answer: str) -> list[str]:
    """Return quoted span texts (canonical parser, citation-safe callers strip)."""
    return [span.span_text.strip() for span in extract_answer_quotes(answer).spans]


def total_quoted_chars(answer: str) -> int:
    """Aggregate characters inside every quoted span (any origin)."""
    extraction = extract_answer_quotes(answer)
    return sum(len(span.span_text) for span in extraction.spans if span.span_text.strip())


class QuoteOriginError(ValueError):
    """A quoted span or claim origin failed closed."""


@dataclass(frozen=True)
class UserMessageRef:
    """Trusted referent for a ``user_report`` span: one HumanMessage slice."""

    message_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class BookSourceAnchor:
    """Exact canonical source range anchoring one book-origin quoted span."""

    source_sha256: str
    section_id: str
    source_id: str
    source_char_start: int
    source_char_end: int
    passage_ids: tuple[str, ...]
    corpus_version: str
    text_sha256: str


@dataclass(frozen=True)
class SpanOverlap:
    """Overlap mapping between one quoted span and verifier response units."""

    answer_char_start: int
    answer_char_end: int
    unit_ids: tuple[str, ...] = ()


def map_spans_to_units(
    spans: tuple[ExtractedQuote, ...] | list[ExtractedQuote],
    units: Any,
) -> list[SpanOverlap]:
    """Map each quoted span to the verifier response unit(s) it overlaps."""
    overlaps: list[SpanOverlap] = []
    unit_spans: list[tuple[str, int, int]] = []
    for unit in units or []:
        try:
            unit_spans.append(
                (
                    str(getattr(unit, "unit_id", "")),
                    int(getattr(unit, "char_start", 0)),
                    int(getattr(unit, "char_end", 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    for span in spans:
        matched = tuple(
            unit_id
            for unit_id, start, end in unit_spans
            if unit_id and span.answer_char_start < end and span.answer_char_end > start
        )
        overlaps.append(
            SpanOverlap(
                answer_char_start=span.answer_char_start,
                answer_char_end=span.answer_char_end,
                unit_ids=matched,
            )
        )
    return overlaps


def build_user_message_index(
    recent: Any,
    user_message: str,
) -> list[dict[str, Any]]:
    """Build the trusted HumanMessage index for ``user_report`` validation.

    Only human-role messages qualify; assistant messages, summaries and
    navigation artifacts are never canonical user evidence. The current
    live turn is ``turn-human-current``; earlier human messages are
    ``turn-human-prior-{n}`` in recency order (``0`` is the most recent
    prior). Text is kept in memory only for substring validation and is
    never logged by callers.
    """
    indexed: list[dict[str, Any]] = []
    current = str(user_message or "")
    if current.strip():
        indexed.append({"message_id": "turn-human-current", "role": "human", "text": current})
    position = 0
    for item in list(recent or []):
        role = str(getattr(item, "type", "") or "").strip().lower()
        content = getattr(item, "content", "")
        text = content if isinstance(content, str) else ""
        if role in ("human", "user") and text.strip():
            indexed.append(
                {"message_id": f"turn-human-prior-{position}", "role": "human", "text": text}
            )
            position += 1
    return indexed


def anchor_user_span(span_text: str, user_index: list[dict[str, Any]]) -> UserMessageRef | None:
    """Anchor ``span_text`` verbatim to one trusted HumanMessage, if present."""
    wanted = str(span_text or "")
    if not wanted.strip():
        return None
    for entry in user_index:
        if str(entry.get("role", "")) != "human":
            continue
        haystack = str(entry.get("text", "") or "")
        if not haystack:
            continue
        found = haystack.find(wanted)
        if found < 0:
            continue
        message_id = str(entry.get("message_id", ""))
        if not message_id:
            continue
        return UserMessageRef(message_id=message_id, char_start=found, char_end=found + len(wanted))
    return None


def _passage_sort_key(item: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return (
        str(item.get("source_id", item.get("source", "") or "")),
        str(item.get("section_id", item.get("section", "") or "")),
        int(item.get("char_start", 0) or 0),
        int(item.get("char_end", 0) or 0),
        str(item.get("passage_id", "") or ""),
    )


def contiguous_source_runs(
    passages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Group passages into exactly contiguous canonical runs.

    Runs never cross a source/section boundary and never silently bridge
    a gap or overlap: concatenation happens only when
    ``next.char_start == prev.char_end`` with identical ``source_id``,
    ``section_id`` and ``source_sha256``. Each run carries the ordered
    passage ids it covers plus the joint text and range.
    """
    ordered = sorted(
        [dict(item) for item in passages if isinstance(item, dict)], key=_passage_sort_key
    )
    runs: list[dict[str, Any]] = []
    for item in ordered:
        text = item.get("text", "")
        passage_id = str(item.get("passage_id", "") or "")
        if not isinstance(text, str) or not text or not passage_id:
            continue
        source_id = str(item.get("source_id", item.get("source", "") or ""))
        section_id = str(item.get("section_id", item.get("section", "") or ""))
        source_sha = str(item.get("source_sha256", "") or "")
        try:
            start = int(item.get("char_start", 0) or 0)
            end = int(item.get("char_end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if not source_id or not section_id or not source_sha or end <= start:
            continue
        corpus_version = str(item.get("corpus_version", "") or "")
        if runs:
            tail = runs[-1]
            if (
                tail["source_id"] == source_id
                and tail["section_id"] == section_id
                and tail["source_sha256"] == source_sha
                and start == int(tail["char_end"])
            ):
                tail["text"] = str(tail["text"]) + text
                tail["char_end"] = end
                tail["passage_ids"] = (*tail["passage_ids"], passage_id)
                if corpus_version and not tail["corpus_version"]:
                    tail["corpus_version"] = corpus_version
                continue
        runs.append(
            {
                "source_id": source_id,
                "section_id": section_id,
                "source_sha256": source_sha,
                "corpus_version": corpus_version,
                "char_start": start,
                "char_end": end,
                "text": text,
                "passage_ids": (passage_id,),
            }
        )
    return runs


def anchor_book_span(
    span_text: str,
    passages: list[dict[str, Any]],
    *,
    cited_passage_ids: list[str] | tuple[str, ...] | set[str] = (),
) -> BookSourceAnchor | None:
    """Anchor a book-origin span to a specific exact contiguous source range.

    The ENTIRE span must appear verbatim inside one contiguous canonical
    run; per-sentence fragments never suffice, so a changed negation, one
    substituted word or a fabricated second sentence fails. When
    ``cited_passage_ids`` is non-empty the covering run must include at
    least one cited passage id, binding the quote to its cited source.
    """
    wanted = str(span_text or "")
    if not wanted.strip():
        return None
    cited = {str(item) for item in (cited_passage_ids or ()) if str(item).strip()}
    for run in contiguous_source_runs(passages):
        run_text = str(run["text"])
        found = run_text.find(wanted)
        if found < 0:
            continue
        covering = tuple(run["passage_ids"])
        if cited and not (set(covering) & cited):
            continue
        run_start = int(run["char_start"])
        span_start = run_start + found
        span_end = span_start + len(wanted)
        return BookSourceAnchor(
            source_sha256=str(run["source_sha256"]),
            section_id=str(run["section_id"]),
            source_id=str(run["source_id"]),
            source_char_start=span_start,
            source_char_end=span_end,
            passage_ids=covering,
            corpus_version=str(run.get("corpus_version", "") or ""),
            text_sha256=hashlib.sha256(wanted.encode("utf-8")).hexdigest(),
        )
    return None


def origin_for_verdict(verdict: Any) -> str:
    """Return the effective claim origin for one internal verdict.

    Book scope dominates: every verdict produced by
    :func:`aa.conversation.verifier.coerce_single_verdict` satisfies
    ``scope == "book"`` iff ``origin == "book_claim"``, and legacy or
    hand-built verdicts without an explicit origin keep the historical
    reading (book scope means a book claim). An explicit non-book origin
    is honored only for non-book scopes, so a book verdict can never
    launder its quotations through a glue/user classification.
    """
    scope = str(getattr(verdict, "scope", "") or "").strip()
    raw = str(getattr(verdict, "origin", "") or "").strip()
    if scope == "book":
        return "book_claim"
    if raw in ORIGINS:
        return raw
    return "conversation_glue"


class SpanProvenance(BaseModel):
    """Serializable provenance for one quoted span of the answer candidate."""

    span_text_sha256: str = ""
    answer_char_start: int = 0
    answer_char_end: int = 0
    answer_utf8_start: int = 0
    answer_utf8_end: int = 0
    origin: str = ""
    unit_ids: list[str] = Field(default_factory=list)
    origin_ref_type: str = ""
    book_source_sha256: str = ""
    book_section_id: str = ""
    book_source_id: str = ""
    book_source_char_start: int = 0
    book_source_char_end: int = 0
    book_passage_ids: list[str] = Field(default_factory=list)
    book_corpus_version: str = ""
    user_message_id: str = ""
    user_char_start: int = 0
    user_char_end: int = 0

    model_config = {"extra": "forbid"}


class AnswerQuoteCertificate(BaseModel):
    """Serializable candidate-level quote/origin artifact for #304.

    Consumed by ``AnswerCandidate`` / ``VerificationCertificate`` at the
    single final-answer boundary. Records the extraction/parser version,
    the final-candidate content hash, exact answer/source offsets (Python
    str offsets plus UTF-8 byte offsets), the trusted origin_ref type,
    the cited source version, and the overlap mapping to all response
    units. This PR checks the actual current answer candidate through
    the existing reachable verifier path; #304 moves the check to the
    single delivery boundary and rechecks after every textual
    modification.
    """

    parser_version: str = QUOTE_PARSER_VERSION
    answer_sha256: str = ""
    answer_chars: int = 0
    answer_utf8_len: int = 0
    total_quote_chars: int = 0
    book_quote_chars: int = 0
    spans: list[SpanProvenance] = Field(default_factory=list)
    dangling: list[dict[str, int | str]] = Field(default_factory=list)
    overlap_units: list[list[str]] = Field(default_factory=list)
    passed: bool = False
    failure_code: str = ""

    model_config = {"extra": "forbid"}


def _verdict_origin_ref(verdict: Any) -> dict[str, Any]:
    ref = getattr(verdict, "origin_ref", None)
    if isinstance(ref, dict):
        return dict(ref)
    return {}


def certify_answer_candidate(
    *,
    answer: str,
    units: Any,
    verdicts: Any,
    passages: list[dict[str, Any]],
    user_index: list[dict[str, Any]] | None = None,
) -> AnswerQuoteCertificate:
    """Validate whole-answer quotes and claim origins, failing closed.

    Raises :class:`QuoteOriginError` (a :class:`ValueError`) on any
    invented book quotation, unanchored user attribution, mixed-unit
    laundering, dangling/ambiguous quotation, or cross-source misuse
    (book quote sourced from a user message/capability/policy, or a user
    report citing an assistant message). Returns the serializable
    candidate-level artifact on success.
    """
    from aa.conversation.verifier_schema import VerifierValidationError

    text = answer if isinstance(answer, str) else ""
    extraction = extract_answer_quotes(text)
    unit_list = list(units or [])
    verdict_list = list(verdicts or [])
    verdict_by_id = {str(getattr(item, "unit_id", "")): item for item in verdict_list}
    trusted_users = list(user_index or [])
    overlaps = map_spans_to_units(tuple(extraction.spans), unit_list)

    if extraction.dangling:
        raise VerifierValidationError(
            "answer candidate carries unmatched quotation; failing closed"
        )

    provenances: list[SpanProvenance] = []
    book_chars = 0
    for span, overlap in zip(extraction.spans, overlaps, strict=True):
        span_text = span.span_text
        if not span_text.strip():
            continue
        unit_ids = list(overlap.unit_ids)
        if not unit_ids:
            raise VerifierValidationError("quoted span maps to no response unit; failing closed")
        # Every overlapped unit must agree on a compatible origin: a mixed
        # unit hiding a new book claim inside a user_report fails closed.
        # Deterministic rule: when several units overlap one span, the
        # span is validated under the strictest overlapped origin (book
        # first), so mixed coverage can never launder unsupported content.
        origins = [origin_for_verdict(verdict_by_id.get(uid)) for uid in unit_ids]
        if "book_claim" in origins:
            effective = "book_claim"
            effective_units = [
                uid for uid, org in zip(unit_ids, origins, strict=True) if org == "book_claim"
            ]
        else:
            effective = origins[0]
            effective_units = [unit_ids[0]]
        if effective == "book_claim":
            cited: list[str] = []
            for uid in effective_units:
                verdict = verdict_by_id.get(uid)
                if verdict is None:
                    raise VerifierValidationError(f"missing verdict for {uid!r}")
                if not bool(getattr(verdict, "supported", False)):
                    continue
                for item in list(getattr(verdict, "evidence_passage_ids", ()) or ()):
                    if isinstance(item, str) and item.strip():
                        cited.append(item.strip())
            if not cited:
                raise VerifierValidationError(
                    "book-origin quote without cited exact passages; failing closed"
                )
            anchor = anchor_book_span(span_text, passages, cited_passage_ids=cited)
            if anchor is None:
                raise VerifierValidationError(
                    "book-origin quote absent verbatim from cited passages; failing closed"
                )
            # A cited book quote must be sourced from canonical book
            # passages, never from a user message, product capability or
            # safety policy carried in origin_ref.
            for uid in effective_units:
                ref = _verdict_origin_ref(verdict_by_id.get(uid))
                if not ref:
                    continue
                kind = str(ref.get("kind", "") or "")
                if kind and kind != "book":
                    raise VerifierValidationError(
                        f"book-origin quote for {uid!r} mis-sourced; failing closed"
                    )
            book_chars += len(span_text)
            provenances.append(
                SpanProvenance(
                    span_text_sha256=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    answer_char_start=span.answer_char_start,
                    answer_char_end=span.answer_char_end,
                    answer_utf8_start=span.answer_utf8_start,
                    answer_utf8_end=span.answer_utf8_end,
                    origin="book_claim",
                    unit_ids=unit_ids,
                    origin_ref_type="book",
                    book_source_sha256=anchor.source_sha256,
                    book_section_id=anchor.section_id,
                    book_source_id=anchor.source_id,
                    book_source_char_start=anchor.source_char_start,
                    book_source_char_end=anchor.source_char_end,
                    book_passage_ids=list(anchor.passage_ids),
                    book_corpus_version=anchor.corpus_version,
                )
            )
        elif effective == "user_report":
            for uid in unit_ids:
                verdict = verdict_by_id.get(uid)
                if verdict is None:
                    raise VerifierValidationError(f"missing verdict for {uid!r}")
                # A mixed unit containing a new book claim must never pass
                # as a harmless user_report just because it repeats user
                # text: any book evidence requirement or book citation on
                # a user_report unit fails closed.
                if bool(getattr(verdict, "requires_book_evidence", False)):
                    raise VerifierValidationError(
                        f"unit {uid!r} mixes user report with book evidence; failing closed"
                    )
                if list(getattr(verdict, "evidence_passage_ids", ()) or ()):
                    raise VerifierValidationError(
                        f"user report {uid!r} must not cite book passages; failing closed"
                    )
                ref = _verdict_origin_ref(verdict)
                if ref:
                    kind = str(ref.get("kind", "") or "")
                    if kind and kind != "user_message":
                        raise VerifierValidationError(
                            f"user report {uid!r} mis-sourced; failing closed"
                        )
                    role = str(ref.get("role", "") or "")
                    if role and role != "human":
                        raise VerifierValidationError(
                            f"user report {uid!r} cites non-human role; failing closed"
                        )
            user_ref = anchor_user_span(span_text, trusted_users)
            if user_ref is None:
                raise VerifierValidationError(
                    "user-attributed quote without matching HumanMessage; failing closed"
                )
            provenances.append(
                SpanProvenance(
                    span_text_sha256=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    answer_char_start=span.answer_char_start,
                    answer_char_end=span.answer_char_end,
                    answer_utf8_start=span.answer_utf8_start,
                    answer_utf8_end=span.answer_utf8_end,
                    origin="user_report",
                    unit_ids=unit_ids,
                    origin_ref_type="user_message",
                    user_message_id=user_ref.message_id,
                    user_char_start=user_ref.char_start,
                    user_char_end=user_ref.char_end,
                )
            )
        else:
            # Glue, capability and safety outcomes carry no substantive
            # quotation: any span that is verbatim book text here, or any
            # quotation without a trusted referent, fails closed instead
            # of passing as harmless glue. Model-led origin classification
            # already ran; this deterministic gate binds it to offsets and
            # roles. No minimum-length exception: ambiguity fails closed.
            book_hit = anchor_book_span(span_text, passages)
            if book_hit is not None:
                raise VerifierValidationError(
                    "non-book unit carries verbatim book quotation; failing closed"
                )
            user_hit = anchor_user_span(span_text, trusted_users)
            if user_hit is None:
                # Quotations in glue/capability/safety units without any
                # trusted referent (neither book citation nor user message)
                # are ambiguous and fail closed; they must be reclassified
                # with provenance instead of passing silently.
                raise VerifierValidationError(
                    "non-book unit carries unattributed quotation; failing closed"
                )
            provenances.append(
                SpanProvenance(
                    span_text_sha256=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    answer_char_start=span.answer_char_start,
                    answer_char_end=span.answer_char_end,
                    answer_utf8_start=span.answer_utf8_start,
                    answer_utf8_end=span.answer_utf8_end,
                    origin=effective,
                    unit_ids=unit_ids,
                    origin_ref_type=("user_message" if user_hit is not None else "none"),
                    user_message_id=user_hit.message_id if user_hit is not None else "",
                    user_char_start=user_hit.char_start if user_hit is not None else 0,
                    user_char_end=user_hit.char_end if user_hit is not None else 0,
                )
            )

    # Units classified as user_report without any quoted span still need a
    # trusted referent when they repeat user content: a bare attribution
    # claim ("you said ...") must resolve to a HumanMessage slice, while
    # pure glue questions without attribution pass through.
    for verdict in verdict_list:
        if origin_for_verdict(verdict) != "user_report":
            continue
        unit_id = str(getattr(verdict, "unit_id", ""))
        unit = next((u for u in unit_list if str(getattr(u, "unit_id", "")) == unit_id), None)
        if unit is None:
            raise VerifierValidationError(f"missing unit for {unit_id!r}")
        unit_text = str(getattr(unit, "text", "") or "")
        covered = any(unit_id in prov.unit_ids for prov in provenances)
        if covered:
            continue
        # Unquoted user_report units pass only when they carry no
        # unattributed book-verbatim content. A unit that anchors verbatim
        # to a trusted HumanMessage is a faithful report even when the
        # same words appear in the book (the user said them); otherwise a
        # book-verbatim unit fails closed instead of passing as a
        # harmless report.
        book_hit = anchor_book_span(unit_text, passages)
        if book_hit is not None:
            if anchor_user_span(unit_text, trusted_users) is None:
                raise VerifierValidationError(
                    f"user report {unit_id!r} carries unattributed book text; failing closed"
                )

    total_chars = sum(len(span.span_text) for span in extraction.spans if span.span_text.strip())
    return AnswerQuoteCertificate(
        parser_version=QUOTE_PARSER_VERSION,
        answer_sha256=extraction.answer_sha256,
        answer_chars=extraction.answer_chars,
        answer_utf8_len=extraction.answer_utf8_len,
        total_quote_chars=total_chars,
        book_quote_chars=book_chars,
        spans=provenances,
        dangling=[
            {
                "answer_char_start": item.answer_char_start,
                "answer_char_end": item.answer_char_end,
                "detail": item.detail,
            }
            for item in extraction.dangling
        ],
        overlap_units=[list(item.unit_ids) for item in overlaps],
        passed=True,
        failure_code="",
    )


def book_quote_chars_for_answer(
    answer: str,
    certificate: AnswerQuoteCertificate | None = None,
) -> int:
    """Return origin-specific verbatim-book quote chars for ``answer``.

    When a certified artifact for the exact same candidate hash is
    supplied, only verified ``book_claim`` spans count: faithful
    ``user_report`` quotations of a person's own messages are exempt from
    the book copyright quota. Without a trusted certificate (ambiguity or
    missing source mapping), the total counts fail closed so an apparent
    book quote can never evade the quota by masquerading as a user
    report.
    """
    text = answer if isinstance(answer, str) else ""
    if certificate is not None:
        try:
            if certificate.answer_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest():
                return int(certificate.book_quote_chars)
        except (AttributeError, TypeError, ValueError):
            pass
    return total_quoted_chars(text)


__all__ = [
    "MAX_QUOTED_SPAN_CHARS",
    "ORIGINS",
    "QUOTE_PARSER_VERSION",
    "AnswerQuoteCertificate",
    "AnswerQuoteExtraction",
    "BookSourceAnchor",
    "DanglingQuote",
    "ExtractedQuote",
    "OriginName",
    "QuoteOriginError",
    "SpanOverlap",
    "SpanProvenance",
    "UserMessageRef",
    "anchor_book_span",
    "anchor_user_span",
    "book_quote_chars_for_answer",
    "build_user_message_index",
    "certify_answer_candidate",
    "contiguous_source_runs",
    "extract_answer_quotes",
    "extract_quoted_span_texts",
    "map_spans_to_units",
    "origin_for_verdict",
    "total_quoted_chars",
]
