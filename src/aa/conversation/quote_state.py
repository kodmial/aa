"""Authoritative delivered-quote history for anti-corpus-export (kodmial/aa#312).

Only real verified-and-delivered book quotations update quote range
state. Read-but-undelivered Evidence Pack passages never count as
delivered quotes: paraphrases, user reports and conversation glue carry
no verbatim book history.

Each stored range carries only provenance and offsets (source identity,
exact quoted ``char_start``/``char_end`` of the quoted substring, never
corpus or user text). Delivery accounting runs at character-range
intersection precision against structured transport receipts, so a
partial multi-segment send records only the confirmed delivered prefix
and ambiguous confirmations keep a distinct ``possibly-delivered``
safety record without lying about confirmed delivery.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

MAX_STORED_RANGES = 8
ADJACENCY_GAP_CHARS = 400

CONFIRMED = "confirmed"
POSSIBLY_DELIVERED = "possibly-delivered"


def _as_int(value: object, default: int = 0) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default


def ranges_from_pack(pack_dicts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Legacy read-range extractor (deprecated, never quote history).

    Retained only for backward import compatibility and for tests that
    pin the old shape. Production quote history must use
    :func:`ranges_from_delivered_quotes` after confirmed delivery;
    whole-pack reads must never silently count as delivered quotes.
    """
    ranges: list[dict[str, Any]] = []
    for item in pack_dicts:
        passage_id = item.get("passage_id")
        source_id = item.get("source_id", item.get("source", ""))
        section_id = item.get("section_id", item.get("section", ""))
        if not isinstance(passage_id, str) or not passage_id:
            continue
        if not isinstance(source_id, str) or not source_id:
            continue
        if not isinstance(section_id, str) or not section_id:
            continue
        ranges.append(
            {
                "passage_id": passage_id,
                "source_id": source_id,
                "section_id": section_id,
                "char_start": _as_int(item.get("char_start", 0)),
                "char_end": _as_int(item.get("char_end", 0)),
            }
        )
    return ranges


def _normalize_range(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one stored range; ``None`` when it carries no provenance."""
    if not isinstance(item, dict):
        return None
    source_id = str(item.get("source_id", item.get("source", "")) or "").strip()
    section_id = str(item.get("section_id", item.get("section", "")) or "").strip()
    start = _as_int(item.get("char_start", 0))
    end = _as_int(item.get("char_end", 0))
    if not source_id or not section_id or end <= start:
        return None
    source_sha = str(item.get("source_sha256", "") or "")
    corpus_version = str(item.get("corpus_version", "") or "")
    passage_ids: list[str] = []
    raw_ids = item.get("passage_ids", None)
    if isinstance(raw_ids, list):
        for raw in raw_ids:
            clean = str(raw or "").strip()
            if clean and clean not in passage_ids:
                passage_ids.append(clean)
    passage_id = str(item.get("passage_id", "") or "").strip()
    if not passage_id and passage_ids:
        passage_id = passage_ids[0]
    if passage_id and passage_id not in passage_ids:
        passage_ids = [passage_id, *passage_ids]
    status = str(item.get("delivery_status", CONFIRMED) or CONFIRMED).strip()
    if status not in (CONFIRMED, POSSIBLY_DELIVERED):
        status = CONFIRMED
    certificate_id = str(item.get("certificate_id", "") or "")
    out: dict[str, Any] = {
        "source_id": source_id,
        "section_id": section_id,
        "source_sha256": source_sha,
        "char_start": start,
        "char_end": end,
        "delivery_status": status,
    }
    if passage_id:
        out["passage_id"] = passage_id
    if passage_ids:
        out["passage_ids"] = passage_ids
    if corpus_version:
        out["corpus_version"] = corpus_version
    if certificate_id:
        out["certificate_id"] = certificate_id
    span_sha = str(item.get("span_text_sha256", "") or "")
    if span_sha:
        out["span_text_sha256"] = span_sha
    answer_start = item.get("answer_char_start", None)
    answer_end = item.get("answer_char_end", None)
    if isinstance(answer_start, int) and isinstance(answer_end, int) and answer_end > answer_start:
        out["answer_char_start"] = answer_start
        out["answer_char_end"] = answer_end
    return out


def _same_source(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether two normalized ranges share one canonical source identity."""
    if first.get("source_id") != second.get("source_id"):
        return False
    if first.get("section_id") != second.get("section_id"):
        return False
    first_sha = str(first.get("source_sha256", "") or "")
    second_sha = str(second.get("source_sha256", "") or "")
    if first_sha and second_sha and first_sha != second_sha:
        return False
    return True


def _intervals_overlap_or_touch(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Whether quoted offsets overlap or directly abut (no gap)."""
    start = int(first.get("char_start", 0))
    end = int(first.get("char_end", 0))
    other_start = int(second.get("char_start", 0))
    other_end = int(second.get("char_end", 0))
    return start <= other_end and end >= other_start


def merge_recent_ranges(
    previous: list[dict[str, Any]], new_ranges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Merge delivered quote spans into bounded recent history.

    Deduplicates by source identity AND actual overlapping/adjacent
    quoted offsets (never passage-id-only): identical spans are
    idempotent, overlapping/touching spans from one passage coalesce
    into their union, and disjoint spans are preserved separately.
    A ``confirmed`` record upgrades an identical ``possibly-delivered``
    one; a duplicate ``possibly-delivered`` never downgrades confirmed
    history. Bounded to the most recent ``MAX_STORED_RANGES`` entries.
    """
    merged: list[dict[str, Any]] = []
    for item in list(previous or []):
        normalized = _normalize_range(item) if isinstance(item, dict) else None
        if normalized is not None:
            merged.append(normalized)
    for item in list(new_ranges or []):
        if not isinstance(item, dict):
            continue
        normalized = _normalize_range(item)
        if normalized is None:
            continue
        # Exact idempotency (same source span): upgrade possibly->confirmed.
        upgraded = False
        duplicate = False
        for existing in merged:
            if not _same_source(existing, normalized):
                continue
            if int(existing.get("char_start", 0)) == int(normalized.get("char_start", 0)) and int(
                existing.get("char_end", 0)
            ) == int(normalized.get("char_end", 0)):
                duplicate = True
                if (
                    existing.get("delivery_status") == POSSIBLY_DELIVERED
                    and normalized.get("delivery_status") == CONFIRMED
                ):
                    existing["delivery_status"] = CONFIRMED
                    if normalized.get("certificate_id"):
                        existing["certificate_id"] = normalized["certificate_id"]
                    existing_ids = list(existing.get("passage_ids", []) or [])
                    for pid in list(normalized.get("passage_ids", []) or []):
                        if pid not in existing_ids:
                            existing_ids.append(pid)
                    if existing_ids:
                        existing["passage_ids"] = existing_ids
                        if not existing.get("passage_id"):
                            existing["passage_id"] = existing_ids[0]
                upgraded = True
                break
        if duplicate:
            # Move the touched entry to the tail (most-recent) for the
            # bound without duplicating it.
            touched = next(
                item_
                for item_ in merged
                if _same_source(item_, normalized)
                and int(item_.get("char_start", 0)) == int(normalized.get("char_start", 0))
                and int(item_.get("char_end", 0)) == int(normalized.get("char_end", 0))
            )
            merged.remove(touched)
            merged.append(touched)
            _ = upgraded
            continue
        # Coalesce overlapping/touching spans on the same source.
        coalesced = False
        for existing in merged:
            if not _same_source(existing, normalized):
                continue
            if _intervals_overlap_or_touch(existing, normalized):
                start = min(int(existing["char_start"]), int(normalized["char_start"]))
                end = max(int(existing["char_end"]), int(normalized["char_end"]))
                existing["char_start"] = start
                existing["char_end"] = end
                if normalized.get("delivery_status") == CONFIRMED:
                    existing["delivery_status"] = CONFIRMED
                existing_ids = list(existing.get("passage_ids", []) or [])
                for pid in list(normalized.get("passage_ids", []) or []):
                    if pid not in existing_ids:
                        existing_ids.append(pid)
                if existing_ids:
                    existing["passage_ids"] = existing_ids
                    if not existing.get("passage_id"):
                        existing["passage_id"] = existing_ids[0]
                if normalized.get("certificate_id") and not existing.get("certificate_id"):
                    existing["certificate_id"] = normalized["certificate_id"]
                # Move coalesced entry to the tail (most-recent).
                merged.remove(existing)
                merged.append(existing)
                coalesced = True
                break
        if coalesced:
            continue
        merged.append(dict(normalized))
    return merged[-MAX_STORED_RANGES:]


def is_adjacent_to_recent(candidate: dict[str, Any], recent: list[dict[str, Any]]) -> bool:
    """Whether ``candidate`` pages a recent delivered quote range."""
    source = str(candidate.get("source_id", candidate.get("source", "")) or "")
    section = str(candidate.get("section_id", candidate.get("section", "")) or "")
    start = _as_int(candidate.get("char_start", 0))
    end = _as_int(candidate.get("char_end", 0))
    if not source or not section or end <= start:
        return False
    for item in recent or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("source_id", "")) != source:
            continue
        if str(item.get("section_id", "")) != section:
            continue
        prev_sha = str(item.get("source_sha256", "") or "")
        cand_sha = str(candidate.get("source_sha256", "") or "")
        if prev_sha and cand_sha and prev_sha != cand_sha:
            continue
        prev_start = _as_int(item.get("char_start", 0))
        prev_end = _as_int(item.get("char_end", 0))
        if prev_end <= prev_start:
            continue
        # Overlap or near-contiguity in either direction pages the source.
        if start <= prev_end + ADJACENCY_GAP_CHARS and end >= prev_start - ADJACENCY_GAP_CHARS:
            return True
    return False


def pack_pages_recent(pack_dicts: list[dict[str, Any]], recent: list[dict[str, Any]]) -> bool:
    """Whether the new pack would continue a recent delivered quote range."""
    if not recent:
        return False
    for item in pack_dicts or []:
        if not isinstance(item, dict):
            continue
        candidate = {
            "source_id": str(item.get("source_id", item.get("source", ""))),
            "section_id": str(item.get("section_id", item.get("section", ""))),
            "source_sha256": str(item.get("source_sha256", "") or ""),
            "char_start": _as_int(item.get("char_start", 0)),
            "char_end": _as_int(item.get("char_end", 0)),
        }
        if is_adjacent_to_recent(candidate, recent):
            return True
    return False


def _confirmed_intervals(receipts: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return confirmed delivered char intervals from transport receipts."""
    intervals: list[tuple[int, int]] = []
    for item in receipts or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")) != CONFIRMED:
            continue
        try:
            start = int(item.get("char_start", 0) or 0)
            end = int(item.get("char_end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if end > start >= 0:
            intervals.append((start, end))
    return intervals


def _possibly_intervals(receipts: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Return ambiguous (unknown/failed-after-partial) intervals for safety."""
    intervals: list[tuple[int, int]] = []
    for item in receipts or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("status", "")) not in ("unknown", "failed"):
            continue
        try:
            start = int(item.get("char_start", 0) or 0)
            end = int(item.get("char_end", 0) or 0)
        except (TypeError, ValueError):
            continue
        if end > start >= 0:
            intervals.append((start, end))
    # Ambiguous intervals that are already fully covered by confirmed
    # delivery stay confirmed-only; safety history covers the rest.
    confirmed = _confirmed_intervals(receipts)
    out: list[tuple[int, int]] = []
    for start, end in intervals:
        covered = any(c_start <= start and end <= c_end for c_start, c_end in confirmed)
        if not covered:
            out.append((start, end))
    return out


def _intersect_lengths(
    span_start: int, span_end: int, intervals: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Intersect one answer span with delivered intervals (all overlaps)."""
    hits: list[tuple[int, int]] = []
    for start, end in intervals:
        overlap_start = max(span_start, start)
        overlap_end = min(span_end, end)
        if overlap_end > overlap_start:
            hits.append((overlap_start, overlap_end))
    hits.sort()
    return hits


def _intersect_length(
    span_start: int, span_end: int, intervals: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Intersect one answer span with delivered intervals (first overlap)."""
    hits = _intersect_lengths(span_start, span_end, intervals)
    return hits[0] if hits else None


def ranges_from_delivered_quotes(
    certificate: Any | None,
    *,
    certificate_id: str = "",
) -> list[dict[str, Any]]:
    """Build exact delivered-quote ranges from a #308 quote certificate.

    Only verified ``book_claim`` spans populate book-quote history with
    their exact ``source_sha256``/passage/source/section and the quoted
    substring ``char_start``-``char_end`` (never the whole read
    passage). ``user_report``/glue/capability/safety spans never count.
    Accepts an :class:`AnswerQuoteCertificate` or its JSON mapping.
    """
    if certificate is None:
        return []
    spans: list[Any] = []
    try:
        raw_spans = getattr(certificate, "spans", None)
        if raw_spans is None and isinstance(certificate, dict):
            raw_spans = certificate.get("spans", [])
        spans = list(raw_spans or [])
    except Exception:
        return []
    cert_id = certificate_id
    try:
        if not cert_id:
            maybe = getattr(certificate, "certificate_id", "")
            if isinstance(maybe, str):
                cert_id = maybe
            elif isinstance(certificate, dict):
                cert_id = str(certificate.get("certificate_id", "") or "")
    except Exception:
        cert_id = certificate_id
    ranges: list[dict[str, Any]] = []
    for span in spans:
        try:
            if isinstance(span, dict):
                origin = str(span.get("origin", "") or "")
                if origin != "book_claim":
                    continue
                source_id = str(span.get("book_source_id", "") or "")
                section_id = str(span.get("book_section_id", "") or "")
                source_sha = str(span.get("book_source_sha256", "") or "")
                start = int(span.get("book_source_char_start", 0) or 0)
                end = int(span.get("book_source_char_end", 0) or 0)
                passage_ids = [
                    str(pid) for pid in list(span.get("book_passage_ids", []) or []) if str(pid)
                ]
                corpus_version = str(span.get("book_corpus_version", "") or "")
                span_sha = str(span.get("span_text_sha256", "") or "")
                answer_start = span.get("answer_char_start", None)
                answer_end = span.get("answer_char_end", None)
            else:
                if str(getattr(span, "origin", "")) != "book_claim":
                    continue
                source_id = str(getattr(span, "book_source_id", "") or "")
                section_id = str(getattr(span, "book_section_id", "") or "")
                source_sha = str(getattr(span, "book_source_sha256", "") or "")
                start = int(getattr(span, "book_source_char_start", 0) or 0)
                end = int(getattr(span, "book_source_char_end", 0) or 0)
                raw_pids = list(getattr(span, "book_passage_ids", []) or [])
                passage_ids = [str(pid) for pid in raw_pids if str(pid)]
                corpus_version = str(getattr(span, "book_corpus_version", "") or "")
                span_sha = str(getattr(span, "span_text_sha256", "") or "")
                answer_start = getattr(span, "answer_char_start", None)
                answer_end = getattr(span, "answer_char_end", None)
        except (TypeError, ValueError):
            continue
        if not source_id or not section_id or not source_sha or end <= start:
            continue
        entry: dict[str, Any] = {
            "source_id": source_id,
            "section_id": section_id,
            "source_sha256": source_sha,
            "char_start": start,
            "char_end": end,
            "delivery_status": CONFIRMED,
        }
        if passage_ids:
            entry["passage_ids"] = passage_ids
            entry["passage_id"] = passage_ids[0]
        if corpus_version:
            entry["corpus_version"] = corpus_version
        if cert_id:
            entry["certificate_id"] = cert_id
        if span_sha:
            entry["span_text_sha256"] = span_sha
        if (
            isinstance(answer_start, int)
            and isinstance(answer_end, int)
            and answer_end > answer_start
        ):
            entry["answer_char_start"] = answer_start
            entry["answer_char_end"] = answer_end
        normalized = _normalize_range(entry)
        if normalized is not None:
            ranges.append(normalized)
    return ranges


def delivered_ranges_for_receipts(
    certificate: Any | None,
    *,
    certified_text: str = "",
    receipts: list[dict[str, Any]] | None = None,
    certificate_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split verified book quotes into confirmed vs possibly-delivered ranges.

    Intersection runs at character-range precision over the certified
    text: a quoted span crossing transport segments records only the
    confirmed delivered prefix (source offsets scale with the delivered
    answer prefix). Ambiguous ``unknown``/``failed`` segments produce a
    separate safety record so retries cannot page contiguous text while
    confirmed history never fabricates an undelivered suffix.
    """
    full_ranges = ranges_from_delivered_quotes(certificate, certificate_id=certificate_id)
    receipt_list = [dict(item) for item in (receipts or []) if isinstance(item, dict)]
    confirmed_intervals = _confirmed_intervals(receipt_list)
    possibly_intervals = _possibly_intervals(receipt_list)
    _ = certified_text
    if not full_ranges or (not confirmed_intervals and not possibly_intervals):
        return [], []
    confirmed: list[dict[str, Any]] = []
    possibly: list[dict[str, Any]] = []
    for entry in full_ranges:
        answer_start = entry.get("answer_char_start", None)
        answer_end = entry.get("answer_char_end", None)
        if not isinstance(answer_start, int) or not isinstance(answer_end, int):
            # Without answer offsets delivery precision is unprovable:
            # never fabricate a confirmed span; keep a safety record only
            # when some delivery actually happened.
            if confirmed_intervals:
                guarded = dict(entry)
                guarded["delivery_status"] = POSSIBLY_DELIVERED
                normalized = _normalize_range(guarded)
                if normalized is not None:
                    possibly.append(normalized)
            continue
        span_len = answer_end - answer_start
        source_start = int(entry.get("char_start", 0))
        source_end = int(entry.get("char_end", 0))
        if span_len <= 0 or source_end <= source_start:
            continue
        confirmed_hits = _intersect_lengths(answer_start, answer_end, confirmed_intervals)
        for confirmed_hit in confirmed_hits:
            delivered_prefix = max(0, confirmed_hit[1] - answer_start)
            delivered_prefix = min(delivered_prefix, span_len)
            # Contiguous delivered prefix from the span start: a confirmed
            # suffix without the opening characters still proves only the
            # overlapping slice, mapped proportionally onto the source.
            overlap_start_offset = max(0, confirmed_hit[0] - answer_start)
            overlap_len = confirmed_hit[1] - confirmed_hit[0]
            mapped_start = source_start + overlap_start_offset
            mapped_end = mapped_start + overlap_len
            mapped_end = min(mapped_end, source_end)
            if mapped_end > mapped_start:
                record = dict(entry)
                record["char_start"] = mapped_start
                record["char_end"] = mapped_end
                record["delivery_status"] = CONFIRMED
                normalized = _normalize_range(record)
                if normalized is not None:
                    confirmed.append(normalized)
            # A partially delivered span may still have an ambiguous
            # remainder; the safety record below covers it without
            # claiming it as confirmed.
            _ = delivered_prefix
        possibly_hits = _intersect_lengths(answer_start, answer_end, possibly_intervals)
        for possibly_hit in possibly_hits:
            overlap_start_offset = max(0, possibly_hit[0] - answer_start)
            overlap_len = possibly_hit[1] - possibly_hit[0]
            mapped_start = source_start + overlap_start_offset
            mapped_end = min(mapped_start + overlap_len, source_end)
            if mapped_end > mapped_start:
                record = dict(entry)
                record["char_start"] = mapped_start
                record["char_end"] = mapped_end
                record["delivery_status"] = POSSIBLY_DELIVERED
                normalized = _normalize_range(record)
                if normalized is not None:
                    possibly.append(normalized)
    return confirmed, possibly


def commit_delivery(
    previous: list[dict[str, Any]],
    certificate: Any | None,
    *,
    certified_text: str = "",
    receipts: list[dict[str, Any]] | None = None,
    certificate_id: str = "",
) -> list[dict[str, Any]]:
    """Commit only receipt-confirmed (plus ambiguous-safety) quote ranges.

    Idempotent over replays/retries: identical spans deduplicate and
    confirmed upgrades possibly-delivered without duplication. Failed
    sends with no confirmed segments commit nothing.
    """
    confirmed, possibly = delivered_ranges_for_receipts(
        certificate,
        certified_text=certified_text,
        receipts=receipts,
        certificate_id=certificate_id,
    )
    merged = merge_recent_ranges(previous, confirmed)
    merged = merge_recent_ranges(merged, possibly)
    return merged


def delivered_ranges_from_candidate(
    *,
    certified_text: str,
    evidence_pack: list[dict[str, Any]],
    claim_verdicts: list[dict[str, Any]],
    stored_unit_texts: list[dict[str, Any]] | None = None,
    receipts: list[dict[str, Any]] | None = None,
    certificate_id: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Derive receipt-scoped book ranges without trusting attempted sends.

    Splits the exact certified text, maps each #308 extracted quote to
    its verifier units, keeps only ``book_claim``-scoped spans anchored
    verbatim to the cited evidence, and intersects with confirmed vs
    ambiguous receipt intervals. ``user_report``/glue spans never count,
    even when their words also occur in the book. Never raises; empty
    means no provable book delivery.
    """
    try:
        from aa.conversation.quote_provenance import anchor_book_span, extract_answer_quotes
        from aa.conversation.response_units import split_response_units
    except Exception:
        return [], []
    try:
        text = certified_text if isinstance(certified_text, str) else ""
        if not text.strip():
            return [], []
        extraction = extract_answer_quotes(text)
        if not extraction.spans:
            return [], []
        try:
            units = split_response_units(text)
        except Exception:
            return [], []
        verdict_by_text: dict[str, dict[str, Any]] = {}
        verdict_by_id: dict[str, dict[str, Any]] = {}
        for item in list(claim_verdicts or []):
            if isinstance(item, dict) and str(item.get("unit_id", "")):
                verdict_by_id[str(item.get("unit_id", ""))] = item
        if stored_unit_texts:
            for entry in list(stored_unit_texts or []):
                if not isinstance(entry, dict):
                    continue
                unit_id = str(entry.get("unit_id", ""))
                unit_text = str(entry.get("text", ""))
                if unit_id and unit_text and unit_id in verdict_by_id:
                    verdict_by_text[unit_text] = verdict_by_id[unit_id]
                    verdict_by_text[" ".join(unit_text.split())] = verdict_by_id[unit_id]
        pack = [dict(item) for item in (evidence_pack or []) if isinstance(item, dict)]
        receipt_list = [dict(item) for item in (receipts or []) if isinstance(item, dict)]
        confirmed_intervals = _confirmed_intervals(receipt_list)
        possibly_intervals = _possibly_intervals(receipt_list)
        if not confirmed_intervals and not possibly_intervals:
            return [], []
        confirmed: list[dict[str, Any]] = []
        possibly: list[dict[str, Any]] = []
        for span in extraction.spans:
            span_text = span.span_text
            if not span_text.strip():
                continue
            overlapped = [
                unit
                for unit in units
                if span.answer_char_start < unit.char_end and span.answer_char_end > unit.char_start
            ]
            if not overlapped:
                continue
            # Strictest overlapped origin wins; mixed book coverage can
            # never launder through a glue/user unit.
            origins: list[str] = []
            for unit in overlapped:
                verdict: dict[str, Any] | None = None
                if verdict_by_text:
                    verdict = verdict_by_text.get(unit.text) or verdict_by_text.get(
                        " ".join(unit.text.split())
                    )
                if verdict is None:
                    # Without stored texts fall back positionally only
                    # when counts align exactly; otherwise fail closed.
                    try:
                        idx = units.index(unit)
                    except ValueError:
                        idx = -1
                    ordered = [verdict_by_id[key] for key in sorted(verdict_by_id.keys())]
                    if 0 <= idx < len(ordered) and len(ordered) == len(units):
                        verdict = ordered[idx]
                scope = str((verdict or {}).get("scope", "") or "")
                origin = str((verdict or {}).get("origin", "") or "")
                if scope == "book":
                    origins.append("book_claim")
                elif origin:
                    origins.append(origin)
                else:
                    origins.append("conversation_glue")
            if not origins or any(origin != "book_claim" for origin in origins):
                continue
            cited: list[str] = []
            for unit in overlapped:
                verdict = None
                if verdict_by_text:
                    verdict = verdict_by_text.get(unit.text) or verdict_by_text.get(
                        " ".join(unit.text.split())
                    )
                if verdict is None:
                    continue
                if str(verdict.get("scope", "")) != "book" or not bool(
                    verdict.get("supported", False)
                ):
                    continue
                for cited_id in list(verdict.get("evidence_passage_ids", []) or []):
                    if isinstance(cited_id, str) and cited_id.strip():
                        cited.append(cited_id.strip())
            if not cited:
                continue
            anchor = anchor_book_span(span_text, pack, cited_passage_ids=cited)
            if anchor is None:
                continue
            source_len = len(span_text)
            if source_len <= 0:
                continue
            confirmed_hits = _intersect_lengths(
                span.answer_char_start, span.answer_char_end, confirmed_intervals
            )
            for confirmed_hit in confirmed_hits:
                overlap_start = max(0, confirmed_hit[0] - span.answer_char_start)
                overlap_len = confirmed_hit[1] - confirmed_hit[0]
                mapped_start = anchor.source_char_start + overlap_start
                mapped_end = min(mapped_start + overlap_len, anchor.source_char_end)
                if mapped_end > mapped_start:
                    record = {
                        "source_id": anchor.source_id,
                        "section_id": anchor.section_id,
                        "source_sha256": anchor.source_sha256,
                        "char_start": mapped_start,
                        "char_end": mapped_end,
                        "passage_id": anchor.passage_ids[0] if anchor.passage_ids else "",
                        "passage_ids": list(anchor.passage_ids),
                        "corpus_version": anchor.corpus_version,
                        "delivery_status": CONFIRMED,
                        "answer_char_start": span.answer_char_start,
                        "answer_char_end": span.answer_char_end,
                        "span_text_sha256": hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    }
                    if certificate_id:
                        record["certificate_id"] = certificate_id
                    normalized = _normalize_range(record)
                    if normalized is not None:
                        confirmed.append(normalized)
            possibly_hits = _intersect_lengths(
                span.answer_char_start, span.answer_char_end, possibly_intervals
            )
            for possibly_hit in possibly_hits:
                overlap_start = max(0, possibly_hit[0] - span.answer_char_start)
                overlap_len = possibly_hit[1] - possibly_hit[0]
                mapped_start = anchor.source_char_start + overlap_start
                mapped_end = min(mapped_start + overlap_len, anchor.source_char_end)
                if mapped_end > mapped_start:
                    record = {
                        "source_id": anchor.source_id,
                        "section_id": anchor.section_id,
                        "source_sha256": anchor.source_sha256,
                        "char_start": mapped_start,
                        "char_end": mapped_end,
                        "passage_id": anchor.passage_ids[0] if anchor.passage_ids else "",
                        "passage_ids": list(anchor.passage_ids),
                        "corpus_version": anchor.corpus_version,
                        "delivery_status": POSSIBLY_DELIVERED,
                        "answer_char_start": span.answer_char_start,
                        "answer_char_end": span.answer_char_end,
                        "span_text_sha256": hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
                    }
                    if certificate_id:
                        record["certificate_id"] = certificate_id
                    normalized = _normalize_range(record)
                    if normalized is not None:
                        possibly.append(normalized)
        return confirmed, possibly
    except Exception:
        return [], []


def quote_ranges_digest(ranges: list[dict[str, Any]]) -> str:
    """Return a privacy-safe digest of stored ranges (no text)."""
    canonical: list[dict[str, Any]] = []
    for item in list(ranges or []):
        if not isinstance(item, dict):
            continue
        canonical.append(
            {
                "source_id": str(item.get("source_id", "")),
                "section_id": str(item.get("section_id", "")),
                "source_sha256": str(item.get("source_sha256", ""))[:16],
                "char_start": _as_int(item.get("char_start", 0)),
                "char_end": _as_int(item.get("char_end", 0)),
                "delivery_status": str(item.get("delivery_status", CONFIRMED)),
            }
        )
    canonical.sort(
        key=lambda entry: (
            str(entry.get("source_id", "")),
            str(entry.get("section_id", "")),
            int(entry.get("char_start", 0)),
            int(entry.get("char_end", 0)),
        )
    )
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "ADJACENCY_GAP_CHARS",
    "CONFIRMED",
    "MAX_STORED_RANGES",
    "POSSIBLY_DELIVERED",
    "commit_delivery",
    "delivered_ranges_for_receipts",
    "delivered_ranges_from_candidate",
    "is_adjacent_to_recent",
    "merge_recent_ranges",
    "pack_pages_recent",
    "quote_ranges_digest",
    "ranges_from_delivered_quotes",
    "ranges_from_pack",
]
