"""Trusted #295 production-selector real-book diagnostics (issue #295).

Separately from the frozen #130 RRF-only benchmark in
:mod:`aa.qualification.real_book_retrieval`, this module exercises the
ACTUAL production selection path
(:func:`aa.conversation.retrieval_node.aretrieve_with_semantic_selection`
with a real ``OpenCodeChatModel`` selector) over the canonical RU
E5/BM25/FAISS substrate.

Privacy contract: diagnostics carry counts/ranks/hashes/latencies only.
Exact book text never leaves the trusted job except inside the
compressed + age-encrypted bundle. Public summaries are metrics-only and
validated by :func:`assert_public_summary_safe`.

Status contract: ``PASS`` / ``FAIL`` / ``INCOMPLETE`` / ``STALE``.
A run that never observes a deep-rank decisive selection (fused rank
>16 selected with essential source text beyond char 500 reaching full
fidelity generation/verifier inputs) is ``INCOMPLETE``, never ``PASS``.
Missing selector/model/E5 prerequisites yield ``INCOMPLETE`` with
``selector_available=False`` / fallback flags, never a fake PASS.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from aa.qualification.real_book_retrieval import (
    RealBookRetrievalError,
    assert_no_oracle_leak,
    assert_public_summary_safe,
    assert_source_exact,
    validate_exact_sha,
)

SCHEMA_VERSION = "aa-real-book-production-selection-295/1"
PUBLIC_SUMMARY_VERSION = "aa-real-book-production-selection-295-summary/1"
PROTECTED_ARTIFACT_VERSION = "aa-real-book-production-selection-295-protected/1"
RESULT_ISSUE = 295
VALID_STATUSES = ("PASS", "FAIL", "INCOMPLETE", "STALE")

# Deep-rank thresholds required by #295 acceptance.
DEEP_RANK_GT5 = 5
DEEP_RANK_GT16 = 16
# Essential information must live beyond the former 500-char cut.
FULL_TEXT_BEYOND_CHARS = 500


@dataclass(frozen=True)
class ProductionSelectionDiagnostics:
    """Privacy-safe per-case #295 diagnostics (no book/user text)."""

    case_id: str
    planner_query_count: int
    fused_pool_unique: int
    preview_count: int
    selected_count: int
    selection_max_rank: int
    deep_rank_gt5: bool
    deep_rank_gt16: bool
    full_chars_total: int
    full_max_chars: int
    beyond_500_chars: bool
    source_sections_covered: int
    fidelity_ok: bool
    selector_available: bool
    fallback_used: bool
    need_more_detail: bool
    followup_added: int
    latency_ms: float
    selection_latency_ms: float
    had_429: bool
    oracle_hit: bool | None

    def public_row(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "planner_query_count": self.planner_query_count,
            "fused_pool_unique": self.fused_pool_unique,
            "preview_count": self.preview_count,
            "selected_count": self.selected_count,
            "selection_max_rank": self.selection_max_rank,
            "deep_rank_gt5": self.deep_rank_gt5,
            "deep_rank_gt16": self.deep_rank_gt16,
            "full_chars_total": self.full_chars_total,
            "full_max_chars": self.full_max_chars,
            "beyond_500_chars": self.beyond_500_chars,
            "source_sections_covered": self.source_sections_covered,
            "fidelity_ok": self.fidelity_ok,
            "selector_available": self.selector_available,
            "fallback_used": self.fallback_used,
            "need_more_detail": self.need_more_detail,
            "followup_added": self.followup_added,
            "latency_ms": round(self.latency_ms, 3),
            "selection_latency_ms": round(self.selection_latency_ms, 3),
            "had_429": self.had_429,
            "oracle_hit": self.oracle_hit,
        }

    def protected_record(self) -> dict[str, Any]:
        row = self.public_row()
        row["evidence_digest"] = ""
        return row


def _sections_of(index: Any, chunk_ids: list[str]) -> set[str]:
    out: set[str] = set()
    chunks = getattr(index, "chunks", {}) or {}
    for cid in chunk_ids:
        rec = chunks.get(cid)
        section = getattr(rec, "section", "") if rec is not None else ""
        if section:
            out.add(str(section))
    return out


def assert_full_evidence_fidelity(pack: Any, index: Any) -> tuple[int, int, int]:
    """Verify every selected passage carries exact full canonical text.

    Returns ``(total_chars, max_chars, section_count)``. Raises
    :class:`RealBookRetrievalError` on any checksum/offset mismatch or on
    empty evidence (fail-closed: never an invented successful coverage).
    """
    passages = list(getattr(pack, "passages", ()) or ())
    if not passages:
        raise RealBookRetrievalError("production selection yielded no evidence passages")
    chunks = getattr(index, "chunks", {}) or {}
    total = 0
    biggest = 0
    sections: set[str] = set()
    for passage in passages:
        text = str(getattr(passage, "exact_text", "") or "")
        if not text:
            raise RealBookRetrievalError("selected passage has empty exact text")
        child_ids = list(getattr(passage, "child_chunk_ids", ()) or ())
        if not child_ids:
            raise RealBookRetrievalError("selected passage has no child chunk ids")
        for cid in child_ids:
            rec = chunks.get(cid)
            if rec is None:
                raise RealBookRetrievalError(f"selected chunk is not indexed: {cid}")
            assert_source_exact(
                text=str(getattr(rec, "text", "")),
                text_sha256=str(getattr(rec, "text_sha256", "")),
                char_start=int(getattr(rec, "char_start", -1)),
                char_end=int(getattr(rec, "char_end", -1)),
            )
        # The pack text must contain each underlying canonical chunk text
        # verbatim (full fidelity, no 500-char cut at generation/verifier).
        for cid in child_ids:
            rec = chunks.get(cid)
            chunk_text = str(getattr(rec, "text", ""))
            if chunk_text and chunk_text not in text:
                raise RealBookRetrievalError(
                    "evidence pack text does not carry exact canonical chunk text"
                )
        total += len(text)
        biggest = max(biggest, len(text))
        section = str(getattr(passage, "section_id", "") or "")
        if section:
            sections.add(section)
    return total, biggest, len(sections)


async def run_production_selection_turn(
    *,
    index: Any,
    case_id: str,
    planner_queries: list[str],
    resolved_intent: str,
    conversation_context: str = "",
    user_message: str = "",
    selection_model: Any | None = None,
    oracle_sections: set[str] | None = None,
    config: Any | None = None,
) -> ProductionSelectionDiagnostics:
    """Run ONE case through the real production selection path.

    Always calls
    :func:`aa.conversation.retrieval_node.aretrieve_with_semantic_selection`
    (never the frozen RRF-only ``_run_turn``). Provider 429 propagates so
    the trusted runner can checkpoint/resume; any other failure raises
    :class:`RealBookRetrievalError` (fail-closed, never silent success).
    """
    from aa.conversation.retrieval_node import aretrieve_with_semantic_selection

    assert_no_oracle_leak({"queries": list(planner_queries)}, case_id)
    started = time.perf_counter()
    had_429 = False
    try:
        pack = await aretrieve_with_semantic_selection(
            index,
            list(planner_queries),
            config=config,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
            user_message=user_message,
            selection_model=selection_model,
        )
    except Exception as exc:
        from aa.opencode.errors import OpenCodeRateLimitError

        if isinstance(exc, OpenCodeRateLimitError):
            raise
        raise RealBookRetrievalError(
            f"production selection path failed: {type(exc).__name__}"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    meta = dict(getattr(pack, "retrieval_metadata", {}) or {})

    def _m(*names: str, default: Any = None) -> Any:
        for name in names:
            if name in meta:
                return meta[name]
        return default

    # Fail closed when the production path was bypassed: the real path
    # always stamps semantic-selection metadata (keys are double-prefixed
    # as selection_selection_* by retrieval_node; accept both spellings).
    if _m("selection_candidates", "selection_selection_candidates") is None:
        raise RealBookRetrievalError(
            "production selection metadata is missing; RRF-only path used?"
        )
    total_chars, max_chars, section_count = assert_full_evidence_fidelity(pack, index)
    max_rank = int(_m("selection_max_rank", "selection_selection_max_rank", default=-1))
    deep5 = bool(_m("selection_deep_rank_gt5", "selection_selection_deep_rank_gt5", default=False))
    deep16 = bool(
        _m("selection_deep_rank_gt16", "selection_selection_deep_rank_gt16", default=False)
    )
    # Cross-check rank flags against the reported max rank (never trust a
    # silently unpromoted run as deep coverage).
    if max_rank > DEEP_RANK_GT16 and not deep16:
        deep16 = True
    if max_rank > DEEP_RANK_GT5 and not deep5:
        deep5 = True
    selector_available = selection_model is not None
    # Heuristic fallback is observable: no model bound means fallback was used.
    fallback_used = selection_model is None
    oracle_hit: bool | None
    if not oracle_sections:
        oracle_hit = None
    else:
        pack_sections = _sections_of(
            index,
            [cid for p in getattr(pack, "passages", ()) for cid in (p.child_chunk_ids or ())],
        )
        oracle_hit = bool(pack_sections & set(oracle_sections))
    return ProductionSelectionDiagnostics(
        case_id=case_id,
        planner_query_count=len(planner_queries),
        fused_pool_unique=int(_m("fused_unique", default=0)),
        preview_count=int(_m("selection_candidates", "selection_selection_candidates", default=0)),
        selected_count=len(list(getattr(pack, "passages", ()) or ())),
        selection_max_rank=max_rank,
        deep_rank_gt5=deep5,
        deep_rank_gt16=deep16,
        full_chars_total=total_chars,
        full_max_chars=max_chars,
        beyond_500_chars=bool(max_chars > FULL_TEXT_BEYOND_CHARS),
        source_sections_covered=section_count,
        fidelity_ok=True,
        selector_available=selector_available,
        fallback_used=fallback_used,
        need_more_detail=bool(
            _m("selection_need_more", "selection_selection_need_more", default=False)
        ),
        followup_added=int(_m("followup_added", default=0)),
        latency_ms=elapsed_ms,
        selection_latency_ms=float(
            _m("selection_latency_ms", "selection_selection_latency_ms", default=0.0)
        ),
        had_429=had_429,
        oracle_hit=oracle_hit,
    )


def decide_status_295(
    *,
    stale: bool,
    incomplete: bool,
    failures: int,
    turns: list[ProductionSelectionDiagnostics],
) -> str:
    """Decide the deterministic #295 status.

    ``PASS`` requires at least one deep-rank decisive case: a selected
    fused rank >16 together with full source text beyond 500 chars at
    verified fidelity. Without such a case the run is ``INCOMPLETE``,
    never ``PASS`` (a benchmark that never reaches deep ranks proves
    nothing about #295).
    """
    if stale:
        return "STALE"
    if incomplete:
        return "INCOMPLETE"
    if failures < 0:
        raise RealBookRetrievalError("failure count must be >= 0")
    if failures > 0:
        return "FAIL"
    decisive = [
        t
        for t in turns
        if t.deep_rank_gt16 and t.beyond_500_chars and t.fidelity_ok and not t.had_429
    ]
    if not decisive:
        return "INCOMPLETE"
    return "PASS"


def summarize_public_295(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    turns: list[ProductionSelectionDiagnostics],
    status: str,
    run_id: str,
    selector_available: bool,
) -> dict[str, Any]:
    if status not in VALID_STATUSES:
        raise RealBookRetrievalError(f"invalid status {status!r}")
    latencies = sorted(t.latency_ms for t in turns)

    def _pct(p: float) -> float:
        if not latencies:
            return 0.0
        return round(latencies[min(len(latencies) - 1, int(p * len(latencies)))], 3)

    payload: dict[str, Any] = {
        "schema_version": PUBLIC_SUMMARY_VERSION,
        "issue": RESULT_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "retrieval_backend": "bm25+e5-faiss/rrf+production-semantic-selection",
        "result": status,
        "run_id": run_id,
        "selector_available": bool(selector_available),
        "turn_count": len(turns),
        "deep_gt5_count": sum(1 for t in turns if t.deep_rank_gt5),
        "deep_gt16_count": sum(1 for t in turns if t.deep_rank_gt16),
        "beyond_500_count": sum(1 for t in turns if t.beyond_500_chars),
        "decisive_count": sum(
            1
            for t in turns
            if t.deep_rank_gt16 and t.beyond_500_chars and t.fidelity_ok and not t.had_429
        ),
        "fallback_count": sum(1 for t in turns if t.fallback_used),
        "oracle_hit_count": sum(1 for t in turns if t.oracle_hit is True),
        "oracle_miss_count": sum(1 for t in turns if t.oracle_hit is False),
        "rate_limited_count": sum(1 for t in turns if t.had_429),
        "latency_p50_ms": _pct(0.5),
        "latency_p95_ms": _pct(0.95),
        "turns": [t.public_row() for t in turns],
    }
    assert_public_summary_safe(payload)
    return payload


def build_protected_payload_295(
    *,
    main_sha: str,
    corpus_sha: str,
    benchmark_sha: str,
    retrieval_sha: str,
    turns: list[ProductionSelectionDiagnostics],
    status: str,
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROTECTED_ARTIFACT_VERSION,
        "issue": RESULT_ISSUE,
        "main_sha": validate_exact_sha(main_sha),
        "corpus_sha256": corpus_sha,
        "benchmark_sha256": benchmark_sha,
        "retrieval_config_sha256": retrieval_sha,
        "result": status,
        "run_id": run_id,
        "turns": [t.protected_record() for t in turns],
    }


def selection_digest(chunk_ids: list[str]) -> str:
    return hashlib.sha256("|".join(chunk_ids).encode("utf-8")).hexdigest()[:16]


def run_async(coro: Any) -> Any:
    return asyncio.run(coro)


__all__ = [
    "DEEP_RANK_GT5",
    "DEEP_RANK_GT16",
    "FULL_TEXT_BEYOND_CHARS",
    "PROTECTED_ARTIFACT_VERSION",
    "PUBLIC_SUMMARY_VERSION",
    "RESULT_ISSUE",
    "SCHEMA_VERSION",
    "VALID_STATUSES",
    "ProductionSelectionDiagnostics",
    "assert_full_evidence_fidelity",
    "build_protected_payload_295",
    "decide_status_295",
    "run_production_selection_turn",
    "selection_digest",
    "summarize_public_295",
]
