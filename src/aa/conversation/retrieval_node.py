"""V2 retrieval node: planner queries to compact Evidence Pack (issue #116).

The node consumes the minimal ``search_queries`` produced by the #113
hidden planner (0 or 10..16 context-resolved Russian queries) and runs
the complete target pipeline from :mod:`aa.retrieval.evidence` over the
#115 RAM-resident canonical index:

- ``search_queries == []`` performs no retrieval and yields an empty
  pack for a purely conversational/glue turn;
- otherwise every query runs BM25 + E5/FAISS branches, global RRF,
  dedup/diversity, the pinned local BGE reranker, small-to-big
  expansion and atomic budget selection.

Only orchestration state is written; ``messages`` is left untouched.
State carries exact passage text plus minimal provenance. Ranking
metadata (RRF/BM25/dense/rerank scores, embeddings, planner reasoning,
search previews) stays in internal retrieval metadata and never enters
the user-facing prompt. Logs carry only routes, counts and token
lengths, never prompts or user text.

Performance note: the pinned BGE path records ~20-22s warm latency per
turn (per-turn RRF plus 64-candidate CPU rerank) against the 5s
interactive budget, so production cutover stays blocked until explicit
performance acceptance or optimization (see
``aa.qualification.v2_retrieval.require_v2_cutover_acceptance``).
Binding the node via :func:`make_retrieval_node` therefore requires
``performance_accepted=True`` and fails closed otherwise, so a slow
turn can never be wired into production without a recorded
acceptance.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from aa.conversation.graph_state import TurnState
from aa.conversation.prompt_builder import EvidencePassage
from aa.retrieval.evidence import (
    INTERACTIVE_LATENCY_BUDGET_MS,
    EvidencePack,
    RetrievalConfig,
    retrieve_evidence,
)
from aa.retrieval.index import HybridIndex, logical_chunk_id
from aa.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger("aa.conversation.retrieval_node")


def pack_to_state(pack: EvidencePack) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Map an Evidence Pack to graph state payloads (exact text + provenance)."""
    hits: list[dict[str, Any]] = []
    pack_dicts: list[dict[str, Any]] = []
    for passage in pack.passages:
        pack_dicts.append(
            {
                "passage_id": passage.passage_id,
                "text": passage.exact_text,
                "source_id": passage.source_id,
                "section_id": passage.section_id,
                "child_chunk_ids": list(passage.child_chunk_ids),
                "char_start": passage.char_start,
                "char_end": passage.char_end,
                "text_sha256": passage.text_sha256,
            }
        )
        for chunk_id in passage.child_chunk_ids:
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "logical_chunk_id": logical_chunk_id(chunk_id),
                    "section_id": passage.section_id,
                    "source_id": passage.source_id,
                    "passage_id": passage.passage_id,
                    "char_start": passage.char_start,
                    "char_end": passage.char_end,
                    "text_sha256": passage.text_sha256,
                }
            )
    return hits, pack_dicts


def state_passages_to_prompt(pack_dicts: list[dict[str, Any]]) -> list[EvidencePassage]:
    """Map stored state passages back to prompt-builder passages."""
    passages: list[EvidencePassage] = []
    for item in pack_dicts:
        text = item.get("text")
        passage_id = item.get("passage_id")
        source_id = item.get("source_id")
        section_id = item.get("section_id")
        if not isinstance(text, str) or not text:
            continue
        if not isinstance(passage_id, str) or not passage_id:
            continue
        passages.append(
            EvidencePassage(
                passage_id=passage_id,
                source=str(source_id),
                section=str(section_id),
                text=text,
            )
        )
    return passages


async def retrieval_node(
    state: TurnState,
    *,
    index: HybridIndex,
    reranker: CrossEncoderReranker | None = None,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    """LangGraph retrieval node: queries to hits plus Evidence Pack.

    Thread safety: concurrent turns share one long-lived ``index`` and
    one long-lived ``reranker`` across ``asyncio.to_thread`` workers.
    Shared access is internally serialized where the underlying
    libraries offer no cross-thread guarantee: lexical FTS via its
    connection lock, dense FAISS search via the dense search lock, and
    FlagReranker scoring via the reranker score lock. No caller-side
    locking is required.
    """
    raw_queries = state.get("search_queries", [])
    if isinstance(raw_queries, (list, tuple)):
        queries = [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
    else:
        queries = []
    if not queries:
        logger.info("v2 retrieval skipped", extra={"queries": 0})
        return {"retrieval_hits": [], "evidence_pack": []}
    started = time.perf_counter()
    pack = await asyncio.to_thread(
        retrieve_evidence, index, queries, config=config, reranker=reranker
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    over_budget = elapsed_ms > INTERACTIVE_LATENCY_BUDGET_MS
    hits, pack_dicts = pack_to_state(pack)
    if over_budget:
        logger.warning(
            "v2 retrieval over interactive budget",
            extra={
                "queries": len(queries),
                "latency_ms": round(elapsed_ms, 1),
                "budget_ms": INTERACTIVE_LATENCY_BUDGET_MS,
            },
        )
    logger.info(
        "v2 retrieval done",
        extra={
            "queries": len(queries),
            "hits": len(hits),
            "passages": len(pack_dicts),
            "tokens": pack.total_tokens,
            "latency_ms": round(elapsed_ms, 1),
            "over_budget": over_budget,
        },
    )
    return {"retrieval_hits": hits, "evidence_pack": pack_dicts}


def make_retrieval_node(
    *,
    index: HybridIndex,
    reranker: CrossEncoderReranker | None = None,
    config: RetrievalConfig | None = None,
    performance_accepted: bool = False,
) -> Any:
    """Build the evidence retrieval node bound to one RAM-resident index.

    ``performance_accepted=True`` records explicit acceptance of the
    ~20-22s warm BGE latency against the 5s interactive budget; the
    default (``False``) fails closed via
    ``aa.qualification.v2_retrieval.require_v2_cutover_acceptance``
    instead of wiring the slow path into production.
    """
    from aa.qualification.v2_retrieval import require_v2_cutover_acceptance

    require_v2_cutover_acceptance(performance_accepted=performance_accepted)

    async def run_evidence_retrieval(state: TurnState) -> dict[str, Any]:
        return await retrieval_node(state, index=index, reranker=reranker, config=config)

    return run_evidence_retrieval


__all__ = [
    "make_retrieval_node",
    "pack_to_state",
    "retrieval_node",
    "state_passages_to_prompt",
]
