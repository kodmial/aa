"""V2 retrieval node: planner queries to compact Evidence Pack (issue #116).

The node consumes the minimal ``search_queries`` produced by the #113
hidden planner (0 or 10..16 context-resolved Russian queries) and runs
the RRF-only target pipeline from :mod:`aa.retrieval.evidence` over the
#115 RAM-resident canonical index:

- ``search_queries == []`` performs no retrieval and yields an empty
  pack for a purely conversational/glue turn;
- otherwise every query runs BM25 + E5/FAISS branches, global RRF,
  dedup/diversity, small-to-big expansion and atomic budget selection.

Only orchestration state is written; ``messages`` is left untouched.
State carries exact passage text plus minimal provenance. Ranking
metadata (RRF/BM25/dense scores, embeddings, planner reasoning,
search previews) stays in internal retrieval metadata and never enters
the user-facing prompt. Logs carry only routes, counts and token
lengths, never prompts or user text.

The hot path is RAM-only with zero network/download dependency and no
second-stage reranker: ``QueryPlan -> BM25+E5 -> RRF -> dedup ->
expansion -> pack``.
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
        # Never coerce a missing id to the literal "None": skip passages
        # without real grounding provenance instead of misattributing them.
        if not isinstance(source_id, str) or not source_id:
            continue
        if not isinstance(section_id, str) or not section_id:
            continue
        passages.append(
            EvidencePassage(
                passage_id=passage_id,
                source=source_id,
                section=section_id,
                text=text,
            )
        )
    return passages


async def retrieval_node(
    state: TurnState,
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
) -> dict[str, Any]:
    """LangGraph retrieval node: queries to hits plus Evidence Pack.

    Runs the RRF-only pipeline in a worker thread on the RAM-resident index.
    The per-turn wall-clock latency against
    ``INTERACTIVE_LATENCY_BUDGET_MS`` is measured and propagated in
    state (``retrieval_latency_ms``/``retrieval_over_budget``) for
    observability. Only counts and latencies are logged, never prompts
    or user text.
    """
    raw_queries = state.get("search_queries", [])
    if isinstance(raw_queries, (list, tuple)):
        queries: list[str] = []
        seen: set[str] = set()
        for raw in raw_queries:
            if not isinstance(raw, str):
                continue
            cleaned = " ".join(raw.split())
            if not cleaned or cleaned.casefold() in seen:
                continue
            seen.add(cleaned.casefold())
            queries.append(cleaned)
            if len(queries) >= 16:
                break
    else:
        queries = []
    if not queries:
        logger.info("v2 retrieval skipped", extra={"queries": 0})
        return {
            "retrieval_hits": [],
            "evidence_pack": [],
            "retrieval_latency_ms": 0.0,
            "retrieval_over_budget": False,
        }
    active_config = config if config is not None else RetrievalConfig()
    started = time.perf_counter()
    pack = await asyncio.to_thread(retrieve_evidence, index, queries, config=active_config)
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
    return {
        "retrieval_hits": hits,
        "evidence_pack": pack_dicts,
        "retrieval_latency_ms": elapsed_ms,
        "retrieval_over_budget": over_budget,
    }


def make_retrieval_node(
    *,
    index: HybridIndex,
    config: RetrievalConfig | None = None,
) -> Any:
    """Build the evidence retrieval node bound to one RAM-resident index."""
    active_config = config if config is not None else RetrievalConfig()

    async def run_evidence_retrieval(state: TurnState) -> dict[str, Any]:
        return await retrieval_node(state, index=index, config=active_config)

    return run_evidence_retrieval


__all__ = [
    "make_retrieval_node",
    "pack_to_state",
    "retrieval_node",
    "state_passages_to_prompt",
]
