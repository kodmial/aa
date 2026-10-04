"""RRF fusion, overlap dedup and section diversity (issue #17).

Fixed production parameters:

- lexical top 40, dense top 40 (retrieval branches);
- RRF ``k=60``;
- max 12 compact candidates per aspect.

Fusion inputs are per-query ranked id lists from the lexical and dense
branches (original query plus planner same-language rewrites). RRF fuses
all inputs; overlapping chunks are deduplicated (exact id plus
same-section character-overlap); section/chapter diversity caps
per-section winners so one chapter cannot crowd out the rest.
"""

from __future__ import annotations

from dataclasses import dataclass

RRF_K = 60
MAX_CANDIDATES_PER_ASPECT = 12
MAX_PER_SECTION = 4


@dataclass(frozen=True)
class FusedCandidate:
    """One fused candidate before provenance enrichment."""

    chunk_id: str
    fused_score: float
    lexical_rank: int | None
    dense_rank: int | None
    lexical_score: float | None
    dense_score: float | None


def rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]],
    *,
    k: int = RRF_K,
) -> dict[str, FusedCandidate]:
    """Fuse ranked lists with Reciprocal Rank Fusion (``1/(k+rank)``)."""
    if k <= 0:
        raise ValueError("k must be > 0")
    scores: dict[str, float] = {}
    lex_rank: dict[str, int] = {}
    dense_rank: dict[str, int] = {}
    lex_score: dict[str, float] = {}
    dense_score: dict[str, float] = {}
    for position, ranked in enumerate(ranked_lists):
        branch = "lexical" if position % 2 == 0 else "dense"
        for rank, (chunk_id, score) in enumerate(ranked, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if branch == "lexical":
                if chunk_id not in lex_rank or rank < lex_rank[chunk_id]:
                    lex_rank[chunk_id] = rank
                    lex_score[chunk_id] = float(score)
            else:
                if chunk_id not in dense_rank or rank < dense_rank[chunk_id]:
                    dense_rank[chunk_id] = rank
                    dense_score[chunk_id] = float(score)
    fused: dict[str, FusedCandidate] = {}
    for chunk_id, score in scores.items():
        fused[chunk_id] = FusedCandidate(
            chunk_id=chunk_id,
            fused_score=score,
            lexical_rank=lex_rank.get(chunk_id),
            dense_rank=dense_rank.get(chunk_id),
            lexical_score=lex_score.get(chunk_id),
            dense_score=dense_score.get(chunk_id),
        )
    return fused


def deduplicate_overlaps(
    candidates: list[FusedCandidate],
    *,
    spans: dict[str, tuple[str, int, int]],
) -> list[FusedCandidate]:
    """Drop same-section character-overlapping losers (keep fused winner)."""
    ordered = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
    kept: list[FusedCandidate] = []
    kept_spans: list[tuple[str, int, int]] = []
    for candidate in ordered:
        span = spans.get(candidate.chunk_id)
        if span is None:
            kept.append(candidate)
            continue
        section, start, end = span
        overlap = False
        for kept_section, kept_start, kept_end in kept_spans:
            if kept_section != section:
                continue
            if start < kept_end and kept_start < end:
                overlap = True
                break
        if not overlap:
            kept.append(candidate)
            kept_spans.append((section, start, end))
    return kept


def enforce_diversity(
    candidates: list[FusedCandidate],
    *,
    sections: dict[str, str],
    max_n: int = MAX_CANDIDATES_PER_ASPECT,
    max_per_section: int = MAX_PER_SECTION,
) -> list[FusedCandidate]:
    """Cap per-section winners, then fill remaining slots by fused score."""
    if max_n <= 0 or max_per_section <= 0:
        raise ValueError("max_n and max_per_section must be > 0")
    ordered = sorted(candidates, key=lambda item: item.fused_score, reverse=True)
    picked: list[FusedCandidate] = []
    counts: dict[str, int] = {}
    deferred: list[FusedCandidate] = []
    for candidate in ordered:
        section = sections.get(candidate.chunk_id, "?")
        if counts.get(section, 0) < max_per_section:
            picked.append(candidate)
            counts[section] = counts.get(section, 0) + 1
            if len(picked) >= max_n:
                break
        else:
            deferred.append(candidate)
    if len(picked) < max_n:
        for candidate in deferred:
            if len(picked) >= max_n:
                break
            picked.append(candidate)
    return picked[:max_n]
