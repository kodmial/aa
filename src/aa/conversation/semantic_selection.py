"""Model-driven semantic candidate selection over broad book candidates.

Issue #295: BM25+E5/FAISS/RRF discovers a genuinely broad ranked
candidate set, but fixed top-5/top-16 caps previously discarded
potentially decisive passages before the answer model could evaluate
them. This module implements the bounded selection layer between
fusion and full-text fetch:

- candidate *previews* (short discovery excerpts + provenance + fused
  rank) are used only for discovery, never as final supporting text;
- an LLM assesses the previews against the user's real context-resolved
  intent plus dialogue context, selects pertinent candidates, and may
  request targeted re-read (neighbor expansion) or another search;
- once selected, callers fetch exact complete canonical passages via
  :func:`fetch_full_passages_for_selection` (small-to-big expansion,
  provenance preserved);
- without a model (tests, outage) a deterministic generic heuristic
  promotion applies: generic token-overlap relevance over the same
  previews, stable by fused rank. No domain-specific regex, intent
  whitelist, keyword table, or fixed qualification routing lives here.

Bounds (never an unbounded dump, never unconditional slow BGE):

- at most ``MAX_SELECTION_CANDIDATES`` previews enter one selection;
- previews carry at most ``PREVIEW_CHARS`` characters each;
- at most ``MAX_SELECTED_CHUNKS`` chunk ids are selected per round;
- follow-up queries are bounded to ``MAX_FOLLOWUP_QUERIES``.

Privacy: previews contain short book excerpts for the selecting model
only; telemetry carries counts/ranks/ids-hashes, never text.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

logger = logging.getLogger("aa.conversation.semantic_selection")

MAX_SELECTION_CANDIDATES = 64
PREVIEW_CHARS = 400
MAX_SELECTED_CHUNKS = 12
MAX_FOLLOWUP_QUERIES = 6
SELECTION_ATTEMPT_BUDGET_S = 8.0

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+", re.UNICODE)


@dataclass(frozen=True)
class CandidatePreview:
    """One broad candidate for discovery-only inspection."""

    chunk_id: str
    fused_rank: int
    fused_score: float
    section: str
    source_id: str
    preview_text: str
    full_text: str = ""
    # Provenance for #311 (never text): stable planner query ids that
    # retrieved this candidate plus the semantic need ids reached through
    # the trusted query->need map. Empty need tuple means unknown/unmapped.
    query_ids: tuple[str, ...] = ()
    need_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class NeedPreviewStatus:
    """Coverage metadata for one semantic need over previews (#311).

    Planning/selection metadata only: ``represented`` reports whether at
    least one preview is associated with the need. It never claims the
    need is sufficiently evidenced; only #306 full-read assessment can
    mark sufficiency.
    """

    need_id: str
    previewed_count: int
    best_fused_rank: int | None
    represented: bool


class SemanticSelection(BaseModel):
    """Bounded model decision over candidate previews (transport only)."""

    selected_chunk_ids: list[str] = Field(default_factory=list)
    need_more_detail: bool = Field(default=False)
    followup_queries: list[str] = Field(default_factory=list)
    # Explicit per-need report (#311): need ids with no reasonable preview
    # or no selectable evidence. Missing-need status is never positive
    # coverage; the caller must request further discovery instead.
    uncovered_need_ids: list[str] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
    # Internal provenance only: never supplied by, or trusted from, the model.
    _used_model: bool = PrivateAttr(default=False)


class SemanticSelectionError(ValueError):
    """Semantic selection output failed validation."""


def _tokenize(text: str) -> list[str]:
    return [tok.casefold() for tok in _WORD_RE.findall(text or "") if len(tok) >= 3]


def preview_candidates(
    *,
    fused_ordered: Sequence[tuple[str, float]],
    texts: dict[str, str],
    sections: dict[str, str] | None = None,
    sources: dict[str, str] | None = None,
    limit: int = MAX_SELECTION_CANDIDATES,
    candidate_query_map: dict[str, list[str]] | None = None,
    candidate_need_map: dict[str, list[str]] | None = None,
) -> list[CandidatePreview]:
    """Build bounded discovery previews in fused-rank order.

    ``fused_ordered`` is ``[(chunk_id, fused_score)]`` best-first over the
    genuinely broad pool (ranks >16 included). Preview text is truncated
    for discovery; :func:`fetch_full_passages_for_selection` later
    supplies the exact complete text for selected ids only.
    Optional provenance maps attach stable query/need ids without
    altering order or text.
    """
    out: list[CandidatePreview] = []
    sections = sections or {}
    sources = sources or {}
    query_map = candidate_query_map or {}
    need_map = candidate_need_map or {}
    for rank, (chunk_id, score) in enumerate(fused_ordered[: max(1, limit)]):
        full = texts.get(chunk_id, "")
        preview = full[:PREVIEW_CHARS] if isinstance(full, str) else ""
        raw_queries = list(query_map.get(chunk_id, []) or [])
        raw_needs = list(need_map.get(chunk_id, []) or [])
        out.append(
            CandidatePreview(
                chunk_id=chunk_id,
                fused_rank=rank,
                fused_score=float(score),
                section=str(sections.get(chunk_id, "")),
                source_id=str(sources.get(chunk_id, "")),
                preview_text=preview,
                full_text=full if isinstance(full, str) else "",
                query_ids=tuple(raw_queries),
                need_ids=tuple(raw_needs),
            )
        )
    return out


def _need_ids_from_info(needs: Sequence[Any] | None) -> list[str]:
    """Extract stable need ids from InformationNeed models or dicts."""
    out: list[str] = []
    seen: set[str] = set()
    for entry in list(needs or []):
        try:
            if isinstance(entry, dict):
                nid = str(entry.get("need_id", "") or "").strip()
            else:
                nid = str(getattr(entry, "need_id", "") or "").strip()
        except Exception:
            continue
        if nid and nid not in seen:
            seen.add(nid)
            out.append(nid)
    return out


def _need_text_for_prompt(needs: Sequence[Any] | None, need_id: str) -> str:
    """Return the model-understood need text for one need id (prompt only)."""
    for entry in list(needs or []):
        try:
            if isinstance(entry, dict):
                nid = str(entry.get("need_id", "") or "").strip()
                text = str(entry.get("text", "") or "")
            else:
                nid = str(getattr(entry, "need_id", "") or "").strip()
                text = str(getattr(entry, "text", "") or "")
        except Exception:
            continue
        if nid == need_id:
            return text
    return ""


def assess_need_preview_coverage(
    previews: Sequence[CandidatePreview],
    needs: Sequence[Any] | None,
) -> tuple[list[NeedPreviewStatus], int]:
    """Report per-need preview representation over actual mappings (#311).

    Coverage metadata for planning/selection only: counts how many
    previews carry each need id and the best global fused rank. A need
    with zero associated previews is unrepresented; the caller must
    report it as uncovered and request further discovery rather than
    fabricating coverage from lexical or chapter diversity. Never a
    sufficiency verdict.
    """
    need_ids = _need_ids_from_info(needs)
    statuses: list[NeedPreviewStatus] = []
    for nid in need_ids:
        count = 0
        best: int | None = None
        for preview in list(previews or []):
            if nid in set(preview.need_ids or ()):
                count += 1
                if best is None or preview.fused_rank < best:
                    best = preview.fused_rank
        statuses.append(
            NeedPreviewStatus(
                need_id=nid,
                previewed_count=count,
                best_fused_rank=best,
                represented=count > 0,
            )
        )
    unmapped = 0
    for preview in list(previews or []):
        if not tuple(preview.need_ids or ()):
            unmapped += 1
    return statuses, unmapped


def select_need_aware_previews(
    *,
    fused_ordered: Sequence[tuple[str, float]],
    texts: dict[str, str],
    sections: dict[str, str] | None = None,
    sources: dict[str, str] | None = None,
    candidate_query_map: dict[str, list[str]] | None = None,
    candidate_need_map: dict[str, list[str]] | None = None,
    information_needs: Sequence[Any] | None = None,
    limit: int = MAX_SELECTION_CANDIDATES,
) -> list[CandidatePreview]:
    """Build a bounded preview set preserving per-need exposure (#311).

    Per-need candidate representation plus remaining global relevance:
    each represented need contributes its best global-ranked candidates
    round-robin so paraphrases for the first need cannot crowd out the
    only reasonable candidate for another need, then remaining slots
    fill by global RRF relevance. Shared candidates (needs A and B)
    appear once with both associations. Needs with no genuine evidence
    receive no fabricated previews. The final set stays within ``limit``
    so the model call fits real prompt/token budgets.
    """
    capped = max(1, int(limit))
    ordered = list(fused_ordered or [])
    if not ordered:
        return []
    need_ids = _need_ids_from_info(information_needs)
    need_map = candidate_need_map or {}
    if not need_ids or not any(list(need_map.get(cid, []) or []) for cid, _ in ordered):
        return preview_candidates(
            fused_ordered=ordered,
            texts=texts,
            sections=sections,
            sources=sources,
            limit=capped,
            candidate_query_map=candidate_query_map,
            candidate_need_map=candidate_need_map,
        )
    # Per-need buckets in global fused-rank order (best-first).
    rank_of: dict[str, int] = {}
    for rank, (cid, _score) in enumerate(ordered):
        if cid not in rank_of:
            rank_of[cid] = rank
    buckets: dict[str, list[str]] = {nid: [] for nid in need_ids}
    for cid, _score in ordered:
        for nid in list(need_map.get(cid, []) or []):
            if nid in buckets and cid not in buckets[nid]:
                buckets[nid].append(cid)
    picked: list[str] = []
    picked_set: set[str] = set()
    # Round-robin: one best remaining candidate per represented need per
    # round, in need order, so every represented need surfaces early.
    active = [nid for nid in need_ids if buckets.get(nid)]
    round_pos: dict[str, int] = {nid: 0 for nid in active}
    progress = True
    while progress and len(picked) < capped:
        progress = False
        for nid in active:
            if len(picked) >= capped:
                break
            bucket = buckets.get(nid, [])
            pos = round_pos.get(nid, 0)
            while pos < len(bucket) and bucket[pos] in picked_set:
                pos += 1
            if pos < len(bucket):
                cid = bucket[pos]
                picked.append(cid)
                picked_set.add(cid)
                round_pos[nid] = pos + 1
                progress = True
            else:
                round_pos[nid] = pos
    # Remaining global relevance (including unmapped) by fused rank.
    if len(picked) < capped:
        for cid, _score in ordered:
            if len(picked) >= capped:
                break
            if cid not in picked_set:
                picked.append(cid)
                picked_set.add(cid)
    score_of: dict[str, float] = {}
    for cid, score in ordered:
        if cid not in score_of:
            score_of[cid] = float(score)
    sections_map = sections or {}
    sources_map = sources or {}
    query_map = candidate_query_map or {}
    out: list[CandidatePreview] = []
    for cid in picked:
        full = texts.get(cid, "")
        preview = full[:PREVIEW_CHARS] if isinstance(full, str) else ""
        out.append(
            CandidatePreview(
                chunk_id=cid,
                fused_rank=int(rank_of.get(cid, 0)),
                fused_score=float(score_of.get(cid, 0.0)),
                section=str(sections_map.get(cid, "")),
                source_id=str(sources_map.get(cid, "")),
                preview_text=preview,
                full_text=full if isinstance(full, str) else "",
                query_ids=tuple(list(query_map.get(cid, []) or [])),
                need_ids=tuple(list(need_map.get(cid, []) or [])),
            )
        )
    return out


def heuristic_relevance_score(preview_text: str, resolved_intent: str) -> float:
    """Generic token-overlap relevance (no domain tables, no regex intents).

    Counts distinct intent tokens (length >=3, casefolded) present in the
    preview, normalized by intent size. Purely generic lexical signal used
    only when no selection model is available, or to order previews
    deterministically before the model call. Never decides support.
    """
    intent_tokens = {tok for tok in _tokenize(resolved_intent)}
    if not intent_tokens:
        return 0.0
    preview_tokens = set(_tokenize(preview_text))
    if not preview_tokens:
        return 0.0
    return len(intent_tokens & preview_tokens) / max(1, len(intent_tokens))


def rerank_previews_semantically(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
) -> list[CandidatePreview]:
    """Promote intent-relevant previews without dropping deep ranks.

    Stable sort: heuristic relevance first, fused rank second. Deep fused
    ranks (>5, >16) with genuine lexical overlap therefore surface while
    every candidate stays inspectable. No content is rewritten.
    """
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    scored = [(heuristic_relevance_score(p.preview_text, context), p) for p in previews]
    scored.sort(key=lambda pair: (-pair[0], pair[1].fused_rank))
    return [preview for _, preview in scored]


def select_by_fused_fallback(
    previews: Sequence[CandidatePreview],
    *,
    limit: int = MAX_SELECTED_CHUNKS,
) -> SemanticSelection:
    """Deterministic fallback: keep fused order (never hide deep ranks)."""
    selected = [p.chunk_id for p in list(previews)[: max(1, limit)]]
    return SemanticSelection(
        selected_chunk_ids=selected, need_more_detail=False, followup_queries=[]
    )


def heuristic_select(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
    limit: int = MAX_SELECTED_CHUNKS,
) -> SemanticSelection:
    """Deterministic generic selection promoting relevant deep candidates."""
    if not previews:
        return SemanticSelection(selected_chunk_ids=[], need_more_detail=True, followup_queries=[])
    reranked = rerank_previews_semantically(
        previews, resolved_intent=resolved_intent, conversation_context=conversation_context
    )
    selected = [p.chunk_id for p in reranked[: max(1, limit)]]
    # Signal targeted re-read when nothing lexically matches: the caller
    # should run one focused additional search instead of regenerating on
    # the same misleading top prefixes.
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    best = heuristic_relevance_score(reranked[0].preview_text, context) if reranked else 0.0
    return SemanticSelection(
        selected_chunk_ids=selected,
        need_more_detail=bool(best <= 0.0 and context),
        followup_queries=[],
    )


def selection_prompt(
    *,
    previews: Sequence[CandidatePreview],
    resolved_intent: str,
    conversation_context: str = "",
    user_message: str = "",
    context_digest: str = "",
    information_needs: Sequence[Any] | None = None,
) -> tuple[str, str]:
    """Render the bounded selection prompt (system, user)."""
    from aa.conversation.prompt_safety import (
        UNTRUSTED_DATA_POLICY_LINE,
        escape_xml_text,
        quote_xml_attr,
    )

    system = (
        "You are a book-evidence selector. Read the context-resolved intent "
        "and the candidate passage previews below. Select the candidate ids "
        "that are materially relevant for answering the intent. Use only "
        "semantic relevance to the actual intent, never keyword counting "
        "alone. Set need_more_detail true when no preview looks sufficient "
        "and a focused follow-up search is needed. Return only the required "
        "JSON decision. "
        + UNTRUSTED_DATA_POLICY_LINE
        + " Candidate previews are discovery references only, never full "
        "evidence; select only ids listed in <candidates>."
    )
    # Shared canonical view: the upstream canonical model view is already
    # bounded, so these bytes travel verbatim. No independent per-stage
    # re-truncation while claiming the same ``context_digest``: the digest
    # must attest to the exact bytes sent to the model.
    lines: list[str] = [
        UNTRUSTED_DATA_POLICY_LINE,
        "<resolved_intent>",
        escape_xml_text((resolved_intent or user_message or "").strip()) or "(no intent)",
        "</resolved_intent>",
    ]
    if user_message.strip():
        lines += [
            "<current_user_message>",
            escape_xml_text(user_message.strip()),
            "</current_user_message>",
        ]
    if conversation_context.strip():
        lines += [
            "<conversation_context>",
            escape_xml_text(conversation_context.strip()),
            "</conversation_context>",
        ]
    need_ids = _need_ids_from_info(information_needs)
    if need_ids:
        lines.append("<information_needs>")
        for nid in need_ids:
            need_text = _need_text_for_prompt(information_needs, nid)
            lines.append(
                f"<need id={quote_xml_attr(nid)}>{escape_xml_text(need_text.strip())}</need>"
            )
        lines.append("</information_needs>")
    lines.append("<candidates>")
    for preview in list(previews)[:MAX_SELECTION_CANDIDATES]:
        query_attr = " ".join(list(preview.query_ids or ()))
        need_attr = " ".join(list(preview.need_ids or ()))
        lines.append(
            f"<candidate id={quote_xml_attr(preview.chunk_id)} "
            f"rank={quote_xml_attr(str(preview.fused_rank))} "
            f"section={quote_xml_attr(preview.section)}"
            + (f" queries={quote_xml_attr(query_attr)}" if query_attr else "")
            + (f" needs={quote_xml_attr(need_attr)}" if need_attr else "")
            + f">{escape_xml_text(preview.preview_text)}</candidate>"
        )
    if not previews:
        lines.append("(no candidates)")
    lines.append("</candidates>")
    if str(context_digest or "").strip():
        lines.append(
            f"<context_digest>{escape_xml_text(str(context_digest).strip())}</context_digest>"
        )
    lines.append(
        "Return ONLY a JSON object with exactly these keys: "
        '{"selected_chunk_ids": array of strings, '
        '"need_more_detail": boolean, "followup_queries": array of strings}. '
        'You may also include optional "uncovered_need_ids": array of '
        "information-need ids from <information_needs> that have no "
        "reasonable preview above; never claim coverage for a need without "
        "selecting its evidence, and when the candidate budget cannot show "
        "a need, report it uncovered rather than selecting unrelated ids. "
        f"Select at most {MAX_SELECTED_CHUNKS} ids from <candidates>. "
        f"Provide at most {MAX_FOLLOWUP_QUERIES} follow-up queries."
    )
    return system, "\n".join(lines)


def semantic_selection_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "selected_chunk_ids": {"type": "array", "items": {"type": "string"}},
            "need_more_detail": {"type": "boolean"},
            "followup_queries": {"type": "array", "items": {"type": "string"}},
            "uncovered_need_ids": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["selected_chunk_ids", "need_more_detail", "followup_queries"],
    }


def validate_semantic_selection(
    data: object,
    *,
    known_chunk_ids: set[str] | None = None,
    known_need_ids: set[str] | None = None,
) -> SemanticSelection:
    """Strictly validate one selection decision (fail-closed)."""
    if isinstance(data, SemanticSelection):
        selection = data
    elif isinstance(data, dict):
        try:
            selection = SemanticSelection.model_validate(data)
        except Exception as exc:
            raise SemanticSelectionError(f"semantic selection invalid: {exc}") from exc
    else:
        raise SemanticSelectionError("semantic selection is not an object")
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in selection.selected_chunk_ids:
        chunk_id = str(raw or "").strip()
        if not chunk_id or chunk_id in seen:
            continue
        if known_chunk_ids is not None and chunk_id not in known_chunk_ids:
            raise SemanticSelectionError(f"selection cites unknown candidate {chunk_id!r}")
        seen.add(chunk_id)
        cleaned.append(chunk_id)
    cleaned = cleaned[:MAX_SELECTED_CHUNKS]
    if not cleaned and not bool(selection.need_more_detail):
        raise SemanticSelectionError("semantic selection selected no candidates")
    followups: list[str] = []
    for raw in selection.followup_queries:
        query = " ".join(str(raw or "").split()).strip()
        if not query:
            continue
        key = query.casefold()
        if key in {q.casefold() for q in followups}:
            continue
        followups.append(query)
        if len(followups) >= MAX_FOLLOWUP_QUERIES:
            break
    uncovered: list[str] = []
    seen_uncovered: set[str] = set()
    for raw in list(selection.uncovered_need_ids or []):
        nid = str(raw or "").strip()
        if not nid or nid in seen_uncovered:
            continue
        if known_need_ids is not None and nid not in known_need_ids:
            raise SemanticSelectionError(f"selection cites unknown need {nid!r}")
        seen_uncovered.add(nid)
        uncovered.append(nid)
    return SemanticSelection(
        selected_chunk_ids=cleaned,
        need_more_detail=bool(selection.need_more_detail),
        followup_queries=followups,
        uncovered_need_ids=uncovered,
    )


def apply_conservative_uncovered(
    selection: SemanticSelection,
    previews: Sequence[CandidatePreview],
    information_needs: Sequence[Any] | None,
) -> SemanticSelection:
    """Add budget-excluded or unselected needs to uncovered (never fake success).

    A need with no preview (budget exhaustion or no genuine evidence) or
    with previews but no selected associated candidate is reported as
    uncovered with ``need_more_detail`` requesting further discovery.
    Only #306 full-read assessment can mark sufficiency. Idempotent.
    """
    try:
        statuses, _ = assess_need_preview_coverage(previews, information_needs)
    except Exception:
        return selection
    if not statuses:
        return selection
    try:
        preview_by_id = {p.chunk_id: p for p in list(previews or [])}
        selected_needs: set[str] = set()
        for cid in list(selection.selected_chunk_ids or []):
            preview = preview_by_id.get(cid)
            if preview is not None:
                for nid in list(preview.need_ids or ()):
                    selected_needs.add(str(nid))
        uncovered = {str(nid) for nid in list(selection.uncovered_need_ids or [])}
        changed = False
        for status in statuses:
            if not status.represented or status.need_id not in selected_needs:
                if status.need_id not in uncovered:
                    uncovered.add(status.need_id)
                    changed = True
        if not changed:
            return selection
        used_model = bool(getattr(selection, "_used_model", False))
        updated = SemanticSelection(
            selected_chunk_ids=list(selection.selected_chunk_ids or []),
            need_more_detail=True,
            followup_queries=list(selection.followup_queries or []),
            uncovered_need_ids=sorted(uncovered),
        )
        try:
            updated._used_model = used_model
        except Exception:
            pass
        return updated
    except Exception:
        return selection


async def aselect_semantic_candidates(
    previews: Sequence[CandidatePreview],
    *,
    resolved_intent: str,
    conversation_context: str = "",
    user_message: str = "",
    model: Any | None = None,
    known_chunk_ids: set[str] | None = None,
    context_digest: str = "",
    information_needs: Sequence[Any] | None = None,
    known_need_ids: set[str] | None = None,
) -> SemanticSelection:
    """Model-driven selection with deterministic fallback (bounded).

    When ``model`` is None or the model call fails, falls back to
    :func:`heuristic_select` so deep relevant candidates still surface
    without inventing ids. Unknown cited ids fail closed to the
    heuristic. Provider 429 propagates.

    ``known_chunk_ids``, when supplied, is the independent retrieval
    ground truth (fused pool / indexed source) that previews are
    verified against before any selector model call. It must never be
    derived from ``previews`` itself.
    """
    preview_list = list(previews)[:MAX_SELECTION_CANDIDATES]
    if not preview_list:
        statuses, _ = assess_need_preview_coverage([], information_needs)
        uncovered = [s.need_id for s in statuses if not s.represented]
        if known_need_ids is not None:
            uncovered = [nid for nid in uncovered if nid in known_need_ids]
        return SemanticSelection(
            selected_chunk_ids=[],
            need_more_detail=True,
            followup_queries=[],
            uncovered_need_ids=uncovered,
        )
    preview_ids = {p.chunk_id for p in preview_list}
    # Discovery-preview integrity before any selector model call
    # (kodmial/aa#310): previews must reference verifiable indexed
    # chunks, but a missing full-read blob never rejects selection.
    # ``known_chunk_ids`` must come from the independent fused/indexed
    # source supplied by the caller; deriving it from ``preview_list``
    # would make membership tautologically true and let an invented
    # chunk_id pass the pre-model gate.
    try:
        from aa.conversation.evidence_integrity import validate_preview_references

        validate_preview_references(preview_list, known_chunk_ids=known_chunk_ids)
    except Exception as exc:
        raise SemanticSelectionError(f"selection previews invalid: {exc}") from exc
    if model is None:
        fallback = heuristic_select(
            preview_list,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
        )
        return apply_conservative_uncovered(fallback, preview_list, information_needs)
    effective_known_needs = known_need_ids
    if effective_known_needs is None and information_needs:
        effective_known_needs = set(_need_ids_from_info(information_needs)) or None
    system, user_text = selection_prompt(
        previews=preview_list,
        resolved_intent=resolved_intent,
        conversation_context=conversation_context,
        user_message=user_message,
        context_digest=context_digest,
        information_needs=information_needs,
    )
    try:
        structured = getattr(model, "ainvoke_structured", None)
        if callable(structured):
            raw = await structured(
                user_text,
                system=system,
                schema=semantic_selection_schema(),
                retry_count=1,
            )
            selected = validate_semantic_selection(
                raw, known_chunk_ids=preview_ids, known_need_ids=effective_known_needs
            )
            selected._used_model = True  # Validated real model verdict, not a fallback.
            return apply_conservative_uncovered(selected, preview_list, information_needs)
        text_invoke = getattr(model, "_ainvoke_text", None) or getattr(model, "ainvoke", None)
        if not callable(text_invoke):
            raise SemanticSelectionError("selection model has no invocation path")
        if getattr(text_invoke, "__name__", "") == "_ainvoke_text":
            reply = await text_invoke(user_text, system=system)
            text = reply if isinstance(reply, str) else str(reply)
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            message = await text_invoke(
                [SystemMessage(content=system), HumanMessage(content=user_text)]
            )
            content = getattr(message, "content", "")
            text = content if isinstance(content, str) else str(content)
        import json as _json

        cleaned = text.strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        payload = cleaned[start : end + 1] if start >= 0 and end > start else cleaned
        try:
            data = _json.loads(payload)
        except Exception as exc:
            raise SemanticSelectionError(f"selection text is not JSON: {exc}") from exc
        selected = validate_semantic_selection(
            data, known_chunk_ids=preview_ids, known_need_ids=effective_known_needs
        )
        selected._used_model = True
        return apply_conservative_uncovered(selected, preview_list, information_needs)
    except Exception as exc:
        # 429 must escape this selector: the workflow retires the runner
        # and resumes from its last verified checkpoint. An inner broad
        # except previously swallowed the re-raised rate-limit error.
        from aa.opencode.errors import OpenCodeRateLimitError

        if isinstance(exc, OpenCodeRateLimitError):
            raise
        if isinstance(exc, SemanticSelectionError):
            logger.info("semantic selection invalid; heuristic fallback used")
        else:
            logger.info(
                "semantic selection unavailable; heuristic fallback used",
                extra={"category": type(exc).__name__},
            )
        fallback = heuristic_select(
            preview_list,
            resolved_intent=resolved_intent,
            conversation_context=conversation_context,
        )
        return apply_conservative_uncovered(fallback, preview_list, information_needs)


def fetch_full_passages_for_selection(
    index: Any,
    selected_chunk_ids: Sequence[str],
    *,
    neighbor_window: int | None = None,
) -> list[Any]:
    """Fetch exact complete passages for selected chunk ids (provenance kept).

    Uses small-to-big expansion (full parent paragraph + bounded neighbor
    window) over the canonical index. Returns
    ``EvidencePassageData`` items in selection order; unknown ids raise.
    No truncation is applied here: the caller budgets the pack.
    """
    from aa.retrieval.evidence import NEIGHBOR_WINDOW, expand_small_to_big
    from aa.retrieval.fusion import FusedCandidate

    window = NEIGHBOR_WINDOW if neighbor_window is None else int(neighbor_window)
    winners = [
        FusedCandidate(
            chunk_id=str(chunk_id),
            fused_score=float(len(selected_chunk_ids) - pos),
            lexical_rank=None,
            dense_rank=None,
            lexical_score=None,
            dense_score=None,
        )
        for pos, chunk_id in enumerate(selected_chunk_ids)
    ]
    return expand_small_to_big(index, winners, neighbor_window=window)


def order_pack_semantically(
    pack_dicts: Sequence[dict[str, Any]],
    *,
    resolved_intent: str,
    conversation_context: str = "",
) -> list[dict[str, Any]]:
    """Reorder an Evidence Pack by generic intent relevance (stable).

    Lexical fallback only (generic token overlap, never an LLM semantic
    verdict): no passage is dropped and no text is altered. Relevant
    deep-rank passages move forward while the full pack stays available
    to the generator and verifier. Deterministic and model-free so
    hermetic tests and outage paths behave identically. Callers must
    label this ordering ``lexical_fallback``, never ``model_selection``.
    """
    context = f"{resolved_intent or ''} {conversation_context or ''}".strip()
    if not context:
        return [dict(item) for item in pack_dicts]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for pos, item in enumerate(pack_dicts):
        text = str(item.get("text", "")) if isinstance(item, dict) else ""
        score = heuristic_relevance_score(text[: PREVIEW_CHARS * 2], context)
        scored.append((score, pos, dict(item)))
    scored.sort(key=lambda triple: (-triple[0], triple[1]))
    return [item for _, _, item in scored]


def serialize_preview_coverage(
    previews: Sequence[CandidatePreview],
    selection: SemanticSelection,
) -> list[dict[str, Any]]:
    """Serialize selected-preview query/need mappings for downstream #306.

    Bounded id-only records (no text): each preview that the selector
    actually saw, with its stable query/need associations and whether it
    was selected. Full evidence sufficiency is never claimed here.
    """
    selected = set(selection.selected_chunk_ids or [])
    out: list[dict[str, Any]] = []
    for preview in list(previews or [])[:MAX_SELECTION_CANDIDATES]:
        out.append(
            {
                "chunk_id": str(preview.chunk_id),
                "fused_rank": int(preview.fused_rank),
                "query_ids": [str(qid) for qid in list(preview.query_ids or ())],
                "need_ids": [str(nid) for nid in list(preview.need_ids or ())],
                "selected": bool(preview.chunk_id in selected),
            }
        )
    return out


def selection_telemetry(
    *,
    previews: Sequence[CandidatePreview],
    selection: SemanticSelection,
    latency_ms: float = 0.0,
    information_needs: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Privacy-safe selection telemetry (ranks/counts/need ids only, never text)."""
    by_id = {p.chunk_id: p for p in previews}
    ranks = [by_id[cid].fused_rank for cid in selection.selected_chunk_ids if cid in by_id]
    digest = hashlib.sha256("|".join(selection.selected_chunk_ids).encode("utf-8")).hexdigest()[:16]
    base: dict[str, Any] = {
        "selection_candidates": len(previews),
        "selection_selected": len(selection.selected_chunk_ids),
        "selection_max_rank": max(ranks) if ranks else -1,
        "selection_deep_rank_gt5": any(rank > 5 for rank in ranks),
        "selection_deep_rank_gt16": any(rank > 16 for rank in ranks),
        "selection_need_more": bool(selection.need_more_detail),
        "selection_followups": len(selection.followup_queries),
        "selection_digest": digest,
        "selection_latency_ms": round(float(latency_ms), 1),
        "selection_model_used": bool(selection._used_model),
        "selection_fallback_used": not bool(selection._used_model),
    }
    try:
        statuses, unmapped_previews = assess_need_preview_coverage(previews, information_needs)
    except Exception:
        statuses = []
        unmapped_previews = 0
    per_need_selected: dict[str, int] = {}
    for preview in list(previews or []):
        if preview.chunk_id in set(selection.selected_chunk_ids or []):
            for nid in list(preview.need_ids or ()):
                per_need_selected[str(nid)] = per_need_selected.get(str(nid), 0) + 1
    unmapped_selected = 0
    for preview in list(previews or []):
        if preview.chunk_id in set(selection.selected_chunk_ids or []) and not tuple(
            preview.need_ids or ()
        ):
            unmapped_selected += 1
    base.update(
        {
            "selection_need_count": len(statuses),
            "selection_need_represented": sum(1 for s in statuses if s.represented),
            "selection_need_unrepresented": [s.need_id for s in statuses if not s.represented],
            "selection_unmapped_previews": int(unmapped_previews),
            "selection_unmapped_selected": int(unmapped_selected),
            "selection_per_need_selected": per_need_selected,
            "selection_uncovered_need_ids": list(selection.uncovered_need_ids or []),
        }
    )
    return base


__all__ = [
    "MAX_FOLLOWUP_QUERIES",
    "MAX_SELECTED_CHUNKS",
    "MAX_SELECTION_CANDIDATES",
    "PREVIEW_CHARS",
    "SELECTION_ATTEMPT_BUDGET_S",
    "CandidatePreview",
    "NeedPreviewStatus",
    "SemanticSelection",
    "SemanticSelectionError",
    "apply_conservative_uncovered",
    "aselect_semantic_candidates",
    "assess_need_preview_coverage",
    "fetch_full_passages_for_selection",
    "heuristic_relevance_score",
    "heuristic_select",
    "order_pack_semantically",
    "preview_candidates",
    "rerank_previews_semantically",
    "select_by_fused_fallback",
    "select_need_aware_previews",
    "selection_prompt",
    "selection_telemetry",
    "semantic_selection_schema",
    "serialize_preview_coverage",
    "validate_semantic_selection",
]
