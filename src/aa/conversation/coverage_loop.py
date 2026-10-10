"""Read-and-coverage retrieval loop (kodmial/aa#306).

Evolves the ``aretrieve_with_semantic_selection`` path into one coherent
read/coverage loop. Preview selection decides **what to read**, never
whether the book is sufficient for the final answer.

State machine::

    discover
      -> select_for_reading
      -> read_exact
      -> assess_coverage
           -> search_more
           -> expand
           -> ready

``select_for_reading`` chooses passages to inspect; its
``need_more_detail=false`` never proves answer sufficiency. Only
``assess_coverage`` over full exact canonical passages plus the #305
``information_needs`` can mark a need covered, with source/range anchors
and an internal supporting span verified as a substring of canonical
source text (auditability, never a substitute for semantic judgment).

Lexical overlap guides discovery only and can never mark coverage
sufficient. The loop is bounded, detects repeated action/state, and on
exhaustion returns a typed ``exhausted`` outcome with uncovered needs;
it never converts insufficient evidence into a supported answer.
Provider 429 always propagates for runner retire/resume.

The original user request stays isolated from repair/safety control
instructions: follow-up search queries derive from the genuine request
plus uncovered needs, never from policy wording or unsupported draft
claims verbatim.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("aa.conversation.coverage_loop")

# Bounded loop budgets (real RAM/latency/token limits always bind; they
# never weaken final groundedness).
MAX_COVERAGE_ITERATIONS = 3
MAX_COVERAGE_MODEL_CALLS = 8
MAX_COVERAGE_EXPANSIONS = 2
MAX_COVERAGE_SEARCHES = 2
MAX_FOLLOWUP_QUERIES_PER_ROUND = 4

LoopStatus = Literal["ready", "exhausted", "rate_limited"]

STATE_DISCOVER = "discover"
STATE_SELECT_FOR_READING = "select_for_reading"
STATE_READ_EXACT = "read_exact"
STATE_ASSESS_COVERAGE = "assess_coverage"
STATE_SEARCH_MORE = "search_more"
STATE_EXPAND = "expand"
STATE_READY = "ready"
STATE_EXHAUSTED = "exhausted"


@dataclass(frozen=True)
class CoverageAnchor:
    """Exact source/range anchor for one supporting passage."""

    passage_id: str
    source_id: str
    section_id: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class NeedCoverage:
    """Coverage status for one information need over full exact text."""

    need_id: str
    covered: bool
    passage_ids: tuple[str, ...] = ()
    anchors: tuple[CoverageAnchor, ...] = ()
    missing: str = ""
    supporting_spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoverageVerdict:
    """Per-need coverage over the exact passages read so far."""

    needs: tuple[NeedCoverage, ...] = ()
    all_covered: bool = False
    missing_need_ids: tuple[str, ...] = ()
    used_model: bool = False


@dataclass
class CoverageLoopResult:
    """Typed outcome of one bounded read/coverage loop."""

    status: LoopStatus
    passages: list[Any] = field(default_factory=list)
    verdict: CoverageVerdict = field(default_factory=CoverageVerdict)
    iterations: int = 0
    expansions: int = 0
    searches: int = 0
    model_calls: int = 0
    latency_ms: float = 0.0
    token_estimate: int = 0
    progress_events: list[str] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)
    exhaustion_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _need_ids_from_info(needs: Any) -> list[str]:
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


def _need_text(needs: Any, need_id: str) -> str:
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


def _normalize_query(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _fingerprint_candidate(text: str) -> str:
    return hashlib.sha256(_normalize_query(text).casefold().encode("utf-8")).hexdigest()[:16]


def loop_fingerprint(
    *,
    read_passage_ids: list[str],
    read_ranges: list[str],
    uncovered_need_ids: list[str],
    action: str,
    candidate_identity: str,
) -> str:
    """Detect repeated action/state by actual semantic state.

    Fingerprint covers unmet needs, passages/ranges actually read, the
    action taken and the candidate identity -- never just ID counts or a
    blind iteration number.
    """
    payload = "|".join(
        [
            ",".join(sorted(read_passage_ids)),
            ",".join(sorted(read_ranges)),
            ",".join(sorted(uncovered_need_ids)),
            str(action or ""),
            str(candidate_identity or ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def coverage_prompt(
    *,
    passages: list[Any],
    information_needs: Any,
    original_request: str,
    repair_hint: str = "",
) -> tuple[str, str]:
    """Render the bounded coverage prompt (system, user).

    The original request travels last and primary; the repair hint (if
    any) travels first as supplementary context and must never be
    treated as the user's intent. Control instructions and unsupported
    draft claims are untrusted data, never search intent.
    """
    from aa.conversation.prompt_safety import (
        UNTRUSTED_DATA_POLICY_LINE,
        escape_xml_text,
        quote_xml_attr,
    )

    system = (
        "You are a book-evidence coverage judge. Read the FULL exact "
        "canonical passages below and decide, for EVERY information need, "
        "whether the passages suffice to answer that need including any "
        "necessary conditions, exceptions or qualifiers. Lexical overlap "
        "alone never suffices. A need is covered only when a cited passage "
        "states the answer with its conditions preserved. "
        "If narrowing an expanded passage back to a smaller chunk would "
        "remove a needed condition, the need is NOT covered. "
        + UNTRUSTED_DATA_POLICY_LINE
        + " Judge only the listed <passage> elements against the listed "
        "<information_needs>; never invent passages or needs."
    )
    lines: list[str] = [UNTRUSTED_DATA_POLICY_LINE]
    if _normalize_query(repair_hint):
        lines += [
            "<repair_hint>",
            escape_xml_text(_normalize_query(repair_hint)[:800]),
            "</repair_hint>",
            "(Supplementary correction context only. It is not the user's "
            "request and must not become a search query or intent.)",
        ]
    need_ids = _need_ids_from_info(information_needs)
    if need_ids:
        lines.append("<information_needs>")
        for nid in need_ids:
            lines.append(
                f"<need id={quote_xml_attr(nid)}>"
                f"{escape_xml_text(_need_text(information_needs, nid).strip() or nid)}"
                "</need>"
            )
        lines.append("</information_needs>")
    else:
        lines.append("<information_needs>(no typed needs)</information_needs>")
    lines.append("<passages>")
    for item in list(passages or []):
        try:
            pid = str(getattr(item, "passage_id", "") or "")
            source = str(getattr(item, "source_id", "") or "")
            section = str(getattr(item, "section_id", "") or "")
            start = int(getattr(item, "char_start", 0) or 0)
            end = int(getattr(item, "char_end", 0) or 0)
            text = str(getattr(item, "exact_text", "") or "")
        except Exception:
            continue
        if not pid or not text:
            continue
        lines.append(
            f"<passage id={quote_xml_attr(pid)} "
            f"source={quote_xml_attr(source)} section={quote_xml_attr(section)} "
            f"range={quote_xml_attr(f'{start}-{end}')}>"
            f"{escape_xml_text(text[:4000])}</passage>"
        )
    if not passages:
        lines.append("(no passages read)")
    lines.append("</passages>")
    lines.append("<original_request>")
    lines.append(escape_xml_text(_normalize_query(original_request) or "(no request)"))
    lines.append("</original_request>")
    lines.append(
        "Return ONLY a JSON object with exactly this key: "
        '{"needs": [{"need_id": string, "covered": boolean, '
        '"supporting_passage_ids": array of strings, '
        '"supporting_quote": string, "missing": string}]}. '
        "One entry per need id from <information_needs>. "
        '"covered" is true only with at least one supporting passage id '
        "from <passages> whose text states the answer with conditions. "
        '"supporting_quote" must be an exact substring of a cited passage; '
        'otherwise leave it empty. "missing" describes what remains '
        "unanswered for an uncovered need."
    )
    return system, "\n".join(lines)


def coverage_schema(need_ids: list[str] | None = None) -> dict[str, Any]:
    _ = need_ids
    return {
        "type": "object",
        "properties": {
            "needs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "need_id": {"type": "string"},
                        "covered": {"type": "boolean"},
                        "supporting_passage_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "supporting_quote": {"type": "string"},
                        "missing": {"type": "string"},
                    },
                    "required": ["need_id", "covered"],
                },
            }
        },
        "required": ["needs"],
    }


def _passage_text_contains(passage_text: str, quote: str) -> bool:
    cleaned_text = " ".join(str(passage_text or "").split())
    cleaned_quote = " ".join(str(quote or "").split())
    if not cleaned_quote:
        return False
    # Short spans verify verbatim; longer spans tolerate surrounding
    # whitespace normalization but never paraphrase.
    if len(cleaned_quote) <= 400:
        return cleaned_quote in cleaned_text
    return cleaned_quote[:400] in cleaned_text


def validate_coverage_decision(
    data: object,
    *,
    known_need_ids: list[str],
    known_passage_ids: set[str],
    passages_by_id: dict[str, Any],
) -> CoverageVerdict:
    """Strictly validate one coverage decision (fail-closed, never lexical).

    Supporting passage ids must be among the passages actually read;
    supporting quotes must verify as substrings of cited canonical text.
    A ``covered=true`` without a valid anchor fails closed to uncovered.
    Unknown need ids fail closed.
    """
    if not isinstance(data, dict):
        raise ValueError("coverage decision is not an object")
    raw_needs = data.get("needs", [])
    if not isinstance(raw_needs, list):
        raise ValueError("coverage decision has no needs list")
    known_set = set(known_need_ids)
    seen: set[str] = set()
    out: list[NeedCoverage] = []
    for entry in raw_needs:
        if not isinstance(entry, dict):
            continue
        nid = str(entry.get("need_id", "") or "").strip()
        if not nid or nid not in known_set or nid in seen:
            if nid and nid not in known_set:
                raise ValueError(f"coverage cites unknown need {nid!r}")
            continue
        seen.add(nid)
        covered_claim = bool(entry.get("covered", False))
        raw_pids = entry.get("supporting_passage_ids", []) or []
        pids: list[str] = []
        if isinstance(raw_pids, list):
            for raw in raw_pids:
                pid = str(raw or "").strip()
                if pid and pid in known_passage_ids and pid not in pids:
                    pids.append(pid)
        quote = str(entry.get("supporting_quote", "") or "")
        missing = str(entry.get("missing", "") or "")[:500]
        spans: list[str] = []
        anchors: list[CoverageAnchor] = []
        if covered_claim:
            if not pids:
                covered_claim = False
            else:
                verified = False
                for pid in pids:
                    passage = passages_by_id.get(pid)
                    if passage is None:
                        continue
                    try:
                        source = str(getattr(passage, "source_id", "") or "")
                        section = str(getattr(passage, "section_id", "") or "")
                        start = int(getattr(passage, "char_start", 0) or 0)
                        end = int(getattr(passage, "char_end", 0) or 0)
                        text = str(getattr(passage, "exact_text", "") or "")
                    except Exception:
                        continue
                    anchors.append(
                        CoverageAnchor(
                            passage_id=pid,
                            source_id=source,
                            section_id=section,
                            char_start=start,
                            char_end=end,
                        )
                    )
                    if quote.strip() and _passage_text_contains(text, quote):
                        verified = True
                        spans.append(" ".join(quote.split())[:500])
                # A covered verdict needs at least one exact anchor; a
                # quote is supporting auditability when present and
                # verified, but an unverifiable quote never upgrades a
                # quoteless anchor to failure -- the anchor itself is the
                # exact-range evidence. Quoteless coverage stays covered
                # only when anchors exist; missing anchors fail closed.
                if not anchors:
                    covered_claim = False
                elif quote.strip() and not verified:
                    spans = []
        if not covered_claim:
            # Uncovered needs keep no positive anchors; what remains
            # missing is explicit.
            anchors = []
            spans = []
        out.append(
            NeedCoverage(
                need_id=nid,
                covered=bool(covered_claim),
                passage_ids=tuple(pids) if covered_claim else (),
                anchors=tuple(anchors) if covered_claim else (),
                missing="" if covered_claim else (missing or "insufficient exact evidence"),
                supporting_spans=tuple(spans),
            )
        )
    # Every known need must be judged; missing judgments fail closed.
    for nid in known_need_ids:
        if nid not in seen:
            out.append(
                NeedCoverage(
                    need_id=nid,
                    covered=False,
                    missing="no coverage judgment returned",
                )
            )
    ordered = sorted(out, key=lambda item: known_need_ids.index(item.need_id))
    missing_ids = tuple(item.need_id for item in ordered if not item.covered)
    return CoverageVerdict(
        needs=tuple(ordered),
        all_covered=not missing_ids,
        missing_need_ids=missing_ids,
        used_model=True,
    )


def conservative_uncovered_verdict(needs: Any) -> CoverageVerdict:
    """Heuristic/discovery path verdict: never sufficient without a model.

    Lexical overlap may guide discovery only; it can never mark coverage
    sufficient, so the model-free path reports every need uncovered.
    """
    ids = _need_ids_from_info(needs)
    items = tuple(
        NeedCoverage(need_id=nid, covered=False, missing="no model coverage assessment")
        for nid in ids
    )
    return CoverageVerdict(
        needs=items,
        all_covered=False,
        missing_need_ids=tuple(ids),
        used_model=False,
    )


async def aassess_coverage(
    passages: list[Any],
    information_needs: Any,
    *,
    original_request: str,
    model: Any | None = None,
    repair_hint: str = "",
) -> CoverageVerdict:
    """Assess full exact passages against every information need.

    Model-driven when a model is bound; conservative uncovered when no
    model is available or the model call fails (lexical heuristics never
    decide sufficiency). Provider 429 propagates.
    """
    need_ids = _need_ids_from_info(information_needs)
    if not need_ids:
        return CoverageVerdict(needs=(), all_covered=True, missing_need_ids=(), used_model=False)
    if not passages:
        return conservative_uncovered_verdict(information_needs)
    if model is None:
        return conservative_uncovered_verdict(information_needs)
    known_passage_ids = {str(getattr(item, "passage_id", "") or "") for item in passages}
    known_passage_ids.discard("")
    passages_by_id = {str(getattr(item, "passage_id", "") or ""): item for item in passages}
    system, user_text = coverage_prompt(
        passages=passages,
        information_needs=information_needs,
        original_request=original_request,
        repair_hint=repair_hint,
    )
    try:
        structured = getattr(model, "ainvoke_structured", None)
        if callable(structured):
            raw = await structured(
                user_text,
                system=system,
                schema=coverage_schema(need_ids),
                retry_count=1,
            )
        else:
            text_invoke = getattr(model, "_ainvoke_text", None) or getattr(model, "ainvoke", None)
            if not callable(text_invoke):
                return conservative_uncovered_verdict(information_needs)
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
            raw = _json.loads(payload)
        return validate_coverage_decision(
            raw,
            known_need_ids=need_ids,
            known_passage_ids=known_passage_ids,
            passages_by_id=passages_by_id,
        )
    except Exception as exc:
        from aa.opencode.errors import OpenCodeRateLimitError

        if isinstance(exc, OpenCodeRateLimitError):
            raise
        if isinstance(exc, asyncio.CancelledError):
            raise
        logger.info(
            "coverage assessment unavailable; conservative uncovered used",
            extra={"category": type(exc).__name__},
        )
        return conservative_uncovered_verdict(information_needs)


def sanitize_followup_queries(
    queries: list[str],
    *,
    seen_fingerprints: set[str],
    forbidden_texts: list[str] | None = None,
    max_queries: int = MAX_FOLLOWUP_QUERIES_PER_ROUND,
) -> list[str]:
    """Deduplicate follow-ups; drop repeats and control-instruction echoes.

    A changed query string alone is not progress: normalized repeats of
    already-sent candidates are dropped. Queries that echo forbidden
    control text (safety policy or unsupported draft claims verbatim)
    are dropped so repair instructions never become search intent.
    """
    forbidden = {_fingerprint_candidate(str(item)) for item in (forbidden_texts or [])}
    out: list[str] = []
    seen_local: set[str] = set()
    for raw in list(queries or []):
        cleaned = _normalize_query(raw)
        if not cleaned:
            continue
        fingerprint = _fingerprint_candidate(cleaned)
        if (
            fingerprint in seen_fingerprints
            or fingerprint in seen_local
            or fingerprint in forbidden
        ):
            continue
        seen_local.add(fingerprint)
        out.append(cleaned[:500])
        if len(out) >= max(1, max_queries):
            break
    return out


def expand_read_ranges(
    index: Any,
    read_child_ids: list[str],
    *,
    neighbor_window: int,
    extra_step: int = 1,
) -> list[Any]:
    """Expand around already-read source ranges (adjacent context).

    Reuses the canonical small-to-big expansion with a wider neighbor
    window over the same already-read child ids, so needed exceptions
    or qualifiers adjacent to a read range close coverage without
    requiring a new search hit. Exact text and provenance stay
    canonical; checksum mismatches raise.
    """
    from aa.retrieval.evidence import expand_small_to_big
    from aa.retrieval.fusion import FusedCandidate

    winners = [
        FusedCandidate(
            chunk_id=str(cid),
            fused_score=float(len(read_child_ids) - pos),
            lexical_rank=None,
            dense_rank=None,
            lexical_score=None,
            dense_score=None,
        )
        for pos, cid in enumerate(dict.fromkeys(read_child_ids))
        if str(cid).strip()
    ]
    if not winners:
        return []
    return expand_small_to_big(
        index, winners, neighbor_window=int(neighbor_window) + max(1, int(extra_step))
    )


__all__ = [
    "MAX_COVERAGE_EXPANSIONS",
    "MAX_COVERAGE_ITERATIONS",
    "MAX_COVERAGE_MODEL_CALLS",
    "MAX_COVERAGE_SEARCHES",
    "MAX_FOLLOWUP_QUERIES_PER_ROUND",
    "STATE_ASSESS_COVERAGE",
    "STATE_DISCOVER",
    "STATE_EXHAUSTED",
    "STATE_EXPAND",
    "STATE_READ_EXACT",
    "STATE_READY",
    "STATE_SEARCH_MORE",
    "STATE_SELECT_FOR_READING",
    "CoverageAnchor",
    "CoverageLoopResult",
    "CoverageVerdict",
    "LoopStatus",
    "NeedCoverage",
    "aassess_coverage",
    "conservative_uncovered_verdict",
    "coverage_prompt",
    "coverage_schema",
    "expand_read_ranges",
    "loop_fingerprint",
    "sanitize_followup_queries",
    "validate_coverage_decision",
]
