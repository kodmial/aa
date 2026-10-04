"""Narrow RU-first OpenCode book tools over the shared hybrid index (issue #18).

The four tools (``book_search``, ``book_read``, ``book_expand``,
``book_section``) share this single retrieval/read implementation over
the qualified RU-first stack from #17/#47:

- ``book_search`` fuses one Russian semantic query plus Russian lexical
  queries through the fixed server-side ranking parameters (lexical top
  40, dense top 40, RRF ``k=60``, max 12 candidates). The model cannot
  choose index names, fusion weights or an unbounded top-K. Results are
  compact navigation candidates with language-neutral logical IDs and
  RU source locators; preview text is navigation-only and never answer
  evidence.
- ``book_read`` returns exact RU canonical text for one logical or
  physical chunk ID with provenance, checksum and version.
- ``book_expand`` returns bounded neighboring exact RU chunks (hard max
  3 before + 3 after).
- ``book_section`` returns a bounded/paginated run of whole exact RU
  chunks for one section under the retrieved-passages token ceiling.

English control: the #47 decision (``ru-first-only`` /
``ru-first-production-v1``) keeps the aligned EN secondary discovery
branch disabled (zero incremental recall at 1.54x latency). Hence
``lexical_query_en`` must stay null; any non-null value fails closed.
EN metadata in search hits is navigation/control data only
(``role: reference-control``). All evidence loading resolves through
aligned RU IDs to exact RU text; generated translation is never
surfaced as source evidence.

Version, provenance and output limits fail closed. Logs carry only
tool names, IDs, counts, sizes and digests, never raw user queries or
corpus text.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS, estimate_text_tokens
from aa.corpus.structure import SECTION_IDS
from aa.retrieval.dense import DENSE_TOP_K
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT, RRF_K
from aa.retrieval.index import HybridIndex, logical_chunk_id, search_aspect
from aa.retrieval.lexical import LEXICAL_TOP_K

logger = logging.getLogger("aa.retrieval.book_tools")

TOOL_NAMES = ("book_search", "book_read", "book_expand", "book_section")

MAX_ASPECT_ID_CHARS = 64
MAX_QUERY_CHARS = 500
MAX_QUERIES_PER_FIELD = 8
MAX_EXPAND_BEFORE = 3
MAX_EXPAND_AFTER = 3
MAX_SECTION_CHUNKS_PER_CALL = 12

EN_SECONDARY_ENABLED = False
EN_DISABLED_REASON = (
    "aligned EN secondary discovery is disabled by the #47 decision "
    "(ru-first-only / ru-first-production-v1): zero incremental recall "
    "at 1.54x latency; RU-only retrieval is production"
)


class BookToolError(ValueError):
    """Raised when a book tool request fails validation (fails closed)."""


class BookNotFoundError(BookToolError):
    """Raised when a chunk/section ID does not resolve to RU evidence."""


class BookStaleError(BookToolError):
    """Raised when a version pin no longer matches the opened index."""


@dataclass(frozen=True)
class SearchRequest:
    """Validated book_search input (compact RU-first schema)."""

    aspect_id: str
    semantic_query_ru: str
    lexical_queries_ru: tuple[str, ...]


def _require_ru_query(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BookToolError(f"{field} must be a non-empty Russian string")
    text = value.strip()
    if len(text) > MAX_QUERY_CHARS:
        raise BookToolError(f"{field} exceeds {MAX_QUERY_CHARS} chars")
    return text


def validate_search_input(payload: object) -> SearchRequest:
    """Validate book_search input; reject EN branch and ranking overrides."""
    if not isinstance(payload, dict):
        raise BookToolError("book_search input must be a JSON object")
    for forbidden in ("index", "index_name", "top_k", "topK", "fusion", "rrf_k", "weights"):
        if forbidden in payload:
            raise BookToolError(f"book_search rejects server-side parameter {forbidden!r}")
    aspect_raw = payload.get("aspect_id")
    if not isinstance(aspect_raw, str) or not aspect_raw.strip():
        raise BookToolError("aspect_id must be a non-empty string")
    aspect_id = aspect_raw.strip()
    if len(aspect_id) > MAX_ASPECT_ID_CHARS:
        raise BookToolError(f"aspect_id exceeds {MAX_ASPECT_ID_CHARS} chars")
    semantic = _require_ru_query(payload.get("semantic_query_ru"), field="semantic_query_ru")
    lexical_raw = payload.get("lexical_queries_ru")
    if not isinstance(lexical_raw, list) or not lexical_raw:
        raise BookToolError("lexical_queries_ru must be a non-empty list of strings")
    if len(lexical_raw) > MAX_QUERIES_PER_FIELD:
        raise BookToolError(f"lexical_queries_ru exceeds {MAX_QUERIES_PER_FIELD} queries")
    lexical = tuple(_require_ru_query(item, field="lexical_queries_ru[]") for item in lexical_raw)
    if "lexical_query_en" in payload and payload["lexical_query_en"] is not None:
        raise BookToolError(f"lexical_query_en must stay null: {EN_DISABLED_REASON}")
    return SearchRequest(
        aspect_id=aspect_id, semantic_query_ru=semantic, lexical_queries_ru=lexical
    )


def _fused_ru_queries(request: SearchRequest) -> list[str]:
    fused: list[str] = []
    seen: set[str] = set()
    for query in (request.semantic_query_ru, *request.lexical_queries_ru):
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            fused.append(query)
    return fused


def _index_versions(index: HybridIndex) -> dict[str, Any]:
    return {
        "index_version": int(index.metadata.get("builder_version", 1)),
        "ru_corpus_version": str(index.metadata.get("ru_artifact_sha256", "")),
        "embedding_model": (
            f"{index.metadata.get('embedding_model_id')}"
            f"@{str(index.metadata.get('embedding_revision', ''))[:12]}"
        ),
    }


def book_search(index: HybridIndex, payload: object) -> dict[str, Any]:
    """Run one RU-first aspect search; return compact navigation candidates."""
    request = validate_search_input(payload)
    queries = _fused_ru_queries(request)
    hits = search_aspect(
        index,
        queries,
        lexical_top_k=LEXICAL_TOP_K,
        dense_top_k=DENSE_TOP_K,
        rrf_k=RRF_K,
        max_n=MAX_CANDIDATES_PER_ASPECT,
    )
    logger.info(
        "book_search aspect_id=%s hits=%d",
        request.aspect_id,
        len(hits),
    )
    return {
        "tool": "book_search",
        "aspect_id": request.aspect_id,
        "candidates": [hit.to_dict() for hit in hits],
        "versions": _index_versions(index),
        "ranking": {
            "lexical_top_k": LEXICAL_TOP_K,
            "dense_top_k": DENSE_TOP_K,
            "rrf_k": RRF_K,
            "max_per_aspect": MAX_CANDIDATES_PER_ASPECT,
        },
        "en_secondary": {"enabled": EN_SECONDARY_ENABLED, "role": "reference-control"},
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_chunk(index: HybridIndex, chunk_id: object) -> Any:
    """Resolve a logical or physical RU chunk ID to its record (fails closed)."""
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise BookToolError("chunk_id must be a non-empty string")
    wanted = chunk_id.strip()
    record = index.chunks.get(wanted)
    if record is not None:
        return record
    for candidate in index.chunks.values():
        if candidate.logical_chunk_id == wanted:
            return candidate
        if logical_chunk_id(wanted) == candidate.logical_chunk_id and wanted == (
            candidate.logical_chunk_id
        ):
            return candidate
    raise BookNotFoundError(f"unknown RU chunk id: {wanted!r}")


def _check_ru_version(index: HybridIndex, expected: object) -> None:
    if expected is None:
        return
    if not isinstance(expected, str) or not expected:
        raise BookToolError("expected_ru_version must be a non-empty string when given")
    live = str(index.metadata.get("ru_artifact_sha256", ""))
    if expected != live:
        raise BookStaleError("RU corpus version is stale for the opened index")


def _read_payload(record: Any, *, index: HybridIndex, tool: str) -> dict[str, Any]:
    if _sha256_text(record.text) != record.text_sha256:
        raise BookToolError(f"RU chunk checksum mismatch: {record.chunk_id!r}")
    if ":ru:" not in record.chunk_id:
        raise BookToolError(f"refusing non-RU evidence chunk: {record.chunk_id!r}")
    return {
        "tool": tool,
        "logical_chunk_id": record.logical_chunk_id,
        "chunk_id": record.chunk_id,
        "section": record.section,
        "text": record.text,
        "ru_locator": {
            "source_id": record.source_id,
            "source_file": record.source_file,
            "char_start": record.char_start,
            "char_end": record.char_end,
            "text_sha256": record.text_sha256,
        },
        "neighbors": {"parent": record.parent, "prev": record.prev, "next": record.next},
        "versions": _index_versions(index),
    }


def book_read(
    index: HybridIndex,
    chunk_id: object,
    *,
    expected_ru_version: object = None,
) -> dict[str, Any]:
    """Read exact RU canonical text for one chunk ID (fails closed)."""
    _check_ru_version(index, expected_ru_version)
    record = resolve_chunk(index, chunk_id)
    logger.info(
        "book_read chunk_id=%s chars=%d",
        record.logical_chunk_id,
        len(record.text),
    )
    return _read_payload(record, index=index, tool="book_read")


def _ordered_section_records(index: HybridIndex, section_id: str) -> list[Any]:
    if section_id not in SECTION_IDS:
        raise BookNotFoundError(f"unknown section id: {section_id!r}")
    ordered = sorted(
        (record for record in index.chunks.values() if record.section == section_id),
        key=lambda item: (item.char_start, item.char_end, item.chunk_id),
    )
    if not ordered:
        raise BookNotFoundError(f"section has no RU chunks: {section_id!r}")
    return ordered


def _validate_expand_window(value: object, *, field: str, hard_max: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BookToolError(f"{field} must be an integer 0..{hard_max}")
    if value < 0 or value > hard_max:
        raise BookToolError(f"{field} must be within 0..{hard_max}")
    return value


def book_expand(
    index: HybridIndex,
    chunk_id: object,
    *,
    before: object = 1,
    after: object = 1,
    expected_ru_version: object = None,
) -> dict[str, Any]:
    """Return bounded neighboring exact RU chunks around one chunk."""
    _check_ru_version(index, expected_ru_version)
    before_n = _validate_expand_window(before, field="before", hard_max=MAX_EXPAND_BEFORE)
    after_n = _validate_expand_window(after, field="after", hard_max=MAX_EXPAND_AFTER)
    center = resolve_chunk(index, chunk_id)
    ordered = _ordered_section_records(index, center.section)
    position = next(i for i, item in enumerate(ordered) if item.chunk_id == center.chunk_id)
    start = max(0, position - before_n)
    stop = min(len(ordered), position + after_n + 1)
    window = ordered[start:stop]
    logger.info(
        "book_expand chunk_id=%s before=%d after=%d window=%d",
        center.logical_chunk_id,
        before_n,
        after_n,
        len(window),
    )
    return {
        "tool": "book_expand",
        "center": center.logical_chunk_id,
        "section": center.section,
        "before": before_n,
        "after": after_n,
        "chunks": [_read_payload(item, index=index, tool="book_expand") for item in window],
        "versions": _index_versions(index),
    }


def _validate_section_window(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BookToolError(f"{field} must be an integer {minimum}..{maximum}")
    if value < minimum or value > maximum:
        raise BookToolError(f"{field} must be within {minimum}..{maximum}")
    return value


def book_section(
    index: HybridIndex,
    section_id: object,
    *,
    chunk_offset: object = 0,
    chunk_limit: object = 8,
    expected_ru_version: object = None,
) -> dict[str, Any]:
    """Return a bounded paginated run of whole exact RU chunks for one section."""
    _check_ru_version(index, expected_ru_version)
    if not isinstance(section_id, str) or not section_id.strip():
        raise BookToolError("section_id must be a non-empty string")
    ordered = _ordered_section_records(index, section_id.strip())
    offset = _validate_section_window(
        chunk_offset, field="chunk_offset", minimum=0, maximum=len(ordered)
    )
    limit = _validate_section_window(
        chunk_limit, field="chunk_limit", minimum=1, maximum=MAX_SECTION_CHUNKS_PER_CALL
    )
    window = ordered[offset : offset + limit]
    if not window:
        raise BookToolError("section read is empty for the requested offset")
    total_tokens = sum(estimate_text_tokens(item.text) for item in window)
    if total_tokens > RETRIEVED_PASSAGES_BUDGET_TOKENS:
        raise BookToolError(
            f"section read needs {total_tokens} tokens "
            f"but the retrieved-passages budget is {RETRIEVED_PASSAGES_BUDGET_TOKENS}"
        )
    next_offset = offset + len(window) if offset + len(window) < len(ordered) else None
    logger.info(
        "book_section section_id=%s offset=%d limit=%d returned=%d tokens=%d",
        section_id,
        offset,
        limit,
        len(window),
        total_tokens,
    )
    return {
        "tool": "book_section",
        "section": section_id,
        "chunk_offset": offset,
        "chunk_limit": limit,
        "total_chunks": len(ordered),
        "next_chunk_offset": next_offset,
        "source_tokens": total_tokens,
        "chunks": [_read_payload(item, index=index, tool="book_section") for item in window],
        "versions": _index_versions(index),
    }
