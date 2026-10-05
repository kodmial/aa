"""Aligned RU/EN AA hierarchy core (issue #8).

Builds one language-neutral hierarchical corpus structure for the Russian
primary corpus and the English reference corpus:

```text
book
  -> section/chapter
     -> paragraph
        -> sentence/chunk
```

Section/chapter alignment is mandatory (both canons share the same twelve
neutral section ids). Paragraph/chunk alignment is never forced across
languages: translations split/merge differently, so paragraph and chunk
nodes are per-language physical units linked only by an explicit
section-level alignment policy (``section-aligned-only`` /
``unaligned-explicit``). No machine translation enters canonical storage;
every chunk round-trips to exact source text.

Public artifacts (``corpus/structure.json``, ``corpus/book-map.md``) are
metadata-only: neutral ids, bilingual display titles, concise English
topics, provenance/checksum references, and alignment status. They never
carry literary text. The full text-bearing hierarchy (paragraphs, sentences
and chunks with exact text) is written to the ignored
``corpus/generated/`` workspace for runtime retrieval.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from aa.corpus.e5_tokens import (
    CHILD_MAX_TOKENS,
    CHUNKER_ID,
    CHUNKER_VERSION,
    E5_HARD_INPUT_TOKENS,
    TOKENIZER_ID,
    ChunkTokenError,
    build_token_chunks,
)
from aa.corpus.sentences import (
    SEGMENTER_ID,
    SEGMENTER_VERSION,
    SentenceSpan,
    split_sentences_razdel,
)

BOOK_ID = "aa-big-book"
STRUCTURE_FORMAT = "aa-aligned-structure/1"
FULL_FORMAT = "aa-corpus-structure-full/2"
BUILDER_VERSION = 2
# Legacy character budget replaced by E5-token budgeting (issue #115).
LEGACY_MAX_CHUNK_CHARS = 1500

SECTION_IDS = (
    "doctors-opinion",
    "chapter-1",
    "chapter-2",
    "chapter-3",
    "chapter-4",
    "chapter-5",
    "chapter-6",
    "chapter-7",
    "chapter-8",
    "chapter-9",
    "chapter-10",
    "chapter-11",
)

SENTENCE_RULE_VERSION = SEGMENTER_VERSION
CHUNKER_RULE_VERSION = CHUNKER_ID

SECTION_ALIGNMENT_STATUS = "aligned"
PARAGRAPH_ALIGNMENT_STATUS = "section-aligned-only"
CHUNK_ALIGNMENT_STATUS = "unaligned-explicit"

ALIGNMENT_POLICY = (
    "Section/chapter alignment is mandatory and one-to-one across RU/EN. "
    "Paragraph/chunk alignment is never forced: translations split/merge "
    "differently, so paragraphs and chunks are per-language physical units "
    "under explicit section-only alignment. Retrieval uses RU chunks as "
    "primary evidence units for Russian users; EN chunks are reference/control."
)

# Concise English navigation topics (metadata only, never book text).
TOPICS_EN = {
    "doctors-opinion": "Medical view: craving and obsession as illness",
    "chapter-1": "Bill's story: descent and turning point",
    "chapter-2": "There is a solution: hope and fellowship",
    "chapter-3": "More about alcoholism: craving, allergy, obsession",
    "chapter-4": "We agnostics: willingness toward spiritual help",
    "chapter-5": "How it works: honesty and action program",
    "chapter-6": "Into action: inventory, amends, daily practice",
    "chapter-7": "Working with others: carrying the message",
    "chapter-8": "To wives: family perspective and support",
    "chapter-9": "Family afterward: rebuilding trust at home",
    "chapter-10": "To employers: workplace perspective and help",
    "chapter-11": "Vision for you: fellowship growth and invitation",
}

# Matches blocks of consecutive non-blank lines (paragraphs). Surrounding
# blank separators are not part of any paragraph span.
_PARAGRAPH_RE = re.compile(r"(?:[^\n]*\S[^\n]*)(?:\n(?!\s*\n)[^\n]*)*")


@dataclass(frozen=True)
class ParagraphSpan:
    """One paragraph slice of a section text (exact offsets)."""

    index: int  # 1-based per section+language
    char_start: int
    char_end: int
    text: str


# Re-exported for backward-compatible imports: production sentences always
# come from the qualified standard segmenter (razdel); see sentences.py.
__all__ = [
    "BOOK_ID",
    "STRUCTURE_FORMAT",
    "FULL_FORMAT",
    "BUILDER_VERSION",
    "SECTION_IDS",
    "CHILD_MAX_TOKENS",
    "SENTENCE_RULE_VERSION",
    "CHUNKER_RULE_VERSION",
    "ParagraphSpan",
    "SentenceSpan",
    "sha256_text",
    "split_paragraphs",
    "split_sentences",
    "build_chunks",
    "paragraph_id",
    "chunk_id",
    "section_prev_next",
    "build_section_units",
    "build_full_structure",
    "strip_text",
    "build_public_structure",
    "render_book_map",
    "chunker_identity",
    "segmenter_identity",
]


def chunker_identity(*, max_tokens: int = CHILD_MAX_TOKENS) -> dict[str, Any]:
    """Return chunker/tokenizer identity for index manifests."""
    from aa.corpus.e5_tokens import chunker_identity as _chunker_identity

    return _chunker_identity(max_tokens=max_tokens)


def segmenter_identity() -> dict[str, str]:
    """Return the pinned production segmenter identity."""
    return {"segmenter_id": SEGMENTER_ID, "segmenter_version": SEGMENTER_VERSION}


def sha256_text(text: str) -> str:
    """Return the hex SHA-256 digest of ``text`` (UTF-8)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def split_paragraphs(section_text: str) -> list[ParagraphSpan]:
    """Split ``section_text`` into paragraphs on blank lines.

    Offsets are section-relative and every span round-trips exactly:
    ``section_text[span.char_start:span.char_end] == span.text``.
    """
    spans: list[ParagraphSpan] = []
    for number, match in enumerate(_PARAGRAPH_RE.finditer(section_text), start=1):
        start, end = match.start(), match.end()
        spans.append(
            ParagraphSpan(
                index=number, char_start=start, char_end=end, text=section_text[start:end]
            )
        )
    if not spans and section_text.strip():
        raise ValueError("paragraph split produced no paragraphs")
    previous_end = 0
    for span in spans:
        gap = section_text[previous_end : span.char_start]
        if gap.strip():
            raise ValueError("paragraph split lost section content")
        previous_end = span.char_end
    tail = section_text[previous_end:]
    if tail.strip():
        raise ValueError("paragraph split lost trailing section content")
    return spans


def split_sentences(paragraph_text: str, base_offset: int) -> list[SentenceSpan]:
    """Split one paragraph into sentences (section-relative offsets).

    Production uses only the qualified standard Russian segmenter
    (``razdel.sentenize``); see :mod:`aa.corpus.sentences`. Every span
    round-trips exactly to the owning section text.
    """
    try:
        return split_sentences_razdel(paragraph_text, base_offset)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc


def build_chunks(
    sentences: list[SentenceSpan],
    *,
    max_tokens: int = CHILD_MAX_TOKENS,
    token_counter: Callable[[str], int] | None = None,
    paragraph_end: int | None = None,
) -> list[tuple[int, int]]:
    """Group sentence spans into chunk ``(char_start, char_end)`` ranges.

    Token-aware: sentences are never split; a chunk holds one or more whole
    adjacent sentences within one paragraph up to ``max_tokens`` E5 tokens.
    A single sentence may exceed ``max_tokens`` and remain one atomic child
    only while it still fits the E5 hard input limit; a harder breach fails
    closed. No overlapping duplicate windows are emitted.
    """
    if paragraph_end is None:
        if not sentences:
            raise ValueError("refusing to chunk an empty sentence list")
        paragraph_end = sentences[-1].char_end
    try:
        return build_token_chunks(
            sentences,
            paragraph_end=paragraph_end,
            max_tokens=max_tokens,
            hard_limit=E5_HARD_INPUT_TOKENS,
            token_counter=token_counter,
        )
    except ChunkTokenError as exc:
        raise ValueError(str(exc)) from exc


def paragraph_id(section_id: str, lang: str, index: int) -> str:
    """Return the physical paragraph id for one language."""
    return f"{section_id}:{lang}:p{index:04d}"


def chunk_id(section_id: str, lang: str, index: int) -> str:
    """Return the physical retrieval-chunk id for one language."""
    return f"{section_id}:{lang}:c{index:04d}"


def section_prev_next(section_id: str) -> tuple[str | None, str | None]:
    """Return ``(prev, next)`` neutral section ids for ``section_id``."""
    position = SECTION_IDS.index(section_id)
    prev_id = SECTION_IDS[position - 1] if position > 0 else None
    next_id = SECTION_IDS[position + 1] if position + 1 < len(SECTION_IDS) else None
    return prev_id, next_id


def build_section_units(
    *,
    section_id: str,
    lang: str,
    section_text: str,
    source_id: str,
    source_file: str,
    source_sha256: str,
    edition: str,
    corpus_version: str,
    max_tokens: int = CHILD_MAX_TOKENS,
    token_counter: Callable[[str], int] | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Build per-language paragraph/sentence/chunk units for one section.

    Every paragraph/sentence/chunk round-trips:
    ``section_text[span.char_start:span.char_end]`` is the exact span text.
    Chunks group whole adjacent sentences within one paragraph up to
    ``max_tokens`` E5 tokens. Units carry parent/previous/next links and
    alignment status; cross-language paragraph equality is never forced.
    """
    if kwargs.get("max_chars") is not None:
        raise ValueError("max_chars was replaced by E5-token max_tokens (issue #115)")
    if section_id not in SECTION_IDS:
        raise ValueError(f"unknown section id: {section_id!r}")
    if lang not in ("en", "ru"):
        raise ValueError(f"unsupported language: {lang!r}")
    paragraphs = split_paragraphs(section_text)
    paragraph_nodes: list[dict[str, object]] = []
    chunk_nodes: list[dict[str, object]] = []
    chunk_index = 0
    for para_position, para in enumerate(paragraphs):
        if section_text[para.char_start : para.char_end] != para.text:
            raise ValueError("paragraph span does not round-trip to section text")
        para_text = section_text[para.char_start : para.char_end]
        sentences = split_sentences(para_text, para.char_start)
        for sentence in sentences:
            if section_text[sentence.char_start : sentence.char_end] != sentence.text:
                raise ValueError("sentence span does not round-trip to section text")
        chunk_ranges = build_chunks(
            sentences,
            max_tokens=max_tokens,
            token_counter=token_counter,
            paragraph_end=para.char_end,
        )
        first_chunk = chunk_index + 1
        for range_start, range_end in chunk_ranges:
            chunk_index += 1
            chunk_text = section_text[range_start:range_end]
            if not chunk_text.strip():
                raise ValueError("refusing an empty chunk")
            if token_counter is not None:
                chunk_tokens = int(token_counter(chunk_text))
            else:
                from aa.corpus.e5_tokens import count_e5_tokens as _count

                chunk_tokens = int(_count(chunk_text))
            chunk_nodes.append(
                {
                    "id": chunk_id(section_id, lang, chunk_index),
                    "parent": paragraph_id(section_id, lang, para.index),
                    "section": section_id,
                    "book": BOOK_ID,
                    "prev": (
                        chunk_id(section_id, lang, chunk_index - 1) if chunk_index > 1 else None
                    ),
                    "next": None,  # linked below
                    "language": lang,
                    "source_id": source_id,
                    "source_file": source_file,
                    "source_sha256": source_sha256,
                    "edition": edition,
                    "corpus_version": corpus_version,
                    "char_start": range_start,
                    "char_end": range_end,
                    "chars": range_end - range_start,
                    "tokens": chunk_tokens,
                    "text_sha256": sha256_text(chunk_text),
                    "text": chunk_text,
                    "alignment": {
                        "status": CHUNK_ALIGNMENT_STATUS,
                        "confidence": 1.0,
                    },
                    "role": ("primary-retrieval" if lang == "ru" else "reference-control"),
                }
            )
        last_chunk = chunk_index
        para_node_id = paragraph_id(section_id, lang, para.index)
        paragraph_nodes.append(
            {
                "id": para_node_id,
                "parent": section_id,
                "section": section_id,
                "book": BOOK_ID,
                "prev": (
                    paragraph_id(section_id, lang, para.index - 1) if para_position > 0 else None
                ),
                "next": None,  # linked below
                "language": lang,
                "source_id": source_id,
                "source_file": source_file,
                "source_sha256": source_sha256,
                "edition": edition,
                "corpus_version": corpus_version,
                "char_start": para.char_start,
                "char_end": para.char_end,
                "chars": para.char_end - para.char_start,
                "text_sha256": sha256_text(para.text),
                "text": para.text,
                "sentences": len(sentences),
                "chunks": [
                    chunk_id(section_id, lang, i) for i in range(first_chunk, last_chunk + 1)
                ],
                "alignment": {
                    "status": PARAGRAPH_ALIGNMENT_STATUS,
                    "confidence": 1.0,
                },
            }
        )
    for first, second in zip(paragraph_nodes, paragraph_nodes[1:], strict=False):
        first["next"] = second["id"]
    for first, second in zip(chunk_nodes, chunk_nodes[1:], strict=False):
        first["next"] = second["id"]
    return {"paragraphs": paragraph_nodes, "chunks": chunk_nodes}


def build_full_structure(
    *,
    en_sections: list[dict[str, object]],
    ru_sections: list[dict[str, object]],
    en_edition: str,
    ru_edition: str,
    en_corpus_version: str,
    ru_corpus_version: str,
    max_tokens: int = CHILD_MAX_TOKENS,
    token_counter: Callable[[str], int] | None = None,
    **kwargs: Any,
) -> dict[str, object]:
    """Build the full text-bearing hierarchy for EN+RU section inputs.

    Each input section needs ``id``, ``title``, ``text``, ``source_id``,
    ``source_file`` and ``source_sha256``. Section ids must match exactly
    across languages in canonical order; paragraph/chunk counts may differ
    (split/merge is represented, never forced).
    """
    if kwargs.get("max_chars") is not None:
        raise ValueError("max_chars was replaced by E5-token max_tokens (issue #115)")
    en_by_id = {str(item["id"]): item for item in en_sections}
    ru_by_id = {str(item["id"]): item for item in ru_sections}
    if [str(item["id"]) for item in en_sections] != list(SECTION_IDS):
        raise ValueError("EN sections are not exactly the canonical twelve")
    if [str(item["id"]) for item in ru_sections] != list(SECTION_IDS):
        raise ValueError("RU sections are not exactly the canonical twelve")
    sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_item = en_by_id[section_id]
        ru_item = ru_by_id[section_id]
        en_text = str(en_item["text"])
        ru_text = str(ru_item["text"])
        if not en_text.strip() or not ru_text.strip():
            raise ValueError(f"section {section_id!r} has blank text in a language")
        prev_id, next_id = section_prev_next(section_id)
        en_units = build_section_units(
            section_id=section_id,
            lang="en",
            section_text=en_text,
            source_id=str(en_item["source_id"]),
            source_file=str(en_item["source_file"]),
            source_sha256=str(en_item["source_sha256"]),
            edition=en_edition,
            corpus_version=en_corpus_version,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        ru_units = build_section_units(
            section_id=section_id,
            lang="ru",
            section_text=ru_text,
            source_id=str(ru_item["source_id"]),
            source_file=str(ru_item["source_file"]),
            source_sha256=str(ru_item["source_sha256"]),
            edition=ru_edition,
            corpus_version=ru_corpus_version,
            max_tokens=max_tokens,
            token_counter=token_counter,
        )
        sections.append(
            {
                "id": section_id,
                "book": BOOK_ID,
                "parent": BOOK_ID,
                "prev": prev_id,
                "next": next_id,
                "titles": {"en": str(en_item["title"]), "ru": str(ru_item["title"])},
                "topic_en": TOPICS_EN[section_id],
                "alignment": {"status": SECTION_ALIGNMENT_STATUS, "confidence": 1.0},
                "en": {
                    "language": "en",
                    "source_id": str(en_item["source_id"]),
                    "source_file": str(en_item["source_file"]),
                    "source_sha256": str(en_item["source_sha256"]),
                    "edition": en_edition,
                    "corpus_version": en_corpus_version,
                    "chars": len(en_text),
                    "text_sha256": sha256_text(en_text),
                    "paragraphs": en_units["paragraphs"],
                    "chunks": en_units["chunks"],
                },
                "ru": {
                    "language": "ru",
                    "source_id": str(ru_item["source_id"]),
                    "source_file": str(ru_item["source_file"]),
                    "source_sha256": str(ru_item["source_sha256"]),
                    "edition": ru_edition,
                    "corpus_version": ru_corpus_version,
                    "chars": len(ru_text),
                    "text_sha256": sha256_text(ru_text),
                    "paragraphs": ru_units["paragraphs"],
                    "chunks": ru_units["chunks"],
                },
            }
        )
    return {
        "format": FULL_FORMAT,
        "builder_version": BUILDER_VERSION,
        "book": BOOK_ID,
        "sentence_rule": SENTENCE_RULE_VERSION,
        "sentence_segmenter": SEGMENTER_ID,
        "chunker": CHUNKER_ID,
        "chunker_version": CHUNKER_VERSION,
        "tokenizer": TOKENIZER_ID,
        "chunk_policy": "adjacent-sentences-within-paragraph",
        "chunk_max_tokens": max_tokens,
        "e5_hard_input_tokens": E5_HARD_INPUT_TOKENS,
        "alignment_policy": ALIGNMENT_POLICY,
        "sections": sections,
    }


def strip_text(value: object) -> object:
    """Return ``value`` with every ``text`` payload removed (public view)."""
    if isinstance(value, dict):
        return {key: strip_text(item) for key, item in value.items() if key != "text"}
    if isinstance(value, list):
        return [strip_text(item) for item in value]
    return value


def build_public_structure(
    *,
    en_manifest: dict[str, object],
    ru_manifest: dict[str, object],
) -> dict[str, object]:
    """Build the public metadata-only aligned structure (no literary text).

    Derived from the committed manifests only, so it is reproducible without
    the private book text. Paragraph/chunk instances stay in the generated
    full hierarchy; the public file carries section alignment, bilingual
    titles, English topics, provenance/checksum references, and the
    split/merge policy.
    """
    en_sections_raw = en_manifest.get("sections")
    ru_sections_raw = ru_manifest.get("sections")
    if not isinstance(en_sections_raw, list) or not isinstance(ru_sections_raw, list):
        raise ValueError("manifests have no sections lists")
    en_by_id = {str(item["id"]): item for item in en_sections_raw if isinstance(item, dict)}
    ru_by_id = {str(item["id"]): item for item in ru_sections_raw if isinstance(item, dict)}
    if sorted(en_by_id) != sorted(SECTION_IDS) or sorted(ru_by_id) != sorted(SECTION_IDS):
        raise ValueError("manifest sections are not the canonical twelve")
    sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_item = en_by_id[section_id]
        ru_item = ru_by_id[section_id]
        assert isinstance(en_item, dict) and isinstance(ru_item, dict)
        prev_id, next_id = section_prev_next(section_id)
        en_entry: dict[str, object] = {
            "language": "en",
            "title": str(en_item.get("title")),
            "text_sha256": str(en_item.get("text_sha256")),
            "source_id": str(en_item.get("source_id", "core-pages-1-164")),
        }
        if en_item.get("chars") is not None:
            en_entry["chars"] = en_item.get("chars")
        if en_item.get("byte_start") is not None:
            en_entry["byte_start"] = en_item.get("byte_start")
        if en_item.get("byte_end") is not None:
            en_entry["byte_end"] = en_item.get("byte_end")
        ru_entry: dict[str, object] = {
            "language": "ru",
            "title": str(ru_item.get("title")),
            "text_sha256": str(ru_item.get("text_sha256")),
            "source_id": "ru-fourth-edition-txt",
        }
        if ru_item.get("chars") is not None:
            ru_entry["chars"] = ru_item.get("chars")
        sections.append(
            {
                "id": section_id,
                "book": BOOK_ID,
                "parent": BOOK_ID,
                "prev": prev_id,
                "next": next_id,
                "titles": {"en": str(en_item.get("title")), "ru": str(ru_item.get("title"))},
                "topic_en": TOPICS_EN[section_id],
                "alignment": {"status": SECTION_ALIGNMENT_STATUS, "confidence": 1.0},
                "paragraph_alignment": {
                    "status": PARAGRAPH_ALIGNMENT_STATUS,
                    "confidence": 1.0,
                    "note": (
                        "Paragraph/chunk alignment is explicitly not forced "
                        "across RU/EN; see alignment_policy."
                    ),
                },
                "chunk_roles": {"ru": "primary-retrieval", "en": "reference-control"},
                "en": en_entry,
                "ru": ru_entry,
            }
        )
    return {
        "format": STRUCTURE_FORMAT,
        "builder_version": BUILDER_VERSION,
        "book": BOOK_ID,
        "sentence_rule": SENTENCE_RULE_VERSION,
        "sentence_segmenter": SEGMENTER_ID,
        "chunker": CHUNKER_ID,
        "chunker_version": CHUNKER_VERSION,
        "tokenizer": TOKENIZER_ID,
        "chunk_policy": "adjacent-sentences-within-paragraph",
        "chunk_max_tokens": CHILD_MAX_TOKENS,
        "e5_hard_input_tokens": E5_HARD_INPUT_TOKENS,
        "alignment_policy": ALIGNMENT_POLICY,
        "chunking": {
            "paragraph_rule": "blank-line-separated blocks; exact section offsets",
            "sentence_rule": SENTENCE_RULE_VERSION,
            "sentence_segmenter": SEGMENTER_ID,
            "chunker": CHUNKER_ID,
            "chunk_policy": "adjacent-sentences-within-paragraph",
            "chunk_max_tokens": CHILD_MAX_TOKENS,
            "tokenizer": TOKENIZER_ID,
            "note": (
                "Chunks group whole adjacent sentences within one paragraph only "
                "up to 256 E5 tokens; sentences are never split. Paragraph/chunk "
                "instances with exact text live in corpus/generated/ and never "
                "in this file."
            ),
        },
        "en": {
            "manifest_format": str(en_manifest.get("format")),
            "artifact_sha256": str(en_manifest.get("artifact_sha256")),
            "edition": str(en_manifest.get("edition")),
        },
        "ru": {
            "manifest_format": str(ru_manifest.get("format")),
            "artifact_sha256": str(ru_manifest.get("artifact_sha256")),
            "edition": str(ru_manifest.get("edition")),
        },
        "sections": sections,
    }


def render_book_map(public_structure: dict[str, object]) -> str:
    """Render the compact primarily-English navigation map (never evidence)."""
    sections = public_structure.get("sections")
    if not isinstance(sections, list):
        raise ValueError("public structure has no sections list")
    en_version = public_structure.get("en")
    ru_version = public_structure.get("ru")
    en_sha = en_version.get("artifact_sha256") if isinstance(en_version, dict) else "?"
    ru_sha = ru_version.get("artifact_sha256") if isinstance(ru_version, dict) else "?"
    lines = [
        "# AA book map (navigation only, never evidence)",
        "",
        "Use language-neutral section ids to route retrieval. Read exact",
        "source passages with the book tools before making AA claims.",
        "RU chunks are primary evidence for Russian users; EN is reference.",
        "",
        f"versions: en={str(en_sha)[:12]} ru={str(ru_sha)[:12]}",
        "",
        "| section | EN title | topic |",
        "|---|---|---|",
    ]
    for entry in sections:
        if not isinstance(entry, dict):
            raise ValueError("public structure has a malformed section")
        section_id = str(entry.get("id"))
        titles = entry.get("titles")
        topic = str(entry.get("topic_en"))
        en_title = titles.get("en") if isinstance(titles, dict) else "?"
        lines.append(f"| {section_id} | {en_title} | {topic} |")
    lines.extend(
        [
            "",
            "Map budget: within the 6000-token compact-map budget (see docs/context-budget.md).",
            "",
        ]
    )
    return "\n".join(lines)
