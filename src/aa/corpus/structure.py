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
from dataclasses import dataclass

BOOK_ID = "aa-big-book"
STRUCTURE_FORMAT = "aa-aligned-structure/1"
FULL_FORMAT = "aa-corpus-structure-full/1"
BUILDER_VERSION = 1

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

MAX_CHUNK_CHARS = 1500
SENTENCE_RULE_VERSION = "sentence-rule/1"

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
# A sentence ends at terminal punctuation plus optional closers.
_SENTENCE_END_RE = re.compile(r"[.!?…]+[\"'\"'\u00bb\)\]]*")
_OPENERS = set('"\'"("«[')
_SENTENCE_WS = set(" \t\r\n")


@dataclass(frozen=True)
class ParagraphSpan:
    """One paragraph slice of a section text (exact offsets)."""

    index: int  # 1-based per section+language
    char_start: int
    char_end: int
    text: str


@dataclass(frozen=True)
class SentenceSpan:
    """One sentence slice of a paragraph (section-relative offsets)."""

    index: int  # 1-based per paragraph
    char_start: int
    char_end: int
    text: str


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

    Boundaries are natural sentence punctuation only; sentences tile the
    paragraph exactly so chunks built from them never split a sentence.
    """
    if base_offset < 0:
        raise ValueError("base_offset must be >= 0")
    if not paragraph_text:
        raise ValueError("refusing to split an empty paragraph")
    boundaries: list[int] = []
    start = 0
    text_len = len(paragraph_text)
    for match in _SENTENCE_END_RE.finditer(paragraph_text):
        end_punct = match.end()
        cursor = end_punct
        while cursor < text_len and paragraph_text[cursor] in _SENTENCE_WS:
            cursor += 1
        if cursor >= text_len:
            boundaries.append(text_len)
            start = text_len
            break
        next_char = paragraph_text[cursor]
        if next_char.isupper() or next_char.isdigit() or next_char in _OPENERS:
            boundaries.append(cursor)
            start = cursor
    if start < text_len:
        boundaries.append(text_len)
    sentences: list[SentenceSpan] = []
    cursor_start = 0
    for number, end in enumerate(boundaries, start=1):
        if end <= cursor_start:
            continue
        sentences.append(
            SentenceSpan(
                index=number,
                char_start=base_offset + cursor_start,
                char_end=base_offset + end,
                text=paragraph_text[cursor_start:end],
            )
        )
        cursor_start = end
    if not sentences:
        raise ValueError("sentence split produced no sentences")
    return sentences


def build_chunks(
    sentences: list[SentenceSpan],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[tuple[int, int]]:
    """Group sentence spans into chunk ``(char_start, char_end)`` ranges.

    Sentences are never split; a chunk holds one or more whole sentences up
    to ``max_chars``. A single oversize sentence becomes its own chunk (it
    must be re-chunked upstream, never truncated here).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be > 0")
    if not sentences:
        raise ValueError("refusing to chunk an empty sentence list")
    chunks: list[tuple[int, int]] = []
    current_start = sentences[0].char_start
    current_end = sentences[0].char_end
    for previous, current in zip(sentences, sentences[1:], strict=False):
        if previous.char_end != current.char_start:
            raise ValueError("sentences must tile the paragraph contiguously")
        candidate_len = current.char_end - current_start
        if candidate_len > max_chars and current_end > current_start:
            chunks.append((current_start, current_end))
            current_start = current.char_start
        current_end = current.char_end
    chunks.append((current_start, current_end))
    return chunks


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
    max_chars: int = MAX_CHUNK_CHARS,
) -> dict[str, object]:
    """Build per-language paragraph/sentence/chunk units for one section.

    Every chunk round-trips: ``section_text[chunk_start:chunk_end]`` is the
    exact chunk text. Units carry parent/previous/next links and alignment
    status; cross-language paragraph equality is never forced.
    """
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
        chunk_ranges = build_chunks(sentences, max_chars=max_chars)
        first_chunk = chunk_index + 1
        for range_start, range_end in chunk_ranges:
            chunk_index += 1
            chunk_text = section_text[range_start:range_end]
            if not chunk_text.strip():
                raise ValueError("refusing an empty chunk")
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
    max_chars: int = MAX_CHUNK_CHARS,
) -> dict[str, object]:
    """Build the full text-bearing hierarchy for EN+RU section inputs.

    Each input section needs ``id``, ``title``, ``text``, ``source_id``,
    ``source_file`` and ``source_sha256``. Section ids must match exactly
    across languages in canonical order; paragraph/chunk counts may differ
    (split/merge is represented, never forced).
    """
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
            max_chars=max_chars,
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
            max_chars=max_chars,
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
        "max_chunk_chars": max_chars,
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
        "max_chunk_chars": MAX_CHUNK_CHARS,
        "alignment_policy": ALIGNMENT_POLICY,
        "chunking": {
            "paragraph_rule": "blank-line-separated blocks; exact section offsets",
            "sentence_rule": SENTENCE_RULE_VERSION,
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "note": (
                "Chunks group whole sentences within one paragraph only; "
                "sentences are never split. Paragraph/chunk instances with "
                "exact text live in corpus/generated/ and never in this file."
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
