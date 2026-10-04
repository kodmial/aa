#!/usr/bin/env python3
"""Build the deterministic hierarchical AA corpus structure and book map.

Reads the canonical runtime artifact produced by the issue-#3 pipeline
(``scripts/fetch_aa_source.py`` + ``scripts/build_canonical.py``) and
derives a stable navigation hierarchy used by retrieval and by the
OpenCode agent for immediate orientation:

.. code-block:: text

    book
      -> section/chapter
         -> paragraph
            -> sentence
               -> retrieval chunk/range

The canonical text itself is never modified, shortened, or paraphrased.
Every paragraph, sentence, and retrieval chunk maps back to exact
canonical source offsets, so each unit round-trips to exact source text.

Public outputs (versioned in Git, metadata only, no literary text):

- ``corpus/structure.json`` -- IDs, hierarchy, offsets, and provenance
  metadata without the full literary text;
- ``corpus/book-map.md`` -- compact chapter/section navigation and topic
  map for the always-loaded agent context;
- ``corpus/structure-report.json`` -- deterministic size/token report.

Private runtime output (ignored by Git, contains substantial text):

- ``corpus/generated/corpus-chunks.json`` -- retrieval chunks with exact
  source text for downstream indexes/tools.

This script never downloads anything. If the canonical artifact is
absent it fails closed and points at the repository bootstrap contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "corpus" / "generated" / "canonical.json"
DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_STRUCTURE = ROOT / "corpus" / "structure.json"
DEFAULT_BOOK_MAP = ROOT / "corpus" / "book-map.md"
DEFAULT_REPORT = ROOT / "corpus" / "structure-report.json"
DEFAULT_CHUNKS_OUT = ROOT / "corpus" / "generated" / "corpus-chunks.json"
RESTORE_ENTRY_POINT = ROOT / "scripts" / "restore_canonical.py"

STRUCTURE_FORMAT = "aa-corpus-structure/1"
STRUCTURE_BUILDER_VERSION = 1
CHUNKS_FORMAT = "aa-corpus-chunks/1"
BOOK_ID = "aa-book"
BOOK_TITLE = (
    "Alcoholics Anonymous, First Edition (1939) "
    "(canonical scope: The Doctor's Opinion + Chapters 1-11)"
)

DOCTORS_OPINION_ID = "doctors-opinion"
DOCTORS_OPINION_TITLE = "The Doctor's Opinion"

CHAPTER_TITLES = (
    "BILL'S STORY",
    "THERE IS A SOLUTION",
    "MORE ABOUT ALCOHOLISM",
    "WE AGNOSTICS",
    "HOW IT WORKS",
    "INTO ACTION",
    "WORKING WITH OTHERS",
    "TO WIVES",
    "THE FAMILY AFTERWARD",
    "TO EMPLOYERS",
    "A VISION FOR YOU",
)

EXPECTED_SECTION_IDS = (
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

CHAPTER_HEADING = re.compile(r"Chapter\s+(\d+)")

# Retrieval chunking bounds (characters of exact canonical text).
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 500
# A trailing chunk smaller than this is merged into its predecessor so a
# section never ends in a tiny orphan chunk (unless it is the only chunk).
TAIL_MERGE_CHARS = 250

# Sentence split: whitespace after end punctuation (optionally via a
# closing quote) followed by a new sentence start. Abbreviation guards in
# _is_abbreviation_split() keep e.g. "Dr. Smith" on one sentence.
SENTENCE_BREAK = re.compile(
    r"(?<=[.!?\"'\"'\u201d\u2019])\s+(?=[\"'\"'(\u201c\u2018\[]?[A-Za-z0-9])"
)

_ABBREVIATIONS = frozenset(
    {
        "Mr",
        "Mrs",
        "Ms",
        "Dr",
        "St",
        "Jr",
        "Sr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "a.m",
        "p.m",
        "No",
    }
)


def sha256_hex(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def estimate_tokens(char_count: int) -> int:
    """Conservative token estimate (ceil(chars/4), mirroring budget.py)."""
    if char_count < 0:
        raise ValueError("char_count must be >= 0")
    return -(-char_count // 4)


def fail(message: str) -> int:
    """Report a fail-closed build error and return the exit status."""
    print(f"corpus structure build failed: {message}", file=sys.stderr)
    return 1


def _is_abbreviation_split(left: str) -> bool:
    """Return True when the split point follows a known abbreviation."""
    stripped = left.rstrip()
    if re.search(r"(?:[A-Za-z]\.){2,}[\"'\"'\u201d\u2019]?$", stripped):
        return True
    if re.search(r"\b[A-Z]\.[\"'\"'\u201d\u2019]?$", stripped):
        return True
    token = stripped.rsplit(None, 1)[-1] if stripped else ""
    while token and token[-1] in "\"'\"'\u201d\u2019])}.,:;":
        token = token[:-1]
    core = token[:-1] if token.endswith(".") else token
    return core in _ABBREVIATIONS


def split_sentences(paragraph_text: str) -> list[tuple[int, int]]:
    """Split paragraph text into ``(start, end)`` sentence spans.

    Spans tile the paragraph text up to whitespace: every span slices the
    exact paragraph substring, and the text between consecutive spans is
    whitespace only, so no sentence is ever cut mid-token.
    """
    spans: list[tuple[int, int]] = []
    start = 0
    length = len(paragraph_text)
    for match in SENTENCE_BREAK.finditer(paragraph_text):
        if _is_abbreviation_split(paragraph_text[: match.start()]):
            continue
        end = match.start()
        piece = paragraph_text[start:end]
        if piece.strip():
            leading = len(piece) - len(piece.lstrip())
            trailing = len(piece) - len(piece.rstrip())
            spans.append((start + leading, end - trailing))
            start = end
    tail = paragraph_text[start:]
    if tail.strip():
        leading = len(tail) - len(tail.lstrip())
        trailing = len(tail) - len(tail.rstrip())
        spans.append((start + leading, length - trailing))
    if not spans:
        raise ValueError("paragraph produced no sentences")
    # Verify tiling: gaps between spans are whitespace only.
    for (prev_start, prev_end), (cur_start, cur_end) in zip(spans, spans[1:], strict=False):
        _ = prev_start
        gap = paragraph_text[prev_end:cur_start]
        if gap.strip():
            raise ValueError(f"sentence spans skip non-whitespace text: {gap!r}")
        if not (prev_end <= cur_start <= cur_end):
            raise ValueError("sentence spans are not ordered")
    return spans


class ParagraphSpan:
    """One body paragraph with exact canonical section offsets."""

    __slots__ = ("index", "text", "char_start", "char_end")

    def __init__(self, *, index: int, text: str, char_start: int, char_end: int) -> None:
        self.index = index
        self.text = text
        self.char_start = char_start
        self.char_end = char_end


def split_paragraphs(section_id: str, section_text: str) -> list[ParagraphSpan]:
    """Split a canonical section into body paragraphs with exact offsets.

    The source uses one indented physical line per paragraph (``\\r\\n``
    separated for chapters, ``\\n\\n`` separated for The Doctor's Opinion),
    so each non-blank line is one paragraph. Section heading lines are
    validated and excluded from the body: the chapter number/title lines
    for chapters, the title line for The Doctor's Opinion.
    """
    # Map every physical line to absolute offsets first (exact slicing).
    lines: list[tuple[int, int, str]] = []
    offset = 0
    for raw in section_text.splitlines(keepends=True):
        lines.append((offset, offset + len(raw), raw))
        offset += len(raw)
    if offset != len(section_text):
        raise ValueError(f"{section_id}: line scan does not cover the section text")

    nonempty = [(s, e, r) for (s, e, r) in lines if r.strip()]
    if not nonempty:
        raise ValueError(f"{section_id}: section has no text lines")

    if section_id == DOCTORS_OPINION_ID:
        if nonempty[0][2].strip() != DOCTORS_OPINION_TITLE:
            raise ValueError("doctors-opinion title line mismatch")
        body = nonempty[1:]
    else:
        number = int(section_id.split("-")[1])
        if not CHAPTER_HEADING.fullmatch(nonempty[0][2].strip()):
            raise ValueError(f"{section_id}: chapter heading line mismatch")
        if nonempty[1][2].strip() != CHAPTER_TITLES[number - 1]:
            raise ValueError(f"{section_id}: chapter title line mismatch")
        body = nonempty[2:]
    if not body:
        raise ValueError(f"{section_id}: section has no body paragraphs")

    paragraphs: list[ParagraphSpan] = []
    for position, (line_start, _line_end, raw) in enumerate(body, start=1):
        stripped = raw.strip()
        leading = len(raw) - len(raw.lstrip(" \t"))
        char_start = line_start + leading
        char_end = char_start + len(stripped)
        if section_text[char_start:char_end] != stripped:
            raise ValueError(f"{section_id}: paragraph offset slice mismatch")
        paragraphs.append(
            ParagraphSpan(index=position, text=stripped, char_start=char_start, char_end=char_end)
        )
    return paragraphs


# Concise orientation metadata: navigation aid only, never evidence.
# Each entry is attributable to its source section and deliberately avoids
# quoting the book, so the map cannot substitute for reading the text.
SECTION_GUIDE: dict[str, dict[str, object]] = {
    "doctors-opinion": {
        "topics": ["medical perspective", "illness model", "craving", "mental obsession"],
        "orientation": (
            "A physician's letter framing alcoholism as an illness with a bodily "
            "reaction to alcohol and a mental obsession that defeats willpower. "
            "Reach for it when the question is whether the condition is a moral "
            "failing or something that needs a program of recovery."
        ),
        "related": ["chapter-2", "chapter-3"],
    },
    "chapter-1": {
        "topics": ["personal story", "descent", "identification", "turning point"],
        "orientation": (
            "A first-person account of early drinking, progressive loss of control, "
            "and the events leading toward recovery. Useful for identification and "
            "for questions about what the downward path can look like."
        ),
        "related": ["chapter-2", "chapter-3", "chapter-11"],
    },
    "chapter-2": {
        "topics": ["fellowship", "hope", "common solution", "overview"],
        "orientation": (
            "Introduces the fellowship's shared recovery and the claim that a common "
            "solution exists even for apparently hopeless cases. A good entry point "
            "for whether a way out exists, and a cross-check against narrow readings "
            "of later chapters."
        ),
        "related": ["doctors-opinion", "chapter-4", "chapter-5", "chapter-11"],
    },
    "chapter-3": {
        "topics": ["illness model", "first drink", "loss of control", "moderation"],
        "orientation": (
            "Describes the nature of the condition: trouble controlling the start and "
            "the amount once started. Central for questions about moderation, "
            "willpower, and why a single drink matters."
        ),
        "related": ["doctors-opinion", "chapter-1", "chapter-2"],
    },
    "chapter-4": {
        "topics": ["higher power", "skepticism", "willingness", "belief"],
        "orientation": (
            "Speaks to doubt about spiritual ideas and invites an open-minded "
            "experiment with help beyond oneself. Key for questions about unbelief or "
            "resistance to spiritual language; it offers a different angle on the "
            "same recovery problem."
        ),
        "related": ["chapter-2", "chapter-5", "chapter-6"],
    },
    "chapter-5": {
        "topics": ["program basis", "honesty", "willingness", "steps"],
        "orientation": (
            "Lays out the foundation of the recovery program and the personal "
            "requirements emphasized throughout the book. Read it for questions about "
            "what participation actually asks of a person."
        ),
        "related": ["chapter-4", "chapter-6", "chapter-7"],
    },
    "chapter-6": {
        "topics": ["action", "inventory", "amends", "daily practice"],
        "orientation": (
            "Moves from principles to concrete conduct: taking stock, repairing harm, "
            "and building a daily practice. Relevant when the question concerns "
            "amends, changed behavior, or what to do next."
        ),
        "related": ["chapter-5", "chapter-7", "chapter-9"],
    },
    "chapter-7": {
        "topics": ["helping others", "sponsorship", "service", "carrying the message"],
        "orientation": (
            "Covers working with others who still struggle and why mutual help "
            "sustains recovery. Useful for questions about helping a friend or family "
            "member; pair it with the family chapters for a fuller picture."
        ),
        "related": ["chapter-5", "chapter-6", "chapter-8", "chapter-11"],
    },
    "chapter-8": {
        "topics": ["family perspective", "spouses", "partners", "household"],
        "orientation": (
            "Written for wives and partners: understanding the condition and "
            "responding sanely. Read it together with the next chapter so both sides "
            "of the household situation are covered."
        ),
        "related": ["chapter-9", "chapter-7"],
    },
    "chapter-9": {
        "topics": ["family recovery", "rebuilding trust", "patience", "home life"],
        "orientation": (
            "Advice for the household after recovery begins: adjusting expectations "
            "and rebuilding trust over time. The companion to the previous chapter "
            "from the recovering household's side."
        ),
        "related": ["chapter-8", "chapter-6"],
    },
    "chapter-10": {
        "topics": ["employers", "workplace", "responsibility", "practical help"],
        "orientation": (
            "Guidance for employers facing alcohol problems at work: understanding, "
            "firmness, and practical steps. A distinct workplace angle on the same "
            "human problem."
        ),
        "related": ["chapter-7", "chapter-8"],
    },
    "chapter-11": {
        "topics": ["outlook", "fellowship vision", "ongoing practice", "hope"],
        "orientation": (
            "Closing vision of the fellowship's purpose and the life recovery makes "
            "possible. Good for questions about what comes next, and as a final "
            "coverage check against an overly narrow reading of earlier chapters."
        ),
        "related": ["chapter-2", "chapter-5", "chapter-7"],
    },
}


def _byte_range(section_text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Map a section-text char range onto UTF-8 byte offsets (fail closed)."""
    prefix = section_text[:char_start].encode("utf-8")
    span = section_text[char_start:char_end].encode("utf-8")
    return len(prefix), len(prefix) + len(span)


def build_structure(
    *,
    canonical: dict[str, object],
    manifest: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build ``(structure, chunk_texts)`` deterministically (fail closed).

    ``structure`` is public navigation metadata without literary text;
    ``chunk_texts`` carries exact chunk text for the ignored runtime
    workspace. Every chunk text is verified to equal the canonical slice.
    """
    sections_in = canonical.get("sections")
    if not isinstance(sections_in, list) or not sections_in:
        raise ValueError("canonical artifact has no sections list")
    manifest_sections = manifest.get("sections")
    if not isinstance(manifest_sections, list):
        raise ValueError("manifest has no sections list")

    artifact_entries = manifest.get("artifact_sha256")
    if not isinstance(artifact_entries, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact_entries):
        raise ValueError("manifest artifact_sha256 is missing or malformed")
    manifest_artifact_sha = artifact_entries

    if [str(s.get("id")) for s in sections_in if isinstance(s, dict)] != list(EXPECTED_SECTION_IDS):
        raise ValueError("canonical section order is not the expected scope")

    canonical_format = canonical.get("format")
    if canonical_format != "aa-canonical/1":
        raise ValueError(f"unsupported canonical format: {canonical_format!r}")

    section_rows: list[dict[str, object]] = []
    paragraph_rows: list[dict[str, object]] = []
    sentence_rows: list[dict[str, object]] = []
    chunk_rows: list[dict[str, object]] = []
    chunk_texts: list[dict[str, object]] = []

    global_chunk_index = 0
    previous_chunk_id: str | None = None

    for order, entry in enumerate(sections_in):
        if not isinstance(entry, dict):
            raise ValueError("canonical artifact has a malformed section")
        section_id = str(entry["id"])
        title = str(entry["title"])
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError(f"canonical section {section_id!r} has no text")
        expected_text_sha = entry.get("text_sha256")
        actual_text_sha = sha256_hex(text.encode("utf-8"))
        if expected_text_sha != actual_text_sha:
            raise ValueError(f"canonical section checksum mismatch: {section_id!r}")
        source_id = str(entry["source_id"])
        source_url = str(entry["source_url"])
        source_file = str(entry["source_file"])
        raw_source_sha = str(entry.get("source_sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", raw_source_sha):
            raise ValueError(f"canonical section {section_id!r} lacks raw source checksum")

        paragraphs = split_paragraphs(section_id, text)

        # Paragraph nodes with sentence children.
        para_ids: list[str] = []
        sentence_id_lists: list[list[str]] = []
        for para in paragraphs:
            para_id = f"{section_id}:p{para.index:03d}"
            para_ids.append(para_id)
            rel_spans = split_sentences(para.text)
            sent_ids: list[str] = []
            for sent_index, (rel_start, rel_end) in enumerate(rel_spans, start=1):
                sent_id = f"{para_id}:s{sent_index:02d}"
                sent_ids.append(sent_id)
                abs_start = para.char_start + rel_start
                abs_end = para.char_start + rel_end
                if text[abs_start:abs_end] != para.text[rel_start:rel_end]:
                    raise ValueError(f"{sent_id}: sentence offset slice mismatch")
            sentence_id_lists.append(sent_ids)

        # Absolute sentence spans, used to prove chunk ends fall on sentence
        # or paragraph boundaries (never mid-sentence).
        abs_sentence_spans: list[tuple[int, int, str, str]] = []  # (start, end, para_id, sent_id)
        for para, sent_ids in zip(paragraphs, sentence_id_lists, strict=False):
            para_id = f"{section_id}:p{para.index:03d}"
            rel_spans = split_sentences(para.text)
            for sent_id, (rel_start, rel_end) in zip(sent_ids, rel_spans, strict=False):
                abs_sentence_spans.append(
                    (para.char_start + rel_start, para.char_start + rel_end, para_id, sent_id)
                )
        sentence_ends = {end for (_, end, _, _) in abs_sentence_spans}
        paragraph_ends = {para.char_end for para in paragraphs}

        # ---- retrieval chunking (greedy over paragraphs, sentences for long ones)
        # A chunk is a contiguous run of whole paragraphs, except that an
        # oversized paragraph is split along sentence boundaries. Chunk text
        # is always the exact canonical slice [char_start:char_end).
        units: list[tuple[str, int, int, list[str], list[str]]] = []
        for para, sent_ids in zip(paragraphs, sentence_id_lists, strict=False):
            if len(para.text) > MAX_CHUNK_CHARS:
                rel_spans = split_sentences(para.text)
                para_id = f"{section_id}:p{para.index:03d}"
                # Greedily pack sentences without exceeding MAX_CHUNK_CHARS.
                run: list[tuple[int, int, str]] = []
                run_len = 0
                for sent_id, (rel_start, rel_end) in zip(sent_ids, rel_spans, strict=False):
                    sent_text = para.text[rel_start:rel_end]
                    if len(sent_text) > MAX_CHUNK_CHARS:
                        raise ValueError(f"{para_id}: single sentence exceeds MAX_CHUNK_CHARS")
                    gap = rel_start - run[-1][1] if run else 0
                    add = len(sent_text) + gap
                    if run and run_len + add > MAX_CHUNK_CHARS:
                        first = run[0]
                        last = run[-1]
                        units.append(
                            (
                                "sentences",
                                para.char_start + first[0],
                                para.char_start + last[1],
                                [para_id],
                                [s for (_, _, s) in run],
                            )
                        )
                        run = []
                        run_len = 0
                        add = len(sent_text)
                    run.append((rel_start, rel_end, sent_id))
                    run_len += add
                if run:
                    first = run[0]
                    last = run[-1]
                    units.append(
                        (
                            "sentences",
                            para.char_start + first[0],
                            para.char_start + last[1],
                            [para_id],
                            [s for (_, _, s) in run],
                        )
                    )
            else:
                para_id = f"{section_id}:p{para.index:03d}"
                units.append(
                    ("paragraph", para.char_start, para.char_end, [para_id], list(sent_ids))
                )

        # Pack paragraph-level units greedily; sentence-split units stand alone.
        section_chunks: list[tuple[int, int, list[str], list[str]]] = []
        pending: list[tuple[str, int, int, list[str], list[str]]] = []
        pending_len = 0
        for unit in units:
            kind, cs, ce, pids, sids = unit
            if kind == "sentences":
                if pending:
                    section_chunks.append(_flush_pending(text, pending))
                    pending = []
                    pending_len = 0
                section_chunks.append((cs, ce, pids, sids))
                continue
            add = 0
            if pending:
                gap = cs - pending[-1][2]
                candidate_len = pending_len + gap + (ce - cs)
            else:
                candidate_len = ce - cs
            if pending and candidate_len > MAX_CHUNK_CHARS:
                section_chunks.append(_flush_pending(text, pending))
                pending = []
                pending_len = 0
                candidate_len = ce - cs
            pending.append(unit)
            pending_len = candidate_len
            if pending_len >= MIN_CHUNK_CHARS:
                section_chunks.append(_flush_pending(text, pending))
                pending = []
                pending_len = 0
        if pending:
            section_chunks.append(_flush_pending(text, pending))

        # Merge a tiny trailing chunk into its predecessor (same section only).
        if len(section_chunks) >= 2:
            last = section_chunks[-1]
            paragraph_starts = {para.char_start for para in paragraphs}
            if (
                (last[1] - last[0]) < TAIL_MERGE_CHARS
                and len(last[2]) == 1
                and last[0] in paragraph_starts
                and last[1] in paragraph_ends
            ):
                prev = section_chunks[-2]
                # Merge only whole-paragraph chunks so a partial sentence-split
                # tail never claims a full paragraph id or duplicates it.
                if prev[0] in paragraph_starts and prev[1] in paragraph_ends:
                    if last[1] - prev[0] <= MAX_CHUNK_CHARS:
                        merged_pids = prev[2] + [pid for pid in last[2] if pid not in prev[2]]
                        section_chunks[-2] = (
                            prev[0],
                            last[1],
                            merged_pids,
                            prev[3] + last[3],
                        )
                        section_chunks.pop()

        # ---- emit paragraph / sentence / chunk rows
        for position, para in enumerate(paragraphs):
            para_id = para_ids[position]
            prev_id = para_ids[position - 1] if position > 0 else None
            next_id = para_ids[position + 1] if position + 1 < len(para_ids) else None
            byte_start, byte_end = _byte_range(text, para.char_start, para.char_end)
            para_text_sha = sha256_hex(para.text.encode("utf-8"))
            paragraph_rows.append(
                {
                    "id": para_id,
                    "kind": "paragraph",
                    "section_id": section_id,
                    "section_title": title,
                    "parent_id": section_id,
                    "prev_id": prev_id,
                    "next_id": next_id,
                    "char_start": para.char_start,
                    "char_end": para.char_end,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "chars": len(para.text),
                    "est_tokens": estimate_tokens(len(para.text)),
                    "text_sha256": para_text_sha,
                    "sentence_ids": sentence_id_lists[position],
                    "sentence_count": len(sentence_id_lists[position]),
                    # Provenance resolves via section_id: the section row
                    # carries the full source URL/file/checksum record, so
                    # paragraphs keep only the source id plus the canonical
                    # version checksum to stay small.
                    "source_id": source_id,
                    "canonical_artifact_sha256": manifest_artifact_sha,
                    "canonical_format": "aa-canonical/1",
                }
            )

        for abs_start, abs_end, para_id, sent_id in abs_sentence_spans:
            siblings = next(
                sids
                for pid, sids in zip(para_ids, sentence_id_lists, strict=False)
                if pid == para_id
            )
            position = siblings.index(sent_id)
            prev_id = siblings[position - 1] if position > 0 else None
            next_id = siblings[position + 1] if position + 1 < len(siblings) else None
            sent_text = text[abs_start:abs_end]
            sentence_rows.append(
                {
                    "id": sent_id,
                    "kind": "sentence",
                    "section_id": section_id,
                    "section_title": title,
                    "parent_id": para_id,
                    "prev_id": prev_id,
                    "next_id": next_id,
                    "char_start": abs_start,
                    "char_end": abs_end,
                    "chars": len(sent_text),
                    "est_tokens": estimate_tokens(len(sent_text)),
                    # Provenance resolves via section_id (see paragraph
                    # comment); byte offsets live on paragraph/chunk rows.
                    "source_id": source_id,
                    "canonical_artifact_sha256": manifest_artifact_sha,
                    "canonical_format": "aa-canonical/1",
                }
            )

        chunk_ids_this_section: list[str] = []
        for local_index, (cs, ce, pids, sids) in enumerate(section_chunks, start=1):
            global_chunk_index += 1
            chunk_id = f"{section_id}:c{local_index:03d}"
            chunk_ids_this_section.append(chunk_id)
            chunk_text = text[cs:ce]
            if not chunk_text.strip():
                raise ValueError(f"{chunk_id}: chunk text is blank")
            # Natural-boundary check: chunks end on paragraph/sentence ends.
            if ce not in paragraph_ends and ce not in sentence_ends:
                raise ValueError(f"{chunk_id}: chunk ends mid-sentence")
            byte_start, byte_end = _byte_range(text, cs, ce)
            chunk_text_sha = sha256_hex(chunk_text.encode("utf-8"))
            chunk_rows.append(
                {
                    "id": chunk_id,
                    "kind": "chunk",
                    "section_id": section_id,
                    "section_title": title,
                    "parent_id": section_id,
                    "prev_id": previous_chunk_id,
                    "next_id": None,  # linked below
                    "index_global": global_chunk_index,
                    "index_in_section": local_index,
                    "paragraph_ids": pids,
                    "sentence_ids": sids,
                    "char_start": cs,
                    "char_end": ce,
                    "byte_start": byte_start,
                    "byte_end": byte_end,
                    "chars": len(chunk_text),
                    "est_tokens": estimate_tokens(len(chunk_text)),
                    "text_sha256": chunk_text_sha,
                    "source_id": source_id,
                    "source_url": source_url,
                    "source_file": source_file,
                    "source_sha256": raw_source_sha,
                    "canonical_artifact_sha256": manifest_artifact_sha,
                    "canonical_format": "aa-canonical/1",
                }
            )
            chunk_texts.append({"id": chunk_id, "section_id": section_id, "text": chunk_text})
            if previous_chunk_id is not None:
                chunk_rows[-2]["next_id"] = chunk_id
            previous_chunk_id = chunk_id

        # Section row (metadata only, no literary text).
        prev_section = EXPECTED_SECTION_IDS[order - 1] if order > 0 else None
        next_section = (
            EXPECTED_SECTION_IDS[order + 1] if order + 1 < len(EXPECTED_SECTION_IDS) else None
        )
        section_byte_start = entry.get("byte_start")
        section_byte_end = entry.get("byte_end")
        if section_byte_start is None and section_byte_end is None:
            pass
        else:
            if (
                not isinstance(section_byte_start, int)
                or not isinstance(section_byte_end, int)
                or isinstance(section_byte_start, bool)
                or isinstance(section_byte_end, bool)
            ):
                raise ValueError(f"{section_id}: canonical section lacks byte offsets")
            if not (0 <= section_byte_start < section_byte_end):
                raise ValueError(f"{section_id}: canonical byte offsets are malformed")
        section_rows.append(
            {
                "id": section_id,
                "kind": "section",
                "order": order,
                "title": title,
                "section_id": section_id,
                "parent_id": BOOK_ID,
                "prev_id": prev_section,
                "next_id": next_section,
                "char_start": 0,
                "char_end": len(text),
                "source_id": source_id,
                "source_url": source_url,
                "source_file": source_file,
                "source_sha256": raw_source_sha,
                "byte_start": section_byte_start,
                "byte_end": section_byte_end,
                "chars": len(text),
                "est_tokens": estimate_tokens(len(text)),
                "text_sha256": actual_text_sha,
                "paragraph_count": len(paragraphs),
                "sentence_count": sum(len(s) for s in sentence_id_lists),
                "chunk_count": len(chunk_ids_this_section),
                "paragraph_ids": [para_ids[0], para_ids[-1]],
                "chunk_ids": [chunk_ids_this_section[0], chunk_ids_this_section[-1]]
                if chunk_ids_this_section
                else [],
                "canonical_artifact_sha256": manifest_artifact_sha,
                "canonical_format": "aa-canonical/1",
            }
        )

    # Global ordering/tiling validation per section.
    _validate_tiling(section_rows, paragraph_rows, chunk_rows)

    structure: dict[str, object] = {
        "format": STRUCTURE_FORMAT,
        "structure_builder_version": STRUCTURE_BUILDER_VERSION,
        "book": {
            "id": BOOK_ID,
            "title": BOOK_TITLE,
            "section_ids": list(EXPECTED_SECTION_IDS),
            "section_count": len(section_rows),
            "paragraph_count": len(paragraph_rows),
            "sentence_count": len(sentence_rows),
            "chunk_count": len(chunk_rows),
        },
        "canonical": {
            "format": "aa-canonical/1",
            "artifact_sha256": manifest_artifact_sha,
        },
        "sections": section_rows,
        "paragraphs": paragraph_rows,
        "sentences": sentence_rows,
        "chunks": chunk_rows,
    }
    return structure, chunk_texts


def _flush_pending(
    section_text: str, pending: list[tuple[str, int, int, list[str], list[str]]]
) -> tuple[int, int, list[str], list[str]]:
    """Close a pending paragraph run into one chunk span (exact slice)."""
    first = pending[0]
    last = pending[-1]
    char_start = first[1]
    char_end = last[2]
    paragraph_ids: list[str] = []
    sentence_ids: list[str] = []
    for _, _, _, pids, sids in pending:
        paragraph_ids.extend(pids)
        sentence_ids.extend(sids)
    if not section_text[char_start:char_end].strip():
        raise ValueError("chunk span is blank")
    return (char_start, char_end, paragraph_ids, sentence_ids)


def _validate_tiling(
    section_rows: list[dict[str, object]],
    paragraph_rows: list[dict[str, object]],
    chunk_rows: list[dict[str, object]],
) -> None:
    """Validate hierarchy tiling: ordered, non-overlapping, gap-free.

    Chunks tile every sentence exactly once (an oversized paragraph split
    across sentence boundaries still lists its paragraph id on each part,
    so sentence coverage is the exact invariant).
    """
    for section in section_rows:
        section_id = str(section["id"])
        paras = [p for p in paragraph_rows if p["section_id"] == section_id]
        paras.sort(key=lambda p: int(str(p["char_start"])))
        for first, second in zip(paras, paras[1:], strict=False):
            if int(str(second["char_start"])) < int(str(first["char_end"])):
                raise ValueError(f"{section_id}: paragraph spans overlap")
        chunks = [c for c in chunk_rows if c["section_id"] == section_id]
        chunks.sort(key=lambda c: int(str(c["char_start"])))
        if not chunks:
            raise ValueError(f"{section_id}: section has no chunks")
        covered: list[str] = []
        for chunk in chunks:
            sentence_ids = chunk["sentence_ids"]
            if not isinstance(sentence_ids, list) or not sentence_ids:
                raise ValueError(f"{section_id}: chunk lacks sentence ids")
            for sentence_id in sentence_ids:
                covered.append(str(sentence_id))
        expected: list[str] = []
        for para in paras:
            sentence_ids = para["sentence_ids"]
            if not isinstance(sentence_ids, list):
                raise ValueError(f"{section_id}: paragraph lacks sentence ids")
            for sentence_id in sentence_ids:
                expected.append(str(sentence_id))
        if covered != expected:
            raise ValueError(f"{section_id}: chunks do not tile sentences exactly once")
        for first, second in zip(chunks, chunks[1:], strict=False):
            if int(str(second["char_start"])) < int(str(first["char_end"])):
                raise ValueError(f"{section_id}: chunk spans overlap")


def render_book_map(
    *,
    structure: dict[str, object],
    book_map_budget_tokens: int = 6000,
) -> tuple[str, int]:
    """Render the compact public book map markdown and its token estimate."""
    sections = structure["sections"]
    if not isinstance(sections, list):
        raise ValueError("structure has no sections list")
    canonical = structure.get("canonical")
    if not isinstance(canonical, dict):
        raise ValueError("structure has no canonical provenance")
    artifact_sha = str(canonical.get("artifact_sha256", ""))
    book = structure.get("book")
    if not isinstance(book, dict):
        raise ValueError("structure has no book node")

    lines: list[str] = []
    lines.append("# AA book map (navigation only)")
    lines.append("")
    lines.append(
        "Versioned routing layer for the AA agent and hybrid retrieval. "
        "It names where to look next; it is never evidence for an answer."
    )
    lines.append("")
    lines.append(f"- structure format: `{STRUCTURE_FORMAT}` (builder v{STRUCTURE_BUILDER_VERSION})")
    lines.append(f"- canonical artifact: `aa-canonical/1` sha256 `{artifact_sha}`")
    lines.append(f"- scope: {len(sections)} sections in canonical order")
    total_chunks = sum(int(str(s.get("chunk_count", 0))) for s in sections if isinstance(s, dict))
    lines.append(f"- retrieval chunks: {total_chunks} addressable ranges")
    lines.append(f"- context budget: compact map must fit {book_map_budget_tokens} tokens")
    lines.append("- grounding rule: substantive claims must cite exact passages read with")
    lines.append("  `book_read` / `book_expand` / `book_section`, never this map.")
    lines.append("")
    lines.append("Coverage planning: for broad personal questions, plan searches across the")
    lines.append("whole book, read exact passages from distinct relevant regions, check for")
    lines.append("missing perspectives, and only then synthesize. One top hit is not enough.")
    lines.append("")
    lines.append("## Canonical order")
    lines.append("")
    lines.append("| # | Section | Stable IDs | Chunks | Est. tokens |")
    lines.append("|---|---|---|---|---|")
    for position, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError("structure has a malformed section")
        section_id = str(section["id"])
        title = str(section["title"])
        chunk_ids = section.get("chunk_ids")
        if not isinstance(chunk_ids, list) or len(chunk_ids) != 2:
            raise ValueError(f"{section_id}: section lacks a chunk range")
        chunk_range = f"`{chunk_ids[0]}`..`{chunk_ids[-1]}`"
        lines.append(
            f"| {position} | {title} (`{section_id}`) | "
            f"section `{section_id}` | {chunk_range} "
            f"({section['chunk_count']}) | {section['est_tokens']} |"
        )
    lines.append("")
    lines.append("## Section guide")
    lines.append("")
    for position, section in enumerate(sections):
        if not isinstance(section, dict):
            raise ValueError("structure has a malformed section")
        section_id = str(section["id"])
        guide = SECTION_GUIDE.get(section_id)
        if guide is None:
            raise ValueError(f"book map has no orientation entry for {section_id!r}")
        topics = ", ".join(str(t) for t in guide["topics"] if isinstance(t, str))
        related = ", ".join(f"`{r}`" for r in guide["related"] if isinstance(r, str))
        chunk_ids = section.get("chunk_ids")
        if not isinstance(chunk_ids, list):
            raise ValueError(f"{section_id}: section lacks a chunk range")
        lines.append(f"### {position}. {section['title']} (`{section_id}`)")
        lines.append("")
        lines.append(f"- topics: {topics}")
        lines.append(f"- orientation: {guide['orientation']}")
        lines.append(
            f"- read via: section `{section_id}`, chunks `{chunk_ids[0]}`..`{chunk_ids[-1]}`"
        )
        lines.append(f"- consider also: {related} (a different perspective on the same problem)")
        lines.append(
            f"- size: {section['chars']} chars / ~{section['est_tokens']} tokens, "
            f"{section['paragraph_count']} paragraphs, {section['chunk_count']} chunks"
        )
        lines.append("")
    lines.append("## Tool routing")
    lines.append("")
    lines.append("- `book_search(query)` -- rank candidate chunks across the whole corpus;")
    lines.append("  start from 2-4 map regions, not one best guess.")
    lines.append(
        "- `book_read(chunk_id)` -- read the exact text of one chunk, e.g. `chapter-3:c001`."
    )
    lines.append("- `book_expand(chunk_id, before, after)` -- bounded exact neighbor context")
    lines.append("  via chunk `prev_id` / `next_id` links.")
    lines.append("- `book_section(section_id)` -- bounded paged read within one section.")
    lines.append("")
    lines.append("## Regenerate")
    lines.append("")
    lines.append("```bash")
    lines.append("python3 scripts/fetch_aa_source.py")
    lines.append("python3 scripts/build_canonical.py")
    lines.append("python3 scripts/build_corpus_structure.py")
    lines.append("```")
    lines.append("")
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    # The footer carries no numbers so the reported estimate always matches
    # the committed file exactly.
    text += "<!-- book-map: versioned routing layer; token counts are reported in\n"
    text += "     corpus/structure-report.json and must fit the context budget -->\n"
    tokens = estimate_tokens(len(text))
    if tokens > book_map_budget_tokens:
        raise ValueError(
            f"book map needs ~{tokens} tokens but the budget is {book_map_budget_tokens}"
        )
    return text, tokens


def build_report(
    *,
    structure: dict[str, object],
    book_map_tokens: int,
    book_map_chars: int,
    structure_bytes: int,
    chunks_bytes: int,
    canonical_bytes: int,
    book_map_budget_tokens: int = 6000,
) -> dict[str, object]:
    """Assemble the deterministic public size/token report."""
    sections = structure["sections"]
    book = structure["book"]
    if not isinstance(sections, list) or not isinstance(book, dict):
        raise ValueError("structure is malformed")
    per_section = []
    for section in sections:
        if not isinstance(section, dict):
            raise ValueError("structure has a malformed section")
        per_section.append(
            {
                "id": str(section["id"]),
                "title": str(section["title"]),
                "chars": section["chars"],
                "est_tokens": section["est_tokens"],
                "paragraph_count": section["paragraph_count"],
                "sentence_count": section["sentence_count"],
                "chunk_count": section["chunk_count"],
                "chunk_range": section["chunk_ids"],
            }
        )
    canonical = structure.get("canonical")
    if not isinstance(canonical, dict):
        raise ValueError("structure has no canonical provenance")
    return {
        "format": "aa-corpus-structure-report/1",
        "structure_builder_version": STRUCTURE_BUILDER_VERSION,
        "structure_format": STRUCTURE_FORMAT,
        "canonical": canonical,
        "counts": {
            "sections": len(sections),
            "paragraphs": book.get("paragraph_count"),
            "sentences": book.get("sentence_count"),
            "chunks": book.get("chunk_count"),
        },
        "book_map": {
            "chars": book_map_chars,
            "est_tokens": book_map_tokens,
            "budget_tokens": book_map_budget_tokens,
            "fits_budget": book_map_tokens <= book_map_budget_tokens,
        },
        "bytes": {
            "canonical_json": canonical_bytes,
            "structure_json": structure_bytes,
            "chunks_json": chunks_bytes,
        },
        "chunking": {
            "max_chunk_chars": MAX_CHUNK_CHARS,
            "min_chunk_chars": MIN_CHUNK_CHARS,
            "tail_merge_chars": TAIL_MERGE_CHARS,
        },
        "sections": per_section,
    }


def serialize_json(payload: dict[str, object] | list[dict[str, object]]) -> bytes:
    """Serialize deterministically (sorted keys, UTF-8)."""
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: build hierarchy, map, and reports (fail closed)."""
    parser = argparse.ArgumentParser(description="Build the hierarchical AA corpus structure.")
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structure", type=Path, default=DEFAULT_STRUCTURE)
    parser.add_argument("--book-map", type=Path, default=DEFAULT_BOOK_MAP)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--chunks-out", type=Path, default=DEFAULT_CHUNKS_OUT)
    args = parser.parse_args(argv)

    try:
        canonical_text = args.canonical.read_bytes()
        manifest_text = args.manifest.read_bytes()
    except FileNotFoundError as exc:
        hint = "python3 scripts/fetch_aa_source.py && python3 scripts/build_canonical.py"
        if RESTORE_ENTRY_POINT.exists():
            hint = "python3 scripts/restore_canonical.py"
        return fail(f"required input file is missing: {exc.filename}; bootstrap with {hint}")

    try:
        canonical = json.loads(canonical_text.decode("utf-8"))
        manifest = json.loads(manifest_text.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return fail(f"required input file is not valid JSON: {exc}")
    if not isinstance(canonical, dict) or not isinstance(manifest, dict):
        return fail("required input files must hold JSON objects")

    # Fail closed when the canonical bytes do not match the committed manifest.
    expected_artifact = manifest.get("artifact_sha256")
    if expected_artifact != sha256_hex(canonical_text):
        return fail(
            "canonical artifact checksum mismatch against corpus/canonical.manifest.json; "
            "refusing to build navigation over an unverified book"
        )

    try:
        structure, chunk_texts = build_structure(canonical=canonical, manifest=manifest)
        book_map_text, book_map_tokens = render_book_map(structure=structure)
        structure_bytes = serialize_json(structure)
        chunks_payload = {
            "format": CHUNKS_FORMAT,
            "structure_builder_version": STRUCTURE_BUILDER_VERSION,
            "canonical": structure["canonical"],
            "chunks": chunk_texts,
        }
        chunks_bytes = serialize_json(chunks_payload)
        # Verify every chunk round-trips to exact canonical text.
        by_section = {
            str(s["id"]): str(s["text"])
            for s in canonical["sections"]
            if isinstance(s, dict) and isinstance(s.get("text"), str)
        }
        for chunk_row, chunk_entry in zip(structure["chunks"], chunk_texts, strict=True):
            if not isinstance(chunk_row, dict):
                raise ValueError("structure has a malformed chunk")
            section_text = by_section[str(chunk_row["section_id"])]
            expected = section_text[
                int(str(chunk_row["char_start"])) : int(str(chunk_row["char_end"]))
            ]
            if chunk_entry["text"] != expected:
                raise ValueError(f"{chunk_row['id']}: chunk text does not round-trip")
        report = build_report(
            structure=structure,
            book_map_tokens=book_map_tokens,
            book_map_chars=len(book_map_text),
            structure_bytes=len(structure_bytes),
            chunks_bytes=len(chunks_bytes),
            canonical_bytes=len(canonical_text),
        )
        report_bytes = serialize_json(report)
    except (ValueError, KeyError, StopIteration, IndexError, TypeError) as exc:
        return fail(str(exc))

    # All validation passed: write every output (no partial writes before this).
    args.structure.parent.mkdir(parents=True, exist_ok=True)
    args.structure.write_bytes(structure_bytes)
    args.book_map.write_text(book_map_text, encoding="utf-8")
    args.report.write_bytes(report_bytes)
    args.chunks_out.parent.mkdir(parents=True, exist_ok=True)
    args.chunks_out.write_bytes(chunks_bytes)

    print(
        json.dumps(
            {
                "sections": len(structure["sections"]),  # type: ignore[arg-type]
                "paragraphs": len(structure["paragraphs"]),  # type: ignore[arg-type]
                "sentences": len(structure["sentences"]),  # type: ignore[arg-type]
                "chunks": len(structure["chunks"]),  # type: ignore[arg-type]
                "book_map_tokens": book_map_tokens,
                "structure_bytes": len(structure_bytes),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
