"""Runtime access to the hierarchical AA corpus structure (issue #8).

The structure is produced by ``scripts/build_corpus_structure.py`` into the
public ``corpus/structure.json`` (navigation metadata only, no literary
text). This module loads it, validates hierarchy invariants (stable IDs,
parent/previous/next links, offsets, provenance), and resolves retrieval
units back to exact canonical text via :class:`CanonicalCorpus`.

Nothing here modifies or shortens the canonical book: the structure is a
routing layer, and substantive text always comes from the canonical
artifact through exact offsets.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aa.corpus.budget import BOOK_MAP_BUDGET_TOKENS, estimate_tokens
from aa.corpus.canonical import CanonicalCorpus

STRUCTURE_FORMAT = "aa-corpus-structure/1"

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


class CorpusStructureError(ValueError):
    """Raised when the structure artifact fails validation."""


@dataclass(frozen=True)
class StructureNode:
    """One hierarchy node: section, paragraph, sentence, or chunk."""

    id: str
    kind: str
    section_id: str
    parent_id: str
    prev_id: str | None
    next_id: str | None
    char_start: int
    char_end: int
    raw: dict[str, Any]


@dataclass
class CorpusStructure:
    """Validated hierarchy with parent/neighbor navigation."""

    nodes: dict[str, StructureNode] = field(default_factory=dict)
    children: dict[str, tuple[str, ...]] = field(default_factory=dict)
    canonical_artifact_sha256: str = ""
    section_ids: tuple[str, ...] = ()

    def get(self, node_id: str) -> StructureNode:
        """Return a node by stable id or raise :class:`CorpusStructureError`."""
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise CorpusStructureError(f"unknown structure node: {node_id!r}") from exc

    def parent(self, node_id: str) -> StructureNode:
        """Return the parent node (the book root has no parent)."""
        node = self.get(node_id)
        if node.parent_id == "":
            raise CorpusStructureError(f"node {node_id!r} is the book root and has no parent")
        return self.get(node.parent_id)

    def previous(self, node_id: str) -> StructureNode | None:
        """Return the previous neighbor or None at the start of a chain."""
        node = self.get(node_id)
        return self.get(node.prev_id) if node.prev_id is not None else None

    def next(self, node_id: str) -> StructureNode | None:
        """Return the next neighbor or None at the end of a chain."""
        node = self.get(node_id)
        return self.get(node.next_id) if node.next_id is not None else None

    def children_of(self, node_id: str) -> tuple[StructureNode, ...]:
        """Return direct children of a node in canonical order."""
        self.get(node_id)  # fail closed on unknown ids
        return tuple(self.get(child_id) for child_id in self.children.get(node_id, ()))

    def section_chunks(self, section_id: str) -> tuple[StructureNode, ...]:
        """Return retrieval chunks of one section in canonical order."""
        chunks = [node for node in self.children_of(section_id) if node.kind == "chunk"]
        chunks.sort(key=lambda node: int(node.raw.get("index_in_section", 0)))
        return tuple(chunks)

    def chunk_text(self, chunk_id: str, corpus: CanonicalCorpus) -> str:
        """Resolve a chunk to its exact canonical text (round-trip read)."""
        node = self.get(chunk_id)
        if node.kind != "chunk":
            raise CorpusStructureError(f"not a retrieval chunk: {chunk_id!r}")
        section = corpus.get(node.section_id)
        if node.char_end > len(section.text):
            raise CorpusStructureError(f"chunk offsets out of range: {chunk_id!r}")
        text = section.text[node.char_start : node.char_end]
        if len(text) != node.char_end - node.char_start:
            raise CorpusStructureError(f"chunk resolves to truncated text: {chunk_id!r}")
        if not text.strip():
            raise CorpusStructureError(f"chunk resolves to blank text: {chunk_id!r}")
        return text


def _require_str(entry: dict[str, Any], key: str, node_id: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise CorpusStructureError(f"node {node_id!r} is missing {key}")
    return value


def _require_int(entry: dict[str, Any], key: str, node_id: str) -> int:
    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CorpusStructureError(f"node {node_id!r} has a non-integer {key}")
    return value


def load_structure(path: str | Path) -> CorpusStructure:
    """Load and validate the public structure artifact (fails closed)."""
    raw_path = Path(path)
    try:
        payload = raw_path.read_bytes()
    except FileNotFoundError as exc:
        raise CorpusStructureError(f"structure artifact is missing: {raw_path}") from exc
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusStructureError(f"structure artifact is not valid JSON: {exc}") from exc
    if not isinstance(artifact, dict) or artifact.get("format") != STRUCTURE_FORMAT:
        raise CorpusStructureError(f"unsupported structure format: {raw_path}")

    canonical = artifact.get("canonical")
    if not isinstance(canonical, dict):
        raise CorpusStructureError("structure artifact has no canonical provenance")
    artifact_sha = canonical.get("artifact_sha256")
    if not isinstance(artifact_sha, str) or not artifact_sha:
        raise CorpusStructureError("structure artifact lacks the canonical checksum")

    book = artifact.get("book")
    if not isinstance(book, dict):
        raise CorpusStructureError("structure artifact has no book node")

    rows: list[dict[str, Any]] = []
    for key in ("sections", "paragraphs", "sentences", "chunks"):
        entries = artifact.get(key)
        if not isinstance(entries, list) or not entries:
            raise CorpusStructureError(f"structure artifact has no {key} list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise CorpusStructureError(f"structure artifact has a malformed {key} row")
            rows.append(entry)

    # The public artifact must stay navigation metadata: no literary text.
    for entry in rows:
        if "text" in entry:
            raise CorpusStructureError(
                f"structure artifact must not carry literary text: {entry.get('id')!r}"
            )

    nodes: dict[str, StructureNode] = {}
    for entry in rows:
        node_id = _require_str(entry, "id", "<unknown>")
        if node_id in nodes:
            raise CorpusStructureError(f"duplicate structure node id: {node_id!r}")
        kind = _require_str(entry, "kind", node_id)
        if kind not in ("section", "paragraph", "sentence", "chunk"):
            raise CorpusStructureError(f"node {node_id!r} has unknown kind {kind!r}")
        prev_id = entry.get("prev_id")
        next_id = entry.get("next_id")
        if prev_id is not None and not isinstance(prev_id, str):
            raise CorpusStructureError(f"node {node_id!r} has a malformed prev_id")
        if next_id is not None and not isinstance(next_id, str):
            raise CorpusStructureError(f"node {node_id!r} has a malformed next_id")
        nodes[node_id] = StructureNode(
            id=node_id,
            kind=kind,
            section_id=_require_str(entry, "section_id", node_id),
            parent_id=_require_str(entry, "parent_id", node_id),
            prev_id=prev_id,
            next_id=next_id,
            char_start=_require_int(entry, "char_start", node_id),
            char_end=_require_int(entry, "char_end", node_id),
            raw=entry,
        )

    structure = CorpusStructure(canonical_artifact_sha256=artifact_sha)
    structure.nodes = nodes

    # Synthesize the book root so every section has a navigable parent.
    total_chars = 0
    for entry in rows:
        if entry.get("kind") == "section":
            total_chars += _require_int(entry, "char_end", str(entry.get("id")))
    nodes["aa-book"] = StructureNode(
        id="aa-book",
        kind="book",
        section_id="aa-book",
        parent_id="",
        prev_id=None,
        next_id=None,
        char_start=0,
        char_end=total_chars,
        raw={
            "id": "aa-book",
            "kind": "book",
            "title": book.get("title", "aa-book"),
            "canonical_artifact_sha256": artifact_sha,
            "canonical_format": "aa-canonical/1",
        },
    )

    children: dict[str, list[str]] = {}
    for node in nodes.values():
        if node.parent_id == "":
            if node.id != "aa-book":
                raise CorpusStructureError(f"node {node.id!r} has no parent")
        elif node.parent_id not in nodes:
            raise CorpusStructureError(f"node {node.id!r} has an unknown parent {node.parent_id!r}")
        if node.prev_id is not None and node.prev_id not in nodes:
            raise CorpusStructureError(
                f"node {node.id!r} has an unknown prev neighbor {node.prev_id!r}"
            )
        if node.next_id is not None and node.next_id not in nodes:
            raise CorpusStructureError(
                f"node {node.id!r} has an unknown next neighbor {node.next_id!r}"
            )
        if node.char_start < 0:
            raise CorpusStructureError(f"node {node.id!r} has a negative offset")
        if node.char_end <= node.char_start:
            raise CorpusStructureError(f"node {node.id!r} has an empty offset range")
        children.setdefault(node.parent_id, []).append(node.id)
    # Deterministic child order follows canonical offsets.
    structure.children = {
        parent_id: tuple(sorted(child_ids, key=lambda cid: nodes[cid].char_start))
        for parent_id, child_ids in children.items()
    }

    section_ids = tuple(
        str(entry["id"])
        for entry in artifact["sections"]
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    )
    if section_ids != EXPECTED_SECTION_IDS:
        raise CorpusStructureError(f"section order mismatch: {section_ids!r}")
    structure.section_ids = section_ids

    # Neighbor links must be reciprocal and ordered.
    for node in nodes.values():
        if node.prev_id is not None and nodes[node.prev_id].next_id != node.id:
            raise CorpusStructureError(f"node {node.id!r} has a broken prev link")
        if node.next_id is not None and nodes[node.next_id].prev_id != node.id:
            raise CorpusStructureError(f"node {node.id!r} has a broken next link")
        if node.prev_id is not None:
            prev = nodes[node.prev_id]
            if prev.section_id == node.section_id and prev.char_end > node.char_start:
                raise CorpusStructureError(f"node {node.id!r} overlaps its prev neighbor")
        if node.next_id is not None:
            nxt = nodes[node.next_id]
            if nxt.section_id == node.section_id and node.char_end > nxt.char_start:
                raise CorpusStructureError(f"node {node.id!r} overlaps its next neighbor")

    # Provenance: every node pins the canonical version checksum.
    for node in nodes.values():
        if node.raw.get("canonical_artifact_sha256") != artifact_sha:
            raise CorpusStructureError(f"node {node.id!r} lacks canonical provenance")
        if node.raw.get("canonical_format") != "aa-canonical/1":
            raise CorpusStructureError(f"node {node.id!r} lacks canonical format")

    return structure


def verify_chunk_round_trip(structure: CorpusStructure, corpus: CanonicalCorpus) -> int:
    """Verify every chunk resolves to exact canonical text; return chunk count."""
    count = 0
    for node in structure.nodes.values():
        if node.kind != "chunk":
            continue
        text = structure.chunk_text(node.id, corpus)
        if len(text) != node.char_end - node.char_start:
            raise CorpusStructureError(f"chunk {node.id!r} length mismatch")
        expected = node.raw.get("text_sha256")
        if not isinstance(expected, str):
            raise CorpusStructureError(f"chunk {node.id!r} lacks a text checksum")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected:
            raise CorpusStructureError(f"chunk {node.id!r} checksum mismatch")
        count += 1
    if count == 0:
        raise CorpusStructureError("structure has no retrieval chunks")
    return count


def check_book_map(
    path: str | Path, structure: CorpusStructure, budget_tokens: int = BOOK_MAP_BUDGET_TOKENS
) -> int:
    """Validate the public book map: order, ids, budget; return token estimate."""
    text = Path(path).read_text(encoding="utf-8")
    tokens = estimate_tokens(len(text))
    if tokens > budget_tokens:
        raise CorpusStructureError(
            f"book map needs ~{tokens} tokens but the budget is {budget_tokens}"
        )
    position = 0
    for section_id in structure.section_ids:
        marker = f"(`{section_id}`)"
        found = text.find(marker, position)
        if found < 0:
            raise CorpusStructureError(f"book map has no entry for {section_id!r}")
        position = found + len(marker)
    return tokens
