"""Hierarchical corpus structure and book-map tests (issue #8).

Builder-logic tests use small synthetic fixtures (no network, no book
text). Committed-artifact tests validate the public navigation files only
and never require the ignored canonical runtime artifact, so CI stays
hermetic.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import tempfile
from typing import Any, Protocol, cast

import pytest

from aa.corpus.budget import BOOK_MAP_BUDGET_TOKENS
from aa.corpus.canonical import CanonicalCorpus, CanonicalSection
from aa.corpus.structure import (
    CorpusStructure,
    CorpusStructureError,
    check_book_map,
    load_structure,
    verify_chunk_round_trip,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]

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

EXPECTED_IDS = (
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


class _StructureBuilder(Protocol):
    """Typed surface of ``scripts/build_corpus_structure.py`` used here."""

    STRUCTURE_FORMAT: str
    STRUCTURE_BUILDER_VERSION: int
    MAX_CHUNK_CHARS: int
    MIN_CHUNK_CHARS: int

    def sha256_hex(self, data: bytes) -> str: ...
    def estimate_tokens(self, char_count: int) -> int: ...
    def split_sentences(self, paragraph_text: str) -> list[tuple[int, int]]: ...
    def build_structure(
        self,
        *,
        canonical: dict[str, Any],
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]: ...
    def render_book_map(
        self, *, structure: dict[str, Any], book_map_budget_tokens: int = ...
    ) -> tuple[str, int]: ...
    def serialize_json(self, payload: dict[str, Any] | list[dict[str, Any]]) -> bytes: ...
    def main(self, argv: list[str] | None = ...) -> int: ...


def _load_builder() -> _StructureBuilder:
    path = ROOT / "scripts" / "build_corpus_structure.py"
    spec = importlib.util.spec_from_file_location("build_corpus_structure_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_StructureBuilder, module)


def _synthetic_section_text(section_id: str, long_paragraph: bool = False) -> str:
    if section_id == "doctors-opinion":
        lines = ["The Doctor's Opinion", ""]
    else:
        number = int(section_id.split("-")[1])
        lines = [f"Chapter {number}", CHAPTER_TITLES[number - 1], ""]
    body = [
        "Fixture paragraph one has two sentences. Here is the second one.",
        "A short note.",
        "He met Dr. Smith today. They talked about the fixture plan.",
    ]
    if long_paragraph:
        sentences = " ".join(
            f"Synthetic sentence number {i} states a fixture fact." for i in range(80)
        )
        body.append(sentences)
    body.append("Closing fixture paragraph ends the section.")
    return "\n".join(lines) + "\n".join(body) + "\n"


def _synthetic_canonical() -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    for section_id in EXPECTED_IDS:
        if section_id == "doctors-opinion":
            title = "The Doctor's Opinion"
            source_id = "doctors-opinion"
        else:
            number = int(section_id.split("-")[1])
            title = CHAPTER_TITLES[number - 1]
            source_id = "core-pages-1-164"
        text = _synthetic_section_text(section_id, long_paragraph=section_id == "chapter-1")
        sections.append(
            {
                "id": section_id,
                "title": title,
                "source_id": source_id,
                "source_url": f"https://example.invalid/{source_id}",
                "source_file": f"corpus/source/raw/{source_id}.txt",
                "source_sha256": hashlib.sha256(section_id.encode()).hexdigest(),
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return {"format": "aa-canonical/1", "sections": sections}


def _synthetic_manifest() -> dict[str, Any]:
    return {
        "format": "aa-canonical-manifest/1",
        "artifact_sha256": hashlib.sha256(b"fixture-artifact").hexdigest(),
        "sections": [{"id": section_id} for section_id in EXPECTED_IDS],
    }


def _fixture_corpus() -> CanonicalCorpus:
    canonical = _synthetic_canonical()
    sections = tuple(
        CanonicalSection(
            id=str(entry["id"]),
            title=str(entry["title"]),
            source_id=str(entry["source_id"]),
            source_url=str(entry["source_url"]),
            source_file=str(entry["source_file"]),
            text=str(entry["text"]),
        )
        for entry in canonical["sections"]
    )
    return CanonicalCorpus(sections=sections)


# ---------------------------------------------------------------------------
# Builder logic on synthetic fixtures (offline).
# ---------------------------------------------------------------------------


def test_sentence_split_never_cuts_mid_token() -> None:
    builder = _load_builder()
    text = "He met Dr. Smith today. They talked."
    spans = builder.split_sentences(text)
    assert [text[s:e] for s, e in spans] == ["He met Dr. Smith today.", "They talked."]
    for start, end in spans:
        assert text[start:end] == text[start:end].strip()
        assert text[start:end]
    # Gaps between spans are whitespace only (tiling).
    for (_, prev_end), (cur_start, _) in zip(spans, spans[1:], strict=False):
        assert text[prev_end:cur_start].strip() == ""


def test_build_fixture_hierarchy_is_stable() -> None:
    builder = _load_builder()
    structure, _ = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    assert structure["format"] == "aa-corpus-structure/1"
    assert [s["id"] for s in structure["sections"]] == list(EXPECTED_IDS)
    assert structure["book"]["section_ids"] == list(EXPECTED_IDS)
    ids = (
        [s["id"] for s in structure["sections"]]
        + [p["id"] for p in structure["paragraphs"]]
        + [s["id"] for s in structure["sentences"]]
        + [c["id"] for c in structure["chunks"]]
    )
    assert len(set(ids)) == len(ids)
    for paragraph in structure["paragraphs"]:
        assert paragraph["parent_id"] == paragraph["section_id"]
        assert paragraph["char_end"] > paragraph["char_start"]
    for chunk in structure["chunks"]:
        assert chunk["parent_id"] == chunk["section_id"]
        assert chunk["paragraph_ids"] and chunk["sentence_ids"]


def test_build_is_deterministic() -> None:
    builder = _load_builder()
    first, first_texts = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    second, second_texts = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    assert builder.serialize_json(first) == builder.serialize_json(second)
    assert builder.serialize_json({"chunks": first_texts}) == builder.serialize_json(
        {"chunks": second_texts}
    )


def test_chunks_cover_exact_source_and_end_on_boundaries() -> None:
    builder = _load_builder()
    canonical = _synthetic_canonical()
    structure, chunk_texts = builder.build_structure(
        canonical=canonical, manifest=_synthetic_manifest()
    )
    by_section = {str(s["id"]): str(s["text"]) for s in canonical["sections"]}
    assert len(chunk_texts) == len(structure["chunks"])
    for chunk_row, chunk_entry in zip(structure["chunks"], chunk_texts, strict=True):
        section_text = by_section[str(chunk_row["section_id"])]
        expected = section_text[int(chunk_row["char_start"]) : int(chunk_row["char_end"])]
        assert chunk_entry["text"] == expected
        assert expected.strip()
        # Natural boundaries: every chunk is within MAX_CHUNK_CHARS.
        assert len(expected) <= builder.MAX_CHUNK_CHARS
    # The long chapter-1 paragraph is split on sentence boundaries (no
    # mid-sentence cuts) and still round-trips.
    chapter_chunks = [c for c in structure["chunks"] if c["section_id"] == "chapter-1"]
    assert len(chapter_chunks) >= 2


def test_parent_neighbor_navigation_spans_sections() -> None:
    builder = _load_builder()
    structure, _ = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    chunks = structure["chunks"]
    assert chunks[0]["prev_id"] is None
    assert chunks[-1]["next_id"] is None
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first["next_id"] == second["id"]
        assert second["prev_id"] == first["id"]
    # The chain crosses section boundaries (whole-book navigation).
    section_of = {c["id"]: c["section_id"] for c in chunks}
    assert any(
        section_of[first["id"]] != section_of[second["id"]]
        for first, second in zip(chunks, chunks[1:], strict=False)
    )


def test_tampered_source_fails_closed() -> None:
    builder = _load_builder()
    canonical = _synthetic_canonical()
    canonical["sections"][3]["text"] += " forged"
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_structure(canonical=canonical, manifest=_synthetic_manifest())


def test_missing_canonical_fails_closed(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    missing = tmp_path / "canonical.json"
    structure_out = tmp_path / "structure.json"
    assert builder.main(["--canonical", str(missing), "--structure", str(structure_out)]) == 1
    assert not structure_out.exists()


def test_runtime_resolves_every_chunk_to_exact_text() -> None:
    builder = _load_builder()
    structure_dict, _ = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    payload = builder.serialize_json(structure_dict)
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "structure.json"
        path.write_bytes(payload)
        structure = load_structure(path)
    corpus = _fixture_corpus()
    assert verify_chunk_round_trip(structure, corpus) == len(structure_dict["chunks"])
    first_chunk = next(n for n in structure.nodes.values() if n.kind == "chunk")
    assert (
        structure.chunk_text(first_chunk.id, corpus)
        == corpus.get(first_chunk.section_id).text[first_chunk.char_start : first_chunk.char_end]
    )
    assert structure.parent(first_chunk.id).id == first_chunk.section_id


def test_runtime_rejects_text_bearing_structure(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    structure_dict, chunk_texts = builder.build_structure(
        canonical=_synthetic_canonical(), manifest=_synthetic_manifest()
    )
    structure_dict["chunks"][0] = dict(structure_dict["chunks"][0])
    structure_dict["chunks"][0]["text"] = str(chunk_texts[0]["text"])
    path = tmp_path / "structure.json"
    path.write_bytes(builder.serialize_json(structure_dict))
    with pytest.raises(CorpusStructureError, match="must not carry literary text"):
        load_structure(path)


# ---------------------------------------------------------------------------
# Committed public artifacts (no canonical artifact required).
# ---------------------------------------------------------------------------


def _load_committed_structure_dict() -> dict[str, Any]:
    path = ROOT / "corpus" / "structure.json"
    assert path.exists(), "corpus/structure.json must be committed"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_committed_structure_loads_with_expected_scope() -> None:
    structure = load_structure(ROOT / "corpus" / "structure.json")
    assert structure.section_ids == EXPECTED_IDS
    kinds = {node.kind for node in structure.nodes.values()}
    assert kinds == {"book", "section", "paragraph", "sentence", "chunk"}
    assert structure.parent("chapter-1").id == "aa-book"
    assert [n.id for n in structure.children_of("aa-book") if n.kind == "section"] == list(
        EXPECTED_IDS
    )
    assert len([n for n in structure.nodes.values() if n.kind == "chunk"]) > 0
    assert len([n for n in structure.nodes.values() if n.kind == "sentence"]) > 0


def test_committed_structure_links_canonical_manifest() -> None:
    structure_dict = _load_committed_structure_dict()
    manifest = json.loads((ROOT / "corpus" / "canonical.manifest.json").read_text())
    assert structure_dict["canonical"]["artifact_sha256"] == manifest["artifact_sha256"]
    assert structure_dict["canonical"]["format"] == "aa-canonical/1"


def test_committed_book_map_fits_budget_and_names_every_section() -> None:
    structure = load_structure(ROOT / "corpus" / "structure.json")
    tokens = check_book_map(ROOT / "corpus" / "book-map.md", structure)
    assert tokens <= BOOK_MAP_BUDGET_TOKENS
    text = (ROOT / "corpus" / "book-map.md").read_text(encoding="utf-8")
    for section_id in EXPECTED_IDS:
        assert f"(`{section_id}`)" in text
    # Navigation aid, never evidence: routing + grounding contract markers.
    assert "book_read" in text
    assert "never this map" in text or "never evidence" in text


def test_committed_book_map_matches_deterministic_render() -> None:
    builder = _load_builder()
    structure_dict = _load_committed_structure_dict()
    rendered, tokens = builder.render_book_map(structure=structure_dict)
    committed = (ROOT / "corpus" / "book-map.md").read_text(encoding="utf-8")
    assert rendered == committed
    report = json.loads((ROOT / "corpus" / "structure-report.json").read_text())
    assert report["book_map"]["est_tokens"] == tokens
    assert report["book_map"]["fits_budget"] is True


def test_committed_report_matches_structure_counts() -> None:
    structure_dict = _load_committed_structure_dict()
    report = json.loads((ROOT / "corpus" / "structure-report.json").read_text())
    assert report["counts"] == {
        "sections": len(structure_dict["sections"]),
        "paragraphs": len(structure_dict["paragraphs"]),
        "sentences": len(structure_dict["sentences"]),
        "chunks": len(structure_dict["chunks"]),
    }
    assert report["canonical"]["artifact_sha256"] == structure_dict["canonical"]["artifact_sha256"]
    assert report["bytes"]["structure_json"] == (ROOT / "corpus" / "structure.json").stat().st_size
    assert report["counts"]["chunks"] == sum(
        int(s["chunk_count"]) for s in structure_dict["sections"]
    )


def test_committed_artifacts_carry_no_book_text() -> None:
    for name in ("corpus/structure.json", "corpus/book-map.md", "corpus/structure-report.json"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert len(text.encode("utf-8")) < 3_200_000
        # Distinctive canonical sentences must never appear in public files.
        for snippet in (
            "WAR FEVER ran high",
            "Here lies a Hampshire Grenadier",
            "Potential alcoholic that I was",
        ):
            assert snippet not in text, f"{name} reproduces book text"
    structure: CorpusStructure = load_structure(ROOT / "corpus" / "structure.json")
    for node in structure.nodes.values():
        assert "text" not in node.raw


def test_canonical_scope_is_not_reduced() -> None:
    builder = _load_builder()
    canonical = _synthetic_canonical()
    before = {str(s["id"]): str(s["text"]) for s in canonical["sections"]}
    builder.build_structure(canonical=canonical, manifest=_synthetic_manifest())
    after = {str(s["id"]): str(s["text"]) for s in canonical["sections"]}
    assert before == after
    assert len(before) == len(EXPECTED_IDS)
