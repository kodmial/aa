"""Aligned RU/EN hierarchy and compact book map tests (issue #8).

All literary content uses invented fixture sentences; no canonical book
text is committed. Real titles/checksums are read from the committed
manifests (metadata only) and never duplicate book prose.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys
from typing import Any

from aa.corpus.budget import BOOK_MAP_BUDGET_TOKENS, estimate_text_tokens
from aa.corpus.structure import (
    BOOK_ID,
    SECTION_IDS,
    build_full_structure,
    build_public_structure,
    build_section_units,
    render_book_map,
    sha256_text,
    split_paragraphs,
    split_sentences,
)


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _fixture_section_texts() -> tuple[dict[str, str], dict[str, str]]:
    """Invented EN/RU section texts with deliberately different splits."""
    en: dict[str, str] = {}
    ru: dict[str, str] = {}
    for position, section_id in enumerate(SECTION_IDS):
        en_paras = [
            f"Fixture EN {section_id} opening paragraph. It carries two sentences.",
            f"Fixture EN {section_id} second paragraph with one sentence.",
            f"Fixture EN {section_id} third paragraph. It also ends here. Done.",
        ]
        # RU merges the last two EN paragraphs differently: only two blocks.
        ru_paras = [
            f"Фиктивный RU {section_id} первый абзац. В нем два предложения.",
            f"Фиктивный RU {section_id} второй абзац объединяет смысл. "
            f"Еще одно предложение здесь. Конец абзаца {position}.",
        ]
        en[section_id] = "\n\n".join(en_paras) + "\n"
        ru[section_id] = "\n\n".join(ru_paras) + "\n"
    return en, ru


def _fixture_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    en_texts, ru_texts = _fixture_section_texts()
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_text = en_texts[section_id]
        ru_text = ru_texts[section_id]
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": en_text,
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": ru_text,
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    return en_sections, ru_sections


def _sections_of(mapping: dict[str, object], key: str = "sections") -> list[dict[str, Any]]:
    value = mapping.get(key)
    assert isinstance(value, list)
    out: list[dict[str, Any]] = []
    for item in value:
        assert isinstance(item, dict)
        out.append(item)
    return out


def _branch_of(section: dict[str, Any], lang: str) -> dict[str, Any]:
    branch = section.get(lang)
    assert isinstance(branch, dict)
    return branch


def _nodes_of(branch: dict[str, Any], key: str) -> list[dict[str, Any]]:
    nodes = branch.get(key)
    assert isinstance(nodes, list)
    out: list[dict[str, Any]] = []
    for item in nodes:
        assert isinstance(item, dict)
        out.append(item)
    return out


def test_section_ids_are_language_neutral_and_ordered() -> None:
    assert list(SECTION_IDS) == [
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
    ]
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-version",
        ru_corpus_version="ru-version",
    )
    assert full["book"] == BOOK_ID
    sections = _sections_of(full)
    ids = [str(section["id"]) for section in sections]
    assert ids == list(SECTION_IDS)
    for section in sections:
        assert section["book"] == BOOK_ID
        assert section["parent"] == BOOK_ID


def test_section_prev_next_chain_is_neutral() -> None:
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-version",
        ru_corpus_version="ru-version",
    )
    sections = _sections_of(full)
    for position, section in enumerate(sections):
        expected_prev = SECTION_IDS[position - 1] if position > 0 else None
        expected_next = SECTION_IDS[position + 1] if position + 1 < len(SECTION_IDS) else None
        assert section["prev"] == expected_prev
        assert section["next"] == expected_next


def test_per_language_provenance_and_checksums() -> None:
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    section = sections[1]
    assert section["alignment"] == {"status": "aligned", "confidence": 1.0}
    for lang in ("en", "ru"):
        branch = _branch_of(section, lang)
        assert branch["language"] == lang
        assert branch["corpus_version"] == f"{lang}-v1"
        assert branch["edition"] == f"{lang}-edition"
        assert len(str(branch["source_sha256"])) == 64
        # Recompute from the fixture section text instead of trusting the node.
        fixture = (
            next(item for item in en_sections if item["id"] == section["id"])["text"]
            if lang == "en"
            else next(item for item in ru_sections if item["id"] == section["id"])["text"]
        )
        assert branch["text_sha256"] == sha256_text(str(fixture))
        for node in (*_nodes_of(branch, "paragraphs"), *_nodes_of(branch, "chunks")):
            assert node["language"] == lang
            assert node["parent"] is not None
            assert node["section"] == section["id"]
            assert node["book"] == BOOK_ID
            assert node["source_sha256"] == branch["source_sha256"]
            alignment = node["alignment"]
            assert isinstance(alignment, dict)
            assert alignment["confidence"] == 1.0


def test_split_merge_is_represented_safely() -> None:
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    for section in sections:
        en_branch = _branch_of(section, "en")
        ru_branch = _branch_of(section, "ru")
        # Fixtures deliberately differ: 3 EN paragraphs vs 2 RU paragraphs.
        assert len(_nodes_of(en_branch, "paragraphs")) == 3
        assert len(_nodes_of(ru_branch, "paragraphs")) == 2
        # No forced equality: paragraph ids are per-language physical units.
        en_ids = {node["id"] for node in _nodes_of(en_branch, "paragraphs")}
        ru_ids = {node["id"] for node in _nodes_of(ru_branch, "paragraphs")}
        assert not (en_ids & ru_ids)
        for node in (*_nodes_of(en_branch, "paragraphs"), *_nodes_of(ru_branch, "paragraphs")):
            alignment = node["alignment"]
            assert isinstance(alignment, dict)
            assert alignment["status"] == "section-aligned-only"
        alignment = section["alignment"]
        assert isinstance(alignment, dict)
        assert alignment["status"] == "aligned"


def test_every_chunk_round_trips_to_exact_source_text() -> None:
    en_sections, ru_sections = _fixture_inputs()
    by_id = {str(item["id"]): str(item["text"]) for item in en_sections + ru_sections}
    # Same text appears in both languages only via distinct fixtures; map each
    # chunk back through its own language branch.
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    for section in sections:
        section_id = str(section["id"])
        for lang in ("en", "ru"):
            branch = _branch_of(section, lang)
            source_text = next(
                str(item["text"])
                for item in (en_sections if lang == "en" else ru_sections)
                if str(item["id"]) == section_id
            )
            assert by_id  # fixtures are keyed per test run
            for chunk in _nodes_of(branch, "chunks"):
                recovered = source_text[int(str(chunk["char_start"])) : int(str(chunk["char_end"]))]
                assert recovered == chunk["text"]
                assert chunk["text_sha256"] == sha256_text(str(chunk["text"]))


def test_chunks_use_natural_boundaries_only() -> None:
    text = "First sentence here. Second sentence follows.\n\nNext paragraph alone."
    units = build_section_units(
        section_id="chapter-5",
        lang="en",
        section_text=text,
        source_id="core-pages-1-164",
        source_file="corpus/source/raw/AA.txt",
        source_sha256="0" * 64,
        edition="en-edition",
        corpus_version="en-v1",
        max_chars=40,
    )
    chunks_raw = units["chunks"]
    assert isinstance(chunks_raw, list) and len(chunks_raw) >= 2
    chunks: list[dict[str, Any]] = []
    for item in chunks_raw:
        assert isinstance(item, dict)
        chunks.append(item)
    # Chunk boundaries fall on sentence ends: every chunk ends with terminal
    # punctuation or at the paragraph end, never mid-sentence.
    for chunk in chunks:
        chunk_text = str(chunk["text"]).strip()
        assert chunk_text
        assert chunk_text[-1] in ".!?…\"'”’»)]" or chunk is chunks[-1]
    # Chunks never cross paragraph boundaries.
    paragraphs_raw = units["paragraphs"]
    assert isinstance(paragraphs_raw, list) and len(paragraphs_raw) == 2
    paragraphs: list[dict[str, Any]] = []
    for item in paragraphs_raw:
        assert isinstance(item, dict)
        paragraphs.append(item)
    first_para_chunks = paragraphs[0]["chunks"]
    assert isinstance(first_para_chunks, list)
    for chunk_id_value in first_para_chunks:
        chunk = next(item for item in chunks if item["id"] == chunk_id_value)
        assert int(str(chunk["char_start"])) >= int(str(paragraphs[0]["char_start"]))
        assert int(str(chunk["char_end"])) <= int(str(paragraphs[0]["char_end"]))


def test_ru_chunks_are_primary_retrieval_units() -> None:
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    for section in sections:
        for chunk in _nodes_of(_branch_of(section, "ru"), "chunks"):
            assert chunk["role"] == "primary-retrieval"
            assert str(chunk["id"]).startswith(f"{section['id']}:ru:c")
            alignment = chunk["alignment"]
            assert isinstance(alignment, dict)
            assert alignment["status"] == "unaligned-explicit"
        for chunk in _nodes_of(_branch_of(section, "en"), "chunks"):
            assert chunk["role"] == "reference-control"
            assert str(chunk["id"]).startswith(f"{section['id']}:en:c")


def test_paragraph_and_chunk_neighbor_links() -> None:
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    branch = _branch_of(sections[0], "ru")
    for key in ("paragraphs", "chunks"):
        nodes = _nodes_of(branch, key)
        assert nodes
        assert nodes[0]["prev"] is None
        assert nodes[-1]["next"] is None
        for first, second in zip(nodes, nodes[1:], strict=False):
            assert first["next"] == second["id"]
            assert second["prev"] == first["id"]


def test_public_structure_is_metadata_only() -> None:
    root = _repo_root()
    en_manifest = json.loads((root / "corpus" / "canonical.manifest.json").read_text())
    ru_manifest = json.loads((root / "corpus" / "canonical.ru.manifest.json").read_text())
    public = build_public_structure(en_manifest=en_manifest, ru_manifest=ru_manifest)
    payload = json.dumps(public, sort_keys=True, ensure_ascii=False)
    assert '"text"' not in payload
    public_sections = _sections_of(public)
    assert [section["id"] for section in public_sections] == list(SECTION_IDS)
    for section in public_sections:
        titles = section["titles"]
        assert isinstance(titles, dict) and set(titles) == {"en", "ru"}
        assert section["topic_en"]
        alignment = section["alignment"]
        assert isinstance(alignment, dict)
        assert alignment["status"] == "aligned"


def test_compact_map_stays_within_budget_and_english() -> None:
    root = _repo_root()
    en_manifest = json.loads((root / "corpus" / "canonical.manifest.json").read_text())
    ru_manifest = json.loads((root / "corpus" / "canonical.ru.manifest.json").read_text())
    public = build_public_structure(en_manifest=en_manifest, ru_manifest=ru_manifest)
    book_map = render_book_map(public)
    assert estimate_text_tokens(book_map) <= BOOK_MAP_BUDGET_TOKENS
    assert "navigation only, never evidence" in book_map
    for section_id in SECTION_IDS:
        assert section_id in book_map
    cyrillic = sum(1 for char in book_map if 0x0400 <= ord(char) <= 0x04FF)
    assert cyrillic == 0


def test_committed_public_artifacts_carry_no_book_text() -> None:
    root = _repo_root()
    structure_path = root / "corpus" / "structure.json"
    map_path = root / "corpus" / "book-map.md"
    assert structure_path.is_file()
    assert map_path.is_file()
    structure_text = structure_path.read_text(encoding="utf-8")
    map_text = map_path.read_text(encoding="utf-8")
    assert len(structure_text.encode("utf-8")) < 30_000
    assert '"text"' not in structure_text
    for snippet in ("Fixture EN chapter-5", "Фиктивный RU chapter-5", "WAR FEVER"):
        assert snippet not in structure_text
        assert snippet not in map_text
    structure = json.loads(structure_text)
    assert structure["format"] == "aa-aligned-structure/1"
    assert [section["id"] for section in structure["sections"]] == list(SECTION_IDS)
    assert estimate_text_tokens(map_text) <= BOOK_MAP_BUDGET_TOKENS


def test_no_machine_translation_in_canonical_path() -> None:
    root = _repo_root()
    for name in ("src/aa/corpus/structure.py", "scripts/build_corpus_structure.py"):
        text = (root / name).read_text(encoding="utf-8").lower()
        for forbidden in ("import openai", "from openai", "import anthropic", "from anthropic"):
            assert forbidden not in text, f"{name} must not use LLM APIs"
        for forbidden in ("translate(", "machine_translation", "googletrans", "deepl"):
            assert forbidden not in text, f"{name} must not machine-translate"
    # Chunk text is always a verbatim slice of its own language section text,
    # never a translation of the other language.
    en_sections, ru_sections = _fixture_inputs()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    sections = _sections_of(full)
    section = sections[3]
    en_branch = _branch_of(section, "en")
    ru_branch = _branch_of(section, "ru")
    en_source = next(item for item in en_sections if item["id"] == section["id"])["text"]
    ru_source = next(item for item in ru_sections if item["id"] == section["id"])["text"]
    assert isinstance(en_source, str) and isinstance(ru_source, str)
    for chunk in _nodes_of(en_branch, "chunks"):
        assert str(chunk["text"]) in en_source
    for chunk in _nodes_of(ru_branch, "chunks"):
        assert str(chunk["text"]) in ru_source


def test_build_is_deterministic() -> None:
    en_sections, ru_sections = _fixture_inputs()
    kwargs: dict[str, Any] = {
        "en_sections": en_sections,
        "ru_sections": ru_sections,
        "en_edition": "en-edition",
        "ru_edition": "ru-edition",
        "en_corpus_version": "en-v1",
        "ru_corpus_version": "ru-v1",
    }
    first = json.dumps(build_full_structure(**kwargs), sort_keys=True, ensure_ascii=False)
    second = json.dumps(build_full_structure(**kwargs), sort_keys=True, ensure_ascii=False)
    assert first == second


def test_sentence_splitting_handles_russian_punctuation() -> None:
    paragraph = "Первое предложение здесь. Второе — там! А третье… Четвертое?"
    sentences = split_sentences(paragraph, 10)
    assert len(sentences) == 4
    assert "".join(item.text for item in sentences) == paragraph
    assert [item.char_start for item in sentences] == [
        10,
        10 + len(sentences[0].text),
        10 + len(sentences[0].text + sentences[1].text),
        10 + len(sentences[0].text + sentences[1].text + sentences[2].text),
    ]


def test_paragraph_split_preserves_exact_offsets() -> None:
    text = "Para one line one.\nStill para one.\n\nPara two here.\n\n\nPara three."
    spans = split_paragraphs(text)
    assert len(spans) == 3
    for span in spans:
        assert text[span.char_start : span.char_end] == span.text


def _load_builder_script() -> Any:
    path = _repo_root() / "scripts" / "build_corpus_structure.py"
    spec = importlib.util.spec_from_file_location("build_corpus_structure_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_script_builds_all_artifacts(tmp_path: pathlib.Path) -> None:
    en_sections, ru_sections = _fixture_inputs()
    en_manifest = {
        "format": "aa-canonical-manifest/1",
        "builder_version": 1,
        "edition": "en-edition",
        "artifact_sha256": "e" * 64,
        "sections": [
            {"id": str(item["id"]), "title": str(item["title"]), "text_sha256": "0" * 64}
            for item in en_sections
        ],
    }
    ru_manifest = {
        "format": "aa-canonical-manifest-ru/1",
        "builder_version": 1,
        "edition": "ru-edition",
        "artifact_sha256": "f" * 64,
        "sections": [
            {
                "id": str(item["id"]),
                "title": str(item["title"]),
                "text_sha256": "1" * 64,
                "chars": len(str(item["text"])),
            }
            for item in ru_sections
        ],
    }
    en_artifact = {
        "format": "aa-canonical/1",
        "sections": [
            {
                "id": str(item["id"]),
                "title": str(item["title"]),
                "text": str(item["text"]),
                "source_id": str(item["source_id"]),
                "source_file": str(item["source_file"]),
                "source_sha256": str(item["source_sha256"]),
            }
            for item in en_sections
        ],
    }
    ru_artifact = {
        "format": "aa-canonical-ru/1",
        "sections": [
            {
                "id": str(item["id"]),
                "title": str(item["title"]),
                "text": str(item["text"]),
                "source_id": str(item["source_id"]),
                "source_file": str(item["source_file"]),
                "source_sha256": str(item["source_sha256"]),
            }
            for item in ru_sections
        ],
    }
    en_manifest_path = tmp_path / "en-manifest.json"
    ru_manifest_path = tmp_path / "ru-manifest.json"
    en_canonical_path = tmp_path / "canonical.json"
    ru_canonical_path = tmp_path / "canonical.ru.json"
    full_path = tmp_path / "corpus_structure.json"
    public_path = tmp_path / "structure.json"
    map_path = tmp_path / "book-map.md"
    en_manifest_path.write_text(json.dumps(en_manifest), encoding="utf-8")
    ru_manifest_path.write_text(json.dumps(ru_manifest), encoding="utf-8")
    en_canonical_path.write_text(json.dumps(en_artifact), encoding="utf-8")
    ru_canonical_path.write_text(json.dumps(ru_artifact), encoding="utf-8")
    script = _repo_root() / "scripts" / "build_corpus_structure.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--en-manifest",
            str(en_manifest_path),
            "--ru-manifest",
            str(ru_manifest_path),
            "--en-canonical",
            str(en_canonical_path),
            "--ru-canonical",
            str(ru_canonical_path),
            "--full-output",
            str(full_path),
            "--public-output",
            str(public_path),
            "--map-output",
            str(map_path),
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    full = json.loads(full_path.read_text(encoding="utf-8"))
    assert isinstance(full, dict)
    assert [section["id"] for section in _sections_of(full)] == list(SECTION_IDS)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    assert '"text"' not in public_path.read_text(encoding="utf-8")
    assert public["format"] == "aa-aligned-structure/1"
    rendered_map = map_path.read_text(encoding="utf-8")
    assert estimate_text_tokens(rendered_map) <= BOOK_MAP_BUDGET_TOKENS


def test_builder_fails_closed_without_canonical(tmp_path: pathlib.Path) -> None:
    module = _load_builder_script()
    missing = tmp_path / "missing.json"
    assert module.main(["--en-canonical", str(missing)]) != 0
