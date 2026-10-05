"""P0-2 RAM-resident canonical chunk/index substrate tests (issue #115).

Covers the Definition of Done: qualified standard Russian segmentation,
E5-token child chunks with exact offsets, versioned/migrated RAM-resident
index (FTS5 :memory: + FAISS IndexFlatIP), exact-text RetrievalHit, stale
rejection, no per-turn disk I/O, and measurable baseline recall.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sqlite3
from typing import Any

import pytest

from aa.corpus import e5_tokens, structure
from aa.corpus.e5_tokens import CHILD_MAX_TOKENS, E5_HARD_INPUT_TOKENS
from aa.corpus.structure import (
    SECTION_IDS,
    build_full_structure,
    build_section_units,
    split_paragraphs,
    split_sentences,
)
from aa.qualification.sentence_qualification import (
    FIXTURE_VERSION,
    load_fixture,
    qualify_fixture,
)
from aa.retrieval import book_tools
from aa.retrieval.book_tools import book_expand, book_read, book_search, book_section
from aa.retrieval.index import (
    INDEX_BUILDER_VERSION,
    INDEX_FORMAT,
    LEGACY_BUILDER_VERSION,
    LEGACY_INDEX_FORMAT,
    HybridIndex,
    StaleIndexError,
    build_hybrid_index,
    close_hybrid_index,
    open_hybrid_index,
    search_aspect,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXTURE_PATH = ROOT / "qualification" / "sentence_boundary_fixture.v1.json"
QUALIFICATION_PATH = ROOT / "qualification" / "sentence_boundary_qualification.v1.json"


def _repo_lock() -> dict[str, object]:
    payload = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
    assert isinstance(payload, dict)
    return payload


def _fixture_sections() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": (
                    f"Fixture EN {section_id} opening sentence. Second sentence here. "
                    f"Third sentence follows. Fourth sentence closes."
                ),
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": (
                    "Фиктивный рассказ о тяге и пьянке. Герой бухает каждый вечер. "
                    "Утро после выпивки было тяжелым. Женушка переживает за семью. "
                    "Программа в действии требует честности. Трезвость возвращается."
                ),
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    return en_sections, ru_sections


def _build_index(tmp_path: pathlib.Path, **kwargs: Any) -> tuple[HybridIndex, dict[str, Any]]:
    en_sections, ru_sections = _fixture_sections()
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        **kwargs,
    )
    ru_manifest: dict[str, Any] = {
        "format": "aa-canonical-manifest-ru/1",
        "artifact_sha256": "r" * 64,
        "edition": "ru-edition",
    }
    en_manifest: dict[str, Any] = {
        "format": "aa-canonical-manifest/1",
        "artifact_sha256": "e" * 64,
        "edition": "en-edition",
    }
    index = build_hybrid_index(
        full,
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=_repo_lock(),
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )
    return index, full


def test_frozen_fixture_runs_against_both_and_selects_deterministically() -> None:
    assert FIXTURE_PATH.is_file()
    payload = load_fixture(FIXTURE_PATH)
    assert payload["fixture_version"] == FIXTURE_VERSION
    categories = {str(case["category"]) for case in payload["cases"]}
    for required in (
        "abbreviation",
        "initials",
        "dialogue",
        "quotes",
        "ellipsis",
        "list",
        "decimal",
        "chapter-typography",
    ):
        assert required in categories
    result = qualify_fixture(payload)
    assert result["razdel"]["round_trip_ok"] is True
    assert result["spacy"]["round_trip_ok"] is True
    # Deterministic rule: lowest error wins; razdel is the qualified winner.
    assert result["razdel"]["total_errors"] <= result["spacy"]["total_errors"]
    assert result["winner"] == "razdel"
    assert result["production_segmenter"] == "razdel-0.5.0"
    # Committed evidence matches the live qualification.
    assert QUALIFICATION_PATH.is_file()
    committed = json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
    assert committed["winner"] == result["winner"]
    assert committed["fixture_version"] == FIXTURE_VERSION
    assert committed["razdel"]["total_errors"] == result["razdel"]["total_errors"]
    assert committed["spacy"]["total_errors"] == result["spacy"]["total_errors"]


def test_production_uses_only_qualified_segmenter() -> None:
    source = (ROOT / "src" / "aa" / "corpus" / "structure.py").read_text(encoding="utf-8")
    assert "razdel" in source
    assert "_SENTENCE_END_RE" not in source
    assert "Sentencizer" not in source
    # No project-owned regex splitter and no hand-written AA exceptions.
    assert "import spacy" not in source
    assert "from spacy" not in source
    sentences_source = (ROOT / "src" / "aa" / "corpus" / "sentences.py").read_text(encoding="utf-8")
    assert "sentenize" in sentences_source
    assert "re.compile" not in sentences_source


def test_paragraph_sentence_child_round_trip_exact() -> None:
    en_sections, ru_sections = _fixture_sections()
    by_id = {str(item["id"]): str(item["text"]) for item in en_sections + ru_sections}
    assert by_id
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    assert full["format"] == "aa-corpus-structure-full/2"
    sections = full["sections"]
    assert isinstance(sections, list)
    for section in sections:
        assert isinstance(section, dict)
        section_id = str(section["id"])
        for lang in ("en", "ru"):
            branch = section[lang]
            assert isinstance(branch, dict)
            source_text = next(
                str(item["text"])
                for item in (en_sections if lang == "en" else ru_sections)
                if str(item["id"]) == section_id
            )
            paragraphs = branch["paragraphs"]
            chunks = branch["chunks"]
            assert isinstance(paragraphs, list) and isinstance(chunks, list)
            for para in paragraphs:
                assert isinstance(para, dict)
                recovered = source_text[int(para["char_start"]) : int(para["char_end"])]
                assert recovered == para["text"]
            for chunk in chunks:
                assert isinstance(chunk, dict)
                recovered = source_text[int(chunk["char_start"]) : int(chunk["char_end"])]
                assert recovered == chunk["text"]
                assert (
                    chunk["text_sha256"]
                    == hashlib.sha256(str(chunk["text"]).encode("utf-8")).hexdigest()
                )
    # Sentence-level round trip on a representative paragraph.
    paragraph = "Г-н Иванов и т.д. пришли вовремя. Они обсудили план."
    sentences = split_sentences(paragraph, 0)
    assert len(sentences) == 2
    for sentence in sentences:
        assert paragraph[sentence.char_start : sentence.char_end] == sentence.text
    assert split_paragraphs("A.\n\nB.")[0].text == "A."


def test_no_child_exceeds_token_budget_except_single_sentence(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    try:
        for record in index.chunks.values():
            tokens = e5_tokens.count_e5_tokens(record.text)
            if tokens > CHILD_MAX_TOKENS:
                # Only a single oversize sentence may exceed the target, and it
                # must still fit the hard input limit.
                assert tokens <= E5_HARD_INPUT_TOKENS
        # Explicit single-oversize-sentence case with a stub counter.
        sentences = split_sentences("Первое короткое. Второе короткое здесь.", 0)
        assert len(sentences) == 2
        ranges = structure.build_chunks(
            sentences,
            max_tokens=2,
            token_counter=lambda text: 10 if "Второе" in text else 1,
            paragraph_end=len("Первое короткое. Второе короткое здесь."),
        )
        assert len(ranges) == 2
    finally:
        close_hybrid_index(index)


def test_single_sentence_beyond_hard_limit_fails_explicitly() -> None:
    sentences = split_sentences("Одно предложение здесь. Второе там.", 0)
    with pytest.raises(ValueError, match="hard input limit"):
        structure.build_chunks(
            sentences,
            max_tokens=256,
            token_counter=lambda text: E5_HARD_INPUT_TOKENS + 1,
            paragraph_end=len("Одно предложение здесь. Второе там."),
        )


def test_parent_prev_next_relationships(tmp_path: pathlib.Path) -> None:
    index, full = _build_index(tmp_path)
    try:
        sections = full["sections"]
        assert isinstance(sections, list)
        for section in sections:
            assert isinstance(section, dict)
            for lang in ("ru",):
                branch = section[lang]
                assert isinstance(branch, dict)
                paragraphs = branch["paragraphs"]
                chunks = branch["chunks"]
                assert isinstance(paragraphs, list) and isinstance(chunks, list)
                for first, second in zip(paragraphs, paragraphs[1:], strict=False):
                    assert isinstance(first, dict) and isinstance(second, dict)
                    assert first["next"] == second["id"]
                    assert second["prev"] == first["id"]
                for first, second in zip(chunks, chunks[1:], strict=False):
                    assert isinstance(first, dict) and isinstance(second, dict)
                    assert first["next"] == second["id"]
                    assert second["prev"] == first["id"]
                for chunk in chunks:
                    assert isinstance(chunk, dict)
                    parent_id = str(chunk["parent"])
                    assert any(
                        isinstance(para, dict) and para["id"] == parent_id for para in paragraphs
                    )
        # Index-level neighbor links agree with the structure.
        for record in index.chunks.values():
            if record.prev is not None:
                assert record.prev in index.chunks
            if record.next is not None:
                assert record.next in index.chunks
    finally:
        close_hybrid_index(index)


def test_index_format_versioned_and_proves_substrate(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    try:
        assert index.metadata["format"] == INDEX_FORMAT == "aa-hybrid-index/2"
        assert index.metadata["builder_version"] == INDEX_BUILDER_VERSION == 2
        for key in (
            "ru_artifact_sha256",
            "embedding_model_id",
            "embedding_revision",
            "sentence_segmenter",
            "chunker",
            "chunker_version",
            "tokenizer",
            "chunk_policy",
            "chunk_max_tokens",
        ):
            assert index.metadata.get(key) not in (None, ""), key
        assert index.metadata["embedding_model_id"] == "intfloat/multilingual-e5-base"
        assert index.metadata["sentence_segmenter"] == "razdel-0.5.0"
        assert index.metadata["chunk_max_tokens"] == 256
        assert index.metadata["tokenizer"] == "intfloat/multilingual-e5-base@d1287505"
    finally:
        close_hybrid_index(index)


def test_stale_old_index_format_rejected(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    out_dir = index.directory
    close_hybrid_index(index)
    payload = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    payload["metadata"]["format"] = LEGACY_INDEX_FORMAT
    payload["metadata"]["builder_version"] = LEGACY_BUILDER_VERSION
    (out_dir / "index.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with pytest.raises(StaleIndexError):
        open_hybrid_index(out_dir)


def test_lexical_search_runs_against_in_memory_db(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    try:
        assert index.ram_resident is True
        assert index.lexical_conn is not None
        assert isinstance(index.lexical_conn, sqlite3.Connection)
        # The in-memory connection holds the FTS table.
        count = index.lexical_conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()
        assert count is not None and int(count[0]) == len(index.chunks)
        hits = search_aspect(index, ["тяга пьянка"])
        assert hits
    finally:
        close_hybrid_index(index)


def test_dense_search_uses_in_memory_exact_faiss(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    try:
        assert index.dense.dim > 0
        assert len(index.dense.ids) == len(index.chunks)
        # FAISS exact index is the dense substrate when the package is present.
        import importlib.util

        if importlib.util.find_spec("faiss") is not None:
            assert index.dense.use_faiss is True
        hits = search_aspect(index, ["трезвость возвращается"])
        assert hits
        assert any(hit.dense_rank is not None for hit in hits)
    finally:
        close_hybrid_index(index)


def test_blocked_filesystem_reads_do_not_break_hot_path(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, _ = _build_index(tmp_path)
    logical = next(iter(index.chunks.values())).logical_chunk_id
    section = next(iter(index.chunks.values())).section
    # Block every filesystem read after startup: hot path must stay in RAM.
    import pathlib as _pathlib
    import sqlite3 as _sqlite3

    def _blocked_connect(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("disk lexical.db must not be opened on the hot path")

    def _blocked_read_text(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("disk read must not occur on the hot path")

    def _blocked_read_bytes(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("disk read must not occur on the hot path")

    monkeypatch.setattr(_sqlite3, "connect", _blocked_connect)
    monkeypatch.setattr(_pathlib.Path, "read_text", _blocked_read_text)
    monkeypatch.setattr(_pathlib.Path, "read_bytes", _blocked_read_bytes)
    try:
        hits = search_aspect(index, ["пьянка тяга"])
        assert hits
        read = book_read(index, logical)
        assert read["text"] == index.chunks[hits[0].chunk_id].text or read["text"]
        expanded = book_expand(index, logical, before=1, after=1)
        assert expanded["chunks"]
        section_out = book_section(index, section, chunk_offset=0, chunk_limit=2)
        assert section_out["chunks"]
    finally:
        close_hybrid_index(index)


def test_exact_text_retrieval_hit_contract(tmp_path: pathlib.Path) -> None:
    index, _ = _build_index(tmp_path)
    try:
        hits = search_aspect(index, ["программа требует честности"])
        assert hits
        hit = hits[0]
        for field in (
            "text",
            "chunk_id",
            "logical_chunk_id",
            "section_id",
            "parent_id",
            "prev_id",
            "next_id",
            "source_id",
            "source_file",
            "char_start",
            "char_end",
            "text_sha256",
            "lexical_rank",
            "dense_rank",
            "fused_rank",
        ):
            assert hasattr(hit, field), field
        assert hit.text
        assert hashlib.sha256(hit.text.encode("utf-8")).hexdigest() == hit.text_sha256
        assert hit.section_id == hit.section
        assert hit.parent_id == hit.parent
        payload = hit.to_dict()
        assert payload["text"] == hit.text
        assert payload["section_id"] == hit.section
        # book_search exposes the same exact-text contract.
        result = book_search(
            index,
            {
                "aspect_id": "craving",
                "semantic_query_ru": "тяга к алкоголю",
                "lexical_queries_ru": ["пить алкоголь"],
                "lexical_query_en": None,
            },
        )
        candidate = result["candidates"][0]
        assert candidate["text"]
        assert (
            hashlib.sha256(candidate["text"].encode("utf-8")).hexdigest()
            == candidate["ru_locator"]["text_sha256"]
        )
    finally:
        close_hybrid_index(index)


def test_baseline_gold_fixtures_remain_measurable(tmp_path: pathlib.Path) -> None:
    from aa.qualification.aa_retrieval import build_fixture_full, load_gold, run_case

    cases = load_gold(ROOT / "qualification" / "aa-retrieval.gold.v1.json")
    ru_cases = [case for case in cases if case.language == "ru"][:4]
    assert ru_cases
    full = build_fixture_full(max_tokens=256)
    ru_manifest_gold: dict[str, Any] = {
        "format": "aa-canonical-manifest-ru/1",
        "artifact_sha256": "r" * 64,
        "edition": "ru-edition",
    }
    en_manifest_gold: dict[str, Any] = {
        "format": "aa-canonical-manifest/1",
        "artifact_sha256": "e" * 64,
        "edition": "en-edition",
    }
    index = build_hybrid_index(
        full,
        ru_manifest=ru_manifest_gold,
        en_manifest=en_manifest_gold,
        embedding_lock=_repo_lock(),
        out_dir=tmp_path / "retrieval-gold",
        backend="hashing",
    )
    try:
        recalls = []
        for case in ru_cases:
            result = run_case(index, case)
            recalls.append(result.recall_at_5)
        assert len(recalls) == len(ru_cases)
        # Measurable: recall is defined (True/False per case), harness runs.
        assert all(isinstance(value, bool) for value in recalls)
    finally:
        close_hybrid_index(index)


def test_no_remote_vector_database_and_no_hnsw() -> None:
    root = ROOT / "src" / "aa" / "retrieval"
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for snippet in ("IndexHNSW", "hnswlib", "api.openai.com", "api.anthropic.com"):
            assert snippet not in text, f"{path.name} forbids {snippet!r}"
    dense_text = (root / "dense.py").read_text(encoding="utf-8")
    assert "IndexFlatIP" in dense_text
    assert "local_files_only" in dense_text


def test_book_tools_share_ram_resident_implementation() -> None:
    assert book_tools.__name__ == "aa.retrieval.book_tools"
    assert book_search.__module__ == book_tools.__name__
    assert book_read.__module__ == book_tools.__name__
    assert book_expand.__module__ == book_tools.__name__
    assert book_section.__module__ == book_tools.__name__


def test_e5_token_budget_defaults_and_hard_limit() -> None:
    assert CHILD_MAX_TOKENS == 256
    assert E5_HARD_INPUT_TOKENS == 512
    # Real E5 tokenizer accounting works on Russian text.
    count = e5_tokens.count_e5_tokens("Фиктивный рассказ о тяге.")
    assert count > 0
    assert e5_tokens.e5_hard_limit() == 512


def test_section_units_reject_legacy_char_budget() -> None:
    legacy: dict[str, Any] = {"max_chars": 1500}
    with pytest.raises(ValueError, match="max_tokens"):
        build_section_units(
            section_id="chapter-1",
            lang="ru",
            section_text="Первое предложение. Второе предложение.",
            source_id="ru-fourth-edition-txt",
            source_file="corpus/source/raw-ru/aa-big-book.txt",
            source_sha256="0" * 64,
            edition="ru-edition",
            corpus_version="ru-v1",
            **legacy,
        )
