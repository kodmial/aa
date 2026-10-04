"""RU-first hybrid retrieval index tests (issue #17).

All literary content uses invented fixture sentences; no canonical book
text is committed. Fixtures exercise Russian morphology, slang,
misspellings and ordinary exact queries, plus stale-index rejection and
exact provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess
import sys
from typing import Any

from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.retrieval.dense import (
    ExactIPIndex,
    hashing_embed,
    l2_normalize,
)
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT, RRF_K
from aa.retrieval.index import (
    HybridIndex,
    StaleIndexError,
    build_hybrid_index,
    logical_chunk_id,
    open_hybrid_index,
    search_aspect,
    search_plan,
)
from aa.retrieval.lexical import LEXICAL_TOP_K, lexical_search
from aa.retrieval.normalize import normalize_ru, ru_stem
from aa.retrieval.planner import (
    PlannerError,
    aspect_search_queries,
    validate_plan,
)

RU_FIXTURES: dict[str, str] = {
    "doctors-opinion": (
        "Фиктивное мнение доктора о тяге. Наблюдение за пациентами продолжается.\n\n"
        "Второй абзац мнения доктора. Выводы записываются аккуратно."
    ),
    "chapter-1": (
        "Фиктивный рассказ о первом глотке. Герой начал пить каждый вечер.\n\n"
        "Второй абзац рассказа. Утро после выпивки было тяжелым."
    ),
    "chapter-2": (
        "Фиктивный выход есть для пьющих. Надежда и поддержка рядом.\n\n"
        "Второй абзац выхода. Сообщество встречает новичков тепло."
    ),
    "chapter-3": (
        "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю приходит внезапно.\n\n"
        "Второй абзац об алкоголизма последствиях. Многие опасаются алкоголизма в семье."
    ),
    "chapter-4": (
        "Фиктивные размышления агностика. Готовность принять помощь растет.\n\n"
        "Второй абзац агностика. Сомнения обсуждаются открыто."
    ),
    "chapter-5": (
        "Фиктивная программа в действии требует честности. Практические шаги каждый день.\n\n"
        "Второй абзац программы. Утренний настрой задает тон."
    ),
    "chapter-6": (
        "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию.\n\n"
        "Второй абзац работы. Вечером подводим итоги дня."
    ),
    "chapter-7": (
        "Фиктивная работа с другими людьми. Несем весть тем кто страдает.\n\n"
        "Второй абзац помощи. Разговор ведется спокойно и честно."
    ),
    "chapter-8": (
        "Фиктивная жена ругает из-за пьянки. Женушка переживает за семью.\n\n"
        "Второй абзац о семье. Пьянство разрушает доверие постепенно."
    ),
    "chapter-9": (
        "Фиктивные новые отношения в семье. Доверие возвращается постепенно.\n\n"
        "Второй абзац семьи. Разговоры становятся спокойнее."
    ),
    "chapter-10": (
        "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна.\n\n"
        "Второй абзац работодателям. Поддержка коллег помогает многим."
    ),
    "chapter-11": (
        "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется, "
        "хочется жить трезво.\n\n"
        "Второй абзац будущего. Планы строятся на трезвую голову."
    ),
}


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _fixture_full() -> dict[str, Any]:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": f"Fixture EN {section_id} opening. Second sentence here.\n\n"
                f"Fixture EN {section_id} second paragraph.",
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": hashlib.sha256(b"en-source").hexdigest(),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": RU_FIXTURES[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": hashlib.sha256(b"ru-source").hexdigest(),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
    )
    return {"full": full, "en_sections": en_sections, "ru_sections": ru_sections}


def _fixture_manifests() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    lock = json.loads((_repo_root() / "corpus" / "embedding.lock.json").read_text())
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
    return ru_manifest, en_manifest, lock


def _build_index(tmp_path: pathlib.Path) -> HybridIndex:
    fixture = _fixture_full()
    ru_manifest, en_manifest, lock = _fixture_manifests()
    return build_hybrid_index(
        fixture["full"],
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=lock,
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )


def _valid_plan(original: str, aspect_id: str = "a1") -> dict[str, Any]:
    return {
        "schema_version": "ru-query-plan-v1",
        "utterance_id": "turn-1",
        "language": "ru",
        "aspects": [
            {
                "aspect_id": aspect_id,
                "meaning": "drinking and alcohol use",
                "semantic_queries_ru": ["употребление алкоголя"],
                "lexical_queries_ru": ["пить алкоголь"],
                "lexical_query_en": None,
                "ambiguity": "low",
                "forbidden_inferences": ["do_not_diagnose_alcoholism"],
            }
        ],
    }


def test_normalization_unifies_yo_and_stems_inflections() -> None:
    assert normalize_ru("Ёлка АЛКОГОЛЬ") == "елка алкоголь"
    assert ru_stem("алкоголизма") == ru_stem("алкоголизм")
    assert ru_stem("алкоголизме") == ru_stem("алкоголизм")
    assert ru_stem("бухаю") == ru_stem("бухать")


def test_ru_bm25_exact_query_finds_chunk(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    hits = search_aspect(index, ["феномен тяги"])
    assert hits
    assert hits[0].section == "chapter-3"
    assert hits[0].lexical_rank is not None
    assert "chapter-3" in hits[0].chunk_id


def test_ru_morphology_query_finds_base_form(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    # Prepositional form absent verbatim from fixtures; stemmed FTS must meet it.
    hits = search_aspect(index, ["рассказ об алкоголизме"])
    assert hits
    assert any(hit.section == "chapter-3" for hit in hits[:3])


def test_original_plus_rewrite_fusion_recovers_slang(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    original = "я бухаю каждый вечер"
    rewrites = ["пить алкоголь каждый вечер", "употребление алкоголя вечером"]
    fused = search_aspect(index, [original, *rewrites])
    sections = [hit.section for hit in fused]
    assert "chapter-1" in sections or "chapter-11" in sections
    # Rewrite fusion must not lose the exact-match section either.
    exact_only = search_aspect(index, ["пить каждый вечер"])
    assert exact_only and exact_only[0].section == "chapter-1"


def test_misspelling_recovered_via_dense_and_rewrite(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    hits = search_aspect(index, ["алкаголь", "алкоголь"])
    assert hits
    assert any(hit.section == "chapter-3" for hit in hits[:3])
    assert any(hit.dense_rank is not None for hit in hits[:3])


def test_dense_embeddings_normalized_and_exact_ip(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    assert index.dense.dim > 0
    vector = hashing_embed("алкоголизм тяга", dim=index.dense.dim)
    norm = math.sqrt(sum(value * value for value in vector))
    assert abs(norm - 1.0) < 1e-9
    scored = index.dense.search(vector, top_k=3)
    assert scored
    assert scored[0][1] <= 1.0 + 1e-6
    # Exact search over stored vectors: self-match scores ~1.0 on top.
    stored = index.dense.vectors[0]
    top = index.dense.search(stored, top_k=1)
    assert top[0][0] == index.dense.ids[0]
    assert abs(top[0][1] - 1.0) < 1e-4


def test_exact_ip_index_rejects_unnormalized() -> None:
    try:
        ExactIPIndex.build(["a"], [[3.0, 4.0]])
    except ValueError:
        return
    raise AssertionError("unnormalized vectors must fail closed")


def test_hybrid_fusion_dedup_diversity_caps(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    hits = search_aspect(index, ["фиктивный"])
    assert hits
    assert len(hits) <= MAX_CANDIDATES_PER_ASPECT
    ids = [hit.chunk_id for hit in hits]
    assert len(set(ids)) == len(ids)
    logical = [hit.logical_chunk_id for hit in hits]
    assert len(set(logical)) == len(logical)
    from collections import Counter

    counts = Counter(hit.section for hit in hits)
    assert max(counts.values()) <= 4
    ranks = [hit.fused_rank for hit in hits]
    assert ranks == list(range(1, len(hits) + 1))
    scores = [hit.fused_score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_overlapping_chunks_deduplicated() -> None:
    from aa.retrieval.fusion import FusedCandidate, deduplicate_overlaps

    candidates = [
        FusedCandidate("c1", 0.9, 1, 1, -1.0, 0.9),
        FusedCandidate("c2", 0.8, 2, 2, -2.0, 0.8),
        FusedCandidate("c3", 0.7, 3, 3, -3.0, 0.7),
    ]
    spans = {"c1": ("chapter-3", 0, 100), "c2": ("chapter-3", 50, 150), "c3": ("chapter-5", 0, 100)}
    kept = deduplicate_overlaps(candidates, spans=spans)
    kept_ids = [item.chunk_id for item in kept]
    assert "c1" in kept_ids and "c3" in kept_ids
    assert "c2" not in kept_ids


def test_search_params_are_fixed(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    params = index.metadata["search_params"]
    assert params["lexical_top_k"] == LEXICAL_TOP_K == 40
    assert params["dense_top_k"] == 40
    assert params["rrf_k"] == RRF_K == 60
    assert params["max_per_aspect"] == MAX_CANDIDATES_PER_ASPECT == 12


def test_stale_index_rejected_on_manifest_change(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    out_dir = index.directory
    ru_manifest, _, lock = _fixture_manifests()
    _, en_manifest, _ = _fixture_manifests()
    opened = open_hybrid_index(out_dir, ru_manifest_path=None)
    assert opened.chunk_count == index.chunk_count
    # Live manifest with a rotated artifact SHA must invalidate the index.
    rotated = tmp_path / "ru-manifest.json"
    tampered = dict(ru_manifest)
    tampered["artifact_sha256"] = "0" * 64
    rotated.write_text(json.dumps(tampered), encoding="utf-8")
    try:
        open_hybrid_index(out_dir, ru_manifest_path=rotated)
    except StaleIndexError:
        pass
    else:
        raise AssertionError("rotated RU artifact must reject the index")
    lock_path = tmp_path / "lock.json"
    tampered_lock = dict(lock)
    tampered_lock["revision"] = "1" * 40
    lock_path.write_text(json.dumps(tampered_lock), encoding="utf-8")
    try:
        open_hybrid_index(out_dir, lock_path=lock_path)
    except StaleIndexError:
        pass
    else:
        raise AssertionError("rotated embedding revision must reject the index")
    assert en_manifest["artifact_sha256"] == "e" * 64


def test_exact_provenance_and_preview_marking(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    hits = search_aspect(index, ["программа в действии требует честности"])
    assert hits
    hit = next(item for item in hits if item.section == "chapter-5")
    record = index.chunks[hit.chunk_id]
    assert (
        hit.text_sha256
        == record.text_sha256
        == hashlib.sha256(record.text.encode("utf-8")).hexdigest()
    )
    assert (hit.char_start, hit.char_end) == (record.char_start, record.char_end)
    assert hit.source_id == "ru-fourth-edition-txt"
    assert "never evidence" in hit.preview
    assert hit.logical_chunk_id == logical_chunk_id(hit.chunk_id)
    assert hit.logical_chunk_id.startswith("chapter-5:")
    assert ":ru:" not in hit.logical_chunk_id
    assert hit.parent.endswith(f":ru:p{int(hit.parent.split('p')[-1]):04d}")
    payload = hit.to_dict()
    assert payload["ru_locator"]["text_sha256"] == hit.text_sha256
    assert payload["versions"]["ru_corpus_version"]


def test_en_control_metadata_available_for_future_branch(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    assert len(index.en_control) == len(SECTION_IDS)
    for section_id in SECTION_IDS:
        control = index.en_control[section_id]
        assert control["role"] == "reference-control"
        assert control["title"].startswith("EN TITLE")
    hits = search_aspect(index, ["жена ругает из-за пьянки"])
    assert hits
    assert any(hit.section == "chapter-8" for hit in hits[:4])
    for hit in hits:
        assert hit.en_control_title.startswith("EN TITLE")
        assert hit.en_control_section == hit.section


def test_planner_validation_and_original_preserved() -> None:
    plan = validate_plan(_valid_plan("я бухаю"), original_query="я бухаю")
    assert plan.original_query == "я бухаю"
    queries = aspect_search_queries(plan.aspects[0], original_query=plan.original_query)
    assert queries[0] == "я бухаю"
    assert "пить алкоголь" in queries
    bad = _valid_plan("я бухаю")
    bad["aspects"][0]["lexical_query_en"] = "drinking alcohol"
    try:
        validate_plan(bad, original_query="я бухаю")
    except PlannerError:
        pass
    else:
        raise AssertionError("EN lexical branch must stay null in the RU-first baseline")
    empty = dict(_valid_plan("я бухаю"))
    empty["aspects"] = []
    try:
        validate_plan(empty, original_query="я бухаю")
    except PlannerError:
        pass
    else:
        raise AssertionError("empty aspects must fail closed")


def test_search_plan_covers_every_aspect(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    data = _valid_plan("жинка ругает из-за пьянки")
    data["aspects"].append(
        {
            "aspect_id": "a2",
            "meaning": "family conflict about drinking",
            "semantic_queries_ru": ["конфликт в семье из-за алкоголя"],
            "lexical_queries_ru": ["жена ругает пьянка"],
            "lexical_query_en": None,
            "ambiguity": "low",
            "forbidden_inferences": ["do_not_assume_breakdown"],
        }
    )
    plan = validate_plan(data, original_query="жинка ругает из-за пьянки")
    results = search_plan(index, plan)
    assert set(results) == {"a1", "a2"}
    for hits in results.values():
        assert len(hits) <= MAX_CANDIDATES_PER_ASPECT


def test_ordinary_turns_reuse_index_without_rebuild(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    before = {path.name: path.stat().st_mtime_ns for path in index.directory.iterdir()}
    first = search_aspect(index, ["трезвость"])
    second = search_aspect(index, ["трезвость"])
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]
    after = {path.name: path.stat().st_mtime_ns for path in index.directory.iterdir()}
    assert before == after


def test_no_remote_search_or_embedding_service() -> None:
    root = _repo_root() / "src" / "aa" / "retrieval"
    forbidden = (
        "import openai",
        "from openai",
        "import anthropic",
        "from anthropic",
        "import httpx",
        "from httpx",
        "import aiohttp",
        "from aiohttp",
        "api.openai.com",
        "api.anthropic.com",
    )
    for path in sorted(root.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for snippet in forbidden:
            assert snippet not in text, f"{path.name}: remote service coupling {snippet!r}"
    dense_text = (root / "dense.py").read_text(encoding="utf-8")
    assert "local_files_only" in dense_text
    assert "HF_HUB_OFFLINE" in dense_text


def test_e5_embed_fails_closed_without_pinned_snapshot(tmp_path: pathlib.Path) -> None:
    from aa.retrieval.dense import DenseError, e5_embed

    bogus_lock = tmp_path / "embedding.lock.json"
    bogus_lock.write_text(json.dumps({"format": "aa-public-embedding-lock/1"}), encoding="utf-8")
    try:
        e5_embed(["алкоголь"], model_dir=str(bogus_lock))
    except DenseError:
        return
    raise AssertionError("e5 without a valid pinned snapshot must fail closed")


def test_l2_normalize_rejects_zero_vector() -> None:
    try:
        l2_normalize([0.0, 0.0])
    except ValueError:
        return
    raise AssertionError("zero vector must fail closed")


def test_lexical_search_rejects_bad_top_k(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    try:
        lexical_search(index.directory / "lexical.db", "алкоголь", top_k=0)
    except ValueError:
        return
    raise AssertionError("top_k=0 must fail closed")


def test_builder_script_builds_runtime_index(tmp_path: pathlib.Path) -> None:
    fixture = _fixture_full()
    full_path = tmp_path / "corpus_structure.json"
    ru_manifest_path = tmp_path / "ru-manifest.json"
    en_manifest_path = tmp_path / "en-manifest.json"
    lock_path = tmp_path / "embedding.lock.json"
    out_dir = tmp_path / "retrieval"
    ru_manifest, en_manifest, lock = _fixture_manifests()
    full_path.write_text(json.dumps(fixture["full"], ensure_ascii=False), encoding="utf-8")
    ru_manifest_path.write_text(json.dumps(ru_manifest), encoding="utf-8")
    en_manifest_path.write_text(json.dumps(en_manifest), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    script = _repo_root() / "scripts" / "build_retrieval_index.py"
    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--full-structure",
            str(full_path),
            "--ru-manifest",
            str(ru_manifest_path),
            "--en-manifest",
            str(en_manifest_path),
            "--embedding-lock",
            str(lock_path),
            "--out-dir",
            str(out_dir),
            "--backend",
            "hashing",
        ],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "index.json").is_file()
    assert (out_dir / "lexical.db").is_file()
    assert (out_dir / "dense.json").is_file()
    opened = open_hybrid_index(out_dir)
    assert opened.chunk_count > 0


def test_generated_index_dir_stays_gitignored() -> None:
    gitignore = (_repo_root() / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/generated/" in gitignore
