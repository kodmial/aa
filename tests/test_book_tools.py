"""Narrow RU-first OpenCode book tools tests (issue #18).

All literary content uses invented fixture sentences; no canonical book
text is committed. Covers RU search/read/expand/section, slang-planner
queries, the disabled EN secondary branch, invalid IDs/stale versions,
exact RU fidelity, output limits, deny-by-default permissions and log
privacy over the single shared implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pathlib
from typing import Any

import pytest

from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.retrieval import book_tools
from aa.retrieval.book_tools import (
    BookNotFoundError,
    BookStaleError,
    BookToolError,
    book_expand,
    book_read,
    book_search,
    book_section,
    validate_search_input,
)
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT
from aa.retrieval.index import HybridIndex, build_hybrid_index

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / ".opencode" / "tools"
CONFIG_PATH = ROOT / "opencode.json"
BRIDGE_PATH = ROOT / "scripts" / "aa_book_tool.py"


def _repo_lock() -> dict[str, Any]:
    payload = json.loads((ROOT / "corpus" / "embedding.lock.json").read_text())
    assert isinstance(payload, dict)
    return payload


def _fixture_full() -> dict[str, Any]:
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": (
                    f"Fixture EN {section_id} opening sentence. Second sentence here. "
                    f"Third sentence follows. Fourth sentence closes the part. "
                    f"Fifth sentence adds control text."
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
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        max_chars=120,
    )
    return full


def _build_index(tmp_path: pathlib.Path) -> HybridIndex:
    full = _fixture_full()
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
    return build_hybrid_index(
        full,
        ru_manifest=ru_manifest,
        en_manifest=en_manifest,
        embedding_lock=_repo_lock(),
        out_dir=tmp_path / "retrieval",
        backend="hashing",
    )


def _search_payload(
    semantic: str = "тяга к алкоголю",
    lexical: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "aspect_id": "craving",
        "semantic_query_ru": semantic,
        "lexical_queries_ru": lexical or ["пить алкоголь", "тяга выпить"],
        "lexical_query_en": None,
    }


def test_search_returns_compact_navigation_candidates(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    result = book_search(index, _search_payload())
    assert result["tool"] == "book_search"
    assert result["aspect_id"] == "craving"
    candidates = result["candidates"]
    assert 0 < len(candidates) <= MAX_CANDIDATES_PER_ASPECT == 12
    for candidate in candidates:
        assert ":ru:" not in candidate["logical_chunk_id"]
        assert "ru_locator" in candidate and "preview" in candidate
        assert "never evidence" in candidate["preview"]
        assert "text" not in candidate
        locator = candidate["ru_locator"]
        for key in ("source_id", "source_file", "char_start", "char_end", "text_sha256"):
            assert key in locator
        assert candidate["en_control"]["role"] == "reference-control"


def test_search_slang_planner_queries(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    for query in (
        "я бухаю каждый вечер",
        "жинка ругает из-за пьянки",
        "женушка переживает за семью",
        "рассказ об алкоголизме",
        "алкаголь",
        "программа в действии требует честности",
    ):
        result = book_search(
            index,
            _search_payload(semantic=query, lexical=["пить алкоголь", "тяга выпить"]),
        )
        assert result["candidates"], f"slang query must hit: {query!r}"


def test_search_rejects_en_secondary_branch(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    payload = _search_payload()
    payload["lexical_query_en"] = "drinking alcohol"
    with pytest.raises(BookToolError, match="lexical_query_en must stay null"):
        book_search(index, payload)
    with pytest.raises(BookToolError, match="lexical_query_en must stay null"):
        validate_search_input(payload)


def test_search_rejects_ranking_overrides(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    for extra in ("top_k", "topK", "fusion", "rrf_k", "weights", "index_name"):
        payload = _search_payload()
        payload[extra] = 5
        with pytest.raises(BookToolError, match="server-side parameter"):
            book_search(index, payload)


def test_search_rejects_empty_and_oversize_input(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    empty_aspect = _search_payload()
    empty_aspect["aspect_id"] = "   "
    with pytest.raises(BookToolError):
        book_search(index, empty_aspect)
    with pytest.raises(BookToolError):
        book_search(index, "not-an-object")
    bad = _search_payload(semantic="x" * 501)
    with pytest.raises(BookToolError, match="exceeds"):
        book_search(index, bad)


def test_read_exact_ru_fidelity_logical_and_physical(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    result = book_search(index, _search_payload())
    candidate = result["candidates"][0]
    for wanted in (candidate["logical_chunk_id"], candidate["chunk_id"]):
        read = book_read(index, wanted)
        assert read["tool"] == "book_read"
        assert read["logical_chunk_id"] == candidate["logical_chunk_id"]
        assert ":ru:" in read["chunk_id"] and ":en:" not in read["chunk_id"]
        assert read["text"]
        assert (
            hashlib.sha256(read["text"].encode("utf-8")).hexdigest()
            == read["ru_locator"]["text_sha256"]
        )
        assert read["ru_locator"]["source_id"] == "ru-fourth-edition-txt"
        assert read["versions"]["ru_corpus_version"] == "r" * 64


def test_read_invalid_id_fails_closed(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    with pytest.raises(BookNotFoundError):
        book_read(index, "chapter-99:c9999")
    with pytest.raises(BookToolError):
        book_read(index, "   ")


def test_read_stale_version_fails_closed(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    logical = next(iter(index.chunks.values())).logical_chunk_id
    with pytest.raises(BookStaleError):
        book_read(index, logical, expected_ru_version="0" * 64)
    live = str(index.metadata.get("ru_artifact_sha256", ""))
    read = book_read(index, logical, expected_ru_version=live)
    assert read["logical_chunk_id"] == logical


def test_expand_bounded_neighbors_and_hard_max(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    ordered = sorted(index.chunks.values(), key=lambda r: (r.section, r.char_start))
    by_section: dict[str, list[str]] = {}
    for record in ordered:
        by_section.setdefault(record.section, []).append(record.logical_chunk_id)
    section = next(key for key, ids in by_section.items() if len(ids) >= 2)
    middle = by_section[section][1]
    result = book_expand(index, middle, before=1, after=1)
    assert result["tool"] == "book_expand"
    assert result["center"] == middle
    assert result["section"] == section
    texts = [item["text"] for item in result["chunks"]]
    assert len(texts) == len(set(texts)) or len(texts) >= 2
    for item in result["chunks"]:
        assert ":ru:" in item["chunk_id"]
        assert (
            hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
            == item["ru_locator"]["text_sha256"]
        )
    with pytest.raises(BookToolError, match="within 0..3"):
        book_expand(index, middle, before=4, after=0)
    with pytest.raises(BookToolError, match="within 0..3"):
        book_expand(index, middle, before=0, after=4)
    with pytest.raises(BookNotFoundError):
        book_expand(index, "no-such-section:c0001")


def test_section_bounded_paginated_under_token_ceiling(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    first = book_section(index, "chapter-1", chunk_offset=0, chunk_limit=2)
    assert first["tool"] == "book_section"
    assert first["section"] == "chapter-1"
    assert first["total_chunks"] >= 2
    assert len(first["chunks"]) == 2
    assert first["source_tokens"] <= RETRIEVED_PASSAGES_BUDGET_TOKENS
    for item in first["chunks"]:
        assert ":ru:" in item["chunk_id"]
        assert (
            hashlib.sha256(item["text"].encode("utf-8")).hexdigest()
            == item["ru_locator"]["text_sha256"]
        )
    if first["next_chunk_offset"] is not None:
        second = book_section(
            index, "chapter-1", chunk_offset=first["next_chunk_offset"], chunk_limit=2
        )
        first_ids = {item["logical_chunk_id"] for item in first["chunks"]}
        second_ids = {item["logical_chunk_id"] for item in second["chunks"]}
        assert not first_ids & second_ids


def test_section_rejects_invalid_and_over_limit(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    with pytest.raises(BookNotFoundError):
        book_section(index, "chapter-99")
    with pytest.raises(BookToolError):
        book_section(index, "chapter-1", chunk_offset=-1, chunk_limit=2)
    with pytest.raises(BookToolError, match="within 1..12"):
        book_section(index, "chapter-1", chunk_offset=0, chunk_limit=13)
    with pytest.raises(BookToolError):
        book_section(index, "chapter-1", chunk_offset=10_000, chunk_limit=2)


def test_output_limits_stay_compact(tmp_path: pathlib.Path) -> None:
    index = _build_index(tmp_path)
    result = book_search(index, _search_payload())
    assert len(result["candidates"]) <= 12
    assert set(result["ranking"]) == {
        "lexical_top_k",
        "dense_top_k",
        "rrf_k",
        "max_per_aspect",
    }
    assert result["en_secondary"] == {"enabled": False, "role": "reference-control"}
    assert len(json.dumps(result, ensure_ascii=False)) < 100_000


def test_four_tools_share_one_implementation() -> None:
    assert book_tools.__name__ == "aa.retrieval.book_tools"
    assert book_search.__module__ == book_tools.__name__
    assert book_read.__module__ == book_tools.__name__
    assert book_expand.__module__ == book_tools.__name__
    assert book_section.__module__ == book_tools.__name__
    assert set(book_tools.TOOL_NAMES) == {
        "book_search",
        "book_read",
        "book_expand",
        "book_section",
    }
    assert book_tools.EN_SECONDARY_ENABLED is False
    assert BRIDGE_PATH.is_file()
    bridge = BRIDGE_PATH.read_text(encoding="utf-8")
    for name in ("book_search", "book_read", "book_expand", "book_section"):
        assert name in bridge


def test_deny_by_default_permissions() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    permission = config["agent"]["aa"]["permission"]
    assert permission.get("*") == "deny"
    allowed = {name for name, value in permission.items() if value == "allow"}
    assert allowed == {"book_search", "book_read", "book_expand", "book_section"}


def test_tool_wrappers_are_narrow_and_share_bridge() -> None:
    for name in ("book_search", "book_read", "book_expand", "book_section"):
        wrapper = TOOLS_DIR / f"{name}.ts"
        assert wrapper.is_file(), f"missing OpenCode tool wrapper: {wrapper}"
        text = wrapper.read_text(encoding="utf-8")
        assert "aa_book_tool.py" in text
        for forbidden in ("top_k", "topK", "rrf", "fusion", "weight", "index_name"):
            assert forbidden not in text.lower(), f"{name}: wrapper leaks {forbidden!r}"
    search_text = (TOOLS_DIR / "book_search.ts").read_text(encoding="utf-8")
    assert "semantic_query_ru" in search_text
    assert "lexical_queries_ru" in search_text
    assert "lexical_query_en" in search_text
    assert "book_read" in (TOOLS_DIR / "book_read.ts").read_text(encoding="utf-8")
    section_text = (TOOLS_DIR / "book_section.ts").read_text(encoding="utf-8")
    assert "section_id" in section_text and "chunk_limit" in section_text
    expand_text = (TOOLS_DIR / "book_expand.ts").read_text(encoding="utf-8")
    assert "before" in expand_text and "after" in expand_text


def test_no_raw_user_or_corpus_leakage_in_logs(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    index = _build_index(tmp_path)
    secret_query = "я бухаю каждый вечер, секретное слово ЖУК"
    target = logging.getLogger("aa.retrieval.book_tools")
    target.setLevel(logging.INFO)
    with caplog.at_level(logging.INFO, logger="aa.retrieval.book_tools"):
        result = book_search(
            index,
            _search_payload(semantic=secret_query, lexical=["пить алкоголь"]),
        )
        logical = result["candidates"][0]["logical_chunk_id"]
        read = book_read(index, logical)
    output = caplog.text
    assert secret_query not in output
    assert "ЖУК" not in output
    assert read["text"] not in output
    assert logical in output or "book_read" in output


def test_bridge_dispatches_all_four_tools(tmp_path: pathlib.Path) -> None:
    import subprocess
    import sys

    index = _build_index(tmp_path)
    out_dir = str(index.directory)
    logical = next(iter(index.chunks.values())).logical_chunk_id
    section = next(iter(index.chunks.values())).section
    calls: list[list[str]] = [
        [
            sys.executable,
            str(BRIDGE_PATH),
            "book_search",
            "--index-dir",
            out_dir,
            "--input-json",
            json.dumps(_search_payload(), ensure_ascii=False),
        ],
        [
            sys.executable,
            str(BRIDGE_PATH),
            "book_read",
            "--index-dir",
            out_dir,
            "--input-json",
            json.dumps({"chunk_id": logical}, ensure_ascii=False),
        ],
        [
            sys.executable,
            str(BRIDGE_PATH),
            "book_expand",
            "--index-dir",
            out_dir,
            "--input-json",
            json.dumps({"chunk_id": logical, "before": 1, "after": 1}, ensure_ascii=False),
        ],
        [
            sys.executable,
            str(BRIDGE_PATH),
            "book_section",
            "--index-dir",
            out_dir,
            "--input-json",
            json.dumps(
                {"section_id": section, "chunk_offset": 0, "chunk_limit": 2},
                ensure_ascii=False,
            ),
        ],
    ]
    for argv in calls:
        proc = subprocess.run(argv, capture_output=True, text=True, check=False)
        assert proc.returncode == 0, proc.stderr
        payload = json.loads(proc.stdout)
        assert payload["tool"] == argv[2]
