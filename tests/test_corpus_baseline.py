"""Canonical AA source baseline, context-budget and corpus-loading tests.

Uses synthetic placeholder text with the required canonical headings only;
no AA book text is committed. Validates deterministic loading, boundaries,
checksums, budget enforcement and privacy-safe observability.
"""

from __future__ import annotations

import pathlib

import pytest

from aa.corpus.budget import (
    MIN_EFFECTIVE_CONTEXT_TOKENS,
    ContextBudgetError,
    compact_conversation_tail,
    validate_context_budget,
)
from aa.corpus.canonical import (
    REQUIRED_SECTIONS,
    estimate_tokens,
    load_canonical_corpus,
)
from aa.corpus.context import CorpusContext
from aa.opencode.runtime import OpenCodeConfig

TITLES_IN_ORDER = [
    "The Doctor's Opinion",
    "Chapter 1 — Bill's Story",
    "Chapter 2 — There Is a Solution",
    "Chapter 3 — More About Alcoholism",
    "Chapter 4 — We Agnostics",
    "Chapter 5 — How It Works",
    "Chapter 6 — Into Action",
    "Chapter 7 — Working With Others",
    "Chapter 8 — To Wives",
    "Chapter 9 — The Family Afterward",
    "Chapter 10 — To Employers",
    "Chapter 11 — A Vision for You",
]


def _write_canonical_dir(
    tmp_path: pathlib.Path, *, titles: list[str] | None = None, extra: str = ""
) -> pathlib.Path:
    titles = titles if titles is not None else TITLES_IN_ORDER
    parts = []
    for title in titles:
        parts.append(f"# {title}\n\nPlaceholder recovery guidance for {title}.\n")
    if extra:
        parts.append(extra)
    (tmp_path / "corpus.md").write_text("\n".join(parts), encoding="utf-8")
    return tmp_path / "corpus.md"


def test_required_section_inventory_is_exactly_twelve() -> None:
    assert len(REQUIRED_SECTIONS) == 12
    ids = [section.section_id for section in REQUIRED_SECTIONS]
    assert ids == [
        "doctors-opinion",
        "chapter-01",
        "chapter-02",
        "chapter-03",
        "chapter-04",
        "chapter-05",
        "chapter-06",
        "chapter-07",
        "chapter-08",
        "chapter-09",
        "chapter-10",
        "chapter-11",
    ]


def test_corpus_loads_deterministically_with_chapter_boundaries(tmp_path: pathlib.Path) -> None:
    source = _write_canonical_dir(tmp_path)
    first = load_canonical_corpus(source, source_version="test-v1")
    second = load_canonical_corpus(source, source_version="test-v1")
    assert first.checksum_sha256 == second.checksum_sha256
    assert len(first.sections) == 12
    assert [s.section_id for s in first.sections] == [s.section_id for s in REQUIRED_SECTIONS]
    # Boundaries are ordered, non-overlapping and cover section bodies.
    for section in first.sections:
        assert section.start_offset < section.end_offset
        assert section.token_estimate > 0
        assert section.title in section.text
    assert first.token_estimate == sum(s.token_estimate for s in first.sections)


def test_required_sections_present_exactly_once(tmp_path: pathlib.Path) -> None:
    source = _write_canonical_dir(tmp_path)
    corpus = load_canonical_corpus(source)
    assert corpus.section_ids().count("chapter-05") == 1


def test_missing_section_fails_closed(tmp_path: pathlib.Path) -> None:
    titles = [t for t in TITLES_IN_ORDER if "We Agnostics" not in t]
    source = _write_canonical_dir(tmp_path, titles=titles)
    with pytest.raises(ValueError, match="We Agnostics"):
        load_canonical_corpus(source)


def test_duplicated_section_fails_closed(tmp_path: pathlib.Path) -> None:
    source = _write_canonical_dir(tmp_path, titles=TITLES_IN_ORDER + ["Chapter 5 — How It Works"])
    with pytest.raises(ValueError, match="[Dd]uplicated"):
        load_canonical_corpus(source)


def test_excluded_sections_are_absent_and_rejected(tmp_path: pathlib.Path) -> None:
    for excluded in (
        "# Foreword to First Edition\n\nPlaceholder.\n",
        "# Personal Stories\n\nPlaceholder.\n",
        "# Appendix I — The Spiritual Experience\n\nPlaceholder.\n",
    ):
        target = tmp_path / "corpus.md"
        _write_canonical_dir(tmp_path, extra=excluded)
        with pytest.raises(ValueError, match="[Ee]xcluded"):
            load_canonical_corpus(target)


def test_checksum_version_token_count_exposed_without_contents(
    tmp_path: pathlib.Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    source = _write_canonical_dir(tmp_path)
    corpus = load_canonical_corpus(source, source_version="test-v1")
    safe = corpus.to_safe_dict()
    assert safe["source_version"] == "test-v1"
    assert len(str(safe["checksum_sha256"])) == 64
    assert safe["token_estimate"] == corpus.token_estimate
    assert safe["section_ids"] == list(corpus.section_ids())
    # Metadata contains no corpus body text.
    with caplog.at_level(logging.INFO):
        logging.getLogger("aa.test").info("corpus loaded %s", safe)
    assert "Placeholder recovery guidance" not in caplog.text


def test_full_corpus_fits_configured_200k_context_without_truncation(
    tmp_path: pathlib.Path,
) -> None:
    source = _write_canonical_dir(tmp_path)
    corpus = load_canonical_corpus(source)
    budget = validate_context_budget(
        effective_context_tokens=MIN_EFFECTIVE_CONTEXT_TOKENS,
        corpus_tokens=corpus.token_estimate,
    )
    assert budget.fits()
    assert budget.remaining_tokens >= 0
    # Larger realistic size still fits: scale the small fixture 10x.
    scaled = corpus.token_estimate * 10
    assert validate_context_budget(effective_context_tokens=200_000, corpus_tokens=scaled).fits()


def test_context_budget_rejects_below_minimum_and_overflow() -> None:
    with pytest.raises(ContextBudgetError, match="below the minimum"):
        validate_context_budget(effective_context_tokens=100_000, corpus_tokens=1000)
    with pytest.raises(ContextBudgetError, match="refusing to truncate"):
        validate_context_budget(
            effective_context_tokens=200_000,
            corpus_tokens=200_000,
        )


def test_opencode_config_enforces_200k_minimum() -> None:
    OpenCodeConfig(
        base_url="http://127.0.0.1:4096", command="opencode", workdir="."
    ).validate_context_contract()
    with pytest.raises(ValueError, match="below the required minimum"):
        OpenCodeConfig(
            base_url="http://127.0.0.1:4096",
            command="opencode",
            workdir=".",
            context_limit_tokens=32_000,
        ).validate_context_contract()


async def test_corpus_context_loads_validated_and_reports_safely(tmp_path: pathlib.Path) -> None:
    source = _write_canonical_dir(tmp_path)
    context = CorpusContext(
        path=str(source),
        version="test-v1",
        effective_context_tokens=MIN_EFFECTIVE_CONTEXT_TOKENS,
    )
    info = await context.load()
    assert info.checksum_sha256
    assert info.token_estimate > 0
    assert info.section_count == 12
    assert context.validated
    assert context.budget is not None and context.budget.fits()
    safe = context.to_safe_dict()
    assert safe["checksum_sha256"] == info.checksum_sha256
    assert "Placeholder recovery guidance" not in str(safe)
    validated = context.ensure_validated()
    assert len(validated.sections) == 12
    await context.unload()
    assert not context.loaded


async def test_corpus_context_pointer_mode_without_source(tmp_path: pathlib.Path) -> None:
    context = CorpusContext(path=str(tmp_path / "missing"), version="local")
    await context.load()
    assert context.loaded
    assert not context.validated
    with pytest.raises(ValueError, match="not validated"):
        context.ensure_validated()


async def test_corpus_context_fails_closed_on_budget_overflow(tmp_path: pathlib.Path) -> None:
    source = _write_canonical_dir(tmp_path)
    context = CorpusContext(path=str(source), version="test-v1", effective_context_tokens=100_000)
    with pytest.raises(ContextBudgetError):
        await context.load()


def test_compact_conversation_keeps_newest_never_corpus() -> None:
    messages = ["old one", "middle two", "latest three"]
    kept = compact_conversation_tail(
        messages, max_conversation_tokens=estimate_tokens("latest three")
    )
    assert kept == ["latest three"]
    kept_all = compact_conversation_tail(messages, max_conversation_tokens=10**9)
    assert kept_all == messages
    assert compact_conversation_tail(["x" * 10**6], max_conversation_tokens=10) == []
