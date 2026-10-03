"""Canonical build contract tests (offline fixtures, no network)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
from typing import Any, Protocol, cast

import pytest

from aa.corpus.canonical import (
    CanonicalCorpusError,
    CanonicalRangeError,
    load_canonical,
)

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


class _BuilderProtocol(Protocol):
    """Typed surface of ``scripts/build_canonical.py`` used by these tests."""

    def sha256(self, data: bytes) -> str: ...
    def extract_doctors_opinion(self, html_bytes: bytes) -> tuple[str, list[str]]: ...
    def slice_chapters(self, source: bytes) -> list[dict[str, Any]]: ...
    def build_canonical(
        self,
        *,
        manifest: dict[str, Any],
        fetch_state: dict[str, Any],
        aa_bytes: bytes,
        opinion_html: bytes,
    ) -> dict[str, Any]: ...
    def serialize_artifact(self, artifact: dict[str, Any]) -> bytes: ...


def _load_builder() -> _BuilderProtocol:
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_canonical.py"
    spec = importlib.util.spec_from_file_location("build_canonical_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_BuilderProtocol, module)


def _fixture_aa_bytes() -> bytes:
    preamble = (
        "The text of the fixture book.\r\n\r\nProvided as fixture software\r\n\r\n"
        "by The Anonymous Press\r\n\r\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\r\n"
    )
    parts = [preamble]
    for number, title in enumerate(CHAPTER_TITLES, start=1):
        body = (
            f"Chapter {number}\r\n    \r\n    {title}\r\n    \r\n"
            f"Fixture body sentence one of chapter {number}. "
            f"Fixture body sentence two of chapter {number}.  "
        )
        if number == 11:
            body += "The road ends here, until then.\n"
        parts.append(body)
    return "".join(parts).encode("utf-8")


def _fixture_opinion_html() -> bytes:
    paragraphs = "".join(
        f'<p id="p{index}" class="mb-4">Fixture paragraph {index} '
        f"with wording intact &#x27;quoted&#x27;.</p>"
        for index in range(1, 42)
    )
    document = (
        "<!DOCTYPE html><html><head><title>Fixture</title></head>"
        f"<body><nav>site chrome</nav><article><h1>The Doctor&#x27;s Opinion</h1>"
        f"{paragraphs}</article><footer>site footer</footer></body></html>"
    )
    return document.encode("utf-8")


def _fixture_manifest(builder: _BuilderProtocol, aa: bytes, opinion: bytes) -> dict[str, Any]:
    sha256 = builder.sha256
    chapters = builder.slice_chapters(aa)
    opinion_text, _ = builder.extract_doctors_opinion(opinion)
    sections: list[dict[str, Any]] = [
        {
            "id": "doctors-opinion",
            "title": "The Doctor's Opinion",
            "source_id": "doctors-opinion",
            "paragraphs": [f"p{i}" for i in range(1, 42)],
            "text_sha256": sha256(opinion_text.encode("utf-8")),
        }
    ]
    for chapter in chapters:
        text = str(chapter["text"])
        sections.append(
            {
                "id": f"chapter-{int(chapter['number'])}",
                "title": str(chapter["title"]),
                "source_id": "core-pages-1-164",
                "byte_start": int(chapter["byte_start"]),
                "byte_end": int(chapter["byte_end"]),
                "text_sha256": sha256(text.encode("utf-8")),
            }
        )
    return {
        "builder_version": 1,
        "sources": [
            {
                "id": "core-pages-1-164",
                "url": "https://example.invalid/AA.txt",
                "raw_path": "corpus/source/raw/AA.txt",
                "sha256": sha256(aa),
                "bytes": len(aa),
            },
            {
                "id": "doctors-opinion",
                "url": "https://example.invalid/doctors-opinion",
                "raw_path": "corpus/source/raw/doctors-opinion.html",
                "sha256": sha256(opinion),
                "bytes": len(opinion),
            },
        ],
        "sections": sections,
    }


def _fixture_fetch_state(builder: _BuilderProtocol, aa: bytes, opinion: bytes) -> dict[str, Any]:
    sha256 = builder.sha256
    return {
        "version": 1,
        "sources": [
            {
                "id": "core-pages-1-164",
                "url": "https://example.invalid/AA.txt",
                "path": "corpus/source/raw/AA.txt",
                "bytes": len(aa),
                "sha256": sha256(aa),
            },
            {
                "id": "doctors-opinion",
                "url": "https://example.invalid/doctors-opinion",
                "path": "corpus/source/raw/doctors-opinion.html",
                "bytes": len(opinion),
                "sha256": sha256(opinion),
            },
        ],
    }


def _build_fixture(builder: _BuilderProtocol) -> tuple[dict[str, Any], bytes, bytes]:
    aa = _fixture_aa_bytes()
    opinion = _fixture_opinion_html()
    manifest = _fixture_manifest(builder, aa, opinion)
    fetch_state = _fixture_fetch_state(builder, aa, opinion)
    artifact = builder.build_canonical(
        manifest=manifest, fetch_state=fetch_state, aa_bytes=aa, opinion_html=opinion
    )
    return artifact, aa, opinion


def test_build_fixture_has_exact_scope() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    assert [section["id"] for section in artifact["sections"]] == list(EXPECTED_IDS)
    titles = [section["title"] for section in artifact["sections"]]
    assert titles[0] == "The Doctor's Opinion"
    assert titles[1:] == list(CHAPTER_TITLES)
    for section in artifact["sections"]:
        assert section["source_url"].startswith("https://example.invalid/")
        assert section["source_file"].startswith("corpus/source/raw/")
        assert section["text_sha256"] == hashlib.sha256(section["text"].encode()).hexdigest()


def test_build_is_deterministic() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    serialize = builder.serialize_artifact
    first = serialize(artifact)
    artifact_again, _, _ = _build_fixture(builder)
    assert serialize(artifact_again) == first


def test_preamble_excluded_and_chapters_contiguous() -> None:
    builder = _load_builder()
    artifact, aa, _ = _build_fixture(builder)
    chapter_sections = [s for s in artifact["sections"] if s["id"] != "doctors-opinion"]
    full = "".join(str(section["text"]) for section in chapter_sections)
    assert "Anonymous Press" not in full
    assert "fixture software" not in full
    assert full == aa[chapter_sections[0]["byte_start"] :].decode("utf-8")
    for previous, current in zip(chapter_sections, chapter_sections[1:], strict=False):
        assert current["byte_start"] == previous["byte_end"]


def test_doctors_opinion_wording_preserved() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    text = str(artifact["sections"][0]["text"])
    assert text.startswith("The Doctor's Opinion\n\n")
    # HTML entities are unescaped; the wording keeps its quotes/apostrophes.
    assert "'quoted'" in text
    assert len(text.split("\n\n")) == 42  # title block + 41 paragraphs
    assert "site chrome" not in text and "site footer" not in text


def test_stale_source_fails_closed() -> None:
    builder = _load_builder()
    aa = _fixture_aa_bytes()
    opinion = _fixture_opinion_html()
    manifest = _fixture_manifest(builder, aa, opinion)
    fetch_state = _fixture_fetch_state(builder, aa, opinion)
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_canonical(
            manifest=manifest, fetch_state=fetch_state, aa_bytes=aa + b"X", opinion_html=opinion
        )
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_canonical(
            manifest=manifest, fetch_state=fetch_state, aa_bytes=aa, opinion_html=opinion + b"X"
        )


def test_missing_chapter_fails_closed() -> None:
    builder = _load_builder()
    aa = _fixture_aa_bytes()
    broken = aa.replace(b"Chapter 6\r\n", b"Section 6\r\n")
    with pytest.raises(ValueError, match="not exactly 1..11"):
        builder.slice_chapters(broken)


def test_reordered_opinion_paragraphs_fail_closed() -> None:
    builder = _load_builder()
    opinion = _fixture_opinion_html().decode("utf-8")
    swapped = (
        opinion.replace('id="p1"', 'id="pX"')
        .replace('id="p2"', 'id="p1"')
        .replace('id="pX"', 'id="p2"')
    )
    with pytest.raises(ValueError, match="not exactly p1..p41"):
        builder.extract_doctors_opinion(swapped.encode("utf-8"))


def test_load_canonical_round_trip_and_exact_reads(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    payload = builder.serialize_artifact(artifact)
    target = tmp_path / "canonical.json"
    target.write_bytes(payload)

    corpus = load_canonical(target, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert corpus.ids() == EXPECTED_IDS
    first = corpus.get("chapter-1")
    assert corpus.read("chapter-1", 0, len(first.text)) == first.text
    assert corpus.read("chapter-1", 0, 7) == first.text[0:7]
    with pytest.raises(CanonicalRangeError):
        corpus.read("chapter-1", 0, len(first.text) + 1)
    with pytest.raises(CanonicalRangeError):
        corpus.read("chapter-1", 5, 5)
    with pytest.raises(CanonicalCorpusError):
        corpus.get("chapter-12")


def test_load_canonical_rejects_tampering(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    payload = builder.serialize_artifact(artifact)
    target = tmp_path / "canonical.json"
    target.write_bytes(payload)
    with pytest.raises(CanonicalCorpusError, match="checksum mismatch"):
        load_canonical(target, expected_sha256="0" * 64)
    tampered = json.loads(payload.decode("utf-8"))
    tampered["sections"][1]["text"] += " forged"
    target.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(CanonicalCorpusError, match="section checksum mismatch"):
        load_canonical(target)


def test_committed_manifest_carries_no_book_text() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = root / "corpus" / "canonical.manifest.json"
    assert manifest_path.exists()
    text = manifest_path.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) < 20_000
    for snippet in ("WAR FEVER", "Fixture body sentence", "7th Tradition"):
        assert snippet not in text
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/generated/" in gitignore
    assert "/corpus/source/raw/" in gitignore


def test_real_manifest_matches_expected_scope() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "corpus" / "canonical.manifest.json").read_text())
    assert [s["id"] for s in manifest["sections"]] == list(EXPECTED_IDS)
    assert re.search(r"^[0-9a-f]{64}$", manifest["artifact_sha256"]) is not None
