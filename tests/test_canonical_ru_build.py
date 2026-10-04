"""Russian canonical build contract tests (issue #50; offline, no network).

All fixtures are short synthetic Russian sentences written for these tests.
No literary text from the Big Book appears here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
from typing import Any, Protocol, cast

import pytest

from aa.corpus.canonical import (
    CanonicalCorpusError,
    CanonicalRangeError,
    load_canonical_ru,
)

SECTION_SPECS = (
    ("doctors-opinion", "nXXVII", "Мнение доктора", "XXVII", "МНЕНИЕ ДОКТОРА", None),
    ("chapter-1", "n1", "Глава 1. Рассказ Билла", "1", "ГЛАВА 1", "РАССКАЗ БИЛЛА"),
    ("chapter-2", "n16", "Глава 2. Выход есть", "16", "ГЛАВА 2", "ВЫХОД ЕСТЬ"),
    ("chapter-3", "n29", "Глава 3. Еще об алкоголизме", "29", "ГЛАВА 3", "ЕЩЕ ОБ АЛКОГОЛИЗМЕ"),
    (
        "chapter-4",
        "n43",
        "Глава 4. А как быть агностикам?",
        "43",
        "ГЛАВА 4",
        "А КАК БЫТЬ АГНОСТИКАМ?",
    ),
    ("chapter-5", "n56", "Глава 5. Программа в действии", "56", "ГЛАВА 5", "ПРОГРАММА В ДЕЙСТВИИ"),
    ("chapter-6", "n70", "Глава 6. За работу!", "70", "ГЛАВА 6", "ЗА РАБОТУ!"),
    ("chapter-7", "n86", "Глава 7. Работая с другими", "86", "ГЛАВА 7", "РАБОТАЯ С ДРУГИМИ"),
    ("chapter-8", "n101", "Глава 8. Обращение к женам", "101", "ГЛАВА 8", "ОБРАЩЕНИЕ К ЖЕНАМ*"),
    (
        "chapter-9",
        "n118",
        "Глава 9. Новые отношения в семье",
        "118",
        "ГЛАВА 9",
        "НОВЫЕ ОТНОШЕНИЯ В СЕМЬЕ",
    ),
    (
        "chapter-10",
        "n132",
        "Глава 10. Обращение к работодателям",
        "132",
        "ГЛАВА 10",
        "ОБРАЩЕНИЕ К РАБОТОДАТЕЛЯМ",
    ),
    (
        "chapter-11",
        "n147",
        "Глава 11. Заглянем в ваше будущее",
        "147",
        "ГЛАВА 11",
        "ЗАГЛЯНЕМ В ВАШЕ БУДУЩЕЕ",
    ),
)

EXPECTED_IDS = tuple(spec[0] for spec in SECTION_SPECS)

EDITION_HTML = """<html><head><title>edition</title></head><body>
<p>АНОНИМНЫЕ АЛКОГОЛИКИ</p><p>4-е издание Перевод с английского</p>
<p>Alcoholics Anonymous World Services, Inc.</p>
<p>Фонд «Единство», 2013</p><p>ISBN 978-5-906531-01-8</p>
<p>Издано с разрешения Alcoholics Anonymous World Services, Inc. (A.A.W.S.)</p>
</body></html>""".encode()


class _RuBuilderProtocol(Protocol):
    """Typed surface of ``scripts/build_canonical_ru.py`` used by these tests."""

    BUILDER_VERSION: int
    ARTIFACT_FORMAT: str
    MANIFEST_FORMAT: str

    def sha256(self, data: bytes) -> str: ...
    def html_to_text(self, fragment: str) -> str: ...
    def parse_section_payload(self, page_html: bytes, *, section: str) -> dict[str, Any]: ...
    def extract_section_blocks(
        self, page_html: bytes, *, section: str, title: str, page: str
    ) -> tuple[list[dict[str, Any]], list[str]]: ...
    def check_headings(
        self, blocks: list[dict[str, Any]], *, first: str, second: str | None, section: str
    ) -> None: ...
    def build_canonical_ru(
        self,
        *,
        manifest: dict[str, Any],
        fetch_state: dict[str, Any],
        edition_html: bytes,
        pages: dict[str, bytes],
    ) -> dict[str, Any]: ...
    def serialize_artifact(self, artifact: dict[str, Any]) -> bytes: ...


def _load_builder() -> _RuBuilderProtocol:
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "build_canonical_ru.py"
    spec = importlib.util.spec_from_file_location("build_canonical_ru_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_RuBuilderProtocol, module)


def _fixture_page(section: str, title: str, page: str, first: str, second: str | None) -> bytes:
    blocks: list[dict[str, Any]] = [
        {"id": f"{section}-p1", "page": page, "html": f"<p><b>{first}</b></p>"},
    ]
    if second is not None:
        blocks.append({"id": f"{section}-p2", "page": page, "html": f"<p><b>{second}</b></p>"})
    body_index = len(blocks) + 1
    blocks.append(
        {
            "id": f"{section}-p{body_index}",
            "page": page,
            "html": "<p>Проверочный абзац один с точной формулировкой.</p>",
            "english": [{"sourceId": "x", "html": "<p>Parallel English must never leak.</p>"}],
            "alternatives": [{"sourceId": "y", "html": "<p>Запасной вариант не входит.</p>"}],
        }
    )
    blocks.append(
        {
            "id": f"{section}-p{body_index + 1}",
            "page": page,
            "html": "<p>Проверочный абзац два со<br/>строкой внутри.</p>",
            "english": [{"sourceId": "z", "html": "<p>More English that must be dropped.</p>"}],
        }
    )
    payload = {"section": {"id": section, "title": title, "page": page, "blocks": blocks}}
    document = (
        "<!DOCTYPE html><html lang='ru'><head><title>fixture</title></head><body>"
        "<nav>chrome</nav>"
        '<script type="application/json" id="book-initial">'
        + json.dumps(payload, ensure_ascii=False)
        + "</script></body></html>"
    )
    return document.encode("utf-8")


def _fixture_pages() -> tuple[bytes, dict[str, bytes]]:
    pages = {}
    for _sid, section, title, page, first, second in SECTION_SPECS:
        pages[section] = _fixture_page(section, title, page, first, second)
    # Chapter 11 carries a trailing stories part divider, dropped by the builder.
    base = pages["n147"].decode("utf-8")
    divider = "," + json.dumps(
        {"id": "n147-p99", "page": "160", "html": "<p><b>ЧАСТЬ 1</b></p>"},
        ensure_ascii=False,
    )
    blocks_close = base.rfind("]")
    assert blocks_close > 0
    pages["n147"] = (base[:blocks_close] + divider + base[blocks_close:]).encode("utf-8")
    return EDITION_HTML, pages


def _fixture_manifest(
    builder: _RuBuilderProtocol, edition: bytes, pages: dict[str, bytes]
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = [
        {
            "id": "edition",
            "section": "edition",
            "url": "https://example.invalid/ru/edition/",
            "raw_path": "corpus/source/raw-ru/edition.html",
            "sha256": builder.sha256(edition),
            "bytes": len(edition),
        }
    ]
    sections: list[dict[str, Any]] = []
    url_of = {"doctors-opinion": "nXXVII"}
    for sid, section, title, page, first, second in SECTION_SPECS:
        url_of.setdefault(sid, section)
        data = pages[section]
        sources.append(
            {
                "id": "doctors-opinion" if sid == "doctors-opinion" else sid,
                "section": section,
                "url": f"https://example.invalid/ru/{section}/",
                "raw_path": f"corpus/source/raw-ru/{section}.html",
                "sha256": builder.sha256(data),
                "bytes": len(data),
            }
        )
        blocks, _ = builder.extract_section_blocks(data, section=section, title=title, page=page)
        builder.check_headings(blocks, first=first, second=second, section=section)
        text = f"{title}\n\n" + "\n\n".join(str(b["text"]) for b in blocks) + "\n"
        sections.append(
            {
                "id": sid,
                "title": title,
                "source_id": "doctors-opinion" if sid == "doctors-opinion" else sid,
                "section": section,
                "page": page,
                "headings": [h for h in (first, second) if h is not None],
                "block_ids": [str(b["block_id"]) for b in blocks],
                "text_sha256": builder.sha256(text.encode("utf-8")),
            }
        )
    return {
        "format": builder.MANIFEST_FORMAT,
        "builder_version": builder.BUILDER_VERSION,
        "language": "ru",
        "edition": "fixture",
        "sku": "RUSSB-30",
        "rights_basis": "fixture",
        "sources": sources,
        "sections": sections,
    }


def _fixture_fetch_state(
    builder: _RuBuilderProtocol, edition: bytes, pages: dict[str, bytes]
) -> dict[str, Any]:
    sources: list[dict[str, Any]] = [
        {
            "id": "edition",
            "section": "edition",
            "url": "https://example.invalid/ru/edition/",
            "path": "corpus/source/raw-ru/edition.html",
            "bytes": len(edition),
            "sha256": builder.sha256(edition),
        }
    ]
    for sid, section, _title, _page, _first, _second in SECTION_SPECS:
        key = "doctors-opinion" if sid == "doctors-opinion" else sid
        data = pages[section]
        sources.append(
            {
                "id": key,
                "section": section,
                "url": f"https://example.invalid/ru/{section}/",
                "path": f"corpus/source/raw-ru/{section}.html",
                "bytes": len(data),
                "sha256": builder.sha256(data),
            }
        )
    return {"version": 1, "language": "ru", "sources": sources}


def _build_fixture(builder: _RuBuilderProtocol) -> tuple[dict[str, Any], bytes, dict[str, bytes]]:
    edition, pages = _fixture_pages()
    manifest = _fixture_manifest(builder, edition, pages)
    fetch_state = _fixture_fetch_state(builder, edition, pages)
    artifact = builder.build_canonical_ru(
        manifest=manifest, fetch_state=fetch_state, edition_html=edition, pages=pages
    )
    return artifact, edition, pages


def test_ru_build_has_exact_scope() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    assert artifact["format"] == "aa-canonical-ru/1"
    assert artifact["language"] == "ru"
    assert [s["id"] for s in artifact["sections"]] == list(EXPECTED_IDS)
    for section in artifact["sections"]:
        assert section["source_url"].startswith("https://example.invalid/ru/")
        assert section["source_file"].startswith("corpus/source/raw-ru/")
        assert section["source_section"] in {spec[1] for spec in SECTION_SPECS}
        assert section["text_sha256"] == hashlib.sha256(section["text"].encode()).hexdigest()
        assert section["text"].startswith(section["title"] + "\n\n")
        # Stable offsets resolve back to exact block text.
        for block in section["blocks"]:
            assert section["text"][block["char_start"] : block["char_end"]]
            assert (
                hashlib.sha256(
                    section["text"][block["char_start"] : block["char_end"]].encode()
                ).hexdigest()
                == block["text_sha256"]
            )


def test_ru_build_is_deterministic() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    first = builder.serialize_artifact(artifact)
    artifact_again, _, _ = _build_fixture(builder)
    assert builder.serialize_artifact(artifact_again) == first


def test_ru_english_parallel_and_alternatives_are_absent() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    for section in artifact["sections"]:
        assert "Parallel English must never leak" not in section["text"]
        assert "More English that must be dropped" not in section["text"]
        assert "Запасной вариант не входит" not in section["text"]
        assert "book-translation" not in section["text"]
    # Verse <br/> structure is preserved as a line break, wording untouched.
    chapter_1 = next(s for s in artifact["sections"] if s["id"] == "chapter-1")
    assert "Проверочный абзац два со\nстрокой внутри." in chapter_1["text"]
    assert "Проверочный абзац один с точной формулировкой." in chapter_1["text"]


def test_ru_chapter_11_part_divider_is_dropped() -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    chapter_11 = next(s for s in artifact["sections"] if s["id"] == "chapter-11")
    assert "ЧАСТЬ 1" not in chapter_11["text"]
    assert "n147-p99" not in chapter_11["block_ids"]


def test_ru_wrong_heading_fails_closed() -> None:
    builder = _load_builder()
    edition, pages = _fixture_pages()
    broken = pages["n1"].replace("РАССКАЗ БИЛЛА".encode(), "НЕ ТОТ ЗАГОЛОВОК".encode())
    blocks, _ = builder.extract_section_blocks(
        broken, section="n1", title="Глава 1. Рассказ Билла", page="1"
    )
    with pytest.raises(ValueError, match="heading mismatch"):
        builder.check_headings(blocks, first="ГЛАВА 1", second="РАССКАЗ БИЛЛА", section="n1")
    manifest = _fixture_manifest(builder, edition, pages)
    fetch_state = _fixture_fetch_state(builder, edition, pages)
    tampered_pages = dict(pages)
    tampered_pages["n1"] = broken
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_canonical_ru(
            manifest=manifest, fetch_state=fetch_state, edition_html=edition, pages=tampered_pages
        )


def test_ru_edition_mismatch_fails_closed() -> None:
    builder = _load_builder()
    _edition, pages = _fixture_pages()
    manifest = _fixture_manifest(builder, EDITION_HTML, pages)
    fetch_state = _fixture_fetch_state(builder, EDITION_HTML, pages)
    bad_edition = EDITION_HTML.replace(b"978-5-906531-01-8", b"000-0-000000-00-0")
    with pytest.raises(ValueError, match="edition mismatch"):
        builder.build_canonical_ru(
            manifest=manifest, fetch_state=fetch_state, edition_html=bad_edition, pages=pages
        )


def test_ru_missing_section_fails_closed() -> None:
    builder = _load_builder()
    edition, pages = _fixture_pages()
    manifest = _fixture_manifest(builder, edition, pages)
    fetch_state = _fixture_fetch_state(builder, edition, pages)
    missing = dict(pages)
    del missing["n43"]
    with pytest.raises(ValueError, match="raw page is missing"):
        builder.build_canonical_ru(
            manifest=manifest, fetch_state=fetch_state, edition_html=edition, pages=missing
        )


def test_ru_moved_section_fails_closed() -> None:
    builder = _load_builder()
    _edition, pages = _fixture_pages()
    moved = pages["n16"].replace(b'"id": "n16"', b'"id": "n29"')
    with pytest.raises(ValueError, match="section id mismatch"):
        builder.extract_section_blocks(moved, section="n16", title="Глава 2. Выход есть", page="16")


def test_ru_stale_source_fails_closed() -> None:
    builder = _load_builder()
    edition, pages = _fixture_pages()
    manifest = _fixture_manifest(builder, edition, pages)
    fetch_state = _fixture_fetch_state(builder, edition, pages)
    with pytest.raises(ValueError, match="checksum mismatch"):
        builder.build_canonical_ru(
            manifest=manifest, fetch_state=fetch_state, edition_html=edition + b"X", pages=pages
        )


def test_ru_load_round_trip_and_exact_reads(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    payload = builder.serialize_artifact(artifact)
    target = tmp_path / "canonical.ru.json"
    target.write_bytes(payload)
    corpus = load_canonical_ru(target, expected_sha256=hashlib.sha256(payload).hexdigest())
    assert corpus.ids() == EXPECTED_IDS
    first = corpus.get("chapter-1")
    assert corpus.read("chapter-1", 0, len(first.text)) == first.text
    with pytest.raises(CanonicalRangeError):
        corpus.read("chapter-1", 0, len(first.text) + 1)
    with pytest.raises(CanonicalCorpusError):
        corpus.get("chapter-12")


def test_ru_load_rejects_english_artifact(tmp_path: pathlib.Path) -> None:
    builder = _load_builder()
    artifact, _, _ = _build_fixture(builder)
    payload = builder.serialize_artifact(artifact)
    target = tmp_path / "canonical.ru.json"
    target.write_bytes(payload)
    tampered = json.loads(payload.decode("utf-8"))
    tampered["format"] = "aa-canonical/1"
    target.write_bytes(json.dumps(tampered).encode("utf-8"))
    with pytest.raises(CanonicalCorpusError, match="unsupported canonical artifact format"):
        load_canonical_ru(target)


def test_ru_committed_manifest_carries_no_book_text() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest_path = root / "corpus" / "canonical.ru.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "aa-canonical-manifest-ru/1"
    assert manifest["language"] == "ru"
    assert manifest["sku"] == "RUSSB-30"
    assert "rights_basis" in manifest and manifest["rights_basis"]
    text = manifest_path.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) < 60_000
    for snippet in ("Проверочный абзац", "Маленький городок в Новой Англии"):
        assert snippet not in text


def test_ru_real_manifest_matches_expected_scope_and_pins_checksums() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "corpus" / "canonical.ru.manifest.json").read_text())
    assert [s["id"] for s in manifest["sections"]] == list(EXPECTED_IDS)
    assert re.search(r"^[0-9a-f]{64}$", manifest["artifact_sha256"]) is not None
    for source in manifest["sources"]:
        assert re.search(r"^[0-9a-f]{64}$", source["sha256"]) is not None
        assert source["bytes"] > 0
        assert source["url"].startswith("https://aarus.fi/read/bigbook/")
    for section in manifest["sections"]:
        assert re.search(r"^[0-9a-f]{64}$", section["text_sha256"]) is not None
        assert len(section["block_ids"]) > 0
    # Source lock records the exact edition/SKU/rights basis.
    lock = json.loads((root / "corpus" / "source.ru.lock.json").read_text())
    assert lock["catalog"]["sku"] == "RUSSB-30"
    assert "978-5-906531-01-8" in lock["edition"]
    assert "Фонд «Единство», 2013" in lock["edition"]
    assert "Owner confirms permission" in lock["rights"]["basis"]
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/source/raw-ru/" in gitignore
    assert "fetch-ru-state.json" in gitignore


def test_ru_refresh_workflow_is_manual_trusted_and_cache_safe() -> None:
    root = pathlib.Path(__file__).resolve().parents[1]
    workflow = (root / ".github" / "workflows" / "encrypted-corpus-refresh-ru.yml").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch" in workflow
    assert "pull_request" not in workflow
    assert "AA_BOOK_AGE_IDENTITY" in workflow
    assert "actions/cache" not in workflow
    assert "administration" not in workflow.lower()
    # RU refresh uses the RU pipeline and the distinct RU snapshot names
    # encrypted to the same recipient (no second private key).
    for marker in (
        "fetch_ru_source.py",
        "build_canonical_ru.py",
        "canonical.ru.tar.zst.age",
        "metadata.ru.json",
        "canonical.ru.manifest.json",
        "--lang ru",
    ):
        assert marker in workflow


def test_ru_refresh_and_restore_round_trip_with_ephemeral_keypair(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from aa.corpus import age_v1
    from aa.corpus import encrypted_snapshot as snap

    identity, recipient = age_v1.generate_identity()
    try:
        builder = _load_builder()
        artifact, _, _ = _build_fixture(builder)
        payload = builder.serialize_artifact(artifact)
        canonical_path = tmp_path / "canonical.ru.json"
        canonical_path.write_bytes(payload)
        manifest = {
            "format": "aa-canonical-manifest-ru/1",
            "builder_version": 1,
            "language": "ru",
            "edition": "fixture",
            "artifact_sha256": hashlib.sha256(payload).hexdigest(),
        }
        manifest_path = tmp_path / "manifest.ru.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        lock_path = tmp_path / "lock.ru.json"
        lock_path.write_text(
            json.dumps({"version": 1, "language": "ru", "edition": "fixture"}), encoding="utf-8"
        )
        encrypted_dir = tmp_path / "encrypted"
        recipient_path = tmp_path / "recipient.txt"
        recipient_path.write_text(recipient + "\n", encoding="utf-8")
        identity_path = tmp_path / "identity.txt"
        identity_path.write_text(identity + "\n", encoding="utf-8")
        monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)

        repo_root = pathlib.Path(__file__).resolve().parents[1]
        env = dict(__import__("os").environ)

        def _run(path: pathlib.Path, args: list[str]) -> Any:
            return subprocess.run(
                [sys.executable, str(path), *args],
                capture_output=True,
                text=True,
                cwd=repo_root,
                env=env,
                check=False,
            )

        refresh = repo_root / "scripts" / "refresh_encrypted_snapshot.py"
        proc = _run(
            refresh,
            [
                "--manifest",
                str(manifest_path),
                "--source-lock",
                str(lock_path),
                "--canonical",
                str(canonical_path),
                "--encrypted-dir",
                str(encrypted_dir),
                "--recipient-file",
                str(recipient_path),
                "--identity-file",
                str(identity_path),
                "--archive-name",
                "canonical.ru.tar.zst.age",
                "--metadata-name",
                "metadata.ru.json",
                "--canonical-name",
                "canonical.ru.json",
            ],
        )
        assert proc.returncode == 0, proc.stderr
        assert "decrypt verification ok" in proc.stdout
        assert identity not in proc.stdout
        assert "Проверочный абзац" not in proc.stdout
        archive = encrypted_dir / "canonical.ru.tar.zst.age"
        metadata_path = encrypted_dir / "metadata.ru.json"
        assert archive.exists()
        assert payload not in archive.read_bytes()
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["canonical_sha256"] == hashlib.sha256(payload).hexdigest()
        assert metadata["encrypted_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert metadata["encryption_format"] == snap.ENCRYPTION_FORMAT
        assert metadata["canonical_name"] == "canonical.ru.json"

        restore = repo_root / "scripts" / "restore_canonical.py"
        output = tmp_path / "generated" / "canonical.ru.json"
        proc = _run(
            restore,
            [
                "--lang",
                "ru",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--encrypted-dir",
                str(encrypted_dir),
                "--identity-file",
                str(identity_path),
                "--no-network-fallback",
            ],
        )
        assert proc.returncode == 0, proc.stderr
        assert output.read_bytes() == payload
    finally:
        identity = "destroyed"  # noqa: F841
