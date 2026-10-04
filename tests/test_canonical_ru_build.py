"""Russian canonical TXT pipeline tests (issue #50).

All literary content uses invented fixture sentences; no canonical book
text is committed. Real validation excerpts live only in
``scripts/build_canonical_ru.py`` as short control markers and are reused
here by import (never duplicated), plus hashes in the manifest.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
import unicodedata
from typing import Any

import pytest

from aa.corpus import age_v1
from aa.corpus import encrypted_snapshot as snap
from aa.corpus.canonical import load_canonical_ru


def _repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


def _load_build() -> Any:
    path = _repo_root() / "scripts" / "build_canonical_ru.py"
    spec = importlib.util.spec_from_file_location("build_canonical_ru_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_fetch() -> Any:
    path = _repo_root() / "scripts" / "fetch_ru_source.py"
    spec = importlib.util.spec_from_file_location("fetch_ru_source_under_test", str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Invented fixture body lines per section (wording is ours, not the book).
_FIXTURE_BODIES = {
    "doctors-opinion": (
        "МНЕНИЕ ДОКТОРА\n\nФиктивный абзац врачебного мнения номер один.\n\n"
        "Фиктивный абзац врачебного мнения номер два про трезвость и поддержку."
    ),
    "chapter-1": (
        "ГЛАВА 1\n\nРАССКАЗ БИЛЛА\n\nФиктивный рассказ Билла: утренние собрания "
        "и поддержка друзей. Буквы ё «кавычки» — тире… конец."
    ),
    "chapter-2": "ГЛАВА 2\n\nВЫХОД ЕСТЬ\n\nФиктивный абзац второй главы про выход и надежду.",
    "chapter-3": "ГЛАВА 3\n\nЕЩЕ ОБ АЛКОГОЛИЗМЕ\n\nФиктивный абзац третьей главы.",
    "chapter-4": "ГЛАВА 4\n\nА КАК БЫТЬ АГНОСТИКАМ?\n\nФиктивный абзац четвёртой главы.",
    "chapter-5": "ГЛАВА 5\n\nПРОГРАММА В ДЕЙСТВИИ\n\nФиктивный абзац пятой главы.",
    "chapter-6": "ГЛАВА 6\n\nЗА РАБОТУ!\n\nФиктивный абзац шестой главы.",
    "chapter-7": "ГЛАВА 7\n\nРАБОТАЯ С ДРУГИМИ\n\nФиктивный абзац седьмой главы.",
    "chapter-8": "ГЛАВА 8\n\nОБРАЩЕНИЕ К ЖЕНАМ\n\nФиктивный абзац восьмой главы.",
    "chapter-9": "ГЛАВА 9\n\nНОВЫЕ ОТНОШЕНИЯ В СЕМЬЕ\n\nФиктивный абзац девятой главы.",
    "chapter-10": "ГЛАВА 10\n\nОБРАЩЕНИЕ К РАБОТОДАТЕЛЯМ\n\nФиктивный абзац десятой главы.",
    "chapter-11": "ГЛАВА 11\n\nЗАГЛЯНЕМ В ВАШЕ БУДУЩЕЕ\n\nФиктивный абзац одиннадцатой главы.",
}

_FIXTURE_TITLES = {
    "doctors-opinion": "Мнение доктора",
    "chapter-1": "Глава 1. Рассказ Билла",
    "chapter-2": "Глава 2. Выход есть",
    "chapter-3": "Глава 3. Еще об алкоголизме",
    "chapter-4": "Глава 4. А как быть агностикам?",
    "chapter-5": "Глава 5. Программа в действии",
    "chapter-6": "Глава 6. За работу!",
    "chapter-7": "Глава 7. Работая с другими",
    "chapter-8": "Глава 8. Обращение к женам",
    "chapter-9": "Глава 9. Новые отношения в семье",
    "chapter-10": "Глава 10. Обращение к работодателям",
    "chapter-11": "Глава 11. Заглянем в ваше будущее",
}

_ORDER = [
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

_PREAMBLE = (
    "АНОНИМНЫЕ АЛКОГОЛИКИ\n\n4-е издание\n\nЧетвертое издание\n\n"
    "Перевод с английского\n\nAlcoholics Anonymous World Services, Inc.\n\n"
    "Нью-Йорк: Alcoholics Anonymous World Services, Inc;\n"
    "Фонд «Единство», 2013. - 192 с.\n"
    "ISBN 978-5-906531-01-8 (Фонд «Единство»)\n\n"
    "Издано с разрешения Alcoholics Anonymous World Services, Inc. (A.A.W.S.)."
)


def _fixture_txt(build: Any) -> bytes:
    """Build a valid fixture TXT embedding the build's own control passages."""
    bodies = dict(_FIXTURE_BODIES)
    # Embed each required control passage verbatim in its expected section so
    # fixtures exercise the real validation without duplicating book text here.
    for section_id, passage in build.CONTROL_PASSAGES:
        bodies[section_id] = bodies[section_id] + "\n\n" + passage
    parts = [_PREAMBLE]
    for section_id in _ORDER:
        parts.append(
            f"@@SECTION:{section_id}|{_FIXTURE_TITLES[section_id]}@@\n\n{bodies[section_id]}"
        )
    normalized: str = build.normalize_text("\n\n".join(parts).strip() + "\n")
    return normalized.encode("utf-8")


def _fixture_manifest(build: Any, raw: bytes) -> dict[str, Any]:
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    canonical_txt, enriched = build.build_canonical_text(sections)
    manifest_sections = []
    for item in enriched:
        body = str(item["body"])
        manifest_sections.append(
            {
                "id": str(item["id"]),
                "title": str(item["title"]),
                "text_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
                "chars": len(body),
            }
        )
    prov: dict[str, Any] = {
        "format": build.MANIFEST_FORMAT,
        "builder_version": 1,
        "edition": "fixture",
        "sources": [
            {
                "id": "ru-fourth-edition-txt",
                "raw_path": "corpus/source/raw-ru/aa-big-book.txt",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        ],
        "sections": manifest_sections,
    }
    fetch_state = {
        "sources": [
            {
                "id": "ru-fourth-edition-txt",
                "path": "corpus/source/raw-ru/aa-big-book.txt",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ]
    }
    artifact = build.build_artifact(
        manifest=prov,
        fetch_state=fetch_state,
        raw=raw,
        normalized=normalized,
        preamble=preamble,
        sections=sections,
        canonical_txt=canonical_txt,
        enriched=enriched,
    )
    payload = build.serialize_artifact(artifact)
    prov["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
    prov["text_artifact_sha256"] = hashlib.sha256(canonical_txt.encode("utf-8")).hexdigest()
    return prov


def _fixture_state(raw: bytes) -> dict[str, Any]:
    return {
        "version": 1,
        "language": "ru",
        "sources": [
            {
                "id": "ru-fourth-edition-txt",
                "path": "corpus/source/raw-ru/aa-big-book.txt",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }


def _build_fixture(build: Any) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    raw = _fixture_txt(build)
    return raw, _fixture_manifest(build, raw), _fixture_state(raw)


def test_fixture_has_exact_scope_and_order() -> None:
    build = _load_build()
    raw, manifest, state = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    preamble, raw_sections = build.split_sections(build.normalize_text(decoded))
    assert [sid for sid, _, _ in raw_sections] == _ORDER
    assert [s["id"] for s in manifest["sections"]] == _ORDER
    assert manifest["sources"][0]["raw_path"] == "corpus/source/raw-ru/aa-big-book.txt"
    assert state["sources"][0]["path"] == "corpus/source/raw-ru/aa-big-book.txt"


def test_identity_validation_rejects_wrong_edition() -> None:
    build = _load_build()
    raw = _fixture_txt(build)
    bad_isbn = raw.replace(b"978-5-906531-01-8", b"000-0-000000-00-0")
    decoded, _ = build.decode_txt(bad_isbn)
    preamble, _ = build.split_sections(build.normalize_text(decoded))
    with pytest.raises(ValueError, match="identity mismatch"):
        build.validate_preamble(preamble)
    bad_title = raw.replace("АНОНИМНЫЕ АЛКОГОЛИКИ".encode(), "ДРУГАЯ КНИГА".encode())
    decoded, _ = build.decode_txt(bad_title)
    preamble, _ = build.split_sections(build.normalize_text(decoded))
    with pytest.raises(ValueError, match="identity mismatch"):
        build.validate_preamble(preamble)


def test_raw_sha_drift_fails_closed() -> None:
    build = _load_build()
    raw, manifest, state = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    canonical_txt, enriched = build.build_canonical_text(sections)
    drifted = raw + b" "
    with pytest.raises(ValueError, match="fetch-ru-state"):
        build.build_artifact(
            manifest=manifest,
            fetch_state=_fixture_state(drifted),
            raw=raw,
            normalized=normalized,
            preamble=preamble,
            sections=sections,
            canonical_txt=canonical_txt,
            enriched=enriched,
        )
    tampered_manifest = json.loads(json.dumps(manifest))
    tampered_manifest["sources"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="manifest"):
        build.build_artifact(
            manifest=tampered_manifest,
            fetch_state=state,
            raw=raw,
            normalized=normalized,
            preamble=preamble,
            sections=sections,
            canonical_txt=canonical_txt,
            enriched=enriched,
        )


def test_deterministic_decoding_bom_nfc_newlines() -> None:
    build = _load_build()
    raw = _fixture_txt(build)
    # BOM is stripped deterministically.
    decoded, had_bom = build.decode_txt(b"\xef\xbb\xbf" + raw)
    assert had_bom is True
    assert decoded.lstrip("\ufeff").encode("utf-8") == raw
    # CRLF/CR collapse to LF.
    crlf = raw.replace(b"\n", b"\r\n")
    assert build.normalize_text(crlf.decode("utf-8")).encode("utf-8") == raw
    # NFC: decomposed e + combining acute becomes a single composed char.
    decomposed = "e\u0301".encode("utf-8")
    assert unicodedata.normalize("NFC", "e\u0301").encode("utf-8") != decomposed
    assert build.normalize_text("e\u0301") == unicodedata.normalize("NFC", "e\u0301")
    # Determinism: normalizing twice is a fixed point.
    once = build.normalize_text(raw.decode("utf-8"))
    assert build.normalize_text(once) == once


def test_normalization_preserves_wording_only() -> None:
    build = _load_build()
    raw, manifest, state = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    chapter1 = next(s for s in sections if s["id"] == "chapter-1")
    # Punctuation and spelling survive: no literary rewriting took place.
    assert "ё" in str(chapter1["body"])
    assert "«кавычки»" in str(chapter1["body"])
    assert "—" in str(chapter1["body"])
    assert "…" in str(chapter1["body"])


def test_missing_reordered_section_fails_closed() -> None:
    build = _load_build()
    raw = _fixture_txt(build).decode("utf-8")
    # Drop chapter-6 delimiter -> body merges into chapter-5; order breaks.
    broken = raw.replace("@@SECTION:chapter-6|Глава 6. За работу!@@\n\n", "")
    with pytest.raises(ValueError, match="exactly"):
        build.validate_sections(build.split_sections(build.normalize_text(broken))[1])
    # Swap two sections.
    part5 = "@@SECTION:chapter-5|Глава 5. Программа в действии@@"
    part6 = "@@SECTION:chapter-6|Глава 6. За работу!@@"
    swapped = raw.replace(part5, "@@TMP@@").replace(part6, part5).replace("@@TMP@@", part6)
    with pytest.raises(ValueError, match="exactly"):
        build.validate_sections(build.split_sections(build.normalize_text(swapped))[1])


def test_headings_and_control_passages_enforced() -> None:
    build = _load_build()
    raw = _fixture_txt(build)
    # Corrupt a heading.
    broken = raw.replace("РАССКАЗ БИЛЛА".encode(), "РАССКАЗ ПЕТИ".encode())
    decoded, _ = build.decode_txt(broken)
    with pytest.raises(ValueError, match="heading mismatch"):
        build.validate_sections(build.split_sections(build.normalize_text(decoded))[1])
    # Remove one control passage.
    _, passage = build.CONTROL_PASSAGES[1]
    truncated = raw.replace(passage.encode(), b"")
    decoded, _ = build.decode_txt(truncated)
    with pytest.raises(ValueError, match="control passage"):
        build.validate_sections(build.split_sections(build.normalize_text(decoded))[1])


def test_rejects_truncated_corrupt_txt() -> None:
    build = _load_build()
    raw = _fixture_txt(build)
    # Truncated: drop the final section.
    cut = raw.rsplit(b"@@SECTION:chapter-11|", 1)[0]
    with pytest.raises(ValueError, match="exactly"):
        sections = build.split_sections(build.normalize_text(cut.decode()))[1]
        build.validate_sections(sections)
    # NUL bytes fail closed.
    with pytest.raises(ValueError, match="NUL"):
        build.decode_txt(raw + b"\x00")
    # Invalid UTF-8 fails closed.
    with pytest.raises(ValueError, match="UTF-8"):
        build.decode_txt(b"\xff\xfe invalid")
    # Empty section fails closed.
    empty = raw.replace(
        _FIXTURE_BODIES["chapter-3"].encode(),
        "\n".join(["ГЛАВА 3", "", "ЕЩЕ ОБ АЛКОГОЛИЗМЕ"]).encode(),
    )
    decoded, _ = build.decode_txt(empty)
    _, raw_sections = build.split_sections(build.normalize_text(decoded))
    # The emptied section still has headings but no literary body; the builder
    # must either accept headings-only (no) or fail. Force a truly blank body:
    blank = re.sub(
        r"(@@SECTION:chapter-3\|[^\n]+@@)\n\n.*?(?=\n@@SECTION:)",
        r"\1\n\n   \n",
        _fixture_txt(build).decode("utf-8"),
        flags=re.S,
    )
    decoded, _ = build.decode_txt(blank.encode())
    with pytest.raises(ValueError, match="blank"):
        build.validate_sections(build.split_sections(build.normalize_text(decoded))[1])


def test_build_round_trip_and_offsets(tmp_path: pathlib.Path) -> None:
    build = _load_build()
    raw, manifest, state = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    canonical_txt, enriched = build.build_canonical_text(sections)
    # Offsets tile the canonical text contiguously in order.
    cursor = 0
    for item in enriched:
        assert item["char_start"] == cursor
        assert item["char_end"] > cursor
        cursor = int(item["char_end"])
    assert cursor == len(canonical_txt)
    artifact = build.build_artifact(
        manifest=manifest,
        fetch_state=state,
        raw=raw,
        normalized=normalized,
        preamble=preamble,
        sections=sections,
        canonical_txt=canonical_txt,
        enriched=enriched,
    )
    payload = build.serialize_artifact(artifact)
    assert hashlib.sha256(payload).hexdigest() == manifest["artifact_sha256"]
    target = tmp_path / "canonical.ru.json"
    target.write_bytes(payload)
    corpus = load_canonical_ru(target, expected_sha256=manifest["artifact_sha256"])
    assert corpus.ids() == tuple(_ORDER)
    assert corpus.get("chapter-1").text in canonical_txt


def test_no_pdf_ocr_dependency() -> None:
    root = _repo_root()
    for name in ("scripts/fetch_ru_source.py", "scripts/build_canonical_ru.py"):
        text = (root / name).read_text(encoding="utf-8").lower()
        for forbidden in (
            "pdfminer",
            "pypdf",
            "pdfplumber",
            "pymupdf",
            "fitz.",
            "ocrmypdf",
            "pytesseract",
            "easyocr",
            "pdf2image",
        ):
            assert forbidden not in text, f"{name} must not depend on {forbidden}"
    lock = json.loads((root / "corpus" / "source.ru.lock.json").read_text(encoding="utf-8"))
    assert lock["policy"]["no_pdf_no_ocr"] is True
    assert lock["control"]["required"] is False


def test_no_llm_rewriting_imports() -> None:
    root = _repo_root()
    for name in ("scripts/fetch_ru_source.py", "scripts/build_canonical_ru.py"):
        text = (root / name).read_text(encoding="utf-8")
        lowered = text.lower()
        for forbidden in ("import openai", "import anthropic", "from openai", "from anthropic"):
            assert forbidden not in lowered, f"{name} must not use LLM APIs"
    build = _load_build()
    # The builder normalizes only transport artifacts (NFC/newlines/BOM).
    import inspect

    source = inspect.getsource(build.normalize_text)
    assert "NFC" in source
    assert "openai" not in source.lower()


def test_encryption_round_trip_same_recipient_contract(tmp_path: pathlib.Path) -> None:
    build = _load_build()
    raw, manifest, _ = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    canonical_txt, enriched = build.build_canonical_text(sections)
    artifact = build.build_artifact(
        manifest=manifest,
        fetch_state=_fixture_state(raw),
        raw=raw,
        normalized=normalized,
        preamble=preamble,
        sections=sections,
        canonical_txt=canonical_txt,
        enriched=enriched,
    )
    payload = build.serialize_artifact(artifact)
    identity, recipient = age_v1.generate_identity()
    try:
        tar_zst = snap.create_tar_zst(
            payload,
            manifest=manifest,
            source_lock={"version": 1},
            canonical_name=snap.RU_CANONICAL_NAME,
        )
        encrypted = age_v1.encrypt_bytes(tar_zst, [recipient])
        assert payload not in encrypted
        recovered_zst = age_v1.decrypt_bytes(encrypted, [identity])
        recovered, _ = snap.extract_tar_zst(
            recovered_zst, expected_canonical_name=snap.RU_CANONICAL_NAME
        )
        assert recovered == payload
        assert snap.RU_ARCHIVE_NAME == "canonical.ru.tar.zst.age"
        assert snap.RU_METADATA_NAME == "metadata.ru.json"
        assert snap.RU_CANONICAL_NAME == "canonical.ru.json"
        # The committed shared recipient parses under the same contract.
        committed = (
            (_repo_root() / "corpus" / "source" / "encrypted" / "recipient.txt")
            .read_text(encoding="utf-8")
            .strip()
        )
        assert committed.startswith("age1")
        age_v1.parse_recipient(committed)
    finally:
        identity = "destroyed"  # noqa: F841


def test_no_plaintext_or_key_leakage() -> None:
    root = _repo_root()
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, cwd=root, check=True
    ).stdout.splitlines()
    assert not [p for p in tracked if p.startswith("corpus/source/raw-ru/")]
    assert not [p for p in tracked if "canonical.ru.json" in p and "manifest" not in p]
    assert not [p for p in tracked if p.endswith("aa-big-book.txt")]
    assert not [p for p in tracked if p == "corpus/source/fetch-ru-state.json"]
    assert not [p for p in tracked if "AGE-SECRET-KEY" in p or p.endswith(".key")]
    manifest_text = (root / "corpus" / "canonical.ru.manifest.json").read_text(encoding="utf-8")
    assert len(manifest_text.encode("utf-8")) < 20_000
    assert "Фиктивный абзац" not in manifest_text
    lock_text = (root / "corpus" / "source.ru.lock.json").read_text(encoding="utf-8")
    assert "Фиктивный абзац" not in lock_text
    gitignore = (root / ".gitignore").read_text(encoding="utf-8")
    assert "/corpus/source/raw-ru/" in gitignore
    assert "fetch-ru-state.json" in gitignore
    assert "/corpus/generated/" in gitignore


def test_refresh_and_restore_ru_round_trip(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _load_build()
    raw, manifest, _ = _build_fixture(build)
    decoded, _ = build.decode_txt(raw)
    normalized = build.normalize_text(decoded)
    preamble, raw_sections = build.split_sections(normalized)
    sections = build.validate_sections(raw_sections)
    canonical_txt, enriched = build.build_canonical_text(sections)
    artifact = build.build_artifact(
        manifest=manifest,
        fetch_state=_fixture_state(raw),
        raw=raw,
        normalized=normalized,
        preamble=preamble,
        sections=sections,
        canonical_txt=canonical_txt,
        enriched=enriched,
    )
    payload = build.serialize_artifact(artifact)
    identity, recipient = age_v1.generate_identity()
    try:
        canonical_path = tmp_path / "canonical.ru.json"
        canonical_path.write_bytes(payload)
        manifest_path = tmp_path / "manifest.ru.json"
        ru_manifest = dict(manifest)
        ru_manifest["artifact_sha256"] = hashlib.sha256(payload).hexdigest()
        manifest_path.write_text(json.dumps(ru_manifest), encoding="utf-8")
        lock_path = tmp_path / "lock.ru.json"
        lock_path.write_text(json.dumps({"version": 1}), encoding="utf-8")
        encrypted_dir = tmp_path / "encrypted"
        recipient_path = tmp_path / "recipient.txt"
        recipient_path.write_text(recipient + "\n", encoding="utf-8")
        identity_path = tmp_path / "identity.txt"
        identity_path.write_text(identity + "\n", encoding="utf-8")
        monkeypatch.delenv("AA_BOOK_AGE_IDENTITY", raising=False)

        refresh = _repo_root() / "scripts" / "refresh_encrypted_snapshot.py"
        env = dict(__import__("os").environ)
        proc = subprocess.run(
            [
                sys.executable,
                str(refresh),
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
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "decrypt verification ok" in proc.stdout
        assert identity not in proc.stdout
        archive = encrypted_dir / "canonical.ru.tar.zst.age"
        assert archive.exists()
        assert payload not in archive.read_bytes()
        metadata = json.loads((encrypted_dir / "metadata.ru.json").read_text(encoding="utf-8"))
        assert metadata["canonical_sha256"] == hashlib.sha256(payload).hexdigest()
        assert metadata["canonical_member"] == "canonical.ru.json"

        restore = _repo_root() / "scripts" / "restore_canonical.py"
        output = tmp_path / "generated" / "canonical.ru.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(restore),
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
                "--archive-name",
                "canonical.ru.tar.zst.age",
                "--metadata-name",
                "metadata.ru.json",
                "--no-network-fallback",
            ],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            env=env,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert output.read_bytes() == payload
        assert hashlib.sha256(output.read_bytes()).hexdigest() == ru_manifest["artifact_sha256"]
        assert identity not in proc.stdout

        # Runtime decrypts once and reuses: restore without identity reuses output.
        mtime = output.stat().st_mtime_ns
        proc = subprocess.run(
            [
                sys.executable,
                str(restore),
                "--lang",
                "ru",
                "--manifest",
                str(manifest_path),
                "--output",
                str(output),
                "--encrypted-dir",
                str(encrypted_dir),
                "--archive-name",
                "canonical.ru.tar.zst.age",
                "--metadata-name",
                "metadata.ru.json",
                "--no-network-fallback",
            ],
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            env={k: v for k, v in env.items() if k != "AA_BOOK_AGE_IDENTITY"},
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert "reused" in proc.stdout
        assert output.stat().st_mtime_ns == mtime
    finally:
        identity = "destroyed"  # noqa: F841


def test_real_manifest_is_pinned_and_text_free() -> None:
    root = _repo_root()
    manifest_path = root / "corpus" / "canonical.ru.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "aa-canonical-manifest-ru/1"
    assert manifest["builder_version"] == 1
    assert manifest["language"] == "ru"
    assert "978-5-906531-01-8" in manifest["edition"]
    assert "2013" in manifest["edition"]
    assert "Фонд" in manifest["edition"]
    assert [s["id"] for s in manifest["sections"]] == _ORDER
    assert re.search(r"^[0-9a-f]{64}$", manifest["artifact_sha256"]) is not None
    assert re.search(r"^[0-9a-f]{64}$", manifest["text_artifact_sha256"]) is not None
    assert re.search(r"^[0-9a-f]{64}$", manifest["sources"][0]["sha256"]) is not None
    assert manifest["sources"][0]["raw_path"] == "corpus/source/raw-ru/aa-big-book.txt"
    text = manifest_path.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) < 20_000


def test_golden_normalization_is_deterministic() -> None:
    build = _load_build()
    decomposed = "é"
    assert len(decomposed) == 2  # e + combining acute; NFC must compose it.
    sample_bytes = b"\xef\xbb\xbf" + ("A\r\nB\r" + decomposed).encode("utf-8")
    decoded, had_bom = build.decode_txt(sample_bytes)
    assert had_bom is True
    normalized = build.normalize_text(decoded)
    assert normalized == "A\nB\né"
    assert build.normalize_text(normalized) == normalized
    assert (
        hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        == hashlib.sha256("A\nB\né".encode()).hexdigest()
    )
