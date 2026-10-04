#!/usr/bin/env python3
"""Build the deterministic Russian canonical artifact from TXT (issue #50).

The builder reads only the normalized TXT
``corpus/source/raw-ru/aa-big-book.txt`` (preserved byte-for-byte by
``scripts/fetch_ru_source.py``). It never downloads anything, never reads
HTML or PDF, and never uses OCR:

- validates the raw TXT against ``corpus/source/fetch-ru-state.json`` and
  the committed ``corpus/canonical.ru.manifest.json`` (SHA-256, bytes);
- decodes deterministically as UTF-8 (BOM stripped deterministically);
- normalizes only transport artifacts (Unicode NFC, newline convention);
- validates Fourth Edition / 2013 / ISBN identity from fixed markers;
- locates Мнение доктора + Chapters 1-11 in exact order with stable
  delimiters, validating headings and fixed control passages;
- writes exactly ``corpus/generated/canonical.ru.json`` and
  ``corpus/generated/canonical.ru.txt`` (both ignored by Git).

No LLM rewriting, translation, summarization, or literary normalization
takes place: spelling, grammar, punctuation, and wording are never altered.
Any stale source, wrong edition, missing/reordered section, undecodable or
corrupt TXT, or unexpected drift fails closed with no artifact written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_FETCH_STATE = ROOT / "corpus" / "source" / "fetch-ru-state.json"
DEFAULT_RAW = ROOT / "corpus" / "source" / "raw-ru" / "aa-big-book.txt"
DEFAULT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.ru.json"
DEFAULT_TEXT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.ru.txt"

BUILDER_VERSION = 1
ARTIFACT_FORMAT = "aa-canonical-ru/1"
MANIFEST_FORMAT = "aa-canonical-manifest-ru/1"

RAW_PATH = "corpus/source/raw-ru/aa-big-book.txt"
SOURCE_ID = "ru-fourth-edition-txt"
# Stable per-section provenance for the single-TXT source (issue #50): every
# section shares the pinned edition-page URL from
# ``corpus/source.ru.lock.json`` (``sources[0].url``), exactly as the English
# builder shares one source URL across chapters from a single raw file.
# Per-section binding additionally comes from ``source_id``/``source_file``,
# ``char_start``/``char_end`` offsets, and ``text_sha256``; the URL must
# never be empty.
SOURCE_URL = "https://aarus.fi/read/bigbook/edition/"

IDENTITY_MARKERS = (
    "АНОНИМНЫЕ АЛКОГОЛИКИ",
    "4-е издание",
    "Четвертое издание",
    "Перевод с английского",
    "Alcoholics Anonymous World Services",
    "Фонд «Единство», 2013",
    "978-5-906531-01-8",
    "с разрешения Alcoholics Anonymous World Services",
)

# (canonical id, required title, required first heading lines).
REQUIRED_SECTIONS = (
    ("doctors-opinion", "Мнение доктора", ("МНЕНИЕ ДОКТОРА",)),
    ("chapter-1", "Глава 1. Рассказ Билла", ("ГЛАВА 1", "РАССКАЗ БИЛЛА")),
    ("chapter-2", "Глава 2. Выход есть", ("ГЛАВА 2", "ВЫХОД ЕСТЬ")),
    ("chapter-3", "Глава 3. Еще об алкоголизме", ("ГЛАВА 3", "ЕЩЕ ОБ АЛКОГОЛИЗМЕ")),
    ("chapter-4", "Глава 4. А как быть агностикам?", ("ГЛАВА 4", "А КАК БЫТЬ АГНОСТИКАМ?")),
    ("chapter-5", "Глава 5. Программа в действии", ("ГЛАВА 5", "ПРОГРАММА В ДЕЙСТВИИ")),
    ("chapter-6", "Глава 6. За работу!", ("ГЛАВА 6", "ЗА РАБОТУ!")),
    ("chapter-7", "Глава 7. Работая с другими", ("ГЛАВА 7", "РАБОТАЯ С ДРУГИМИ")),
    ("chapter-8", "Глава 8. Обращение к женам", ("ГЛАВА 8", "ОБРАЩЕНИЕ К ЖЕНАМ")),
    ("chapter-9", "Глава 9. Новые отношения в семье", ("ГЛАВА 9", "НОВЫЕ ОТНОШЕНИЯ В СЕМЬЕ")),
    (
        "chapter-10",
        "Глава 10. Обращение к работодателям",
        ("ГЛАВА 10", "ОБРАЩЕНИЕ К РАБОТОДАТЕЛЯМ"),
    ),
    (
        "chapter-11",
        "Глава 11. Заглянем в ваше будущее",
        ("ГЛАВА 11", "ЗАГЛЯНЕМ В ВАШЕ БУДУЩЕЕ"),
    ),
)

SECTION_IDS = tuple(item[0] for item in REQUIRED_SECTIONS)

# Fixed control passages across beginning/middle/end. Short validation
# excerpts only (not a substitute copy); each must be present verbatim in
# the expected section or the build fails closed.
CONTROL_PASSAGES = (
    (
        "doctors-opinion",
        "Мы, члены Сообщества Анонимных Алкоголиков, полагаем, "
        "что читателю будет интересно познакомиться с медицинской оценкой",
    ),
    (
        "chapter-5",
        "Приняли решение препоручить нашу волю и нашу жизнь Богу, как мы Его понимали",
    ),
    (
        "chapter-11",
        "Да благословит вас Господь и да хранит вас ныне и присно",
    ),
)

DELIMITER_RE = re.compile(r"^@@SECTION:([a-z0-9-]+)\|(.+)@@$")


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> int:
    """Report a fail-closed build error and return the exit status."""
    print(f"canonical ru build failed: {message}", file=sys.stderr)
    return 1


def decode_txt(raw: bytes) -> tuple[str, bool]:
    """Decode raw TXT deterministically as UTF-8, stripping one leading BOM."""
    had_bom = raw.startswith(b"\xef\xbb\xbf")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"aa-big-book.txt is not valid UTF-8: {exc}") from exc
    if "\x00" in text:
        raise ValueError("aa-big-book.txt contains NUL bytes; refusing a corrupt source")
    return text, had_bom


def normalize_text(text: str) -> str:
    """Normalize only transport artifacts: NFC plus newline convention."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def split_sections(normalized: str) -> tuple[str, list[tuple[str, str, str]]]:
    """Split the normalized TXT into preamble plus 12 ordered sections."""
    lines = normalized.split("\n")
    preamble_lines: list[str] = []
    sections: list[tuple[str, str, list[str]]] = []
    current_id: str | None = None
    current_title: str | None = None
    current_body: list[str] = []
    seen_before_first = True
    for line in lines:
        match = DELIMITER_RE.match(line.strip())
        if match:
            if current_id is not None:
                sections.append((current_id, current_title or "", "\n".join(current_body)))
            current_id, current_title = match.group(1), match.group(2).strip()
            current_body = []
            seen_before_first = False
            continue
        if seen_before_first:
            preamble_lines.append(line)
        elif current_id is None:
            raise ValueError("text before the first section delimiter")
        else:
            current_body.append(line)
    if current_id is None:
        raise ValueError("no section delimiters found in aa-big-book.txt")
    sections.append((current_id, current_title or "", "\n".join(current_body)))
    return "\n".join(preamble_lines), sections


def validate_preamble(preamble: str) -> None:
    """Validate the Fourth Edition / 2013 / ISBN identity in the preamble."""
    missing = [marker for marker in IDENTITY_MARKERS if marker not in preamble]
    if missing:
        raise ValueError(f"edition identity mismatch in TXT preamble: missing {missing!r}")


def validate_sections(sections: list[tuple[str, str, str]]) -> list[dict[str, object]]:
    """Validate order, titles, headings, and control passages of sections."""
    if [(sid, title) for sid, title, _ in sections] != [
        (sid, title) for sid, title, _ in REQUIRED_SECTIONS
    ]:
        got = [(sid, title) for sid, title, _ in sections]
        want = [(sid, title) for sid, title, _ in REQUIRED_SECTIONS]
        raise ValueError(
            f"sections are not exactly Мнение доктора + Chapters 1-11: {got!r} != {want!r}"
        )
    validated: list[dict[str, object]] = []
    for (section_id, title, body), (_, _, headings) in zip(
        sections, REQUIRED_SECTIONS, strict=True
    ):
        stripped = body.strip("\n")
        if not stripped.strip():
            raise ValueError(f"section {section_id!r} decoded to blank text")
        # Headings are the first non-empty lines of the section body. The
        # chapter-8 source heading carries a footnote asterisk that is not
        # literary wording, so it is accepted with or without it.
        body_lines = [line for line in stripped.split("\n") if line.strip()]
        for position, heading in enumerate(headings):
            if position >= len(body_lines):
                raise ValueError(f"section {section_id!r} is missing heading {heading!r}")
            candidate = body_lines[position].strip()
            if section_id == "chapter-8" and heading == "ОБРАЩЕНИЕ К ЖЕНАМ":
                if candidate.rstrip("*").strip() != heading:
                    raise ValueError(f"section {section_id!r} heading mismatch: {candidate!r}")
                continue
            if candidate != heading:
                raise ValueError(f"section {section_id!r} heading mismatch: {candidate!r}")
        validated.append({"id": section_id, "title": title, "body": stripped})
    by_id = {item["id"]: str(item["body"]) for item in validated}
    for section_id, passage in CONTROL_PASSAGES:
        if passage not in by_id.get(section_id, ""):
            raise ValueError(f"control passage missing in {section_id!r}; refusing drift")
    return validated


def build_canonical_text(sections: list[dict[str, object]]) -> tuple[str, list[dict[str, object]]]:
    """Build the deterministic canonical.ru.txt plus per-section offsets."""
    chunks: list[str] = []
    enriched: list[dict[str, object]] = []
    offset = 0
    for index, item in enumerate(sections):
        section_id = str(item["id"])
        title = str(item["title"])
        body = str(item["body"])
        delimiter = f"@@SECTION:{section_id}|{title}@@"
        block = f"{delimiter}\n\n{body.strip()}\n"
        if index > 0:
            block = "\n" + block
        start = offset
        chunks.append(block)
        offset += len(block)
        enriched.append(
            {
                "id": section_id,
                "title": title,
                "delimiter": delimiter,
                "char_start": start,
                "char_end": offset,
                "body": body.strip(),
            }
        )
    return "".join(chunks), enriched


def build_artifact(
    *,
    manifest: dict[str, object],
    fetch_state: dict[str, object],
    raw: bytes,
    normalized: str,
    preamble: str,
    sections: list[dict[str, object]],
    canonical_txt: str,
    enriched: list[dict[str, object]],
) -> dict[str, object]:
    """Build the canonical artifact dict, failing closed on any mismatch."""
    # Re-check the caller-derived identity inputs against the raw bytes so a
    # drifted preamble/normalized/sections/canonical view cannot succeed when
    # only ``enriched`` is consistent.
    validate_preamble(preamble)
    decoded_check, _ = decode_txt(raw)
    if normalize_text(decoded_check) != normalized:
        raise ValueError("normalized text mismatch against raw TXT")
    preamble_check, raw_sections_check = split_sections(normalized)
    if preamble_check != preamble:
        raise ValueError("preamble mismatch against normalized TXT")
    revalidated = validate_sections(raw_sections_check)
    if revalidated != sections:
        raise ValueError("sections mismatch against normalized TXT")
    rebuilt_txt, rebuilt_enriched = build_canonical_text(sections)
    if rebuilt_txt != canonical_txt:
        raise ValueError("canonical text mismatch against sections")
    if rebuilt_enriched != enriched:
        raise ValueError("enriched sections mismatch against canonical text")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValueError("manifest builder_version is not supported by this builder")
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("manifest format is not supported by this builder")
    state_sources = fetch_state.get("sources")
    if not isinstance(state_sources, list) or not state_sources:
        raise ValueError("fetch-ru-state has no sources")
    recorded = state_sources[0]
    if not isinstance(recorded, dict):
        raise ValueError("fetch-ru-state source entry is malformed")
    digest = sha256(raw)
    if recorded.get("sha256") != digest:
        raise ValueError("raw TXT checksum mismatch against fetch-ru-state")
    if recorded.get("bytes") != len(raw):
        raise ValueError("raw TXT byte length mismatch against fetch-ru-state")
    if recorded.get("path") != RAW_PATH:
        raise ValueError(f"raw TXT path mismatch: {recorded.get('path')!r}")
    manifest_sources = manifest.get("sources")
    if not isinstance(manifest_sources, list) or not manifest_sources:
        raise ValueError("manifest has no sources")
    expected_source = manifest_sources[0]
    if not isinstance(expected_source, dict):
        raise ValueError("manifest source entry is malformed")
    if expected_source.get("sha256") != digest:
        raise ValueError("raw TXT checksum mismatch against manifest (source drift)")
    if expected_source.get("bytes") != len(raw):
        raise ValueError("raw TXT byte length mismatch against manifest")
    if expected_source.get("raw_path") != RAW_PATH:
        raise ValueError("manifest raw_path mismatch")

    manifest_sections = manifest.get("sections")
    if not isinstance(manifest_sections, list):
        raise ValueError("manifest has no sections list")
    expected_by_id = {str(item["id"]): item for item in manifest_sections if isinstance(item, dict)}
    if sorted(expected_by_id) != sorted(SECTION_IDS):
        raise ValueError("manifest sections are not exactly Мнение доктора + Chapters 1-11")

    artifact_sections: list[dict[str, object]] = []
    for item in enriched:
        section_id = str(item["id"])
        expected = expected_by_id.get(section_id)
        if not isinstance(expected, dict):
            raise ValueError(f"manifest has no expectations for {section_id!r}")
        if expected.get("title") != item["title"]:
            raise ValueError(f"{section_id} title mismatch against manifest")
        body = str(item["body"])
        text_sha = sha256(body.encode("utf-8"))
        if expected.get("text_sha256") != text_sha:
            raise ValueError(f"{section_id} derived text mismatch against manifest")
        artifact_sections.append(
            {
                "id": section_id,
                "title": item["title"],
                "source_id": SOURCE_ID,
                "source_url": SOURCE_URL,
                "source_file": RAW_PATH,
                "source_sha256": digest,
                "char_start": item["char_start"],
                "char_end": item["char_end"],
                "text_sha256": text_sha,
                "chars": len(body),
                "text": body,
            }
        )
    return {
        "format": ARTIFACT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "language": "ru",
        "edition": manifest.get("edition"),
        "sources": [
            {
                "id": SOURCE_ID,
                "file": RAW_PATH,
                "sha256": digest,
                "bytes": len(raw),
            }
        ],
        "sections": artifact_sections,
    }


def serialize_artifact(artifact: dict[str, object]) -> bytes:
    """Serialize the artifact deterministically (sorted keys, UTF-8)."""
    return (json.dumps(artifact, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: validate the TXT and write the RU canonical artifacts."""
    parser = argparse.ArgumentParser(description="Build the Russian canonical artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fetch-state", type=Path, default=DEFAULT_FETCH_STATE)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        fetch_state = json.loads(args.fetch_state.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return fail(f"required input file is missing: {exc.filename}")
    except json.JSONDecodeError as exc:
        return fail(f"required input file is not valid JSON: {exc}")

    try:
        raw = args.raw.read_bytes()
    except FileNotFoundError as exc:
        return fail(f"raw TXT is missing; run scripts/fetch_ru_source.py first: {exc}")

    try:
        decoded, _ = decode_txt(raw)
        normalized = normalize_text(decoded)
        preamble, raw_sections = split_sections(normalized)
        validate_preamble(preamble)
        sections = validate_sections(raw_sections)
        canonical_txt, enriched = build_canonical_text(sections)
        artifact = build_artifact(
            manifest=manifest,
            fetch_state=fetch_state,
            raw=raw,
            normalized=normalized,
            preamble=preamble,
            sections=sections,
            canonical_txt=canonical_txt,
            enriched=enriched,
        )
    except ValueError as exc:
        return fail(str(exc))

    payload = serialize_artifact(artifact)
    expected_artifact = manifest.get("artifact_sha256")
    if expected_artifact != sha256(payload):
        return fail(f"artifact checksum mismatch: expected={expected_artifact!r}")
    expected_text = manifest.get("text_artifact_sha256")
    canonical_txt_bytes = canonical_txt.encode("utf-8")
    if expected_text != sha256(canonical_txt_bytes):
        return fail("canonical text checksum mismatch against manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    if args.output.read_bytes() != payload:
        return fail("written JSON artifact differs from the validated payload")
    args.text_output.parent.mkdir(parents=True, exist_ok=True)
    args.text_output.write_bytes(canonical_txt_bytes)
    if args.text_output.read_bytes() != canonical_txt_bytes:
        return fail("written text artifact differs from the validated payload")

    print(
        json.dumps(
            {
                "sections": len(artifact["sections"]),  # type: ignore[arg-type]
                "artifact_sha256": sha256(payload),
                "artifact_bytes": len(payload),
                "text_sha256": sha256(canonical_txt_bytes),
                "text_bytes": len(canonical_txt_bytes),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
