#!/usr/bin/env python3
"""Build the deterministic Russian canonical AA runtime artifact (issue #50).

The builder reuses the ``corpus/source.ru.lock.json`` acquisition path and
never downloads anything itself:

- ``scripts/fetch_ru_source.py`` populates ``corpus/source/raw-ru/`` unchanged;
- ``corpus/source/fetch-ru-state.json`` records the fetched SHA-256 values.

This script reads those raw files read-only, validates them against the
committed ``corpus/canonical.ru.manifest.json`` expectations (failing closed
on any stale/mismatched source, edition mismatch, or unexpected page
structure), and writes exactly one artifact,
``corpus/generated/canonical.ru.json``, containing only:

- Мнение доктора (``nXXVII``);
- Chapters 1-11 (``n1``, ``n16``, ``n29``, ``n43``, ``n56``, ``n70``,
  ``n86``, ``n101``, ``n118``, ``n132``, ``n147``).

Only the Russian ``html`` field of each ``book-initial`` block becomes
canonical text. English parallel text (``english``), alternative Russian
renderings (``alternatives``), page markers, navigation, and UI chrome are
discarded and never enter the artifact. No paraphrase, translation,
summarization, or linguistic normalization takes place: tags are stripped,
entities are unescaped, and whitespace is collapsed deterministically while
the wording itself is untouched.

The full Russian text is never committed to Git: the repository stores this
builder plus the manifest (checksums, ranges, provenance), while the artifact
itself lives in the ignored ``corpus/generated/`` workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.ru.manifest.json"
DEFAULT_FETCH_STATE = ROOT / "corpus" / "source" / "fetch-ru-state.json"
DEFAULT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.ru.json"
DEFAULT_RAW_DIR = ROOT / "corpus" / "source" / "raw-ru"

BUILDER_VERSION = 1
ARTIFACT_FORMAT = "aa-canonical-ru/1"
MANIFEST_FORMAT = "aa-canonical-manifest-ru/1"

# (canonical id, page section, JSON section id, JSON title, page,
#  first heading, second heading or None)
REQUIRED_SECTIONS = (
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

SECTION_IDS = tuple(item[0] for item in REQUIRED_SECTIONS)

EDITION_MARKERS = (
    "АНОНИМНЫЕ АЛКОГОЛИКИ",
    "4-е издание",
    "Перевод с английского",
    "Alcoholics Anonymous World Services",
    "Фонд «Единство», 2013",
    "978-5-906531-01-8",
    "с разрешения Alcoholics Anonymous World Services",
)

# A trailing "ЧАСТЬ N" divider opens the excluded stories part, not Chapter 11.
PART_DIVIDER = re.compile(r"^ЧАСТЬ\s+\d+\.?$")

BOOK_INITIAL = re.compile(
    r'<script\s+type="application/json"\s+id="book-initial">(.*?)</script>', re.S
)


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> int:
    """Report a fail-closed build error and return the exit status."""
    print(f"canonical ru build failed: {message}", file=sys.stderr)
    return 1


class _RussianText(HTMLParser):
    """Strip tags from a Russian block, preserving ``<br>`` line breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._chunks.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br":
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        self._chunks.append(data)

    def text(self) -> str:
        """Return the deterministic plain-text rendering of the block."""
        raw = html.unescape("".join(self._chunks))
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        lines = [line for line in lines if line]
        return "\n".join(lines)


def html_to_text(fragment: str) -> str:
    """Convert one Russian ``html`` fragment to plain literary text."""
    parser = _RussianText()
    parser.feed(fragment)
    parser.close()
    return parser.text()


def parse_section_payload(page_html: bytes, *, section: str) -> dict[str, object]:
    """Parse the single embedded ``book-initial`` section payload."""
    try:
        document = page_html.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"section {section!r} source is not valid UTF-8: {exc}") from exc
    matches = BOOK_INITIAL.findall(document)
    if len(matches) != 1:
        raise ValueError(
            f"section {section!r} has {len(matches)} embedded book payloads (expected 1)"
        )
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"section {section!r} payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("section"), dict):
        raise ValueError(f"section {section!r} payload has no section object")
    result = payload["section"]
    assert isinstance(result, dict)
    return result


def extract_section_blocks(
    page_html: bytes, *, section: str, title: str, page: str
) -> tuple[list[dict[str, object]], list[str]]:
    """Extract ordered Russian block texts for one section (RU ``html`` only).

    Returns ``(blocks, dropped_part_dividers)`` where each block carries its
    ``block_id``, ``page``, and ``text``. English parallel text, alternative
    renderings, page markers, and UI chrome never enter the result.
    """
    payload = parse_section_payload(page_html, section=section)
    if payload.get("id") != section:
        raise ValueError(f"section id mismatch: expected={section!r} actual={payload.get('id')!r}")
    if payload.get("title") != title:
        raise ValueError(
            f"section {section!r} title mismatch: "
            f"expected={title!r} actual={payload.get('title')!r}"
        )
    if payload.get("page") != page:
        raise ValueError(
            f"section {section!r} page mismatch: expected={page!r} actual={payload.get('page')!r}"
        )
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"section {section!r} has no content blocks")
    if not isinstance(raw_blocks[0], dict):
        raise ValueError(f"section {section!r} has a malformed block")

    blocks: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in raw_blocks:
        if not isinstance(entry, dict):
            raise ValueError(f"section {section!r} has a malformed block")
        block_id = entry.get("id")
        fragment = entry.get("html")
        if not isinstance(block_id, str) or not block_id.startswith(f"{section}-p"):
            raise ValueError(f"section {section!r} has an unexpected block id: {block_id!r}")
        if block_id in seen:
            raise ValueError(f"section {section!r} has a duplicate block id: {block_id!r}")
        seen.add(block_id)
        if not isinstance(fragment, str) or not fragment.strip():
            raise ValueError(f"section {section!r} block {block_id!r} has no Russian html")
        text = html_to_text(fragment)
        if not text:
            raise ValueError(f"section {section!r} block {block_id!r} decoded to blank text")
        block_page = entry.get("page")
        blocks.append(
            {
                "block_id": block_id,
                "page": block_page if isinstance(block_page, str) else "",
                "text": text,
            }
        )

    # A trailing "ЧАСТЬ N" divider opens the excluded stories part; it is not
    # Chapter 11 literary text and is dropped deterministically.
    dropped: list[str] = []
    if section == "n147":
        while blocks and PART_DIVIDER.match(str(blocks[-1]["text"])):
            dropped.append(str(blocks.pop()["block_id"]))
    elif any(PART_DIVIDER.match(str(block["text"])) for block in blocks):
        raise ValueError(f"section {section!r} contains an unexpected part divider")
    if not blocks:
        raise ValueError(f"section {section!r} has no Russian text blocks")
    return blocks, dropped


def check_headings(
    blocks: list[dict[str, object]], *, first: str, second: str | None, section: str
) -> None:
    """Fail closed unless the section opens with its exact required headings."""
    if str(blocks[0]["text"]) != first:
        raise ValueError(
            f"section {section!r} first heading mismatch: "
            f"expected={first!r} actual={str(blocks[0]['text'])[:80]!r}"
        )
    if second is not None:
        if len(blocks) < 2 or str(blocks[1]["text"]) != second:
            actual = str(blocks[1]["text"])[:80] if len(blocks) > 1 else None
            raise ValueError(
                f"section {section!r} second heading mismatch: "
                f"expected={second!r} actual={actual!r}"
            )


def check_source(
    *,
    source_id: str,
    raw_path: str,
    data: bytes,
    manifest_sources: dict[str, dict[str, object]],
    fetch_state: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Validate one raw source against manifest and fetch-state (fail closed)."""
    expected = manifest_sources.get(source_id)
    if expected is None:
        raise ValueError(f"manifest has no expectations for source {source_id!r}")
    recorded = fetch_state.get(source_id)
    if recorded is None:
        raise ValueError(f"fetch-state has no record for source {source_id!r}")
    actual_digest = sha256(data)
    for label, entry in (("manifest", expected), ("fetch-state", recorded)):
        if entry.get("sha256") != actual_digest:
            raise ValueError(
                f"source {source_id!r} checksum mismatch against {label}: "
                f"expected={entry.get('sha256')!r} actual={actual_digest!r}"
            )
        if entry.get("bytes") != len(data):
            raise ValueError(f"source {source_id!r} byte length mismatch against {label}")
    if expected.get("raw_path") != raw_path or recorded.get("path") != raw_path:
        raise ValueError(f"source {source_id!r} path mismatch: {raw_path!r}")
    return expected


def build_canonical_ru(
    *,
    manifest: dict[str, object],
    fetch_state: dict[str, object],
    edition_html: bytes,
    pages: dict[str, bytes],
) -> dict[str, object]:
    """Build the Russian canonical artifact dict, failing closed on mismatch."""
    if manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"manifest format is not {MANIFEST_FORMAT!r}")
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValueError("manifest builder_version is not supported by this builder")
    if manifest.get("language") != "ru":
        raise ValueError("manifest language is not Russian")

    try:
        edition_text = edition_html.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"edition source is not valid UTF-8: {exc}") from exc
    missing = [marker for marker in EDITION_MARKERS if marker not in edition_text]
    if missing:
        raise ValueError(f"edition mismatch: missing markers {missing!r}")

    manifest_sources = {
        str(item["id"]): item
        for item in manifest.get("sources", [])  # type: ignore[union-attr]
    }
    state_sources = {
        str(item["id"]): item
        for item in fetch_state.get("sources", [])  # type: ignore[union-attr]
    }
    check_source(
        source_id="edition",
        raw_path="corpus/source/raw-ru/edition.html",
        data=edition_html,
        manifest_sources=manifest_sources,
        fetch_state=state_sources,
    )

    manifest_sections = {str(item["id"]): item for item in manifest.get("sections", [])}  # type: ignore[union-attr]
    if sorted(manifest_sections) != sorted(SECTION_IDS):
        raise ValueError("manifest sections are not exactly doctors-opinion + chapters 1-11")

    sections: list[dict[str, object]] = []
    source_entries: list[dict[str, object]] = [
        {
            "id": "edition",
            "section": "edition",
            "url": str(manifest_sources["edition"].get("url")),
            "file": "corpus/source/raw-ru/edition.html",
            "sha256": sha256(edition_html),
            "bytes": len(edition_html),
        }
    ]

    for section_id, page_section, title, page, first, second in REQUIRED_SECTIONS:
        if section_id == "doctors-opinion":
            source_key = "doctors-opinion"
        else:
            source_key = section_id
        raw_path = f"corpus/source/raw-ru/{page_section}.html"
        data = pages.get(page_section)
        if data is None:
            raise ValueError(f"raw page is missing for section {page_section!r}")
        expectation = check_source(
            source_id=source_key,
            raw_path=raw_path,
            data=data,
            manifest_sources=manifest_sources,
            fetch_state=state_sources,
        )
        blocks, _dropped = extract_section_blocks(
            data, section=page_section, title=title, page=page
        )
        check_headings(blocks, first=first, second=second, section=page_section)

        expected_section = manifest_sections[section_id]
        if expected_section.get("title") != title:
            raise ValueError(f"{section_id} manifest title mismatch")
        manifest_block_ids = list(expected_section.get("block_ids", []))  # type: ignore[union-attr]
        actual_block_ids = [str(block["block_id"]) for block in blocks]
        if manifest_block_ids != actual_block_ids:
            raise ValueError(
                f"{section_id} block sequence mismatch: "
                f"expected {len(manifest_block_ids)} blocks, got {len(actual_block_ids)}"
            )

        text = f"{title}\n\n" + "\n\n".join(str(block["text"]) for block in blocks) + "\n"
        text_sha = sha256(text.encode("utf-8"))
        if expected_section.get("text_sha256") != text_sha:
            raise ValueError(
                f"{section_id} derived text mismatch: "
                f"expected={expected_section.get('text_sha256')!r} actual={text_sha!r}"
            )

        proven_blocks: list[dict[str, object]] = []
        offset = len(f"{title}\n\n")
        for block in blocks:
            block_text = str(block["text"])
            start = offset
            end = start + len(block_text)
            proven_blocks.append(
                {
                    "block_id": str(block["block_id"]),
                    "page": str(block["page"]),
                    "char_start": start,
                    "char_end": end,
                    "text_sha256": sha256(block_text.encode("utf-8")),
                }
            )
            offset = end + len("\n\n")
        # The trailing newline replaces the final block separator.
        if len(text) != offset - 1 or not text.endswith("\n"):
            raise ValueError(f"{section_id} offset accounting is inconsistent")

        sections.append(
            {
                "id": section_id,
                "title": title,
                "source_id": source_key,
                "source_section": page_section,
                "source_url": str(expectation.get("url")),
                "source_file": raw_path,
                "source_sha256": sha256(data),
                "page": page,
                "headings": [h for h in (first, second) if h is not None],
                "block_ids": actual_block_ids,
                "blocks": proven_blocks,
                "text_sha256": text_sha,
                "chars": len(text),
                "text": text,
            }
        )
        source_entries.append(
            {
                "id": source_key,
                "section": page_section,
                "url": str(expectation.get("url")),
                "file": raw_path,
                "sha256": sha256(data),
                "bytes": len(data),
            }
        )

    return {
        "format": ARTIFACT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "language": "ru",
        "edition": manifest.get("edition"),
        "sku": manifest.get("sku"),
        "rights_basis": manifest.get("rights_basis"),
        "sources": source_entries,
        "sections": sections,
    }


def serialize_artifact(artifact: dict[str, object]) -> bytes:
    """Serialize the artifact deterministically (sorted keys, UTF-8)."""
    return (json.dumps(artifact, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: validate Russian sources and write the RU artifact."""
    parser = argparse.ArgumentParser(description="Build the Russian canonical artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fetch-state", type=Path, default=DEFAULT_FETCH_STATE)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        fetch_state = json.loads(args.fetch_state.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return fail(f"required input file is missing: {exc.filename}")
    except json.JSONDecodeError as exc:
        return fail(f"required input file is not valid JSON: {exc}")

    raw_dir = Path(args.raw_dir)
    try:
        edition_html = (raw_dir / "edition.html").read_bytes()
    except FileNotFoundError as exc:
        return fail(f"raw Russian source is missing; run scripts/fetch_ru_source.py first: {exc}")
    pages: dict[str, bytes] = {}
    for _, page_section, *_ in REQUIRED_SECTIONS:
        try:
            pages[page_section] = (raw_dir / f"{page_section}.html").read_bytes()
        except FileNotFoundError as exc:
            return fail(
                f"raw Russian source is missing; run scripts/fetch_ru_source.py first: {exc}"
            )

    try:
        artifact = build_canonical_ru(
            manifest=manifest,
            fetch_state=fetch_state,
            edition_html=edition_html,
            pages=pages,
        )
    except ValueError as exc:
        return fail(str(exc))

    payload = serialize_artifact(artifact)
    expected_artifact = manifest.get("artifact_sha256")
    if expected_artifact != sha256(payload):
        return fail(
            f"artifact checksum mismatch: expected={expected_artifact!r} actual={sha256(payload)!r}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    if args.output.read_bytes() != payload:
        return fail("written artifact differs from the validated payload")

    print(
        json.dumps(
            {
                "language": "ru",
                "sections": len(artifact["sections"]),  # type: ignore[arg-type]
                "artifact_sha256": sha256(payload),
                "artifact_bytes": len(payload),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
