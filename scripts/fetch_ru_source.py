#!/usr/bin/env python3
"""Acquire the Russian Fourth Edition TXT (issue #50).

Single canonical acquisition path for Russian: the normalized TXT file
``corpus/source/raw-ru/aa-big-book.txt``. The canonical build
(``scripts/build_canonical_ru.py``) reads only this TXT; PDF extraction and
OCR are never part of the canonical path.

Modes:

- reuse (default): when the TXT already exists in the trusted runtime, it is
  preserved byte-for-byte, validated for Fourth Edition / 2013 / ISBN
  identity, and recorded (SHA-256, bytes) in the ignored
  ``corpus/source/fetch-ru-state.json``. No network is used.
- ``--bootstrap-from-provider``: one-time trusted bootstrap that downloads
  the text-native provider pages pinned by ``corpus/source.ru.lock.json``,
  validates edition markers and section identity, extracts only the Russian
  literary text deterministically, and writes ``aa-big-book.txt`` once.
  Subsequent runs reuse the TXT without network.

Raw bytes are never modified in place; provenance and SHA-256 are recorded.
Logs contain only paths, sizes, and digests, never literary text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "corpus" / "source.ru.lock.json"
RAW_TXT = ROOT / "corpus" / "source" / "raw-ru" / "aa-big-book.txt"
STATE = ROOT / "corpus" / "source" / "fetch-ru-state.json"

USER_AGENT = "kodmial-aa-corpus-fetch-ru/1.0"
REQUEST_TIMEOUT_SECONDS = 25
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
MAX_BYTES_PER_FILE = 5 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

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

# (provider section, required page title, required first heading).
BOOTSTRAP_SECTIONS = (
    ("nXXVII", "Мнение доктора", "МНЕНИЕ ДОКТОРА"),
    ("n1", "Глава 1. Рассказ Билла", "ГЛАВА 1"),
    ("n16", "Глава 2. Выход есть", "ГЛАВА 2"),
    ("n29", "Глава 3. Еще об алкоголизме", "ГЛАВА 3"),
    ("n43", "Глава 4. А как быть агностикам?", "ГЛАВА 4"),
    ("n56", "Глава 5. Программа в действии", "ГЛАВА 5"),
    ("n70", "Глава 6. За работу!", "ГЛАВА 6"),
    ("n86", "Глава 7. Работая с другими", "ГЛАВА 7"),
    ("n101", "Глава 8. Обращение к женам", "ГЛАВА 8"),
    ("n118", "Глава 9. Новые отношения в семье", "ГЛАВА 9"),
    ("n132", "Глава 10. Обращение к работодателям", "ГЛАВА 10"),
    ("n147", "Глава 11. Заглянем в ваше будущее", "ГЛАВА 11"),
)

# Canonical delimiter written into aa-big-book.txt (never collides with prose).
DELIMITER_RE = re.compile(r"^@@SECTION:([a-z0-9-]+)\|(.+)@@$")
PART_DIVIDER_RE = re.compile(r"^ЧАСТЬ\s+\d+\.?$")

BOOK_INITIAL_RE = re.compile(
    r'<script\s+type="application/json"\s+id="book-initial">(.*?)</script>', re.S
)


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def _fail(message: str) -> int:
    print(f"ru source fetch failed: {message}", file=sys.stderr)
    return 1


def check_identity_markers(text: str, *, where: str) -> None:
    """Fail closed unless every Fourth Edition identity marker is present."""
    missing = [marker for marker in IDENTITY_MARKERS if marker not in text]
    if missing:
        raise ValueError(f"edition identity mismatch at {where}: missing {missing!r}")


class _PlainText(HTMLParser):
    """Strip tags from one Russian block, preserving ``<br>`` line breaks."""

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
        raw = "".join(self._chunks)
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in raw.split("\n")]
        return "\n".join(line for line in lines if line)


def html_to_text(fragment: str) -> str:
    """Convert one Russian ``html`` fragment to plain literary text."""
    parser = _PlainText()
    parser.feed(fragment)
    parser.close()
    return parser.text()


def normalize_text(text: str) -> str:
    """Normalize only transport artifacts: NFC plus newline convention."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))


def _download(url: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"unexpected HTTP status {status}")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES_PER_FILE:
                        raise RuntimeError(f"response exceeds size limit: {url}")
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RuntimeError(f"section is missing (HTTP 404): {url}") from exc
            if exc.code is not None and 500 <= exc.code < 600 and attempt < MAX_RETRIES - 1:
                last_error = exc
            else:
                raise RuntimeError(f"fetch failed for {url}: {exc}") from exc
        except Exception as exc:
            last_error = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"fetch failed after {MAX_RETRIES} attempts for {url}: {last_error}")


def _parse_section_payload(page_html: bytes, *, section: str) -> dict[str, object]:
    try:
        document = page_html.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"section {section!r} is not valid UTF-8: {exc}") from exc
    matches = BOOK_INITIAL_RE.findall(document)
    if len(matches) != 1:
        raise ValueError(f"section {section!r} has {len(matches)} payloads (expected 1)")
    try:
        payload = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"section {section!r} payload is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("section"), dict):
        raise ValueError(f"section {section!r} payload has no section object")
    result = payload["section"]
    if not isinstance(result, dict):
        raise ValueError(f"section {section!r} payload has no section object")
    return result


def _extract_blocks(
    page_html: bytes, *, section: str, title: str, heading: str
) -> tuple[list[str], list[str]]:
    payload = _parse_section_payload(page_html, section=section)
    if payload.get("id") != section:
        raise ValueError(f"section id mismatch: expected={section!r}")
    if payload.get("title") != title:
        raise ValueError(f"section {section!r} title mismatch: {payload.get('title')!r}")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError(f"section {section!r} has no content blocks")
    texts: list[str] = []
    seen: set[str] = set()
    for entry in raw_blocks:
        if not isinstance(entry, dict):
            raise ValueError(f"section {section!r} has a malformed block")
        block_id = entry.get("id")
        fragment = entry.get("html")
        if not isinstance(block_id, str) or not block_id.startswith(f"{section}-p"):
            raise ValueError(f"section {section!r} unexpected block id: {block_id!r}")
        if block_id in seen:
            raise ValueError(f"section {section!r} duplicate block id: {block_id!r}")
        seen.add(block_id)
        if not isinstance(fragment, str) or not fragment.strip():
            raise ValueError(f"section {section!r} block {block_id!r} has no html")
        text = html_to_text(fragment)
        if text:
            texts.append(text)
    first_lines = [line.strip() for text in texts[:3] for line in text.split("\n") if line.strip()]
    if not texts or heading not in first_lines:
        raise ValueError(f"required heading {heading!r} is missing in {section!r}")
    dropped: list[str] = []
    if section == "n147":
        while texts and PART_DIVIDER_RE.match(texts[-1]):
            dropped.append(texts.pop())
    elif any(PART_DIVIDER_RE.match(line) for text in texts for line in text.split("\n")):
        raise ValueError(f"section {section!r} contains an unexpected part divider")
    if not texts:
        raise ValueError(f"section {section!r} has no Russian text")
    return texts, dropped


def _extract_edition_text(page_html: bytes) -> str:
    payload = _parse_section_payload(page_html, section="edition")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ValueError("edition page has no content blocks")
    texts: list[str] = []
    for entry in raw_blocks:
        if not isinstance(entry, dict):
            raise ValueError("edition page has a malformed block")
        fragment = entry.get("html")
        if not isinstance(fragment, str):
            raise ValueError("edition page block has no html")
        text = html_to_text(fragment)
        if text:
            texts.append(text)
    edition_text = "\n\n".join(texts)
    check_identity_markers(edition_text, where="edition page")
    return edition_text


def assemble_txt(*, edition_text: str, sections: list[tuple[str, str, list[str]]]) -> bytes:
    """Assemble the normalized ``aa-big-book.txt`` bytes deterministically."""
    parts: list[str] = [edition_text.strip()]
    for section_id, title, blocks in sections:
        body = "\n\n".join(block.strip() for block in blocks if block.strip())
        if not body:
            raise ValueError(f"section {section_id!r} assembled to blank text")
        parts.append(f"@@SECTION:{section_id}|{title}@@\n\n{body}")
    assembled = normalize_text("\n\n".join(parts).strip() + "\n")
    check_identity_markers(assembled, where="assembled TXT")
    return assembled.encode("utf-8")


def _write_state(*, raw_sha: str, raw_bytes_len: int) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps(
            {
                "version": 1,
                "language": "ru",
                "sources": [
                    {
                        "id": "ru-fourth-edition-txt",
                        "path": "corpus/source/raw-ru/aa-big-book.txt",
                        "bytes": raw_bytes_len,
                        "sha256": raw_sha,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _reuse_existing() -> int:
    try:
        raw = RAW_TXT.read_bytes()
    except FileNotFoundError:
        return _fail(
            "aa-big-book.txt is missing in the trusted runtime; "
            "provision it or run with --bootstrap-from-provider"
        )
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return _fail(f"aa-big-book.txt is not valid UTF-8: {exc}")
    try:
        check_identity_markers(normalize_text(text), where="aa-big-book.txt")
    except ValueError as exc:
        return _fail(str(exc))
    digest = sha256(raw)
    _write_state(raw_sha=digest, raw_bytes_len=len(raw))
    print(json.dumps({"id": "ru-fourth-edition-txt", "bytes": len(raw), "sha256": digest}))
    return 0


def _bootstrap() -> int:
    # Preserve-bytes: never re-download over a trusted preserved TXT.
    # Reuse it without network; intentional requalification deletes the
    # file first and then bootstraps explicitly.
    if RAW_TXT.exists():
        return _fail(
            "aa-big-book.txt already exists; refusing to overwrite preserved bytes "
            "(run without --bootstrap-from-provider to reuse, or delete the file "
            "for intentional requalification)"
        )
    try:
        config = json.loads(LOCK.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _fail(f"lock file is missing: {LOCK}")
    except json.JSONDecodeError as exc:
        return _fail(f"lock file is not valid JSON: {exc}")
    provider = config.get("provider", {}) if isinstance(config, dict) else {}
    root = str(provider.get("edition_root", "https://aarus.fi/read/bigbook/"))
    if not root.endswith("/"):
        root += "/"
    try:
        edition_raw = _download(f"{root}edition/")
        edition_text = _extract_edition_text(edition_raw)
        sections: list[tuple[str, str, list[str]]] = []
        for provider_section, title, heading in BOOTSTRAP_SECTIONS:
            page_raw = _download(f"{root}{provider_section}/")
            blocks, _ = _extract_blocks(
                page_raw, section=provider_section, title=title, heading=heading
            )
            lock_sections = {
                item["section"]: item
                for item in config.get("sections", [])
                if isinstance(item, dict)
            }
            canonical = lock_sections.get(provider_section)
            if canonical is None:
                return _fail(f"lock has no section for provider page {provider_section!r}")
            sections.append((str(canonical["id"]), str(canonical["title"]), blocks))
        expected_ids = ["doctors-opinion"] + [f"chapter-{i}" for i in range(1, 12)]
        if [sid for sid, _, _ in sections] != expected_ids:
            raise ValueError(
                "assembled sections are not exactly "
                f"doctors-opinion + chapters 1-11: {[sid for sid, _, _ in sections]!r}"
            )
        assembled = assemble_txt(edition_text=edition_text, sections=sections)
    except (RuntimeError, ValueError) as exc:
        return _fail(str(exc))
    RAW_TXT.parent.mkdir(parents=True, exist_ok=True)
    tmp = RAW_TXT.with_name(RAW_TXT.name + ".tmp")
    tmp.write_bytes(assembled)
    if tmp.read_bytes() != assembled:
        try:
            tmp.unlink()
        except OSError:
            pass
        return _fail("assembled TXT changed while writing")
    tmp.replace(RAW_TXT)
    _write_state(raw_sha=sha256(assembled), raw_bytes_len=len(assembled))
    print(
        json.dumps(
            {
                "id": "ru-fourth-edition-txt",
                "bytes": len(assembled),
                "sha256": sha256(assembled),
                "bootstrap": "provider",
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: reuse the TXT or bootstrap it once from the provider."""
    parser = argparse.ArgumentParser(description="Acquire the Russian Fourth Edition TXT.")
    parser.add_argument(
        "--bootstrap-from-provider",
        action="store_true",
        help="Assemble aa-big-book.txt once from the text-native provider pages.",
    )
    args = parser.parse_args(argv)
    if args.bootstrap_from_provider:
        return _bootstrap()
    return _reuse_existing()


if __name__ == "__main__":
    raise SystemExit(main())
