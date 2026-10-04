#!/usr/bin/env python3
"""Fetch the immutable Russian Big Book source artifacts (issue #50).

Reads ``corpus/source.ru.lock.json``, downloads every required section page
with bounded timeouts/retries and content-size limits, validates the Russian
Fourth Edition markers and the required section/headings *before* accepting
the bytes, and stores the raw downloads only under the ignored
``corpus/source/raw-ru/`` workspace. SHA-256 metadata goes to
``corpus/source/fetch-ru-state.json`` (also ignored by Git).

Logs contain only section ids, byte sizes, and SHA-256 digests — never the
literary text.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "corpus" / "source.ru.lock.json"
RAW_DIR = ROOT / "corpus" / "source" / "raw-ru"
STATE = ROOT / "corpus" / "source" / "fetch-ru-state.json"

USER_AGENT = "kodmial-aa-corpus-fetch-ru/1.0"
REQUEST_TIMEOUT_SECONDS = 20
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = (1, 2, 4)
MAX_BYTES_PER_FILE = 5 * 1024 * 1024
CHUNK_SIZE = 64 * 1024

# Edition markers that must all be present (in extracted Russian block text or
# in the raw page) before the fetched edition page is accepted.
EDITION_MARKERS = (
    "АНОНИМНЫЕ АЛКОГОЛИКИ",
    "4-е издание",
    "Перевод с английского",
    "Alcoholics Anonymous World Services",
    "Фонд «Единство», 2013",
    "978-5-906531-01-8",
    "с разрешения Alcoholics Anonymous World Services",
)

# Required ``<section id -> (page title, first heading)>`` expectations.
# Section page-ids are print-page based (n1, n16, n29, ...), not 1..11.
REQUIRED_SECTIONS = (
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


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    """Download ``url`` with bounded timeouts/retries and a size limit."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise RuntimeError(f"unexpected HTTP status {status} for {url}")
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_BYTES_PER_FILE:
                        raise RuntimeError(
                            f"response exceeds size limit ({MAX_BYTES_PER_FILE} bytes): {url}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except urllib.error.HTTPError as exc:
            # Missing/moved sections fail closed immediately; server errors retry.
            if exc.code == 404:
                raise RuntimeError(f"section is missing (HTTP 404): {url}") from exc
            if exc.code is not None and 500 <= exc.code < 600 and attempt < MAX_RETRIES - 1:
                last_error = exc
            else:
                raise RuntimeError(f"fetch failed for {url}: {exc}") from exc
        except Exception as exc:  # timeouts, DNS, connection resets
            last_error = exc
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    raise RuntimeError(f"fetch failed after {MAX_RETRIES} attempts for {url}: {last_error}")


def _decode(data: bytes, *, url: str) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"source at {url} is not valid UTF-8: {exc}") from exc


def validate_edition(data: bytes, *, url: str) -> None:
    """Fail closed unless the edition page carries every required marker."""
    text = _decode(data, url=url)
    missing = [marker for marker in EDITION_MARKERS if marker not in text]
    if missing:
        raise RuntimeError(f"edition mismatch at {url}: missing markers {missing!r}")


def validate_section(data: bytes, *, section: str, title: str, heading: str, url: str) -> None:
    """Fail closed on missing/moved sections or unexpected page structure."""
    text = _decode(data, url=url)
    match = re.search(
        r'<script\s+type="application/json"\s+id="book-initial">(.*?)</script>',
        text,
        re.S,
    )
    if match is None:
        raise RuntimeError(f"section payload is missing at {url}")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"section payload is not valid JSON at {url}: {exc}") from exc
    current = payload.get("section", {}) if isinstance(payload, dict) else {}
    if not isinstance(current, dict) or current.get("id") != section:
        raise RuntimeError(
            f"section payload mismatch at {url}: expected id {section!r}, got {current.get('id')!r}"
            if isinstance(current, dict)
            else "malformed"
        )
    if current.get("title") != title:
        raise RuntimeError(
            f"section title mismatch at {url}: expected {title!r}, got {current.get('title')!r}"
        )
    if heading not in text:
        raise RuntimeError(f"required heading {heading!r} is missing at {url}")


def main() -> int:
    """CLI entrypoint: fetch, validate, and record the Russian sources."""
    try:
        config = json.loads(LOCK.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"source fetch failed: lock file is missing: {LOCK}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"source fetch failed: lock file is not valid JSON: {exc}", file=sys.stderr)
        return 1

    required = [s for s in config.get("sources", []) if s.get("required", True)]
    if not required:
        print("source fetch failed: lock declares no required sources", file=sys.stderr)
        return 1

    expectations = {section: (title, heading) for section, title, heading in REQUIRED_SECTIONS}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    fetched = []
    try:
        for source in required:
            data = fetch(source["url"])
            target = ROOT / source["raw_path"]
            if target.resolve().parent != RAW_DIR.resolve():
                raise RuntimeError(f"refusing to write outside the raw-ru workspace: {target}")
            if source["id"] == "edition":
                validate_edition(data, url=source["url"])
            else:
                title, heading = expectations[source["section"]]
                validate_section(
                    data,
                    section=source["section"],
                    title=title,
                    heading=heading,
                    url=source["url"],
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError(f"raw source changed while writing: {target}")
            fetched.append(
                {
                    "id": source["id"],
                    "section": source["section"],
                    "url": source["url"],
                    "path": source["raw_path"],
                    "bytes": len(data),
                    "sha256": sha256(data),
                }
            )
    except RuntimeError as exc:
        print(f"source fetch failed: {exc}", file=sys.stderr)
        return 1

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(
        json.dumps({"version": 1, "language": "ru", "sources": fetched}, indent=2) + "\n",
        encoding="utf-8",
    )

    for item in fetched:
        print(f"{item['id']}: {item['bytes']} bytes sha256={item['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
