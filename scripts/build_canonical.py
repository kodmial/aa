#!/usr/bin/env python3
"""Build the deterministic canonical AA runtime artifact.

The builder reuses the PR #11 acquisition path and never downloads anything:

- ``corpus/source.lock.json`` declares the pinned source URLs;
- ``scripts/fetch_aa_source.py`` populates ``corpus/source/raw/`` unchanged;
- ``corpus/source/fetch-state.json`` records the fetched SHA-256 values.

This script reads those raw files read-only, validates them against the
committed ``corpus/canonical.manifest.json`` expectations (failing closed on
any stale or mismatched source), and writes exactly one artifact,
``corpus/generated/canonical.json``, containing only:

- The Doctor's Opinion (HTML converted to readable literary text
  deterministically, without paraphrase or any change of wording);
- Chapters 1-11 sliced deterministically from the plain-text source.

No shortening, summarization, semantic rewriting, translation, or LLM
curation takes place. The full book text is never committed to Git: the
repository stores this builder plus the manifest (checksums, ranges,
provenance), while the artifact itself lives in the ignored
``corpus/generated/`` workspace.
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
DEFAULT_MANIFEST = ROOT / "corpus" / "canonical.manifest.json"
DEFAULT_FETCH_STATE = ROOT / "corpus" / "source" / "fetch-state.json"
DEFAULT_OUTPUT = ROOT / "corpus" / "generated" / "canonical.json"

BUILDER_VERSION = 1
ARTIFACT_FORMAT = "aa-canonical/1"

DOCTORS_OPINION_ID = "doctors-opinion"
DOCTORS_OPINION_TITLE = "The Doctor's Opinion"
EXPECTED_PARAGRAPH_IDS = [f"p{i}" for i in range(1, 42)]

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

CHAPTER_HEADING = re.compile(rb"Chapter\s+(\d+)\s*\r?\n\s*\r?\n?\s*([^\r\n]+)")
PREAMBLE_MARKER = b"Anonymous Press"
CLOSING_MARKER = b"until then."


def sha256(data: bytes) -> str:
    """Return the hex SHA-256 digest of ``data``."""
    return hashlib.sha256(data).hexdigest()


def fail(message: str) -> int:
    """Report a fail-closed build error and return the exit status."""
    print(f"canonical build failed: {message}", file=sys.stderr)
    return 1


class _ArticleParagraphs(HTMLParser):
    """Collect ``<p id=...>`` texts inside ``<article>`` in document order."""

    def __init__(self) -> None:
        # Keep the default convert_charrefs=True so character and entity
        # references (e.g. &#x27;, &quot;) reach handle_data as real
        # characters. Dropping them would silently alter the wording, which
        # the canonical contract forbids.
        super().__init__()
        self._depth = 0
        self._current_id: str | None = None
        self._current_chunks: list[str] = []
        self.ids: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self._depth += 1
            return
        if self._depth == 0 or tag != "p":
            return
        raw_id: str | None = None
        for name, value in attrs:
            if name == "id":
                raw_id = value
        self._current_id = raw_id
        self._current_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self._depth = max(0, self._depth - 1)
            return
        if self._depth == 0 or tag != "p" or self._current_id is None:
            return
        text = collapse_whitespace("".join(self._current_chunks))
        self.ids.append(self._current_id)
        self.paragraphs.append(text)
        self._current_id = None
        self._current_chunks = []

    def handle_data(self, data: str) -> None:
        if self._depth > 0 and self._current_id is not None:
            self._current_chunks.append(data)


def collapse_whitespace(text: str) -> str:
    """Collapse whitespace runs deterministically (wording is untouched)."""
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def extract_doctors_opinion(html_bytes: bytes) -> tuple[str, list[str]]:
    """Convert the Doctor's Opinion HTML to readable literary text.

    Only ``<p id="pN">`` elements inside ``<article>`` are kept, in document
    order. HTML entities are unescaped and whitespace runs are collapsed so
    the result is readable; the wording itself is never altered.
    """
    try:
        document = html_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"doctors-opinion source is not valid UTF-8: {exc}") from exc
    if "<article>" not in document or "</article>" not in document:
        raise ValueError("doctors-opinion source has no <article> element")
    parser = _ArticleParagraphs()
    parser.feed(document)
    parser.close()
    if parser.ids != EXPECTED_PARAGRAPH_IDS:
        raise ValueError(
            f"doctors-opinion paragraphs are not exactly p1..p41 in order: {parser.ids!r}"
        )
    if any(not paragraph for paragraph in parser.paragraphs):
        raise ValueError("doctors-opinion contains an empty paragraph")
    text = f"{DOCTORS_OPINION_TITLE}\n\n" + "\n\n".join(parser.paragraphs) + "\n"
    return text, list(parser.ids)


def slice_chapters(source: bytes) -> list[dict[str, object]]:
    """Slice Chapters 1-11 deterministically from the plain-text source."""
    matches = list(CHAPTER_HEADING.finditer(source))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, 12)):
        raise ValueError(f"chapter headings are not exactly 1..11 in order: {numbers!r}")
    titles = [match.group(2).decode("utf-8", errors="strict").strip() for match in matches]
    if list(titles) != list(CHAPTER_TITLES):
        raise ValueError(f"chapter titles do not match the canonical scope: {titles!r}")

    preamble = source[: matches[0].start()]
    if PREAMBLE_MARKER not in preamble:
        raise ValueError("preamble marker missing; source scope cannot be validated")
    tail = source[matches[-1].start() :]
    if not source.endswith(CLOSING_MARKER + b"\n") and CLOSING_MARKER not in tail:
        raise ValueError("closing marker missing at the end of Chapter 11")

    sections: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        if end <= start:
            raise ValueError(f"chapter {index + 1} has an empty byte range")
        chunk = source[start:end]
        try:
            text = chunk.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"chapter {index + 1} is not valid UTF-8: {exc}") from exc
        if not text.strip():
            raise ValueError(f"chapter {index + 1} decoded to blank text")
        sections.append(
            {
                "number": index + 1,
                "title": titles[index],
                "byte_start": start,
                "byte_end": end,
                "text": text,
            }
        )
    # Chapters must tile the canonical span contiguously: no gaps, no overlaps.
    for previous, current in zip(sections, sections[1:], strict=False):
        if current["byte_start"] != previous["byte_end"]:
            raise ValueError("chapter ranges are not contiguous")
    if sections[0]["byte_start"] == 0:
        raise ValueError("preamble was not excluded from Chapter 1")
    return sections


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


def build_canonical(
    *,
    manifest: dict[str, object],
    fetch_state: dict[str, object],
    aa_bytes: bytes,
    opinion_html: bytes,
) -> dict[str, object]:
    """Build the canonical artifact dict, failing closed on any mismatch."""
    if manifest.get("builder_version") != BUILDER_VERSION:
        raise ValueError("manifest builder_version is not supported by this builder")

    manifest_sources = {
        str(item["id"]): item
        for item in manifest.get("sources", [])  # type: ignore[union-attr]
    }
    state_sources = {
        str(item["id"]): item
        for item in fetch_state.get("sources", [])  # type: ignore[union-attr]
    }
    aa_expectation = check_source(
        source_id="core-pages-1-164",
        raw_path="corpus/source/raw/AA.txt",
        data=aa_bytes,
        manifest_sources=manifest_sources,
        fetch_state=state_sources,
    )
    opinion_expectation = check_source(
        source_id="doctors-opinion",
        raw_path="corpus/source/raw/doctors-opinion.html",
        data=opinion_html,
        manifest_sources=manifest_sources,
        fetch_state=state_sources,
    )

    opinion_text, paragraph_ids = extract_doctors_opinion(opinion_html)
    chapters = slice_chapters(aa_bytes)

    manifest_sections = {str(item["id"]): item for item in manifest.get("sections", [])}  # type: ignore[union-attr]
    if sorted(manifest_sections) != sorted(
        ["doctors-opinion", *[f"chapter-{i}" for i in range(1, 12)]]
    ):
        raise ValueError("manifest sections are not exactly doctors-opinion + chapters 1-11")

    sections: list[dict[str, object]] = []
    opinion_section = manifest_sections["doctors-opinion"]
    if opinion_section.get("title") != DOCTORS_OPINION_TITLE:
        raise ValueError("manifest doctors-opinion title mismatch")
    if list(opinion_section.get("paragraphs", [])) != EXPECTED_PARAGRAPH_IDS:  # type: ignore[union-attr]
        raise ValueError("manifest doctors-opinion paragraphs mismatch")
    opinion_text_sha = sha256(opinion_text.encode("utf-8"))
    if opinion_section.get("text_sha256") != opinion_text_sha:
        raise ValueError(
            "doctors-opinion derived text mismatch: "
            f"expected={opinion_section.get('text_sha256')!r} actual={opinion_text_sha!r}"
        )
    sections.append(
        {
            "id": DOCTORS_OPINION_ID,
            "title": DOCTORS_OPINION_TITLE,
            "source_id": "doctors-opinion",
            "source_url": str(opinion_expectation.get("url")),
            "source_file": "corpus/source/raw/doctors-opinion.html",
            "source_sha256": sha256(opinion_html),
            "paragraph_ids": paragraph_ids,
            "text_sha256": opinion_text_sha,
            "chars": len(opinion_text),
            "text": opinion_text,
        }
    )

    for chapter in chapters:
        number = int(chapter["number"])  # type: ignore[arg-type]
        section_id = f"chapter-{number}"
        expected = manifest_sections[section_id]
        for key in ("byte_start", "byte_end", "title"):
            if expected.get(key) != chapter[key]:
                raise ValueError(
                    f"{section_id} {key} mismatch: "
                    f"expected={expected.get(key)!r} actual={chapter[key]!r}"
                )
        text = str(chapter["text"])
        text_sha = sha256(text.encode("utf-8"))
        if expected.get("text_sha256") != text_sha:
            raise ValueError(
                f"{section_id} derived text mismatch: "
                f"expected={expected.get('text_sha256')!r} actual={text_sha!r}"
            )
        sections.append(
            {
                "id": section_id,
                "title": str(chapter["title"]),
                "source_id": "core-pages-1-164",
                "source_url": str(aa_expectation.get("url")),
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": sha256(aa_bytes),
                "byte_start": chapter["byte_start"],
                "byte_end": chapter["byte_end"],
                "text_sha256": text_sha,
                "chars": len(text),
                "text": text,
            }
        )

    return {
        "format": ARTIFACT_FORMAT,
        "builder_version": BUILDER_VERSION,
        "sources": [
            {
                "id": "core-pages-1-164",
                "url": str(aa_expectation.get("url")),
                "file": "corpus/source/raw/AA.txt",
                "sha256": sha256(aa_bytes),
                "bytes": len(aa_bytes),
            },
            {
                "id": "doctors-opinion",
                "url": str(opinion_expectation.get("url")),
                "file": "corpus/source/raw/doctors-opinion.html",
                "sha256": sha256(opinion_html),
                "bytes": len(opinion_html),
            },
        ],
        "sections": sections,
    }


def serialize_artifact(artifact: dict[str, object]) -> bytes:
    """Serialize the artifact deterministically (sorted keys, UTF-8)."""
    return (json.dumps(artifact, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: validate sources and write the canonical artifact."""
    parser = argparse.ArgumentParser(description="Build the canonical AA runtime artifact.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--fetch-state", type=Path, default=DEFAULT_FETCH_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        fetch_state = json.loads(args.fetch_state.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        return fail(f"required input file is missing: {exc.filename}")
    except json.JSONDecodeError as exc:
        return fail(f"required input file is not valid JSON: {exc}")

    aa_path = ROOT / "corpus" / "source" / "raw" / "AA.txt"
    opinion_path = ROOT / "corpus" / "source" / "raw" / "doctors-opinion.html"
    try:
        aa_bytes = aa_path.read_bytes()
        opinion_html = opinion_path.read_bytes()
    except FileNotFoundError as exc:
        return fail(f"raw source is missing; run scripts/fetch_aa_source.py first: {exc}")

    try:
        artifact = build_canonical(
            manifest=manifest,
            fetch_state=fetch_state,
            aa_bytes=aa_bytes,
            opinion_html=opinion_html,
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
                "sections": len(artifact["sections"]),  # type: ignore[arg-type]
                "artifact_sha256": sha256(payload),
                "artifact_bytes": len(payload),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
