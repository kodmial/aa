"""Runtime access to the canonical AA artifact (no silent truncation).

The artifact is produced by ``scripts/build_canonical.py`` into the ignored
``corpus/generated/`` workspace; this module only reads it. Every load
re-verifies per-section SHA-256 digests, and every read enforces exact
bounds, so a corrupt artifact or an out-of-range request fails closed
instead of returning silently truncated text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

ARTIFACT_FORMAT = "aa-canonical/1"

EXPECTED_SECTION_IDS = (
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


class CanonicalCorpusError(ValueError):
    """Raised when the canonical artifact fails validation."""


class CanonicalRangeError(CanonicalCorpusError):
    """Raised when a read request exceeds section bounds (never truncated)."""


@dataclass(frozen=True)
class CanonicalSection:
    """One canonical section with provenance back to the raw source."""

    id: str
    title: str
    source_id: str
    source_url: str
    source_file: str
    text: str


@dataclass(frozen=True)
class CanonicalCorpus:
    """Validated canonical sections in canonical order."""

    sections: tuple[CanonicalSection, ...]

    def ids(self) -> tuple[str, ...]:
        """Return section ids in canonical order."""
        return tuple(section.id for section in self.sections)

    def get(self, section_id: str) -> CanonicalSection:
        """Return a section by id or raise :class:`CanonicalCorpusError`."""
        for section in self.sections:
            if section.id == section_id:
                return section
        raise CanonicalCorpusError(f"unknown canonical section: {section_id!r}")

    def read(self, section_id: str, start: int, end: int) -> str:
        """Return an exact ``[start:end)`` slice of a section's text.

        Bounds violations fail with :class:`CanonicalRangeError`; the text
        is never silently truncated.
        """
        section = self.get(section_id)
        if start < 0 or end < 0 or end < start:
            raise CanonicalRangeError(f"invalid range [{start}:{end}) for {section_id!r}")
        if end > len(section.text):
            raise CanonicalRangeError(
                f"range [{start}:{end}) exceeds section {section_id!r} length {len(section.text)}"
            )
        if end == start:
            raise CanonicalRangeError(f"refusing empty read [{start}:{end}) for {section_id!r}")
        return section.text[start:end]


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_canonical(path: str | Path, *, expected_sha256: str | None = None) -> CanonicalCorpus:
    """Load and validate the canonical artifact (fails closed on mismatch)."""
    raw_path = Path(path)
    try:
        payload = raw_path.read_bytes()
    except FileNotFoundError as exc:
        raise CanonicalCorpusError(f"canonical artifact is missing: {raw_path}") from exc
    try:
        artifact = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CanonicalCorpusError(f"canonical artifact is not valid JSON: {exc}") from exc

    if not isinstance(artifact, dict) or artifact.get("format") != ARTIFACT_FORMAT:
        raise CanonicalCorpusError(f"unsupported canonical artifact format: {raw_path}")
    if expected_sha256 is not None and _digest(payload) != expected_sha256:
        raise CanonicalCorpusError(f"canonical artifact checksum mismatch: {raw_path}")

    entries = artifact.get("sections")
    if not isinstance(entries, list):
        raise CanonicalCorpusError("canonical artifact has no sections list")
    sections: list[CanonicalSection] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise CanonicalCorpusError("canonical artifact has a malformed section")
        text = entry.get("text")
        if not isinstance(text, str) or not text:
            raise CanonicalCorpusError("canonical artifact has an empty section text")
        if _digest(text.encode("utf-8")) != entry.get("text_sha256"):
            raise CanonicalCorpusError(f"canonical section checksum mismatch: {entry.get('id')!r}")
        for key in ("id", "title", "source_id", "source_url", "source_file"):
            if not isinstance(entry.get(key), str) or not entry.get(key):
                raise CanonicalCorpusError(f"canonical section is missing {key}")
        sections.append(
            CanonicalSection(
                id=str(entry["id"]),
                title=str(entry["title"]),
                source_id=str(entry["source_id"]),
                source_url=str(entry["source_url"]),
                source_file=str(entry["source_file"]),
                text=text,
            )
        )

    corpus = CanonicalCorpus(sections=tuple(sections))
    if corpus.ids() != EXPECTED_SECTION_IDS:
        raise CanonicalCorpusError(f"canonical section order mismatch: {corpus.ids()!r}")
    return corpus
