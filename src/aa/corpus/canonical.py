"""Canonical AA source baseline.

Defines the authoritative section inventory for the AA Big Book core used by
this project, plus deterministic ingestion and validation over an operator
provided source directory.

Copyright handling: this repository ships no AA book text. The operator
supplies the canonical source separately from a lawfully obtained copy via
``AA_CORPUS_PATH``. Validation records checksum/version/token counts only and
never logs corpus contents. Runtime quotations must stay short and
source-exact (see README section "AA source and copyright").
"""

from __future__ import annotations

import hashlib
import math
import pathlib
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RequiredSection:
    """One required canonical section."""

    section_id: str
    title: str
    pattern: str


def _title_pattern(title: str) -> str:
    """Build a tolerant regex for a section title.

    Accepts any dash variant, flexible whitespace and an optional leading
    ``Chapter N`` / ``Chapter N.`` prefix so both file-per-section layouts
    and single-file layouts with markdown headers validate identically.
    """
    escaped = re.escape(title)
    escaped = escaped.replace(r"\ ", r"\s+")
    escaped = escaped.replace(r"\-", r"[\-–—]")
    return escaped


REQUIRED_SECTIONS: tuple[RequiredSection, ...] = (
    RequiredSection(
        "doctors-opinion", "The Doctor's Opinion", _title_pattern("The Doctor's Opinion")
    ),
    RequiredSection("chapter-01", "Bill's Story", _title_pattern("Bill's Story")),
    RequiredSection("chapter-02", "There Is a Solution", _title_pattern("There Is a Solution")),
    RequiredSection("chapter-03", "More About Alcoholism", _title_pattern("More About Alcoholism")),
    RequiredSection("chapter-04", "We Agnostics", _title_pattern("We Agnostics")),
    RequiredSection("chapter-05", "How It Works", _title_pattern("How It Works")),
    RequiredSection("chapter-06", "Into Action", _title_pattern("Into Action")),
    RequiredSection("chapter-07", "Working With Others", _title_pattern("Working With Others")),
    RequiredSection("chapter-08", "To Wives", _title_pattern("To Wives")),
    RequiredSection("chapter-09", "The Family Afterward", _title_pattern("The Family Afterward")),
    RequiredSection("chapter-10", "To Employers", _title_pattern("To Employers")),
    RequiredSection("chapter-11", "A Vision for You", _title_pattern("A Vision for You")),
)

#: Markers that must NOT appear as section headers. Matching is restricted to
#: header-like lines (short lines, optionally markdown-prefixed) so ordinary
#: body mentions do not trigger false positives.
EXCLUDED_HEADER_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"^\s*#{1,6}\s*forewords?\b.*$",
        r"^\s*#{1,6}\s*prefaces?\b.*$",
        r"^\s*#{1,6}\s*personal\s+stor(?:y|ies)\b.*$",
        r"^\s*#{1,6}\s*appendi(?:x|ces)\b.*$",
        r"^\s*#{1,6}\s*appendix\s+[ivx\d]+\b.*$",
        r"^\s*#{1,6}\s*the\s+spiritual\s+experience\s*$",
        r"^\s*#{1,6}\s*the\s+medical\s+view\b.*$",
        r"^\s*#{1,6}\s*the\s+lasker\s+award\b.*$",
        r"^\s*#{1,6}\s*the\s+religious\s+view\b.*$",
        r"^\s*#{1,6}\s*how\s+to\s+get\s+in\s+touch\b.*$",
        r"^\s*forewords?\s+to\s+(the\s+)?(first|second|third|fourth)\s+edition\s*$",
        r"^\s*preface\s*$",
        r"^\s*personal\s+stor(?:y|ies)\s*$",
        r"^\s*appendi(?:x|ces)\s*.*$",
        r"^\s*appendix\s+[ivx\d]+\b.*$",
        r"^\s*they\s+stopped\s+in\s+time\s*$",
        r"^\s*they\s+lost\s+nearly\s+all\s*$",
        r"^\s*women\s+suffer\s+too\s*$",
    )
)

_CHAPTER_PREFIX = re.compile(r"(?i)^\s*(?:chapter\s+\d{1,2}\s*[\.\-–—:]?\s*)?")


def _is_header_line(line: str) -> bool:
    stripped = line.strip().lstrip("#").strip()
    return 0 < len(stripped) <= 120


def _normalize_title(text: str) -> str:
    normalized = text.lower()
    normalized = normalized.replace("–", "-").replace("—", "-")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def find_section_occurrences(text: str, section: RequiredSection) -> list[int]:
    """Return start offsets where ``section`` header occurs in ``text``.

    Only markdown headers (``# ...``) or standalone title lines count, so
    ordinary body sentences that mention a title never match.
    """
    pattern = re.compile(section.pattern, re.IGNORECASE)
    expected = _normalize_title(section.title)
    occurrences: list[int] = []
    for match in re.finditer(r"^.*$", text, re.MULTILINE):
        line = match.group(0)
        if not _is_header_line(line):
            continue
        is_markdown = line.lstrip().startswith("#")
        candidate = line.strip().lstrip("#").strip()
        candidate = _CHAPTER_PREFIX.sub("", candidate)
        if not pattern.search(candidate):
            continue
        if is_markdown:
            # A markdown header must be mostly the title itself.
            title_words = len(section.title.split())
            candidate_words = len(candidate.split())
            if candidate_words <= title_words + 6:
                occurrences.append(match.start())
        elif _normalize_title(candidate) == expected:
            occurrences.append(match.start())
    return occurrences


def find_excluded_headers(text: str) -> list[str]:
    """Return excluded header lines found in ``text``."""
    found: list[str] = []
    for line in text.splitlines():
        if not _is_header_line(line):
            continue
        for pattern in EXCLUDED_HEADER_PATTERNS:
            if pattern.match(line):
                found.append(line.strip())
                break
    return found


class CorpusValidationError(ValueError):
    """Raised when the canonical source fails validation (fail-closed)."""


@dataclass(frozen=True)
class CorpusSection:
    """One validated canonical section with deterministic boundaries."""

    section_id: str
    title: str
    text: str
    start_offset: int
    end_offset: int
    token_estimate: int


@dataclass(frozen=True)
class CanonicalCorpus:
    """Validated canonical corpus snapshot (no logging of ``sections`` text)."""

    sections: tuple[CorpusSection, ...]
    checksum_sha256: str
    token_estimate: int
    char_count: int
    source_version: str = field(default="local")

    def section_ids(self) -> tuple[str, ...]:
        """Return ordered canonical section ids."""
        return tuple(section.section_id for section in self.sections)

    def to_safe_dict(self) -> dict[str, object]:
        """Return metadata only; never includes corpus contents."""
        return {
            "source_version": self.source_version,
            "checksum_sha256": self.checksum_sha256,
            "token_estimate": self.token_estimate,
            "char_count": self.char_count,
            "section_count": len(self.sections),
            "section_ids": list(self.section_ids()),
        }


def estimate_tokens(text: str) -> int:
    """Deterministically estimate token count for ``text``.

    Uses ``ceil(chars / 4)``: a conservative, language-agnostic upper-bound
    heuristic suitable for EN/RU budget checks without a model tokenizer.
    """
    if not text:
        return 0
    return int(math.ceil(len(text) / 4))


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_source_text(path: pathlib.Path) -> str:
    if path.is_file():
        return _normalize_newlines(path.read_text(encoding="utf-8"))
    if path.is_dir():
        files = sorted(
            [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".txt", ".md"}]
        )
        if not files:
            raise CorpusValidationError(
                f"corpus directory {path} contains no .txt/.md source files"
            )
        parts = [_normalize_newlines(p.read_text(encoding="utf-8")) for p in files]
        return "\n\n".join(part.strip("\n") for part in parts) + "\n"
    raise CorpusValidationError(f"corpus path does not exist: {path}")


def _canonical_bytes(sections_in_order: list[tuple[str, str]]) -> bytes:
    normalized = "\n\n".join(f"## {title}\n\n{body.strip()}" for title, body in sections_in_order)
    return (normalized.strip() + "\n").encode("utf-8")


def load_canonical_corpus(
    corpus_path: str | pathlib.Path, *, source_version: str = "local"
) -> CanonicalCorpus:
    """Load and validate the canonical corpus at ``corpus_path``.

    Fails closed with :class:`CorpusValidationError` when a required section
    is missing or duplicated, when an excluded header is present, or when any
    section body is empty. Never logs corpus contents; callers must use
    :meth:`CanonicalCorpus.to_safe_dict` for observability.
    """
    path = pathlib.Path(corpus_path)
    raw = _read_source_text(path)

    excluded = find_excluded_headers(raw)
    if excluded:
        preview = "; ".join(excluded[:5])
        raise CorpusValidationError(f"excluded sections present: {preview}")

    boundaries: list[tuple[RequiredSection, int]] = []
    for section in REQUIRED_SECTIONS:
        occurrences = find_section_occurrences(raw, section)
        if len(occurrences) == 0:
            raise CorpusValidationError(f"required section missing: {section.title}")
        if len(occurrences) > 1:
            raise CorpusValidationError(
                f"required section duplicated ({len(occurrences)}x): {section.title}"
            )
        boundaries.append((section, occurrences[0]))

    boundaries.sort(key=lambda item: item[1])
    ordered_ids = [section.section_id for section, _ in boundaries]
    expected_ids = [section.section_id for section in REQUIRED_SECTIONS]
    # Single-file layouts may list sections in canonical order only; accept
    # any deterministic file-concatenation order only if all sections exist
    # exactly once, but record canonical order for checksum stability.
    _ = (ordered_ids, expected_ids)

    # Slice bodies between consecutive headers.
    sections: list[CorpusSection] = []
    for index, (section, start) in enumerate(boundaries):
        end = boundaries[index + 1][1] if index + 1 < len(boundaries) else len(raw)
        body = raw[start:end].strip()
        if not body:
            raise CorpusValidationError(f"required section empty: {section.title}")
        sections.append(
            CorpusSection(
                section_id=section.section_id,
                title=section.title,
                text=body,
                start_offset=start,
                end_offset=end,
                token_estimate=estimate_tokens(body),
            )
        )

    # Order sections canonically so checksums are stable regardless of file
    # concatenation order.
    by_id = {section.section_id: section for section in sections}
    ordered = tuple(by_id[section.section_id] for section in REQUIRED_SECTIONS)
    canonical = _canonical_bytes([(section.title, section.text) for section in ordered])
    checksum = hashlib.sha256(canonical).hexdigest()
    return CanonicalCorpus(
        sections=ordered,
        checksum_sha256=checksum,
        token_estimate=sum(section.token_estimate for section in ordered),
        char_count=sum(len(section.text) for section in ordered),
        source_version=source_version,
    )
