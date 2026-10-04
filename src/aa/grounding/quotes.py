"""Russian quotation policy (issue #48).

Authoritative, testable rules for user-visible Russian quotations:

- Russian conversations display quotations in Russian.
- Exact Russian quotations come verbatim from the version-pinned
  authoritative Russian corpus (fourth-edition Russian Big Book lineage
  qualified in issue #45: title ``RUSSIAN_EDITION_TITLE``, publisher
  marker ``RUSSIAN_EDITION_PUBLISHER_MARKER``, ISBN
  ``RUSSIAN_EDITION_ISBN``). The literary bytes are never normalized,
  summarized, translated, or rewritten in storage.
- When no authoritative Russian source is available, a generated
  translation may be shown only when the caller explicitly allows the
  translation fallback, only with the ``TRANSLATION_MARKER_RU`` label,
  and only with provenance to the exact source unit(s) it derives from.
  A translation is never presented as source-exact Russian text.
- Provenance is pinned to exact source unit(s): corpus version, source
  id, section/chunk locators, and offsets. A bare ``source_id`` without
  source-exact text is not provenance.
- Navigation aids (book map, search previews, embeddings, rankings,
  scores, reranker output) and query artifacts (original, normalized, or
  rewritten queries) are never evidence and never quotable as the book.

This module owns quotation classification and display formatting. The
semantic grounding gate lives in :mod:`aa.grounding.gate`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

RUSSIAN_EDITION_TITLE = "АНОНИМНЫЕ АЛКОГОЛИКИ"
RUSSIAN_EDITION_ISBN = "978-5-906531-01-8"
RUSSIAN_EDITION_PUBLISHER_MARKER = "Фонд «Единство», 2013"

TRANSLATION_MARKER_RU = "[перевод — не точная цитата источника]"
"""Required label for any Russian rendering that is not source-exact.

The marker must appear in the user-visible text of every translated
quotation so exact-source and translated quotations are unambiguous.
"""


class QuoteKind(Enum):
    """How a Russian user-visible quotation relates to the corpus."""

    EXACT_SOURCE = "exact-source"
    TRANSLATION = "translation"


class EvidenceKind(Enum):
    """What a candidate evidence unit is.

    Only :attr:`SOURCE_TEXT` can ground a claim. Every other kind is a
    navigation aid or a query artifact and is never evidence, even when
    its text happens to match the claim.
    """

    SOURCE_TEXT = "source-text"
    NAVIGATION = "navigation"
    QUERY = "query"
    TRANSLATION_DRAFT = "translation-draft"


@dataclass(frozen=True)
class Provenance:
    """Exact, version-pinned provenance for one source unit."""

    corpus_version: str
    source_id: str
    section_id: str
    chunk_id: str = ""
    char_start: int = 0
    char_end: int = 0
    source_checksum: str = ""
    source_language: str = "en"

    def validate(self) -> None:
        """Fail closed on incomplete or incoherent provenance."""
        if not self.corpus_version.strip():
            raise ValueError("provenance corpus_version must be pinned")
        if not self.source_id.strip():
            raise ValueError("provenance source_id must not be empty")
        if not self.section_id.strip():
            raise ValueError("provenance section_id must not be empty")
        if self.source_language not in ("ru", "en"):
            raise ValueError("provenance source_language must be 'ru' or 'en'")
        if self.char_start < 0 or self.char_end < 0:
            raise ValueError("provenance offsets must be >= 0")
        if self.char_end < self.char_start:
            raise ValueError("provenance char_end must be >= char_start")


@dataclass(frozen=True)
class EvidenceUnit:
    """One candidate evidence unit with its kind and provenance."""

    kind: EvidenceKind
    language: str
    text: str
    provenance: Provenance

    @property
    def is_source_text(self) -> bool:
        """Whether this unit can ground a claim (exact text only)."""
        return self.kind is EvidenceKind.SOURCE_TEXT and bool(self.text.strip())


def contains_translation_label(text: str) -> bool:
    """Return whether ``text`` carries an explicit translation label."""
    return TRANSLATION_MARKER_RU in text


def format_russian_quotation(
    text: str,
    kind: QuoteKind,
    provenance: Provenance,
) -> str:
    """Format a Russian user-visible quotation with kind and provenance.

    Exact-source quotations are returned verbatim with a provenance
    citation. Translations always carry ``TRANSLATION_MARKER_RU`` plus
    the provenance of the exact source unit(s) they derive from, so a
    generated translation can never be mistaken for source-exact text.
    """
    if not text.strip():
        raise ValueError("quotation text must not be empty")
    provenance.validate()
    locator = provenance.section_id
    if provenance.chunk_id:
        locator = f"{locator}#{provenance.chunk_id}"
    citation = f"[{provenance.source_id}/{locator}]"
    if kind is QuoteKind.EXACT_SOURCE:
        if contains_translation_label(text):
            raise ValueError("exact-source quotation must not carry a translation label")
        return f"{text}\n{citation}"
    return f"{TRANSLATION_MARKER_RU} {text}\n{citation}"
