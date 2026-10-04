"""AA grounding boundary (issue #48).

Owns the authoritative Russian quotation and multilingual grounding
policy as deterministic Python. The OpenCode agent prompt states the
user-facing duties; this package enforces them.
"""

from __future__ import annotations

from aa.grounding.gate import (
    GroundingGate,
    GroundingVerdict,
    check_grounding,
    default_entails,
)
from aa.grounding.quotes import (
    RUSSIAN_EDITION_ISBN,
    RUSSIAN_EDITION_PUBLISHER_MARKER,
    RUSSIAN_EDITION_TITLE,
    TRANSLATION_MARKER_RU,
    EvidenceKind,
    EvidenceUnit,
    Provenance,
    QuoteKind,
    contains_translation_label,
    format_russian_quotation,
)

__all__ = [
    "RUSSIAN_EDITION_ISBN",
    "RUSSIAN_EDITION_PUBLISHER_MARKER",
    "RUSSIAN_EDITION_TITLE",
    "TRANSLATION_MARKER_RU",
    "EvidenceKind",
    "EvidenceUnit",
    "GroundingGate",
    "GroundingVerdict",
    "Provenance",
    "QuoteKind",
    "check_grounding",
    "contains_translation_label",
    "default_entails",
    "format_russian_quotation",
]
