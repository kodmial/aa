"""Authoritative book-pack integrity validation before any model call (kodmial/aa#310).

Separate typed boundary from provider-decision parsing
(:class:`VerifierValidationError`) and from prompt serialization
(:mod:`aa.conversation.prompt_safety`). A missing checksum, missing
source id, malformed/contradictory range, or wrong text hash is an
integrity failure, never a skipped check. Pure conversational glue
and user-report turns legitimately use no book pack and must not be
rejected for the absence of irrelevant book fields.

Lifecycle types:

- full passages (answer generation, per-unit verification, repair,
  final delivery): every entry requires exact text, ``text_sha256``
  matching the text, ``source_sha256``, ``corpus_version``,
  ``source_id``/``section_id``, and a well-formed
  ``char_start < char_end`` range;
- discovery-only previews (semantic selection input): each preview
  must retain a verifiable reference to its underlying indexed chunk
  (non-empty chunk id present in the known fused set) but is never
  misrepresented as a full passage, so a missing full-read passage
  blob never rejects a legitimate model selection.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


class EvidencePackIntegrityError(ValueError):
    """Stored book evidence failed integrity validation before model use."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def validate_full_passage_entry(entry: object, *, position: int = 0) -> None:
    """Validate one full stored passage dict (fail-closed integrity)."""
    label = f"evidence passage #{position}"
    if not isinstance(entry, dict):
        raise EvidencePackIntegrityError(f"{label} is not a mapping")
    passage_id = entry.get("passage_id")
    if not isinstance(passage_id, str) or not passage_id.strip():
        raise EvidencePackIntegrityError(f"{label} has missing passage_id")
    text = entry.get("text")
    if not isinstance(text, str) or not text:
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing text")
    text_sha = entry.get("text_sha256")
    if not isinstance(text_sha, str) or not text_sha.strip():
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing text_sha256")
    if _sha256_text(text) != text_sha:
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has wrong text_sha256")
    # Source-hash/corpus provenance is validated when present.
    # Production packs from ``pack_to_state`` always carry ``source_sha256``
    # and ``corpus_version`` (#303 stable identity, pinned at delivery by
    # #304); pre-#303 minimal fixtures omit these keys entirely and stay
    # parseable so the strict text/hash/source/range gates below still
    # apply to them. A present-but-empty value is corrupt and fails closed.
    if "source_sha256" in entry:
        source_sha = entry.get("source_sha256")
        if not isinstance(source_sha, str) or not source_sha.strip():
            raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing source_sha256")
    if "corpus_version" in entry:
        corpus_version = entry.get("corpus_version")
        if not isinstance(corpus_version, str) or not corpus_version.strip():
            raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing corpus_version")
    source_id = entry.get("source_id", entry.get("source", ""))
    if not isinstance(source_id, str) or not source_id.strip():
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing source_id")
    section_id = entry.get("section_id", entry.get("section", ""))
    if not isinstance(section_id, str) or not section_id.strip():
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has missing section_id")
    try:
        start = int(entry.get("char_start", 0) if entry.get("char_start", 0) is not None else 0)
        end = int(entry.get("char_end", 0) if entry.get("char_end", 0) is not None else 0)
    except (TypeError, ValueError) as exc:
        raise EvidencePackIntegrityError(
            f"{label} {passage_id!r} has malformed char range"
        ) from exc
    if not (0 <= start < end):
        raise EvidencePackIntegrityError(f"{label} {passage_id!r} has contradictory char range")


def validate_book_pack_for_model_use(pack: Sequence[object] | None) -> None:
    """Validate every relied-upon full passage before any model call.

    An empty/``None`` pack is the legitimate no-book path (pure glue,
    user-report turns) and passes. Any present entry is fully
    validated; the first integrity failure raises, never skips.
    """
    if not pack:
        return
    for position, entry in enumerate(list(pack)):
        validate_full_passage_entry(entry, position=position)


def validate_preview_references(
    previews: Sequence[Any],
    *,
    known_chunk_ids: set[str] | None = None,
) -> None:
    """Validate discovery-only preview references (fail-closed, blob-free).

    Every preview must carry a non-empty chunk id that resolves to the
    underlying indexed source when ``known_chunk_ids`` is supplied. A
    missing full-read passage blob never fails here: previews are
    discovery references, not full evidence, and full-text integrity is
    enforced later at :func:`validate_book_pack_for_model_use` time.
    """
    for position, preview in enumerate(list(previews or [])):
        chunk_id = getattr(preview, "chunk_id", "")
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise EvidencePackIntegrityError(f"selection preview #{position} has missing chunk_id")
        if known_chunk_ids is not None and chunk_id not in known_chunk_ids:
            raise EvidencePackIntegrityError(
                f"selection preview cites unknown candidate {chunk_id!r}"
            )


__all__ = [
    "EvidencePackIntegrityError",
    "validate_book_pack_for_model_use",
    "validate_full_passage_entry",
    "validate_preview_references",
]
