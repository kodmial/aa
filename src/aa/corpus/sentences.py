"""Standardized Russian sentence segmentation (issue #115).

Production uses exactly one qualified standard segmenter: ``razdel``
(``razdel.sentenize``, Russian-specific, exact source offsets). The
alternative candidate (spaCy Russian statistical segmentation) is evaluated
only by the one-time qualification harness
(:mod:`aa.qualification.sentence_qualification`); it is never imported on the
production chunking path.

A hand-written sentence splitter is forbidden here: this module contains no
regex boundary detector and no AA-specific sentence exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass

RAZDEL_VERSION = "0.5.0"
SEGMENTER_ID = "razdel-0.5.0"
SEGMENTER_VERSION = "sentence-seg/razdel-0.5.0"


@dataclass(frozen=True)
class SentenceSpan:
    """One sentence slice of a paragraph (section-relative offsets)."""

    index: int  # 1-based per paragraph
    char_start: int
    char_end: int
    text: str


class SentenceSegmentationError(ValueError):
    """Raised when sentence segmentation fails closed."""


def split_sentences_razdel(paragraph_text: str, base_offset: int) -> list[SentenceSpan]:
    """Split one paragraph into sentences with ``razdel.sentenize``.

    Offsets are section-relative and every span round-trips exactly:
    ``section_text[span.char_start:span.char_end] == span.text`` for the
    owning section text. Inter-sentence gaps must be whitespace-only;
    otherwise the build fails closed instead of losing content.
    """
    if base_offset < 0:
        raise SentenceSegmentationError("base_offset must be >= 0")
    if not paragraph_text:
        raise SentenceSegmentationError("refusing to split an empty paragraph")
    try:
        from razdel import sentenize  # type: ignore[import-untyped]
    except Exception as exc:
        raise SentenceSegmentationError(f"razdel segmenter is unavailable: {exc}") from exc
    raw: list[tuple[int, int]] = []
    for substring in sentenize(paragraph_text):
        raw.append((int(substring.start), int(substring.stop)))
    if not raw:
        raise SentenceSegmentationError("sentence split produced no sentences")
    # Validate exact round-trip and whitespace-only gaps.
    previous_stop: int | None = None
    for start, stop in raw:
        if start < 0 or stop <= start or stop > len(paragraph_text):
            raise SentenceSegmentationError("razdel returned an out-of-range span")
        if paragraph_text[start:stop] == "":
            raise SentenceSegmentationError("razdel returned an empty span")
        if previous_stop is not None:
            gap = paragraph_text[previous_stop:start]
            if gap.strip():
                raise SentenceSegmentationError("sentence split lost paragraph content")
            if start < previous_stop:
                raise SentenceSegmentationError("sentence spans overlap")
        previous_stop = stop
    spans: list[SentenceSpan] = []
    for number, (start, stop) in enumerate(raw, start=1):
        spans.append(
            SentenceSpan(
                index=number,
                char_start=base_offset + start,
                char_end=base_offset + stop,
                text=paragraph_text[start:stop],
            )
        )
    return spans


def segmenter_identity() -> dict[str, str]:
    """Return the pinned production segmenter identity for index manifests."""
    return {
        "segmenter_id": SEGMENTER_ID,
        "segmenter_version": SEGMENTER_VERSION,
        "razdel_version": RAZDEL_VERSION,
    }
