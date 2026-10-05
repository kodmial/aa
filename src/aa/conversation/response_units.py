"""Ordered response-unit segmentation for claim-level verification.

Every non-empty draft unit receives exactly one verifier verdict. The
splitter is the same qualified standard sentence-boundary component
selected for production chunking: ``razdel.sentenize``
(``razdel==0.5.0``). No hand-written regex boundary detector and no
AA-specific sentence exceptions live here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResponseUnitDraft:
    """One ordered draft unit awaiting a verifier verdict."""

    unit_id: str
    text: str
    char_start: int
    char_end: int


class ResponseUnitError(ValueError):
    """Raised when draft segmentation fails closed."""


def split_response_units(draft: str) -> list[ResponseUnitDraft]:
    """Split ``draft`` into ordered non-empty response units with razdel.

    Unit ids are stable ``u1``..``uN`` in draft order. Offsets are
    draft-relative and every span round-trips exactly. Empty or
    whitespace-only drafts yield no units.
    """
    if not draft.strip():
        return []
    try:
        from razdel import sentenize  # type: ignore[import-untyped]
    except Exception as exc:
        raise ResponseUnitError(f"razdel segmenter is unavailable: {exc}") from exc
    spans: list[tuple[int, int]] = []
    for substring in sentenize(draft):
        spans.append((int(substring.start), int(substring.stop)))
    if not spans:
        raise ResponseUnitError("sentence split produced no sentences")
    units: list[ResponseUnitDraft] = []
    position = 0
    for start, stop in spans:
        if start < 0 or stop <= start or stop > len(draft):
            raise ResponseUnitError("razdel returned an out-of-range span")
        raw = draft[start:stop]
        text = raw.strip()
        if not text:
            continue
        position += 1
        # Re-anchor stripped offsets so spans round-trip to unit text.
        lead = raw.index(text)
        units.append(
            ResponseUnitDraft(
                unit_id=f"u{position}",
                text=text,
                char_start=start + lead,
                char_end=start + lead + len(text),
            )
        )
    # Renumber sequentially over non-empty units only.
    renumbered = [
        ResponseUnitDraft(
            unit_id=f"u{pos}",
            text=unit.text,
            char_start=unit.char_start,
            char_end=unit.char_end,
        )
        for pos, unit in enumerate(units, start=1)
    ]
    void = [unit for unit in renumbered if not unit.text.strip()]
    if void:
        raise ResponseUnitError("segmentation produced an empty unit")
    return renumbered


def segmenter_identity() -> dict[str, str]:
    """Return the pinned production segmenter identity for observability."""
    from aa.corpus.sentences import SEGMENTER_ID, SEGMENTER_VERSION

    return {"segmenter_id": SEGMENTER_ID, "segmenter_version": SEGMENTER_VERSION}


__all__ = ["ResponseUnitDraft", "ResponseUnitError", "segmenter_identity", "split_response_units"]
