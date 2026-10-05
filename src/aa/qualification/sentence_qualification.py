"""One-time qualification: razdel vs spaCy Russian segmentation (issue #115).

Bounded deterministic comparison between the two maintained standard
Russian-capable segmenters:

1. ``razdel.sentenize`` (Russian-specific, exact start/stop offsets);
2. spaCy Russian statistical segmentation via the maintained
   ``ru_core_news_sm`` pipeline (parser-driven statistical boundaries;
   the pipeline carries no ``sentencizer`` component, so the simple
   punctuation-only segmenter is never used).

Selection rule (in order):

1. exact source-offset round trip is mandatory (span slices exactly,
   inter-sentence gaps whitespace-only);
2. lowest boundary error on the frozen fixture wins;
3. if tied, prefer the lower startup/RAM/dependency footprint;
4. the winner is recorded in the index manifest.

Production uses only the winner; this harness is qualification-only and is
never imported on the hot path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aa.corpus.sentences import RAZDEL_VERSION

SPACY_VERSION = "3.8.7"
SPACY_MODEL_ID = "ru_core_news_sm"
SPACY_MODEL_VERSION = "3.8.0"
FIXTURE_VERSION = "aa-sentence-boundary-fixture/1"


class SentenceQualificationError(ValueError):
    """Raised when the sentence qualification cannot be evaluated exactly."""


def fixture_path(repo_root: str | Path) -> Path:
    """Return the frozen fixture path."""
    return Path(repo_root) / "qualification" / "sentence_boundary_fixture.v1.json"


def load_fixture(path: str | Path) -> dict[str, Any]:
    """Load and validate the frozen boundary fixture."""
    raw = Path(path).read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SentenceQualificationError(f"fixture is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise SentenceQualificationError("fixture must be a JSON object")
    if payload.get("fixture_version") != FIXTURE_VERSION:
        raise SentenceQualificationError(f"fixture_version must be {FIXTURE_VERSION!r}")
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SentenceQualificationError("fixture must carry a non-empty cases list")
    required_categories = {
        "abbreviation",
        "initials",
        "dialogue",
        "quotes",
        "ellipsis",
        "list",
        "decimal",
        "chapter-typography",
    }
    seen: set[str] = set()
    for position, raw_case in enumerate(cases):
        owner = f"case[{position}]"
        if not isinstance(raw_case, dict):
            raise SentenceQualificationError(f"{owner} must be an object")
        case_id = raw_case.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise SentenceQualificationError(f"{owner}: case_id must be non-empty")
        if case_id in seen:
            raise SentenceQualificationError(f"{owner}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        for key in ("category", "paragraph", "gold_sentences"):
            if raw_case.get(key) is None:
                raise SentenceQualificationError(f"{case_id}: missing {key}")
        paragraph = raw_case["paragraph"]
        gold = raw_case["gold_sentences"]
        if not isinstance(paragraph, str) or not paragraph:
            raise SentenceQualificationError(f"{case_id}: paragraph must be non-empty")
        if not isinstance(gold, list) or not gold:
            raise SentenceQualificationError(f"{case_id}: gold_sentences must be non-empty")
        for sentence in gold:
            if not isinstance(sentence, str) or not sentence:
                raise SentenceQualificationError(f"{case_id}: gold holds an empty sentence")
        # Gold must tile the paragraph exactly (concatenation with single-space
        # gaps is not enough: gold sentences joined by "" must reconstruct the
        # paragraph only when gold already contains spacing; here gold stores
        # exact sentence texts and gaps are single spaces, so join with "" of
        # tiled spans must equal the paragraph).
        cursor = 0
        for sentence in gold:
            start = paragraph.find(sentence, cursor)
            if start != cursor and not (paragraph[cursor:start].strip() == "" and start > cursor):
                # Allow leading-gap whitespace: gold sentences tile modulo
                # whitespace-only gaps.
                raise SentenceQualificationError(f"{case_id}: gold does not tile the paragraph")
            cursor = start + len(sentence)
        tail = paragraph[cursor:]
        if tail.strip():
            raise SentenceQualificationError(f"{case_id}: gold does not cover the paragraph tail")
    categories = {str(case["category"]) for case in cases if isinstance(case, dict)}
    missing = required_categories - categories
    if missing:
        raise SentenceQualificationError(f"fixture is missing categories: {sorted(missing)}")
    return payload


def _spans_round_trip(paragraph: str, spans: list[tuple[int, int]]) -> bool:
    if not spans:
        return False
    if paragraph[0 : spans[0][0]].strip():
        return False
    previous_end: int | None = None
    for start, stop in spans:
        if start < 0 or stop <= start or stop > len(paragraph):
            return False
        if paragraph[start:stop] == "":
            return False
        if previous_end is not None:
            if start < previous_end:
                return False
            if paragraph[previous_end:start].strip():
                return False
        previous_end = stop
    if previous_end is not None and paragraph[previous_end:].strip():
        return False
    return True


def _boundary_errors(
    paragraph: str, gold_sentences: list[str], spans: list[tuple[int, int]]
) -> int:
    """Count boundary errors vs gold (break-position mismatches + count diff)."""
    # Derive gold break offsets (end of each non-terminal sentence).
    gold_breaks: list[int] = []
    cursor = 0
    for sentence in gold_sentences[:-1]:
        start = paragraph.find(sentence, cursor)
        gold_breaks.append(start + len(sentence))
        cursor = start + len(sentence)
    # Predicted breaks: end of each non-terminal span, normalized to skip
    # trailing gap whitespace (break falls inside the gap).
    predicted_breaks = [stop for _, stop in spans[:-1]]
    # Align by count: missing/extra breaks each count fully, plus position
    # mismatches beyond whitespace tolerance.
    errors = abs(len(predicted_breaks) - len(gold_breaks))
    for predicted, gold in zip(predicted_breaks, gold_breaks, strict=False):
        if predicted == gold:
            continue
        gap = paragraph[min(predicted, gold) : max(predicted, gold)]
        if gap.strip():
            errors += 1
    return errors


def run_razdel(paragraph: str) -> list[tuple[int, int]]:
    """Run ``razdel.sentenize`` and return ``[(start, stop)]`` offsets."""
    from razdel import sentenize  # type: ignore[import-untyped]

    return [(int(item.start), int(item.stop)) for item in sentenize(paragraph)]


_SPACY_NLP: Any | None = None


def run_spacy(paragraph: str) -> list[tuple[int, int]]:
    """Run spaCy Russian statistical segmentation (no Sentencizer)."""
    import spacy

    global _SPACY_NLP
    if _SPACY_NLP is None:
        model = spacy.load(SPACY_MODEL_ID)
        pipe_names = list(model.pipe_names)
        if "sentencizer" in pipe_names:
            raise SentenceQualificationError("spaCy pipeline must not use Sentencizer")
        if "parser" not in pipe_names and "senter" not in pipe_names:
            raise SentenceQualificationError(
                "spaCy Russian pipeline must provide a statistical segmenter"
            )
        _SPACY_NLP = model
    doc = _SPACY_NLP(paragraph)
    return [(int(sent.start_char), int(sent.end_char)) for sent in doc.sents]


def qualify_fixture(payload: dict[str, Any]) -> dict[str, Any]:
    """Evaluate both candidates against the frozen fixture (deterministic)."""
    cases = payload["cases"]
    assert isinstance(cases, list)
    razdel_total = 0
    spacy_total = 0
    razdel_round_trip_ok = True
    spacy_round_trip_ok = True
    per_case: list[dict[str, Any]] = []
    for raw in cases:
        assert isinstance(raw, dict)
        case_id = str(raw["case_id"])
        paragraph = str(raw["paragraph"])
        gold = [str(item) for item in raw["gold_sentences"]]
        razdel_spans = run_razdel(paragraph)
        spacy_spans = run_spacy(paragraph)
        razdel_ok = _spans_round_trip(paragraph, razdel_spans)
        spacy_ok = _spans_round_trip(paragraph, spacy_spans)
        razdel_round_trip_ok = razdel_round_trip_ok and razdel_ok
        spacy_round_trip_ok = spacy_round_trip_ok and spacy_ok
        razdel_errors = _boundary_errors(paragraph, gold, razdel_spans)
        spacy_errors = _boundary_errors(paragraph, gold, spacy_spans)
        razdel_total += razdel_errors
        spacy_total += spacy_errors
        per_case.append(
            {
                "case_id": case_id,
                "category": str(raw["category"]),
                "razdel_spans": [[a, b] for a, b in razdel_spans],
                "spacy_spans": [[a, b] for a, b in spacy_spans],
                "razdel_round_trip": razdel_ok,
                "spacy_round_trip": spacy_ok,
                "razdel_errors": razdel_errors,
                "spacy_errors": spacy_errors,
            }
        )
    # Deterministic selection.
    if not razdel_round_trip_ok and not spacy_round_trip_ok:
        raise SentenceQualificationError("neither candidate round-trips exactly")
    if razdel_round_trip_ok and not spacy_round_trip_ok:
        winner = "razdel"
        reason = "spacy fails exact source-offset round trip; razdel round-trips"
    elif spacy_round_trip_ok and not razdel_round_trip_ok:
        winner = "spacy"
        reason = "razdel fails exact source-offset round trip; spacy round-trips"
    elif razdel_total < spacy_total:
        winner = "razdel"
        reason = f"lowest boundary error wins (razdel {razdel_total} < spacy {spacy_total})"
    elif spacy_total < razdel_total:
        winner = "spacy"
        reason = f"lowest boundary error wins (spacy {spacy_total} < razdel {razdel_total})"
    else:
        # Effectively tied: prefer the lower startup/RAM/dependency footprint.
        winner = "razdel"
        reason = (
            f"boundary quality tied ({razdel_total} == {spacy_total}); "
            "razdel wins on lower startup/RAM/dependency footprint "
            "(pure-python, no statistical model load)"
        )
    return {
        "fixture_version": FIXTURE_VERSION,
        "razdel": {
            "version": RAZDEL_VERSION,
            "total_errors": razdel_total,
            "round_trip_ok": razdel_round_trip_ok,
        },
        "spacy": {
            "spacy_version": SPACY_VERSION,
            "model": f"{SPACY_MODEL_ID}=={SPACY_MODEL_VERSION}",
            "statistical_pipeline": "parser (no sentencizer)",
            "total_errors": spacy_total,
            "round_trip_ok": spacy_round_trip_ok,
        },
        "per_case": per_case,
        "winner": winner,
        "reason": reason,
        "production_segmenter": ("razdel-0.5.0" if winner == "razdel" else "spacy-ru-3.8.0"),
    }
