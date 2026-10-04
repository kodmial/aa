"""Production Russian-first grounded conversational turn orchestrator (issue #9).

Deterministic state machine for every non-trivial substantive user turn:

0. deterministic safety/operations gate (safety routing lives in
   :mod:`aa.safety.router`; the operations half checks the opened RU
   index, pinned versions and the named ``aa`` agent/model contract);
1. validated slang-aware retrieval planning (``ru-query-plan-v1``);
2. RU-first whole-corpus search (qualified #17/#47 configuration);
3. deterministic cross-aspect deduplication/diversity;
4. exact RU ``book_read`` / bounded ``book_expand`` loading;
5. one bounded coverage check and optional second retrieval round;
6. exact-RU evidence-pack construction under the source/context budget;
7. final synthesis through the named ``aa`` agent;
8. semantic grounding validation for every substantive answer unit;
9. one bounded regeneration for unsupported units, otherwise fail closed.

Planner/search metadata never enters the evidence pack. English is
control/optional candidate discovery only and can never silently
substitute for RU evidence. Logs carry only ids, counts, digests and
model pointers, never user text, corpus text, secrets or snapshots.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aa.conversation.output_limits import (
    ENVELOPE_FALLBACK_REPLY,
    HARD_CHARS,
    HARD_WORDS,
    MAX_COMPACT_REGENERATIONS,
    QUOTE_BUDGET_CHARS,
    TARGET_CHARS,
    TARGET_WORDS,
    aggregate_quote_chars,
    compact_retry_instruction,
    compact_text_to_envelope,
    count_graphemes,
    count_words,
    envelope_passes,
    generation_budget_instruction,
    is_bulk_reproduction_request,
    is_continuation_request,
    resolve_generation_budget,
)
from aa.corpus.budget import RETRIEVED_PASSAGES_BUDGET_TOKENS, estimate_text_tokens
from aa.grounding.gate import EntailmentFn, check_grounding, default_entails
from aa.grounding.quotes import (
    EvidenceKind,
    EvidenceUnit,
    Provenance,
    QuoteKind,
)
from aa.opencode.errors import (
    OpenCodeError,
    OpenCodeSessionNotFoundError,
    OpenCodeTimeoutError,
    OpenCodeTransientError,
)
from aa.retrieval.book_tools import (
    MAX_EXPAND_AFTER,
    MAX_EXPAND_BEFORE,
    BookToolError,
    book_expand,
    book_read,
)
from aa.retrieval.fusion import MAX_PER_SECTION, enforce_diversity
from aa.retrieval.index import HybridIndex, RetrievalHit, search_aspect, search_plan
from aa.retrieval.normalize import normalize_ru, ru_stem, ru_tokens
from aa.retrieval.planner import (
    MAX_ASPECTS,
    PLANNER_LANGUAGE,
    PlannerError,
    QueryPlan,
    aspect_search_queries,
    validate_plan,
)
from aa.retrieval.planner import (
    SCHEMA_VERSION as PLANNER_SCHEMA_VERSION,
)

logger = logging.getLogger("aa.conversation.orchestrator")

RUNTIME_VERSION = "aa-conversation-runtime/1"
COVERAGE_SCHEMA_VERSION = "aa-coverage-v1"
SUPPORT_SCHEMA_VERSION = "aa-synthesis-support-v1"

AGENT_NAME = "aa"

MAX_PLANNER_ATTEMPTS = 2
MAX_RETRIEVAL_ROUNDS = 2
MAX_REGENERATIONS = 1
MAX_SEND_ATTEMPTS = 3
SEND_RETRY_DELAYS = (0.5, 1.0)
MAX_EVIDENCE_CHUNKS = 8
MAX_READ_PER_TURN = 6

FAIL_CLOSED_REPLY = (
    "Не могу дать обоснованный ответ по имеющимся отрывкам книги. Попробуйте уточнить вопрос."
)

_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")

_EN_FALLBACK_PATTERNS = (
    r"cannot\s+provide",
    r"grounded\s+answer",
    r"available\s+book",
    r"book\s+passages",
    r"refine\s+your\s+question",
    r"please\s+refine",
    r"please\s+try\s+again",
    r"could\s+not\s+process",
    r"bot\s+is\s+ready",
    r"new\s+conversation",
    r"send\s+a\s+message",
    r"i\s+cannot\s+give",
    r"ask\s+.*clarif",
    r"insufficient\s+grounding",
    r"fail[-\s]?closed",
    r"opencode",
    r"model\s+unavailable",
    r"provider\s+error",
    r"http\s*=",
    r"transient",
)

_EN_FALLBACK_RES = tuple(re.compile(item, re.IGNORECASE) for item in _EN_FALLBACK_PATTERNS)


def _strip_citations_for_language_check(text: str) -> str:
    """Remove ``[source/section#chunk]`` pointers before language checks."""
    return _CITATION_RE.sub(" ", text)


def contains_english_fallback(text: str) -> bool:
    """Return whether ``text`` leaks an English error/fallback fragment."""
    cleaned = _strip_citations_for_language_check(text)
    return any(item.search(cleaned) is not None for item in _EN_FALLBACK_RES)


def meets_russian_only(text: str) -> bool:
    """Return whether user-visible ``text`` obeys the RU-only contract.

    Citations are pointers, not prose: they are stripped first. Valid
    Russian text must contain Cyrillic and must not contain any Latin
    (visible English) prose, English error/fallback fragment, or
    internal provider token.
    """
    cleaned = _strip_citations_for_language_check(text)
    if contains_english_fallback(text):
        return False
    if _CYRILLIC_RE.search(cleaned) is None:
        return False
    if _LATIN_RE.search(cleaned) is not None:
        return False
    return True


def ensure_russian_only(text: str) -> None:
    """Raise :class:`TurnFailed` when ``text`` violates the RU-only contract."""
    if not meets_russian_only(text):
        raise TurnFailed("language-violation", "synthesis violated the RU-only contract")


_TRIVIAL_NORMALIZED = frozenset(
    {
        "привет",
        "здравствуйте",
        "здравствуй",
        "добрый день",
        "добрый вечер",
        "спасибо",
        "благодарю",
        "пока",
        "до свидания",
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
        "/start",
        "/new",
        "start",
    }
)

_FOLLOWUP_INTERROGATIVES = (
    "почему",
    "зачем",
    "отчего",
    "откуда",
    "куда",
    "где",
    "когда",
    "сколько",
    "какой",
    "какая",
    "какое",
    "какие",
    "какого",
    "что ",
    "что?",
    "как ",
    "как?",
    "а дальше",
    "а еще",
    "а ещ",
    "а потом",
    "и дальше",
    "и потом",
    "что дальше",
    "ну и",
    "и что",
    "расскажи",
    "продолж",
    "поясни",
    "объясни",
    "уточни",
    "поподробн",
    "правда",
    "серьезн",
    "why",
    "what",
    "how",
    "really",
)

_SUBSTANTIVE_KEYWORDS = (
    "книг",
    "книж",
    "страх",
    "боюсь",
    "боязнь",
    "шаг",
    "инвентар",
    "обид",
    "молитв",
    "медитац",
    "срыв",
    "сорвал",
    "рецидив",
    "тяга",
    "алког",
    "трезв",
    "пьян",
    "пить",
    "бух",
    "выпив",
    "запой",
    "похмел",
    "семь",
    "жен",
    "жинк",
    "женушк",
    "муж",
    "работ",
    "начальник",
    "содруж",
    "сообществ",
    "спонсор",
    "программ",
    "высш",
    "сил",
    "вера",
    "бог",
    "book",
    "fear",
    "step",
    "drink",
    "sober",
    "relapse",
    "craving",
    "family",
)

_SLANG_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "бухаю": ("употребляю алкоголь", "пью каждый вечер", "выпивка"),
    "бухать": ("употреблять алкоголь", "пить", "выпивка"),
    "бухает": ("употребляет алкоголь", "пьет", "выпивка"),
    "нажрался": ("сильное опьянение", "выпил лишнего", "опьянение"),
    "нажираться": ("сильное опьянение", "выпивка", "опьянение"),
    "тяпнул": ("выпил", "рюмка", "небольшая доза алкоголя"),
    "тяпнуть": ("выпить", "рюмка", "алкоголь"),
    "жинка": ("жена", "семья", "супруга"),
    "женушка": ("жена", "семья", "супруга"),
    "жнка": ("жена", "семья", "супруга"),
    "сорвался": ("срыв", "рецидив", "снова начал пить"),
    "сорваться": ("срыв", "рецидив", "возврат к выпивке"),
    "сорвусь": ("срыв", "рецидив", "страх срыва"),
    "запой": ("запой", "длительная выпивка", "потеря контроля"),
    "похмелье": ("похмелье", "утро после выпивки", "последствия выпивки"),
    "тяга": ("тяга", "навязчивое желание выпить", "одержимость"),
    "пьянка": ("пьянство", "выпивка", "семейный конфликт из-за выпивки"),
    "пьянки": ("пьянство", "выпивка", "семья"),
}

_THEME_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "drinking",
        ("алкогол", "трезв", "пьян", "пить", "бух", "выпив", "тяга", "срыв", "сорвал", "запой"),
    ),
    ("family", ("семь", "жен", "жинк", "женушк", "муж", "дом", "развод", "дет")),
    ("work", ("работ", "начальник", "работодател", "увольн", "коллег")),
    ("fear", ("страх", "боюсь", "боязнь", "тревог", "опасен")),
    ("inventory", ("инвентар", "обид", "четверт", "шаг")),
    ("faith", ("высш", "сил", "вера", "молитв", "медитац", "агностик")),
    ("fellowship", ("содруж", "сообществ", "групп", "собран", "спонсор", "служени")),
)

# Coverage-oriented additive queries for broad/general turns (issue #98).
# Same-language generic program vocabulary only: no factual strengthening,
# no diagnosis, no loss-of-control/divorce/medical assumptions. Used to
# sample distinct book regions when the user wording carries no specific
# theme marker, so a broad query does not collapse to one top hit.
_BROAD_COVERAGE_QUERIES: dict[str, tuple[str, ...]] = {
    "general": (),
    "program": ("программа трезвость шаги", "трезвость программа выздоровление"),
    "support": ("сообщество поддержка помощь", "помощь сообщества трезвость"),
}

_CITATION_RE = re.compile(r"\[([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+(?:#[A-Za-z0-9_:.\-]+)?)\]")
# A period followed by a citation belongs to the preceding claim: the split
# must not strand ``[source/section#chunk]`` away from its sentence.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+(?!\s*\[)|(?<=\])\s+|\n+")
_QUOTED_SPAN_RE = re.compile(r"[«\"„]([^«»\"]{8,400})[»\"“]")


class TurnFailed(ValueError):
    """Deterministic fail-closed turn failure (caught by the app boundary)."""

    def __init__(self, category: str, detail: str = "") -> None:
        super().__init__(f"turn failed [{category}]" + (f": {detail}" if detail else ""))
        self.category = category
        self.detail = detail


@dataclass(frozen=True)
class CoverageResult:
    """Outcome of the deterministic coverage check."""

    covered: bool
    gaps: tuple[str, ...]
    distinct_sections: int
    aspects_covered: int
    aspects_total: int


@dataclass(frozen=True)
class EvidencePack:
    """Exact-RU evidence for one turn (never planner/search metadata)."""

    units: tuple[EvidenceUnit, ...]
    token_count: int
    source_ids: tuple[str, ...]
    locators: tuple[str, ...]
    checksums: tuple[str, ...]
    ru_corpus_version: str
    retrieval_rounds: int
    tool_call_count: int

    def by_locator(self) -> dict[str, EvidenceUnit]:
        """Index source-exact units by ``source_id/section#chunk`` locators."""
        mapping: dict[str, EvidenceUnit] = {}
        for unit in self.units:
            provenance = unit.provenance
            if not provenance.chunk_id:
                continue
            short = f"{provenance.source_id}/{provenance.section_id}#{provenance.chunk_id}"
            mapping[short] = unit
        return mapping


@dataclass(frozen=True)
class AnswerUnit:
    """One answer sentence plus its parsed citations."""

    unit_id: str
    text: str
    citations: tuple[str, ...]
    substantive: bool


@dataclass(frozen=True)
class ResponseUnit:
    """One user-visible unit with grounding provenance for #83.

    ``priority`` is zero-based in answer order: lower values are more
    important, so downstream compaction (#83) drops higher values first
    while retained units keep their validated grounding.
    """

    unit_id: str
    text: str
    kind: str
    priority: int
    cited_locators: tuple[str, ...]
    grounding_passed: bool | None
    source_exact: bool


@dataclass(frozen=True)
class TurnDiagnostics:
    """Privacy-safe turn metrics (no user/corpus text, no secrets)."""

    substantive: bool
    aspects: int
    retrieval_rounds: int
    candidates: int
    evidence_chunks: int
    evidence_tokens: int
    tool_call_count: int
    coverage_gaps: tuple[str, ...]
    regeneration_count: int
    grounding_passed: bool | None
    served_model: str
    fallback_used: bool
    error_category: str
    retry_count: int


@dataclass(frozen=True)
class GroundedResponse:
    """Structured grounded output owned by #9 for downstream #83."""

    text: str
    units: tuple[ResponseUnit, ...]
    evidence: EvidencePack | None
    diagnostics: TurnDiagnostics

    def compact(self, *, max_units: int) -> GroundedResponse:
        """Return a shorter response keeping top-priority grounded units.

        Lower-priority units are dropped without inventing content; every
        retained unit keeps its validated provenance and grounding verdict.
        """
        if max_units < 0:
            raise TurnFailed("invalid-compact", "max_units must be >= 0")
        kept = self.units[:max_units]
        text = " ".join(unit.text for unit in kept)
        diagnostics = TurnDiagnostics(
            substantive=self.diagnostics.substantive,
            aspects=self.diagnostics.aspects,
            retrieval_rounds=self.diagnostics.retrieval_rounds,
            candidates=self.diagnostics.candidates,
            evidence_chunks=self.diagnostics.evidence_chunks,
            evidence_tokens=self.diagnostics.evidence_tokens,
            tool_call_count=self.diagnostics.tool_call_count,
            coverage_gaps=self.diagnostics.coverage_gaps,
            regeneration_count=self.diagnostics.regeneration_count,
            grounding_passed=self.diagnostics.grounding_passed,
            served_model=self.diagnostics.served_model,
            fallback_used=self.diagnostics.fallback_used,
            error_category=self.diagnostics.error_category,
            retry_count=self.diagnostics.retry_count,
        )
        return GroundedResponse(
            text=text, units=kept, evidence=self.evidence, diagnostics=diagnostics
        )


@dataclass(frozen=True)
class SynthesisResult:
    """Raw synthesis output plus transport metadata."""

    text: str
    served_model: str
    fallback_used: bool
    error_category: str
    retry_count: int


def is_substantive(text: str) -> bool:
    """Return whether ``text`` needs the full grounded pipeline.

    Empty text is not substantive (the safety layer blocks it first).
    Exact trivial greetings/thanks bypass retrieval; short follow-up
    questions and interrogatives (``почему?``, ``а дальше?``) always
    take the full state machine so no substantive turn skips grounding.
    """
    stripped = text.strip()
    if not stripped:
        return False
    normalized = normalize_ru(stripped)
    if normalized in _TRIVIAL_NORMALIZED:
        return False
    if " ".join(ru_tokens(stripped)) in _TRIVIAL_NORMALIZED:
        return False
    if "?" in stripped or "？" in stripped:
        return True
    if any(marker in normalized for marker in _FOLLOWUP_INTERROGATIVES):
        return True
    if len(normalized) <= 24 and not any(key in normalized for key in _SUBSTANTIVE_KEYWORDS):
        return False
    return True


def _expand_token_queries(token: str) -> list[str]:
    queries: list[str] = []
    stem = ru_stem(token)
    if stem and stem != token:
        queries.append(stem)
    for slang, expansions in _SLANG_EXPANSIONS.items():
        if token == slang or stem == ru_stem(slang):
            queries.extend(expansions)
    return queries


def _detect_themes(normalized: str) -> list[str]:
    themes: list[str] = []
    for theme, markers in _THEME_MARKERS:
        if any(marker in normalized for marker in markers):
            themes.append(theme)
    return themes


def build_local_plan_payload(text: str, *, utterance_id: str = "turn-1") -> dict[str, Any]:
    """Build a deterministic slang-aware planner JSON payload (RU-only).

    The original wording is preserved by the caller in ``validate_plan``;
    rewrites are additive same-language expansions that never strengthen
    the user's factual meaning. ``lexical_query_en`` always stays null.
    """
    stripped = text.strip()
    if not stripped:
        raise PlannerError("original_query must be a non-empty string")
    normalized = normalize_ru(stripped)
    tokens = ru_tokens(stripped)
    themes = _detect_themes(normalized)
    if not themes:
        # Broad/coverage-oriented turn: no specific theme marker. Plan
        # multiple whole-book searches so retrieval samples distinct
        # regions instead of collapsing to one top hit (issue #98).
        themes = ["general", "program", "support"]

    ambiguity = "none"
    if any(token in ("сорвался", "сорвусь", "сорваться") for token in tokens):
        ambiguity = "material"
    elif len(stripped) <= 24:
        ambiguity = "low"

    aspects: list[dict[str, Any]] = []
    for position, theme in enumerate(themes[:MAX_ASPECTS]):
        rewrites: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            for query in _expand_token_queries(token):
                key = query.casefold()
                if key not in seen and key != normalized:
                    seen.add(key)
                    rewrites.append(query)
        lexical: list[str] = [stripped]
        for rewrite in rewrites:
            if len(lexical) >= 4:
                break
            lexical.append(rewrite)
        for extra in _BROAD_COVERAGE_QUERIES.get(theme, ()):
            if len(lexical) >= 4:
                break
            if extra.casefold() not in {item.casefold() for item in lexical}:
                lexical.append(extra)
        semantic: list[str] = [stripped]
        if normalized != stripped:
            semantic.append(normalized)
        stemmed = " ".join(ru_stem(token) for token in tokens if token)
        if stemmed and stemmed not in semantic and len(semantic) < 4:
            semantic.append(stemmed)
        for extra in _BROAD_COVERAGE_QUERIES.get(theme, ()):
            if len(semantic) >= 4:
                break
            if extra.casefold() not in {item.casefold() for item in semantic}:
                semantic.append(extra)
        aspects.append(
            {
                "aspect_id": theme if len(themes) > 1 else "main",
                "meaning": f"поиск по теме {theme} без усиления утверждений пользователя",
                "semantic_queries_ru": semantic[:4],
                "lexical_queries_ru": lexical[:4],
                "lexical_query_en": None,
                "ambiguity": ambiguity if position == 0 else "low",
                "forbidden_inferences": [
                    "do_not_diagnose",
                    "do_not_assume_loss_of_control",
                    "do_not_assume_divorce_or_abuse",
                    "do_not_assume_medical_facts",
                ],
            }
        )
    return {
        "schema_version": PLANNER_SCHEMA_VERSION,
        "utterance_id": utterance_id,
        "language": PLANNER_LANGUAGE,
        "aspects": aspects,
    }


PlannerFn = Callable[[str], Mapping[str, Any] | Awaitable[Mapping[str, Any]]]


def _coerce_mapping(payload: object, *, owner: str) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PlannerError(f"{owner} planner output must be a JSON object")
    return dict(payload)


async def run_planner(
    text: str,
    *,
    planner_fn: PlannerFn | None = None,
    utterance_id: str = "turn-1",
) -> QueryPlan:
    """Validate planner output with one bounded retry, otherwise fail closed."""
    last_error: Exception | None = None
    for _attempt in range(MAX_PLANNER_ATTEMPTS):
        try:
            if planner_fn is None:
                payload: Mapping[str, Any] = build_local_plan_payload(
                    text, utterance_id=utterance_id
                )
            else:
                try:
                    produced = planner_fn(text)
                    if isinstance(produced, Awaitable):
                        produced = await produced
                    payload = _coerce_mapping(produced, owner="planner")
                except PlannerError:
                    raise
                except Exception as exc:
                    raise PlannerError(f"planner failed: {exc}") from exc
            return validate_plan(dict(payload), original_query=text)
        except PlannerError as exc:
            last_error = exc
            continue
    raise TurnFailed("planner-invalid", str(last_error) if last_error else "invalid plan")


def validate_coverage_payload(payload: object) -> CoverageResult:
    """Validate a coverage JSON payload (fails closed on malformed)."""
    if not isinstance(payload, Mapping):
        raise TurnFailed("coverage-invalid", "coverage output must be a JSON object")
    if payload.get("schema_version") != COVERAGE_SCHEMA_VERSION:
        raise TurnFailed("coverage-invalid", "unexpected coverage schema_version")
    covered = payload.get("covered")
    if not isinstance(covered, bool):
        raise TurnFailed("coverage-invalid", "covered must be a boolean")
    gaps_raw = payload.get("gaps", [])
    if not isinstance(gaps_raw, list) or any(not isinstance(item, str) for item in gaps_raw):
        raise TurnFailed("coverage-invalid", "gaps must be a list of strings")
    for key in ("distinct_sections", "aspects_covered", "aspects_total"):
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TurnFailed("coverage-invalid", f"{key} must be a non-negative integer")
    return CoverageResult(
        covered=covered,
        gaps=tuple(str(item) for item in gaps_raw),
        distinct_sections=int(payload["distinct_sections"]),
        aspects_covered=int(payload["aspects_covered"]),
        aspects_total=int(payload["aspects_total"]),
    )


def validate_support_payload(payload: object) -> list[dict[str, Any]]:
    """Validate a synthesis-support mapping payload (fails closed on malformed)."""
    if not isinstance(payload, Mapping):
        raise TurnFailed("support-invalid", "support output must be a JSON object")
    if payload.get("schema_version") != SUPPORT_SCHEMA_VERSION:
        raise TurnFailed("support-invalid", "unexpected support schema_version")
    raw_units = payload.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise TurnFailed("support-invalid", "units must be a non-empty list")
    cleaned: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_units):
        owner = f"support.units[{position}]"
        if not isinstance(raw, Mapping):
            raise TurnFailed("support-invalid", f"{owner} must be an object")
        unit_id = raw.get("unit_id")
        claim = raw.get("claim")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise TurnFailed("support-invalid", f"{owner}.unit_id must be non-empty")
        if not isinstance(claim, str) or not claim.strip():
            raise TurnFailed("support-invalid", f"{owner}.claim must be non-empty")
        cited = raw.get("cited", [])
        if not isinstance(cited, list) or any(not isinstance(item, str) for item in cited):
            raise TurnFailed("support-invalid", f"{owner}.cited must be a list of strings")
        kind = raw.get("quote_kind", "prose")
        if kind not in ("exact-source", "prose"):
            raise TurnFailed("support-invalid", f"{owner}.quote_kind must be exact-source|prose")
        cleaned.append(
            {
                "unit_id": unit_id.strip(),
                "claim": claim.strip(),
                "cited": [str(item) for item in cited],
                "quote_kind": str(kind),
            }
        )
    return cleaned


def search_first_round(index: HybridIndex, plan: QueryPlan) -> dict[str, list[RetrievalHit]]:
    """Run RU-first whole-corpus search for every planned aspect."""
    try:
        return search_plan(index, plan)
    except (ValueError, OSError, RuntimeError) as exc:
        raise TurnFailed("retrieval-failed", str(exc)) from exc


def deduplicate_cross_aspect(
    hits_by_aspect: Mapping[str, Sequence[RetrievalHit]],
    *,
    max_total: int = 24,
) -> list[RetrievalHit]:
    """Deterministic cross-aspect dedup/diversity (RU candidates only)."""
    ordered: list[RetrievalHit] = []
    seen: set[str] = set()
    for aspect_id in sorted(hits_by_aspect):
        for hit in hits_by_aspect[aspect_id]:
            if hit.logical_chunk_id in seen:
                continue
            if ":ru:" not in hit.chunk_id:
                raise TurnFailed("retrieval-non-ru", "non-RU candidate in RU-first search")
            seen.add(hit.logical_chunk_id)
            ordered.append(hit)
    ordered.sort(key=lambda item: item.fused_score, reverse=True)
    sections = {hit.chunk_id: hit.section for hit in ordered}
    if not ordered:
        return []
    from aa.retrieval.fusion import FusedCandidate, deduplicate_overlaps

    fused = [
        FusedCandidate(
            chunk_id=hit.chunk_id,
            fused_score=hit.fused_score,
            lexical_rank=hit.lexical_rank,
            dense_rank=hit.dense_rank,
            lexical_score=hit.lexical_score,
            dense_score=hit.dense_score,
        )
        for hit in ordered
    ]
    spans = {hit.chunk_id: (hit.section, hit.char_start, hit.char_end) for hit in ordered}
    try:
        deduped_ids = {item.chunk_id for item in deduplicate_overlaps(fused, spans=spans)}
    except (ValueError, OSError, RuntimeError) as exc:
        raise TurnFailed("retrieval-failed", str(exc)) from exc
    filtered = [hit for hit in ordered if hit.chunk_id in deduped_ids]
    if not filtered:
        return []
    try:
        diverse = enforce_diversity(
            [
                FusedCandidate(
                    chunk_id=hit.chunk_id,
                    fused_score=hit.fused_score,
                    lexical_rank=hit.lexical_rank,
                    dense_rank=hit.dense_rank,
                    lexical_score=hit.lexical_score,
                    dense_score=hit.dense_score,
                )
                for hit in filtered
            ],
            sections=sections,
            max_n=min(max_total, len(filtered)),
            max_per_section=MAX_PER_SECTION,
        )
    except (ValueError, OSError, RuntimeError) as exc:
        raise TurnFailed("retrieval-failed", str(exc)) from exc
    wanted = {item.chunk_id for item in diverse}
    result = [hit for hit in filtered if hit.chunk_id in wanted]
    result.sort(key=lambda item: item.fused_score, reverse=True)
    return result


def check_coverage(plan: QueryPlan, merged: Sequence[RetrievalHit]) -> CoverageResult:
    """Deterministic coverage check over merged RU candidates."""
    sections = {hit.section for hit in merged}
    gaps: list[str] = []
    if plan.aspects and len(plan.aspects) > 1 and len(sections) < 2:
        gaps.append("multi-aspect turn collapsed to a single section")
    if not merged:
        gaps.append("no candidates retrieved")
    if plan.aspects and len(merged) < len(plan.aspects):
        gaps.append("fewer candidates than planned aspects")
    aspect_ids = {aspect.aspect_id for aspect in plan.aspects}
    broad_ids = {"general", "main", "program", "support"}
    if merged and len(sections) < 2 and (aspect_ids & broad_ids):
        gaps.append("broad turn needs evidence from at least two sections")
    if merged and len(plan.aspects) > 1 and len(sections) < min(3, len(plan.aspects)):
        gaps.append("coverage-oriented turn needs at least three distinct sections")
    payload: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA_VERSION,
        "covered": not gaps,
        "gaps": gaps,
        "distinct_sections": len(sections),
        "aspects_covered": min(len(sections), len(plan.aspects)),
        "aspects_total": len(plan.aspects),
    }
    return validate_coverage_payload(payload)


def _provenance_for_hit(hit: RetrievalHit, *, ru_corpus_version: str) -> Provenance:
    return Provenance(
        corpus_version=ru_corpus_version,
        source_id=hit.source_id,
        section_id=hit.section,
        chunk_id=hit.logical_chunk_id,
        char_start=hit.char_start,
        char_end=hit.char_end,
        source_checksum=hit.text_sha256,
        source_language="ru",
    )


def load_exact_evidence(
    index: HybridIndex,
    merged: Sequence[RetrievalHit],
    *,
    ru_corpus_version: str,
    expand_multi_aspect: bool = False,
) -> tuple[EvidencePack, int]:
    """Load exact RU passages via ``book_read``/bounded ``book_expand``.

    Returns ``(evidence_pack_without_budget, tool_call_count)``. The
    source/context budget is applied by :func:`fit_evidence_budget`.
    Planner/search metadata never enters the pack: only ``SOURCE_TEXT``
    units with version-pinned provenance do.
    """
    if not ru_corpus_version.strip():
        raise TurnFailed("evidence-unpinned", "RU corpus version must be pinned")
    live_version = str(index.metadata.get("ru_artifact_sha256", ""))
    if not live_version:
        raise TurnFailed("evidence-unpinned", "opened index carries no RU version")
    tool_calls = 0
    collected: list[EvidenceUnit] = []
    seen: set[str] = set()
    for hit in list(merged)[:MAX_READ_PER_TURN]:
        if hit.logical_chunk_id in seen:
            continue
        try:
            read = book_read(index, hit.logical_chunk_id, expected_ru_version=live_version)
        except (BookToolError, ValueError) as exc:
            raise TurnFailed("evidence-read", str(exc)) from exc
        tool_calls += 1
        text = str(read.get("text", ""))
        locator = read.get("ru_locator", {})
        if not isinstance(locator, dict):
            raise TurnFailed("evidence-read", "book_read returned no RU locator")
        if not text.strip():
            raise TurnFailed("evidence-read", "book_read returned empty RU text")
        if hashlib.sha256(text.encode("utf-8")).hexdigest() != str(locator.get("text_sha256", "")):
            raise TurnFailed("evidence-checksum", "RU chunk checksum mismatch")
        provenance = Provenance(
            corpus_version=ru_corpus_version,
            source_id=str(locator.get("source_id", hit.source_id)),
            section_id=hit.section,
            chunk_id=hit.logical_chunk_id,
            char_start=int(locator.get("char_start", hit.char_start)),
            char_end=int(locator.get("char_end", hit.char_end)),
            source_checksum=str(locator.get("text_sha256", "")),
            source_language="ru",
        )
        try:
            provenance.validate()
        except ValueError as exc:
            raise TurnFailed("evidence-provenance", str(exc)) from exc
        seen.add(hit.logical_chunk_id)
        collected.append(
            EvidenceUnit(
                kind=EvidenceKind.SOURCE_TEXT,
                language="ru",
                text=text,
                provenance=provenance,
            )
        )
    if expand_multi_aspect and merged:
        center = merged[0]
        try:
            expanded = book_expand(
                index,
                center.logical_chunk_id,
                before=min(1, MAX_EXPAND_BEFORE),
                after=min(1, MAX_EXPAND_AFTER),
                expected_ru_version=live_version,
            )
        except (BookToolError, ValueError) as exc:
            raise TurnFailed("evidence-expand", str(exc)) from exc
        tool_calls += 1
        chunks = expanded.get("chunks", [])
        if isinstance(chunks, list):
            for item in chunks:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", ""))
                locator = item.get("ru_locator", {})
                if not isinstance(locator, dict) or not text.strip():
                    continue
                chunk_id = str(item.get("logical_chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                expected_digest = str(locator.get("text_sha256", ""))
                actual_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if not expected_digest or actual_digest != expected_digest:
                    raise TurnFailed("evidence-checksum", "RU expanded chunk checksum mismatch")
                provenance = Provenance(
                    corpus_version=ru_corpus_version,
                    source_id=str(locator.get("source_id", center.source_id)),
                    section_id=str(item.get("section", center.section)),
                    chunk_id=chunk_id,
                    char_start=int(locator.get("char_start", 0)),
                    char_end=int(locator.get("char_end", 0)),
                    source_checksum=str(locator.get("text_sha256", "")),
                    source_language="ru",
                )
                try:
                    provenance.validate()
                except ValueError as exc:
                    raise TurnFailed("evidence-provenance", str(exc)) from exc
                seen.add(chunk_id)
                collected.append(
                    EvidenceUnit(
                        kind=EvidenceKind.SOURCE_TEXT,
                        language="ru",
                        text=text,
                        provenance=provenance,
                    )
                )
                if len(collected) >= MAX_EVIDENCE_CHUNKS:
                    break
    if not collected:
        raise TurnFailed("evidence-empty", "no exact RU evidence could be loaded")
    for unit in collected:
        if unit.language != "ru" or not unit.is_source_text:
            raise TurnFailed("evidence-non-ru", "evidence pack must be exact RU source text")
    token_count = sum(estimate_text_tokens(unit.text) for unit in collected)
    pack = EvidencePack(
        units=tuple(collected),
        token_count=token_count,
        source_ids=tuple(sorted({unit.provenance.source_id for unit in collected})),
        locators=tuple(
            f"{unit.provenance.source_id}/{unit.provenance.section_id}#{unit.provenance.chunk_id}"
            for unit in collected
        ),
        checksums=tuple(unit.provenance.source_checksum for unit in collected),
        ru_corpus_version=ru_corpus_version,
        retrieval_rounds=1,
        tool_call_count=tool_calls,
    )
    return pack, tool_calls


def fit_evidence_budget(pack: EvidencePack, *, budget_tokens: int) -> EvidencePack:
    """Keep the highest-priority prefix of ``pack`` that fits the budget.

    Atomic per passage: a passage either fits entirely or is excluded by
    truncating the tail. Dropping lower-priority tail units preserves the
    grounding of every retained unit for downstream #83 compaction.
    """
    if budget_tokens <= 0:
        raise TurnFailed("budget-invalid", "evidence budget must be > 0")
    kept: list[EvidenceUnit] = []
    total = 0
    for unit in pack.units:
        need = estimate_text_tokens(unit.text)
        if not kept and need > budget_tokens:
            raise TurnFailed(
                "budget-exceeded",
                "top evidence passage exceeds the retrieved-passages budget",
            )
        if total + need > budget_tokens:
            break
        kept.append(unit)
        total += need
    if not kept:
        raise TurnFailed("budget-exceeded", "no evidence passage fits the budget")
    return EvidencePack(
        units=tuple(kept),
        token_count=total,
        source_ids=tuple(sorted({unit.provenance.source_id for unit in kept})),
        locators=tuple(
            f"{unit.provenance.source_id}/{unit.provenance.section_id}#{unit.provenance.chunk_id}"
            for unit in kept
        ),
        checksums=tuple(unit.provenance.source_checksum for unit in kept),
        ru_corpus_version=pack.ru_corpus_version,
        retrieval_rounds=pack.retrieval_rounds,
        tool_call_count=pack.tool_call_count,
    )


def build_synthesis_prompt(
    *, user_text: str, pack: EvidencePack, repair: str = "", generation_budget_tokens: int = 0
) -> str:
    """Build the synthesis prompt sent to the named ``aa`` agent.

    The prompt carries the validated evidence pack with pinned provenance
    and requires citations per claim. Planner/search metadata is never
    included. The caller must never log the returned prompt.

    The prompt also carries the concise output policy (#83): short
    conversational answers, the ordinary target, the hard envelope, and
    the anti-corpus-dump rule. ``generation_budget_tokens`` is an
    efficiency hint only (``0`` selects the conservative default); the
    deterministic character validator stays authoritative.
    """
    try:
        budget = resolve_generation_budget(generation_budget_tokens)
    except ValueError:
        budget = resolve_generation_budget(0)
    lines: list[str] = [
        "Ответь ТОЛЬКО по-русски, используя ТОЛЬКО приведённые ниже точные отрывки.",
        "Весь видимый ответ — на русском языке; английский текст запрещён.",
        "Каждое существенное утверждение снабди цитатой-ссылкой вида [source/section#chunk].",
        "Прямые цитаты — дословный русский текст отрывков без изменений.",
        "Если отрывки не подтверждают просьбу, так и скажи и предложи только близкий",
        "подтверждённый материал. Не выдумывай факты и цитаты.",
        "Если обоснованный ответ невозможен, ответи строго: "
        "Не могу дать обоснованный ответ по имеющимся отрывкам книги. "
        "Попробуйте уточнить вопрос.",
        "Никогда не отвечай по-английски и не сообщай технические детали.",
        "",
        "ФОРМАТ ОТВЕТА (обязательно): отвечай кратко, как в переписке, "
        "обычно 2-5 коротких предложений. Обычная цель: не более "
        f"{TARGET_CHARS} символов и {TARGET_WORDS} слов. "
        f"Жёсткий предел: не более {HARD_CHARS} символов и {HARD_WORDS} слов "
        "в одном сообщении. Один главный смысл и не более одного "
        "уточняющего вопроса. Не дели ответ на несколько сообщений.",
        "НИКОГДА не воспроизводи главу, раздел или длинный связный отрывок "
        "целиком. Просьбу выдать главу/большой кусок преврати в краткое "
        "изложение своими словами и при необходимости добавь одну короткую "
        "точную цитату. Суммарно все дословные цитаты корпуса в ответе — "
        f"не более {QUOTE_BUDGET_CHARS} символов.",
        generation_budget_instruction(budget),
    ]
    if repair:
        lines.extend(["", "ИСПРАВЛЕНИЕ (обязательно):", repair])
    if is_bulk_reproduction_request(user_text) or is_continuation_request(user_text):
        lines.append(
            "Просьба пользователя касается выдачи главы/большого куска или "
            "продолжения печати: останься в режиме краткого обсуждения — "
            "дай сжатое изложение и предложи разобрать конкретную тему. "
            "Не листай канонический текст по частям."
        )
    lines.extend(["", "ТОЧНЫЕ ОТРЫВКИ:"])
    for unit in pack.units:
        provenance = unit.provenance
        lines.append(
            f"[{provenance.source_id}/{provenance.section_id}#{provenance.chunk_id}] {unit.text}"
        )
    lines.extend(["", "ВОПРОС ПОЛЬЗОВАТЕЛЯ:", user_text])
    return "\n".join(lines)


_CITATION_ONLY_RE = re.compile(r"^(?:\[[^\]]+\]\s*)+$")


def split_answer_units(answer: str) -> list[AnswerUnit]:
    """Split synthesis text into units and parse per-unit citations.

    A fragment holding only citation brackets is re-attached to the
    preceding unit so ``Sentence. [source/section#chunk]`` stays one
    citable claim instead of an orphaned uncited sentence.
    """
    parts = [item.strip() for item in _SENTENCE_SPLIT_RE.split(answer) if item.strip()]
    merged_parts: list[str] = []
    for part in parts:
        if _CITATION_ONLY_RE.match(part) and merged_parts:
            merged_parts[-1] = f"{merged_parts[-1]} {part}"
        else:
            merged_parts.append(part)
    units: list[AnswerUnit] = []
    for position, part in enumerate(merged_parts):
        citations = tuple(_CITATION_RE.findall(part))
        units.append(
            AnswerUnit(
                unit_id=f"u{position + 1}",
                text=part,
                citations=citations,
                substantive=_is_substantive_unit(part),
            )
        )
    return units


def _significant_tokens(text: str) -> set[str]:
    return {token for token in ru_tokens(text) if len(token) >= 4}


def _is_substantive_unit(text: str) -> bool:
    """Return whether an answer fragment needs semantic grounding.

    Fail closed: every fragment carrying any alphanumeric claim text --
    including short factual or imperative claims -- is substantive and
    must pass citation/entailment validation. Only empty,
    citation-only, or punctuation-only scaffolding is boilerplate.
    """
    cleaned = _CITATION_RE.sub("", text).strip()
    if not cleaned:
        return False
    return any(ch.isalnum() for ch in cleaned)


def _quote_kind_for_unit(unit: AnswerUnit, pack: EvidencePack) -> QuoteKind:
    cleaned = _CITATION_RE.sub("", unit.text).strip()
    for evidence in pack.units:
        if cleaned and cleaned in evidence.text:
            return QuoteKind.EXACT_SOURCE
        for match in _QUOTED_SPAN_RE.findall(unit.text):
            if match.strip() and match.strip() in evidence.text:
                return QuoteKind.EXACT_SOURCE
    return QuoteKind.TRANSLATION


def _resolve_cited(
    citations: Sequence[str], pack: EvidencePack
) -> tuple[list[Provenance], list[EvidenceUnit], list[str]]:
    mapping = pack.by_locator()
    provenances: list[Provenance] = []
    units: list[EvidenceUnit] = []
    validated: list[str] = []
    for citation in citations:
        if "#" not in citation:
            continue
        unit = mapping.get(citation)
        if unit is None:
            continue
        provenances.append(unit.provenance)
        units.append(unit)
        validated.append(citation)
    return provenances, units, validated


def judge_unit(
    unit: AnswerUnit,
    pack: EvidencePack,
    *,
    entails: EntailmentFn | None = None,
) -> tuple[bool, bool, QuoteKind, tuple[str, ...]]:
    """Judge one answer unit; return ``(passed, source_exact, kind, cited)``."""
    if not unit.substantive:
        return True, False, QuoteKind.TRANSLATION, ()
    provenances, cited_units, validated = _resolve_cited(unit.citations, pack)
    if not provenances:
        return False, False, QuoteKind.TRANSLATION, ()
    quoted = _CITATION_RE.sub("", unit.text).strip()
    kind = _quote_kind_for_unit(unit, pack)
    # Citations are provenance pointers, not claim content: semantic support
    # is judged on the citation-stripped claim so locator tokens can never
    # dilute same-language overlap or smuggle cross-language support.
    if kind is QuoteKind.TRANSLATION:
        # Same-language Russian prose (own-words summary required by the
        # anti-corpus-dump policy): grounded by stemmed entailment against
        # the cited source text. No translation label is required here;
        # cross-language translation policy stays in ``check_grounding``.
        # Language purity is enforced separately by ``ensure_russian_only``.
        support_text = " ".join(item.text for item in cited_units)
        if not quoted.strip() or not support_text.strip():
            return False, False, kind, tuple(validated)
        judge = entails if entails is not None else default_entails
        passed = bool(judge(quoted, support_text))
        return passed, False, kind, tuple(validated)
    verdict = check_grounding(
        russian_claim=quoted,
        quoted_text=quoted,
        quote_kind=kind,
        cited=provenances,
        evidence=list(pack.units),
        ru_corpus_available=True,
        allow_translation_fallback=False,
        entails=entails if entails is not None else default_entails,
    )
    return verdict.passed, verdict.source_exact, kind, tuple(validated)


def build_grounded_response(
    *,
    answer: str,
    pack: EvidencePack,
    diagnostics: TurnDiagnostics,
    entails: EntailmentFn | None = None,
) -> GroundedResponse:
    """Validate every substantive unit and build the #83-ready response."""
    split = split_answer_units(answer)
    if not split:
        raise TurnFailed("synthesis-empty", "synthesis returned no answer units")
    mapping = pack.by_locator()
    units: list[ResponseUnit] = []
    all_passed = True
    for priority, item in enumerate(split):
        if not item.substantive:
            units.append(
                ResponseUnit(
                    unit_id=item.unit_id,
                    text=item.text,
                    kind="boilerplate",
                    priority=priority,
                    cited_locators=(),
                    grounding_passed=None,
                    source_exact=False,
                )
            )
            continue
        passed, source_exact, kind, cited = judge_unit(item, pack, entails=entails)
        if not passed:
            all_passed = False
        kind_name = "exact-quote" if kind is QuoteKind.EXACT_SOURCE else "prose"
        units.append(
            ResponseUnit(
                unit_id=item.unit_id,
                text=item.text,
                kind=kind_name,
                priority=priority,
                cited_locators=cited,
                grounding_passed=passed,
                source_exact=source_exact,
            )
        )
    _ = mapping
    merged = TurnDiagnostics(
        substantive=diagnostics.substantive,
        aspects=diagnostics.aspects,
        retrieval_rounds=diagnostics.retrieval_rounds,
        candidates=diagnostics.candidates,
        evidence_chunks=len(pack.units),
        evidence_tokens=pack.token_count,
        tool_call_count=diagnostics.tool_call_count,
        coverage_gaps=diagnostics.coverage_gaps,
        regeneration_count=diagnostics.regeneration_count,
        grounding_passed=all_passed,
        served_model=diagnostics.served_model,
        fallback_used=diagnostics.fallback_used,
        error_category=diagnostics.error_category,
        retry_count=diagnostics.retry_count,
    )
    return GroundedResponse(text=answer, units=tuple(units), evidence=pack, diagnostics=merged)


def response_quote_chars(response: GroundedResponse) -> int:
    """Return aggregate verbatim quoted characters in a grounded response."""
    return aggregate_quote_chars(response.text)


def response_envelope_ok(response: GroundedResponse) -> bool:
    """Whether a grounded response fits the hard Telegram envelope."""
    return envelope_passes(response.text)


def compact_grounded_response(response: GroundedResponse) -> GroundedResponse:
    """Deterministically keep leading complete units that fit the hard cap.

    Lower-priority (trailing) units are dropped first; every retained unit
    keeps its validated provenance and grounding verdict. Cuts happen only
    at complete-unit boundaries, so quotations, Markdown constructs, URLs,
    and combining sequences stay intact. Raises :class:`TurnFailed` when
    even the first unit cannot fit (the caller fails closed to a bounded
    operational reply instead of emitting partial text).
    """
    if envelope_passes(response.text):
        return response
    kept: list[ResponseUnit] = []
    for unit in response.units:
        candidate_units = [*kept, unit]
        candidate_text = " ".join(item.text for item in candidate_units)
        if not envelope_passes(candidate_text):
            break
        kept.append(unit)
    if not kept:
        raise TurnFailed("output-envelope", "no complete unit fits the hard envelope")
    compacted_text = " ".join(item.text for item in kept)
    if compact_text_to_envelope(response.text) == ENVELOPE_FALLBACK_REPLY:
        # String-level compaction agrees nothing beyond the fallback fits;
        # unit-level kept prefix is still authoritative when it passes.
        if not envelope_passes(compacted_text):
            raise TurnFailed("output-envelope", "no complete unit fits the hard envelope")
    logger.info(
        "grounded response compacted to envelope",
        extra={
            "kept_units": len(kept),
            "total_units": len(response.units),
            "graphemes": count_graphemes(compacted_text),
            "words": count_words(compacted_text),
            "quoted": aggregate_quote_chars(compacted_text),
        },
    )
    return GroundedResponse(
        text=compacted_text,
        units=tuple(kept),
        evidence=response.evidence,
        diagnostics=response.diagnostics,
    )


async def enforce_grounded_envelope(
    response: GroundedResponse,
    *,
    user_text: str,
    pack: EvidencePack,
    session_id: str,
    send: Callable[..., Awaitable[str]],
    agent: str,
    primary_model: str,
    fallback_model: str,
    entails: EntailmentFn | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    generation_budget_tokens: int = 0,
) -> tuple[GroundedResponse, int]:
    """Enforce the #83 envelope with at most one compact regeneration.

    Uses the same validated ``pack`` (no new retrieval, no new claims)
    and an explicit remaining size budget. Returns ``(response, extra_sends)``
    where ``extra_sends`` is ``0`` (first answer fit) or ``1`` (exactly one
    compact regeneration was performed). When the regenerated answer still
    exceeds the hard envelope, deterministic complete-unit compaction
    applies. No synthetic user turn enters Telegram-visible history: the
    repair is an internal synthesis retry on the same OpenCode session.
    """
    if envelope_passes(response.text):
        return response, 0
    logger.info(
        "grounded answer exceeds envelope; compact regeneration",
        extra={
            "graphemes": count_graphemes(response.text),
            "words": count_words(response.text),
            "quoted": aggregate_quote_chars(response.text),
        },
    )
    repair = compact_retry_instruction(
        remaining_chars=HARD_CHARS,
        remaining_words=HARD_WORDS,
        quote_remaining=QUOTE_BUDGET_CHARS,
    )
    repair_prompt = build_synthesis_prompt(
        user_text=user_text,
        pack=pack,
        repair=repair,
        generation_budget_tokens=generation_budget_tokens,
    )
    try:
        repaired = await send_with_fallback(
            send,
            session_id,
            repair_prompt,
            agent=agent,
            primary_model=primary_model,
            fallback_model=fallback_model,
            sleep=sleep,
        )
    except OpenCodeSessionNotFoundError as exc:
        raise TurnFailed("session-not-found", "opencode session is gone") from exc
    repaired_diag = TurnDiagnostics(
        substantive=response.diagnostics.substantive,
        aspects=response.diagnostics.aspects,
        retrieval_rounds=response.diagnostics.retrieval_rounds,
        candidates=response.diagnostics.candidates,
        evidence_chunks=len(pack.units),
        evidence_tokens=pack.token_count,
        tool_call_count=response.diagnostics.tool_call_count + 1,
        coverage_gaps=response.diagnostics.coverage_gaps,
        regeneration_count=response.diagnostics.regeneration_count + MAX_COMPACT_REGENERATIONS,
        grounding_passed=None,
        served_model=repaired.served_model,
        fallback_used=repaired.fallback_used,
        error_category=repaired.error_category,
        retry_count=repaired.retry_count,
    )
    second = build_grounded_response(
        answer=repaired.text,
        pack=pack,
        diagnostics=repaired_diag,
        entails=entails,
    )
    unsupported = [unit for unit in second.units if unit.grounding_passed is False]
    if unsupported:
        raise TurnFailed(
            "grounding-failed",
            f"{len(unsupported)} regenerated unit(s) lack semantic support",
        )
    if envelope_passes(second.text):
        logger.info(
            "compact regeneration fit the envelope",
            extra={
                "graphemes": count_graphemes(second.text),
                "words": count_words(second.text),
            },
        )
        return second, 1
    compacted = compact_grounded_response(second)
    return compacted, 1


def _classify_send_error(exc: BaseException) -> str:
    text = str(exc).casefold()
    if re.search(r"(?<!\d)429(?!\d)", text) is not None or "too many requests" in text:
        return "provider-429"
    if isinstance(exc, (OpenCodeTransientError, OpenCodeTimeoutError)):
        return "provider-transient"
    if isinstance(exc, OpenCodeError):
        for marker in (
            "freeusagelimit",
            "free-usage-limit",
            "overloaded",
            "service unavailable",
            "bad gateway",
            "gateway timeout",
            "provider",
            "model-unavailable",
            "model unavailable",
        ):
            if marker in text:
                return "provider-unavailable"
        return "deterministic-error"
    return "deterministic-error"


async def send_with_fallback(
    send: Callable[..., Awaitable[str]],
    session_id: str,
    prompt: str,
    *,
    agent: str,
    primary_model: str,
    fallback_model: str,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> SynthesisResult:
    """Send one synthesis prompt with bounded retry and technical fallback.

    Provider 429/availability failures use bounded retry/backoff and the
    configured technical fallback model. Retrieval/grounding failures never
    reach this path; corpus/index state is never touched here, so a cache
    hit/miss or a 429 can never trigger corpus/index invalidation.
    Only the failure category and served model are recorded.
    """
    if not primary_model.strip() or not fallback_model.strip():
        raise TurnFailed("model-unpinned", "primary and fallback models must be pinned")
    if primary_model == fallback_model:
        raise TurnFailed("model-unpinned", "primary and fallback models must differ")
    sleeper = sleep or asyncio.sleep
    last_error: BaseException | None = None
    retry_count = 0
    for attempt in range(1, MAX_SEND_ATTEMPTS + 1):
        try:
            text = await send(session_id, prompt, agent=agent, model=primary_model)
            if not text.strip():
                raise TurnFailed("synthesis-empty", "synthesis returned an empty response")
            return SynthesisResult(
                text=text,
                served_model=primary_model,
                fallback_used=False,
                error_category="ok",
                retry_count=attempt - 1,
            )
        except TurnFailed:
            raise
        except (OpenCodeTransientError, OpenCodeTimeoutError) as exc:
            last_error = exc
            retry_count = attempt
            if attempt < MAX_SEND_ATTEMPTS:
                await sleeper(SEND_RETRY_DELAYS[min(attempt - 1, len(SEND_RETRY_DELAYS) - 1)])
                continue
        except OpenCodeSessionNotFoundError:
            raise
        except OpenCodeError as exc:
            category = _classify_send_error(exc)
            if category in ("provider-429", "provider-transient", "provider-unavailable"):
                last_error = OpenCodeTransientError(str(exc))
                retry_count = attempt
                if attempt < MAX_SEND_ATTEMPTS:
                    await sleeper(SEND_RETRY_DELAYS[min(attempt - 1, len(SEND_RETRY_DELAYS) - 1)])
                    continue
                break
            raise TurnFailed("synthesis-failed", category) from exc
    if last_error is not None:
        try:
            text = await send(session_id, prompt, agent=agent, model=fallback_model)
            if not text.strip():
                raise TurnFailed("synthesis-empty", "fallback synthesis was empty")
            return SynthesisResult(
                text=text,
                served_model=fallback_model,
                fallback_used=True,
                error_category="fallback-used",
                retry_count=retry_count,
            )
        except TurnFailed:
            raise
        except OpenCodeSessionNotFoundError:
            raise
        except OpenCodeError as exc:
            raise TurnFailed("synthesis-failed", _classify_send_error(exc)) from exc
    raise TurnFailed("synthesis-failed", "synthesis produced no result")


@dataclass
class TurnRunner:
    """Injectable turn pipeline (index + synthesis boundary)."""

    index: HybridIndex | None = None
    ru_corpus_version: str = ""
    agent: str = AGENT_NAME
    primary_model: str = ""
    fallback_model: str = ""
    entails: EntailmentFn | None = None
    generation_budget_tokens: int = 0

    def require_index(self) -> HybridIndex:
        """Return the opened RU index or fail closed on stale/missing corpus."""
        if self.index is None:
            raise TurnFailed("corpus-unavailable", "RU corpus/index is unavailable")
        live = str(self.index.metadata.get("ru_artifact_sha256", ""))
        if not live:
            raise TurnFailed("corpus-unavailable", "opened index has no RU version")
        pinned = self.ru_corpus_version.strip()
        # ``local`` is the default filesystem pointer (Settings.aa_corpus_version),
        # not a checksum: only a concrete non-local pin is compared for staleness.
        if pinned and pinned != "local" and pinned != live:
            raise TurnFailed("corpus-stale", "RU corpus version does not match the index")
        return self.index

    async def run_grounded_turn(
        self,
        text: str,
        *,
        session_id: str,
        send: Callable[..., Awaitable[str]],
        planner_fn: PlannerFn | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> GroundedResponse:
        """Execute phases 1-9 for one substantive turn (fails closed)."""
        if not text.strip():
            raise TurnFailed("empty-turn", "refusing an empty turn")
        index = self.require_index()
        ru_version = str(index.metadata.get("ru_artifact_sha256", ""))

        plan = await run_planner(text, planner_fn=planner_fn)
        if len(plan.aspects) > MAX_ASPECTS:
            raise TurnFailed("planner-invalid", "too many planned aspects")

        first_hits = search_first_round(index, plan)
        tool_calls = sum(1 for _ in first_hits)
        merged = deduplicate_cross_aspect(first_hits)
        coverage = check_coverage(plan, merged)

        retrieval_rounds = 1
        if not coverage.covered and MAX_RETRIEVAL_ROUNDS > 1:
            # One bounded second round with broadened additive queries.
            # The original wording stays; only same-language additive
            # vocabulary (stems, normalized form, broad program terms)
            # is added and never strengthens factual meaning.
            normalized_query = normalize_ru(plan.original_query)
            stemmed_query = " ".join(
                ru_stem(token) for token in ru_tokens(plan.original_query) if token
            )
            second_hits: dict[str, list[RetrievalHit]] = {}
            for aspect in plan.aspects:
                broadened = aspect_search_queries(aspect, original_query=plan.original_query)
                seen_queries = {item.casefold() for item in broadened}
                for extra in (
                    normalized_query,
                    stemmed_query,
                    *_BROAD_COVERAGE_QUERIES.get(aspect.aspect_id, ()),
                ):
                    cleaned = extra.strip()
                    if not cleaned or cleaned.casefold() in seen_queries:
                        continue
                    seen_queries.add(cleaned.casefold())
                    broadened.append(cleaned)
                try:
                    second_hits[aspect.aspect_id] = search_aspect(index, broadened)
                except (ValueError, OSError, RuntimeError) as exc:
                    raise TurnFailed("retrieval-failed", str(exc)) from exc
                tool_calls += 1
            combined: dict[str, list[RetrievalHit]] = {}
            for key in set(first_hits) | set(second_hits):
                combined[key] = [*first_hits.get(key, []), *second_hits.get(key, [])]
            merged = deduplicate_cross_aspect(combined)
            coverage = check_coverage(plan, merged)
            retrieval_rounds = 2

        distinct_sections = {hit.section for hit in merged}
        pack, read_calls = load_exact_evidence(
            index,
            merged,
            ru_corpus_version=ru_version,
            expand_multi_aspect=len(plan.aspects) > 1 or len(distinct_sections) < 2,
        )
        tool_calls += read_calls
        pack = fit_evidence_budget(pack, budget_tokens=RETRIEVED_PASSAGES_BUDGET_TOKENS)
        pack = EvidencePack(
            units=pack.units,
            token_count=pack.token_count,
            source_ids=pack.source_ids,
            locators=pack.locators,
            checksums=pack.checksums,
            ru_corpus_version=pack.ru_corpus_version,
            retrieval_rounds=retrieval_rounds,
            tool_call_count=tool_calls,
        )

        prompt = build_synthesis_prompt(user_text=text, pack=pack)
        try:
            synthesis = await send_with_fallback(
                send,
                session_id,
                prompt,
                agent=self.agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                sleep=sleep,
            )
        except OpenCodeSessionNotFoundError as exc:
            raise TurnFailed("session-not-found", "opencode session is gone") from exc

        base_diagnostics = TurnDiagnostics(
            substantive=True,
            aspects=len(plan.aspects),
            retrieval_rounds=retrieval_rounds,
            candidates=len(merged),
            evidence_chunks=len(pack.units),
            evidence_tokens=pack.token_count,
            tool_call_count=tool_calls,
            coverage_gaps=coverage.gaps,
            regeneration_count=0,
            grounding_passed=None,
            served_model=synthesis.served_model,
            fallback_used=synthesis.fallback_used,
            error_category=synthesis.error_category,
            retry_count=synthesis.retry_count,
        )
        try:
            first = build_grounded_response(
                answer=synthesis.text,
                pack=pack,
                diagnostics=base_diagnostics,
                entails=self.entails,
            )
        except TurnFailed:
            # Empty synthesis: nothing to regenerate from, fail closed.
            logger.warning(
                "grounded turn synthesis empty",
                extra={"aspects": len(plan.aspects), "evidence": len(pack.units)},
            )
            raise
        try:
            ensure_russian_only(synthesis.text)
            language_ok_first = True
        except TurnFailed:
            logger.warning(
                "grounded turn synthesis violated RU-only contract",
                extra={"aspects": len(plan.aspects), "evidence": len(pack.units)},
            )
            language_ok_first = False
        unsupported_first = [unit for unit in first.units if unit.grounding_passed is False]
        if language_ok_first and not unsupported_first:
            logger.info(
                "grounded turn completed",
                extra={
                    "aspects": len(plan.aspects),
                    "rounds": retrieval_rounds,
                    "candidates": len(merged),
                    "evidence": len(pack.units),
                    "units": len(first.units),
                    "regenerated": False,
                    "fallback": synthesis.fallback_used,
                },
            )
            enveloped, _extra = await enforce_grounded_envelope(
                first,
                user_text=text,
                pack=pack,
                session_id=session_id,
                send=send,
                agent=self.agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                entails=self.entails,
                sleep=sleep,
                generation_budget_tokens=self.generation_budget_tokens,
            )
            return enveloped
        # One bounded regeneration for unsupported units, else fail closed.
        logger.info(
            "grounded turn regenerating unsupported units",
            extra={"unsupported": len(unsupported_first), "evidence": len(pack.units)},
        )
        repair_note = (
            "Перепиши ответ ТОЛЬКО по-русски, опираясь только на отрывки. "
            f"Неподтверждённых мест: {len(unsupported_first)}. "
            "Каждое существенное утверждение — с ссылкой; "
            "прямые цитаты — дословно. Английский текст запрещён."
        )
        repair_prompt = build_synthesis_prompt(user_text=text, pack=pack, repair=repair_note)
        try:
            repaired = await send_with_fallback(
                send,
                session_id,
                repair_prompt,
                agent=self.agent,
                primary_model=self.primary_model,
                fallback_model=self.fallback_model,
                sleep=sleep,
            )
        except OpenCodeSessionNotFoundError as exc2:
            raise TurnFailed("session-not-found", "opencode session is gone") from exc2
        repaired_diag = TurnDiagnostics(
            substantive=True,
            aspects=len(plan.aspects),
            retrieval_rounds=retrieval_rounds,
            candidates=len(merged),
            evidence_chunks=len(pack.units),
            evidence_tokens=pack.token_count,
            tool_call_count=tool_calls + 1,
            coverage_gaps=coverage.gaps,
            regeneration_count=1,
            grounding_passed=None,
            served_model=repaired.served_model,
            fallback_used=repaired.fallback_used,
            error_category=repaired.error_category,
            retry_count=repaired.retry_count,
        )
        ensure_russian_only(repaired.text)
        response = build_grounded_response(
            answer=repaired.text,
            pack=pack,
            diagnostics=repaired_diag,
            entails=self.entails,
        )
        unsupported = [unit for unit in response.units if unit.grounding_passed is False]
        if unsupported:
            raise TurnFailed(
                "grounding-failed",
                f"{len(unsupported)} answer unit(s) lack semantic support",
            )
        ensure_russian_only(response.text)
        logger.info(
            "grounded turn completed after regeneration",
            extra={
                "aspects": len(plan.aspects),
                "rounds": retrieval_rounds,
                "candidates": len(merged),
                "evidence": len(pack.units),
                "units": len(response.units),
                "regenerated": True,
                "fallback": repaired.fallback_used,
            },
        )
        enveloped, _extra = await enforce_grounded_envelope(
            response,
            user_text=text,
            pack=pack,
            session_id=session_id,
            send=send,
            agent=self.agent,
            primary_model=self.primary_model,
            fallback_model=self.fallback_model,
            entails=self.entails,
            sleep=sleep,
            generation_budget_tokens=self.generation_budget_tokens,
        )
        return enveloped


def build_trivial_prompt(*, user_text: str) -> str:
    """Build the RU-only prompt for a non-substantive turn."""
    return (
        "Ответь ТОЛЬКО по-русски. Весь видимый ответ — на русском языке; "
        "английский текст запрещён.\n"
        f"{user_text}"
    )


async def run_trivial_turn(
    text: str,
    *,
    session_id: str,
    send: Callable[..., Awaitable[str]],
    agent: str = AGENT_NAME,
    primary_model: str,
    fallback_model: str,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> SynthesisResult:
    """Answer a non-substantive turn directly through the named agent.

    The prompt demands a Russian-only reply. Any synthesis violating
    the RU-only contract (missing Cyrillic, visible English/Latin, or
    an English fallback/error/provider fragment) fails the turn closed
    for a deterministic Russian fallback upstream.
    """
    if not text.strip():
        raise TurnFailed("empty-turn", "refusing an empty turn")
    try:
        result = await send_with_fallback(
            send,
            session_id,
            build_trivial_prompt(user_text=text),
            agent=agent,
            primary_model=primary_model,
            fallback_model=fallback_model,
            sleep=sleep,
        )
    except OpenCodeSessionNotFoundError as exc:
        raise TurnFailed("session-not-found", "opencode session is gone") from exc
    if not meets_russian_only(result.text):
        raise TurnFailed("language-violation", "trivial synthesis violated RU-only contract")
    if not envelope_passes(result.text):
        compacted = compact_text_to_envelope(result.text)
        if not meets_russian_only(compacted):
            raise TurnFailed("language-violation", "trivial synthesis violated RU-only contract")
        logger.info(
            "trivial turn compacted to envelope",
            extra={
                "graphemes": count_graphemes(compacted),
                "words": count_words(compacted),
            },
        )
        return SynthesisResult(
            text=compacted,
            served_model=result.served_model,
            fallback_used=result.fallback_used,
            error_category=result.error_category,
            retry_count=result.retry_count,
        )
    return result


__all__ = [
    "AGENT_NAME",
    "COVERAGE_SCHEMA_VERSION",
    "FAIL_CLOSED_REPLY",
    "MAX_EVIDENCE_CHUNKS",
    "MAX_PLANNER_ATTEMPTS",
    "MAX_REGENERATIONS",
    "MAX_RETRIEVAL_ROUNDS",
    "MAX_SEND_ATTEMPTS",
    "RUNTIME_VERSION",
    "SUPPORT_SCHEMA_VERSION",
    "AnswerUnit",
    "CoverageResult",
    "EvidencePack",
    "GroundedResponse",
    "ResponseUnit",
    "SynthesisResult",
    "TurnDiagnostics",
    "TurnFailed",
    "TurnRunner",
    "build_grounded_response",
    "build_local_plan_payload",
    "build_synthesis_prompt",
    "build_trivial_prompt",
    "check_coverage",
    "compact_grounded_response",
    "contains_english_fallback",
    "deduplicate_cross_aspect",
    "enforce_grounded_envelope",
    "ensure_russian_only",
    "fit_evidence_budget",
    "is_substantive",
    "judge_unit",
    "load_exact_evidence",
    "meets_russian_only",
    "response_envelope_ok",
    "response_quote_chars",
    "run_planner",
    "run_trivial_turn",
    "search_first_round",
    "send_with_fallback",
    "split_answer_units",
    "validate_coverage_payload",
    "validate_support_payload",
]
