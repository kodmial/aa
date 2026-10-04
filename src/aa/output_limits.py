"""Deterministic Telegram output envelope (issue #83).

Authoritative user-facing product limits for every Telegram text reply:

- ordinary substantive target: ``<= 500`` characters / ``<= 80`` words;
- hard maximum: ``<= 900`` grapheme clusters / ``<= 130`` words;
- verbatim quoted corpus text aggregate: ``<= 300`` characters;
- never reproduce a whole chapter/section or auto-split overflow.

Length here means user-visible Unicode text. Grapheme counting is
combining-aware (base character plus combining marks, variation
selectors, and ZWJ sequences count as one cluster); ``len()`` on
codepoints is a conservative ceiling of that count, so enforcing both
keeps the check fail-closed.

Privacy contract: this module never logs response or corpus text, only
lengths and categories.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

logger = logging.getLogger("aa.output_limits")

TARGET_CHARS = 500
TARGET_WORDS = 80
HARD_CHARS = 900
HARD_WORDS = 130
ACK_CHARS = 300
QUOTE_BUDGET_CHARS = 300
MAX_COMPACT_REGENERATIONS = 1

DEFAULT_GENERATION_BUDGET_TOKENS = 256
"""Conservative bounded output-token budget (efficiency guard only).

Token counts are not authoritative: character/token ratios vary by
language and model. The deterministic character validator below is the
authoritative enforcement boundary.
"""

COMPACT_RETRY_INSTRUCTION = (
    "Rewrite the previous draft concisely: at most 2-4 short sentences, "
    "at most 500 characters and 80 words total, keeping only the "
    "highest-value point and at most one short exact quote "
    "(300 characters total at most). Do not split across messages."
)

BULK_EXPORT_SUMMARY_RU = (
    "Не могу выводить главы целиком: отвечаю коротко по существу. "
    "Напишите, какая мысль сейчас важна, и разберём её по книге. "
    "Какой эпизод обсудить?"
)

BULK_EXPORT_SUMMARY_EN = (
    "I cannot print whole chapters here; I keep answers short. "
    "Tell me which point matters right now and we will look at it. "
    "Which episode should we discuss?"
)

OVERLONG_FALLBACK_TEXT = (
    "Ответ вышел слишком длинным, вот короткая версия. "
    "Задайте более узкий вопрос, и продолжим. / "
    "The answer was too long, here is the short version. "
    "Ask a narrower follow-up to continue."
)


def grapheme_len(text: str) -> int:
    """Count Unicode grapheme clusters (combining-aware approximation).

    A codepoint continues the previous cluster when it is a combining
    mark, a variation selector (U+FE00-U+FE0F), a zero-width joiner, or
    a modifier (emoji modifiers, regional indicators are counted
    pairwise below). Everything else starts a new cluster.
    """
    count = 0
    pending_regional = False
    for char in text:
        code = ord(char)
        if char == "\u200d":
            # Zero-width joiner glues emoji sequences: no new cluster.
            continue
        if unicodedata.combining(char) != 0:
            continue
        if 0xFE00 <= code <= 0xFE0F:
            continue
        if 0x1F3FB <= code <= 0x1F3FF:
            # Emoji skin-tone modifiers extend the previous cluster.
            continue
        if 0x1F1E6 <= code <= 0x1F1FF:
            # Regional indicators pair into one flag cluster.
            if pending_regional:
                pending_regional = False
                continue
            pending_regional = True
            count += 1
            continue
        pending_regional = False
        count += 1
    return count


def word_count(text: str) -> int:
    """Count whitespace-separated words in user-visible text."""
    return len(text.split())


def within_hard_cap(text: str) -> bool:
    """Whether ``text`` fits the hard Telegram envelope."""
    return grapheme_len(text) <= HARD_CHARS and word_count(text) <= HARD_WORDS


_QUOTED_PAIR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"«(.+?)»", re.DOTALL),
    re.compile(r"“(.+?)”", re.DOTALL),
    re.compile(r"„(.+?)“", re.DOTALL),
    re.compile(r'"(.+?)"', re.DOTALL),
)

_FENCED_CODE_PATTERN = re.compile(r"```.*?```", re.DOTALL)
_BLOCKQUOTE_LINE = re.compile(r"(?m)^\s*>\s?(.*)$")
_ENTITY_PATTERN = re.compile(r"&(?:[A-Za-z]+|#[0-9]+|#x[0-9A-Fa-f]+);")
_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+")
_MARKDOWN_TOKEN_PATTERN = re.compile(r"(\*\*|__|`|\[|\]\(|\)|\|)")


def quoted_spans(text: str) -> list[str]:
    """Return inner texts treated as source-exact quotation candidates."""
    spans: list[str] = []
    for pattern in _QUOTED_PAIR_PATTERNS:
        for match in pattern.finditer(text):
            inner = match.group(1)
            if inner.strip():
                spans.append(inner)
    for match in _BLOCKQUOTE_LINE.finditer(text):
        inner = match.group(1)
        if inner.strip():
            spans.append(inner)
    for match in _FENCED_CODE_PATTERN.finditer(text):
        inner = match.group(0)[3:-3].strip()
        if inner:
            spans.append(inner)
    return spans


def aggregate_quoted_chars(text: str) -> int:
    """Aggregate verbatim-quote candidate characters in ``text``."""
    return sum(len(span) for span in quoted_spans(text))


def within_quote_budget(text: str) -> bool:
    """Whether aggregate quoted text fits the 300-character budget."""
    return aggregate_quoted_chars(text) <= QUOTE_BUDGET_CHARS


_BULK_EXPORT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"выве[дс]и\b.{0,60}?\bглав\w*", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bглав\w*.{0,40}?\bцеликом\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bвтор\w+\s+глав\w*", re.IGNORECASE),
    re.compile(r"\bвс[юе]\s+глав\w*", re.IGNORECASE),
    re.compile(r"\bцел\w+\s+глав\w*", re.IGNORECASE),
    re.compile(r"напечатай\b.{0,80}?\b(глав\w*|раздел\w*|текст\w*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"покажи\b.{0,80}?\b(вс[юе]|целиком|полностью)\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"пришли\b.{0,80}?\b(глав\w*|текст\w*|книг\w*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"продолж\w*.{0,40}?(глав\w*|печата\w*|выводи\w*)", re.IGNORECASE | re.DOTALL),
    re.compile(r"давай\s+дальше\b.{0,40}?глав\w*", re.IGNORECASE | re.DOTALL),
    re.compile(r"следующ\w*\s+(часть|кусок|продолжение)\b", re.IGNORECASE),
    re.compile(r"сними\b.{0,40}?огранич", re.IGNORECASE | re.DOTALL),
    re.compile(r"игнорируй\b.{0,40}?огранич", re.IGNORECASE | re.DOTALL),
    re.compile(r"убери\b.{0,40}?лимит", re.IGNORECASE | re.DOTALL),
    re.compile(r"без\s+ограничений\b", re.IGNORECASE),
    re.compile(
        r"print\b.{0,80}?\b(whole|entire|full|complete)\b.{0,40}?\b(chapter|section|passage)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"\b(whole|entire|full|complete)\b.{0,40}?\b(chapter|section)\b.{0,40}?\b(dump|text|verbatim)\b",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(r"continue\b.{0,40}?(printing|chapter|passage)", re.IGNORECASE | re.DOTALL),
    re.compile(r"give\b.{0,40}?next\s+part\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"ignore\b.{0,40}?\blimits?\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"print\b.{0,40}?\d{4,}\s*characters?", re.IGNORECASE | re.DOTALL),
)

_CONTINUATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*продолж\w*\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*дальше\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*давай\s+дальше\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*следующ\w*\s+(часть|кусок)\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*ещ[её]\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*continue\s*[.!?…]*\s*$", re.IGNORECASE),
    re.compile(r"^\s*(give\s+me\s+)?the?\s*next\s+part\s*[.!?…]*\s*$", re.IGNORECASE),
)


def is_bulk_export_request(text: str) -> bool:
    """Whether ``text`` primarily asks for bulk corpus reproduction."""
    normalized = text.strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) is not None for pattern in _BULK_EXPORT_PATTERNS)


def is_continuation_request(text: str) -> bool:
    """Whether ``text`` asks to continue paging through source text."""
    normalized = text.strip()
    if not normalized:
        return False
    if any(pattern.search(normalized) is not None for pattern in _BULK_EXPORT_PATTERNS):
        return True
    return any(pattern.search(normalized) is not None for pattern in _CONTINUATION_PATTERNS)


def detect_language(text: str) -> str:
    """Detect ``ru`` vs ``en`` with a Cyrillic-presence heuristic."""
    for char in text:
        if "\u0400" <= char <= "\u04ff":
            return "ru"
    return "en"


def bulk_export_summary_reply(language: str = "ru") -> str:
    """Bounded summary-mode reply for chapter-dump requests (no dump)."""
    if language == "ru":
        return BULK_EXPORT_SUMMARY_RU
    return BULK_EXPORT_SUMMARY_EN


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+|\n+")


def _grapheme_boundary_indices(text: str) -> list[int]:
    """Character offsets that are safe grapheme boundaries."""
    boundaries: list[int] = [0]
    pending_regional = False
    for index, char in enumerate(text):
        code = ord(char)
        starts_new = True
        if char == "\u200d":
            starts_new = False
        elif unicodedata.combining(char) != 0:
            starts_new = False
        elif 0xFE00 <= code <= 0xFE0F:
            starts_new = False
        elif 0x1F3FB <= code <= 0x1F3FF:
            starts_new = False
        elif 0x1F1E6 <= code <= 0x1F1FF:
            if pending_regional:
                starts_new = False
            else:
                starts_new = True
        if starts_new:
            pending_regional = 0x1F1E6 <= code <= 0x1F1FF and not pending_regional
            boundaries.append(index)
        else:
            pending_regional = False
    boundaries.append(len(text))
    return sorted(set(boundaries))


def _snap_to_grapheme(text: str, limit: int) -> int:
    """Largest safe grapheme-boundary offset ``<= limit``."""
    if limit >= len(text):
        return len(text)
    boundaries = _grapheme_boundary_indices(text)
    best = 0
    for boundary in boundaries:
        if boundary <= limit:
            best = boundary
        else:
            break
    return best


def _cut_inside_span(cut: int, spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    for start, end in spans:
        if start < cut < end:
            return (start, end)
    return None


def _unsafe_spans(text: str) -> list[tuple[int, int]]:
    """Spans that must never be cut inside (URLs, entities, quotes)."""
    spans: list[tuple[int, int]] = []
    for match in _URL_PATTERN.finditer(text):
        spans.append((match.start(), match.end()))
    for match in _ENTITY_PATTERN.finditer(text):
        spans.append((match.start(), match.end()))
    for pattern in _QUOTED_PAIR_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end()))
    for match in _FENCED_CODE_PATTERN.finditer(text):
        spans.append((match.start(), match.end()))
    return spans


def safe_cut(text: str, limit: int) -> str:
    """Cut ``text`` to ``limit`` chars without breaking safe constructs."""
    if len(text) <= limit:
        return text
    cut = _snap_to_grapheme(text, limit)
    spans = _unsafe_spans(text)
    hit = _cut_inside_span(cut, spans)
    if hit is not None:
        # Back off to the start of the unsafe span instead of cutting it.
        cut = _snap_to_grapheme(text[: hit[0]], hit[0])
    # Prefer a word boundary so no word is split.
    if cut < len(text) and cut > 0 and not text[cut].isspace() and not text[cut - 1].isspace():
        space = text.rfind(" ", 0, cut)
        if space > 0:
            cut = _snap_to_grapheme(text, space)
    candidate = text[:cut].rstrip()
    # Never leave unclosed markdown/code/entity constructs behind.
    if candidate.count("```") % 2 == 1:
        fence = candidate.rfind("```")
        candidate = candidate[:fence].rstrip()
    if candidate.count("**") % 2 == 1:
        marker = candidate.rfind("**")
        candidate = candidate[:marker].rstrip()
    if candidate.count("__") % 2 == 1:
        marker = candidate.rfind("__")
        candidate = candidate[:marker].rstrip()
    if candidate.count("`") % 2 == 1:
        marker = candidate.rfind("`")
        candidate = candidate[:marker].rstrip()
    dangling_entity = re.search(r"&[A-Za-z0-9#]*$", candidate)
    if dangling_entity is not None:
        candidate = candidate[: dangling_entity.start()].rstrip()
    dangling_bracket = candidate.rfind("[")
    dangling_paren = candidate.rfind("](")
    if dangling_bracket > candidate.rfind("]"):
        candidate = candidate[:dangling_bracket].rstrip()
    elif dangling_paren != -1 and candidate.find(")", dangling_paren) == -1:
        candidate = candidate[:dangling_paren].rstrip()
    return candidate


def compact_to_hard_cap(text: str) -> str:
    """Deterministically keep complete leading units fitting the hard cap.

    Units are sentences/paragraphs; lower-priority trailing units are
    dropped before any unsafe string truncation. A quotation, URL,
    Markdown/entity construct, or combining sequence is never cut
    inside. The result fits ``<= 900`` graphemes, ``<= 130`` words, and
    the ``<= 300``-character aggregate quote budget when possible.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if within_hard_cap(stripped) and within_quote_budget(stripped):
        return stripped
    units = [unit for unit in _SENTENCE_SPLIT.split(stripped) if unit.strip()]
    kept: list[str] = []
    for unit in units:
        candidate = " ".join([*kept, unit.strip()])
        if within_hard_cap(candidate) and within_quote_budget(candidate):
            kept.append(unit.strip())
        else:
            break
    if kept:
        return " ".join(kept)
    # The first unit alone overflows: fall back to a safe word-truncate
    # of that unit so output is never empty for non-empty input.
    first = units[0].strip() if units else stripped
    words = first.split()
    while len(words) > HARD_WORDS:
        words = words[:HARD_WORDS]
        candidate_text = " ".join(words)
        if within_hard_cap(candidate_text) and within_quote_budget(candidate_text):
            return candidate_text
    candidate_text = " ".join(words)
    if within_hard_cap(candidate_text) and within_quote_budget(candidate_text):
        return candidate_text
    truncated = safe_cut(candidate_text, HARD_CHARS)
    truncated_words = truncated.split()
    while len(truncated_words) > HARD_WORDS:
        truncated_words = truncated_words[:-1]
    truncated = " ".join(truncated_words)
    if within_quote_budget(truncated):
        return truncated
    # Quote budget still exceeded inside one unit: drop trailing quoted
    # spans entirely rather than cutting inside a quotation.
    spans = quoted_spans(truncated)
    for span in reversed(spans):
        truncated = truncated.replace(span, "", 1).strip()
        truncated = re.sub(r"\s{2,}", " ", truncated)
        if within_quote_budget(truncated) and truncated:
            break
    if truncated and within_hard_cap(truncated) and within_quote_budget(truncated):
        return truncated
    return safe_cut(truncated or first, min(HARD_CHARS, len(truncated or first)))


def resolve_generation_budget(configured_max_output_tokens: int) -> int:
    """Resolve the bounded output-token efficiency guard.

    A positive configured ``OPENCODE_MAX_OUTPUT_TOKENS`` value wins;
    otherwise the conservative default applies. Negative values are a
    configuration error. The result is never treated as the product
    contract: the deterministic character validator is authoritative.
    """
    if configured_max_output_tokens < 0:
        raise ValueError("OPENCODE_MAX_OUTPUT_TOKENS must be >= 0")
    if configured_max_output_tokens > 0:
        return configured_max_output_tokens
    return DEFAULT_GENERATION_BUDGET_TOKENS


@dataclass(frozen=True)
class LimitAssessment:
    """Deterministic verdict for one user-facing candidate reply."""

    graphemes: int
    words: int
    quoted_chars: int
    within_hard_cap: bool
    within_quote_budget: bool
    bulk_export_request: bool = False

    @property
    def acceptable(self) -> bool:
        """Whether the candidate may be delivered as-is."""
        return self.within_hard_cap and self.within_quote_budget


def assess(text: str) -> LimitAssessment:
    """Assess ``text`` against the hard envelope (lengths only logged)."""
    graphemes = grapheme_len(text)
    words = word_count(text)
    quoted = aggregate_quoted_chars(text)
    assessment = LimitAssessment(
        graphemes=graphemes,
        words=words,
        quoted_chars=quoted,
        within_hard_cap=graphemes <= HARD_CHARS and words <= HARD_WORDS,
        within_quote_budget=quoted <= QUOTE_BUDGET_CHARS,
    )
    logger.info(
        "output limit assessment",
        extra={
            "graphemes": graphemes,
            "words": words,
            "quoted_chars": quoted,
            "acceptable": assessment.acceptable,
        },
    )
    return assessment


@dataclass(frozen=True)
class EnforcementOutcome:
    """Result of deterministic output-limit enforcement for one turn."""

    text: str
    regenerations: int
    compacted: bool


def enforce_sync(
    first_answer: str,
    regenerate: Callable[[str], str] | None = None,
) -> EnforcementOutcome:
    """Enforce the envelope with at most one compact regeneration.

    ``regenerate`` receives the explicit remaining-size budget
    instruction and returns the second candidate. When no regeneration
    is available or the second answer still overflows, output is
    deterministically compacted to complete leading units. Never
    splits overflow into additional messages and never logs text.
    """
    first = first_answer.strip()
    if assess(first).acceptable:
        return EnforcementOutcome(text=first, regenerations=0, compacted=False)
    logger.info(
        "output limit regeneration requested",
        extra={
            "graphemes": grapheme_len(first),
            "words": word_count(first),
            "quoted_chars": aggregate_quoted_chars(first),
        },
    )
    if regenerate is not None:
        try:
            second = regenerate(COMPACT_RETRY_INSTRUCTION).strip()
        except Exception:
            logger.warning("output limit regeneration failed")
            second = ""
        if second:
            if assess(second).acceptable:
                return EnforcementOutcome(text=second, regenerations=1, compacted=False)
            compacted = compact_to_hard_cap(second)
            logger.info(
                "output limit deterministic compaction",
                extra={
                    "graphemes": grapheme_len(compacted),
                    "words": word_count(compacted),
                    "quoted_chars": aggregate_quoted_chars(compacted),
                },
            )
            return EnforcementOutcome(text=compacted, regenerations=1, compacted=True)
    compacted = compact_to_hard_cap(first)
    logger.info(
        "output limit deterministic compaction",
        extra={
            "graphemes": grapheme_len(compacted),
            "words": word_count(compacted),
            "quoted_chars": aggregate_quoted_chars(compacted),
        },
    )
    return EnforcementOutcome(text=compacted, regenerations=0, compacted=True)


async def enforce_async(
    first_answer: str,
    regenerate: Callable[[str], Awaitable[str]] | None = None,
) -> EnforcementOutcome:
    """Async variant of :func:`enforce_sync` for orchestrator wiring."""
    first = first_answer.strip()
    if assess(first).acceptable:
        return EnforcementOutcome(text=first, regenerations=0, compacted=False)
    logger.info(
        "output limit regeneration requested",
        extra={
            "graphemes": grapheme_len(first),
            "words": word_count(first),
            "quoted_chars": aggregate_quoted_chars(first),
        },
    )
    if regenerate is not None:
        try:
            second = (await regenerate(COMPACT_RETRY_INSTRUCTION)).strip()
        except Exception:
            logger.warning("output limit regeneration failed")
            second = ""
        if second:
            if assess(second).acceptable:
                return EnforcementOutcome(text=second, regenerations=1, compacted=False)
            compacted = compact_to_hard_cap(second)
            logger.info(
                "output limit deterministic compaction",
                extra={
                    "graphemes": grapheme_len(compacted),
                    "words": word_count(compacted),
                    "quoted_chars": aggregate_quoted_chars(compacted),
                },
            )
            return EnforcementOutcome(text=compacted, regenerations=1, compacted=True)
    compacted = compact_to_hard_cap(first)
    logger.info(
        "output limit deterministic compaction",
        extra={
            "graphemes": grapheme_len(compacted),
            "words": word_count(compacted),
            "quoted_chars": aggregate_quoted_chars(compacted),
        },
    )
    return EnforcementOutcome(text=compacted, regenerations=0, compacted=True)


__all__ = [
    "ACK_CHARS",
    "BULK_EXPORT_SUMMARY_EN",
    "BULK_EXPORT_SUMMARY_RU",
    "COMPACT_RETRY_INSTRUCTION",
    "DEFAULT_GENERATION_BUDGET_TOKENS",
    "HARD_CHARS",
    "HARD_WORDS",
    "MAX_COMPACT_REGENERATIONS",
    "OVERLONG_FALLBACK_TEXT",
    "QUOTE_BUDGET_CHARS",
    "TARGET_CHARS",
    "TARGET_WORDS",
    "EnforcementOutcome",
    "LimitAssessment",
    "aggregate_quoted_chars",
    "assess",
    "bulk_export_summary_reply",
    "compact_to_hard_cap",
    "detect_language",
    "enforce_async",
    "enforce_sync",
    "grapheme_len",
    "is_bulk_export_request",
    "is_continuation_request",
    "quoted_spans",
    "resolve_generation_budget",
    "safe_cut",
    "within_hard_cap",
    "within_quote_budget",
    "word_count",
]
