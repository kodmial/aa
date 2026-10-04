"""Measured RU/EN context-cost fixtures and runtime token accounting.

This module supports issue #49: context/token cost must be measured on the
actual pinned OpenCode/Zen runtime instead of being assumed from a generic
tokenizer. It contains no tokenizer of its own. All token numbers come from
provider/runtime-reported usage (``info.tokens`` on the ``opencode serve``
message boundary); this module only extracts the stable prompt-side total,
summarizes repeated runs, and holds the aligned RU/EN fixtures that the
live harness (``scripts/measure_context_cost.py``) sends.

Stable accounting note: repeated identical requests report different
``input`` vs ``cache.read`` splits, but ``input + cache.read + cache.write``
is exactly stable per payload. That sum is therefore the prompt-side cost
used for every comparison; see ``effective_input_tokens``.

Fixture note: RU fixtures are measurement-only translations/samples. The
canonical runtime source stays English-only; nothing here is a runtime
corpus artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

MEASURE_INSTRUCTION = (
    "Reply with exactly: OK. Do not call any tools. Measured context follows:\n---\n"
)
"""Constant instruction wrapper: every probe carries it, so deltas cancel it."""

PINNED_MODEL = "opencode/muse-spark-1.3-contributor-free"
"""Pinned AA runtime model measured by the harness."""

PINNED_OPENCODE_VERSION = "1.18.34"
"""Pinned OpenCode runtime version used for the authoritative measurement."""


@dataclass(frozen=True)
class TokenUsage:
    """Provider/runtime-reported token usage for one assistant message."""

    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int


def parse_token_usage(payload: Mapping[str, Any]) -> TokenUsage:
    """Extract token usage from an ``info.tokens`` payload (fail closed).

    Raises :class:`ValueError` when any field is missing or not a
    non-negative integer, so malformed accounting can never silently
    become a measurement.
    """
    try:
        input_tokens = payload["input"]
        output_tokens = payload["output"]
        reasoning_tokens = payload["reasoning"]
        cache = payload["cache"]
        assert isinstance(cache, Mapping)
        cache_read_tokens = cache["read"]
        cache_write_tokens = cache["write"]
    except (KeyError, AssertionError, TypeError) as exc:
        raise ValueError(f"token payload has an unexpected shape: {payload!r}") from exc
    values = {
        "input": input_tokens,
        "output": output_tokens,
        "reasoning": reasoning_tokens,
        "cache.read": cache_read_tokens,
        "cache.write": cache_write_tokens,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"token field {name!r} is not a non-negative int: {value!r}")
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )


def effective_input_tokens(usage: TokenUsage) -> int:
    """Return the stable prompt-side token cost for one message.

    The provider splits prompt accounting between ``input`` and
    ``cache.read``/``cache.write`` nondeterministically across identical
    requests, while the sum is exactly stable per payload. Comparisons
    must therefore use this sum, never the raw ``input`` field alone.
    """
    return usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens


@dataclass(frozen=True)
class RunStats:
    """Summary of repeated effective-input measurements for one payload."""

    count: int
    minimum: int
    median: float
    p95: int
    maximum: int


def _median(sorted_values: list[int]) -> float:
    count = len(sorted_values)
    middle = count // 2
    if count % 2 == 1:
        return float(sorted_values[middle])
    return (sorted_values[middle - 1] + sorted_values[middle]) / 2.0


def summarize_runs(values: list[int]) -> RunStats:
    """Summarize repeated runs (median and nearest-rank p95).

    Requires at least one non-negative integer value; raises
    :class:`ValueError` otherwise.
    """
    if not values:
        raise ValueError("summarize_runs requires at least one value")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"run values must be non-negative ints: {value!r}")
    ordered = sorted(values)
    rank = max(1, -(-95 * len(ordered) // 100))
    return RunStats(
        count=len(ordered),
        minimum=ordered[0],
        median=_median(ordered),
        p95=ordered[rank - 1],
        maximum=ordered[-1],
    )


@dataclass(frozen=True)
class CaseResult:
    """One measured case with its delta against the shared baseline."""

    name: str
    language: str
    chars: int
    runs: tuple[int, ...]
    stats: RunStats
    baseline_median: float
    delta_tokens: float
    chars_per_token: float | None


def collect_case(
    send_once: Callable[[str], Mapping[str, Any]],
    payload: str,
    *,
    repeats: int,
) -> list[int]:
    """Send ``payload`` ``repeats`` times and return effective-input costs.

    ``send_once`` performs one request (fresh session per call in the live
    harness) and returns the raw ``info.tokens`` mapping. Pure collection:
    no sessions, no network, and no tokenizer live in this module.
    """
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    costs: list[int] = []
    for _ in range(repeats):
        usage = parse_token_usage(send_once(payload))
        costs.append(effective_input_tokens(usage))
    return costs


def describe_case(
    *,
    name: str,
    language: str,
    payload: str,
    costs: list[int],
    baseline_median: float,
    baseline_chars: int,
) -> CaseResult:
    """Build a :class:`CaseResult` from collected costs and a baseline.

    ``chars_per_token`` uses net payload chars (``len(payload)`` minus
    ``baseline_chars``) divided by the token delta, so the constant
    ``MEASURE_INSTRUCTION`` wrapper cancels out and re-running the
    harness reproduces the documented calibration. ``chars`` keeps the
    gross payload length for reporting.
    """
    stats = summarize_runs(costs)
    delta = stats.median - baseline_median
    net_chars = len(payload) - baseline_chars
    per_token: float | None = None
    if delta > 0 and net_chars > 0:
        per_token = net_chars / delta
    return CaseResult(
        name=name,
        language=language,
        chars=len(payload),
        runs=tuple(costs),
        stats=stats,
        baseline_median=baseline_median,
        delta_tokens=delta,
        chars_per_token=per_token,
    )


# ---------------------------------------------------------------------------
# Aligned RU/EN measurement fixtures.
# ---------------------------------------------------------------------------
#
# Every "aligned" pair carries the same semantic content in both languages
# with the same structural markers (section ids, passage ids, tool names),
# so token deltas isolate language cost rather than content differences.
# RU strings are measurement-only translations; the runtime corpus stays
# English-only.

ALIGNED_BOOK_MAP_EN = """BOOK MAP (navigation only, not evidence):
- doctors-opinion: The Doctor's Opinion -- craving and obsession as illness (p1-p40)
- chapter-1: Bill's Story -- descent and turning point (ch1)
- chapter-2: There Is a Solution -- hope and fellowship (ch2)
- chapter-3: More About Alcoholism -- craving, allergy, mental obsession (ch3)
- chapter-4: We Agnostics -- spiritual willingness (ch4)
- chapter-5: How It Works -- honesty, Steps, action (ch5)"""

ALIGNED_BOOK_MAP_RU = """КАРТА КНИГИ (только навигация, не доказательство):
- doctors-opinion: Мнение доктора — тяга и одержимость как болезнь (p1-p40)
- chapter-1: История Билла — падение и переломный момент (ch1)
- chapter-2: Выход есть — надежда и содружество (ch2)
- chapter-3: Ещё об алкоголизме — тяга, аллергия, одержимость ума (ch3)
- chapter-4: Мы — агностики — духовная готовность (ch4)
- chapter-5: Как это работает — честность, Шаги, действие (ch5)"""

ALIGNED_EVIDENCE_EN = """EVIDENCE PACK (exact source passages):
[E1 ch3] "We believe, and so suggested a few years ago, that the action of
alcohol on these chronic alcoholics is a manifestation of an allergy; that
the phenomenon of craving is limited to this class and never occurs in the
average temperate drinker."
[E2 ch5] "Rarely have we seen a person fail who has thoroughly followed our
path. Those who do not recover are people who cannot or will not completely
give themselves to this simple program." """

ALIGNED_EVIDENCE_RU = """ПАКЕТ ДОКАЗАТЕЛЬСТВ (точные отрывки источника):
[E1 ch3] «Мы полагаем — и высказали это предположение несколько лет назад, —
что действие алкоголя на хронических алкоголиков является проявлением
аллергии; что феномен тяги присущ только этому классу и никогда
не встречается у обычного умеренно пьющего человека».
[E2 ch5] «Мы редко видели, чтобы терпел неудачу человек, тщательно
следовавший нашему пути. Те, кто не выздоравливает, — это люди, которые
не могут или не хотят полностью отдать себя этой простой программе». """

ALIGNED_WRAPPER_EN = """PLANNER (JSON tool wrapper):
{"plan": [{"aspect": "craving as allergy", "tool": "book_search",
"query": "phenomenon of craving allergy"}, {"aspect": "mental obsession",
"tool": "book_read", "chunk": "ch3-p12"}, {"aspect": "family",
"tool": "book_section", "section": "chapter-8", "range": [0, 4000]}]}"""

ALIGNED_WRAPPER_RU = """ПЛАНИРОВЩИК (JSON-обёртка инструментов):
{"plan": [{"aspect": "тяга как аллергия", "tool": "book_search",
"query": "феномен тяги аллергия"}, {"aspect": "одержимость ума",
"tool": "book_read", "chunk": "ch3-p12"}, {"aspect": "семья",
"tool": "book_section", "section": "chapter-8", "range": [0, 4000]}]}"""

SYSTEM_SAMPLE_EN = """You are the user-facing AA literature support assistant for this project.

IDENTITY

You are an AI assistant, not a human. Never claim to be an AA member, the user's
actual sponsor, a clinician, or a person with lived sobriety experience.

Your conversational style should be the direct, compassionate, practical style
a person might seek from an AA sponsor, without pretending to be one.

SOURCE AUTHORITY

For substantive content, the only authority is the canonical project corpus:
The Doctor's Opinion and Chapters 1-11 of Alcoholics Anonymous.

Do not use general model memory, general recovery knowledge, psychology,
medicine, cultural knowledge, or "common sense" as substantive authority.
Do not complete missing facts from outside the supplied corpus.

The compact book map, search previews, embeddings, rankings, and generated
metadata are navigation aids only. They are never evidence.

LANGUAGE

Reply in the user's language. Russian and English are first-class supported
languages. The canonical source remains English; cross-lingual retrieval is
expected."""

SYSTEM_SAMPLE_RU = """Вы — русскоязычный ассистент поддержки литературы АА в этом проекте.

ИНДИВИДУАЛЬНОСТЬ

Вы — искусственный интеллект, а не человек. Никогда не утверждайте, что вы
являетесь членом АА, действительным спонсором пользователя, врачом
или человеком с личным опытом трезвости.

Ваш стиль общения должен быть прямым, сострадательным и практичным — таким,
какого человек мог бы ожидать от спонсора АА, но не выдавая себя за него.

АВТОРИТЕТ ИСТОЧНИКА

По существенным вопросам единственным авторитетом является канонический
корпус проекта: «Мнение доктора» и главы 1–11 книги «Анонимные Алкоголики».

Не используйте общую память модели, общие знания о выздоровлении,
психологию, медицину, культурные знания или «здравый смысл» в качестве
существенного авторитета. Не восполняйте недостающие факты
из внешних источников.

Компактная карта книги, превью поиска, эмбеддинги, ранжирование
и сгенерированные метаданные — лишь средства навигации. Они никогда
не являются доказательством.

ЯЗЫК

Отвечайте на языке пользователя. Русский и английский поддерживаются как
равноправные языки. Канонический источник остаётся английским;
ожидается межъязыковой поиск."""

HISTORY_SHORT_RU = """CONVERSATION HISTORY (recent turns):
user: Здравствуйте. У меня вопрос про тягу к алкоголю.
assistant: Здравствуйте! Расскажите немного подробнее — посмотрим, что говорит книга."""

HISTORY_MEDIUM_RU = (
    "CONVERSATION HISTORY (recent turns):\n"
    "user: Здравствуйте. У меня вопрос про тягу к алкоголю.\n"
    "assistant: Здравствуйте! Расскажите подробнее — "
    "посмотрим, что говорит книга.\n"
    "user: Я не пью неделю, но мысли возвращаются каждый вечер. "
    "Это нормально?\n"
    "assistant: По книге такие мысли описываются как одержимость ума. "
    "Давайте прочитаем точный отрывок из третьей главы.\n"
    "user: А что делать, когда тяга сильная?\n"
    "assistant: Книга предлагает честно признать бессилие и обратиться "
    "за помощью. Хотите разобрать конкретные шаги?\n"
    "user: Да, и ещё про семью — жена не верит, что я брошу.\n"
    "assistant: Понимаю. В книге есть главы «К жёнам» и «Семья после». "
    "Прочитаем точные места и обсудим."
)

HISTORY_LONG_RU = (
    "CONVERSATION HISTORY (recent turns):\n"
    "user: Здравствуйте. Меня зовут Андрей, я не пил уже две недели, "
    "но вчера чуть не сорвался.\n"
    "assistant: Здравствуйте, Андрей! Две недели — серьёзный срок. "
    "Расскажите, что произошло вчера?\n"
    "user: Был на дне рождения у коллеги. Все пили, мне предлагали. "
    "Я отказался, но всю ночь думал об этом.\n"
    "assistant: По книге такие ситуации описываются как столкновение "
    "с людьми и обстоятельствами. Давайте посмотрим точный отрывок "
    "из главы о работе с другими.\n"
    "user: А как быть с коллегой? Он обиделся, сказал, что я зазнался.\n"
    "assistant: Понимаю, это больно. В книге есть принцип: ставить "
    "духовный рост выше мнения окружающих. Прочитаем точное место.\n"
    "user: Жена говорит, что я стал раздражительным. "
    "Раньше пил и был весёлым.\n"
    "assistant: Раздражительность в начале трезвости описывается в книге "
    "как часть перестройки. Есть точные отрывки в главе «Семья после».\n"
    "user: А что делать со свободным временем по вечерам? "
    "Раньше это было время бутылки.\n"
    "assistant: Книга предлагает наполнить его действием: помощь другим, "
    "собрания, шаги. Обсудим конкретные примеры из текста?\n"
    "user: Я боюсь, что на Новый год не выдержу. Вся семья будет пить.\n"
    "assistant: Страх будущих ситуаций разбирается в книге "
    "через принцип «одного дня». Прочитаем точные места "
    "и подготовимся заранее.\n"
    "user: Спасибо. А можно коротко: в чём главная мысль третьей главы?\n"
    "assistant: Главная мысль: аллергическая тяга тела плюс одержимость "
    "ума лишают выбора. Подтвердим точными цитатами?"
)

TURN_BROAD_MIXED = """BROAD TURN (history + map + evidence):
HISTORY user(RU): У меня проблемы с тягой, семьёй и работой — всё сразу. С чего начать?
MAP: doctors-opinion(craving/illness); ch3(craving/allergy/obsession);
ch5(honesty/steps); ch7(working with others); ch8(to wives); ch10(to employers).
EVIDENCE [E1 ch3 allergy passage][E2 ch5 thorough path passage]
[E3 ch8 employer passage][E4 ch10 family passage]. Task: plan multi-aspect retrieval."""

ALIGNED_PAIRS: dict[str, dict[str, str]] = {
    "book_map": {"en": ALIGNED_BOOK_MAP_EN, "ru": ALIGNED_BOOK_MAP_RU},
    "evidence_pack": {"en": ALIGNED_EVIDENCE_EN, "ru": ALIGNED_EVIDENCE_RU},
    "planner_wrapper": {"en": ALIGNED_WRAPPER_EN, "ru": ALIGNED_WRAPPER_RU},
    "system_sample": {"en": SYSTEM_SAMPLE_EN, "ru": SYSTEM_SAMPLE_RU},
}
"""Aligned RU/EN pairs: same semantics and markers, language differs."""

RUSSIAN_HISTORIES: dict[str, str] = {
    "history_short": HISTORY_SHORT_RU,
    "history_medium": HISTORY_MEDIUM_RU,
    "history_long": HISTORY_LONG_RU,
}
"""Typical Russian Telegram conversation histories by length."""


def build_turn(*parts: str) -> str:
    """Compose one measured turn payload from ordered context parts."""
    return "\n".join(part for part in (MEASURE_INSTRUCTION, *parts) if part)
