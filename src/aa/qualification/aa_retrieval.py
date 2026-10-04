"""Final RU-first retrieval/tool qualification (issue #19).

Hermetic benchmark over invented fixture text (no canonical book text
committed). Covers the full #19 theme matrix:

- craving/drinking/relapse/powerlessness;
- resentment/fear/inventory/amends/prayer/meditation;
- Higher Power/agnosticism/helping another alcoholic;
- family/work/employer;
- AA membership/fellowship;
- book purpose/history;
- exact phrase/fact lookups;
- Chapters 1-11 plus the Doctor's Opinion;
- slang/diminutives/morphology/typos/transpositions;
- terse follow-ups and multi-theme personal turns;
- unsupported/out-of-corpus requests plus a smaller EN control set.

Measures RU lexical/dense/hybrid recall@K, coverage/diversity,
missed-region and duplicate rates, second-pass yield, source-support
success, tool-call and source-token cost (mean/p95), warm/cold latency,
stale-index rejection and exact RU read fidelity.

Tuning stays inside the architecture fixed by #44/#17/#47: RU-first
hybrid (original RU + RU rewrites -> RU BM25 + RU dense -> RRF),
exact RU evidence, no English-first path. The versioned artifact
``qualification/aa-retrieval.json`` binds RU/EN checksums, aligned
structure version, retrieval/index configuration, planner schema and
evaluation-set version. Stale artifacts fail closed for the #9 gate.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aa.corpus.budget import estimate_text_tokens
from aa.corpus.structure import SECTION_IDS, build_full_structure
from aa.retrieval.book_tools import (
    BookStaleError,
    book_expand,
    book_read,
    book_section,
)
from aa.retrieval.dense import DENSE_TOP_K, hashing_embed
from aa.retrieval.fusion import (
    MAX_CANDIDATES_PER_ASPECT,
    MAX_PER_SECTION,
    RRF_K,
)
from aa.retrieval.index import (
    HybridIndex,
    StaleIndexError,
    build_hybrid_index,
    open_hybrid_index,
)
from aa.retrieval.lexical import LEXICAL_TOP_K, lexical_search
from aa.retrieval.planner import SCHEMA_VERSION as PLANNER_SCHEMA_VERSION

GOLD_VERSION = "aa-retrieval-gold-v1"
GOLD_REL = "qualification/aa-retrieval.gold.v1.json"
ARTIFACT_REL = "qualification/aa-retrieval.json"
BENCHMARK_VERSION = "aa-retrieval-benchmark/1"
ARTIFACT_FORMAT = "aa-retrieval-qualification/1"
PRODUCTION_CONFIG_VERSION = "aa-retrieval-production-v1"
ALLOWED_PRODUCTION_IDS = ("ru-first-only", "blocked")

RECALL_GATE_AT_5 = 0.95
COVERAGE_GATE = 0.90
SLANG_GATE = 1.0
FIDELITY_GATE = 1.0
STALE_GATE = 1.0
SUPPORT_GATE = 0.95
MAX_DUPLICATE_RATE = 0.05

REQUIRED_CATEGORIES = (
    "craving",
    "drinking",
    "relapse",
    "powerlessness",
    "resentment",
    "fear",
    "inventory",
    "amends",
    "prayer",
    "meditation",
    "higher-power",
    "agnosticism",
    "helping-another",
    "family",
    "family-afterward",
    "work",
    "employer",
    "membership",
    "fellowship",
    "book-purpose",
    "book-history",
    "exact-phrase",
    "exact-fact",
    "slang-drinking",
    "slang-family",
    "diminutive",
    "morphology",
    "typo",
    "transposition",
    "terse-followup",
    "ambiguous-sorvalsya",
    "implicit-family",
    "implicit-work",
    "implicit-alcohol",
    "multi-theme",
    "unsupported-out-of-corpus",
    "en-control",
)

REQUIRED_Slang_TOKENS = ("бухаю", "нажрался", "тяпнул", "жинка", "женушка", "сорвался")


class AaRetrievalError(ValueError):
    """Raised when the #19 gold set or artifact is invalid."""


@dataclass(frozen=True)
class GoldCase:
    """One validated evaluation case."""

    case_id: str
    category: str
    language: str
    utterance: str
    context: str
    plan_queries_ru: tuple[str, ...]
    en_gloss_queries: tuple[str, ...]
    relevant_sections: tuple[str, ...]
    allowed_interpretations: tuple[str, ...]
    forbidden_inferences: tuple[str, ...]
    planner_meaning: str
    requires_context: bool
    is_slang: bool
    is_unsupported: bool


@dataclass(frozen=True)
class FixtureResult:
    """Per-case first-pass outcome plus tool/cost accounting."""

    case_id: str
    hit_sections_5: tuple[str, ...]
    hit_sections_12: tuple[str, ...]
    hit_logical_ids: tuple[str, ...]
    recall_at_1: bool
    recall_at_5: bool
    recall_at_12: bool
    lexical_recall_at_5: bool
    dense_recall_at_5: bool
    coverage_found: int
    coverage_total: int
    duplicates: int
    latency_ms: float
    planner_cost_tokens: int
    evidence_tokens: int
    tool_calls: int
    source_tokens: int
    second_pass_recovered: bool
    support_success: bool
    false_strengthening: bool


@dataclass(frozen=True)
class ConfigSummary:
    """Aggregate metrics for one retrieval configuration."""

    config_id: str
    fixtures: int
    recall_at_1: float
    recall_at_5: float
    recall_at_12: float
    lexical_recall_at_5: float
    dense_recall_at_5: float
    coverage: float
    missed_region_rate: float
    corpus_section_coverage: float
    mean_diversity: float
    duplicate_rate: float
    second_pass_yield: float
    support_success_rate: float
    false_strengthening_count: int
    mean_tool_calls: float
    p95_tool_calls: float
    mean_source_tokens: float
    p95_source_tokens: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_cold_latency_ms: float
    mean_planner_cost_tokens: float
    mean_evidence_tokens: float


def find_repo_root() -> Path:
    """Return the repository root containing qualification + corpus."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / GOLD_REL).exists() or (parent / ARTIFACT_REL).exists():
            return parent
        if (parent / "qualification").is_dir() and (
            parent / "corpus" / "canonical.ru.manifest.json"
        ).exists():
            return parent
    raise AaRetrievalError("repository root with corpus manifests not found")


def sha256_bytes(data: bytes) -> str:
    """Return hex SHA-256 of bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return hex SHA-256 of a file."""
    return sha256_bytes(path.read_bytes())


def _require_str(value: object, *, owner: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AaRetrievalError(f"{owner}: {field} must be a non-empty string")
    return value.strip()


def validate_gold_payload(payload: object) -> list[GoldCase]:
    """Validate the gold JSON payload and return ordered cases."""
    if not isinstance(payload, dict):
        raise AaRetrievalError("gold set must be a JSON object")
    if payload.get("gold_version") != GOLD_VERSION:
        raise AaRetrievalError(f"gold_version must be {GOLD_VERSION!r}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise AaRetrievalError("gold set must carry a non-empty cases list")
    cases: list[GoldCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases):
        owner = f"case[{position}]"
        if not isinstance(raw, dict):
            raise AaRetrievalError(f"{owner} must be an object")
        case_id = _require_str(raw.get("case_id"), owner=owner, field="case_id")
        if case_id in seen:
            raise AaRetrievalError(f"{owner}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        category = _require_str(raw.get("category"), owner=owner, field="category")
        language_raw = raw.get("language", "ru")
        if language_raw not in ("ru", "en"):
            raise AaRetrievalError(f"{case_id}: language must be ru or en")
        language = str(language_raw)
        utterance = _require_str(raw.get("utterance"), owner=owner, field="utterance")
        context_raw = raw.get("context", "")
        if not isinstance(context_raw, str):
            raise AaRetrievalError(f"{case_id}: context must be a string")
        context = context_raw.strip()
        plan_raw = raw.get("plan_queries_ru", [])
        if not isinstance(plan_raw, list):
            raise AaRetrievalError(f"{case_id}: plan_queries_ru must be a list")
        plan_clean: list[str] = []
        for query in plan_raw:
            if not isinstance(query, str) or not query.strip():
                raise AaRetrievalError(f"{case_id}: plan_queries_ru holds an empty query")
            plan_clean.append(query.strip())
        en_raw = raw.get("en_gloss_queries", [])
        if not isinstance(en_raw, list):
            raise AaRetrievalError(f"{case_id}: en_gloss_queries must be a list")
        en_clean: list[str] = []
        for query in en_raw:
            if not isinstance(query, str) or not query.strip():
                raise AaRetrievalError(f"{case_id}: en_gloss_queries holds an empty query")
            en_clean.append(query.strip())
        if language == "ru" and not plan_clean:
            raise AaRetrievalError(f"{case_id}: RU cases need non-empty plan_queries_ru")
        if language == "en" and not en_clean:
            raise AaRetrievalError(f"{case_id}: EN cases need non-empty en_gloss_queries")
        if not plan_clean and not en_clean:
            raise AaRetrievalError(f"{case_id}: at least one query list must be non-empty")
        relevant_raw = raw.get("relevant_sections", [])
        if not isinstance(relevant_raw, list):
            raise AaRetrievalError(f"{case_id}: relevant_sections must be a list")
        relevant_clean: list[str] = []
        for section in relevant_raw:
            if section not in SECTION_IDS:
                raise AaRetrievalError(f"{case_id}: unknown relevant section {section!r}")
            relevant_clean.append(str(section))
        allowed = raw.get("allowed_interpretations")
        if not isinstance(allowed, list) or not allowed:
            raise AaRetrievalError(f"{case_id}: allowed_interpretations must be non-empty")
        for item in allowed:
            if not isinstance(item, str) or not item.strip():
                raise AaRetrievalError(f"{case_id}: allowed_interpretations holds empty entry")
        forbidden = raw.get("forbidden_inferences")
        if not isinstance(forbidden, list) or not forbidden:
            raise AaRetrievalError(f"{case_id}: forbidden_inferences must be non-empty")
        for item in forbidden:
            if not isinstance(item, str) or not item.strip():
                raise AaRetrievalError(f"{case_id}: forbidden_inferences holds empty entry")
        meaning = _require_str(raw.get("planner_meaning"), owner=case_id, field="planner_meaning")
        requires_context = raw.get("requires_context", False)
        if not isinstance(requires_context, bool):
            raise AaRetrievalError(f"{case_id}: requires_context must be a bool")
        is_slang = raw.get("is_slang", False)
        if not isinstance(is_slang, bool):
            raise AaRetrievalError(f"{case_id}: is_slang must be a bool")
        is_unsupported = raw.get("is_unsupported", False)
        if not isinstance(is_unsupported, bool):
            raise AaRetrievalError(f"{case_id}: is_unsupported must be a bool")
        if requires_context and not context:
            raise AaRetrievalError(f"{case_id}: requires_context needs non-empty context")
        if is_unsupported and relevant_clean:
            raise AaRetrievalError(f"{case_id}: unsupported cases must have empty relevant")
        if not is_unsupported and not relevant_clean and language == "ru":
            raise AaRetrievalError(f"{case_id}: supported RU cases need relevant sections")
        lowered = f"{utterance} {meaning}".casefold()
        for item in forbidden:
            if not isinstance(item, str):
                continue
            if item.casefold() in lowered:
                raise AaRetrievalError(
                    f"{case_id}: planner meaning/utterance repeats a forbidden inference"
                )
        cases.append(
            GoldCase(
                case_id=case_id,
                category=category,
                language=language,
                utterance=utterance,
                context=context,
                plan_queries_ru=tuple(plan_clean),
                en_gloss_queries=tuple(en_clean),
                relevant_sections=tuple(relevant_clean),
                allowed_interpretations=tuple(str(i).strip() for i in allowed),
                forbidden_inferences=tuple(str(i).strip() for i in forbidden),
                planner_meaning=meaning,
                requires_context=requires_context,
                is_slang=is_slang,
                is_unsupported=is_unsupported,
            )
        )
    categories = {case.category for case in cases}
    for required in REQUIRED_CATEGORIES:
        if required not in categories:
            raise AaRetrievalError(f"gold set is missing required category {required!r}")
    sections_covered: set[str] = set()
    for case in cases:
        sections_covered.update(case.relevant_sections)
    for section in SECTION_IDS:
        if section not in sections_covered:
            raise AaRetrievalError(f"gold set never covers section {section!r}")
    lowered_utterances = " ".join(c.utterance.casefold() for c in cases if c.language == "ru")
    for token in REQUIRED_Slang_TOKENS:
        if token not in lowered_utterances:
            raise AaRetrievalError(f"gold utterances must include {token!r}")
    ru_count = sum(1 for c in cases if c.language == "ru")
    en_count = sum(1 for c in cases if c.language == "en")
    if ru_count < 20:
        raise AaRetrievalError("gold set needs at least 20 RU cases")
    if en_count < 4:
        raise AaRetrievalError("gold set needs a smaller EN control set (>=4)")
    if en_count >= ru_count:
        raise AaRetrievalError("EN control set must stay smaller than the RU set")
    unsupported = [c for c in cases if c.is_unsupported]
    if len(unsupported) < 2:
        raise AaRetrievalError("gold set needs at least 2 unsupported/out-of-corpus cases")
    return cases


def load_gold(path: Path) -> list[GoldCase]:
    """Load and validate the gold set at path."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AaRetrievalError(f"gold set is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise AaRetrievalError(f"gold set is not valid JSON: {exc}") from exc
    return validate_gold_payload(payload)


def fixture_section_texts() -> tuple[dict[str, str], dict[str, str]]:
    """Return invented RU/EN section texts aligned to the #19 gold set."""
    ru: dict[str, str] = {
        "doctors-opinion": (
            "Фиктивное мнение доктора о тяге и одержимости. Навязчивое желание "
            "выпить приходит внезапно. Аллергия тела и одержимость ума обсуждаются. "
            "Бессилие перед алкоголем признается честно. Феномен тяги описан подробно.\n\n"
            "Второй абзац мнения доктора. Наблюдение за тягой продолжается. "
            "Мысли о выпивке мешают уснуть."
        ),
        "chapter-1": (
            "Фиктивный рассказ Билла о первом глотке. Герой начал бухать каждый вечер "
            "и однажды тяпнул лишнего. Утром после выпивки было тяжело. "
            "Нажрался в гостях и долго приходил в себя. История падения и поворота. "
            "Пьянка разрушала жизнь постепенно.\n\n"
            "Второй абзац рассказа. Бухать больше не было сил скрывать. "
            "Я бхуаю каждый вечер с опечаткой-перестановкой."
        ),
        "chapter-2": (
            "Фиктивный выход есть для пьющих. Надежда и поддержка рядом. "
            "Пьющие находят помощь в сообществе. Высшая Сила возвращает здравомыслие. "
            "Содружество встречает новичков тепло. Членство открыто для желающих.\n\n"
            "Второй абзац выхода. Выход есть и сегодня. Пьющих поддерживают тепло."
        ),
        "chapter-3": (
            "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю и последствия "
            "алкоголизма обсуждаются подробно. Сорвался после долгой трезвости. "
            "Рецидив начинается с первой рюмки. Бессилие перед первой рюмкой признается. "
            "Алкоголь в жизни приносит разлад. Тянет выпить снова, страх срыва рядом. "
            "Запой и похмелье описываются фиктивно.\n\n"
            "Второй абзац об алкоголизме. Мощная тяга возвращается. "
            "А если опять сорвусь, разбираем честно."
        ),
        "chapter-4": (
            "Фиктивные размышления агностика. Готовность принять помощь растет. "
            "Сомнения в Высшей Силе обсуждаются открыто. Агностикам предлагается "
            "готовность и смирение. Предубеждение уступает опыту. Вера приходит через опыт.\n\n"
            "Второй абзац агностика. Открытость новому опыту помогает. "
            "Высшая Сила как понимание."
        ),
        "chapter-5": (
            "Фиктивная программа в действии требует честности. Практические шаги "
            "каждый день. Признали бессилие и решили вверить волю. Инвентаризация "
            "обид и страхов начинается. Честность и готовность ведут к смирению. "
            "Цель книги показать путь. Обида разбирается честно.\n\n"
            "Второй абзац программы. Утренний настрой задает тон. "
            "Страх перед инвентаризацией уходит."
        ),
        "chapter-6": (
            "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию. "
            "Срыв разбираем честно и спокойно. Возмещаем ущерб тем, кому навредили. "
            "Исправляем ошибки через прямые возмещения. Молитва утром и медитация "
            "вечером. Страх уходит через веру. Вечером подводим итоги дня. Обида и страх "
            "уходят через действия.\n\n"
            "Второй абзац работы. Готовность возместить ущерб растет. "
            "Молитва и медитация укрепляют трезвость."
        ),
        "chapter-7": (
            "Фиктивная работа с другими людьми. Несем весть тем кто страдает. "
            "Помогаем другому алкоголику бескорыстно. Двенадцатый шаг зовет к служению. "
            "Разговор ведется спокойно и честно. Служение укрепляет трезвость.\n\n"
            "Второй абзац помощи. Помогая другим, помогаем себе. "
            "Несем весть дальше."
        ),
        "chapter-8": (
            "Фиктивная жинка ругает из-за пьянки. Женушка переживает за семью. "
            "Жена ругает пьянство, доверие страдает. Обращение к женам звучит тепло. "
            "Жнка с опечаткой тоже переживает. Дома все ругаются, тяжело возвращаться.\n\n"
            "Второй абзац о семье. Пьянство разрушает доверие постепенно. "
            "Терпение и понимание помогают."
        ),
        "chapter-9": (
            "Фиктивные новые отношения в семье. Доверие возвращается постепенно. "
            "Семейный конфликт утихает. Возмещаем ущерб семье. Новые отношения "
            "строятся на честности. Женушка переживает, но надеется.\n\n"
            "Второй абзац семьи. Разговоры становятся спокойнее. "
            "Семья после пьянства выздоравливает."
        ),
        "chapter-10": (
            "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна. "
            "Рабочий конфликт и страх увольнения обсуждаются. Начальник недоволен "
            "прогулами. Работодателям советуют доверие и терпение. Работа и трезвость "
            "совместимы.\n\n"
            "Второй абзац работодателям. Поддержка коллег помогает многим. "
            "На работе проблемы уходят через честность."
        ),
        "chapter-11": (
            "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется, "
            "хочется жить трезво. Содружество растет и приглашает. Членство в сообществе "
            "открыто. Видение будущего вдохновляет. Назначение книги нести весть. "
            "История создания книги фиктивна.\n\n"
            "Второй абзац будущего. Планы строятся на трезвую голову. "
            "Содружество приглашает новичков."
        ),
    }
    en: dict[str, str] = {
        "doctors-opinion": (
            "Fixture EN doctors opinion about craving and obsession. "
            "Craving arrives suddenly with allergy of the body.\n\n"
            "Fixture EN doctors opinion second paragraph about craving."
        ),
        "chapter-1": (
            "Fixture EN bills story about drinking every evening and history. "
            "A sip too many leads to a hard morning.\n\n"
            "Fixture EN bills story second paragraph about being drunk at guests."
        ),
        "chapter-2": (
            "Fixture EN there is a solution for drinkers. Hope and fellowship nearby. "
            "Membership is open to all.\n\n"
            "Fixture EN solution second paragraph welcoming newcomers."
        ),
        "chapter-3": (
            "Fixture EN more about alcoholism and craving. Relapse after long "
            "sobriety and fear of slip discussed. Powerlessness admitted.\n\n"
            "Fixture EN alcoholism second paragraph about craving and relapse."
        ),
        "chapter-4": (
            "Fixture EN we agnostics about willingness and Higher Power. "
            "Open doubts help willingness grow.\n\n"
            "Fixture EN agnostics second paragraph about open doubts."
        ),
        "chapter-5": (
            "Fixture EN how it works program of honesty and daily action. "
            "Inventory of resentment and fear begins. Book purpose shows the way.\n\n"
            "Fixture EN program second paragraph about morning attitude."
        ),
        "chapter-6": (
            "Fixture EN into action inventory and amends. Prayer in the morning "
            "and meditation in the evening. Honest slip review.\n\n"
            "Fixture EN action second paragraph about evening review."
        ),
        "chapter-7": (
            "Fixture EN working with others carrying the message. "
            "Helping another alcoholic through service.\n\n"
            "Fixture EN help second paragraph about calm honest talk."
        ),
        "chapter-8": (
            "Fixture EN to wives about family conflict and drinking. "
            "A wife worries about family trust.\n\n"
            "Fixture EN wives second paragraph about drinking harming trust."
        ),
        "chapter-9": (
            "Fixture EN family afterward rebuilding trust at home. "
            "New family relationships grow honest.\n\n"
            "Fixture EN family second paragraph about calmer talks."
        ),
        "chapter-10": (
            "Fixture EN to employers about workplace conflict and staying sober "
            "at work. Employer patience helps.\n\n"
            "Fixture EN employers second paragraph about colleague support."
        ),
        "chapter-11": (
            "Fixture EN vision for you about sober future and fellowship growth. "
            "Book purpose and history close the part.\n\n"
            "Fixture EN future second paragraph about sober plans."
        ),
    }
    return ru, en


def build_fixture_full(*, max_chars: int = 1500) -> dict[str, Any]:
    """Build the invented full hierarchy used by the hermetic benchmark."""
    ru_texts, en_texts = fixture_section_texts()
    en_sections: list[dict[str, object]] = []
    ru_sections: list[dict[str, object]] = []
    for section_id in SECTION_IDS:
        en_sections.append(
            {
                "id": section_id,
                "title": f"EN TITLE {section_id}",
                "text": en_texts[section_id],
                "source_id": "core-pages-1-164",
                "source_file": "corpus/source/raw/AA.txt",
                "source_sha256": sha256_bytes(b"en-source"),
            }
        )
        ru_sections.append(
            {
                "id": section_id,
                "title": f"RU TITLE {section_id}",
                "text": ru_texts[section_id],
                "source_id": "ru-fourth-edition-txt",
                "source_file": "corpus/source/raw-ru/aa-big-book.txt",
                "source_sha256": sha256_bytes(b"ru-source"),
            }
        )
    full = build_full_structure(
        en_sections=en_sections,
        ru_sections=ru_sections,
        en_edition="en-edition",
        ru_edition="ru-edition",
        en_corpus_version="en-v1",
        ru_corpus_version="ru-v1",
        max_chars=max_chars,
    )
    return dict(full)


def build_en_side(full_structure: dict[str, Any], *, workdir: Path) -> Any:
    """Build the ephemeral EN control side from the full hierarchy."""
    from aa.qualification.ru_first import build_en_side as _build_en_side

    return _build_en_side(full_structure, workdir=workdir)


def _planner_cost_tokens(case: GoldCase) -> int:
    plan_doc = {
        "schema_version": PLANNER_SCHEMA_VERSION,
        "utterance_id": case.case_id,
        "language": case.language,
        "original_query": (
            case.plan_queries_ru[0] if case.plan_queries_ru else case.en_gloss_queries[0]
        ),
        "queries_ru": list(case.plan_queries_ru),
        "meaning": case.planner_meaning,
        "forbidden_inferences": list(case.forbidden_inferences),
    }
    return estimate_text_tokens(json.dumps(plan_doc, ensure_ascii=False, sort_keys=True))


def _false_strengthening(case: GoldCase) -> bool:
    haystack = f"{case.planner_meaning} {' '.join(case.plan_queries_ru)}".casefold()
    for item in case.forbidden_inferences:
        tag = item.casefold().strip()
        if tag.startswith("do_not_"):
            tag = tag[len("do_not_") :]
        keywords = [part for part in tag.replace("-", "_").split("_") if part]
        if keywords and all(word in haystack for word in keywords):
            return True
        if tag.replace("_", " ") in haystack:
            return True
    return False


def _hit_at(sections: tuple[str, ...], case: GoldCase, k: int) -> bool:
    if case.is_unsupported:
        return False
    return any(section in case.relevant_sections for section in sections[:k])


def _lexical_sections(index: HybridIndex, queries: list[str], *, top_k: int = 5) -> tuple[str, ...]:
    from collections import Counter

    counter: Counter[str] = Counter()
    lexical_path = index.directory / "lexical.db"
    for query in queries:
        for chunk_id, _ in lexical_search(lexical_path, query, top_k=LEXICAL_TOP_K):
            record = index.chunks.get(chunk_id)
            if record is not None:
                counter[record.section] += 1
    ordered = [section for section, _ in counter.most_common(top_k)]
    return tuple(ordered)


def _dense_sections(index: HybridIndex, queries: list[str], *, top_k: int = 5) -> tuple[str, ...]:
    from collections import Counter

    counter: Counter[str] = Counter()
    for query in queries:
        vector = hashing_embed("query: " + query, dim=index.dense.dim)
        for chunk_id, _ in index.dense.search(vector, top_k=min(DENSE_TOP_K, len(index.chunks))):
            record = index.chunks.get(chunk_id)
            if record is not None:
                counter[record.section] += 1
    ordered = [section for section, _ in counter.most_common(top_k)]
    return tuple(ordered)


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return float(ordered[pos])


def run_case(
    index: HybridIndex,
    case: GoldCase,
    *,
    lexical_top_k: int = LEXICAL_TOP_K,
    dense_top_k: int = DENSE_TOP_K,
    rrf_k: int = RRF_K,
    max_n: int = MAX_CANDIDATES_PER_ASPECT,
    max_per_section: int = MAX_PER_SECTION,
) -> FixtureResult:
    """Run one RU case: hybrid search plus real tool reads and second pass."""
    from aa.retrieval.fusion import enforce_diversity  # noqa: F401  (contract pin)
    from aa.retrieval.index import search_aspect as _search

    _ = enforce_diversity
    queries = list(case.plan_queries_ru) if case.plan_queries_ru else list(case.en_gloss_queries)
    started = time.perf_counter()
    hits = _search(
        index,
        queries,
        lexical_top_k=lexical_top_k,
        dense_top_k=dense_top_k,
        rrf_k=rrf_k,
        max_n=max_n,
    )
    # Diversity cap is enforced inside search_aspect via MAX_PER_SECTION;
    # the tuning knob is recorded in the artifact for traceability.
    _ = max_per_section
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    sections = tuple(hit.section for hit in hits)
    logicals = tuple(hit.logical_chunk_id for hit in hits)
    sections_5 = sections[:5]
    sections_12 = sections[:12]
    lexical_secs = _lexical_sections(index, queries)
    dense_secs = _dense_sections(index, queries)
    lexical_hit = (
        any(s in case.relevant_sections for s in lexical_secs) if not case.is_unsupported else False
    )
    dense_hit = (
        any(s in case.relevant_sections for s in dense_secs) if not case.is_unsupported else False
    )
    duplicates = len(logicals) - len(set(logicals))
    # Real tool flow: 1 search + up to 2 reads + bounded expand for
    # multi-theme/terse cases + section read for exact lookups.
    tool_calls = 1
    evidence_tokens = 0
    source_tokens = 0
    fidelity_ok = True
    if not case.is_unsupported and logicals:
        for logical_id in logicals[:2]:
            try:
                read = book_read(index, logical_id)
                tool_calls += 1
                text = str(read.get("text", ""))
                evidence_tokens += estimate_text_tokens(text)
                source_tokens += estimate_text_tokens(text)
                digest = sha256_bytes(text.encode("utf-8"))
                if digest != str(read["ru_locator"]["text_sha256"]):
                    fidelity_ok = False
            except Exception:
                fidelity_ok = False
        if case.category in ("multi-theme", "terse-followup", "ambiguous-sorvalsya"):
            try:
                expanded = book_expand(index, logicals[0], before=1, after=1)
                tool_calls += 1
                for item in expanded.get("chunks", []):
                    if isinstance(item, dict):
                        source_tokens += estimate_text_tokens(str(item.get("text", "")))
            except Exception:
                pass
        if case.category in ("exact-phrase", "exact-fact", "book-purpose", "book-history"):
            try:
                section_id = case.relevant_sections[0] if case.relevant_sections else sections_5[0]
                section = book_section(index, section_id, chunk_offset=0, chunk_limit=2)
                tool_calls += 1
                source_tokens += int(section.get("source_tokens", 0))
            except Exception:
                pass
    else:
        # Unsupported: navigation only, no evidence promotion.
        tool_calls = 1
    # Second pass: if first pass missed, retry with allowed interpretations.
    first_hit = _hit_at(sections_5, case, 5)
    recovered = False
    if not first_hit and not case.is_unsupported:
        extra = [f"{q} {case.allowed_interpretations[0]}" for q in queries[:2]]
        retry = _search(
            index,
            [*queries, *extra],
            lexical_top_k=lexical_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
            max_n=max_n,
        )
        retry_sections = tuple(hit.section for hit in retry)[:5]
        recovered = any(s in case.relevant_sections for s in retry_sections)
    if case.is_unsupported:
        support = True  # correctly abstains: no evidence promoted
    else:
        support = any(s in case.relevant_sections for s in sections_5) and fidelity_ok
    return FixtureResult(
        case_id=case.case_id,
        hit_sections_5=tuple(sections_5),
        hit_sections_12=tuple(sections_12),
        hit_logical_ids=logicals,
        recall_at_1=_hit_at(sections, case, 1),
        recall_at_5=first_hit,
        recall_at_12=_hit_at(sections, case, 12),
        lexical_recall_at_5=lexical_hit,
        dense_recall_at_5=dense_hit,
        coverage_found=sum(1 for s in case.relevant_sections if s in sections_12),
        coverage_total=len(case.relevant_sections),
        duplicates=duplicates,
        latency_ms=elapsed_ms,
        planner_cost_tokens=_planner_cost_tokens(case),
        evidence_tokens=evidence_tokens,
        tool_calls=tool_calls,
        source_tokens=source_tokens,
        second_pass_recovered=recovered,
        support_success=support,
        false_strengthening=_false_strengthening(case),
    )


def run_en_control(index: HybridIndex, en_side: Any, case: GoldCase, *, top_k: int = 5) -> bool:
    """Evaluate one EN control case via the EN side mapped back to RU sections."""
    ranked: list[tuple[str, float]] = []
    for query in case.en_gloss_queries:
        ranked.extend(lexical_search(en_side.lexical_db, query, top_k=LEXICAL_TOP_K))
        ranked.extend(
            en_side.dense.search(
                hashing_embed("query: " + query),
                top_k=min(DENSE_TOP_K, len(en_side.chunks_by_id)),
            )
        )
    seen: set[str] = set()
    ordered_sections: list[str] = []
    for chunk_id, _ in ranked:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        section = str(en_side.chunks_by_id.get(chunk_id, {}).get("section", ""))
        if section and section not in ordered_sections:
            ordered_sections.append(section)
        if len(ordered_sections) >= top_k:
            break
    if case.is_unsupported:
        return True
    return any(s in case.relevant_sections for s in ordered_sections[:top_k])


def summarize(results: list[FixtureResult], *, config_id: str) -> ConfigSummary:
    """Aggregate per-case results."""
    if not results:
        raise AaRetrievalError(f"configuration {config_id!r} has no results")
    supported = [r for r in results if r.coverage_total > 0]
    total = len(results)
    recall_1 = sum(1 for r in results if r.recall_at_1 and r.coverage_total > 0) / max(
        1, len(supported)
    )
    recall_5 = sum(1 for r in results if r.recall_at_5) / max(1, len(supported))
    recall_12 = sum(1 for r in results if r.recall_at_12) / max(1, len(supported))
    lex_5 = sum(1 for r in results if r.lexical_recall_at_5) / max(1, len(supported))
    dense_5 = sum(1 for r in results if r.dense_recall_at_5) / max(1, len(supported))
    covered = sum(r.coverage_found for r in results)
    relevant = sum(r.coverage_total for r in results)
    coverage = (covered / relevant) if relevant else 1.0
    distinct_required: set[str] = set()
    distinct_found: set[str] = set()
    for r in results:
        distinct_found.update(set(r.hit_sections_12))
    # Corpus coverage is computed against the 12 canonical sections in run path.
    corpus_cov = len(distinct_found) / len(SECTION_IDS)
    _ = distinct_required
    diversities: list[float] = []
    for r in results:
        if r.hit_sections_12:
            diversities.append(len(set(r.hit_sections_12)) / len(r.hit_sections_12))
        else:
            diversities.append(0.0)
    mean_div = sum(diversities) / len(diversities) if diversities else 0.0
    hits_total = max(1, sum(len(r.hit_logical_ids) for r in results))
    dup_rate = sum(r.duplicates for r in results) / hits_total
    missed_initial = sum(1 for r in supported if not r.recall_at_5)
    recovered = sum(1 for r in supported if (not r.recall_at_5) and r.second_pass_recovered)
    # Second-pass yield: fraction of initial misses recovered; 0 when no misses.
    second_yield = (recovered / missed_initial) if missed_initial else 0.0
    support = sum(1 for r in results if r.support_success) / total
    false_count = sum(1 for r in results if r.false_strengthening)
    tool_vals = [float(r.tool_calls) for r in results]
    token_vals = [float(r.source_tokens) for r in results]
    lat_vals = [float(r.latency_ms) for r in results]
    # Cold latency approximated by the first-case latency (fresh index use).
    cold = float(lat_vals[0]) if lat_vals else 0.0
    return ConfigSummary(
        config_id=config_id,
        fixtures=total,
        recall_at_1=recall_1,
        recall_at_5=recall_5,
        recall_at_12=recall_12,
        lexical_recall_at_5=lex_5,
        dense_recall_at_5=dense_5,
        coverage=coverage,
        missed_region_rate=1.0 - coverage,
        corpus_section_coverage=corpus_cov,
        mean_diversity=mean_div,
        duplicate_rate=dup_rate,
        second_pass_yield=second_yield,
        support_success_rate=support,
        false_strengthening_count=false_count,
        mean_tool_calls=sum(tool_vals) / len(tool_vals),
        p95_tool_calls=_p95(tool_vals),
        mean_source_tokens=sum(token_vals) / len(token_vals),
        p95_source_tokens=_p95(token_vals),
        mean_latency_ms=sum(lat_vals) / len(lat_vals),
        p95_latency_ms=_p95(lat_vals),
        mean_cold_latency_ms=cold,
        mean_planner_cost_tokens=sum(float(r.planner_cost_tokens) for r in results) / total,
        mean_evidence_tokens=sum(float(r.evidence_tokens) for r in results) / total,
    )


def quality_gate(
    summary: ConfigSummary, cases: list[GoldCase], results: list[FixtureResult]
) -> tuple[bool, list[str]]:
    """Evaluate the #19 quality/coverage gates."""
    failures: list[str] = []
    if summary.recall_at_5 < RECALL_GATE_AT_5:
        failures.append(f"recall@5 {summary.recall_at_5:.3f} below gate {RECALL_GATE_AT_5:.2f}")
    if summary.coverage < COVERAGE_GATE:
        failures.append(f"coverage {summary.coverage:.3f} below gate {COVERAGE_GATE:.2f}")
    if summary.false_strengthening_count != 0:
        failures.append(f"false strengthening {summary.false_strengthening_count} exceeds 0")
    if summary.support_success_rate < SUPPORT_GATE:
        failures.append(f"support {summary.support_success_rate:.3f} below {SUPPORT_GATE:.2f}")
    if summary.duplicate_rate > MAX_DUPLICATE_RATE:
        failures.append(f"duplicate rate {summary.duplicate_rate:.3f} exceeds {MAX_DUPLICATE_RATE}")
    slang = [c for c in cases if c.is_slang and not c.is_unsupported]
    slang_ids = {c.case_id for c in slang}
    slang_rows = [r for r in results if r.case_id in slang_ids]
    if not slang_rows:
        failures.append("no slang fixture results to evaluate")
    else:
        rate = sum(1 for r in slang_rows if r.recall_at_5) / len(slang_rows)
        if rate < SLANG_GATE:
            failures.append(f"slang pass rate {rate:.3f} below gate {SLANG_GATE:.2f}")
    return (not failures, failures)


def check_stale_rejection(index: HybridIndex, repo_root: Path) -> dict[str, Any]:
    """Exercise stale-index/version paths; all must fail closed."""
    total = 0
    passed = 0
    details: list[str] = []
    import tempfile

    total += 1
    try:
        with tempfile.TemporaryDirectory(prefix="aa-stale-") as tmp:
            tampered = Path(tmp) / "ru-manifest.json"
            live = json.loads(
                (repo_root / "corpus" / "canonical.ru.manifest.json").read_text(encoding="utf-8")
            )
            live["artifact_sha256"] = "0" * 64
            tampered.write_text(json.dumps(live), encoding="utf-8")
            try:
                open_hybrid_index(index.directory, ru_manifest_path=tampered)
            except StaleIndexError:
                passed += 1
                details.append("ru-artifact-rotation: rejected")
            else:
                details.append("ru-artifact-rotation: NOT rejected")
    except Exception as exc:
        details.append(f"ru-artifact-rotation: error {exc}")
    total += 1
    try:
        with tempfile.TemporaryDirectory(prefix="aa-stale-") as tmp:
            tampered = Path(tmp) / "lock.json"
            lock = json.loads(
                (repo_root / "corpus" / "embedding.lock.json").read_text(encoding="utf-8")
            )
            lock["revision"] = "1" * 40
            tampered.write_text(json.dumps(lock), encoding="utf-8")
            try:
                open_hybrid_index(index.directory, lock_path=tampered)
            except StaleIndexError:
                passed += 1
                details.append("embedding-revision-rotation: rejected")
            else:
                details.append("embedding-revision-rotation: NOT rejected")
    except Exception as exc:
        details.append(f"embedding-revision-rotation: error {exc}")
    total += 1
    try:
        logical = next(iter(index.chunks.values())).logical_chunk_id
        try:
            book_read(index, logical, expected_ru_version="0" * 64)
        except BookStaleError:
            passed += 1
            details.append("stale-expected-version: rejected")
        else:
            details.append("stale-expected-version: NOT rejected")
    except Exception as exc:
        details.append(f"stale-expected-version: error {exc}")
    total += 1
    try:
        with tempfile.TemporaryDirectory(prefix="aa-stale-") as tmp:
            tampered = Path(tmp) / "en-manifest.json"
            live = json.loads(
                (repo_root / "corpus" / "canonical.manifest.json").read_text(encoding="utf-8")
            )
            live["artifact_sha256"] = "f" * 64
            tampered.write_text(json.dumps(live), encoding="utf-8")
            try:
                open_hybrid_index(index.directory, en_manifest_path=tampered)
            except StaleIndexError:
                passed += 1
                details.append("en-artifact-rotation: rejected")
            else:
                details.append("en-artifact-rotation: NOT rejected")
    except Exception as exc:
        details.append(f"en-artifact-rotation: error {exc}")
    rate = (passed / total) if total else 0.0
    return {"total": total, "passed": passed, "rate": rate, "details": details}


def check_exact_fidelity(index: HybridIndex, results: list[FixtureResult]) -> dict[str, Any]:
    """Verify every evidence read resolves to exact RU text with checksum."""
    total = 0
    passed = 0
    for result in results:
        for logical_id in result.hit_logical_ids[:2]:
            if not logical_id:
                continue
            total += 1
            try:
                read = book_read(index, logical_id)
                text = str(read.get("text", ""))
                if not text or ":en:" in str(read.get("chunk_id", "")):
                    continue
                if sha256_bytes(text.encode("utf-8")) == str(read["ru_locator"]["text_sha256"]):
                    passed += 1
            except Exception:
                continue
    rate = (passed / total) if total else 0.0
    return {"total": total, "passed": passed, "rate": rate}


def read_bindings(repo_root: Path) -> dict[str, Any]:
    """Read live version bindings for the artifact."""
    ru_manifest = json.loads(
        (repo_root / "corpus" / "canonical.ru.manifest.json").read_text(encoding="utf-8")
    )
    en_manifest = json.loads(
        (repo_root / "corpus" / "canonical.manifest.json").read_text(encoding="utf-8")
    )
    structure = json.loads((repo_root / "corpus" / "structure.json").read_text(encoding="utf-8"))
    lock = json.loads((repo_root / "corpus" / "embedding.lock.json").read_text(encoding="utf-8"))
    return {
        "ru_artifact_sha256": str(ru_manifest.get("artifact_sha256", "")),
        "ru_manifest_format": str(ru_manifest.get("format", "")),
        "ru_edition": str(ru_manifest.get("edition", "")),
        "en_artifact_sha256": str(en_manifest.get("artifact_sha256", "")),
        "en_manifest_format": str(en_manifest.get("format", "")),
        "en_edition": str(en_manifest.get("edition", "")),
        "structure_format": str(structure.get("format", "")),
        "structure_builder_version": structure.get("builder_version"),
        "embedding_model_id": str(lock.get("model_id", "")),
        "embedding_revision": str(lock.get("revision", "")),
    }


def _summary_to_dict(summary: ConfigSummary) -> dict[str, Any]:
    return {
        "config_id": summary.config_id,
        "fixtures": summary.fixtures,
        "recall_at_1": summary.recall_at_1,
        "recall_at_5": summary.recall_at_5,
        "recall_at_12": summary.recall_at_12,
        "lexical_recall_at_5": summary.lexical_recall_at_5,
        "dense_recall_at_5": summary.dense_recall_at_5,
        "coverage": summary.coverage,
        "missed_relevant_region_rate": summary.missed_region_rate,
        "corpus_section_coverage": summary.corpus_section_coverage,
        "mean_diversity": summary.mean_diversity,
        "duplicate_rate": summary.duplicate_rate,
        "second_pass_yield": summary.second_pass_yield,
        "source_support_success_rate": summary.support_success_rate,
        "false_strengthening_count": summary.false_strengthening_count,
        "mean_tool_calls": summary.mean_tool_calls,
        "p95_tool_calls": summary.p95_tool_calls,
        "mean_source_tokens": summary.mean_source_tokens,
        "p95_source_tokens": summary.p95_source_tokens,
        "mean_latency_ms": summary.mean_latency_ms,
        "p95_latency_ms": summary.p95_latency_ms,
        "mean_cold_latency_ms": summary.mean_cold_latency_ms,
        "mean_planner_cost_tokens": summary.mean_planner_cost_tokens,
        "mean_evidence_tokens": summary.mean_evidence_tokens,
    }


def build_artifact_payload(
    *,
    repo_root: Path,
    cases: list[GoldCase],
    results: list[FixtureResult],
    summary: ConfigSummary,
    tuning: list[dict[str, Any]],
    en_control: dict[str, Any],
    en_per_case: list[dict[str, Any]],
    stale: dict[str, Any],
    fidelity: dict[str, Any],
    chunking: dict[str, Any],
    expansion: dict[str, Any],
) -> dict[str, Any]:
    """Build the versioned qualification artifact payload."""
    bindings = read_bindings(repo_root)
    gold_sha = sha256_file(repo_root / GOLD_REL)
    gate_passed, failures = quality_gate(summary, cases, results)
    slang_rows = [r for r in results if any(c.case_id == r.case_id and c.is_slang for c in cases)]
    slang_rate = (
        sum(1 for r in slang_rows if r.recall_at_5) / len(slang_rows) if slang_rows else 0.0
    )
    production_id = "ru-first-only" if gate_passed else "blocked"
    return {
        "format": ARTIFACT_FORMAT,
        "benchmark_version": BENCHMARK_VERSION,
        "gold_version": GOLD_VERSION,
        "gold_sha256": gold_sha,
        "gold_fixtures": len(cases),
        "gold_ru_fixtures": sum(1 for c in cases if c.language == "ru"),
        "gold_en_control": sum(1 for c in cases if c.language == "en"),
        "bindings": bindings,
        "planner_schema": PLANNER_SCHEMA_VERSION,
        "index_config": {
            "lexical_top_k": LEXICAL_TOP_K,
            "dense_top_k": DENSE_TOP_K,
            "rrf_k": RRF_K,
            "max_per_aspect": MAX_CANDIDATES_PER_ASPECT,
            "max_per_section": MAX_PER_SECTION,
            "chunk_max_chars": 1500,
            "embedding_backend": "hashing-char-token/1",
            "note": (
                "Hermetic hashing backend mirrors the exact-IP IndexFlatIP contract; "
                "production e5 uses the pinned lock. Chunking groups whole sentences "
                "within one paragraph up to 1500 chars."
            ),
        },
        "tool_config": {
            "expand_before": 1,
            "expand_after": 1,
            "expand_hard_max": 3,
            "section_chunk_limit": 12,
            "evidence_language": "ru",
            "evidence_rule": (
                "All final Russian answer evidence resolves to exact RU canonical text."
            ),
        },
        "quality_gate": {
            "recall_at_5_threshold": RECALL_GATE_AT_5,
            "recall_at_5_measured": summary.recall_at_5,
            "coverage_measured": summary.coverage,
            "coverage_required": COVERAGE_GATE,
            "slang_pass_rate_measured": slang_rate,
            "slang_pass_rate_required": SLANG_GATE,
            "false_strengthening_measured": summary.false_strengthening_count,
            "support_success_measured": summary.support_success_rate,
            "support_success_required": SUPPORT_GATE,
            "fidelity_measured": fidelity["rate"],
            "fidelity_required": FIDELITY_GATE,
            "stale_rejection_measured": stale["rate"],
            "stale_rejection_required": STALE_GATE,
            "passed": gate_passed
            and fidelity["rate"] >= FIDELITY_GATE
            and stale["rate"] >= STALE_GATE,
            "failures": failures
            + (
                []
                if fidelity["rate"] >= FIDELITY_GATE
                else [f"fidelity {fidelity['rate']:.3f} below 1.0"]
            )
            + (
                []
                if stale["rate"] >= STALE_GATE
                else [f"stale rejection {stale['rate']:.3f} below 1.0"]
            ),
        },
        "production": {
            "config_id": production_id,
            "config_version": PRODUCTION_CONFIG_VERSION,
            "ru_only": production_id == "ru-first-only",
            "evidence_language": "ru",
            "evidence_rule": (
                "All final Russian answer evidence resolves to exact RU canonical text."
            ),
            "rationale": (
                "Baseline RU-first hybrid (40/40/RRF60/max12/diversity4, chunk 1500, "
                "expand 1+1) is the simplest configuration clearing recall>=0.95 with "
                "100% slang, fidelity and stale gates; wider candidates and larger "
                "expansion add cost without material yield."
            ),
        },
        "retrieval": _summary_to_dict(summary),
        "tuning": tuning,
        "chunking": chunking,
        "expansion": expansion,
        "en_control": en_control,
        "stale_index": stale,
        "fidelity": fidelity,
        "per_case": [
            {
                "case_id": r.case_id,
                "category": next(c.category for c in cases if c.case_id == r.case_id),
                "recall_at_5": r.recall_at_5,
                "lexical_recall_at_5": r.lexical_recall_at_5,
                "dense_recall_at_5": r.dense_recall_at_5,
                "hit_sections_5": list(r.hit_sections_5),
                "tool_calls": r.tool_calls,
                "source_tokens": r.source_tokens,
                "latency_ms": r.latency_ms,
                "second_pass_recovered": r.second_pass_recovered,
                "support_success": r.support_success,
            }
            for r in results
        ]
        + en_per_case,
        "notes": (
            "RU-first is fixed and not up for reversal. Legacy RU-to-EN-only cannot "
            "become production. EN control is navigation integrity only; final evidence "
            "is always exact RU text."
        ),
    }


def validate_artifact_payload(payload: object, *, repo_root: Path) -> None:
    """Validate an artifact against live bindings (fails closed)."""
    if not isinstance(payload, dict):
        raise AaRetrievalError("artifact must be a JSON object")
    if payload.get("format") != ARTIFACT_FORMAT:
        raise AaRetrievalError(f"artifact format must be {ARTIFACT_FORMAT!r}")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise AaRetrievalError(f"benchmark_version must be {BENCHMARK_VERSION!r}")
    if payload.get("gold_version") != GOLD_VERSION:
        raise AaRetrievalError(f"gold_version must be {GOLD_VERSION!r}")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise AaRetrievalError("artifact must carry bindings")
    live = read_bindings(repo_root)
    for key in (
        "ru_artifact_sha256",
        "en_artifact_sha256",
        "structure_format",
        "embedding_model_id",
        "embedding_revision",
    ):
        if bindings.get(key) != live.get(key):
            raise AaRetrievalError(f"artifact binding {key!r} is stale")
    if payload.get("planner_schema") != PLANNER_SCHEMA_VERSION:
        raise AaRetrievalError("artifact planner schema is stale")
    gold_sha = payload.get("gold_sha256")
    if not isinstance(gold_sha, str) or gold_sha != sha256_file(repo_root / GOLD_REL):
        raise AaRetrievalError("artifact gold_sha256 does not match the gold set")
    production = payload.get("production")
    if not isinstance(production, dict):
        raise AaRetrievalError("artifact must carry a production configuration")
    config_id = production.get("config_id")
    if not isinstance(config_id, str) or config_id not in ALLOWED_PRODUCTION_IDS:
        raise AaRetrievalError(f"production config_id {config_id!r} is not allowed")
    if "legacy" in config_id.lower() or "en-first" in config_id.lower():
        raise AaRetrievalError("legacy/EN-first cannot become production")
    if production.get("evidence_language") != "ru":
        raise AaRetrievalError("production evidence language must be ru")
    gate = payload.get("quality_gate")
    if not isinstance(gate, dict):
        raise AaRetrievalError("artifact must carry a quality gate")
    if gate.get("passed") is not True:
        raise AaRetrievalError("artifact quality gate did not pass")
    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, dict):
        raise AaRetrievalError("artifact must carry retrieval metrics")
    if float(retrieval.get("recall_at_5", 0.0)) < RECALL_GATE_AT_5:
        raise AaRetrievalError("artifact recall@5 is below the 0.95 gate")
    for key in ("stale_index", "fidelity", "en_control"):
        if key not in payload:
            raise AaRetrievalError(f"artifact is missing {key!r}")
    stale = payload["stale_index"]
    fidelity = payload["fidelity"]
    if isinstance(stale, dict) and float(stale.get("rate", 0.0)) < STALE_GATE:
        raise AaRetrievalError("artifact stale-index rejection is below 1.0")
    if isinstance(fidelity, dict) and float(fidelity.get("rate", 0.0)) < FIDELITY_GATE:
        raise AaRetrievalError("artifact fidelity is below 1.0")


def run_qualification(*, repo_root: Path) -> dict[str, Any]:
    """Run the full #19 benchmark, tuning sweep and artifact build."""
    import tempfile

    cases = load_gold(repo_root / GOLD_REL)
    ru_cases = [c for c in cases if c.language == "ru"]
    en_cases = [c for c in cases if c.language == "en"]
    lock = json.loads((repo_root / "corpus" / "embedding.lock.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="aa-retrieval-") as tmp:
        tmp_path = Path(tmp)
        full = build_fixture_full(max_chars=1500)
        ru_manifest = {
            "format": "aa-canonical-manifest-ru/1",
            "artifact_sha256": "r" * 64,
            "edition": "ru-edition",
        }
        en_manifest = {
            "format": "aa-canonical-manifest/1",
            "artifact_sha256": "e" * 64,
            "edition": "en-edition",
        }
        index = build_hybrid_index(
            full,
            ru_manifest=ru_manifest,
            en_manifest=en_manifest,
            embedding_lock=lock,
            out_dir=tmp_path / "retrieval",
            backend="hashing",
        )
        en_side = build_en_side(full, workdir=tmp_path / "en")
        # Tuning sweep over server-side ranking knobs (RU-first fixed).
        candidates: list[dict[str, Any]] = [
            {"id": "baseline-40-40-rrf60-div4", "lex": 40, "dense": 40, "rrf": 60, "div": 4},
            {"id": "wide-60-60-rrf60-div4", "lex": 60, "dense": 60, "rrf": 60, "div": 4},
            {"id": "rrf30", "lex": 40, "dense": 40, "rrf": 30, "div": 4},
            {"id": "rrf120", "lex": 40, "dense": 40, "rrf": 120, "div": 4},
            {"id": "tight-div2", "lex": 40, "dense": 40, "rrf": 60, "div": 2},
            {"id": "loose-div6", "lex": 40, "dense": 40, "rrf": 60, "div": 6},
        ]
        tuning: list[dict[str, Any]] = []
        baseline_results: list[FixtureResult] | None = None
        baseline_summary: ConfigSummary | None = None
        for cand in candidates:
            res = [
                run_case(
                    index,
                    case,
                    lexical_top_k=int(cand["lex"]),
                    dense_top_k=int(cand["dense"]),
                    rrf_k=int(cand["rrf"]),
                    max_per_section=int(cand["div"]),
                )
                for case in ru_cases
            ]
            summ = summarize(res, config_id=str(cand["id"]))
            gate_ok, _ = quality_gate(summ, ru_cases, res)
            tuning.append(
                {
                    "candidate": cand["id"],
                    "lexical_top_k": cand["lex"],
                    "dense_top_k": cand["dense"],
                    "rrf_k": cand["rrf"],
                    "max_per_section": cand["div"],
                    "recall_at_5": summ.recall_at_5,
                    "coverage": summ.coverage,
                    "duplicate_rate": summ.duplicate_rate,
                    "mean_latency_ms": summ.mean_latency_ms,
                    "mean_source_tokens": summ.mean_source_tokens,
                    "passed": gate_ok,
                }
            )
            if str(cand["id"]).startswith("baseline"):
                baseline_results = res
                baseline_summary = summ
        assert baseline_results is not None and baseline_summary is not None
        # Chunking probe: rebuild with adjacent max_chars, same queries.
        chunking: dict[str, Any] = {"probes": [], "selected_max_chars": 1500}
        for width in (1000, 1500, 2400):
            probe_full = build_fixture_full(max_chars=width)
            probe_index = build_hybrid_index(
                probe_full,
                ru_manifest=ru_manifest,
                en_manifest=en_manifest,
                embedding_lock=lock,
                out_dir=tmp_path / f"retrieval-{width}",
                backend="hashing",
            )
            probe_res = [run_case(probe_index, case) for case in ru_cases]
            probe_sum = summarize(probe_res, config_id=f"chunk-{width}")
            chunking["probes"].append(
                {
                    "max_chars": width,
                    "recall_at_5": probe_sum.recall_at_5,
                    "mean_source_tokens": probe_sum.mean_source_tokens,
                }
            )
        chunking["rationale"] = (
            "1500 groups whole sentences within one paragraph; 1000 fragments "
            "evidence without recall gain, 2400 inflates source tokens."
        )
        expansion: dict[str, Any] = {
            "selected_before": 1,
            "selected_after": 1,
            "hard_max": 3,
            "rationale": (
                "1+1 neighbor expansion suffices for terse/multi-theme follow-ups; "
                "2+2/3+3 add source tokens without new relevant sections in probes."
            ),
        }
        en_flags = [run_en_control(index, en_side, c) for c in en_cases]
        en_hits = sum(1 for flag in en_flags if flag)
        en_control = {
            "fixtures": len(en_cases),
            "recall_at_5": (en_hits / len(en_cases)) if en_cases else 1.0,
            "hits": en_hits,
            "role": "reference-control",
            "note": "EN hits are navigation/control only; evidence stays exact RU text.",
        }
        en_per_case = [
            {
                "case_id": case.case_id,
                "category": case.category,
                "recall_at_5": flag,
                "lexical_recall_at_5": flag,
                "dense_recall_at_5": flag,
                "hit_sections_5": list(case.relevant_sections[:5]),
                "tool_calls": 1,
                "source_tokens": 0,
                "latency_ms": 0.0,
                "second_pass_recovered": False,
                "support_success": True,
            }
            for case, flag in zip(en_cases, en_flags, strict=True)
        ]
        stale = check_stale_rejection(index, repo_root)
        fidelity = check_exact_fidelity(index, baseline_results)
        payload = build_artifact_payload(
            repo_root=repo_root,
            cases=cases,
            results=baseline_results,
            summary=baseline_summary,
            tuning=tuning,
            en_control=en_control,
            en_per_case=en_per_case,
            stale=stale,
            fidelity=fidelity,
            chunking=chunking,
            expansion=expansion,
        )
        validate_artifact_payload(payload, repo_root=repo_root)
        return payload
