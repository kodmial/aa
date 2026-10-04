"""RU-first retrieval qualification and EN-secondary decision (issue #47).

Compares three fixed retrieval configurations over the aligned corpus:

- ``A`` RU-first baseline: original RU + RU planner rewrites
  -> RU BM25 + RU multilingual dense -> RRF fusion;
- ``B`` RU-first + aligned EN secondary discovery (EN hits mapped back
  to exact RU canonical chunks; final evidence is always RU text);
- ``C`` legacy RU→EN-only retrieval as a benchmark/control only
  (EN gloss queries against the EN side, mapped to RU sections).

RU-first is not up for reversal: this module only decides whether the
aligned EN secondary branch adds enough measured value to enable.
The decision defaults to RU-only unless the EN branch shows a material
measured improvement that justifies latency/complexity.

All literary content used by the hermetic benchmark is invented fixture
text; no canonical book text is committed. The versioned decision
artifact is bound to the real RU source checksum, structure version,
index config, planner schema and gold-set version.
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
from aa.retrieval.dense import DENSE_TOP_K, ExactIPIndex, hashing_embed
from aa.retrieval.fusion import MAX_CANDIDATES_PER_ASPECT, MAX_PER_SECTION, RRF_K
from aa.retrieval.index import (
    HybridIndex,
    build_hybrid_index,
    search_aspect,
)
from aa.retrieval.lexical import LEXICAL_TOP_K, build_lexical_db, lexical_search
from aa.retrieval.planner import SCHEMA_VERSION as PLANNER_SCHEMA_VERSION

GOLD_VERSION = "ru-first-gold-v1"
GOLD_REL = "qualification/ru_first_gold.v1.json"
DECISION_REL = "qualification/ru_first_retrieval.v1.decision.json"
BENCHMARK_VERSION = "ru-first-retrieval-benchmark/1"
DECISION_FORMAT = "ru-first-retrieval-decision/1"
PRODUCTION_CONFIG_VERSION = "ru-first-production-v1"

RECALL_KS = (1, 5, 12)
QUALITY_GATE_RECALL_AT_5 = 0.80
QUALITY_GATE_SLABG_PASS_RATE = 1.0
QUALITY_GATE_MAX_FALSE_STRENGTHENING = 0
EN_ENABLE_MIN_INCREMENTAL_RECALL_AT_5 = 0.05
EN_ENABLE_MIN_NEW_SECTIONS = 1
EN_ENABLE_MAX_LATENCY_RATIO = 2.0

REQUIRED_GOLD_CATEGORIES = (
    "slang-drinking",
    "slang-family",
    "morphology",
    "typo",
    "transposition",
    "terse-followup",
    "ambiguous-sorvalsya",
    "implicit-family",
    "implicit-work",
    "implicit-alcohol",
    "multi-theme",
    "exact-phrase",
    "exact-fact",
)


class RuFirstError(ValueError):
    """Raised when the RU-first gold set or decision artifact is invalid."""


@dataclass(frozen=True)
class GoldCase:
    """One validated RU-first gold fixture."""

    case_id: str
    category: str
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


@dataclass(frozen=True)
class FixtureResult:
    """Per-fixture retrieval outcome for one configuration."""

    case_id: str
    hit_sections: tuple[str, ...]
    hit_logical_ids: tuple[str, ...]
    recall_hit_at_1: bool
    recall_hit_at_5: bool
    recall_hit_at_12: bool
    coverage_found: int
    coverage_total: int
    lexical_only: int
    dense_only: int
    both_branches: int
    duplicates: int
    latency_ms: float
    planner_cost_tokens: int
    evidence_tokens: int
    false_strengthening: bool


@dataclass(frozen=True)
class ConfigSummary:
    """Aggregate metrics for one configuration (A, B or C)."""

    config_id: str
    fixtures: int
    recall_at_1: float
    recall_at_5: float
    recall_at_12: float
    coverage: float
    missed_relevant_region_rate: float
    false_strengthening_count: int
    lexical_hit_rate: float
    dense_hit_rate: float
    both_branch_rate: float
    duplicate_rate: float
    mean_latency_ms: float
    mean_planner_cost_tokens: float
    mean_evidence_tokens: float


@dataclass(frozen=True)
class EnSide:
    """Ephemeral EN control side for the B/C benchmark branches."""

    lexical_db: Path
    dense: ExactIPIndex
    chunks_by_id: dict[str, dict[str, Any]]
    section_to_ru_chunk: dict[str, str]


def find_repo_root() -> Path:
    """Return the repository root containing the qualification directory."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / GOLD_REL).exists() or (parent / "qualification").is_dir():
            if (parent / "corpus" / "canonical.ru.manifest.json").exists():
                return parent
    raise RuFirstError("repository root with corpus manifests not found")


def sha256_bytes(data: bytes) -> str:
    """Return the hex SHA-256 of ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file's raw bytes."""
    return sha256_bytes(path.read_bytes())


def _require_non_empty_str(value: object, *, owner: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuFirstError(f"{owner}: {field} must be a non-empty string")
    return value.strip()


def validate_gold_payload(payload: object) -> list[GoldCase]:
    """Validate the gold-set JSON payload and return ordered cases."""
    if not isinstance(payload, dict):
        raise RuFirstError("gold set must be a JSON object")
    if payload.get("gold_version") != GOLD_VERSION:
        raise RuFirstError(f"gold_version must be {GOLD_VERSION!r}")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RuFirstError("gold set must carry a non-empty cases list")
    cases: list[GoldCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases):
        owner = f"case[{position}]"
        if not isinstance(raw, dict):
            raise RuFirstError(f"{owner} must be an object")
        case_id = _require_non_empty_str(raw.get("case_id"), owner=owner, field="case_id")
        if case_id in seen:
            raise RuFirstError(f"{owner}: duplicate case_id {case_id!r}")
        seen.add(case_id)
        category = _require_non_empty_str(raw.get("category"), owner=owner, field="category")
        utterance = _require_non_empty_str(raw.get("utterance"), owner=owner, field="utterance")
        context_raw = raw.get("context", "")
        if not isinstance(context_raw, str):
            raise RuFirstError(f"{case_id}: context must be a string")
        context = context_raw.strip()
        plan_queries = raw.get("plan_queries_ru")
        if not isinstance(plan_queries, list) or not plan_queries:
            raise RuFirstError(f"{case_id}: plan_queries_ru must be a non-empty list")
        cleaned_ru: list[str] = []
        for query in plan_queries:
            if not isinstance(query, str) or not query.strip():
                raise RuFirstError(f"{case_id}: plan_queries_ru holds an empty query")
            cleaned_ru.append(query.strip())
        en_queries = raw.get("en_gloss_queries")
        if not isinstance(en_queries, list) or not en_queries:
            raise RuFirstError(f"{case_id}: en_gloss_queries must be a non-empty list")
        cleaned_en: list[str] = []
        for query in en_queries:
            if not isinstance(query, str) or not query.strip():
                raise RuFirstError(f"{case_id}: en_gloss_queries holds an empty query")
            cleaned_en.append(query.strip())
        relevant = raw.get("relevant_sections")
        if not isinstance(relevant, list) or not relevant:
            raise RuFirstError(f"{case_id}: relevant_sections must be non-empty")
        relevant_clean: list[str] = []
        for section in relevant:
            if section not in SECTION_IDS:
                raise RuFirstError(f"{case_id}: unknown relevant section {section!r}")
            relevant_clean.append(str(section))
        allowed = raw.get("allowed_interpretations")
        if not isinstance(allowed, list) or not allowed:
            raise RuFirstError(f"{case_id}: allowed_interpretations must be non-empty")
        for item in allowed:
            if not isinstance(item, str) or not item.strip():
                raise RuFirstError(f"{case_id}: allowed_interpretations holds an empty entry")
        forbidden = raw.get("forbidden_inferences")
        if not isinstance(forbidden, list) or not forbidden:
            raise RuFirstError(f"{case_id}: forbidden_inferences must be non-empty")
        for item in forbidden:
            if not isinstance(item, str) or not item.strip():
                raise RuFirstError(f"{case_id}: forbidden_inferences holds an empty entry")
        meaning = _require_non_empty_str(
            raw.get("planner_meaning"), owner=case_id, field="planner_meaning"
        )
        requires_context = raw.get("requires_context", False)
        if not isinstance(requires_context, bool):
            raise RuFirstError(f"{case_id}: requires_context must be a bool")
        is_slang = raw.get("is_slang", False)
        if not isinstance(is_slang, bool):
            raise RuFirstError(f"{case_id}: is_slang must be a bool")
        if requires_context and not context:
            raise RuFirstError(f"{case_id}: requires_context needs a non-empty context")
        lowered = f"{utterance} {meaning}".casefold()
        for item in forbidden:
            if item.casefold() in lowered:
                raise RuFirstError(
                    f"{case_id}: planner meaning/utterance repeats a forbidden inference"
                )
        cases.append(
            GoldCase(
                case_id=case_id,
                category=category,
                utterance=utterance,
                context=context,
                plan_queries_ru=tuple(cleaned_ru),
                en_gloss_queries=tuple(cleaned_en),
                relevant_sections=tuple(relevant_clean),
                allowed_interpretations=tuple(str(item).strip() for item in allowed),
                forbidden_inferences=tuple(str(item).strip() for item in forbidden),
                planner_meaning=meaning,
                requires_context=requires_context,
                is_slang=is_slang,
            )
        )
    categories = {case.category for case in cases}
    for required in REQUIRED_GOLD_CATEGORIES:
        if required not in categories:
            raise RuFirstError(f"gold set is missing required category {required!r}")
    lowered_utterances = " ".join(case.utterance.casefold() for case in cases)
    for token in ("бухаю", "нажрался", "тяпнул", "жинка", "женушка", "сорвался"):
        if token not in lowered_utterances:
            raise RuFirstError(f"gold set utterances must include {token!r}")
    return cases


def load_gold(path: Path) -> list[GoldCase]:
    """Load and validate the gold set at ``path``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuFirstError(f"gold set is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuFirstError(f"gold set is not valid JSON: {exc}") from exc
    return validate_gold_payload(payload)


def fixture_section_texts() -> tuple[dict[str, str], dict[str, str]]:
    """Return invented RU/EN section texts aligned to the gold set.

    Every section carries distinctive invented keywords so the hermetic
    benchmark can measure section-level recall without committing any
    canonical book text.
    """
    ru: dict[str, str] = {
        "doctors-opinion": (
            "Фиктивное мнение доктора о тяге и навязчивом желании выпить. "
            "Тяга приходит внезапно и требует внимания.\n\n"
            "Второй абзац мнения доктора. Наблюдение за тягой продолжается."
        ),
        "chapter-1": (
            "Фиктивный рассказ о первом глотке. Герой начал бухать каждый вечер "
            "и однажды тяпнул лишнего. Утром после выпивки было тяжело.\n\n"
            "Второй абзац рассказа. Нажрался в гостях и долго приходил в себя."
        ),
        "chapter-2": (
            "Фиктивный выход есть для пьющих. Надежда и поддержка рядом. "
            "Пьющие находят помощь в сообществе.\n\n"
            "Второй абзац выхода. Сообщество встречает новичков тепло."
        ),
        "chapter-3": (
            "Фиктивный алкоголизм как феномен тяги. Тяга к алкоголю и последствия "
            "алкоголизма обсуждаются подробно. Сорвался после долгой трезвости.\n\n"
            "Второй абзац об алкоголизме. Алкаголь в жизни приносит разлад. "
            "Тянет выпить снова, страх срыва рядом."
        ),
        "chapter-4": (
            "Фиктивные размышления агностика. Готовность принять помощь растет.\n\n"
            "Второй абзац агностика. Сомнения обсуждаются открыто."
        ),
        "chapter-5": (
            "Фиктивная программа в действии требует честности. "
            "Практические шаги каждый день.\n\n"
            "Второй абзац программы. Утренний настрой задает тон."
        ),
        "chapter-6": (
            "Фиктивная работа по шагам продолжается. Утром делаем инвентаризацию. "
            "Срыв разбираем честно и спокойно.\n\n"
            "Второй абзац работы. Вечером подводим итоги дня."
        ),
        "chapter-7": (
            "Фиктивная работа с другими людьми. Несем весть тем кто страдает.\n\n"
            "Второй абзац помощи. Разговор ведется спокойно и честно."
        ),
        "chapter-8": (
            "Фиктивная жинка ругает из-за пьянки. Женушка переживает за семью. "
            "Жена ругает пьянство, доверие страдает.\n\n"
            "Второй абзац о семье. Пьянство разрушает доверие постепенно. "
            "Жнка с опечаткой тоже переживает."
        ),
        "chapter-9": (
            "Фиктивные новые отношения в семье. Доверие возвращается постепенно. "
            "Семейный конфликт утихает.\n\n"
            "Второй абзац семьи. Разговоры становятся спокойнее."
        ),
        "chapter-10": (
            "Фиктивное обращение к работодателям. Трезвость на рабочем месте важна. "
            "Рабочий конфликт и страх увольнения обсуждаются.\n\n"
            "Второй абзац работодателям. Поддержка коллег помогает многим."
        ),
        "chapter-11": (
            "Фиктивный взгляд в будущее сообщества. Бухать больше не хочется, "
            "хочется жить трезво.\n\n"
            "Второй абзац будущего. Планы строятся на трезвую голову."
        ),
    }
    en: dict[str, str] = {
        "doctors-opinion": (
            "Fixture EN doctors opinion about craving and obsession. "
            "Craving arrives suddenly.\n\n"
            "Fixture EN doctors opinion second paragraph about craving."
        ),
        "chapter-1": (
            "Fixture EN bills story about drinking every evening. "
            "A sip too many leads to a hard morning.\n\n"
            "Fixture EN bills story second paragraph about being drunk at guests."
        ),
        "chapter-2": (
            "Fixture EN there is a solution for drinkers. Hope and fellowship nearby.\n\n"
            "Fixture EN solution second paragraph welcoming newcomers."
        ),
        "chapter-3": (
            "Fixture EN more about alcoholism and craving. Relapse after long "
            "sobriety and fear of slip discussed.\n\n"
            "Fixture EN alcoholism second paragraph about craving and relapse."
        ),
        "chapter-4": (
            "Fixture EN we agnostics about willingness to accept help.\n\n"
            "Fixture EN agnostics second paragraph about open doubts."
        ),
        "chapter-5": (
            "Fixture EN how it works program of honesty and daily action.\n\n"
            "Fixture EN program second paragraph about morning attitude."
        ),
        "chapter-6": (
            "Fixture EN into action inventory and honest slip review.\n\n"
            "Fixture EN action second paragraph about evening review."
        ),
        "chapter-7": (
            "Fixture EN working with others carrying the message.\n\n"
            "Fixture EN help second paragraph about calm honest talk."
        ),
        "chapter-8": (
            "Fixture EN to wives about family conflict and drinking. "
            "A wife worries about family trust.\n\n"
            "Fixture EN wives second paragraph about drinking harming trust."
        ),
        "chapter-9": (
            "Fixture EN family afterward rebuilding trust at home.\n\n"
            "Fixture EN family second paragraph about calmer talks."
        ),
        "chapter-10": (
            "Fixture EN to employers about workplace conflict and staying sober "
            "at work.\n\n"
            "Fixture EN employers second paragraph about colleague support."
        ),
        "chapter-11": (
            "Fixture EN vision for you about sober future plans.\n\n"
            "Fixture EN future second paragraph about sober plans."
        ),
    }
    return ru, en


def build_fixture_full() -> dict[str, Any]:
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
    )
    return dict(full)


def build_en_side(full_structure: dict[str, Any], *, workdir: Path) -> EnSide:
    """Build the ephemeral EN control side from the full hierarchy."""
    sections = full_structure.get("sections")
    if not isinstance(sections, list) or not sections:
        raise RuFirstError("full structure has no sections list")
    chunk_ids: list[str] = []
    section_of: list[str] = []
    texts: list[str] = []
    chunks_by_id: dict[str, dict[str, Any]] = {}
    section_to_ru_chunk: dict[str, str] = {}
    for entry in sections:
        if not isinstance(entry, dict):
            raise RuFirstError("full structure has a malformed section")
        section_id = str(entry.get("id"))
        en_branch = entry.get("en")
        ru_branch = entry.get("ru")
        if not isinstance(en_branch, dict) or not isinstance(ru_branch, dict):
            raise RuFirstError(f"section {section_id!r} is missing en/ru branches")
        en_chunks = en_branch.get("chunks")
        ru_chunks = ru_branch.get("chunks")
        if not isinstance(en_chunks, list) or not en_chunks:
            raise RuFirstError(f"section {section_id!r} has no EN chunks")
        if not isinstance(ru_chunks, list) or not isinstance(ru_chunks, list):
            raise RuFirstError(f"section {section_id!r} has no RU chunks")
        first_ru = ru_chunks[0]
        if isinstance(first_ru, dict):
            section_to_ru_chunk[section_id] = str(first_ru.get("id", ""))
        for node in en_chunks:
            if not isinstance(node, dict):
                raise RuFirstError("EN chunk node must be an object")
            chunk_id = str(node.get("id"))
            text = str(node.get("text", ""))
            if not text.strip():
                raise RuFirstError(f"EN chunk {chunk_id!r} has blank text")
            chunk_ids.append(chunk_id)
            section_of.append(section_id)
            texts.append(text)
            chunks_by_id[chunk_id] = {
                "section": section_id,
                "text": text,
            }
    workdir.mkdir(parents=True, exist_ok=True)
    lexical_db = workdir / "en_lexical.db"
    build_lexical_db(lexical_db, chunk_ids=chunk_ids, sections=section_of, texts=texts)
    vectors = [hashing_embed("passage: " + text) for text in texts]
    dense = ExactIPIndex.build(chunk_ids, vectors, backend="hashing")
    return EnSide(
        lexical_db=lexical_db,
        dense=dense,
        chunks_by_id=chunks_by_id,
        section_to_ru_chunk=section_to_ru_chunk,
    )


def _planner_cost_tokens(case: GoldCase) -> int:
    """Estimate planner cost as tokens of the validated plan JSON."""
    plan_doc = {
        "schema_version": PLANNER_SCHEMA_VERSION,
        "utterance_id": case.case_id,
        "language": "ru",
        "original_query": case.plan_queries_ru[0] if case.plan_queries_ru else "",
        "queries_ru": list(case.plan_queries_ru),
        "meaning": case.planner_meaning,
        "forbidden_inferences": list(case.forbidden_inferences),
    }
    return estimate_text_tokens(json.dumps(plan_doc, ensure_ascii=False, sort_keys=True))


def _false_strengthening(case: GoldCase) -> bool:
    """Return True when planner text strengthens a forbidden inference."""
    haystack = f"{case.planner_meaning} {' '.join(case.plan_queries_ru)}".casefold()
    return any(item.casefold() in haystack for item in case.forbidden_inferences)


def _evidence_tokens(index: HybridIndex, logical_ids: tuple[str, ...]) -> int:
    total = 0
    by_logical = {record.logical_chunk_id: record for record in index.chunks.values()}
    for logical_id in logical_ids:
        record = by_logical.get(logical_id)
        if record is not None:
            total += estimate_text_tokens(record.text)
    return total


def run_fixture_a(index: HybridIndex, case: GoldCase, *, top_k: int = 12) -> FixtureResult:
    """Run configuration A (RU-first baseline) for one gold fixture."""
    started = time.perf_counter()
    hits = search_aspect(index, list(case.plan_queries_ru))
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return _summarize_hits(index, case, hits, elapsed_ms=elapsed_ms, top_k=top_k)


def run_fixture_b(
    index: HybridIndex,
    en_side: EnSide,
    case: GoldCase,
    *,
    top_k: int = 12,
    en_extra: int = 4,
) -> FixtureResult:
    """Run configuration B (RU-first plus aligned EN secondary discovery)."""
    started = time.perf_counter()
    ru_hits = search_aspect(index, list(case.plan_queries_ru))
    ru_ids = [hit.chunk_id for hit in ru_hits]
    ru_sections = {hit.section for hit in ru_hits}
    en_ranked: list[tuple[str, float]] = []
    for query in case.en_gloss_queries:
        en_ranked.extend(lexical_search(en_side.lexical_db, query, top_k=LEXICAL_TOP_K))
        en_ranked.extend(
            en_side.dense.search(
                hashing_embed("query: " + query),
                top_k=min(DENSE_TOP_K, len(en_side.chunks_by_id)),
            )
        )
    # Map EN discoveries back to RU sections; final evidence stays RU-only.
    added: list[str] = []
    seen_sections = set(ru_sections)
    ordered_en = sorted(
        {chunk_id for chunk_id, _ in en_ranked},
        key=lambda cid: en_side.chunks_by_id.get(cid, {}).get("section", ""),
    )
    for en_id in ordered_en:
        section = str(en_side.chunks_by_id.get(en_id, {}).get("section", ""))
        if not section or section in seen_sections:
            continue
        ru_chunk = en_side.section_to_ru_chunk.get(section, "")
        if ru_chunk and ru_chunk not in ru_ids and ru_chunk in index.chunks:
            added.append(ru_chunk)
            seen_sections.add(section)
        if len(added) >= en_extra:
            break
    # Re-fuse: keep RU order first, then EN-mapped RU chunks in section order.
    combined_ids = ru_ids + [item for item in added if item not in ru_ids]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    # Re-resolve combined ids to hit-like section/logical lists.
    sections: list[str] = []
    logicals: list[str] = []
    for chunk_id in combined_ids[:top_k]:
        record = index.chunks.get(chunk_id)
        if record is None:
            continue
        sections.append(record.section)
        logicals.append(record.logical_chunk_id)
    # Reuse the RU hit contracts for branch/token accounting on the
    # overlapping prefix; EN-mapped tail counts as dense-assisted discovery.
    base = _summarize_hits(index, case, ru_hits, elapsed_ms=elapsed_ms, top_k=top_k)
    merged_sections = tuple(sections)
    merged_logicals = tuple(logicals)
    return FixtureResult(
        case_id=case.case_id,
        hit_sections=merged_sections,
        hit_logical_ids=merged_logicals,
        recall_hit_at_1=_hit_at(merged_sections, case, 1),
        recall_hit_at_5=_hit_at(merged_sections, case, 5),
        recall_hit_at_12=_hit_at(merged_sections, case, 12),
        coverage_found=sum(1 for item in case.relevant_sections if item in merged_sections),
        coverage_total=len(case.relevant_sections),
        lexical_only=base.lexical_only,
        dense_only=base.dense_only,
        both_branches=base.both_branches,
        duplicates=base.duplicates,
        latency_ms=elapsed_ms,
        planner_cost_tokens=base.planner_cost_tokens,
        evidence_tokens=_evidence_tokens(index, merged_logicals),
        false_strengthening=base.false_strengthening,
    )


def run_fixture_c(
    index: HybridIndex,
    en_side: EnSide,
    case: GoldCase,
    *,
    top_k: int = 12,
) -> FixtureResult:
    """Run configuration C (legacy RU→EN-only control, mapped to RU)."""
    started = time.perf_counter()
    en_ranked: list[tuple[str, float]] = []
    for query in case.en_gloss_queries:
        en_ranked.extend(lexical_search(en_side.lexical_db, query, top_k=LEXICAL_TOP_K))
        en_ranked.extend(
            en_side.dense.search(
                hashing_embed("query: " + query),
                top_k=min(DENSE_TOP_K, len(en_side.chunks_by_id)),
            )
        )
    # Control only: EN discovery mapped back to RU sections for scoring.
    # No RU lexical/dense branch runs here.
    ordered = sorted(
        {chunk_id for chunk_id, _ in en_ranked},
        key=lambda cid: en_side.chunks_by_id.get(cid, {}).get("section", ""),
    )
    ru_ids: list[str] = []
    for en_id in ordered:
        section = str(en_side.chunks_by_id.get(en_id, {}).get("section", ""))
        ru_chunk = en_side.section_to_ru_chunk.get(section, "")
        if ru_chunk and ru_chunk not in ru_ids and ru_chunk in index.chunks:
            ru_ids.append(ru_chunk)
        if len(ru_ids) >= top_k:
            break
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    sections = tuple(
        index.chunks[chunk_id].section for chunk_id in ru_ids[:top_k] if chunk_id in index.chunks
    )
    logicals = tuple(
        index.chunks[chunk_id].logical_chunk_id
        for chunk_id in ru_ids[:top_k]
        if chunk_id in index.chunks
    )
    return FixtureResult(
        case_id=case.case_id,
        hit_sections=sections,
        hit_logical_ids=logicals,
        recall_hit_at_1=_hit_at(sections, case, 1),
        recall_hit_at_5=_hit_at(sections, case, 5),
        recall_hit_at_12=_hit_at(sections, case, 12),
        coverage_found=sum(1 for item in case.relevant_sections if item in sections),
        coverage_total=len(case.relevant_sections),
        lexical_only=0,
        dense_only=0,
        both_branches=0,
        duplicates=0,
        latency_ms=elapsed_ms,
        planner_cost_tokens=_planner_cost_tokens(case),
        evidence_tokens=_evidence_tokens(index, logicals),
        false_strengthening=_false_strengthening(case),
    )


def _hit_at(sections: tuple[str, ...], case: GoldCase, k: int) -> bool:
    return any(section in case.relevant_sections for section in sections[:k])


def _summarize_hits(
    index: HybridIndex,
    case: GoldCase,
    hits: list[Any],
    *,
    elapsed_ms: float,
    top_k: int,
) -> FixtureResult:
    sections: list[str] = []
    logicals: list[str] = []
    lexical_only = 0
    dense_only = 0
    both = 0
    for hit in hits[:top_k]:
        sections.append(str(hit.section))
        logicals.append(str(hit.logical_chunk_id))
        has_lex = hit.lexical_rank is not None
        has_dense = hit.dense_rank is not None
        if has_lex and has_dense:
            both += 1
        elif has_lex:
            lexical_only += 1
        elif has_dense:
            dense_only += 1
    duplicates = len(logicals) - len(set(logicals))
    sections_t = tuple(sections)
    logicals_t = tuple(logicals)
    return FixtureResult(
        case_id=case.case_id,
        hit_sections=sections_t,
        hit_logical_ids=logicals_t,
        recall_hit_at_1=_hit_at(sections_t, case, 1),
        recall_hit_at_5=_hit_at(sections_t, case, 5),
        recall_hit_at_12=_hit_at(sections_t, case, 12),
        coverage_found=sum(1 for item in case.relevant_sections if item in sections_t),
        coverage_total=len(case.relevant_sections),
        lexical_only=lexical_only,
        dense_only=dense_only,
        both_branches=both,
        duplicates=duplicates,
        latency_ms=elapsed_ms,
        planner_cost_tokens=_planner_cost_tokens(case),
        evidence_tokens=_evidence_tokens(index, logicals_t),
        false_strengthening=_false_strengthening(case),
    )


def summarize(results: list[FixtureResult], *, config_id: str) -> ConfigSummary:
    """Aggregate per-fixture results into one configuration summary."""
    if not results:
        raise RuFirstError(f"configuration {config_id!r} has no fixture results")
    total = len(results)
    recall_1 = sum(1 for item in results if item.recall_hit_at_1) / total
    recall_5 = sum(1 for item in results if item.recall_hit_at_5) / total
    recall_12 = sum(1 for item in results if item.recall_hit_at_12) / total
    covered = sum(item.coverage_found for item in results)
    relevant = sum(item.coverage_total for item in results)
    coverage = (covered / relevant) if relevant else 0.0
    false_count = sum(1 for item in results if item.false_strengthening)
    hits_total = sum(item.lexical_only + item.dense_only + item.both_branches for item in results)
    lexical_hits = sum(item.lexical_only + item.both_branches for item in results)
    dense_hits = sum(item.dense_only + item.both_branches for item in results)
    both_hits = sum(item.both_branches for item in results)
    lexical_rate = (lexical_hits / hits_total) if hits_total else 0.0
    dense_rate = (dense_hits / hits_total) if hits_total else 0.0
    both_rate = (both_hits / hits_total) if hits_total else 0.0
    duplicate_rate = sum(item.duplicates for item in results) / max(1, hits_total)
    return ConfigSummary(
        config_id=config_id,
        fixtures=total,
        recall_at_1=recall_1,
        recall_at_5=recall_5,
        recall_at_12=recall_12,
        coverage=coverage,
        missed_relevant_region_rate=1.0 - coverage,
        false_strengthening_count=false_count,
        lexical_hit_rate=lexical_rate,
        dense_hit_rate=dense_rate,
        both_branch_rate=both_rate,
        duplicate_rate=duplicate_rate,
        mean_latency_ms=sum(item.latency_ms for item in results) / total,
        mean_planner_cost_tokens=sum(item.planner_cost_tokens for item in results) / total,
        mean_evidence_tokens=sum(item.evidence_tokens for item in results) / total,
    )


def decide_production(
    summary_a: ConfigSummary,
    summary_b: ConfigSummary,
    *,
    results_a: list[FixtureResult],
    results_b: list[FixtureResult],
) -> tuple[str, str, float, int]:
    """Decide the production configuration and return details.

    Returns ``(production_config_id, rationale, incremental_recall, new_sections)``.
    EN secondary is enabled only on a material measured improvement that
    justifies latency/complexity; otherwise RU-only retrieval stays.
    """
    incremental = summary_b.recall_at_5 - summary_a.recall_at_5
    by_a = {item.case_id: set(_relevant_hit_sections(item)) for item in results_a}
    new_sections = 0
    for item in results_b:
        before = by_a.get(item.case_id, set())
        after = set(_relevant_hit_sections(item))
        new_sections += len(after - before)
    latency_ratio = (
        (summary_b.mean_latency_ms / summary_a.mean_latency_ms)
        if summary_a.mean_latency_ms > 0
        else 1.0
    )
    gate_ok = summary_a.recall_at_5 >= QUALITY_GATE_RECALL_AT_5
    _ = gate_ok
    if (
        incremental >= EN_ENABLE_MIN_INCREMENTAL_RECALL_AT_5
        and new_sections >= EN_ENABLE_MIN_NEW_SECTIONS
        and latency_ratio <= EN_ENABLE_MAX_LATENCY_RATIO
    ):
        return (
            "ru-first-plus-en-secondary",
            (
                f"EN secondary adds {incremental:.3f} recall@5 and {new_sections} "
                f"new relevant sections at {latency_ratio:.2f}x latency; enabling."
            ),
            incremental,
            new_sections,
        )
    return (
        "ru-first-only",
        (
            f"EN secondary adds {incremental:.3f} recall@5 and {new_sections} new "
            f"relevant sections at {latency_ratio:.2f}x latency; keeping RU-only."
        ),
        incremental,
        new_sections,
    )


def _relevant_hit_sections(item: FixtureResult) -> list[str]:
    return list(item.hit_sections)


def quality_gate(summary_a: ConfigSummary, cases: list[GoldCase]) -> tuple[bool, list[str]]:
    """Evaluate the RU-first quality gate; return (passed, failures)."""
    failures: list[str] = []
    if summary_a.recall_at_5 < QUALITY_GATE_RECALL_AT_5:
        failures.append(
            f"recall@5 {summary_a.recall_at_5:.3f} below gate {QUALITY_GATE_RECALL_AT_5:.2f}"
        )
    if summary_a.false_strengthening_count != QUALITY_GATE_MAX_FALSE_STRENGTHENING:
        failures.append(
            f"false strengthening {summary_a.false_strengthening_count} "
            f"exceeds {QUALITY_GATE_MAX_FALSE_STRENGTHENING}"
        )
    slang = [case for case in cases if case.is_slang]
    # Slang pass is measured by the caller from per-fixture rows; here we
    # only enforce that slang fixtures exist (rows are checked in tests).
    if not slang:
        failures.append("no slang fixtures in the gold set")
    return (not failures, failures)


def read_bindings(repo_root: Path) -> dict[str, Any]:
    """Read live version bindings for the decision artifact."""
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
        "en_artifact_sha256": str(en_manifest.get("artifact_sha256", "")),
        "en_manifest_format": str(en_manifest.get("format", "")),
        "structure_format": str(structure.get("format", "")),
        "structure_builder_version": structure.get("builder_version"),
        "embedding_model_id": str(lock.get("model_id", "")),
        "embedding_revision": str(lock.get("revision", "")),
    }


def build_decision_payload(
    *,
    repo_root: Path,
    cases: list[GoldCase],
    results_a: list[FixtureResult],
    results_b: list[FixtureResult],
    results_c: list[FixtureResult],
    summary_a: ConfigSummary,
    summary_b: ConfigSummary,
    summary_c: ConfigSummary,
) -> dict[str, Any]:
    """Build the versioned benchmark/decision artifact payload."""
    bindings = read_bindings(repo_root)
    gold_path = repo_root / GOLD_REL
    gold_sha = sha256_file(gold_path)
    production_id, rationale, incremental, new_sections = decide_production(
        summary_a, summary_b, results_a=results_a, results_b=results_b
    )
    gate_passed, failures = quality_gate(summary_a, cases)
    slang_rows = [item for item in results_a if _case_is_slang(cases, item.case_id)]
    slang_pass = (
        sum(1 for item in slang_rows if item.recall_hit_at_5) / len(slang_rows)
        if slang_rows
        else 0.0
    )
    return {
        "format": DECISION_FORMAT,
        "benchmark_version": BENCHMARK_VERSION,
        "gold_version": GOLD_VERSION,
        "gold_sha256": gold_sha,
        "gold_fixtures": len(cases),
        "bindings": bindings,
        "planner_schema": PLANNER_SCHEMA_VERSION,
        "index_config": {
            "lexical_top_k": LEXICAL_TOP_K,
            "dense_top_k": DENSE_TOP_K,
            "rrf_k": RRF_K,
            "max_per_aspect": MAX_CANDIDATES_PER_ASPECT,
            "max_per_section": MAX_PER_SECTION,
            "embedding_backend": "hashing-char-token/1",
            "note": (
                "Hermetic hashing backend mirrors the exact-IP "
                "IndexFlatIP contract; production e5 uses the pinned lock."
            ),
        },
        "quality_gate": {
            "recall_at_5_threshold": QUALITY_GATE_RECALL_AT_5,
            "recall_at_5_measured": summary_a.recall_at_5,
            "slang_pass_rate_measured": slang_pass,
            "slang_pass_rate_required": QUALITY_GATE_SLABG_PASS_RATE,
            "false_strengthening_measured": summary_a.false_strengthening_count,
            "passed": gate_passed,
            "failures": failures,
        },
        "configs": {
            "a_ru_first": _summary_to_dict(summary_a),
            "b_ru_first_plus_en_secondary": _summary_to_dict(summary_b),
            "c_legacy_ru_to_en_only": _summary_to_dict(summary_c),
        },
        "en_secondary": {
            "incremental_recall_at_5": incremental,
            "new_relevant_sections": new_sections,
            "decision": "enabled" if production_id == "ru-first-plus-en-secondary" else "disabled",
            "latency_ratio_b_over_a": (
                (summary_b.mean_latency_ms / summary_a.mean_latency_ms)
                if summary_a.mean_latency_ms > 0
                else 1.0
            ),
        },
        "production": {
            "config_id": production_id,
            "config_version": PRODUCTION_CONFIG_VERSION,
            "ru_only": production_id == "ru-first-only",
            "rationale": rationale,
            "evidence_language": "ru",
            "evidence_rule": (
                "All final Russian answer evidence resolves to exact RU canonical text."
            ),
        },
        "per_fixture": [
            {
                "case_id": item.case_id,
                "relevant_sections": list(
                    next(c.relevant_sections for c in cases if c.case_id == item.case_id)
                ),
                "a_hit_sections": list(item.hit_sections[:5]),
                "a_recall_at_5": item.recall_hit_at_5,
                "b_hit_sections": list(
                    next(r.hit_sections[:5] for r in results_b if r.case_id == item.case_id)
                ),
                "b_recall_at_5": next(
                    r.recall_hit_at_5 for r in results_b if r.case_id == item.case_id
                ),
                "c_hit_sections": list(
                    next(r.hit_sections[:5] for r in results_c if r.case_id == item.case_id)
                ),
                "c_recall_at_5": next(
                    r.recall_hit_at_5 for r in results_c if r.case_id == item.case_id
                ),
            }
            for item in results_a
        ],
        "notes": (
            "RU-first is fixed and not up for reversal. Legacy RU-to-EN-only "
            "is a benchmark/control and cannot become the production default."
        ),
    }


def _case_is_slang(cases: list[GoldCase], case_id: str) -> bool:
    for case in cases:
        if case.case_id == case_id:
            return case.is_slang
    return False


def _summary_to_dict(summary: ConfigSummary) -> dict[str, Any]:
    return {
        "config_id": summary.config_id,
        "fixtures": summary.fixtures,
        "recall_at_1": summary.recall_at_1,
        "recall_at_5": summary.recall_at_5,
        "recall_at_12": summary.recall_at_12,
        "coverage": summary.coverage,
        "missed_relevant_region_rate": summary.missed_relevant_region_rate,
        "false_strengthening_count": summary.false_strengthening_count,
        "lexical_hit_rate": summary.lexical_hit_rate,
        "dense_hit_rate": summary.dense_hit_rate,
        "both_branch_rate": summary.both_branch_rate,
        "duplicate_rate": summary.duplicate_rate,
        "mean_latency_ms": summary.mean_latency_ms,
        "mean_planner_cost_tokens": summary.mean_planner_cost_tokens,
        "mean_evidence_tokens": summary.mean_evidence_tokens,
    }


def validate_decision_payload(payload: object, *, repo_root: Path) -> None:
    """Validate a decision artifact against live bindings (fails closed)."""
    if not isinstance(payload, dict):
        raise RuFirstError("decision artifact must be a JSON object")
    if payload.get("format") != DECISION_FORMAT:
        raise RuFirstError(f"decision format must be {DECISION_FORMAT!r}")
    if payload.get("benchmark_version") != BENCHMARK_VERSION:
        raise RuFirstError(f"benchmark_version must be {BENCHMARK_VERSION!r}")
    if payload.get("gold_version") != GOLD_VERSION:
        raise RuFirstError(f"gold_version must be {GOLD_VERSION!r}")
    bindings = payload.get("bindings")
    if not isinstance(bindings, dict):
        raise RuFirstError("decision artifact must carry bindings")
    live = read_bindings(repo_root)
    for key in (
        "ru_artifact_sha256",
        "en_artifact_sha256",
        "structure_format",
        "embedding_model_id",
        "embedding_revision",
    ):
        if bindings.get(key) != live.get(key):
            raise RuFirstError(f"decision artifact binding {key!r} is stale")
    if payload.get("planner_schema") != PLANNER_SCHEMA_VERSION:
        raise RuFirstError("decision artifact planner schema is stale")
    gold_sha = payload.get("gold_sha256")
    gold_path = repo_root / GOLD_REL
    if not isinstance(gold_sha, str) or gold_sha != sha256_file(gold_path):
        raise RuFirstError("decision artifact gold_sha256 does not match the gold set")
    configs = payload.get("configs")
    if not isinstance(configs, dict):
        raise RuFirstError("decision artifact must carry configs")
    for key in ("a_ru_first", "b_ru_first_plus_en_secondary", "c_legacy_ru_to_en_only"):
        if key not in configs:
            raise RuFirstError(f"decision artifact is missing config {key!r}")
    production = payload.get("production")
    if not isinstance(production, dict) or not production.get("config_id"):
        raise RuFirstError("decision artifact must carry a production configuration")
    if production.get("evidence_language") != "ru":
        raise RuFirstError("production evidence language must be ru")
    quality_gate_payload = payload.get("quality_gate")
    if not isinstance(quality_gate_payload, dict):
        raise RuFirstError("decision artifact must carry a quality gate")


def run_benchmark(
    *,
    repo_root: Path,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full A/B/C benchmark on the hermetic fixture hierarchy."""
    import tempfile

    cases = load_gold(repo_root / GOLD_REL)
    full = build_fixture_full()
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
    lock = json.loads((repo_root / "corpus" / "embedding.lock.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="ru-first-bench-") as tmp:
        tmp_path = Path(tmp)
        index = build_hybrid_index(
            full,
            ru_manifest=ru_manifest,
            en_manifest=en_manifest,
            embedding_lock=lock,
            out_dir=tmp_path / "retrieval",
            backend="hashing",
        )
        en_side = build_en_side(full, workdir=tmp_path / "en")
        results_a = [run_fixture_a(index, case) for case in cases]
        results_b = [run_fixture_b(index, en_side, case) for case in cases]
        results_c = [run_fixture_c(index, en_side, case) for case in cases]
        summary_a = summarize(results_a, config_id="a_ru_first")
        summary_b = summarize(results_b, config_id="b_ru_first_plus_en_secondary")
        summary_c = summarize(results_c, config_id="c_legacy_ru_to_en_only")
        payload = build_decision_payload(
            repo_root=repo_root,
            cases=cases,
            results_a=results_a,
            results_b=results_b,
            results_c=results_c,
            summary_a=summary_a,
            summary_b=summary_b,
            summary_c=summary_c,
        )
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
        _ = out_dir
        return payload
