"""Deterministic emergency/medical-safety classifier (Russian + English).

This module owns the pre-model / pre-answer emergency decision described in
GitHub issue #21. It is intentionally independent of the AA retrieval
pipeline and of any LLM:

- pure functions over the input text (no network, no external medical
  service, no randomness);
- small transport-independent API so Telegram integration can call it
  before any OpenCode work is scheduled;
- privacy-safe by construction: classification returns categories and
  language only, never echoes the input.

Design notes (kept deterministic and testable):

- Text is normalized (lowercase, ``ё`` -> ``е``) and split into sentences.
  Every sentence is evaluated independently so that one historical sentence
  (``I had a seizure five years ago``) does not mask a current one
  (``right now I am convulsing``) and vice versa.
- Strong literal patterns trigger the emergency route unless they are
  negated (``no seizures``, ``нет суицидальных мыслей``), historical
  without current-acute markers, resolved (``I am fine now``), or part of
  an informational / AA literature question without acute markers and
  without strong first-person distress.
- Weak patterns (bare nouns, colloquialisms, metaphor-prone words such as
  ``suicide``, ``overdose``, ``collapsed`` or ``белочка``) additionally
  require an acute marker, an alcohol context, or strong first-person
  distress in the same sentence.
- Ordinary AA discussion (past drinking, fear/resentment inventory,
  relapse, steps, the Big Book) contains none of the strong patterns and
  therefore stays on the normal path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class EmergencyCategory(Enum):
    """Acute situations owned by the deterministic safety layer."""

    SEVERE_WITHDRAWAL = "severe-withdrawal"
    SEIZURE = "seizure"
    HALLUCINATION_DELIRIUM = "hallucination-delirium"
    POISONING_OVERDOSE = "poisoning-overdose"
    UNCONSCIOUSNESS = "unconsciousness"
    SELF_HARM_SUICIDE = "self-harm-suicide"
    LIFE_THREATENING_OTHER = "life-threatening-other"


@dataclass(frozen=True)
class EmergencyClassification:
    """Deterministic routing verdict for one message."""

    is_emergency: bool
    categories: tuple[EmergencyCategory, ...] = ()
    language: str = "en"
    matched: tuple[str, ...] = ()


def detect_language(text: str) -> str:
    """Detect ``ru`` vs ``en`` with a Cyrillic-presence heuristic."""
    for char in text:
        if "\u0400" <= char <= "\u04ff":
            return "ru"
    return "en"


def _normalize(text: str) -> str:
    return text.lower().replace("ё", "е")


_SENTENCE_SPLIT = re.compile(r"[.!?…\n;]+")

# ---------------------------------------------------------------------------
# Context markers
# ---------------------------------------------------------------------------

_PAST_MARKERS = (
    r"years?\s+ago",
    r"months?\s+ago",
    r"weeks?\s+ago",
    r"days?\s+ago",
    r"long\s+ago",
    r"in\s+the\s+past",
    r"last\s+(year|month|summer|winter)",
    r"when\s+i\s+was",
    r"\bused\s+to\b",
    r"history\s+of",
    r"back\s+then",
    r"previously",
    r"formerly",
    r"\bin\s+childhood\b",
    r"recovered\s+from",
    r"год\s+назад",
    r"года\s+назад",
    r"лет\s+назад",
    r"месяц(?:а|ев)?\s+назад",
    r"недел\w*\s+назад",
    r"давно",
    r"в\s+прошлом",
    r"раньше",
    r"когда-то",
    r"когда\s+я\s+(?:пил|пила|был|была)",
    r"в\s+молодости",
    r"в\s+детстве",
    r"много\s+лет\s+назад",
    r"прошлый\s+год",
    r"в\s+прошлый\s+раз",
    r"тогда",
)

_ACUTE_MARKERS = (
    r"right\s+now",
    r"\bnow\b",
    r"currently",
    r"at\s+the\s+moment",
    r"suddenly",
    r"just\s+now",
    r"just\s+happened",
    r"just\s+had",
    r"minutes?\s+ago",
    r"an?\s+hour\s+ago",
    r"hours?\s+ago",
    r"ongoing",
    r"still\s+happening",
    r"\btoday\b",
    r"\btonight\b",
    r"this\s+morning",
    r"this\s+evening",
    r"не\s+приходит\s+в\s+себя",
    r"прямо\s+сейчас",
    r"\bсейчас\b",
    r"\bсегодня\b",
    r"только\s+что",
    r"\bвдруг\b",
    r"внезапно",
    r"минут\w*\s+назад",
    r"час\s+назад",
    r"часа\s+назад",
    r"продолжается",
    r"до\s+сих\s+пор",
    r"в\s+данный\s+момент",
    r"сегодня\s+(?:утром|ночью|вечером)",
)

_RESOLVED_MARKERS = (
    r"i'?m\s+fine\s+now",
    r"fine\s+now",
    r"all\s+better",
    r"feeling\s+better",
    r"it\s+passed",
    r"сейчас\s+все\s+хорошо",
    r"все\s+хорошо",
    r"все\s+в\s+порядке",
    r"все\s+прошло",
    r"мне\s+(?:уже\s+)?лучше",
    r"чувствую\s+себя\s+хорошо",
)

_NEGATION_MARKERS_EN = (
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bwithout\b",
    r"\bnone\b",
    r"\bneither\b",
    r"\bhasn'?t\b",
    r"\bhaven'?t\b",
    r"\bhadn'?t\b",
    r"\bdon'?t\b",
    r"\bdoesn'?t\b",
    r"\bdidn'?t\b",
    r"\bisn'?t\b",
    r"\baren'?t\b",
    r"\bwasn'?t\b",
    r"\bweren'?t\b",
    r"\bdeny\b",
    r"\bdenies\b",
    r"\bdenied\b",
)

_NEGATION_MARKERS_RU = (
    r"\bне\b",
    r"\bнет\b",
    r"\bникаких\b",
    r"\bникогда\b",
    r"\bбез\b",
    r"отрицаю",
    r"\bни\b",
)

# Informational / AA-literature question frames. When a sentence carries one
# of these frames and shows no acute marker and no strong first-person
# distress, strong patterns inside it are treated as a topic question, not
# an acute event (bounds false positives on literature discussion).
_INFORMATIONAL_FRAMES = (
    r"what\s+does",
    r"what\s+is",
    r"what\s+are",
    r"what\s+happens\s+if",
    r"what\s+if\s+i",
    r"what\s+should\s+i\s+do\s+about",
    r"tell\s+me\s+about",
    r"tell\s+us\s+about",
    r"\bexplain\b",
    r"according\s+to",
    r"big\s+book",
    r"doctor'?s\s+opinion",
    r"\bstep\s+\d",
    r"fear\s+inventory",
    r"resentment",
    r"how\s+does\s+aa",
    r"how\s+to\s+treat",
    r"how\s+is\s+\w+\s+treated",
    r"i\s+(?:read|heard|learnt|learned)\s+about",
    r"medicat\w*",
    r"\bsafe\s+with\b",
    r"drug\s+interaction",
    r"\bwe\s+discussed\b",
    r"\btalked\s+about\b",
    r"\bdiscussed\b",
    r"что\s+говорит\s+книга",
    r"что\s+такое",
    r"расскажи(?:те)?(?:\s+мне)?\s+о(?:б)?",
    r"объясни",
    r"большая\s+книга",
    r"мнение\s+доктора",
    r"\bшаг\s+\d",
    r"четвертый\s+шаг",
    r"инвентаризац\w*\s+страха",
    r"обид\w*",
    r"что\s+будет\s+если",
    r"а\s+если",
    r"а\s+что\s+если",
    r"что\s+делать\s+с",
    r"что\s+мне\s+делать\s+с",
    r"как\s+лечить",
    r"читал\w*(?:\s+я)?\s+о(?:б)?",
    r"слышал\w*\s+о(?:б)?",
    r"узнал\w*\s+о(?:б)?",
    r"лекарств\w*",
    r"медикамент\w*",
    r"совместим\w*\s+с",
    r"обсуждали",
    r"говорили\s+о(?:б)?",
    r"разбирали",
)

# Strong first-person distress signals. Bare possessives (``my``) and bare
# ``me`` are intentionally excluded: they appear inside informational frames
# (``tell me about ...``) which are stripped before this check runs.
_STRONG_DISTRESS = (
    r"\bi\s+am\b",
    r"\bi'?m\b",
    r"\bi\s+have\b",
    r"\bi'?ve\b",
    r"\bi\s+feel\b",
    r"\bi\s+can'?t\b",
    r"\bi\s+had\b",
    r"\bi\s+just\b",
    r"\bhelp\s+me\b",
    r"\bу\s+меня\b",
    r"\bмне\b",
    r"\bменя\b",
    r"помогите",
    r"помоги",
    r"\bя\s+(?:сейчас|только\s+что|чувствую|вижу|вижу|слышу|хочу|боюсь|"
    r"не\s+могу|потерял|потеряла|упал|упала|выпил|выпила|принял|приняла)\b",
)

_PAST_RE = re.compile("|".join(f"(?:{item})" for item in _PAST_MARKERS))
_ACUTE_RE = re.compile("|".join(f"(?:{item})" for item in _ACUTE_MARKERS))
_RESOLVED_RE = re.compile("|".join(f"(?:{item})" for item in _RESOLVED_MARKERS))
_NEGATION_RE = re.compile(
    "|".join(f"(?:{item})" for item in (*_NEGATION_MARKERS_EN, *_NEGATION_MARKERS_RU))
)
_INFORMATIONAL_RE = re.compile("|".join(f"(?:{item})" for item in _INFORMATIONAL_FRAMES))
_STRONG_DISTRESS_RE = re.compile("|".join(f"(?:{item})" for item in _STRONG_DISTRESS))


@dataclass(frozen=True)
class _Rule:
    category: EmergencyCategory
    pattern: re.Pattern[str]
    # Human-readable tag used in classification metadata (never user text).
    tag: str
    # When True the pattern is weak on its own and additionally requires an
    # acute marker, an alcohol/withdrawal context, or strong first-person
    # distress in the same sentence.
    needs_context: bool = False


def _compile(items: tuple[str, ...]) -> re.Pattern[str]:
    return re.compile("|".join(f"(?:{item})" for item in items))


_ALCOHOL_CONTEXT_RE = _compile(
    (
        r"alcohol",
        r"drink(?:ing|s)?",
        r"drunk",
        r"binge",
        r"withdrawal",
        r"abstinen\w*",
        r"quit\s+drinking",
        r"stopped\s+drinking",
        r"stop\s+drinking",
        r"алкогол\w*",
        r"запой\w*",
        r"запил\w*",
        r"выпив\w*",
        r"похмел\w*",
        r"отмен\w*",
        r"абстин\w*",
        r"бросить\s+пить",
        r"бросил\w*\s+пить",
        r"пить",
    )
)

_RULES: tuple[_Rule, ...] = (
    # -- Severe alcohol withdrawal ----------------------------------------
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile((r"delirium\s+tremens",)),
        tag="en:delirium-tremens",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile((r"alcohol\s+withdrawal", r"withdrawal\s+from\s+alcohol")),
        tag="en:alcohol-withdrawal",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile((r"withdrawal\s+seizures?",)),
        tag="en:withdrawal-seizure",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile(
            (
                r"severe\s+(?:alcohol\s+)?withdrawal",
                r"serious\s+(?:alcohol\s+)?withdrawal",
                r"strong\s+(?:alcohol\s+)?withdrawal",
            )
        ),
        tag="en:severe-withdrawal",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile(
            (
                r"stopped\s+drinking\s+and\b.{0,60}?"
                r"(?:shak\w*|trembl\w*|tremor|sweat\w*|vomit\w*|hallucinat\w*|"
                r"seeing\s+things|hearing\s+voices|confus\w*|seizures?|convuls\w*)",
                r"quit\s+drinking\s+and\b.{0,60}?"
                r"(?:shak\w*|trembl\w*|tremor|sweat\w*|vomit\w*|hallucinat\w*|"
                r"seeing\s+things|hearing\s+voices|confus\w*|seizures?|convuls\w*)",
            )
        ),
        tag="en:withdrawal-after-quit",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile(
            (
                r"белая\s+горячка",
                r"алкогольный\s+делирий",
                r"алкогольный\s+психоз",
                r"абстинентный\s+синдром",
                r"абстиненция",
                r"синдром\s+отмены",
                r"тяжел\w*\s+абстин\w*",
            )
        ),
        tag="ru:withdrawal-syndrome",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile(
            (
                r"тряс\w*\s+после\s+(?:запо\w*|отказа|отмены)",
                r"тряс\w*.{0,40}?после\s+(?:запо\w*|отказа|отмены)",
                r"после\s+запо\w*.{0,40}?тряс\w*",
                r"трясет\s+после\s+отказа",
                r"колбасит\s+после\s+(?:отмены|запо\w*)",
                r"судороги\s+после\s+отказа\s+от\s+алкоголя",
            )
        ),
        tag="ru:shaking-after-binge",
    ),
    _Rule(
        EmergencyCategory.SEVERE_WITHDRAWAL,
        _compile((r"белочек|белочка|белочку",)),
        tag="ru:delirium-colloquial",
        needs_context=True,
    ),
    # -- Seizure ------------------------------------------------------------
    _Rule(
        EmergencyCategory.SEIZURE,
        _compile(
            (
                r"having\s+a\s+seizure",
                r"i\s+am\s+convulsing",
                r"body\s+is\s+convulsing",
                r"shaking\s+uncontrollably",
                r"i\s+(?:just\s+)?had\s+a\s+seizure",
            )
        ),
        tag="en:seizure-acute",
    ),
    _Rule(
        EmergencyCategory.SEIZURE,
        _compile(
            (
                r"seizures?",
                r"convulsions?",
                r"convulsing",
                r"epileptic\s+(?:fit|seizure)",
            )
        ),
        tag="en:seizure",
        needs_context=True,
    ),
    _Rule(
        EmergencyCategory.SEIZURE,
        _compile(
            (
                r"припад\w+",
                r"судорог\w*",
                r"конвульси\w*",
                r"эпилептический\s+при(?:падок|ступ)",
                r"эпилепси\w*\s+приступ",
                r"бьюсь\s+в\s+судорогах",
                r"начались\s+судороги",
                r"трясет\s+все\s+тело\s+и\s+не\s+могу\s+остановиться",
                r"у\s+меня\s+судороги",
                r"меня\s+трясет\s+и\s+я\s+не\s+могу\s+остановиться",
            )
        ),
        tag="ru:seizure",
    ),
    # -- Hallucination / delirium-like ---------------------------------------
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile((r"hallucinat\w*", r"\bdelirium\b", r"\bdelirious\b")),
        tag="en:hallucination",
    ),
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile((r"hearing\s+voices", r"voices\s+telling\s+me")),
        tag="en:hearing-voices",
    ),
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile(
            (
                r"seeing\s+things\s+that\s+(?:are\s+not|aren'?t)\s+(?:there|real)",
                r"seeing\s+things\s+no\s+one\s+else\s+sees",
                r"visions?\s+that\s+(?:are\s+not|aren'?t)\s+real",
            )
        ),
        tag="en:seeing-things-qualified",
    ),
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile((r"seeing\s+things", r"hearing\s+things")),
        tag="en:sensory-disturbance-bare",
        needs_context=True,
    ),
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile(
            (
                r"галлюцинац\w*",
                r"слышу\s+голоса",
                r"слышутся\s+голоса",
                r"вижу\s+то,\s*чего\s+нет",
                r"вижу\s+то\s+чего\s+нет",
                r"мерещатся",
                r"мерещится",
            )
        ),
        tag="ru:hallucination",
    ),
    _Rule(
        EmergencyCategory.HALLUCINATION_DELIRIUM,
        _compile((r"бред\s+и\s+галлюцинац\w*", r"спутанность\s+сознания")),
        tag="ru:delirium-phrase",
    ),
    # -- Poisoning / overdose -------------------------------------------------
    _Rule(
        EmergencyCategory.POISONING_OVERDOSE,
        _compile((r"overdos(?:e|ed|ing)",)),
        tag="en:overdose",
    ),
    _Rule(
        EmergencyCategory.POISONING_OVERDOSE,
        _compile(
            (
                r"took\s+too\s+many\s+pills",
                r"too\s+many\s+(?:pills|tablets)",
                r"swallowed\s+(?:a\s+handful\s+of\s+)?pills",
                r"handful\s+of\s+pills",
                r"drank\s+poison",
                r"drank\s+methanol",
                r"methyl\s+alcohol",
                r"surrogate\s+alcohol",
                r"drank\s+antifreeze",
            )
        ),
        tag="en:poisoning-phrase",
    ),
    _Rule(
        EmergencyCategory.POISONING_OVERDOSE,
        _compile((r"poison(?:ed|ing)?\s+by\s+(?:alcohol|chemicals|methanol|pills)",)),
        tag="en:poisoned-by",
    ),
    _Rule(
        EmergencyCategory.POISONING_OVERDOSE,
        _compile(
            (
                r"передозировка",
                r"передоз",
                r"отравление",
                r"отравил\w+",
                r"наглотал\w*\s+таблеток",
                r"выпил\w*\s+много\s+таблеток",
                r"слишком\s+много\s+таблеток",
                r"горсть\s+таблеток",
                r"метанол\w*",
                r"метиловый\s+спирт",
                r"суррогат\w*",
                r"незамерзайк\w*",
                r"выпил\s+яд",
                r"выпила\s+яд",
                r"выпил\s+отраву",
                r"тосол",
            )
        ),
        tag="ru:poisoning",
    ),
    # -- Loss of consciousness ------------------------------------------------
    _Rule(
        EmergencyCategory.UNCONSCIOUSNESS,
        _compile(
            (
                r"\bunconscious\b",
                r"loss\s+of\s+consciousness",
                r"passed\s+out",
                r"blacked\s+out",
                r"i\s+(?:just\s+)?(?:collapsed|fainted)",
                r"won'?t\s+wake\s+up",
                r"can'?t\s+wake\s+(?:him|her|them|me|my)",
                r"not\s+responding",
                r"collapsed\s+and\s+(?:not|can'?t|won'?t)",
                r"fainted\s+and\s+(?:not|can'?t|won'?t|still)",
                r"stopped\s+responding",
            )
        ),
        tag="en:unconsciousness",
    ),
    _Rule(
        EmergencyCategory.UNCONSCIOUSNESS,
        _compile((r"\bfainted\b", r"\bcollapsed\b", r"\bblackout\b")),
        tag="en:faint-collapse-bare",
        needs_context=True,
    ),
    _Rule(
        EmergencyCategory.UNCONSCIOUSNESS,
        _compile(
            (
                r"без\s+сознания",
                r"потерял\s+сознание",
                r"потеряла\s+сознание",
                r"потеря\s+сознания",
                r"потеряли\s+сознание",
                r"упал\s+в\s+обморок",
                r"упала\s+в\s+обморок",
                r"не\s+приходит\s+в\s+себя",
                r"не\s+могу\s+очнуться",
                r"не\s+может\s+очнуться",
                r"отключился\s+и\s+не\s+реагирует",
                r"не\s+реагирует",
                r"не\s+просыпается",
                r"не\s+дышит",
            )
        ),
        tag="ru:unconsciousness",
    ),
    _Rule(
        EmergencyCategory.UNCONSCIOUSNESS,
        _compile((r"\bобморок\b",)),
        tag="ru:faint-bare",
        needs_context=True,
    ),
    # -- Self-harm / suicide ---------------------------------------------------
    _Rule(
        EmergencyCategory.SELF_HARM_SUICIDE,
        _compile(
            (
                r"want\s+to\s+die",
                r"wanna\s+die",
                r"feel\s+like\s+dying",
                r"don'?t\s+want\s+to\s+live",
                r"want\s+to\s+kill\s+myself",
                r"going\s+to\s+kill\s+myself",
                r"plan\s+to\s+kill\s+myself",
                r"kill\s+myself",
                r"end\s+my\s+life",
                r"take\s+my\s+own\s+life",
                r"end\s+it\s+all",
                r"\bsuicidal\b",
                r"thoughts\s+of\s+suicide",
                r"thinking\s+about\s+suicide",
                r"hurt\s+myself",
                r"harm\s+myself",
                r"cutting\s+myself",
                r"cut\s+myself",
                r"hang\s+myself",
            )
        ),
        tag="en:self-harm",
    ),
    _Rule(
        EmergencyCategory.SELF_HARM_SUICIDE,
        _compile((r"\bsuicide\b", r"\bself-?harm\b")),
        tag="en:suicide-bare",
        needs_context=True,
    ),
    _Rule(
        EmergencyCategory.SELF_HARM_SUICIDE,
        _compile(
            (
                r"хочу\s+умереть",
                r"хочется\s+умереть",
                r"не\s+хочу\s+жить",
                r"незачем\s+жить",
                r"покончить\s+с\s+собой",
                r"покончить\s+жизнь",
                r"самоубийство",
                r"суицид\w*",
                r"повеситься",
                r"повешусь",
                r"вскрыть\s+вены",
                r"режу\s+себя",
                r"порезал\w*\s+себя",
                r"нанесу\s+себе\s+вред",
                r"причиню\s+себе\s+вред",
                r"убью\s+себя",
                r"мысли\s+о\s+суициде",
                r"думаю\s+о\s+суициде",
                r"думаю\s+о\s+смерти",
                r"лучше\s+умереть",
            )
        ),
        tag="ru:self-harm",
    ),
    # -- Other life-threatening ------------------------------------------------
    _Rule(
        EmergencyCategory.LIFE_THREATENING_OTHER,
        _compile(
            (
                r"can'?t\s+breathe",
                r"cann?ot\s+breathe",
                r"difficulty\s+breathing",
                r"trouble\s+breathing",
                r"\bchoking\b",
                r"severe\s+shortness\s+of\s+breath",
                r"chest\s+pain",
                r"chest\s+pressure",
                r"chest\s+tightness",
                r"heart\s+attack",
                r"\bstroke\b",
                r"face\s+is\s+drooping",
                r"slurred\s+speech",
                r"bleeding\s+heavily",
                r"severe\s+bleeding",
                r"blood\s+won'?t\s+stop",
                r"coughing\s+up\s+blood",
                r"stopped\s+breathing",
                r"\bno\s+pulse\b",
                r"throat\s+is\s+closing",
                r"\banaphylaxis\b",
            )
        ),
        tag="en:life-threatening",
    ),
    _Rule(
        EmergencyCategory.LIFE_THREATENING_OTHER,
        _compile(
            (
                r"не\s+могу\s+дышать",
                r"нечем\s+дышать",
                r"задыхаюсь",
                r"задыхается",
                r"удушье",
                r"трудно\s+дышать",
                r"тяжело\s+дышать",
                r"боль\s+в\s+груди",
                r"давление\s+в\s+груди",
                r"сжатие\s+в\s+груди",
                r"сердечный\s+приступ",
                r"инфаркт",
                r"инсульт",
                r"перекосило\s+лицо",
                r"невнятная\s+речь",
                r"онемела\s+рука",
                r"сильное\s+кровотечение",
                r"кровь\s+не\s+останавливается",
                r"кашляю\s+кровью",
                r"нет\s+пульса",
                r"горло\s+отекает",
                r"отек\s+квинке",
                r"анафилаксия",
            )
        ),
        tag="ru:life-threatening",
    ),
)


def _sentence_has_negation_before(sentence: str, match_start: int) -> bool:
    window = sentence[max(0, match_start - 48) : match_start]
    return _NEGATION_RE.search(window) is not None


def _strip_informational(sentence: str) -> str:
    """Remove informational-frame substrings before distress detection.

    This keeps phrases such as ``tell me about`` from contributing their
    ``me`` to the first-person distress check.
    """
    return _INFORMATIONAL_RE.sub(" ", sentence)


def _strip_resolved(sentence: str) -> str:
    """Remove resolved-state phrases before acute-marker detection.

    This keeps the ``now`` in ``fine now`` / ``сейчас все хорошо`` from
    counting as a current-acute signal for an otherwise historical event.
    """
    return _RESOLVED_RE.sub(" ", sentence)


def _classify_sentence(sentence: str) -> set[tuple[EmergencyCategory, str]]:
    """Classify one normalized sentence; returns {(category, tag)}."""
    found: set[tuple[EmergencyCategory, str]] = set()
    if not sentence.strip():
        return found
    historical = _PAST_RE.search(sentence) is not None
    denoisified = _strip_resolved(sentence)
    acute = _ACUTE_RE.search(denoisified) is not None
    informational = _INFORMATIONAL_RE.search(sentence) is not None
    resolved = _RESOLVED_RE.search(sentence) is not None
    if historical and not acute:
        return found
    if resolved and not acute:
        return found
    stripped = _strip_informational(sentence)
    distress = _STRONG_DISTRESS_RE.search(stripped) is not None
    alcohol_context = _ALCOHOL_CONTEXT_RE.search(sentence) is not None
    if informational and not (acute or distress):
        return found
    for rule in _RULES:
        match = rule.pattern.search(sentence)
        if match is None:
            continue
        if _sentence_has_negation_before(sentence, match.start()):
            continue
        if rule.needs_context and not (acute or alcohol_context or distress):
            continue
        found.add((rule.category, rule.tag))
    return found


def classify_emergency(text: str) -> EmergencyClassification:
    """Deterministically classify ``text`` for emergency routing.

    The function is pure: no I/O, no logging, no randomness. ``language``
    is ``ru`` when Cyrillic characters are present, otherwise ``en``.
    """
    language = detect_language(text)
    normalized = _normalize(text)
    categories: dict[EmergencyCategory, None] = {}
    tags: list[str] = []
    seen_tags: set[str] = set()
    for sentence in _SENTENCE_SPLIT.split(normalized):
        for category, tag in sorted(_classify_sentence(sentence), key=lambda item: item[1]):
            if category not in categories:
                categories[category] = None
            if tag not in seen_tags:
                seen_tags.add(tag)
                tags.append(tag)
    ordered = tuple(sorted(categories, key=lambda item: item.value))
    return EmergencyClassification(
        is_emergency=bool(ordered),
        categories=ordered,
        language=language,
        matched=tuple(tags),
    )


def is_emergency(text: str) -> bool:
    """Return True when ``text`` must take the emergency response path."""
    return classify_emergency(text).is_emergency


__all__ = [
    "EmergencyCategory",
    "EmergencyClassification",
    "classify_emergency",
    "detect_language",
    "is_emergency",
]
