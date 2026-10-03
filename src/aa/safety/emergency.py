"""Deterministic acute-case detection for safety guardrails.

Detects (EN + RU, case-insensitive, regex-based, no ML) acute cases that must
receive immediate emergency/medical guidance before any ordinary AA-oriented
response:

- severe withdrawal;
- seizure;
- hallucination;
- poisoning/overdose;
- loss of consciousness;
- immediate self-harm risk.

The emergency template contains no medication dosing and no unsupervised
detox instructions by construction; see ``contains_prohibited_medical_advice``
and the corresponding tests.
"""

from __future__ import annotations

import re

#: Deterministic acute-case patterns: (category, compiled regex).
ACUTE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "severe-withdrawal",
        re.compile(
            r"(?i)(delirium\s+tremens|\bdt\b.*withdraw|withdraw.*\bdt\b|severe\s+withdraw|"
            r"heavy\s+withdraw|withdrawal\s+seizure|"
            r"сильная\s+ломка|тяж[её]лый\s+абстинент|белая\s+горячка|алкогольный\s+делирий)",
        ),
    ),
    (
        "withdrawal",
        re.compile(
            r"(?i)(withdrawal|withdraw(ing|al)?\s+(symptom|shak|tremor|sweat|nausea|vomit)|"
            r"shaking\s+(uncontrollably|violently|badly)|can't\s+stop\s+shaking|"
            r"абстинент|ломка|трясет|трясёт|дрожь|похмелье\s+с\s+судорогами)",
        ),
    ),
    (
        "seizure",
        re.compile(
            r"(?i)(seizure|convulsion|epilep|having\s+a\s+fit|fits?\s+and\s+(shaking|blackout)|"
            r"судорог|припадок|эпилепс|конвульси)",
        ),
    ),
    (
        "hallucination",
        re.compile(
            r"(?i)(hallucinat|hearing\s+voices|seeing\s+things|visions?\s+that.*not\s+real|"
            r"deliri(um|ous)|галлюцинац|вижу\s+то.*чего\s+нет|слышу\s+голоса|голоса\s+в\s+голове)",
        ),
    ),
    (
        "poisoning-overdose",
        re.compile(
            r"(?i)(overdose|overdos(ed|ing)?|poison(ing|ed)?|took\s+too\s+(much|many)|"
            r"drank\s+(way\s+)?too\s+much|methanol|surrogate\s+alcohol|"
            r"передоз|отравлен|отравился|отравилась|выпил\s+слишком\s+много|"
            r"суррогат|метанол)",
        ),
    ),
    (
        "loss-of-consciousness",
        re.compile(
            r"(?i)(unconscious|passed\s+out|passing\s+out|black(ed)?\s+out|fainted|fainting|"
            r"can't\s+wake|cannot\s+wake|unresponsive|collapsed\s+and.*not\s+waking|"
            r"без\s+сознания|потерял\s+сознание|потеряла\s+сознание|отключился|"
            r"отключилась|обморок|не\s+приходит\s+в\s+себя|не\s+могу\s+очнуться)",
        ),
    ),
    (
        "self-harm",
        re.compile(
            r"(?i)(kill\s+myself|killing\s+myself|suicid|end\s+my\s+life|take\s+my\s+life|"
            r"don't\s+want\s+to\s+live|do\s+not\s+want\s+to\s+live|hurt\s+myself|"
            r"harm\s+myself|cut\s+myself|self[\s\-]?harm|"
            r"суицид|покончить\s+(с\s+собой|жизнью)|убью\s+себя|убить\s+себя|"
            r"не\s+хочу\s+(жить|больше\s+жить)|причинить\s+себе\s+вред|"
            r"нанесу\s+себе\s+вред|порежу\s+себя|вскрою\s+вены)",
        ),
    ),
)

#: Patterns that must never appear in bot output (dosing / detox protocols).
PROHIBITED_MEDICAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?i)\b\d+\s*(mg|milligram|g\b|gram|ml|tablet|pill|dose)\b.*(take|drink|inject|use)"
    ),
    re.compile(r"(?i)(take|drink|inject|use)\s+\d+\s*(mg|g\b|ml|tablet|pill)"),
    re.compile(r"(?i)detox\s+(protocol|schedule|plan|regimen)"),
    re.compile(r"(?i)(tapering|taper)\s+(schedule|plan|protocol|regimen)"),
    re.compile(r"(?i)unsupervised\s+(detox|withdrawal)"),
    re.compile(r"(?i)\bсхема\s+(детокс|выхода|отмены)"),
    re.compile(r"(?i)\bдозиров"),
    re.compile(r"(?i)\bпринимай(те)?\s+\d+"),
    re.compile(r"(?i)\b\d+\s*(мг|г\b|мл|таблетк)"),
)

EMERGENCY_RESPONSE_EN = (
    "This sounds like it could be a medical emergency. "
    "If you or someone near you may be in immediate danger — "
    "for example severe withdrawal, seizure, hallucinations, suspected "
    "poisoning/overdose, loss of consciousness, or thoughts of self-harm — "
    "call your local emergency number now (for example 112 or 103) or go to "
    "the nearest emergency department. "
    "Do not stay alone if possible; ask someone nearby for help. "
    "I cannot provide medication doses or detox instructions. "
    "Once you are safe and, if needed, with medical professionals, "
    "we can talk about AA literature support."
)

EMERGENCY_RESPONSE_RU = (
    "Похоже, это может быть неотложное состояние. "
    "Если вам или человеку рядом может угрожать опасность — например сильная "
    "ломка/абстиненция, судороги, галлюцинации, возможное отравление или "
    "передозировка, потеря сознания либо мысли о самоповреждении — "
    "немедленно позвоните в скорую (112 или 103) или обратитесь в ближайшее "
    "отделение неотложной помощи. "
    "По возможности не оставайтесь в одиночестве, попросите о помощи тех, кто "
    "рядом. "
    "Я не называю дозы лекарств и не даю инструкций по самостоятельному "
    "выходу из запоя. "
    "Когда вы будете в безопасности и, при необходимости, под наблюдением "
    "врачей, мы сможем поговорить о поддержке по литературе АА."
)


def detect_acute_category(text: str) -> str | None:
    """Return the first matching acute category, or ``None``.

    Deterministic: normalized case-insensitive regex scan in the fixed order
    of :data:`ACUTE_PATTERNS`. Empty/blank input returns ``None``.
    """
    if not text or not text.strip():
        return None
    normalized = text.strip()
    for category, pattern in ACUTE_PATTERNS:
        if pattern.search(normalized):
            return category
    return None


def is_emergency(text: str) -> bool:
    """Whether ``text`` deterministically routes to emergency guidance."""
    return detect_acute_category(text) is not None


def build_emergency_response(*, lang: str = "en") -> str:
    """Return the fixed emergency template (never dosing/detox).

    ``lang`` selects ``"ru"`` for Russian, anything else for English. The
    template is static so routing never depends on model output.
    """
    if lang.lower().startswith("ru"):
        return EMERGENCY_RESPONSE_RU
    return EMERGENCY_RESPONSE_EN


def detect_language(text: str) -> str:
    """Detect ``"ru"`` vs ``"en"`` by Cyrillic presence (deterministic)."""
    if re.search(r"[А-Яа-яЁё]", text):
        return "ru"
    return "en"


def contains_prohibited_medical_advice(text: str) -> bool:
    """Whether ``text`` contains dosing/detox instructions (must be False)."""
    return any(pattern.search(text) for pattern in PROHIBITED_MEDICAL_PATTERNS)


__all__ = [
    "ACUTE_PATTERNS",
    "EMERGENCY_RESPONSE_EN",
    "EMERGENCY_RESPONSE_RU",
    "PROHIBITED_MEDICAL_PATTERNS",
    "build_emergency_response",
    "contains_prohibited_medical_advice",
    "detect_acute_category",
    "detect_language",
    "is_emergency",
]
