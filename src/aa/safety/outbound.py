"""Mandatory outbound response safety gate (kodmial/aa#252).

The gate certifies the outbound recommended action after generation and
before Telegram delivery. Its decision is independent of book-grounding:
a response that is genuinely supported by an authentic AA book passage
can still FAIL here when it advises the user to drink.

Policy (independent of citation validity):

- never recommend beginning or resuming alcohol use to test whether one
  can stop;
- never invite self-diagnosis by drinking, experimental controlled
  drinking, or an abrupt stop after a drinking trial;
- never recommend dangerous unsupported self-detox procedures.

Discussing a historical book passage as a subject of analysis or
cautionary context is not the same as advising it as a present-day
behavioral action and must not fail. The AA source is never redacted
and authentic quote text is never marked fake: provenance stays exact
and the gate distinguishes a historical passage under discussion from
a recommended action.

Deterministic and offline: pure functions over the response text (no
network, no LLM, no randomness). Intent plus action semantics over
Russian stems, not exact-phrase filters, so paraphrases are caught.
Privacy-safe: verdicts carry only booleans, categories and lengths,
never message bodies.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger("aa.safety.outbound")

# Bounded outbound recovery attempts after a harmful draft is blocked.
OUTBOUND_SAFETY_MAX_REPAIRS = 2

# Recovery control instruction for safe regeneration (kodmial/aa#300).
# This is a SEPARATE control instruction, never concatenated to raw user
# content or to a semantic retrieval query. Callers must pass it through
# a dedicated safety-policy channel (separate prompt block), keeping the
# resolved user intent as the retrieval/generation request verbatim.
SAFE_RECOVERY_INSTRUCTION = (
    "Ответ должен помогать оставаться трезвым без употребления. "
    "Никогда не советуй начинать, возобновлять или пробовать пить, "
    "в том числе чтобы проверить себя или резко прекратить после пробы."
)


def _normalize(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


_SENTENCE_SPLIT = re.compile(r"[.!?…\n;]+")

# Drinking as an action (stems, not exact phrases so paraphrases match).
# Short imperatives keep word boundaries so ordinary words such as
# "успейте" never count as a drinking action.
_DRINK_RE = re.compile(
    r"пить|пью|пьешь|пьет|пьем|пьете|\bпе[йи]\b|\bпейте\b|выпей|выпейте|выпить|"
    r"выпив|выпива|выпивк|употреб\w*\s+алкогол|алкогол\w*|спиртн\w*|пив[оа]?\w*|"
    r"рюмк\w*|бокал\w*|стопк\w*|стакан\w*|буха|бухл|drink|alcohol|beer"
)

# Directive / invitation modality toward the user (recommendation,
# invitation, imperative or infinitive trial framed as help).
_ADVICE_RE = re.compile(
    r"попробуй|попробуйте|попробовать|стоит|следует|нужно|надо|необходимо|"
    r"рекоменду|совет|предлагаю|предложи|лучше|давай|давайте|можешь|можете|можно\s+попроб|"
    r"начни|начните|начать\s+пить|возобнови|возобновить|продолжи|продолжить|"
    r"продолжай|выпей|выпейте|пе[йи](те)?|проверь|проверьте|убедис|выясни|"
    r"испытай|испытайте|помогает\s+такой\s+опыт|поможет\s+такой\s+опыт|"
    r"помогает\s+опыт|попробуй\s+начать|попробовать\s+начать|честно\s+разобраться\s+помогает"
)

# Purpose of a drinking trial: self-test / controlled experiment.
_TEST_PURPOSE_RE = re.compile(
    r"провер|убеди|выясн|испыта|эксперимент|тест|контролир|умерен|огранич\w*\s+количеств|"
    r"только\s+пиво|доказать.*исключен|можешь\s+ли\s+остановить|можете\s+ли\s+остановить|"
    r"сможешь\s+ли|сможете\s+ли|как\s+обстоят\s+дела|станет\s+яснее|диагности|самодиагности"
)

# Abrupt stop framed as part of a drinking trial.
_ABRUPT_STOP_RE = re.compile(
    r"резко\s+(прекрат|брос|останов)|"
    r"резко\s+прекрати|несколько\s+раз|остановиться\s+после|прекратить\s+после|"
    r"бросить\s+после|остановить\w*\s+после"
)

# Negation / abstinence direction: the sentence tells the user NOT to
# drink, which is safe and must never fail.
_NEGATION_RE = re.compile(
    r"\bне\s+(пей|пейте|выпей|выпейте|пить|выпить|стоит\s+пить|нужно\s+пить|"
    r"надо\s+пить|пробуй|пробуйте|начинай|начинайте|продолжай|продолжайте)\b|"
    r"\bне\s+следует\s+пить\b|\bвоздержи|откажись\s+от|оставайся\s+трезв|"
    r"оставайтесь\s+трезв|без\s+употребления|не\s+употребляй"
)

# Explicit refusal of the drink-to-test action: stating that the
# assistant cannot recommend drinking is safe, never advice.
_REFUSAL_RE = re.compile(
    r"не\s+могу\s+(посоветовать|предложить|порекомендовать|помочь\s+с\s+этим)|"
    r"не\s+буду\s+советовать|не\s+советую\s+пить|не\s+рекомендую\s+пить"
)

# Strong invitation/imperative modality: a direct call to act. Weak
# modals such as "следует" also appear in cautionary context
# ("повторять его опасно и не следует"), so cautionary sentences need a
# strong invitation to count as advice.
_STRONG_ADVICE_RE = re.compile(
    r"попробуй|попробуйте|попробовать|начни|начните|возобнови|возобновить|"
    r"продолжи|продолжить|продолжай|выпей|выпейте|\bпе[йи]\b|\bпейте\b|"
    r"давай|давайте|предлагаю|совет|рекоменду|проверь|проверьте|испытай|"
    r"испытайте|помогает\s+такой\s+опыт|поможет\s+такой\s+опыт|"
    r"честно\s+разобраться\s+помогает"
)

# Cautionary framing: the sentence warns against the action instead of
# inviting it. With only a weak modal and no strong invitation, the
# sentence discusses history or warns; it does not advise drinking.
_CAUTION_RE = re.compile(
    r"не\s+следует|нельзя|опасно|не\s+является\s+рекомендацией|"
    r"не\s+рекомендация|не\s+совет|не\s+следует\s+повторять"
)

# Dangerous unsupported self-detox instruction as a recommended action.
_DETOX_ACTION_RE = re.compile(r"детокс|сняти[ея]\s+абстиненц|выход\s+из\s+запоя|похмели|похмел")
_DETOX_DIRECTIVE_RE = re.compile(
    r"принимай|принимайте|выпейте?\s+\d|доза|дозировка|доз\w*|расписание|schedule|"
    r"пейте\s+меньше\s+каждый\s+день|taper"
)


# Drinking mentioned as subordinate craving context or as an avoided outcome
# is not an advised action. These markers exempt a drinking clause from the
# cross-clause fallback so abstinence-direction guidance stays safe. Stems
# only, never exact phrases.
_CRAVING_CONTEXT_RE = re.compile(
    r"тянет|тяг\w*|хочет\w*|желани\w*|потянуло",
    re.IGNORECASE,
)
_AVOIDANCE_RE = re.compile(
    r"избеж\w*|избег\w*|справ\w*\s+с|борьб\w*|удержа\w*|воздерж|преодол\w*",
    re.IGNORECASE,
)

# Clause boundary for governance-local advice detection (kodmial/aa#257
# recurrence 2): drinking advice holds only when the directive governs the
# drinking action inside the same clause. Split on punctuation plus
# subordinate conjunctions so context clauses never license the advised
# clause. Cross-clause coordination below restores protection when a
# directive in one clause governs a non-exempt drinking action in another.
_CLAUSE_SPLIT_RE = re.compile(
    r"[,;:\u2014\u2013\-\(\)]+|\b(?:когда|если|чтобы|чтоб|потому|поскольку|"
    r"так\s+как|пока|где|куда|откуда|но|а|однако|зато|хотя)\b",
    re.IGNORECASE,
)


def _split_clauses(sentence: str) -> list[str]:
    """Split one sentence into governance-local clauses (generic, no text)."""
    return [part for part in _CLAUSE_SPLIT_RE.split(sentence or "") if part.strip()]


@dataclass(frozen=True)
class OutboundSafetyVerdict:
    """Outcome of certifying one outbound response."""

    safe: bool
    reason: str = ""
    category: str = ""


def _sentence_is_negated(sentence: str) -> bool:
    return _NEGATION_RE.search(sentence) is not None


def _sentence_advises_drinking(sentence: str) -> tuple[bool, str]:
    """Judge one sentence; return (unsafe, category).

    Governance-local (kodmial/aa#257 recurrence 2): the directive must
    govern the drinking action inside the same clause. Sentence-level
    drink+advice co-occurrence falsely blocked abstinence-direction
    guidance where drinking is subordinate context ("when it pulls to
    drink, try calling ...") while the advised action stays sober.
    """
    if not sentence.strip():
        return False, ""
    if (
        _REFUSAL_RE.search(sentence) is not None
        or _sentence_is_negated(sentence)
        or (_CAUTION_RE.search(sentence) is not None and _STRONG_ADVICE_RE.search(sentence) is None)
    ):
        # Mixed-sentence guard: a refusal, negation or cautionary prefix
        # does not excuse a later clause that itself invites a drinking
        # trial with directive modality. Clauses without such an invitation
        # (discussion, warning, abstinence direction) stay safe; a clause
        # carrying directive drink-trial signal falls through to the strict
        # drink-plus-advice detection below.
        clauses = [
            c
            for c in re.split(r"[,:\u2014\u2013\-]+|\bно\b|\bа\b|\bоднако\b|\bхотя\b", sentence)
            if c.strip()
        ]
        for clause in clauses:
            if (
                _REFUSAL_RE.search(clause) is not None
                or _NEGATION_RE.search(clause) is not None
                or (
                    _CAUTION_RE.search(clause) is not None
                    and _STRONG_ADVICE_RE.search(clause) is None
                )
                or (_DRINK_RE.search(clause) is None and _DETOX_ACTION_RE.search(clause) is None)
            ):
                continue
            has_action = (
                _DRINK_RE.search(clause) is not None or _DETOX_ACTION_RE.search(clause) is not None
            )
            if _STRONG_ADVICE_RE.search(clause) is not None and has_action:
                break
            if (
                has_action
                and _ADVICE_RE.search(clause) is not None
                and (
                    _TEST_PURPOSE_RE.search(clause) is not None
                    or _ABRUPT_STOP_RE.search(clause) is not None
                    or _DETOX_DIRECTIVE_RE.search(clause) is not None
                )
            ):
                break
        else:
            return False, ""
    # Governance-local main detection: each clause must itself advise
    # drinking. Purpose words stay scoped to governed clauses via the
    # cross-clause fallback below, never as a sentence-level drink+advice
    # fast-path, so sober purpose guidance with negated or avoided drinking
    # stays safe.
    clauses = _split_clauses(sentence)
    for clause in clauses:
        if not clause.strip():
            continue
        # A clause that refuses, negates or warns (without a strong
        # invitation) discusses drinking instead of advising it.
        if (
            _REFUSAL_RE.search(clause) is not None
            or _NEGATION_RE.search(clause) is not None
            or (_CAUTION_RE.search(clause) is not None and _STRONG_ADVICE_RE.search(clause) is None)
        ):
            continue
        clause_drink = _DRINK_RE.search(clause) is not None
        clause_advice = _ADVICE_RE.search(clause) is not None
        clause_test = _TEST_PURPOSE_RE.search(clause) is not None
        clause_abrupt = _ABRUPT_STOP_RE.search(clause) is not None
        if clause_drink and clause_advice:
            if clause_abrupt:
                return True, "abrupt-stop-after-drinking"
            if clause_test:
                return True, "drink-test-advice"
            return True, "resume-drinking-advice"
        # Infinitive trial without an explicit modal: "начать пить и резко
        # прекратить" as helped experience is still an invitation to act.
        clause_helps = "помогает" in clause or "поможет" in clause or "опыт" in clause
        if clause_drink and (clause_test or clause_abrupt) and clause_helps:
            if clause_abrupt:
                return True, "abrupt-stop-after-drinking"
            return True, "drink-test-advice"
        # Dangerous self-detox instruction as an action.
        if _DETOX_ACTION_RE.search(clause) is not None and (
            _DETOX_DIRECTIVE_RE.search(clause) is not None or clause_advice
        ):
            if _NEGATION_RE.search(clause) is None:
                return True, "unsafe-detox-advice"
    # Cross-clause fallback: a directive in one clause governing a
    # non-exempt drinking action in another (coordination with comma or
    # words such as "a/potom", or a purpose link). Drinking mentioned as
    # craving context or as an avoided outcome never counts as advised.
    if _ADVICE_RE.search(sentence) is not None and _DRINK_RE.search(sentence) is not None:
        candidate = False
        for clause in clauses:
            if _DRINK_RE.search(clause) is None:
                continue
            if (
                _REFUSAL_RE.search(clause) is not None
                or _NEGATION_RE.search(clause) is not None
                or (
                    _CAUTION_RE.search(clause) is not None
                    and _STRONG_ADVICE_RE.search(clause) is None
                )
                or _CRAVING_CONTEXT_RE.search(clause) is not None
                or _AVOIDANCE_RE.search(clause) is not None
            ):
                continue
            candidate = True
            break
        if candidate:
            if _ABRUPT_STOP_RE.search(sentence) is not None:
                return True, "abrupt-stop-after-drinking"
            if _TEST_PURPOSE_RE.search(sentence) is not None:
                return True, "drink-test-advice"
            return True, "resume-drinking-advice"
    return False, ""


def classify_outbound_safety(response_text: str) -> OutboundSafetyVerdict:
    """Certify one outbound response independently of book-grounding.

    Returns safe when no sentence advises the user to drink or to run
    an unsafe self-detox. Pure historical discussion, cautionary
    context and abstinence direction stay safe. Logs only the decision,
    category and length, never the response body.
    """
    text = response_text or ""
    normalized = _normalize(text)
    for sentence in _SENTENCE_SPLIT.split(normalized):
        unsafe, category = _sentence_advises_drinking(sentence)
        if unsafe:
            verdict = OutboundSafetyVerdict(
                safe=False, reason=f"unsafe:{category}", category=category
            )
            logger.info(
                "outbound safety decision",
                extra={
                    "safe": False,
                    "reason": verdict.reason,
                    "text_len": len(text),
                },
            )
            return verdict
    verdict = OutboundSafetyVerdict(safe=True, reason="outbound-safe")
    logger.info(
        "outbound safety decision",
        extra={"safe": True, "reason": verdict.reason, "text_len": len(text)},
    )
    return verdict


def is_outbound_safe(response_text: str) -> bool:
    """Return True when ``response_text`` passes the outbound safety gate."""
    return classify_outbound_safety(response_text).safe


__all__ = [
    "OUTBOUND_SAFETY_MAX_REPAIRS",
    "SAFE_RECOVERY_INSTRUCTION",
    "OutboundSafetyVerdict",
    "classify_outbound_safety",
    "is_outbound_safe",
]
