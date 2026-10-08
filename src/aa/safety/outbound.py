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

# Transparent safe-unavailability reply for the outbound path. It refuses
# the drink-to-test action explicitly, stays short and inside the #83
# envelope, carries no substantive book claim (so it needs no verifier
# provenance), and is distinct from the generic clarification and the
# generic retry reply so Gate C never counts it as a completed grounded
# answer.
SAFE_UNAVAILABLE_REPLY = (
    "Не могу посоветовать пробовать пить, чтобы проверить себя, "
    "это опасно. Расскажите, что сейчас важнее всего, "
    "и разберём ближайшие шаги без употребления."
)

# Bounded outbound recovery attempts after a harmful draft is blocked.
OUTBOUND_SAFETY_MAX_REPAIRS = 2

# Recovery instruction appended to the user message when regenerating a
# safe replacement from diversified evidence. It constrains the action
# without adding external clinical guidance as a knowledge source.
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


@dataclass(frozen=True)
class OutboundSafetyVerdict:
    """Outcome of certifying one outbound response."""

    safe: bool
    reason: str = ""
    category: str = ""


def _sentence_is_negated(sentence: str) -> bool:
    return _NEGATION_RE.search(sentence) is not None


def _sentence_advises_drinking(sentence: str) -> tuple[bool, str]:
    """Judge one sentence; return (unsafe, category)."""
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
    has_drink = _DRINK_RE.search(sentence) is not None
    has_advice = _ADVICE_RE.search(sentence) is not None
    has_test = _TEST_PURPOSE_RE.search(sentence) is not None
    has_abrupt = _ABRUPT_STOP_RE.search(sentence) is not None
    if has_drink and has_advice:
        if has_test or has_abrupt:
            return True, "drink-test-advice"
        return True, "resume-drinking-advice"
    # Infinitive trial without an explicit modal: "начать пить и резко
    # прекратить" as helped experience is still an invitation to act.
    helps = "помогает" in sentence or "поможет" in sentence or "опыт" in sentence
    if has_drink and (has_test or has_abrupt) and helps:
        return True, "drink-test-advice"
    # Abrupt stop after a drinking trial recommended as a procedure.
    if has_abrupt and has_advice and has_drink:
        return True, "abrupt-stop-after-drinking"
    # Dangerous self-detox instruction as an action.
    if _DETOX_ACTION_RE.search(sentence) is not None and (
        _DETOX_DIRECTIVE_RE.search(sentence) is not None or has_advice
    ):
        if not _sentence_is_negated(sentence):
            return True, "unsafe-detox-advice"
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
    "SAFE_UNAVAILABLE_REPLY",
    "OutboundSafetyVerdict",
    "classify_outbound_safety",
    "is_outbound_safe",
]
