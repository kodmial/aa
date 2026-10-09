"""Independent model-driven whole-turn helpfulness judge (kodmial/aa#286).

This module provides the independent real-product relevance oracle that
live Gate C was missing: production telemetry (groundedness plus
per-unit ``addresses_intent`` from the same verifier invocation) is
never independent proof on its own. The judge below is a separately
instantiated model control with its own audit identity, system prompt
and bounded schema. It judges the whole delivered reply against the
context-resolved intent: whether the turn as a whole helpfully answers
the practical request instead of padding, repetition, off-topic
digressions, or background explanation as a substitute for actionable
guidance.

Design constraints (all mandatory):

- Model-driven semantics only: no domain regexes, keyword tables,
  literal prompt cases, stem lists, or canned responses. The prompt
  carries only the resolved intent, the delivered reply and bounded
  conversation context.
- Bounded schema: the model returns exactly two strict booleans plus
  no free-form aggregate. Missing, null or wrong-type verdicts fail
  closed. Unknown envelope keys are discarded, never trusted.
- Independence where feasible: the caller must supply a separately
  instantiated model (distinct agent identity and fresh session per
  call, e.g. ``planner.with_agent(WHOLE_TURN_JUDGE_AGENT_V2)``).
  Generator/verifier family independence is explicitly limited: live
  AA serves one pinned provider family, so the judge shares the
  configured primary/fallback model family while remaining
  procedurally independent (separate agent, session, prompt and
  schema). Callers must surface that limitation in telemetry instead
  of claiming family independence. Reuse of production semantic
  booleans as independent proof is never allowed: see
  :func:`combine_telemetry_with_judge`.
- Whole-turn scope: pure empathy/glue sentences alongside substantive
  guidance need no action of their own; the verdict covers the turn as
  a whole. Natural dialog is never replaced with citations.
- Privacy: only booleans/counts/ids/hashes travel outward. Prompts,
  replies and evidence text never enter logs or telemetry.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, StrictBool

logger = logging.getLogger("aa.conversation.whole_turn_judge")

# Separately instantiated audit identity for the independent judge.
# Shares the configured provider model family (see module docstring for
# the explicit family limitation) but never shares a session, prompt or
# schema with the generator or the per-unit verifier.
WHOLE_TURN_JUDGE_AGENT_V2 = "aa-judge-v2"

# Explicit independence limitation surfaced in telemetry/metrics so no
# caller can claim family independence it does not have.
JUDGE_INDEPENDENCE_LIMITATION = (
    "procedurally-independent-separate-agent-session-prompt-schema; "
    "shares-configured-provider-model-family-with-generator-verifier"
)

WHOLE_TURN_JUDGE_MAX_ATTEMPTS = 1

# Repair generation uses the entire Evidence Pack (issues #286, #295).
# The former top-5 initial / top-8 repair rank windows are removed: both
# initial drafts and repair regens receive all selected canonical
# passages with full text. Zero means no generation-stage cap (kept for
# backwards import compatibility).
REPAIR_GENERATION_MAX_PASSAGES = 0


class WholeTurnDecision(BaseModel):
    """Provider-native whole-turn judge decision (transport only)."""

    helpful: StrictBool
    addresses_intent: StrictBool
    contains_substantive_claim: StrictBool

    model_config = {"extra": "forbid"}


class WholeTurnJudgeError(ValueError):
    """Whole-turn judge output failed completeness validation."""


@dataclass(frozen=True)
class WholeTurnJudgement:
    """Validated whole-turn judge verdict (internal)."""

    helpful: bool
    addresses_intent: bool
    contains_substantive_claim: bool
    model_identity: str = ""
    available: bool = True
    failure_category: str = ""


@dataclass(frozen=True)
class EvidenceWindowCoverage:
    """Privacy-safe evidence-window coverage (counts only, never text)."""

    pack_passages: int
    generation_window: int
    verifier_window: int
    omitted_from_generation: int
    omitted_from_verifier: int
    generation_omitted_chars: int
    verifier_omitted_chars: int
    cited_outside_generation: int
    cited_outside_verifier: int


def whole_turn_judge_schema() -> dict[str, object]:
    """Build the minimal whole-turn judge transport schema."""
    return {
        "type": "object",
        "properties": {
            "helpful": {"type": "boolean"},
            "addresses_intent": {"type": "boolean"},
            "contains_substantive_claim": {"type": "boolean"},
        },
        "required": ["helpful", "addresses_intent", "contains_substantive_claim"],
    }


def validate_whole_turn_decision(data: object) -> WholeTurnDecision:
    """Pydantic-validate one whole-turn judge decision (fail-closed)."""
    from pydantic import ValidationError as PydanticValidationError

    if isinstance(data, WholeTurnDecision):
        return data
    if isinstance(data, dict):
        try:
            return WholeTurnDecision.model_validate(data)
        except PydanticValidationError as exc:
            raise WholeTurnJudgeError(f"whole-turn judge output invalid: {exc}") from exc
    raise WholeTurnJudgeError("whole-turn judge output is not a structured object")


_DECISION_KEYS = frozenset({"helpful", "addresses_intent", "contains_substantive_claim"})


def parse_whole_turn_text_decision(text: str) -> dict[str, Any]:
    """Parse bounded text-JSON fallback for the whole-turn judge.

    Same envelope tolerance as the per-unit verifier: fences and
    surrounding prose are normalized, unknown keys are discarded, and
    only the three strict booleans decide. Missing, null or
    wrong-type verdicts fail closed at validation time.
    """
    from aa.conversation.verifier import _strip_text_json_fences, _tolerant_json_loads

    cleaned = _strip_text_json_fences(text or "")
    if not cleaned:
        raise WholeTurnJudgeError("whole-turn judge text output is empty")
    candidate = cleaned
    if not candidate.lstrip().startswith("{"):
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise WholeTurnJudgeError("whole-turn judge text output is not a JSON object")
        candidate = candidate[start : end + 1]
    try:
        data = _tolerant_json_loads(candidate)
    except (json.JSONDecodeError, ValueError, SyntaxError, TypeError) as exc:
        raise WholeTurnJudgeError(f"whole-turn judge text output is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise WholeTurnJudgeError("whole-turn judge text output is not a JSON object")
    filtered = {key: value for key, value in data.items() if key in _DECISION_KEYS}
    validate_whole_turn_decision(filtered)
    return dict(filtered)


def build_whole_turn_judge_prompt(
    *,
    resolved_intent: str,
    reply: str,
    context: str = "",
) -> str:
    """Render the whole-turn judge payload (no domain vocabulary).

    Generic wording only: the model judges whether the whole delivered
    reply helpfully answers the context-resolved intent. Brief empathy
    or acknowledgement alongside guidance is natural glue and needs no
    action of its own. A reply that only explains background, repeats
    generic statements, drifts off-topic, or substitutes difficulty
    description for practical guidance is not helpful.
    """
    lines: list[str] = [
        "Judge the whole delivered reply against the context-resolved intent.",
        "Return helpful true only when the reply as a whole helpfully answers "
        "the practical request: it gives usable guidance or a directly relevant "
        "explanation tied to what was asked. Return helpful false for padding, "
        "repetition, off-topic digressions, generic background restatement, or "
        "difficulty description offered as a substitute for practical guidance.",
        "Brief empathy or acknowledgement alongside guidance is natural and "
        "needs no action of its own; do not require every sentence to act.",
        "Set addresses_intent true only when the reply addresses the resolved "
        "intent described below. Set contains_substantive_claim true when the "
        "reply contains any substantive external claim, fact, mechanism, or "
        "recommended action; set it false only for pure conversational glue, "
        "a truthful assistant capability statement, or a safety notice.",
        "Judge semantic helpfulness to the actual intent, never keyword overlap.",
        "<resolved_intent>",
        (resolved_intent or "").strip()[:2000] or "(no resolved intent supplied)",
        "</resolved_intent>",
        "<delivered_reply>",
        (reply or "").strip()[:4000] or "(no reply supplied)",
        "</delivered_reply>",
    ]
    if (context or "").strip():
        lines.extend(
            [
                "<conversation_context>",
                context.strip()[:2000],
                "</conversation_context>",
            ]
        )
    return "\n".join(lines)


def _judge_identity(model: Any) -> str:
    for attr in ("agent", "primary_model", "model"):
        try:
            value = getattr(model, attr, "")
        except Exception:
            continue
        if isinstance(value, str) and value.strip():
            return value.strip()[:64]
    return type(model).__name__[:64]


async def judge_whole_turn(
    *,
    resolved_intent: str,
    reply: str,
    context: str = "",
    model: Any,
) -> WholeTurnJudgement:
    """Invoke the independent whole-turn judge once (fail-closed).

    ``model`` must be a separately instantiated control (distinct agent
    identity from generator/verifier). Empty intent or reply fails
    closed to not-helpful without a model call. Any transport or
    validation failure fails closed to not-helpful, never to helpful.
    Only booleans travel outward.
    """
    identity = _judge_identity(model)
    if not (resolved_intent or "").strip() or not (reply or "").strip():
        return WholeTurnJudgement(
            helpful=False,
            addresses_intent=False,
            contains_substantive_claim=False,
            model_identity=identity,
        )
    prompt = build_whole_turn_judge_prompt(
        resolved_intent=resolved_intent, reply=reply, context=context
    )
    system = (
        "You are an independent whole-turn response-quality judge. "
        "Return exactly the required strict boolean JSON decision."
    )
    from aa.opencode.errors import OpenCodeRateLimitError

    try:
        structured = getattr(model, "ainvoke_structured", None)
        if callable(structured):
            try:
                raw = await structured(
                    prompt,
                    system=system,
                    schema=whole_turn_judge_schema(),
                    retry_count=WHOLE_TURN_JUDGE_MAX_ATTEMPTS,
                )
                decision = validate_whole_turn_decision(raw)
            except OpenCodeRateLimitError:
                raise
            except Exception:
                text_invoke = getattr(model, "_ainvoke_text", None) or getattr(
                    model, "ainvoke", None
                )
                if not callable(text_invoke):
                    raise
                first = text_invoke
                if getattr(first, "__name__", "") == "_ainvoke_text":
                    text_reply = await first(prompt, system=system)
                else:
                    from langchain_core.messages import HumanMessage, SystemMessage

                    message = await first(
                        [SystemMessage(content=system), HumanMessage(content=prompt)]
                    )
                    content = getattr(message, "content", "")
                    text_reply = content if isinstance(content, str) else str(content)
                decision = validate_whole_turn_decision(parse_whole_turn_text_decision(text_reply))
        else:
            from langchain_core.messages import HumanMessage, SystemMessage

            plain = getattr(model, "ainvoke", None)
            if not callable(plain):
                raise WholeTurnJudgeError("judge model has no invocation path")
            message = await plain([SystemMessage(content=system), HumanMessage(content=prompt)])
            content = getattr(message, "content", "")
            text_reply = content if isinstance(content, str) else str(content)
            decision = validate_whole_turn_decision(parse_whole_turn_text_decision(text_reply))
    except Exception as exc:
        # A rate limit requires fresh-runner checkpoint recovery, not a
        # fabricated negative semantic verdict. Keep the 429 lifecycle.
        if isinstance(exc, OpenCodeRateLimitError):
            raise
        category = type(exc).__name__
        logger.warning("whole-turn judge unavailable; failing closed", extra={"category": category})
        return WholeTurnJudgement(
            helpful=False,
            addresses_intent=False,
            contains_substantive_claim=False,
            model_identity=identity,
            available=False,
            failure_category=category,
        )
    return WholeTurnJudgement(
        helpful=bool(decision.helpful),
        addresses_intent=bool(decision.addresses_intent),
        contains_substantive_claim=bool(decision.contains_substantive_claim),
        model_identity=identity,
    )


def combine_telemetry_with_judge(*, telemetry_pass: bool, judge: WholeTurnJudgement | None) -> bool:
    """Combine production telemetry with the independent judge (fail-closed).

    An independent FAIL always overrides a telemetry PASS: reuse of
    production semantic booleans is never independent proof. Without a
    judge verdict this returns the telemetry signal unchanged (used by
    offline/unit paths); live Gate C must always supply the judge, and
    its absence there is recorded as a limitation, never as proof.
    """
    if judge is None:
        return bool(telemetry_pass)
    if not bool(telemetry_pass):
        return False
    return bool(judge.helpful and judge.addresses_intent)


def assess_evidence_window_coverage(
    pack: Sequence[dict[str, Any]],
    *,
    generation_window: int,
    verifier_window: int,
    generation_max_chars: int = 0,
    verifier_max_chars: int = 0,
    cited_ids: Sequence[str] = (),
) -> EvidenceWindowCoverage:
    """Measure window coverage on counts only (never text).

    A window value of ``0`` (or any value covering the whole pack) means
    the full Evidence Pack reaches that stage with full text: nothing is
    omitted there. Positive windows preserve the legacy bounded counting
    for observability. Purely mechanical counting; no semantic judgement.
    """
    items = [item for item in (pack or []) if isinstance(item, dict)]
    total = len(items)
    gen_window = total if int(generation_window) <= 0 else min(total, int(generation_window))
    ver_window = total if int(verifier_window) <= 0 else min(total, int(verifier_window))
    window_ids = [str(item.get("passage_id", "")) for item in items[:gen_window]]
    verifier_ids = [str(item.get("passage_id", "")) for item in items[:ver_window]]
    window_set = {value for value in window_ids if value}
    verifier_set = {value for value in verifier_ids if value}
    gen_omitted_chars = 0
    ver_omitted_chars = 0
    # A max-chars value of 0 means full source text (no display cut).
    for item in items[:gen_window]:
        text = item.get("text", "")
        if (
            isinstance(text, str)
            and int(generation_max_chars) > 0
            and len(text) > generation_max_chars
        ):
            gen_omitted_chars += len(text) - generation_max_chars
    for item in items[:ver_window]:
        text = item.get("text", "")
        if isinstance(text, str) and int(verifier_max_chars) > 0 and len(text) > verifier_max_chars:
            ver_omitted_chars += len(text) - verifier_max_chars
    # Passages beyond the window still hold full stored text; count the
    # total stored characters outside each window as omitted context.
    for item in items[gen_window:]:
        text = item.get("text", "")
        if isinstance(text, str):
            gen_omitted_chars += len(text)
    for item in items[ver_window:]:
        text = item.get("text", "")
        if isinstance(text, str):
            ver_omitted_chars += len(text)
    cited = [str(value) for value in (cited_ids or []) if str(value)]
    cited_out_gen = sum(1 for value in cited if value not in window_set)
    cited_out_ver = sum(1 for value in cited if value not in verifier_set)
    return EvidenceWindowCoverage(
        pack_passages=total,
        generation_window=min(total, gen_window),
        verifier_window=min(total, ver_window),
        omitted_from_generation=max(0, total - gen_window),
        omitted_from_verifier=max(0, total - ver_window),
        generation_omitted_chars=int(gen_omitted_chars),
        verifier_omitted_chars=int(ver_omitted_chars),
        cited_outside_generation=int(cited_out_gen),
        cited_outside_verifier=int(cited_out_ver),
    )


def coverage_to_metrics(coverage: EvidenceWindowCoverage) -> dict[str, int]:
    """Render coverage as privacy-safe numeric metrics."""
    return {
        "pack_passages": coverage.pack_passages,
        "generation_window": coverage.generation_window,
        "verifier_window": coverage.verifier_window,
        "omitted_from_generation": coverage.omitted_from_generation,
        "omitted_from_verifier": coverage.omitted_from_verifier,
        "generation_omitted_chars": coverage.generation_omitted_chars,
        "verifier_omitted_chars": coverage.verifier_omitted_chars,
        "cited_outside_generation": coverage.cited_outside_generation,
        "cited_outside_verifier": coverage.cited_outside_verifier,
    }


def _no_forbidden_text_signals() -> tuple[str, ...]:
    return ()


__all__ = [
    "JUDGE_INDEPENDENCE_LIMITATION",
    "REPAIR_GENERATION_MAX_PASSAGES",
    "WHOLE_TURN_JUDGE_AGENT_V2",
    "WHOLE_TURN_JUDGE_MAX_ATTEMPTS",
    "EvidenceWindowCoverage",
    "WholeTurnDecision",
    "WholeTurnJudgeError",
    "WholeTurnJudgement",
    "assess_evidence_window_coverage",
    "build_whole_turn_judge_prompt",
    "combine_telemetry_with_judge",
    "coverage_to_metrics",
    "judge_whole_turn",
    "parse_whole_turn_text_decision",
    "validate_whole_turn_decision",
    "whole_turn_judge_schema",
]
