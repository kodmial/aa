"""Qualification campaign mode resolver (kodmial/aa#431).

The self-proving qualification currently re-runs the full A/B/C/E/F chain
for every problematic live series. During the transient Gate C focus the
repository must instead:

- run exactly the 18 unchanged Gate C scenarios;
- never execute Gate A/B (no placeholders, no fake evidence);
- never hand a C-only PASS out as a trusted full qualification;
- keep ``full`` available explicitly via a dispatch input or a repository
  setting without a second automatic qualifier.

This pure module owns the single mode resolution rule used by the AA
workflow: the ``workflow_dispatch`` input has priority over the repository
variable, an invalid explicit value fails closed, and automatic triggers
(push, post-merge lifecycle dispatches, workflow-run recovery) default to
``gate-c-focus`` while the transient focus mode is in force. Returning a
typed :class:`CampaignModeDecision` keeps every workflow branch honest
about why it runs.
"""

from __future__ import annotations

from dataclasses import dataclass

MODE_GATE_C_FOCUS = "gate-c-focus"
MODE_FULL = "full"
VALID_MODES: tuple[str, ...] = (MODE_GATE_C_FOCUS, MODE_FULL)
AUTOMATIC_DEFAULT_MODE = MODE_GATE_C_FOCUS

SOURCE_DISPATCH_INPUT = "dispatch-input"
SOURCE_REPO_VARIABLE = "repo-variable"
SOURCE_AUTOMATIC_DEFAULT = "automatic-default"


class CampaignModeError(ValueError):
    """Raised when an explicit mode value is invalid (fail closed)."""


@dataclass(frozen=True)
class CampaignModeDecision:
    """The resolved campaign mode plus the resolution provenance."""

    mode: str
    source: str
    requested_value: str = ""

    def __post_init__(self) -> None:
        if self.mode not in VALID_MODES:
            raise CampaignModeError(f"invalid resolved campaign mode: {self.mode!r}")
        allowed_sources = (
            SOURCE_DISPATCH_INPUT,
            SOURCE_REPO_VARIABLE,
            SOURCE_AUTOMATIC_DEFAULT,
        )
        if self.source not in allowed_sources:
            raise CampaignModeError(f"invalid campaign mode source: {self.source!r}")


def _validate_explicit(value: str, origin: str) -> str:
    normalized = str(value or "").strip()
    if normalized not in VALID_MODES:
        raise CampaignModeError(
            f"invalid {origin} qualification mode {value!r}; "
            f"expected one of {', '.join(VALID_MODES)}"
        )
    return normalized


def resolve_campaign_mode(
    *,
    dispatch_input: str = "",
    repo_variable: str = "",
    automatic_default: str = AUTOMATIC_DEFAULT_MODE,
) -> CampaignModeDecision:
    """Resolve one campaign mode with the documented priority chain.

    1. A non-empty ``workflow_dispatch`` input wins (owner's explicit
       choice); an invalid value fails closed.
    2. Otherwise a non-empty repository variable wins; an invalid value
       fails closed.
    3. Otherwise the automatic default applies. The default itself is
       validated and currently ``gate-c-focus`` while the transient Gate C
       focus mode is in force; ``full`` remains available by owner choice
       via the input or the repository variable only.
    """
    validated_default = _validate_explicit(automatic_default, "automatic default")
    if str(dispatch_input or "").strip():
        return CampaignModeDecision(
            mode=_validate_explicit(dispatch_input, "workflow_dispatch"),
            source=SOURCE_DISPATCH_INPUT,
            requested_value=str(dispatch_input).strip(),
        )
    if str(repo_variable or "").strip():
        return CampaignModeDecision(
            mode=_validate_explicit(repo_variable, "repository variable"),
            source=SOURCE_REPO_VARIABLE,
            requested_value=str(repo_variable).strip(),
        )
    return CampaignModeDecision(mode=validated_default, source=SOURCE_AUTOMATIC_DEFAULT)


def resolve_or_block(
    *,
    dispatch_input: str = "",
    repo_variable: str = "",
    automatic_default: str = AUTOMATIC_DEFAULT_MODE,
) -> CampaignModeDecision:
    """Resolve the mode or raise the typed normalized error for callers.

    The workflow steps wrap this into a fail-closed BLOCKED exit so a
    misconfigured mode can never silently run the wrong gate set.
    """
    return resolve_campaign_mode(
        dispatch_input=dispatch_input,
        repo_variable=repo_variable,
        automatic_default=automatic_default,
    )


def automatic_trigger_uses_full_mode(decision: CampaignModeDecision) -> bool:
    """Whether an automatic trigger would run the trailing full reducer.

    Automatic triggers (push, lifecycle dispatches without the mode input,
    workflow-run recovery) never run hidden FULL: entering full mode on
    them only happens through an owner-set repository variable, which is
    the documented explicit way back to the full qualification.
    """
    if decision.source == SOURCE_AUTOMATIC_DEFAULT:
        return False
    if decision.source in (SOURCE_DISPATCH_INPUT, SOURCE_REPO_VARIABLE):
        return decision.mode == MODE_FULL
    return False


__all__ = [
    "AUTOMATIC_DEFAULT_MODE",
    "MODE_FULL",
    "MODE_GATE_C_FOCUS",
    "SOURCE_AUTOMATIC_DEFAULT",
    "SOURCE_DISPATCH_INPUT",
    "SOURCE_REPO_VARIABLE",
    "VALID_MODES",
    "CampaignModeDecision",
    "CampaignModeError",
    "automatic_trigger_uses_full_mode",
    "resolve_campaign_mode",
    "resolve_or_block",
]
