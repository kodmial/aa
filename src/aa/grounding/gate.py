"""Deterministic multilingual grounding gate (issue #48).

The gate validates the actual Russian claim, not the mere presence of a
``source_id``. It enforces the deterministic subset of grounding:

- At least one cited unit must be source-exact text
  (:attr:`EvidenceKind.SOURCE_TEXT`) with complete version-pinned
  provenance. Navigation aids (book map, previews, embeddings, rankings,
  scores) and query artifacts (original, normalized, or rewritten
  queries) are never evidence, even when their text matches the claim.
- Exact-source Russian quotations must be verbatim substrings of the
  cited Russian source-exact text while the version-pinned Russian
  corpus is available. Without the Russian corpus the exact-source path
  fails closed; a generated translation must never pass as exact.
- Translated Russian renderings must carry an explicit translation
  label and provenance to the exact source unit(s), and pass only when
  the caller explicitly allows the translation fallback.
- Semantic support (entailment) of the actual Russian claim is checked
  through an injectable predicate over ``(claim, source_text)``. The
  default predicate is a conservative same-language lexical support
  check; cross-language production use must supply the qualified
  entailment function measured for the retrieval architecture (issue
  #47), never a bare ``source_id``. Unsupported interpretation
  introduced during translation or paraphrase therefore fails the gate.

Normalization (lowercasing, ``ё`` -> ``е``, punctuation handling) is a
matching aid only. Its outputs are never evidence.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from aa.grounding.quotes import (
    EvidenceKind,
    EvidenceUnit,
    Provenance,
    QuoteKind,
    contains_translation_label,
)
from aa.retrieval.normalize import ru_stem

EntailmentFn = Callable[[str, str], bool]
"""Predicate over ``(russian_claim, source_text)`` reporting support."""

_TOKEN_PATTERN = re.compile(r"[а-яa-z0-9]+")

_VERDICT_OK_EXACT = "ok-exact-source"
_VERDICT_OK_TRANSLATION = "ok-translation"
_VERDICT_EMPTY_CLAIM = "empty-claim"
_VERDICT_NO_SOURCE_TEXT = "no-source-text"
_VERDICT_UNKNOWN_PROVENANCE = "unknown-provenance"
_VERDICT_RU_CORPUS_UNAVAILABLE = "ru-corpus-unavailable"
_VERDICT_NO_RU_SOURCE = "no-ru-source-text"
_VERDICT_VERBATIM_MISMATCH = "verbatim-mismatch"
_VERDICT_TRANSLATION_AS_EXACT = "translation-as-exact"
_VERDICT_TRANSLATION_UNLABELED = "translation-unlabeled"
_VERDICT_TRANSLATION_FALLBACK_DISABLED = "translation-fallback-disabled"
_VERDICT_UNSUPPORTED_INTERPRETATION = "unsupported-interpretation"


@dataclass(frozen=True)
class GroundingVerdict:
    """Outcome of grounding one Russian claim."""

    passed: bool
    code: str
    reason: str
    source_exact: bool = False


def normalize_for_support(text: str) -> str:
    """Normalize ``text`` for support matching (never evidence itself)."""
    return text.lower().replace("ё", "е")


def _significant_tokens(text: str) -> set[str]:
    normalized = normalize_for_support(text)
    stems: set[str] = set()
    for token in _TOKEN_PATTERN.findall(normalized):
        if len(token) < 4:
            continue
        stem = ru_stem(token)
        if len(stem) >= 3:
            stems.add(stem)
    return stems


def default_entails(russian_claim: str, source_text: str) -> bool:
    """Conservative same-language support check (fail-closed proxy).

    Requires at least two shared stemmed significant tokens covering at
    least half of the claim's significant tokens. Stemming makes
    inflected Russian forms (``трезвости``/``трезвость``) meet without
    loosening the overlap threshold. Paraphrase or translation beyond
    that overlap must be judged by a qualified entailment function
    supplied by the caller; this default never passes cross-language
    pairs with disjoint vocabularies.
    """
    claim_tokens = _significant_tokens(russian_claim)
    if len(claim_tokens) < 2:
        return False
    source_tokens = _significant_tokens(source_text)
    shared = claim_tokens & source_tokens
    return len(shared) >= 2 and len(shared) / len(claim_tokens) >= 0.5


def _provenance_key(provenance: Provenance) -> tuple[str, str, str, str]:
    return (
        provenance.corpus_version,
        provenance.source_id,
        provenance.section_id,
        provenance.chunk_id,
    )


def check_grounding(
    *,
    russian_claim: str,
    quoted_text: str,
    quote_kind: QuoteKind,
    cited: Sequence[Provenance],
    evidence: Sequence[EvidenceUnit],
    ru_corpus_available: bool,
    allow_translation_fallback: bool,
    entails: EntailmentFn | None = None,
) -> GroundingVerdict:
    """Judge whether ``russian_claim`` is grounded in ``evidence``."""
    if not russian_claim.strip():
        return GroundingVerdict(False, _VERDICT_EMPTY_CLAIM, "claim must not be empty")

    usable = [unit for unit in evidence if unit.is_source_text]
    if not usable:
        return GroundingVerdict(
            False,
            _VERDICT_NO_SOURCE_TEXT,
            "no source-exact evidence: source_id, navigation aids, "
            "normalization, and query rewrites are never evidence",
        )

    for provenance in cited:
        try:
            provenance.validate()
        except ValueError as exc:
            return GroundingVerdict(False, _VERDICT_UNKNOWN_PROVENANCE, str(exc))
    cited_keys = {_provenance_key(provenance) for provenance in cited}
    cited_units = [unit for unit in usable if _provenance_key(unit.provenance) in cited_keys]
    if not cited:
        return GroundingVerdict(False, _VERDICT_UNKNOWN_PROVENANCE, "claim cites no provenance")
    if not cited_units:
        return GroundingVerdict(
            False,
            _VERDICT_UNKNOWN_PROVENANCE,
            "cited provenance resolves to no source-exact evidence unit",
        )

    if quote_kind is QuoteKind.EXACT_SOURCE:
        if not ru_corpus_available:
            return GroundingVerdict(
                False,
                _VERDICT_RU_CORPUS_UNAVAILABLE,
                "exact Russian quotation requires the version-pinned "
                "Russian corpus; translation fallback must be labeled",
            )
        ru_text = "".join(unit.text for unit in cited_units if unit.language == "ru")
        if not ru_text.strip():
            return GroundingVerdict(
                False,
                _VERDICT_NO_RU_SOURCE,
                "exact Russian quotation requires cited Russian source-exact text",
            )
        if not quoted_text.strip():
            return GroundingVerdict(
                False, _VERDICT_VERBATIM_MISMATCH, "exact quotation text must not be empty"
            )
        if contains_translation_label(quoted_text) or contains_translation_label(russian_claim):
            return GroundingVerdict(
                False,
                _VERDICT_TRANSLATION_AS_EXACT,
                "translation-labeled text must never pass as source-exact",
            )
        if quoted_text not in ru_text:
            return GroundingVerdict(
                False,
                _VERDICT_VERBATIM_MISMATCH,
                "exact quotation must be a verbatim substring of the cited "
                "Russian source-exact text",
            )
    else:
        display = f"{russian_claim}\n{quoted_text}"
        if not contains_translation_label(display):
            return GroundingVerdict(
                False,
                _VERDICT_TRANSLATION_UNLABELED,
                "translated quotation must carry an explicit translation label",
            )
        if not allow_translation_fallback:
            return GroundingVerdict(
                False,
                _VERDICT_TRANSLATION_FALLBACK_DISABLED,
                "translation fallback is not allowed here",
            )

    support_text = "".join(unit.text for unit in cited_units)
    judge = entails if entails is not None else default_entails
    if not judge(russian_claim, support_text):
        return GroundingVerdict(
            False,
            _VERDICT_UNSUPPORTED_INTERPRETATION,
            "cited source text does not semantically support the Russian claim",
        )

    if quote_kind is QuoteKind.EXACT_SOURCE:
        return GroundingVerdict(True, _VERDICT_OK_EXACT, "exact Russian quote grounded", True)
    return GroundingVerdict(True, _VERDICT_OK_TRANSLATION, "labeled translation grounded", False)


def evidence_kinds(evidence: Sequence[EvidenceUnit]) -> set[EvidenceKind]:
    """Return the distinct evidence kinds present (for policy tests)."""
    return {unit.kind for unit in evidence}


@dataclass(frozen=True)
class GroundingGate:
    """Configured deterministic gate owned by the Python orchestrator.

    Production defaults fail closed: without the version-pinned Russian
    corpus, exact-source Russian quotations cannot pass, and translation
    fallback passes only when explicitly allowed by the caller.
    """

    corpus_version: str
    ru_corpus_available: bool = False
    allow_translation_fallback: bool = False

    def judge(
        self,
        *,
        russian_claim: str,
        quoted_text: str = "",
        quote_kind: QuoteKind = QuoteKind.EXACT_SOURCE,
        cited: Sequence[Provenance] = (),
        evidence: Sequence[EvidenceUnit] = (),
        entails: EntailmentFn | None = None,
    ) -> GroundingVerdict:
        """Judge one Russian claim under this gate configuration."""
        if not self.corpus_version.strip():
            return GroundingVerdict(
                False, _VERDICT_UNKNOWN_PROVENANCE, "gate corpus_version must be pinned"
            )
        return check_grounding(
            russian_claim=russian_claim,
            quoted_text=quoted_text,
            quote_kind=quote_kind,
            cited=cited,
            evidence=evidence,
            ru_corpus_available=self.ru_corpus_available,
            allow_translation_fallback=self.allow_translation_fallback,
            entails=entails,
        )
