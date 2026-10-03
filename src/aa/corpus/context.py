"""AA corpus/context provider boundary.

Loads the canonical AA source snapshot from ``AA_CORPUS_PATH`` with
deterministic validation (chapter boundaries, required/excluded sections,
checksum, token count) and enforces the >=200k context-budget contract
without ever silently truncating the corpus.

Copyright handling: no AA book text ships with this repository. The operator
provides the canonical source from a lawfully obtained copy. Only
checksum/version/token counts are exposed for logging; corpus contents are
never logged.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from aa.corpus.budget import (
    DEFAULT_RESERVED_CONVERSATION_TOKENS,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    DEFAULT_RESERVED_SYSTEM_TOKENS,
    MIN_EFFECTIVE_CONTEXT_TOKENS,
    ContextBudget,
    ContextBudgetError,
)
from aa.corpus.canonical import (
    REQUIRED_SECTIONS,
    CanonicalCorpus,
    CorpusValidationError,
    load_canonical_corpus,
)


@dataclass(frozen=True)
class CorpusInfo:
    """Describes the resolved AA corpus snapshot (metadata only)."""

    path: str
    version: str
    checksum_sha256: str = ""
    token_estimate: int = 0
    section_count: int = 0


class CorpusContext:
    """Validated canonical corpus held in the stable session prefix.

    The complete corpus stays resident once loaded. Old conversation turns
    may be compacted elsewhere; this class never drops source sections.
    When ``AA_CORPUS_PATH`` points at a real source snapshot, startup fails
    closed (raises) when validation or the context budget fails, so the
    worker never runs with a silently truncated corpus. When the path is
    absent (for example a local ``--check`` without a lawfully obtained
    copy), the context loads in unvalidated pointer mode with
    ``validated == False`` so the foundation boot path stays green; bot
    sessions must call :meth:`ensure_validated` before serving traffic.
    """

    def __init__(
        self,
        path: str,
        version: str,
        *,
        effective_context_tokens: int = MIN_EFFECTIVE_CONTEXT_TOKENS,
        reserved_system_tokens: int = DEFAULT_RESERVED_SYSTEM_TOKENS,
        reserved_conversation_tokens: int = DEFAULT_RESERVED_CONVERSATION_TOKENS,
        reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
    ) -> None:
        self._path = path
        self._version = version
        self._effective_context_tokens = effective_context_tokens
        self._reserved_system_tokens = reserved_system_tokens
        self._reserved_conversation_tokens = reserved_conversation_tokens
        self._reserved_output_tokens = reserved_output_tokens
        self._corpus: CanonicalCorpus | None = None
        self._budget: ContextBudget | None = None
        self._loaded = False
        self._validated = False

    @staticmethod
    def _path_has_source(path: str) -> bool:
        candidate = pathlib.Path(path)
        if candidate.is_file():
            return True
        if candidate.is_dir():
            return any(
                p.is_file() and p.suffix.lower() in {".txt", ".md"} for p in candidate.iterdir()
            )
        return False

    async def load(self) -> CorpusInfo:
        """Validate/pin the corpus when present, else pointer mode."""
        if not self._path_has_source(self._path):
            # No operator-provided source present: stay in pointer mode so
            # local boot/CI without the copyrighted text still succeeds.
            self._corpus = None
            self._budget = None
            self._validated = False
            self._loaded = True
            return self.info
        corpus = load_canonical_corpus(self._path, source_version=self._version)
        budget = ContextBudget(
            effective_context_tokens=self._effective_context_tokens,
            corpus_tokens=corpus.token_estimate,
            reserved_system_tokens=self._reserved_system_tokens,
            reserved_conversation_tokens=self._reserved_conversation_tokens,
            reserved_output_tokens=self._reserved_output_tokens,
        )
        try:
            budget.validate()
        except ContextBudgetError as exc:
            raise ContextBudgetError(f"{exc}; refusing to truncate the canonical corpus") from exc
        self._corpus = corpus
        self._budget = budget
        self._validated = True
        self._loaded = True
        return self.info

    def ensure_validated(self) -> CanonicalCorpus:
        """Return the validated corpus or raise when unvalidated."""
        if not self._validated or self._corpus is None:
            raise CorpusValidationError(
                f"canonical corpus at {self._path!r} is not validated; "
                "provide AA_CORPUS_PATH with the required 12 sections"
            )
        return self._corpus

    async def unload(self) -> None:
        """Release corpus resources."""
        self._corpus = None
        self._budget = None
        self._loaded = False
        self._validated = False

    @property
    def loaded(self) -> bool:
        """Whether the corpus context is loaded."""
        return self._loaded

    @property
    def validated(self) -> bool:
        """Whether a canonical source was strictly validated."""
        return self._validated

    @property
    def info(self) -> CorpusInfo:
        """Return the corpus pointer plus validated metadata (no contents)."""
        if self._corpus is not None:
            return CorpusInfo(
                path=self._path,
                version=self._version,
                checksum_sha256=self._corpus.checksum_sha256,
                token_estimate=self._corpus.token_estimate,
                section_count=len(self._corpus.sections),
            )
        return CorpusInfo(path=self._path, version=self._version)

    @property
    def corpus(self) -> CanonicalCorpus | None:
        """Return the validated corpus (in-memory stable prefix)."""
        return self._corpus

    @property
    def budget(self) -> ContextBudget | None:
        """Return the validated context budget, if loaded."""
        return self._budget

    def required_section_ids(self) -> tuple[str, ...]:
        """Return canonical required section ids in order."""
        return tuple(section.section_id for section in REQUIRED_SECTIONS)

    def to_safe_dict(self) -> dict[str, object]:
        """Return log-safe metadata; never includes corpus contents."""
        info = self.info
        payload: dict[str, object] = {
            "path": info.path,
            "version": info.version,
            "checksum_sha256": info.checksum_sha256,
            "token_estimate": info.token_estimate,
            "section_count": info.section_count,
            "loaded": self._loaded,
            "validated": self._validated,
            "min_effective_context_tokens": MIN_EFFECTIVE_CONTEXT_TOKENS,
        }
        if self._budget is not None:
            payload["budget"] = self._budget.to_safe_dict()
        return payload


__all__ = [
    "CorpusContext",
    "CorpusInfo",
    "CorpusValidationError",
    "ContextBudgetError",
    "MIN_EFFECTIVE_CONTEXT_TOKENS",
]
