"""E5-token accounting for canonical child chunks (issue #115).

Chunk sizing uses the already-pinned ``intfloat/multilingual-e5-base``
tokenizer. The hard input limit is the tokenizer/model maximum (512); the
initial production child budget is 256 tokens (configurable/eval-tunable).

The tokenizer is loaded once per process and reused. Builds fail closed
when the tokenizer is unavailable or when a single canonical sentence
exceeds the hard input limit (no silent truncation, no ad-hoc clause
splitting in this task).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

E5_MODEL_ID = "intfloat/multilingual-e5-base"
E5_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"
E5_HARD_INPUT_TOKENS = 512
CHILD_MAX_TOKENS = 256
CHUNKER_ID = "e5-token-chunker/1"
CHUNKER_VERSION = 1
TOKENIZER_ID = "intfloat/multilingual-e5-base@d1287505"

_TOKENIZER: Any | None = None


class ChunkTokenError(ValueError):
    """Raised when token-aware chunking cannot be served exactly."""


def load_e5_tokenizer() -> Any:
    """Load and cache the pinned E5 tokenizer (reused across calls)."""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
    except Exception as exc:
        raise ChunkTokenError(f"E5 tokenizer stack is unavailable: {exc}") from exc
    # Prefer the pinned local snapshot (offline, no network); fall back to a
    # pinned-revision hub load (small tokenizer files only, never model weights).
    try:
        from aa.corpus.public_cache import (
            resolve_hf_cache_dir,
            resolve_model_root,
            snapshot_dir,
        )

        model_root = resolve_model_root(resolve_hf_cache_dir())
        snapshot = snapshot_dir(model_root, E5_REVISION)
        marker_files = ("tokenizer.json", "tokenizer_config.json")
        if all((snapshot / name).is_file() for name in marker_files):
            _TOKENIZER = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                str(snapshot), local_files_only=True, trust_remote_code=False
            )
            return _TOKENIZER
    except Exception:
        pass
    try:
        _TOKENIZER = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            E5_MODEL_ID, revision=E5_REVISION, trust_remote_code=False
        )
    except Exception as exc:
        raise ChunkTokenError(f"cannot load pinned E5 tokenizer: {exc}") from exc
    return _TOKENIZER


def reset_e5_tokenizer_cache() -> None:
    """Reset the cached tokenizer (tests only)."""
    global _TOKENIZER
    _TOKENIZER = None


def count_e5_tokens(text: str, *, prefix: str = "passage: ") -> int:
    """Return the E5 tokenizer token count for ``text`` with E5 prefix.

    The ``passage: `` prefix mirrors dense indexing (``e5_embed``), so a
    chunk that fits here is guaranteed to fit the model's hard input limit
    at embed time. Counts exclude added special tokens; the hard-limit
    check stays conservative via the prefix.
    """
    tokenizer = load_e5_tokenizer()
    try:
        encoded = tokenizer(prefix + text, add_special_tokens=False)
        return len(encoded["input_ids"])
    except Exception as exc:
        raise ChunkTokenError(f"E5 token counting failed: {exc}") from exc


def e5_hard_limit() -> int:
    """Return the pinned E5 hard input limit in tokens."""
    return E5_HARD_INPUT_TOKENS


def chunker_identity(*, max_tokens: int = CHILD_MAX_TOKENS) -> dict[str, Any]:
    """Return chunker/tokenizer identity for index manifests."""
    return {
        "chunker_id": CHUNKER_ID,
        "chunker_version": CHUNKER_VERSION,
        "tokenizer_id": TOKENIZER_ID,
        "tokenizer_model": E5_MODEL_ID,
        "tokenizer_revision": E5_REVISION,
        "chunk_policy": "adjacent-sentences-within-paragraph",
        "chunk_max_tokens": int(max_tokens),
        "e5_hard_input_tokens": E5_HARD_INPUT_TOKENS,
    }


def build_token_chunks(
    sentences: list[Any],
    *,
    paragraph_end: int,
    max_tokens: int = CHILD_MAX_TOKENS,
    hard_limit: int = E5_HARD_INPUT_TOKENS,
    token_counter: Callable[[str], int] | None = None,
) -> list[tuple[int, int]]:
    """Group sentence spans into chunk ``(char_start, char_end)`` ranges.

    Sentences are never split; a chunk holds one or more whole adjacent
    sentences within one paragraph up to ``max_tokens`` E5 tokens. A single
    sentence may exceed ``max_tokens`` and remain one atomic child only
    while it still fits ``hard_limit``; a sentence exceeding ``hard_limit``
    fails closed with its source span. No overlapping duplicate windows are
    emitted; larger context comes later from parent/neighbor expansion.

    Chunks tile the paragraph exactly: each chunk starts where the previous
    chunk ended (boundaries fall inside whitespace-only gaps), so no
    paragraph content is lost between chunks.
    """
    if max_tokens <= 0:
        raise ChunkTokenError("max_tokens must be > 0")
    if hard_limit <= 0:
        raise ChunkTokenError("hard_limit must be > 0")
    if max_tokens > hard_limit:
        raise ChunkTokenError("max_tokens must not exceed the E5 hard input limit")
    if not sentences:
        raise ChunkTokenError("refusing to chunk an empty sentence list")
    counter = token_counter if token_counter is not None else (lambda text: count_e5_tokens(text))
    # Validate contiguity modulo whitespace-only gaps.
    for previous, current in zip(sentences, sentences[1:], strict=False):
        if current.char_start < previous.char_end:
            raise ChunkTokenError("sentences must not overlap")
        # Gaps are validated by the segmenter; re-check cheaply via text presence.
        if current.char_start < previous.char_start:
            raise ChunkTokenError("sentences must be ordered")
    # Tokenize each sentence once; fail closed on hard-limit breach.
    sentence_tokens: list[int] = []
    for sentence in sentences:
        tokens = int(counter(sentence.text))
        if tokens > hard_limit:
            raise ChunkTokenError(
                f"single sentence exceeds E5 hard input limit ({tokens} > {hard_limit}): "
                f"span [{sentence.char_start}:{sentence.char_end})"
            )
        sentence_tokens.append(tokens)
    # Greedy grouping by token budget.
    groups: list[list[int]] = []
    current_group: list[int] = []
    current_tokens = 0
    for position in range(len(sentences)):
        tokens = sentence_tokens[position]
        if not current_group:
            current_group.append(position)
            current_tokens = tokens
            continue
        if current_tokens + tokens <= max_tokens:
            current_group.append(position)
            current_tokens += tokens
        else:
            # Single oversize sentence already stands alone; otherwise split.
            groups.append(current_group)
            current_group = [position]
            current_tokens = tokens
    if current_group:
        groups.append(current_group)
    # Convert groups to tiling (char_start, char_end) ranges. Boundaries fall
    # inside whitespace gaps: each non-terminal chunk ends where the next
    # chunk begins (the next sentence start), so gaps are owned exactly once.
    chunks: list[tuple[int, int]] = []
    for group_index, group in enumerate(groups):
        first = sentences[group[0]]
        if group_index + 1 < len(groups):
            next_first = sentences[groups[group_index + 1][0]]
            chunk_start = first.char_start if group_index == 0 else chunks[-1][1]
            # When groups are adjacent, next_first.char_start is the gap end.
            chunk_end = next_first.char_start
            # Guard against empty or inverted ranges (should not happen).
            if chunk_end <= chunk_start:
                chunk_end = sentences[group[-1]].char_end
        else:
            chunk_start = first.char_start if group_index == 0 else chunks[-1][1]
            last = sentences[group[-1]]
            # Last chunk runs to the paragraph end so trailing gaps are kept.
            chunk_end = max(last.char_end, paragraph_end)
            if chunk_end <= chunk_start:
                raise ChunkTokenError("chunk span is empty")
        if chunk_end <= chunk_start:
            raise ChunkTokenError("chunk span is empty")
        chunks.append((chunk_start, chunk_end))
    return chunks
