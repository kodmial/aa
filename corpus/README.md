# AA corpus and retrieval source

The canonical AA literary text is **immutable source authority**. The project
does not build a permanently shortened runtime edition and does not preload the
whole book into every LLM prompt.

## Source acquisition

Run:

```bash
python3 scripts/fetch_aa_source.py
```

The command downloads the source URLs pinned by `corpus/source.lock.json` into
`corpus/source/raw/` and records SHA-256 metadata in
`corpus/source/fetch-state.json`.

The raw files are intentionally ignored by Git. The repository stores the
acquisition/validation contract and hashes rather than republishing the complete
book in normal Git history.

## Canonical scope

The runtime knowledge source is limited to:

- The Doctor's Opinion;
- Chapters 1–11.

Unrelated front matter, publishing/history material, later personal stories and
unrelated appendices are outside the MVP corpus.

Issue #3 owns deterministic extraction/validation of this scope and records
source provenance and checksums.

## Derived artifacts

Later stages may derive navigation and retrieval metadata without changing the
literary source:

- stable chapter/paragraph/sentence/chunk IDs;
- exact source offsets and parent/neighbor links;
- a compact book map;
- lexical and multilingual semantic indexes;
- ranking/reranking metadata.

Every retrieval unit must map back to exact canonical source text. Generated
book-map text, embeddings, scores, contextual metadata and reranker output are
navigation aids only and must never be quoted or presented as the book.

## Runtime access

The dedicated OpenCode agent receives a compact book map and read-only project
tools:

- `book_search`;
- `book_read`;
- `book_expand`;
- bounded `book_section`.

Only passages needed for the current question are injected into the model
context. Source text returned by read/expand tools is exact; no silent
truncation or paraphrased replacement is permitted.

See `docs/aa-knowledge-architecture.md` for the architecture decision.
