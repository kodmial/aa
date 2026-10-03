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

## Canonical build (The Doctor's Opinion + Chapters 1-11)

`scripts/build_canonical.py` is the single deterministic builder for the
canonical runtime artifact. It reuses the acquisition path above and never
downloads anything itself:

- reads `corpus/source/raw/AA.txt` and `corpus/source/raw/doctors-opinion.html` read-only;
- validates both files against `corpus/source/fetch-state.json` and the
  committed `corpus/canonical.manifest.json` (SHA-256, byte lengths, paths);
- converts The Doctor's Opinion HTML to readable literary text
  deterministically (article paragraphs `p1`..`p41` in order, entities
  unescaped, wording untouched);
- slices Chapters 1-11 contiguously from the plain-text source, validating
  chapter numbers, titles, order and boundaries while excluding the
  publisher preamble;
- verifies every derived section SHA and the whole-artifact SHA against the
  manifest, then writes `corpus/generated/canonical.json` (ignored by Git).

Any stale or mismatched source fails closed with a non-zero exit status and
no artifact is written. Rebuilding while sources are unchanged produces
byte-identical output.

Run:

```bash
python3 scripts/fetch_aa_source.py
python3 scripts/build_canonical.py
```

Runtime context budgets for the retrieval system are documented in
`docs/context-budget.md`.

## Encrypted snapshot and restore (issue #24)

The reproducible encrypted snapshot in `corpus/source/encrypted/` is the
durable cross-run cache for this small corpus. Plaintext book text and
plaintext retrieval indexes are never stored in Git or in GitHub Actions
cache. `scripts/restore_canonical.py` is the single restore entry point:
it reuses a valid `corpus/generated/canonical.json`, otherwise decrypts
the committed snapshot with `AA_BOOK_AGE_IDENTITY`, otherwise falls back
to the deterministic #3 fetch/build only when explicitly allowed, and
fails closed otherwise. Production activation is deferred to #28. See
`docs/encrypted-snapshot.md`.

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
