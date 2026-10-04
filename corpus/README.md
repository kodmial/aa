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

## Russian canonical build (issue #50)

The Russian Fourth Edition pipeline mirrors the English one but its single
canonical input is the normalized TXT `corpus/source/raw-ru/aa-big-book.txt`
(ignored by Git). PDF extraction and OCR are never part of the canonical path.

- `scripts/fetch_ru_source.py` reuses the TXT in the trusted runtime, or
  assembles it once via `--bootstrap-from-provider` from the text-native
  provider pages pinned by `corpus/source.ru.lock.json`, recording SHA-256
  in the ignored `corpus/source/fetch-ru-state.json`;
- `scripts/build_canonical_ru.py` reads only that TXT: deterministic UTF-8
  decoding (BOM stripped), NFC/newline transport normalization, Fourth
  Edition / 2013 / ISBN `978-5-906531-01-8` identity validation, exact
  Мнение доктора + Chapters 1-11 order/boundaries/headings/control passages,
  then writes `corpus/generated/canonical.ru.json` and
  `corpus/generated/canonical.ru.txt` (both ignored by Git) after verifying
  every section SHA and both artifact SHAs against the committed
  `corpus/canonical.ru.manifest.json`.

```bash
python3 scripts/fetch_ru_source.py --bootstrap-from-provider
python3 scripts/build_canonical_ru.py
```

See `docs/russian-corpus.md` for the full contract.

## Encrypted snapshot and restore (issues #24, #50)

The reproducible encrypted snapshots in `corpus/source/encrypted/` are the
durable cross-run cache for this small corpus. Plaintext book text and
plaintext retrieval indexes are never stored in Git or in GitHub Actions
cache. `scripts/restore_canonical.py` is the single restore entry point
(`--lang ru` selects the Russian artifact):
it reuses a valid `corpus/generated/canonical.json`, otherwise decrypts
the committed snapshot with `AA_BOOK_AGE_IDENTITY`, otherwise falls back
to the deterministic #3 fetch/build only when explicitly allowed, and
fails closed otherwise. The Russian counterpart
(`corpus/generated/canonical.ru.json` via `canonical.ru.tar.zst.age` +
`metadata.ru.json`, same age recipient, no second key) works the same way.
Production activation is deferred to #28. See
`docs/encrypted-snapshot.md`.

## Derived artifacts

Later stages may derive navigation and retrieval metadata without changing the
literary source:

- stable chapter/paragraph/sentence/chunk IDs;
- exact source offsets and parent/neighbor links;
- a compact book map;
- lexical and multilingual semantic indexes;
- ranking/reranking metadata.

Issue #8 builds the aligned RU/EN hierarchy with
`scripts/build_corpus_structure.py` (`src/aa/corpus/structure.py`):

- `corpus/structure.json` — public metadata-only aligned structure
  (language-neutral section ids, bilingual display titles, concise English
  topics, provenance/checksum references, section-only alignment policy);
- `corpus/book-map.md` — public compact primarily-English routing map
  (navigation only, within the 6000-token budget);
- `corpus/generated/corpus_structure.json` — private text-bearing full
  hierarchy (per-language paragraphs/sentences/chunks with exact offsets,
  parent/previous/next links, and RU-primary retrieval roles; ignored by Git).

Section alignment is mandatory; paragraph/chunk alignment is never forced
across languages. Every chunk round-trips to exact source text.

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
