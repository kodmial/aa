# AA corpus

The runtime corpus is built from the original source by **deletion only**.

## Invariant

The authoritative downloaded source is never rewritten, summarized, corrected, translated, or paraphrased.

A curated corpus may only be produced as an ordered concatenation of exact byte ranges from the canonical source. A range may be kept or dropped; text inside a kept range is immutable.

This makes the core rule mechanically enforceable:

> curated text = exact source text minus selected source spans

No LLM is part of corpus construction.

## Source acquisition

Run:

```bash
python3 scripts/fetch_aa_source.py
```

The command downloads the pinned source URLs from `corpus/source.lock.json` into `corpus/source/raw/` and records SHA-256 values in `corpus/source/fetch-state.json`.

The raw files are intentionally ignored by Git. AA World Services states that the First and Second Editions are public domain in the United States only and asks for permission for Internet postings of all editions. Because this repository is public and globally accessible, the repository stores the acquisition contract and hashes rather than republishing the complete text in Git history.

The runtime/build workspace still contains the complete downloaded original unchanged.

## Curation

`scripts/build_curated.py` accepts a byte-range manifest and copies exact ranges from a source file. It:

- verifies the source SHA-256;
- requires ordered, non-overlapping ranges;
- performs no decoding/re-encoding or text normalization;
- writes exact source bytes only;
- rejects malformed ranges.

Curation manifests should select complete sentence/paragraph spans. They must never split a sentence merely to alter its meaning.

The full original always remains available separately from the reduced runtime corpus.

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
