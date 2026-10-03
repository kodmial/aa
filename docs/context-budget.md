# Runtime context budget

This note records the measured OpenCode/Zen context and the explicit budgets
used by the hierarchical retrieval system (issue #3). The steady LLM context
contains only bounded parts; the full canonical book is never always-loaded.

## Effective context (measured)

Zen serves models whose input tiers are explicitly split at 200K tokens: the
Claude Sonnet/Opus and Gemini families price `<= 200K` input separately from
`> 200K` input. `200_000` tokens is therefore the conservative effective
context that is safe regardless of which Zen model backs the session.

- `EFFECTIVE_CONTEXT_TOKENS_DEFAULT = 200_000` (`src/aa/corpus/budget.py`).
- A deployment with a known-larger model may set
  `OPENCODE_CONTEXT_LIMIT_TOKENS`; `resolve_effective_context()` honours a
  positive value and otherwise keeps the 200K floor.
- Token estimates use `ceil(chars / 4)`, a conservative upper bound for
  English prose, so budgets err on the side of reserving too much.

## Canonical corpus (measured 2026-10-03)

Built by `scripts/build_canonical.py` from the PR #11 acquisition path
(`scripts/fetch_aa_source.py` + `corpus/source/raw/`, validated against
`corpus/canonical.manifest.json`). The artifact (`corpus/generated/`,
ignored by Git) holds exactly The Doctor's Opinion + Chapters 1-11.

| Section | Chars | Est. tokens |
|---|---|---|
| doctors-opinion | 12434 | 3109 |
| chapter-1 | 25544 | 6386 |
| chapter-2 | 20684 | 5171 |
| chapter-3 | 22902 | 5726 |
| chapter-4 | 22027 | 5507 |
| chapter-5 | 21138 | 5285 |
| chapter-6 | 27180 | 6795 |
| chapter-7 | 24261 | 6066 |
| chapter-8 | 28359 | 7090 |
| chapter-9 | 22837 | 5710 |
| chapter-10 | 23122 | 5781 |
| chapter-11 | 21944 | 5486 |
| **Total** | **272432** | **68108** |

Raw sources: `AA.txt` 260330 bytes (`sha256:7cef8d86…1788`),
`doctors-opinion.html` 48912 bytes (`sha256:311ebcc8…dda1d`).
Artifact `canonical.json`: 281291 bytes,
`sha256:7bd1b398…ed7463b` (pinned in the manifest).

## Budgets (tokens)

| Category | Budget | Notes |
|---|---|---|
| system/policy | 6000 | behaviour + policy prompt |
| compact book map | 6000 | navigation only, never evidence |
| retrieved source passages | 16000 | dynamic, per-need passages |
| conversation history | 32000 | rolling session state |
| response headroom | 16000 | completion reservation |
| **Reserved** | **76000** | enforced by `ContextBudget.validate()` |
| spare (wrappers/overhead) | 124000 | `200000 - 76000` |

The full canonical text (~68108 est. tokens) is larger than the combined
retrieved + map budgets and nearly equals the whole reservation: always
loading it would starve history and headroom and break smaller effective
limits. Retrieval therefore loads only ranked passages per turn
(~16000-token cap fits, e.g., four ~4000-token passages), keeping steady
context far below the effective limit by construction.

## No silent truncation

- `fit_passages()` packs passages atomically: the full ordered set must fit
  the retrieved-passages budget or it raises `TruncationRefusedError`.
  Callers re-rank and explicitly request a smaller set; passages are never
  cut and trailing passages are never silently dropped.
- `CanonicalCorpus.read()` enforces exact `[start:end)` bounds and raises
  `CanonicalRangeError` on any out-of-range or empty read.
- A single passage larger than the budget is refused (it must be re-chunked
  upstream at sentence boundaries, never truncated here).

## Reproduce

```bash
python3 scripts/fetch_aa_source.py
python3 scripts/build_canonical.py
pytest tests/test_canonical_build.py tests/test_context_budget.py -q
```
