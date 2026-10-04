# RU/EN context-cost measurement (issue #49)

Measured on the pinned AA runtime — OpenCode `1.18.34`, model
`opencode/muse-spark-1.3-contributor-free` — via
`scripts/measure_context_cost.py` (2026-10-04). No generic tokenizer was
used: every number below is provider/runtime-reported input-token usage
for controlled requests that differ only by language payload, with
wrappers, model, agent and reasoning settings held constant.

## Method

- A local `opencode serve` (pinned version) receives one probe per fresh
  session: `POST /session` then `POST /session/{id}/message` with
  `{"parts": [{"type": "text", ...}]}` plus the pinned model pointer.
- The stable prompt-side metric is `input + cache.read + cache.write`
  from `info.tokens`. Repeated identical requests split that sum between
  `input` and `cache.read` nondeterministically, but the sum is exactly
  stable per payload (all 5 repeats identical in every case below, so
  median == p95 everywhere).
- Every payload carries the same constant instruction wrapper, so deltas
  against the shared baseline isolate the payload's token cost
  (incremental-delta method; per-category isolation is unavailable from
  provider accounting).
- "Chars/token" below uses net payload chars (wrapper subtracted).
- Fallback spot-check uses `opencode/space-bunny-free` (3 repeats).

## Runtime constraints found

- The dedicated `aa` agent (deny-by-default `"*": "deny"`) is rejected by
  the pinned Muse Spark free tier with 403 `FreeTierError`, while the
  server-default agent on the identical model succeeds. Measurement
  therefore uses the default agent on the same pinned model/tokenizer
  path and carries the authoritative system prompt as a measured payload.
  The deny-by-default agent needs a model-permission qualification before
  it can run on the primary model (out of scope for #49).
- `opencode export` session totals confirm the same accounting
  independently of the per-message boundary.

## Results (primary model, baseline 8367 tokens, 5 repeats)

| Case | Lang | Chars | Median | p95 | Delta | Net chars/token |
|---|---|---|---|---|---|---|
| baseline | mixed | 84 | 8367 | 8367 | 0 | n/a |
| system_full_en | en | 4161 | 9134 | 9134 | 767 | 5.32 |
| system_sample_en | en | 1147 | 8574 | 8574 | 207 | 5.14 |
| system_sample_ru | ru | 1224 | 8655 | 8655 | 288 | 3.96 |
| book_map_en | en | 521 | 8482 | 8482 | 115 | 3.80 |
| book_map_ru | ru | 513 | 8506 | 8506 | 139 | 3.09 |
| evidence_pack_en | en | 563 | 8476 | 8476 | 109 | 4.39 |
| evidence_pack_ru | ru | 599 | 8494 | 8494 | 127 | 4.06 |
| history_short_ru | ru | 255 | 8404 | 8404 | 37 | 4.62 |
| history_medium_ru | ru | 745 | 8535 | 8535 | 168 | 3.93 |
| history_long_ru | ru | 1539 | 8721 | 8721 | 354 | 4.11 |
| planner_wrapper_en | en | 379 | 8456 | 8456 | 89 | 3.31 |
| planner_wrapper_ru | ru | 380 | 8465 | 8465 | 98 | 3.02 |
| turn_short | mixed | 547 | 8471 | 8471 | 104 | 4.45 |
| turn_medium | mixed | 1675 | 8765 | 8765 | 398 | 4.00 |
| turn_broad | mixed | 1968 | 8844 | 8844 | 477 | 3.95 |

## Results (fallback spot-check, baseline 8406 tokens, 3 repeats)

| Case | Lang | Chars | Median | p95 | Delta | Net chars/token |
|---|---|---|---|---|---|---|
| book_map_en | en | 521 | 8520 | 8520 | 114 | 3.83 |
| book_map_ru | ru | 513 | 8567 | 8567 | 161 | 2.66 |
| evidence_pack_en | en | 563 | 8513 | 8513 | 107 | 4.48 |
| evidence_pack_ru | ru | 599 | 8582 | 8582 | 176 | 2.93 |

## RU/EN cost ratios (same semantics, token deltas)

| Pair | Primary | Fallback |
|---|---|---|
| system sample | RU +39% | n/a |
| book map | RU +21% | RU +41% |
| evidence pack | RU +17% | RU +64% |
| planner wrapper | RU +10% | n/a |

Russian is consistently more expensive per meaning, and the penalty grows
on the fallback model. Dense structured content (planner JSON, map rows)
is the most token-dense per character in both languages.

## Estimator calibration (applied in `src/aa/corpus/budget.py`)

- The old `ceil(chars / 4)` undercounts real payloads: English planner
  JSON (3.31), English map slice (3.80), Russian medium history (3.93)
  and the broad turn (3.95) all cost more than `chars / 4` predicts.
- `CHARS_PER_TOKEN = 3` is a true ceiling for every primary-model
  payload (measured floor 3.02) and every fallback-English payload.
- `RU_CHARS_PER_TOKEN = 2` is the ceiling for Cyrillic text on both
  models (measured floor 2.66 on fallback dense Russian evidence).
- `estimate_text_tokens()` budgets Cyrillic spans at the Russian rate
  and all other spans at the English rate; conversation history must use
  it because history arrives in the user's language.

## Language decision (from this evidence, not assumptions)

- System/policy stays English: the authoritative prompt costs 767
  tokens (far inside the 6000 budget); the equivalent Russian sample
  costs +39% more for identical semantics and repeats every turn.
- Compact book map stays English: navigation-only, always loaded, and
  Russian costs +21% (primary) to +41% (fallback) for the same rows.
- Planner/tool wrappers stay English for internal keys/structure;
  only user-visible values may be Russian (+10% is small but steady).
- Evidence stays canonical English: exact source text is authoritative
  by contract, and Russian translation costs +17% to +64% more.
- Conversation history and user-facing answers follow the user's
  language (Russian budgeted at the Russian rate).

## Reproduce

```bash
opencode serve --port 4096 --hostname 127.0.0.1  # pinned 1.18.34
python3 scripts/measure_context_cost.py --base-url http://127.0.0.1:4096 \
  --json-out /tmp/context-cost.json
```

Fixtures live in `src/aa/corpus/token_measurement.py`; statistics and
accounting extraction are covered by `tests/test_token_measurement.py`.
