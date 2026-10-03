# AA

Public repository for the AA Telegram support bot.

## Architecture

The MVP runs as one bounded GitHub Actions job containing:

- a local OpenCode runtime configured with Zen;
- a small Python 3.12+ asyncio Telegram worker;
- outbound Telegram Bot API long polling.

OpenCode is the sole LLM boundary. The Telegram worker handles transport and orchestration.

The AA source/context and conversational policy are implemented as dedicated OpenCode agent context with deterministic source validation, source mapping, context budgeting and safety guardrails.

## AA source and copyright

No AA book text ships with this repository. The operator supplies the
canonical source separately from a lawfully obtained copy via
`AA_CORPUS_PATH` (file or directory with `.txt`/`.md` sections) and pins the
edition with `AA_CORPUS_VERSION`.

Canonical baseline (validated by `aa.corpus`, owned by issue #3):

- The Doctor's Opinion plus Chapters 1–11, each required exactly once;
- edition front matter, publishing/history material, post-core personal
  stories and unrelated appendices are rejected when present as section
  headers;
- startup records checksum (SHA-256), version and token estimate, and fails
  closed on missing/duplicated/excluded sections or on context overflow —
  the corpus is never silently truncated;
- effective model context must be >= 200k tokens (`OPENCODE_CONTEXT_LIMIT_TOKENS`);
  deterministic headroom is reserved for system instructions, conversation
  and output, and only old conversation turns are compacted — never the
  source corpus;
- only checksum/version/token counts are logged, never corpus contents;
- bot quotations must stay short and source-exact; runtime corpus reduction
  is owned by #8 and conversational policy by #9.

## Safety guardrails

Acute cases (severe withdrawal, seizure, hallucination, suspected
poisoning/overdose, loss of consciousness, immediate self-harm risk) are
detected deterministically (offline EN/RU patterns in `aa.safety`, no model
call) and receive immediate emergency/medical guidance before any ordinary
AA-oriented response. The bot never provides medication dosing or
unsupervised detox instructions.

## Local development

```sh
python -m pip install -e ".[dev]"
python -m aa --check
pytest -q
ruff check .
ruff format --check .
mypy src tests
```

## Automation

Continuum core is installed from and intentionally tracks `kodmial/continuum@main`.

Implementation is tracked in repository issues #1–#9. Product-level tracking lives in `kodmial/work-lock#46`.
