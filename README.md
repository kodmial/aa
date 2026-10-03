# AA

Public repository for the AA Telegram support bot.

## Architecture

The MVP runs as one bounded GitHub Actions job containing:

- a local OpenCode runtime configured with Zen;
- a small Python 3.12+ asyncio Telegram worker;
- outbound Telegram Bot API long polling.

OpenCode is the sole LLM boundary. The Telegram worker handles transport and orchestration.

The AA source/context and conversational policy are implemented as dedicated OpenCode agent context with deterministic source validation, source mapping, context budgeting and safety guardrails.

## Automation

Continuum core is installed from and intentionally tracks `kodmial/continuum@main`.

Implementation is tracked in repository issues #1–#9. Product-level tracking lives in `kodmial/work-lock#46`.
