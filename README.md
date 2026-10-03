# AA

Public repository for the AA Telegram support bot.

## Runtime architecture

The MVP runs as one bounded GitHub Actions job containing:

- a local OpenCode runtime configured with Zen;
- a small Python 3.12+ asyncio Telegram worker;
- outbound Telegram Bot API long polling.

OpenCode is the sole LLM boundary. The Telegram worker owns transport and orchestration only.

Local OpenCode runtime integration (loopback `opencode serve`, per-chat
sessions, failure taxonomy, persistence/recovery) is documented in
[`docs/opencode-runtime.md`](docs/opencode-runtime.md).

## Bounded live runtime

Live/test sessions are controlled from repository issue **#31** with owner-only
commands:

```text
/bot start 15m
/bot start 1h
/bot start 2h
/bot start 3h
/bot stop
/bot status
```

A start dispatches one bounded GitHub Actions job. The job bootstraps and
validates the AA knowledge artifacts, starts the local OpenCode runtime, waits
for readiness, starts Telegram long polling, and only then arms the requested
15-minute/1-hour/2-hour/3-hour live window. Duplicate runtime starts are
rejected so only one poller can own the bot token.

Telegram uses outbound `getUpdates` long polling; there is no inbound web
server. Pending Telegram updates are preserved across normal offline periods
instead of being deliberately discarded. An update offset is committed only
after application handling succeeds, reducing message loss across transient
failures and shutdowns.

The live workflow fails closed before polling if the canonical source,
hierarchical corpus, retrieval index, dedicated AA OpenCode agent, or required
read-only book tools are missing/stale. This means the runtime-control
infrastructure can exist before the remaining retrieval issues are complete
without accidentally launching an ungrounded assistant.

The user-facing OpenCode agent is `aa`, backed by
`prompts/aa-agent-system.md`. Its runtime model defaults are:

- primary: `opencode/muse-spark-1.3-contributor-free`;
- technical fallback: `opencode/space-bunny-free`.

These are AA-assistant runtime models; they do not define the model Continuum
uses to implement repository tasks.

## AA knowledge architecture

The bot uses **agentic RAG**, not a permanently shortened book and not a
full-book prompt.

The canonical AA source is kept immutable outside normal Git history and is
fetched/validated deterministically from `corpus/source.lock.json`.

At runtime OpenCode receives:

1. the behavioral/system policy;
2. a compact versioned **book map** for immediate orientation;
3. the current conversation;
4. only the exact source passages retrieved for the current question;
5. response headroom.

The external knowledge layer keeps a hierarchical source model and a
multilingual hybrid retrieval index. A Russian or English user query can locate
the English source, after which OpenCode reads the exact original text through
project-local read-only tools:

- `book_search`;
- `book_read`;
- `book_expand`;
- bounded `book_section`.

The book map and retrieval metadata are navigation aids, never evidentiary
source text. Substantive AA claims must be grounded in source-exact passages.

For broad personal/support questions, retrieval is coverage-oriented: OpenCode
uses the map to plan multiple searches across the whole book, reads exact
passages from distinct relevant regions, checks for missing perspectives, and
only then synthesizes the answer. One top hit is not treated as sufficient
grounding when the question spans multiple parts of the book.

The dedicated AA agent uses a deny-by-default tool policy: unrelated shell,
write/edit, arbitrary web and coding capabilities are not available to the
conversational agent unless a separate runtime requirement explicitly needs
them.

The complete design and decision record are in
[`docs/aa-knowledge-architecture.md`](docs/aa-knowledge-architecture.md).

## Implementation order

Repository issues are the authoritative task graph:

`#20 -> #3 -> #8 -> #17 -> #18 -> #19 -> #9 -> #5 -> #6 -> #7`

Issue #21 (deterministic emergency/medical-safety routing) starts after #20 and
may proceed in parallel with the knowledge pipeline; it must be complete before
#5.

Completed foundations:

- #1 / PR #10 — Python worker/test harness;
- #4 / PR #15 — Telegram transport;
- #2 / PR #16 — local OpenCode/Zen sessions;
- PR #11 — immutable AA source acquisition foundation.

## Automation

Continuum core is installed from and intentionally tracks
`kodmial/continuum@main`.

All implementation planning and execution for this project is tracked only in
this repository.
