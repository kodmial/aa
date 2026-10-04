# OpenCode local runtime: observed capabilities

Observed against `opencode 1.18.34` by probing a real local `opencode serve`
instance on loopback. This note records what the worker may rely on; it is
not a copy of upstream documentation.

## Invocation boundary

The narrowest stable local boundary is the `opencode serve` HTTP API on
loopback (`--port <n> --hostname 127.0.0.1`):

| Operation | Request | Notes |
|---|---|---|
| Readiness | `GET /global/health` → `{"healthy": true, "version": "..."}` | Gates Telegram traffic. |
| Create session | `POST /session` body `{"title": ...}` → `{"id": "ses_..."}` | Returns the opaque session identity. |
| Get session | `GET /session/{id}` | `404` when the id is unknown. |
| Delete session | `DELETE /session/{id}` | Permanent; `404` when unknown. |
| Prompt | `POST /session/{id}/message` body `{"parts": [{"type": "text", "text": ...}]}` | Returns `{"info": {...}, "parts": [...]}`. |
| History | `GET /session/{id}/message?limit=N` | Oldest first. |
| Abort | `POST /session/{id}/abort` | Cancels in-flight work. |

The worker uses only the standard library (`urllib` in a worker thread) for
these calls, so no second LLM client and no HTTP vendor dependency exist.

The OpenAPI document is served at `GET /doc` and was used to confirm the
shapes above; the worker depends only on the rows in this table.

## Failure shapes (observed)

- Unknown session: HTTP `404` → create a fresh session and rebind.
- Bad request (malformed payload, unknown model pointer): HTTP `400` →
  deterministic, do not retry unchanged.
- Rate limiting / concurrency: HTTP `408`/`425`/`429` → transient, retry
  with backoff.
- Server errors: HTTP `5xx` → transient.
- Provider/model failure inside a `200` prompt response: the assistant
  message carries `info.error = {"name": ..., "data": {"statusCode": int,
  "isRetryable": bool, ...}}`. `isRetryable: true` (or a `5xx` status) is
  transient; otherwise deterministic. The provider message text is never
  logged because it may echo request content.

## Session persistence and recovery (verified)

- Sessions are persisted by OpenCode in SQLite at
  `~/.local/share/opencode/opencode.db` (tables `session`, `message`,
  `part`, `session_message`, ...).
- Verified: a session created on one `opencode serve` process is still
  retrievable via `GET /session/{id}` after the server restarts on a
  different port against the same state directory.
- CLI equivalents: `opencode session list`, `opencode session delete
  <sessionID>`, `opencode export [sessionID]`, `opencode import <file>`.

Recovery contract for the worker:

1. The worker keeps only the `chat_id -> opencode session id` map in
   memory plus a reset generation counter. No external state service is
   used.
2. After a worker restart the map starts empty. The next turn for a chat
   creates a new session; if a previous session id is supplied out of band,
   the worker re-attaches with `GET /session/{id}` and falls back to
   creating a new session on `404`.
3. Reset = best-effort `DELETE` of the old remote session (a `404` there is
   not an error) + create + rebind. The old conversation can never leak
   into the new one because the id changes.

## Configuration ownership

Model/provider (Zen) configuration lives entirely on the OpenCode side
(`~/.config/opencode/opencode.jsonc` and the job environment). The AA runtime
uses the anonymous/keyless OpenCode free tier and must not require an OpenCode
or Zen account, API key, provider credential, or authentication secret. The
worker forwards at most the opaque `OPENCODE_MODEL` `provider/model` pointer.

## Concurrency and restart state (issue #5)

The runtime uses one authoritative Telegram poller and one local
``opencode serve`` process per AA worker, not one OpenCode process per
Telegram user. Each private ``chat_id`` maps to exactly one opaque OpenCode
session identity; different chats never share a session, history, pending
turn, reset generation, or mutable orchestration state.

- Turns for the same chat are strict FIFO with at most one active
  substantive turn; session create/rebind/reset and ``/new`` run inside the
  same per-chat serialization, so concurrent first messages cannot create
  competing sessions and ``/new`` replaces only the requesting chat.
- Turns for different chats may execute concurrently under a configurable
  global ``MAX_CONCURRENT_TURNS`` bound (default 4, conservative for the
  pinned OpenCode/provider/runtime) plus a bounded per-chat pending queue
  (``PER_CHAT_QUEUE_SIZE``, default 8) with safe backpressure.
- One slow chat never head-of-line blocks unrelated chats: the poller only
  enqueues and returns, while per-chat workers proceed independently.

Conversation continuity is guaranteed only for the lifetime of the current
worker/runtime; a new runtime starts a fresh OpenCode conversation per chat
and no external database is added for history.

Crash window (explicit): the transport commits an update offset once the
update is accepted into the per-chat queue. Clean shutdown stops the poller
first and then drains queued/in-flight turns (bounded). If the worker
crashes after acceptance but before the reply is sent, that in-flight turn
is lost and will not be redelivered. No exactly-once reply across crashes is
claimed without durable transactional state.

## Generation budget (issue #83)

Pinned-runtime finding (OpenCode 1.18.34, verified against `GET /doc`):
`POST /session/{id}/message` accepts only `parts`/`agent`/`model` (plus
`messageID`/`noReply`/`tools`/`format`/`system`/`variant`) and exposes no
documented per-message `max_tokens`/`maxTokens` field; the agent config
schema likewise exposes no output-token cap (only `model`/`variant`/
`temperature`/`top_p`/`prompt`/`tools`/`options`/`steps`/`maxSteps`).
Sending an undocumented token-cap field would risk being silently ignored,
so the worker deliberately does not send one: `OPENCODE_MAX_OUTPUT_TOKENS`
is honored as a bounded prompt-advertised generation budget (default 256
output tokens when unset) plus an explicit concise-answer instruction in
every synthesis prompt. Token count is never the product contract because
character/token ratios vary by language and model; the deterministic
character validator (`aa.conversation.output_limits`, hard cap 900
graphemes / 130 words, verbatim quote aggregate 300 chars, exactly one
compact regeneration, then complete-unit compaction, final transport guard)
is authoritative. Qualification fails if a `max_tokens`-shaped field ever
appears on the message boundary (see the contract test).
