# OpenCode local runtime: observed capabilities

Observed against the pinned `opencode 2.2.6` binary by probing a real local
`opencode serve` instance on loopback and reading its `GET /openapi.json`
document. This note records what the worker may rely on; it is not a copy of
upstream documentation. The pre-2.0 (v1) routes are retained only for
historical fixtures and are not used by the production runtime.

## Invocation boundary

The narrowest stable local boundary is the `opencode serve` HTTP API on
loopback (`--port <n> --hostname 127.0.0.1`). Every request carries HTTP
Basic auth with the fixed `opencode` username and an explicit
`OPENCODE_PASSWORD`; the managed spawn passes the password through the child
environment, never through command-line flags, and the secret is never
logged.

| Operation | Request | Notes |
|---|---|---|
| Readiness | `GET /api/info` → `{"version": "...", "pid": ...}` | Gates Telegram traffic; `401`/non-JSON is unhealthy. |
| Create session | `POST /api/session` body `{"title": ...}` → `{"data": {"id": "ses_...", ...}}` | Returns the opaque session identity. |
| Get session | `GET /api/session/{id}` → `{"data": {...}}` | `404` when the id is unknown. |
| Delete session | `DELETE /api/session/{id}` | Permanent; `404` when unknown. |
| Pin selectors | `POST /api/session/{id}/model` body `{"model": {"providerID", "id"}}`; `POST /api/session/{id}/agent` body `{"agent": ...}` | Applied before the prompt; `204` on success. |
| Prompt (enqueue) | `POST /api/session/{id}/prompt` body `{"text": ...}` → `{"data": {"id", "type": "user", ...}}` | Durable admission only; **never** a reply. |
| Await idle | `POST /api/experimental/session/{id}/wait` | `204` when the agent loop is idle. |
| History | `GET /api/session/{id}/message?limit=N&order=desc` → `{"data": [...], "cursor": {"previous", "next"}}` | Newest first; follow `cursor.next` (never combine with `order`). |
| Interrupt | `POST /api/session/{id}/interrupt` → `{"interrupted": bool}` | `true` only when an active execution was interrupted. |

The worker uses only the standard library (`urllib` in a worker thread) for
these calls, so no second LLM client and no HTTP vendor dependency exist.

The v2 success bodies are mostly wrapped in a single-field `{"data": ...}`
envelope; the client unwraps only that exact shape and leaves paged
responses (`data` plus `cursor`) intact for the caller.

## Prompt completion contract (v2)

A `200` from the prompt route proves the input was durably admitted, not that
the model has replied. The client therefore:

1. records the admitted user-message `id` and `time.created`;
2. calls `/wait`, then reads history newest-first and stops at that user
   message, so a previous turn's assistant/idle message can never be
   mistaken for this turn;
3. treats completion as proven only by a terminal `idle` message created
   after this turn's user message, whose `outcome` is `succeeded`;
4. re-issues `/wait` and polls until that condition holds, bounded by the
   caller's timeout (or a 600 s finite fallback), then raises a timeout
   rather than inventing a reply;
5. classifies a non-`succeeded` idle message from its structured `error`
   (status taxonomy only; provider message text is never logged);
6. requires a completed assistant message with `finish: "stop"` and extracts
   its `content[type=text]` parts.

Structured output has no native `format` field on the v2 prompt route, so
the client sends an explicit JSON-Schema-only instruction, parses the reply
strictly (bare object or one fenced JSON block), and Pydantic-validates it in
the caller.

## Failure shapes (observed)

- Unknown session: HTTP `404` → create a fresh session and rebind.
- Bad request (malformed payload, unknown model pointer): HTTP `400` →
  deterministic, do not retry unchanged.
- Unauthorized (missing/wrong `OPENCODE_PASSWORD`): HTTP `401` →
  deterministic and fail closed; the response body is never surfaced.
- Rate limiting / concurrency: HTTP `408`/`425`/`429` → transient, retry
  with backoff.
- Server errors / unavailable wait: HTTP `5xx` (including `503`) →
  transient.
- Provider/model failure inside a completed turn: the terminal `idle`
  message carries `outcome` (for example `failed`) and a structured
  `error = {"type": ..., "message": ..., "status": int}`. The status is
  classified with the same taxonomy; the provider message text is ignored
  because it may echo request content.
- A non-JSON body (for example an HTML fallback) or an unknown v2 route is a
  deterministic contract mismatch, never a usable protocol response.

## Session persistence and recovery (verified)

- Sessions are persisted by OpenCode in SQLite at
  `~/.local/share/opencode/opencode.db` (tables `session`, `message`,
  `part`, `session_message`, ...); the v2 routes read and write the same
  store.
- Verified on the pinned `2.2.6` binary: a session created by one
  `opencode serve` process is still retrievable via
  `GET /api/session/{id}` after a fresh server process starts against the
  same state directory, and its paged history still contains the completed
  turn. On `404` the worker falls back to creating a fresh session.
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

Pinned-runtime finding (OpenCode 2.2.6, verified against `GET /openapi.json`):
`POST /api/session/{id}/prompt` accepts only `id`/`text`/`files`/`agents`/
`skills`/`metadata`/`delivery`/`resume` and exposes no documented
per-message `max_tokens`/`maxTokens` field; the agent config schema likewise
exposes no output-token cap (only `model`/`variant`/`temperature`/`top_p`/
`prompt`/`tools`/`options`/`steps`/`maxSteps`).
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
