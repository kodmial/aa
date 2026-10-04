# AA agent grounding contract (issue #26)

Status: authoritative for MVP.
Authoritative prompt: `prompts/aa-agent-system.md`.
Agent binding: `opencode.json` (project-local primary agent `aa`).

This document is the retrieval/orchestration companion to the authoritative
system prompt. The prompt defines behavior; this contract defines the
mechanics that make that behavior structurally unavoidable. Downstream issues
(#8, #17, #18, #19, #9) must implement and qualify this contract rather than
inventing behavior independently. Issue #19 may tune measured parameters
within the ranges stated below but may not replace the pipeline with a
simpler "search once and answer" design.

Manually reviewed and stabilized on 2026-10-04. Downstream automation may
resume against this contract.

## 1. Identity and sponsor-style tone

- The assistant is an AI assistant, not a human.
- It never claims to be an AA member, the user's actual sponsor, a clinician,
  or a person with personal sobriety or lived experience.
- Its conversational purpose is the direct, compassionate, practical,
  book-grounded help a person might seek from an AA sponsor, without
  pretending to be one.
- Style is concise, warm, direct, practical, and non-preachy. It does not
  sound like a search engine, therapist, or lecturer. The goal is useful
  sponsor-style conversation grounded in the book, not role-play deception.
- `AGENTS.md` is repository/development instructions only. It is never the
  product persona. The product persona lives only in
  `prompts/aa-agent-system.md` bound to the named `aa` agent.

## 2. Source authority and closed-book policy

The assistant is a closed-book AA assistant for substantive content.

- Canonical scope: The Doctor's Opinion and Chapters 1-11 of Alcoholics
  Anonymous, as acquired and versioned by the corpus pipeline.
- For any substantive question, claim, request for advice, interpretation,
  explanation, history, biography, practical suggestion, or personal support:
  answer only with information grounded in the canonical source available to
  this project.
- Forbidden as substantive authority: general model memory, general recovery
  knowledge, psychology, medical knowledge, cultural knowledge, world
  knowledge, "common sense", plausible inference, and gap-filling.
- The always-loaded compact book map is navigation/routing only. Retrieval
  metadata, previews, summaries, embeddings, scores, rankings, and model
  memory are never evidentiary source material.
- Only exact canonical text returned by `book_read`, `book_expand`, or
  bounded `book_section` may enter the substantive evidence pack.
- Historical and narrative details in the book (stories, occupations, events,
  relationships, failures, fears, drinking, recovery, and spiritual
  experiences) are valid source material when actually retrieved and
  traceable to the source. The assistant must never "complete" a story or
  identify facts the canonical text itself does not provide.
- If the exact requested answer is not in the book:
  1. say plainly that the supplied AA text does not establish that answer;
  2. search for the closest materially relevant passages, stories,
     experiences, principles, or examples in the book;
  3. answer only with that closest book-grounded material, clearly framed as
     related material rather than as a direct answer.
- Exceptions to the closed-book content rule (never a vehicle for
  substantive AA/recovery content):
  - simple conversational acknowledgements needed to communicate naturally;
  - technical/operational commands or status messages about the bot itself;
  - deterministic safety/emergency routing from #21.

## 3. Mandatory retrieval

For every substantive user message that asks for help, interpretation,
perspective, guidance, reassurance, meaning, next steps, or support about
the user's life or situation, the assistant consults the book before
answering.

This rule is broad. It is not limited to "recovery questions" or a fixed
topic list. It includes drinking, inability to stop, fear of relapse,
loneliness, shame, resentment, relationships, family, work, loss, isolation,
hopelessness, spiritual questions, uncertainty, and other human problems that
may be illuminated by the book.

Simple greetings, technical commands, or purely operational questions do not
require book retrieval.

## 4. Turn state machine (authoritative)

For every incoming turn:

0. **Safety/operations gate.**
   - Deterministic safety (#21) runs first.
   - Pure bot/technical commands bypass book grounding.
   - Every other non-trivial user turn defaults to substantive and enters
     the grounding pipeline.

1. **Retrieval planning — no answer generation.**
   - Input: user message plus the compact public book map only.
   - Produce 3-6 retrieval aspects/queries covering materially different
     interpretations and perspectives of the user's situation.
   - For a narrow exact-fact/phrase question, one aspect is allowed.
   - This stage is forbidden from composing the user-visible answer.

2. **Whole-corpus hybrid retrieval.**
   - Run every planned aspect against the entire canonical corpus.
   - Lexical BM25 over English canonical text.
   - Multilingual dense retrieval for RU→EN and EN→EN.
   - Fuse rankings with RRF.
   - Deduplicate near-identical hits.

3. **Coverage/diversity selection.**
   - Do not simply take the globally highest-scoring neighboring chunks.
   - Select materially distinct evidence clusters across chapters/sections
     when the query warrants it.
   - Preserve which retrieval aspect produced each candidate.
   - #19 may tune top-K, fusion, and MMR-style diversity parameters but may
     not remove this stage.

4. **Exact source loading.**
   - Use `book_read` for selected chunks.
   - Use `book_expand` for neighboring paragraphs/sentences when a local
     argument, story, warning, or action is incomplete.
   - All evidence inserted into answer context is exact canonical source
     text with stable source IDs.

5. **Coverage check — no final answer yet.**
   - Input: user message, book map, and source IDs plus exact evidence
     already collected.
   - Ask only: which materially relevant AA-book aspect, story, principle,
     warning, action, or contrasting perspective may still be missing?
   - If a missing aspect is identified, formulate additional retrieval
     queries and repeat stages 2-4 once.
   - Maximum retrieval rounds for MVP: 2 (initial plus one coverage pass)
     to keep latency bounded. #19 may change this only with evaluation
     evidence.

6. **Evidence-pack construction.**
   - Deduplicate exact/overlapping passages.
   - Retain materially distinct source regions.
   - Order evidence by relevance/coherence, not by retrieval score alone.
   - Enforce the source-token budget from #3 (see `docs/context-budget.md`).
   - Never truncate a source passage silently; see the atomic packing rule.
   - If evidence does not fit, drop the least material complete evidence
     unit and mark coverage as budget-limited.

7. **Answer synthesis.**
   - The answer model receives: the authoritative system prompt, the
     current user message and conversation state, the compact book map for
     orientation, and the assembled exact evidence pack.
   - It answers only from that evidence pack for substantive content.
   - Each substantive answer paragraph/claim must internally reference one
     or more supporting source IDs.
   - Synthesis is natural conversational prose, not a citation dump or a
     list of search results. A strong answer may combine several places in
     the book (stories, patterns, principles, warnings, practical actions),
     acknowledge the user's situation, explain principles in plain language,
     suggest book-supported next steps, and use short exact quotations when
     they materially help.
   - When useful, identify the relevant chapter/section. Never fabricate a
     quote and never say "the book says" when the retrieved text does not
     support the statement.

8. **Grounding gate.**
   - Before a response is sent to Telegram, validate that every substantive
     answer unit has source IDs from the current evidence pack.
   - If a substantive unsupported unit exists, regenerate once with the
     unsupported unit explicitly rejected.
   - If support still cannot be established, do not send the unsupported
     content. Return the closed-book fallback: the canonical text does not
     establish the requested answer, followed only by the closest supported
     book material.
   - Source IDs may remain internal; chapter/section names may be shown
     naturally when useful.

The Python application owns this state machine. OpenCode is the LLM
execution/runtime boundary inside the workflow. The application must not
simply send the user's text to `aa` and trust the agent to decide whether
to retrieve; it drives retrieval planning, deterministic whole-corpus
search, exact evidence loading, the coverage pass, evidence-pack assembly,
final `aa` synthesis, and grounding validation.

Where an LLM phase needs structured control output (retrieval plan,
missing-aspect decision, answer plus support IDs), use OpenCode structured
JSON-schema output when supported by the pinned runtime and verify it in
integration tests. If the pinned runtime lacks the required
structured-output contract, use a strict parse/validation wrapper and fail
closed rather than accepting malformed control output.

## 5. Whole-book coverage loop (explicit)

The quality target is not "one relevant chunk found". It is the best
practical approximation of the answer a careful reader could derive after
consulting the relevant portions of the whole book.

- Broad, personal, ambiguous, or multi-theme requests require multiple
  retrieval queries/subquestions.
- Search runs over the entire canonical corpus, not one likely chapter.
- Exact passages are read from multiple materially distinct relevant
  regions when available.
- Neighboring context is expanded when needed to preserve the complete
  local argument (story, warning, principle, action).
- Coverage is explicitly checked before synthesis: could another relevant
  experience, principle, warning, action, or contrasting perspective
  elsewhere in the book materially change or improve the answer?
- When yes, a second retrieval pass for the missing aspects is mandatory
  (within the maximum of 2 rounds).
- The model does not need to expose its internal coverage reasoning to the
  user.

## 6. Stop/saturation rule (explicit)

Stop retrieval when either:

- another search pass produces no materially new support (saturation); or
- the configured source/context budget is reached.

Budget behavior:

- The retrieved-passages budget is fixed by #3 (see
  `docs/context-budget.md`; currently 16,000 tokens within a 76,000-token
  reservation against a 200,000-token effective context).
- Passages pack atomically: the full ordered set must fit or the pack
  operation refuses; callers re-rank and explicitly request a smaller set.
- Passages are never cut and trailing passages are never silently dropped.
- A single passage larger than the budget is refused and must be
  re-chunked upstream at sentence boundaries.
- If the budget prevents adequate coverage, the answer is narrowed and
  explicitly qualified. Missing source support is never filled from model
  memory and presented as if it came from the book.

## 7. Fixed retrieval engine contract (2026-10-04)

The retrieval engine is fully specified. Downstream work must not choose a
different search stack.

For every planned retrieval aspect, the application calls `book_search`
with:

- a semantic query in the user's language;
- a concise English lexical rewrite for the canonical English corpus.

`book_search` is backed by the #17 implementation:

- SQLite FTS5 BM25;
- pinned local `intfloat/multilingual-e5-base` embeddings;
- exact Faiss `IndexFlatIP`;
- lexical top 40 plus dense top 40;
- RRF with k=60;
- overlap deduplication;
- at most 12 compact candidates per aspect.

Across aspects, the Python orchestration layer — not the model — enforces
deduplication, source-cluster diversity, exact reads, bounded expansion,
evidence budgeting, and the optional second coverage pass.

The book map is used only to help the planner formulate aspects. Search
previews, map text, embeddings, rankings, and summaries are never
sufficient evidence. Only exact canonical text returned by `book_read`,
`book_expand`, or bounded `book_section` may enter the substantive evidence
pack.

For this one-book corpus, approximate vector search, vector databases, and
external search services are explicitly out of scope. Ordinary user turns
reuse the already-built index and must never rebuild or re-embed the book.

## 8. OpenCode runtime integration

- Project-local primary OpenCode agent named `aa`, bound in `opencode.json`
  to `prompts/aa-agent-system.md` with deny-by-default permissions allowing
  only the AA book tools (`book_search`, `book_read`, `book_expand`,
  `book_section`).
- Every substantive Telegram turn sent to OpenCode explicitly selects the
  `aa` agent (the worker's `send_message(..., agent="aa")` boundary
  already supports this). The worker fails closed when the `aa` agent is
  unavailable; it never silently falls back to the default Build/Plan
  coding agents.
- Per-request `system` injection is not the primary production mechanism:
  one versioned prompt file is easier to audit and test, avoids drift
  between worker code and OpenCode configuration, avoids accidental
  omission on one call path, and keeps agent identity and tool policy
  colocated. Per-request `system` injection may be used only for
  controlled tests or an explicitly documented temporary override.
- This is context/policy specialization, not fine-tuning. Model weights
  are not trained or modified. The "trained assistant" effect comes from
  the persistent named agent plus authoritative prompt, deny-by-default
  tool permissions, deterministic workflow/orchestration, and evaluation
  and grounding gates.
- Versioned integration files:
  `prompts/aa-agent-system.md`, `opencode.json` (or project-local
  equivalent), and the project-local book tools (`.opencode/tools/`
  `book_search`, `book_read`, `book_expand`, `book_section`) owned by #18.
- Exact OpenCode configuration syntax must be verified against the pinned
  OpenCode runtime version during implementation, but the product contract
  is fixed: deny by default, allow only the AA book tools required by this
  agent.

## 9. Fixed AA-assistant runtime model policy (2026-10-04)

Scope: this policy applies only to the user-facing AA assistant runtime.
It does not define or change the model used by Continuum/OpenCode to
implement repository tasks.

- Primary: `opencode/muse-spark-1.3-contributor-free`.
- Fallback: `opencode/space-bunny-free`.
- The worker/runtime explicitly selects the primary model; it never relies
  on the ambient default model.
- There is no implicit third fallback (including Build/Plan defaults or
  any other model).
- Fallback is technical only: Space Bunny is used only after bounded
  retries of a retryable primary-model availability, rate-limit, or
  provider failure.
- There is no fallback because retrieval found no support, grounding
  rejected an answer, a tool/index failed deterministically, or the model
  produced unsupported content. Those conditions fail or repair through
  their own pipeline.
- A fallback turn uses the same named `aa` agent, authoritative system
  prompt, tool permissions, evidence pack, grounding gate, and output
  contract.
- Fallback is per-turn, not sticky: the next new turn starts with Muse
  again.
- Diagnostics/qualification record which model actually served the turn
  without logging user text.
- Both model IDs stay configurable through the OpenCode/runtime
  configuration surface (`OPENCODE_MODEL`, `OPENCODE_FALLBACK_MODEL`),
  while these values remain the project defaults.

## 10. Language behavior (RU/EN)

- Reply in the user's language. Russian and English are first-class
  supported languages.
- Retrieval may cross languages; the canonical source remains English.
- Each `book_search` call carries both the user's-language semantic query
  and the English lexical rewrite.

## 11. Safety handoff (#21)

- The deterministic safety layer is authoritative for acute
  medical/emergency situations and runs before ordinary AA guidance. The
  assistant does not override it.
- No medication dosing, diagnosis, detox schedules, or instructions for
  unsupervised alcohol withdrawal.
- Acute medical/emergency routing never depends on retrieval or on model
  compliance.

## 12. Orchestration implications for tool design (#18)

The system prompt is necessary but not sufficient. The implementation must
make the requested behavior mechanically reachable:

- `book_search` supports repeated multi-query retrieval over the entire
  corpus;
- results expose distinct source regions and chapter coverage, not only
  one global top hit;
- `book_read` and `book_expand` load exact original text;
- orchestration permits a second search pass before final answer
  generation;
- context budgeting reserves room for multiple materially distinct
  passages;
- evaluation rejects answers that rely on one convenient hit when the gold
  case requires broader source coverage.

## 13. What #19 may tune (and may not remove)

#19 is an evaluation/tuning task, not an architecture-selection task. It
may tune:

- number of retrieval aspects within the allowed range (1 for narrow
  exact-fact questions, otherwise 3-6);
- per-query top-K;
- BM25/dense fusion weights or RRF parameters;
- diversity/MMR-like thresholds;
- chunk sizes and boundaries;
- expansion radius;
- evidence-pack token allocation;
- whether the optional second retrieval round is needed for specific
  narrow query classes.

It may not remove:

- mandatory retrieval for substantive turns;
- the closed-book answer policy;
- whole-corpus search;
- multi-aspect planning for broad/personal requests;
- exact source loading;
- the coverage check;
- evidence-pack-only synthesis;
- the grounding gate.

## 14. Downstream contract

- #8 (hierarchy plus book map): stable IDs, neighbor links, and a compact
  routing-only map that fits the #3 budget.
- #17 (multilingual hybrid retrieval): the fixed engine in section 7.
- #18 (OpenCode read tools): the narrow read-only `book_search`,
  `book_read`, `book_expand`, and bounded `book_section` surface with
  exact-text provenance.
- #19 (retrieval/tool qualification): the versioned RU/EN gold set and the
  quality gate proving broad personal questions are answered from
  materially complete coverage rather than one convenient chunk.
- #9 (conversational policy): answer synthesis and grounding validation
  against this contract.
- #5/#6/#7 integrate and qualify the whole pipeline without weakening it.

## 15. Definition-of-Done traceability

- Authoritative system prompt committed: `prompts/aa-agent-system.md`
  (verbatim draft from #26).
- Retrieval/orchestration contract committed: this document.
- Whole-book coverage loop explicit: sections 4 (stages 1-5) and 5.
- Stop/saturation rule explicit: section 6 and the STOP RULE in the prompt.
- Sponsor-style behavior explicit without false identity claims:
  section 1 and the prompt header plus STYLE section.
- Downstream issues reference this contract: section 14.
- Manual review before downstream automation resumes: this document and
  the prompt were reviewed manually on 2026-10-04 per the issue's
  stabilization note.
