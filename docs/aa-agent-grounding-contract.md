# AA agent grounding contract

Status: implementation for issue #26.
Matches the fixed contracts in #44 (Russian-first architecture),
#46 (slang-aware query planner), and #48 (Russian quotation and
multilingual grounding policy).

This note defines what the named OpenCode `aa` agent prompt owns and
what deterministic Python orchestration owns. The design is fixed;
OpenCode must not redesign the product architecture.

## Named agent binding

The project-local primary agent is `aa`:

- `opencode.json` declares `agent.aa` with `mode: primary`,
  `prompt: {file:./prompts/aa-agent-system.md}`, and
  `temperature: 0.2`.
- `Settings.opencode_agent` defaults to `"aa"`; the worker forwards
  turns to that agent with a technical model fallback only.
- The agent is deny-by-default. Only four read-only AA tools are
  exposed: `book_search`, `book_read`, `book_expand`, `book_section`.
  Shell execution, file edits, coding work, and arbitrary web access
  are denied.
- The agent is an AI AA-literature-based support assistant with a
  direct, compassionate, practical sponsor-like conversational style.
  It never claims to be human, an AA member, the user's actual
  sponsor, a clinician, or to have lived sobriety experience.

## Language split

- The system prompt and stable tool/control instructions stay in
  English.
- The user-facing response follows the user's language. Russian is
  the primary product language; Russian conversation is the default.
- The user's original Russian wording is preserved in ephemeral turn
  state and is never replaced by one normalized interpretation.
- English is never the canonical source for Russian production.
  English evidence is reference/control only and never authorizes a
  Russian direct quotation.
- An English quotation must never be translated and labeled as a
  source-exact Russian quote.

## Canonical sources

- For Russian users the primary production evidence source is the
  qualified Russian Fourth Edition lineage: title
  `RUSSIAN_EDITION_TITLE`, publisher marker
  `RUSSIAN_EDITION_PUBLISHER_MARKER`, ISBN `RUSSIAN_EDITION_ISBN`
  (see `src/aa/grounding/quotes.py`).
- English (The Doctor's Opinion and Chapters 1-11) is a separately
  versioned reference/control corpus.
- RU and EN use language-neutral logical section/chunk IDs with
  per-language exact locators and checksums.
- Plaintext literary text is never committed to Git. Encrypted
  snapshots restore deterministically and are SHA-verified.
- Russian production fails closed when the qualified Russian corpus
  is unavailable. It must not silently substitute a generated
  translation of English text.

## Evidence policy

The agent may synthesize natural Russian prose, but every
substantive AA claim must be supported by the current validated
evidence pack of exact source text.

Never treated as evidence:

- model memory and general recovery/psychology/medical knowledge;
- planner output, normalizations, and query rewrites;
- compact book map and routing metadata;
- search previews, embeddings, rankings, scores, and reranker output;
- generated translations and translation drafts.

Additionally:

- Every substantive answer unit must be semantically supported by
  exact current-turn evidence text. A bare `source_id` or locator
  without source-exact text is not provenance.
- Interpretation added during translation, paraphrase, or synthesis
  that the cited text does not support fails grounding.
- When current evidence does not establish a requested substantive
  claim, the agent qualifies or declines it and offers only the
  closest supported material.
- The deterministic gate is `aa.grounding` (`src/aa/grounding/`,
  issues #48): verbatim-substring checks, translation-label checks,
  provenance-to-evidence resolution, pinned corpus versions, and
  semantic-support checks with an injectable entailment predicate.

## Russian quotation policy

- Russian conversations display quotations in Russian.
- Direct Russian quotations in normal production are verbatim
  substrings of the version-pinned authoritative Russian corpus.
  Spelling and punctuation in quoted source text are not repaired.
- When no authoritative Russian source is available, a generated
  translation may be shown only when explicitly allowed, only with
  the `TRANSLATION_MARKER_RU` label
  (`[перевод — не точная цитата источника]`), and only with
  provenance to the exact source unit(s) it derives from.
- A translation is never presented as source-exact Russian text,
  including translation-labeled text passed through the exact-source
  path.
- Chapter/section attribution uses Russian display titles where
  practical and always carries pinned provenance (corpus version,
  source id, section/chunk locators, offsets).

## Planner boundary (#46)

- The pre-retrieval planner preserves the original user text and
  produces additive retrieval formulations: spelling corrections,
  slang/synonym expansions, and multiple semantic aspects.
- Planner output may influence where to search, never what is true.
- The planner must not strengthen uncertain wording into
  diagnosis, severity, or facts the user did not state.
- Planner JSON is schema-validated control data. It is navigation
  metadata only and can never enter the evidence pack as source
  authority.

## Prompt/orchestrator boundary

The English system prompt (`prompts/aa-agent-system.md`) stays
compact. It defines identity and role boundaries, source and
evidence authority, the language and exact-quote rule, closed-book
behavior, the safety handoff, and concise conversational style.

Deterministic Python orchestration owns the retrieval state machine
and enforcement, and that machinery is not duplicated in prompt
prose:

`safety/operations gate -> query planning -> retrieval -> diversity
-> exact reads/expansion -> coverage pass -> evidence pack ->
synthesis -> grounding gate -> one bounded repair -> fail closed.`

Concretely:

- Prompt owns: who the agent is, what counts as authority, Russian
  quotation duties, closed-book response duties, safety handoff,
  conversational style, and fail-closed behavior when sources or
  tools are unavailable.
- Orchestrator owns: phase ordering, query-plan schema validation,
  RU-first retrieval branches and fusion, diversity and
  deduplication, read/expansion budgets, coverage checks and second
  passes, evidence-pack construction, grounding-gate enforcement
  (`aa.grounding`), one bounded repair, and deterministic safety
  routing (#21).

## Safety handoff (#21)

- The deterministic emergency/medical-safety layer runs before
  ordinary AA guidance and is authoritative for acute danger. The
  prompt is not the sole safety boundary and never overrides it.
- The agent does not provide diagnosis, medication dosing, detox
  schedules, or instructions for unsupervised alcohol withdrawal.

## DoD traceability (#26)

- Named `aa` primary agent: `opencode.json`, `Settings.opencode_agent`.
- English concise system prompt: `prompts/aa-agent-system.md`.
- Russian primary user-facing language: prompt `LANGUAGE` section.
- Exact Russian quote rule: prompt quotation section plus
  `src/aa/grounding/quotes.py` and `src/aa/grounding/gate.py`.
- Memory/map/rewrites non-evidentiary: prompt source/grounding
  sections plus `EvidenceKind` gate (`tests/test_russian_grounding.py`).
- No redundant orchestration mechanics in prompt: prompt defers to
  `aa.grounding`; orchestrator owns the state machine above.
- Only four read-only tools: `opencode.json` permission map.
- Safety handoff: prompt `SAFETY` section plus `src/aa/safety/`.
- Matches #44/#46/#48: sections above; Russian Fourth Edition
  primary, planner never evidence, grounding gate authoritative.
