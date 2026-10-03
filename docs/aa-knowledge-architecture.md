# AA knowledge architecture

Status: accepted for MVP  
Decision issue: #20

## Decision

Use a repository-local **agentic RAG** knowledge layer around one immutable
canonical AA source.

We explicitly reject two earlier approaches:

1. loading the complete book into every OpenCode request/session;
2. maintaining a permanently shortened or deletion-curated edition as runtime
   source authority.

The full canonical scope remains addressable. Runtime context contains only a
small navigation map plus source-exact passages retrieved for the current need.

## Why

A full-book stable prompt spends context on material unrelated to most turns,
reduces conversation/output headroom, and makes context-cost behavior harder to
control.

A permanently shortened corpus creates an irreversible relevance decision
before the user's question is known and can discard a narrative detail that is
important for a later question.

Retrieval lets relevance be decided at query time while preserving exact
provenance to the original text.

## Canonical source contract

The MVP source scope is:

- The Doctor's Opinion;
- Chapters 1–11.

The repository reuses the source-acquisition foundation merged in PR #11.

Normal Git history stores:

- source URLs/identity;
- fetch and validation code;
- checksums/version metadata;
- non-literary derived metadata where appropriate.

The complete downloaded text lives in the ignored runtime/build workspace.

Canonical literary wording is never normalized, summarized, translated,
corrected, or permanently reduced. Every evidentiary passage returned to the
assistant must map back to exact source bytes/text and source location.

## Hierarchical source model

Issue #8 builds a stable navigation hierarchy:

```text
book
  -> section/chapter
     -> paragraph
        -> sentence
           -> retrieval chunk/range
```

Each searchable unit retains:

- stable ID;
- chapter/section identity;
- exact source locator/offsets;
- parent ID;
- previous/next neighbor IDs;
- source checksum/version.

Chunk boundaries follow natural sentence/paragraph boundaries. They do not split
sentences solely to satisfy token sizes.

## Compact book map

A small, versioned book map is always present in the dedicated AA agent context.

Its purpose is fast orientation: section order, topics and stable IDs/ranges.
It is not source authority. The agent must read source text before making
substantive AA/recovery claims.

The map must fit the explicit context budget established by #3.

## Retrieval baseline

The canonical text is English, while users may ask in Russian or English.
Cross-lingual retrieval is therefore mandatory.

Issue #17 first implements and measures the simplest strong baseline:

1. lexical/BM25 search over English canonical chunks;
2. multilingual semantic embeddings supporting RU -> EN and EN -> EN retrieval;
3. reciprocal-rank/rank fusion and deduplication.

For non-English queries, lexical query expansion/translation may be benchmarked
rather than assuming Russian tokens can search English BM25 effectively.

Contextual chunk enrichment and reranking are optional improvements, not fixed
requirements. They are adopted only when #19 demonstrates a material quality
benefit that justifies latency/complexity.

This follows the general hybrid-retrieval pattern described by Anthropic's
Contextual Retrieval work and current retrieval/ranking guidance, while keeping
the concrete stack benchmark-driven for this small corpus.

References:

- https://www.anthropic.com/engineering/contextual-retrieval
- https://platform.openai.com/docs/guides/retrieval

## OpenCode book tools

Issue #18 exposes a narrow project-local read-only interface:

### `book_search(query)`

Returns compact ranked candidates and provenance metadata. It does not dump
large source passages.

### `book_read(chunk_id)`

Returns exact canonical source text for one selected retrieval unit.

### `book_expand(chunk_id, before, after)`

Returns bounded exact neighboring context when the initial hit cuts through a
larger argument.

### `book_section(section_id, cursor/range)`

Supports a deliberate bounded larger read with pagination/ranges. It never
implicitly loads the whole book.

Project-local OpenCode custom tools are preferred over MCP because there is no
current cross-repository tool-reuse requirement.

## OpenCode capability boundary

The conversational AA agent is deny-by-default.

It receives the AA read tools it needs. Unrelated shell execution, write/edit,
arbitrary web access and coding tools are disabled unless a separate documented
runtime requirement explicitly needs one.

The Telegram worker does not implement a second retrieval/LLM stack.

## Runtime context contract

Steady context contains only:

1. system/behavior policy;
2. compact book map;
3. per-chat conversation state;
4. source-exact passages fetched for the current question;
5. output headroom.

Retrieved passages may not be silently truncated. Context limits and expansion
bounds are measured and fixed by #3/#19.

## Hybrid whole-book grounding

The compact book map is an always-loaded routing layer only. It must never be used as the sole basis of a substantive answer.

For every substantive personal/support request, the agent performs coverage-oriented retrieval across the canonical book:

1. interpret the user's situation and derive multiple search aspects when needed;
2. use the book map to route toward likely sections and alternative perspectives;
3. run hybrid search across the whole corpus;
4. diversify and deduplicate candidates across distinct source regions;
5. read exact passages from multiple materially distinct candidates;
6. expand neighboring context where the local argument requires it;
7. run a coverage check and a second search pass for missing aspects;
8. stop when further retrieval adds no materially new support or the explicit source budget is reached;
9. synthesize the final answer only from retrieved canonical material.

The quality target is therefore not "one relevant chunk found". It is the best practical approximation of the answer a careful reader could derive after consulting the relevant portions of the whole book.

If the source-token budget prevents complete coverage, the response must narrow/qualify its claims rather than fill gaps from model memory.

## Grounding contract

For substantive AA/recovery claims the agent:

1. searches;
2. reads the strongest candidate passages;
3. expands only if local context is insufficient;
4. answers only to the extent supported by retrieved canonical text;
5. identifies the relevant chapter/section where practical.

Map summaries, embedding metadata, scores and reranker output are never
presented as the book.

## Evaluation

Issue #19 uses a versioned RU/EN gold set and measures at least:

- lexical recall@K;
- semantic recall@K;
- hybrid recall@K;
- optional reranker recall@K;
- RU vs EN retrieval quality;
- source-support success after tool use;
- tool-call count;
- source tokens injected;
- unnecessary expansion rate.

The preferred implementation is the simplest configuration that meets the
quality gate. The current target is recall@5 >= 0.95 on the project gold set;
failures remain visible instead of being hidden by adding unmeasured machinery.

## Safety boundary

Emergency/medical-risk routing is independent of RAG and ordinary prompt
compliance. Issue #21 implements a deterministic RU/EN safety path before normal
AA handling.

## Task graph

The repository-local implementation chain is:

```text
#20 architecture
  -> #3 canonical source + context budget
  -> #8 hierarchy + book map
  -> #17 multilingual hybrid retrieval
  -> #18 OpenCode read tools
  -> #19 retrieval/tool qualification
  -> #9 conversational policy
  -> #5 Telegram/OpenCode integration
  -> #6 bounded runtime control
  -> #7 live E2E qualification
```

After #20, #21 safety may run in parallel with the knowledge chain. #5 requires
both #9 and #21.
