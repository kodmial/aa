# Guard `guard-internal-terms` audit (kodmial/aa#450)

Part A of issue #450. The guard rejects user-visible text that leaks
retrieval/verification mechanics. Live Telegram runs (reports
#39972781629, job failures 120479539082) exposed two defect families:
Russian-language mechanics disclosures passed the guard unchanged, and a
safe grounded Russian answer that was locally `guard-internal-terms`-rejected
collapsed into the generic `verifier_outcome=unavailable` path with zero
repair, so the user received a `[service-error]` after a full
planner/retrieval/answer chain.

This document records the vocabulary audit, the exact post-#450
behavior, and the known conservative limitations (separate defects,
intentionally not fixed here).

The live vocabulary is the closed tuple `_INTERNAL_TERMS` in
`src/aa/conversation/turn_pipeline.py`. Matching is a casefolded,
Unicode-token-bounded rule match performed BEFORE the semantic verifier
is invoked, so no leaky draft ever reaches a verifier call and no draft
text travels in telemetry: `_matched_internal_rule` returns only the
fixed rule id from the module's own vocabulary and `_verify_draft`
attaches `detail=f"rule:{term}"` (`source=local`, stage `verification`).
A Latin rule matches only as a whole token or phrase, never embedded
inside a longer ordinary word; Cyrillic rules also accept inflected
continuations on every word of a phrase, not only the final one (see
Cyrillic inflection handling).

## Audit tables

### A1: allowed samples (must stay allowed)

All rows are pinned by `test_guard_allowed_samples_pass` and
`test_added_rules_each_have_negative_and_positive_case`
(`tests/test_p0_450_internal_terms_guard_repair.py`); every sample is
synthetic, never real user text or canonical excerpt text.

| Sample | Why it passes |
| --- | --- |
| `Поддержка рядом помогает пережить тягу сегодня. Расскажите, что сейчас важнее всего?` | Ordinary supportive Russian; no rule substring. |
| `Честный разговор о трудностях и ближайшие трезвые шаги помогают сегодня.` | Ordinary grounded Russian. |
| `Спокойный вечерний распорядок и поддержка рядом помогают сегодня.` | Ordinary grounded Russian. |
| `Поговорите со спонсором до того, как примете решение, — это опора группы.` | Book-grounded phrasing. |
| `Большая Книга — верный спутник; приходите на собрание и расскажите о трудном дне.` | Book-grounded phrasing. |
| `Alcoholics Anonymous — история сообщества, где каждый может начать заново.` | Proper-name English does not collide with any rule substring. |
| `Завтра — день поиска поддержки; собрание в семь вечера.` | Contains the ordinary word `поиск` but not any mechanics phrase such as `поиск по корпусу`. |
| `Корпус здания виден от входа в общину, подходят любые одежды.` | Contains the ordinary noun `корпус` but not `corpus` (Latin-script rule) nor `корпус текстов`. |
| `Ваш запрос важен; приходите на собрание и спрашивайте о прошлом опыте.` | Contains `запрос` but not `поисковый запрос`-shaped mechanics phrasing. |
| `The indexing of the group notes is complete.` | `index` is embedded in the ordinary word `indexing`, not a standalone rule token. |
| `Providers of support meet on Tuesdays.` | `provider` is embedded in the ordinary plural `providers`. |
| `Planners and coordinators share the schedule.` | `planner` is embedded in the ordinary plural `planners`. |
| `Qualifications for the volunteer role are simple.` | `qualification` is embedded in the ordinary plural `qualifications`. |
| `Evidenced progress matters more than perfection.` | `evidence` is embedded in `evidenced`, not exposed as a mechanics term. |

### A2: forbidden samples (must stay blocked)

Real disclosure shapes drawn from the live job failures and run reports,
schematized for the regression suite.

| Sample | Live failure evidence |
| --- | --- |
| `Мы переписали ваш вопрос в поисковый запрос и проверяем эмбеддинги.` | Russian answer family exposing planner query + embedding internals. |
| `Система запустила поиск по корпусу текстов, но индекс не откликнулся.` | Russian answer family explicitly naming the corpus-index mechanics. |
| `The retrieval pipeline failed while re-ranking; bm25 index unavailable.` | English disclosure family from job logs. |
| `Fail-Closed: ответ сервера не прошёл json-schema проверку.` | Status/code echo family (`fail_closed`, structured-validation leak). |
| `Модель использует векторную базу из Faiss и Bm-25.` | Russian tooling-name leak with mixed-case English names. |
| `Structured-Output parsing failed, повторите запрос позже.` | Structured-vocabulary leak carried verbatim from verification. |
| `Диагностика: rerank внутри FAISS упал в выдаче.` | Mixed RU+EN mechanics diagnostics leak. |
| `code=ru-fourth-edition-txt, status unavailable, provider returned 429` | Source-ID/status/provider-error echo family. |

## Added rules (post-#450, additive only; nothing removed)

Every addition is pinned by the audit tests with a direct example, and
the A1 corpus keeps the near-miss whitelist intact.

| Rule (casefold token/phrase) | Negative (allowed) | Positive (blocked) |
| --- | --- | --- |
| `embeddings` (explicit English plural of `embedding`) | ordinary `embedding`-free book phrasing | `Dense embeddings store the passages.` |
| `эмбеддинг` | ordinary `поддержка`/`распорядок` answers | `...и проверяем эмбеддинги` |
| `риранк` | — | `Риранк по корпусу текстов не дал ответа` |
| `фаисс` | — | `Фаисс упал` (RU transliteration of `faiss`) |
| `бм25` | — | `Бм25 индекс недоступен` (RU transliteration of `bm25`) |
| `векторная база`, `векторное хранилище` | `векторная`-free ordinary phrasing | `векторная база не открылась` |
| `поиск по корпусу` | `день поиска поддержки` | `запустила поиск по корпусу текстов` |
| `корпус текстов` | `Корпус здания виден от входа` | `по корпусу текстов, но индекс не откликнулся` |
| `fail-closed` (joining existing `fail_closed`, `fail closed`) | ordinary `не работает` phrasing | `Fail-Closed: ответ закрыт` |
| `search-failure` (joining `search failure`) | — | `Search-Failure in the service layer` |
| `structured-output` (joining `structured output`) | — | `Structured-Output parsing failed` |
| `json-schema` (joining `json_schema`, `json schema`) | — | `не прошёл json-schema проверку` |
| `bm-25` (joining `bm25`) | — | `bm-25 index unavailable` |

Case/format covering: matching casefolds the text first and then
requires Unicode word boundaries, so `Fail-Closed`, `BM25`, `FAISS`,
`РИРАНК` all block while containing words such as `indexing`,
`providers`, `planners` and `qualifications` stay allowed. The
historical English rules (`corpus`, `retrieval`, `grounding`,
`evidence`, `index`, `embedding`, `planner`, `provider`,
`qualification`, `fail_closed`, `fail closed`, `search failure`,
`structured output`, `json_schema`, `json schema`, `rerank`, `bm25`,
`faiss`) are unchanged; no rule was removed and direct rule tokens still
block. The trailing boundary means Latin plurals are not matched
morphologically, so `embeddings` is listed explicitly rather than
widening the family (which would also match ordinary `providers` and
`planners`). This is the accepted product tradeoff: other Latin plurals
such as `planners`, `qualifications` and `retrievals` remain outside the
guard because the strict token boundary deliberately avoids blocking
ordinary English word forms.

## Cyrillic inflection handling

Unlike the Latin trailing boundary, Cyrillic rules accept Russian case
endings so `эмбеддинги`, `риранка` and `фаиссе` cannot bypass the guard.
For a multi-word rule every word is anchored on an explicit audited
stable root from `_CYRILLIC_WORD_ROOTS` and accepts any word
continuation, so `векторной базе`, `корпуса текстов` and `векторною
базою` are blocked as well as the exact phrases. The suffix table is
only used to normalize the canonical rule word; it is not re-applied to
draft text. Consequently an inflectional ending that is absent from
`_CYRILLIC_INFLECTION_SUFFIXES` (for example the diminutive
`векторненькая` or the augmentative `корпусище`) still matches, as
pinned in `test_cyrillic_endings_outside_suffix_table_blocked`. Every
current guarded word has an explicit root, enforced by
`test_cyrillic_word_roots_cover_every_guarded_phrase_word`, so a future
rule word cannot silently rely on suffix stripping. The audit samples
found no ordinary book-grounded word that shares a guarded root, and
unrelated prefix words such as `корпус здания` / `векторный анализ`
stay allowed (pinned in `test_cyrillic_unrelated_prefix_allowed`). A
future ordinary-word collision would require narrowing the inflection
set rather than removing the rule.

## Documented limitations (separate defects; #450 scope deliberately leaves them)

1. **Single-Cyrillic language gate.** `contains_cyrillic` accepts any
   text containing at least one Cyrillic symbol (`_CYRILLIC_RE`), so a
   mostly-English draft with one Russian word passes the RU-output
   language gate (`"Please say something. Раз."` is pinned in
   `test_language_gate_single_cyrillic_limitation_is_documented`).
   Fully tightening it is a separate issue inside the language policy
   contract; #450 pins the current behavior so a future change is
   deliberate and visible.
2. **Standalone English words stay conservatively blocked.** Rules
   such as `evidence`, `provider`, `qualification`, `planner`, `index`
   are common standalone English words. The review follow-up added
   token boundaries, so a rule embedded inside a longer ordinary word
   (`indexing`, `providers`, `planners`, `qualifications`) no longer
   trips the guard; those collisions are pinned in A1. A benign English
   sentence that uses one of the rule words as a standalone token still
   stays blocked (conservative direction), because the same tokens are
   genuine mechanics vocabulary in draft disclosures.
3. **No live-draft knowledge in CI.** The verdicts above are pinned on
   schematizations of the live reports; the actual live drafts after
   #450 are only observed in the next live-side qualification run.
4. **Stem alternation needs an explicit audited root.** The matcher is
   not a general morphological analyzer. A future guarded word whose
   stem alternates (fleeting vowel or suppletion) must list every
   alternant in `_CYRILLIC_WORD_ROOTS`. The audit test requires an
   explicit entry for every current rule word, so an unaudited word
   fails CI instead of silently bypassing the guard. The current closed
   vocabulary contains no stem-alternating word.
