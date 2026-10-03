# AA book map (navigation only)

Versioned routing layer for the AA agent and hybrid retrieval. It names where to look next; it is never evidence for an answer.

- structure format: `aa-corpus-structure/1` (builder v1)
- canonical artifact: `aa-canonical/1` sha256 `7bd1b398090ecc9dbea829b7107841972a2a55daf4484935a33d7ec33ed7463b`
- scope: 12 sections in canonical order
- retrieval chunks: 361 addressable ranges
- context budget: compact map must fit 6000 tokens
- grounding rule: substantive claims must cite exact passages read with
  `book_read` / `book_expand` / `book_section`, never this map.

Coverage planning: for broad personal questions, plan searches across the
whole book, read exact passages from distinct relevant regions, check for
missing perspectives, and only then synthesize. One top hit is not enough.

## Canonical order

| # | Section | Stable IDs | Chunks | Est. tokens |
|---|---|---|---|---|
| 0 | The Doctor's Opinion (`doctors-opinion`) | section `doctors-opinion` | `doctors-opinion:c001`..`doctors-opinion:c018` (18) | 3109 |
| 1 | BILL'S STORY (`chapter-1`) | section `chapter-1` | `chapter-1:c001`..`chapter-1:c034` (34) | 6386 |
| 2 | THERE IS A SOLUTION (`chapter-2`) | section `chapter-2` | `chapter-2:c001`..`chapter-2:c027` (27) | 5171 |
| 3 | MORE ABOUT ALCOHOLISM (`chapter-3`) | section `chapter-3` | `chapter-3:c001`..`chapter-3:c029` (29) | 5726 |
| 4 | WE AGNOSTICS (`chapter-4`) | section `chapter-4` | `chapter-4:c001`..`chapter-4:c028` (28) | 5507 |
| 5 | HOW IT WORKS (`chapter-5`) | section `chapter-5` | `chapter-5:c001`..`chapter-5:c030` (30) | 5285 |
| 6 | INTO ACTION (`chapter-6`) | section `chapter-6` | `chapter-6:c001`..`chapter-6:c037` (37) | 6795 |
| 7 | WORKING WITH OTHERS (`chapter-7`) | section `chapter-7` | `chapter-7:c001`..`chapter-7:c030` (30) | 6066 |
| 8 | TO WIVES (`chapter-8`) | section `chapter-8` | `chapter-8:c001`..`chapter-8:c037` (37) | 7090 |
| 9 | THE FAMILY AFTERWARD (`chapter-9`) | section `chapter-9` | `chapter-9:c001`..`chapter-9:c029` (29) | 5710 |
| 10 | TO EMPLOYERS (`chapter-10`) | section `chapter-10` | `chapter-10:c001`..`chapter-10:c032` (32) | 5781 |
| 11 | A VISION FOR YOU (`chapter-11`) | section `chapter-11` | `chapter-11:c001`..`chapter-11:c030` (30) | 5486 |

## Section guide

### 0. The Doctor's Opinion (`doctors-opinion`)

- topics: medical perspective, illness model, craving, mental obsession
- orientation: A physician's letter framing alcoholism as an illness with a bodily reaction to alcohol and a mental obsession that defeats willpower. Reach for it when the question is whether the condition is a moral failing or something that needs a program of recovery.
- read via: section `doctors-opinion`, chunks `doctors-opinion:c001`..`doctors-opinion:c018`
- consider also: `chapter-2`, `chapter-3` (a different perspective on the same problem)
- size: 12434 chars / ~3109 tokens, 41 paragraphs, 18 chunks

### 1. BILL'S STORY (`chapter-1`)

- topics: personal story, descent, identification, turning point
- orientation: A first-person account of early drinking, progressive loss of control, and the events leading toward recovery. Useful for identification and for questions about what the downward path can look like.
- read via: section `chapter-1`, chunks `chapter-1:c001`..`chapter-1:c034`
- consider also: `chapter-2`, `chapter-3`, `chapter-11` (a different perspective on the same problem)
- size: 25544 chars / ~6386 tokens, 75 paragraphs, 34 chunks

### 2. THERE IS A SOLUTION (`chapter-2`)

- topics: fellowship, hope, common solution, overview
- orientation: Introduces the fellowship's shared recovery and the claim that a common solution exists even for apparently hopeless cases. A good entry point for whether a way out exists, and a cross-check against narrow readings of later chapters.
- read via: section `chapter-2`, chunks `chapter-2:c001`..`chapter-2:c027`
- consider also: `doctors-opinion`, `chapter-4`, `chapter-5`, `chapter-11` (a different perspective on the same problem)
- size: 20684 chars / ~5171 tokens, 52 paragraphs, 27 chunks

### 3. MORE ABOUT ALCOHOLISM (`chapter-3`)

- topics: illness model, first drink, loss of control, moderation
- orientation: Describes the nature of the condition: trouble controlling the start and the amount once started. Central for questions about moderation, willpower, and why a single drink matters.
- read via: section `chapter-3`, chunks `chapter-3:c001`..`chapter-3:c029`
- consider also: `doctors-opinion`, `chapter-1`, `chapter-2` (a different perspective on the same problem)
- size: 22902 chars / ~5726 tokens, 41 paragraphs, 29 chunks

### 4. WE AGNOSTICS (`chapter-4`)

- topics: higher power, skepticism, willingness, belief
- orientation: Speaks to doubt about spiritual ideas and invites an open-minded experiment with help beyond oneself. Key for questions about unbelief or resistance to spiritual language; it offers a different angle on the same recovery problem.
- read via: section `chapter-4`, chunks `chapter-4:c001`..`chapter-4:c028`
- consider also: `chapter-2`, `chapter-5`, `chapter-6` (a different perspective on the same problem)
- size: 22027 chars / ~5507 tokens, 49 paragraphs, 28 chunks

### 5. HOW IT WORKS (`chapter-5`)

- topics: program basis, honesty, willingness, steps
- orientation: Lays out the foundation of the recovery program and the personal requirements emphasized throughout the book. Read it for questions about what participation actually asks of a person.
- read via: section `chapter-5`, chunks `chapter-5:c001`..`chapter-5:c030`
- consider also: `chapter-4`, `chapter-6`, `chapter-7` (a different perspective on the same problem)
- size: 21138 chars / ~5285 tokens, 82 paragraphs, 30 chunks

### 6. INTO ACTION (`chapter-6`)

- topics: action, inventory, amends, daily practice
- orientation: Moves from principles to concrete conduct: taking stock, repairing harm, and building a daily practice. Relevant when the question concerns amends, changed behavior, or what to do next.
- read via: section `chapter-6`, chunks `chapter-6:c001`..`chapter-6:c037`
- consider also: `chapter-5`, `chapter-7`, `chapter-9` (a different perspective on the same problem)
- size: 27180 chars / ~6795 tokens, 52 paragraphs, 37 chunks

### 7. WORKING WITH OTHERS (`chapter-7`)

- topics: helping others, sponsorship, service, carrying the message
- orientation: Covers working with others who still struggle and why mutual help sustains recovery. Useful for questions about helping a friend or family member; pair it with the family chapters for a fuller picture.
- read via: section `chapter-7`, chunks `chapter-7:c001`..`chapter-7:c030`
- consider also: `chapter-5`, `chapter-6`, `chapter-8`, `chapter-11` (a different perspective on the same problem)
- size: 24261 chars / ~6066 tokens, 48 paragraphs, 30 chunks

### 8. TO WIVES (`chapter-8`)

- topics: family perspective, spouses, partners, household
- orientation: Written for wives and partners: understanding the condition and responding sanely. Read it together with the next chapter so both sides of the household situation are covered.
- read via: section `chapter-8`, chunks `chapter-8:c001`..`chapter-8:c037`
- consider also: `chapter-9`, `chapter-7` (a different perspective on the same problem)
- size: 28359 chars / ~7090 tokens, 63 paragraphs, 37 chunks

### 9. THE FAMILY AFTERWARD (`chapter-9`)

- topics: family recovery, rebuilding trust, patience, home life
- orientation: Advice for the household after recovery begins: adjusting expectations and rebuilding trust over time. The companion to the previous chapter from the recovering household's side.
- read via: section `chapter-9`, chunks `chapter-9:c001`..`chapter-9:c029`
- consider also: `chapter-8`, `chapter-6` (a different perspective on the same problem)
- size: 22837 chars / ~5710 tokens, 47 paragraphs, 29 chunks

### 10. TO EMPLOYERS (`chapter-10`)

- topics: employers, workplace, responsibility, practical help
- orientation: Guidance for employers facing alcohol problems at work: understanding, firmness, and practical steps. A distinct workplace angle on the same human problem.
- read via: section `chapter-10`, chunks `chapter-10:c001`..`chapter-10:c032`
- consider also: `chapter-7`, `chapter-8` (a different perspective on the same problem)
- size: 23122 chars / ~5781 tokens, 51 paragraphs, 32 chunks

### 11. A VISION FOR YOU (`chapter-11`)

- topics: outlook, fellowship vision, ongoing practice, hope
- orientation: Closing vision of the fellowship's purpose and the life recovery makes possible. Good for questions about what comes next, and as a final coverage check against an overly narrow reading of earlier chapters.
- read via: section `chapter-11`, chunks `chapter-11:c001`..`chapter-11:c030`
- consider also: `chapter-2`, `chapter-5`, `chapter-7` (a different perspective on the same problem)
- size: 21944 chars / ~5486 tokens, 57 paragraphs, 30 chunks

## Tool routing

- `book_search(query)` -- rank candidate chunks across the whole corpus;
  start from 2-4 map regions, not one best guess.
- `book_read(chunk_id)` -- read the exact text of one chunk, e.g. `chapter-3:c001`.
- `book_expand(chunk_id, before, after)` -- bounded exact neighbor context
  via chunk `prev_id` / `next_id` links.
- `book_section(section_id)` -- bounded paged read within one section.

## Regenerate

```bash
python3 scripts/fetch_aa_source.py
python3 scripts/build_canonical.py
python3 scripts/build_corpus_structure.py
```
<!-- book-map: versioned routing layer; token counts are reported in
     corpus/structure-report.json and must fit the context budget -->
