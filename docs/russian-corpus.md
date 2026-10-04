# Russian canonical corpus (issue #50)

Status: implementation for issue #50. Production snapshot activation is
tracked in #28 and uses the existing `AA_BOOK_AGE_IDENTITY` / shared public
recipient (no second password/key).

## Goal

Build the authoritative machine-readable Russian AA Big Book canonical
corpus from the selected Fourth Edition TXT and publish the encrypted
Russian snapshot for production.

## Fixed canonical source: TXT

The single canonical text input is the normalized TXT
`corpus/source/raw-ru/aa-big-book.txt` (ignored by Git, preserved
byte-for-byte, pinned by raw SHA-256 in `corpus/canonical.ru.manifest.json`
and `corpus/source/fetch-ru-state.json`).

Required identity (validated deterministically from the TXT preamble):

- title: `АНОНИМНЫЕ АЛКОГОЛИКИ`;
- Fourth Edition (`4-е издание` / `Четвертое издание` / `FOURTH EDITION`);
- Alcoholics Anonymous World Services, Inc. / Фонд «Единство»;
- Russian publication lineage: 2013;
- ISBN: `978-5-906531-01-8`;
- permission line: `с разрешения Alcoholics Anonymous World Services`;
- scope: Мнение доктора + Chapters 1-11.

Owner confirms permission for project use/storage; there is no licensing
blocker. PDF extraction and OCR are never part of the canonical path and
never block this task.

The one-time trusted TXT bootstrap assembles `aa-big-book.txt` from the
text-native provider pages pinned by `corpus/source.ru.lock.json`
(`https://aarus.fi/read/bigbook/`, edition page plus `nXXVII`/`n1`…`n147`);
the canonical build itself reads only the TXT (a second acquisition path
is never introduced).

## TXT layout

UTF-8, LF newlines, Unicode NFC, no BOM (a single leading BOM is stripped
deterministically on read):

```text
<edition/identity preamble with all markers above>

@@SECTION:doctors-opinion|Мнение доктора@@

<МНЕНИЕ ДОКТОРА + literary blocks, blank-line separated>

@@SECTION:chapter-1|Глава 1. Рассказ Билла@@

<ГЛАВА 1 / РАССКАЗ БИЛЛА + literary blocks>
...
```

Sections appear exactly once in the order above (Мнение доктора, Chapters
1-11); headings and fixed control passages (beginning/middle/end) are
validated verbatim. Only NFC/BOM/newline transport artifacts are
normalized; spelling, grammar, punctuation, and wording are never altered.

## Build

```bash
python3 scripts/fetch_ru_source.py --bootstrap-from-provider
python3 scripts/build_canonical_ru.py
```

Outputs (both ignored by Git, SHA-pinned in the manifest):

- `corpus/generated/canonical.ru.json` (`aa-canonical-ru/1`);
- `corpus/generated/canonical.ru.txt` (deterministic section rendering).

Every section carries `source_file`, `source_sha256`, `char_start`/`char_end`
offsets into the canonical text, `text_sha256`, and `chars`. Wrong edition,
wrong ISBN, missing/reordered sections, undecodable/corrupt TXT, or source
drift fails closed with no artifact written.

## Encrypted storage

The Russian snapshot reuses the #24 age pipeline and the shared recipient
(`corpus/source/encrypted/recipient.txt`):

```bash
python3 scripts/refresh_encrypted_snapshot.py \
  --manifest corpus/canonical.ru.manifest.json \
  --source-lock corpus/source.ru.lock.json \
  --canonical corpus/generated/canonical.ru.json \
  --recipient-file corpus/source/encrypted/recipient.txt \
  --archive-name canonical.ru.tar.zst.age \
  --metadata-name metadata.ru.json \
  --canonical-name canonical.ru.json
```

Only `canonical.ru.tar.zst.age` + `metadata.ru.json` (+ the shared public
recipient) are committed. Encrypt -> decrypt -> exact canonical SHA-256 is
verified before publication. Runtime decrypts once per startup/job
(`python3 scripts/restore_canonical.py --lang ru`) and reuses the restored
artifact; never per Telegram message.
