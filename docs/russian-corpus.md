# Russian Big Book corpus (issue #50)

Status: implementation for issue #50. The encrypted production snapshot is
published by the trusted refresh workflow; until it exists, restore uses the
deterministic network fetch/build fallback only when explicitly allowed, and
fails closed otherwise.

## Goal

Russian canonical evidence is **mandatory for production**. The production
bot must not depend on LLM translation of the English Big Book for normal
Russian quotations or Russian-source retrieval.

## Edition and rights

- Title: `АНОНИМНЫЕ АЛКОГОЛИКИ (Четвертое издание)`.
- Russian Fourth Edition translation: `Фонд «Единство», 2013`,
  ISBN `978-5-906531-01-8`.
- Catalog: Russian Big Book, SKU `RUSSB-30`, General Service
  Conference-approved, Russian, softcover. (The AAWS catalog lists 228
  pages; the Russian imprint states 192 pages.)
- Rights basis: the owner confirms permission for the Russian edition. No
  additional licensing approval is required for this implementation.
- Constraints: private server-side retrieval/indexing, bounded passage
  retrieval per request, no public redistribution of the source file,
  Russian-language AA support bot, immutable source storage with
  checksum/version controls.
- Exact edition/SKU/rights metadata lives in `corpus/source.ru.lock.json`;
  pinned checksums live in `corpus/canonical.ru.manifest.json`.

## Source

Text-native Russian Fourth Edition pages served by Группа АА «Контакт»:

- edition/root validation: `https://aarus.fi/read/bigbook/edition/`;
- Doctor's Opinion: `https://aarus.fi/read/bigbook/nXXVII/`;
- Chapters 1-11: `n1`, `n16`, `n29`, `n43`, `n56`, `n70`, `n86`, `n101`,
  `n118`, `n132`, `n147` (section ids are print-page based, not 1-11).

Each page is accepted only after validating the title
(`АНОНИМНЫЕ АЛКОГОЛИКИ`), Fourth Edition / Russian translation metadata,
AAWS attribution, `Фонд «Единство», 2013`, ISBN `978-5-906531-01-8`, and
the exact required headings for Мнение доктора and Chapters 1-11.

The Conference-approved `BigBook-4th.pdf` linked by the same AA group (see
`https://aarus.fi/books/`) is a verification/control source only. OCR is
never the primary pipeline while the text-native source is available.

## Fetch/build

Deterministic Russian source path parallel to English:

```bash
python3 scripts/fetch_ru_source.py
python3 scripts/build_canonical_ru.py
```

- `scripts/fetch_ru_source.py` reads `corpus/source.ru.lock.json`, fetches
  with bounded timeouts (20s), retries (3, backoff), and content-size limits
  (5 MiB per file), saves raw downloads only under the ignored
  `corpus/source/raw-ru/` workspace, and records SHA-256 metadata in the
  ignored `corpus/source/fetch-ru-state.json`.
- `scripts/build_canonical_ru.py` reads the raw files read-only, validates
  them against the committed `corpus/canonical.ru.manifest.json` plus
  fetch-state (SHA-256, byte lengths, paths), re-validates the edition
  markers, extracts Russian literary text from the `html` field of each
  `book-initial` block (wording untouched; tags stripped, entities
  unescaped, whitespace collapsed; `<br/>` verse breaks preserved as line
  breaks), and discards navigation/UI, English parallel text (`english`),
  alternative renderings (`alternatives`), and page markers.
- Scope is exactly Мнение доктора + Chapters 1-11. The trailing `ЧАСТЬ 1`
  divider on the Chapter 11 page opens the excluded stories part and is
  dropped deterministically.
- Every section preserves source URL, page section id, source checksum,
  block ids, page numbers, and stable char offsets/provenance.
- Any missing/moved section, edition mismatch, unexpected page structure,
  or checksum/version mismatch fails closed with a non-zero exit status
  and no artifact is written. The artifact is
  `corpus/generated/canonical.ru.json` (ignored by Git).

## Encrypted snapshot and restore

The design reuses the #24/#28 age encryption architecture and the same
`AA_BOOK_AGE_IDENTITY` secret. There is no second private key/password.

Committed (public) Russian snapshot files, independently versioned from
English:

```text
corpus/source/encrypted/
  canonical.ru.tar.zst.age   # age-encrypted tar.zst: canonical.ru.json + provenance.json
  metadata.ru.json           # non-secret metadata (canonical/encrypted SHA-256, versions)
  recipient.txt              # shared public age1... recipient (same as English)
```

Refresh (trusted context only; verifies decrypt -> SHA-256 round trip
before writing anything):

```bash
python3 scripts/refresh_encrypted_snapshot.py \
  --manifest corpus/canonical.ru.manifest.json \
  --source-lock corpus/source.ru.lock.json \
  --canonical corpus/generated/canonical.ru.json \
  --encrypted-dir corpus/source/encrypted \
  --recipient-file corpus/source/encrypted/recipient.txt \
  --archive-name canonical.ru.tar.zst.age \
  --metadata-name metadata.ru.json \
  --canonical-name canonical.ru.json
```

Or dispatch the manually triggered
`.github/workflows/encrypted-corpus-refresh-ru.yml` workflow.

Restore (single entry point alongside English; decrypts once per job at
startup and reuses the local copy; never decrypts per Telegram message):

```bash
python3 scripts/restore_canonical.py --lang ru
python3 scripts/restore_canonical.py --lang ru --allow-network-fallback
```

Raw downloaded HTML/PDF is never committed. It exists transiently during
trusted refresh and is deleted with the job workspace.

## Production source policy

Russian is the primary source/evidence language for Russian users
(`src/aa/corpus/canonical.py::load_canonical_ru`). English remains a
separately versioned control/reference corpus. Generated translation must
never be silently substituted for a source-exact Russian quotation in
production.
