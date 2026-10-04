# Encrypted canonical snapshot (issue #24)

This directory holds the **public, committed** encrypted snapshot of the
canonical AA artifact. It is an access/reliability mechanism, not a
replacement for the canonical source/provenance contract owned by #3.

Planned layout after production activation (tracked in #28;
Russian snapshot in #50, encrypted to the same recipient — never a second
private key/password):

```text
corpus/source/encrypted/
  canonical.tar.zst.age
  metadata.json
  canonical.ru.tar.zst.age
  metadata.ru.json
  recipient.txt
```

- `canonical.tar.zst.age` — age-encrypted deterministic `tar.zst` archive
  containing exactly `canonical.json` (byte-identical to the validated
  `corpus/generated/canonical.json`) plus minimum `provenance.json`.
- `canonical.ru.tar.zst.age` — Russian counterpart (issue #50) containing
  exactly `canonical.ru.json` (byte-identical to the validated
  `corpus/generated/canonical.ru.json`) plus minimum Russian
  `provenance.json`; independently versioned, same recipient.
- `metadata.json` — non-secret metadata: canonical SHA-256, encrypted
  archive SHA-256, manifest/source versions, encryption format/version, and
  creation/update instructions.
- `metadata.ru.json` — Russian counterpart of `metadata.json`.
- `recipient.txt` — public `age1...` recipient shared by both snapshots.
  Safe to commit.

Security rules:

- Never commit the private identity (`AGE-SECRET-KEY-...`), the plaintext
  book, the plaintext canonical artifact, or a plaintext retrieval index.
- The private identity lives only in the `AA_BOOK_AGE_IDENTITY` GitHub
  Actions repository secret.
- Until production activation exists (see #28), no production snapshot is
  present and runtime restore uses the deterministic #3 network
  fetch/build fallback only when explicitly allowed, and fails closed
  otherwise.

Refresh (English):

```bash
python3 scripts/fetch_aa_source.py
python3 scripts/build_canonical.py
python3 scripts/refresh_encrypted_snapshot.py \
  --recipient-file corpus/source/encrypted/recipient.txt
```

Refresh (Russian, issue #50):

```bash
python3 scripts/fetch_ru_source.py
python3 scripts/build_canonical_ru.py
python3 scripts/refresh_encrypted_snapshot.py \
  --manifest corpus/canonical.ru.manifest.json \
  --source-lock corpus/source.ru.lock.json \
  --canonical corpus/generated/canonical.ru.json \
  --recipient-file corpus/source/encrypted/recipient.txt \
  --archive-name canonical.ru.tar.zst.age \
  --metadata-name metadata.ru.json \
  --canonical-name canonical.ru.json
```

Or dispatch the manually triggered
`.github/workflows/encrypted-corpus-refresh.yml` (English) or
`.github/workflows/encrypted-corpus-refresh-ru.yml` (Russian) workflow.

Restore (single entry point for all later tasks):

```bash
python3 scripts/restore_canonical.py
python3 scripts/restore_canonical.py --lang ru
python3 scripts/restore_canonical.py --allow-network-fallback
```

One-time production key provisioning (run locally, never in CI):

```bash
python3 scripts/age_provision.py \
  --write-recipient corpus/source/encrypted/recipient.txt
```

See `docs/encrypted-snapshot.md` for the full procedure and cache policy.
