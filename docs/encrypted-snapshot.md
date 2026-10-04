# Encrypted canonical-book snapshot and deterministic restore

Status: implementation for issue #24. Production key/secret/snapshot
activation is deferred to #28 and must not block #8 or downstream
corpus/retrieval development.

## Goal

After #3 produces the validated canonical AA artifact, keep a reproducible
encrypted snapshot so later tasks and runtimes do not depend on the external
source sites being reachable on every run. This is an access/reliability
mechanism, not a replacement for the canonical source/provenance contract.

## Storage model

Only the encrypted snapshot is committed:

```text
corpus/source/encrypted/
  canonical.tar.zst.age
  metadata.json
  canonical.ru.tar.zst.age
  metadata.ru.json
  recipient.txt
```

Never committed: the private identity, the plaintext book, the plaintext
canonical artifact, or a plaintext retrieval index.

`metadata.json` records at least the canonical artifact SHA-256, the
encrypted archive SHA-256, the source/manifest version, the encryption
format/version, and creation/update instructions. The Russian counterparts
(`canonical.ru.tar.zst.age` + `metadata.ru.json`, issue #50) use the same
age recipient/identity contract with independently versioned artifact
metadata/checksums; see `docs/russian-corpus.md`.

## One-time production setup (owner action, tracked in #28)

Run locally on a trusted machine. Never generate the production key inside
an ordinary repository workflow, and never grant the default `GITHUB_TOKEN`
secret administration.

```bash
python3 scripts/age_provision.py \
  --write-recipient corpus/source/encrypted/recipient.txt
```

1. Copy the printed `AGE-SECRET-KEY-...` value into the repository secret
   named `AA_BOOK_AGE_IDENTITY` (Settings -> Secrets and variables ->
   Actions). The secret contract is the only representation of the private
   identity in this repository.
2. Commit only `corpus/source/encrypted/recipient.txt` (the public
   `age1...` recipient).
3. Dispatch the `encrypted-corpus-refresh` workflow to build, encrypt,
   verify, and publish `canonical.tar.zst.age` + `metadata.json`.
4. Delete any local copy of the private identity after storing it.

## Build/update workflow

`.github/workflows/encrypted-corpus-refresh.yml` is manually dispatched
(`workflow_dispatch`) from a trusted repository context and:

1. restores/uses `AA_BOOK_AGE_IDENTITY` only for validation when required;
2. runs the #3 canonical source bootstrap (`fetch_aa_source.py`,
   `build_canonical.py`);
3. verifies canonical SHA/version against `corpus/canonical.manifest.json`;
4. packs the exact canonical artifact plus minimum provenance metadata;
5. encrypts it with the committed public age recipient;
6. verifies that decrypting with the secret identity reproduces the
   expected canonical SHA;
7. updates only the encrypted archive + non-secret metadata in Git;
8. keeps logs free of plaintext corpus and private key material (only
   paths, sizes, and SHA-256 digests are logged).

## Runtime restore

`scripts/restore_canonical.py` is the single canonical bootstrap/restore
entry point (`--lang ru` selects the Russian artifact from
`corpus/canonical.ru.manifest.json` into
`corpus/generated/canonical.ru.json` via `canonical.ru.tar.zst.age`).
Later tasks must call it instead of inventing their own
source-loading path.

Runtime order:

1. Reuse a valid decrypted `corpus/generated/canonical.json` already
   present in the current runner workspace.
2. Otherwise, when the encrypted snapshot exists and
   `AA_BOOK_AGE_IDENTITY` is available, decrypt it into
   `corpus/generated/canonical.json`.
3. Verify SHA/version against the committed manifest (artifact SHA plus
   per-section validation in `src/aa/corpus/canonical.py`).
4. Otherwise fall back to the deterministic #3 network fetch/build only
   when explicitly allowed (`--allow-network-fallback` or
   `AA_ALLOW_NETWORK_FETCH=1`).
5. Fail closed when no verified canonical artifact can be obtained.

Call it once per job at startup and reuse the restored copy for all
requests. Never download or decrypt the book per Telegram message.

```bash
python3 scripts/restore_canonical.py
python3 scripts/restore_canonical.py --allow-network-fallback
AA_BOOK_AGE_IDENTITY="$(cat /run/secrets/age)" python3 scripts/restore_canonical.py
```

## Cache policy

- Do not put plaintext book text or a plaintext text-bearing retrieval
  index into GitHub Actions cache.
- The committed encrypted snapshot is the durable cross-run cache for
  this small corpus.
- Inside one running job, reuse the decrypted
  `corpus/generated/canonical.json` and the derived in-memory/on-disk
  index for all user requests.
- A future cross-run Actions cache may contain only non-sensitive
  metadata or encrypted payloads unless a separate security review says
  otherwise.

## Public derived material

Public, versioned derived artifacts remain expected and encouraged
(`corpus/book-map.md`, `corpus/structure.json`, evaluation questions,
topic labels, summaries, short necessary snippets). They are
navigation/evaluation material, not a substitute copy of the canonical
literary text.
