# Persistent AA derived retrieval index (issue #429)

Two cache levels protect against a full E5 re-embed when the Actions
(L1) derived retrieval cache is evicted:

- **L1** — the existing exact-content, encrypted GZip Actions cache
  (`aa-derived-retrieval-v1-<OS>-<ARCH>-<digest>`). Save/restore lives
  on the same timing as issue #58 and stays best effort.
- **L2** — the same encrypted bundle published as one immutable OCI
  artifact `ghcr.io/kodmial/aa-index` (tag `idx-<cache_digest>`,
  artifact type `application/vnd.aa.derived-index.artifact.v1`,
  layer media type `application/vnd.aa.derived-index.bundle.v1`).
  Only the `docker/aa-index.digest`-committed `ghcr.io/kodmial/aa-index@sha256:<64-hex>`
  reference authorizes a pull; mutable tags never validate a restore.

## Cache keys and indices

The derived-cache key stays purely content-addressed: schema version,
RU/EN canonical artifacts, corpus structure, retrieval production
config, planner schema, golden evidence, E5 embedding identity/revision,
embedding lock, ABI (Python/SQLite/index-format/planner/RRF), and
OS/arch. The new bindings `production_dense_backend` (always
`intfloat-multilingual-e5-base/1`) and `production_dense_dim` (always
768) changed the key, so old bundles miss safely.

Rotation (`AA_DERIVED_CACHE_ROTATION`) appends `-rot-<N>` **only** to the
L1 Actions cache key; the encrypted bundle's internal manifest
`cache_key` remains the base key and self-validates after any restore.

## Strict E5 backend enforcement

`src/aa/retrieval/derived_cache.py::assert_required_backend` reads the
`embedding_backend`/`embedding_dim` metadata from `index.json` inside
the restored bundle and rejects anything that does not prove the
production `e5` backend (E5 dimensions). The hashing semantic-cache
backend is offline-only. Every restore path
(`restore`, `restore-or-rebuild`, `restore-l2`) accepts
`--require-backend {none,hashing,e5}`; production AA jobs always pass
`--require-backend e5` and `--backend e5`, so a restored hashing bundle
is never accepted as a production hit and is rejected with the reason
`wrong-backend`. `publish-l2` refuses non-E5 backends outright.

## Publication and consumption flows

- `scripts/manage_derived_cache.py --mode publish-l2` (trusted
  `.github/workflows/aa-derived-index.yml` only, `packages: write`):
  strict E5 `restore-or-rebuild`, encrypt/package, then an idempotent
  `push_artifact` (HEAD-before-push; same manifest digest short-circuits
  the write; the digest-pinned bundle is re-verified end-to-end before
  writing any file into the cache directory). A promotable digest
  report and a content fingerprint then open one reviewed PR via
  `scripts/promote_derived_index.py` (TAP_PAT required; direct main
  writes are refused).
- Consumers (`aa-runtime.yml`, both `aa-self-proving-qualification.yml`
  jobs, and the measurement workflow) bootstrap in the fixed
  L1 -> L2 -> rebuild order (issue #461): the actual restored L1
  directory is validated first with `--mode restore --require-backend
  e5` (never merely trusted from the Actions-restore output), then —
  only after a missing/rejected L1 — the digest-pinned L2 artifact is
  pulled into an isolated L2-owned staging directory, and only when
  both fail does `restore-or-rebuild --backend e5 --require-backend e5`
  run the deterministic bounded emergency rebuild. Verified L1/L2 hits
  skip redundant checks and embeddings; a verified hit emits the typed
  statuses `l1-hit`/`l2-hit`/`rebuilt`/`failed` (plus non-secret reason
  categories and per-source timings) into the step summary. L2
  download, decryption, verification and cleanup only ever touch the
  staging directory (`--staging-dir`, default `<cache-dir>.l2-staging`
  outside the checkout): an L2 401/403, transport failure, partial
  blob, corrupt artifact, bad manifest/backend, decryption or
  self-test failure can never mutate, overwrite or delete the restored
  L1 directory. `actions/cache/save` stays exact-key immutable: valid
  L1 hits skip re-save, and a corrupt L1 entry is never claimed as
  repaired by overwriting the same key (rotation via
  `AA_DERIVED_CACHE_ROTATION` is an owner action only).
- `docker/aa-index.digest` starts as the honest
  "L2 publication pin is still a placeholder" placeholder
  (`sha256:000…0`): `parse_pinned_digest` raises a fail-closed category,
  no bundle is fetched, and consumers surface `l2-unpublished` while the
  trusted publisher's promotion PR is still open.

## Trust and privacy

The registry-v2 client (`src/aa/retrieval/oci_artifact.py`) is stdlib
only. Identity = a strict immutable digest; locator tags may not mix
run ids, dates, branches, or source SHAs. Basic/Bearer auth is resolved
through the standard WWW-Authenticate challenge (realm host must match
the registry host, fail closed). Every blob and manifest is verified
against its sha256 digest before writing; only the exact expected file
set (`derived-retrieval.tar.zst.age`, `derived-retrieval.meta.json`)
is atomically restored (`<name>.l2tmp` then rename) after full
verification, and a failed verification restores nothing.

Annotations are non-secret metadata only: cache key/fingerprint, bundle
format, chunk/index version. No book plaintext, decrypted artifact, or
ephemeral runtime identity (run id, timestamp, date, branch, ordinary
source SHA) may enter the publish path.

## Boot timing metrics

`.github/workflows/aa-bootstrap-metrics.yml` runs a clean-runner
`legacy-path` job (hosted runner + setup-python + `pip install -e ".[dev]"`)
and a `prepared-image-path` job in the promoted container image reporting
the L2 restoration route (`L2 GHCR restore (strict E5)`). Both write
phase timings to `GITHUB_STEP_SUMMARY` plus JSON artifacts; the
prepared-image leg surfaces the real `l2_state` and never picks a fake
success when the L2 pin is still a placeholder.
