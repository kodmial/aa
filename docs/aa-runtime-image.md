# Prebuilt AA dependency runtime (issue #298)

## Separation of code and environment

`docker/Dockerfile.aa-runtime` builds the reusable Linux/amd64 environment
(Python 3.12, native runtime Python libraries, ffmpeg/Opus, pinned OpenCode).
It does **not** embed AA application code, prompts, private book text, data,
secrets, session state, or public model weight blobs.

The image build context is **only `./docker`**, restricted by
`docker/.dockerignore` to `Dockerfile.aa-runtime` and
`opencode.version`. Do not switch context to the repository root.
At bot startup `actions/checkout` supplies the exact current AA SHA;
`pip install -e . --no-deps` connects its code to prebuilt dependencies.
The runtime verifies that SHA and refuses an untrusted or missing image.

## Public models and sensitive source data

E5, GigaAM, Silero and the presentation classifier stay outside the base
environment image and use content-addressed Actions caches and lockfile
identity checks. A cache hit avoids model download; a miss downloads and
validates before the bot can become READY. A cache miss must not return a
false PASS. Measure cold/hot image pulls alongside model restore/download,
then decide whether separate model layers would improve *total* startup.
Do not assert that all model caches are already warm. Review the actual
model licenses before redistributing model weights in any future image.

Book/corpus plaintext, keys, decrypted retrieval indexes, user messages,
Telegram audio and sessions never enter Docker build context or image layers.
The existing encrypted derived-retrieval cache remains a separate runtime
artifact and must be validated after restore.

## Safe two-stage activation

**Stage A — build without affecting existing bot sessions:**

1. PR CI builds an image locally and MUST pass native Python import, OpenCode,
   and codec smoke checks. It publishes nothing and does not start Telegram.
2. After the PR merges, trusted `main` push builds using Buildx registry
   layer cache, publishes `ghcr.io/kodmial/aa-runtime:sha-<commit>`, and
   records an immutable `sha256:...` image digest. No mutable `:latest`
   tag is used by the bot.
3. A **separate GitHub-hosted runner** pulls the image by digest and runs the
   canary with no Telegram poller and no private secrets. An image failing
   canary cannot be activated.

**Stage B — automatic, reviewed promotion:**

4. Following a successful image canary, the workflow (using the existing
   `TAP_PAT` required for PR-triggered CI) runs
   `scripts/promote_runtime_image.py --digest sha256:<digest> --apply`.
   It creates a separate normal PR updating `docker/aa-runtime.digest`,
   `aa-runtime.yml`, both qualification jobs in
   `aa-self-proving-qualification.yml`, the trusted derived-index
   publisher `aa-derived-index.yml`, the bootstrap metrics consumer
   `aa-bootstrap-metrics.yml` and the container Git trust regression
   `aa-container-git-trust-regression.yml` together (issues #429 and
   #441: every AA job kind shares the exact same prepared environment —
   no mixed digests). No direct writes to protected `main`.
   `scripts/check_runtime_image_consumers.py` (run by the repository test
   suite) fails the build when any AA workflow pins any other
   `aa-runtime` digest, when a consumer is missing from the promotion
   contract, or when the static `--pins-only` and actual
   binary-availability `--tools-only` preflight pair is dropped.
5. CI and automated review must pass before that PR merges. Until this
   stage merges, the **old ubuntu-latest AA runtime is unchanged** and
   `/run` continues to work.
6. Promotion removes repeated setup-python and uses the prebuilt package
   layer, but still validates a synthetic ffmpeg encode/decode, OpenCode
   pin, current AA checkout, native dependencies and external model identity.
   The existing fixed 5h campaign and single-poller guard are unchanged.
7. A changed dependency manifest during canary triggers a fresh main image
   build rather than promoting stale bytes. Repeated promotion dispatches
   reuse an already open promotion PR for the same digest.

## Prepared qualification/dev toolchain layer (issue #429)

The runtime image also carries the exact pinned qualification toolchain:
`pytest`, `pytest-asyncio`, `ruff`, `mypy`, `click`, `typer`, `spacy` and
the `ru-core-news-sm` RU NLP model. Gate A/B (ruff/mypy/pytest) in
`aa-self-proving-qualification.yml` runs inside the same image, so no
qualification job reinstalls tools on the hot path. The baked versions
must stay compatible with the selector ranges in
`pyproject.toml [project.optional-dependencies]`; the image canary and
`scripts/verify_runtime_image.py --pins-only` fail closed on drift.

Both qualification jobs validate the prepared environment before work:
static pins check (`verify_runtime_image.py --pins-only`), the actual
binary availability check (`verify_runtime_image.py --tools-only`: pinned
`pytest`/`ruff`/`mypy` console scripts, the spaCy RU model and the exact
`docker/opencode.version` OpenCode pin — a green `--pins-only` alone never
validated the selected container; issue #441), then
`pip install -e . --no-deps` (exact checked-out source only). The reusable
`.github/actions/aa-opencode-toolchain` action
remains for the legacy hosted-runner workflows only; `aa-runtime.yml` and
`aa-self-proving-qualification.yml` consume the pin from the image.

## Rollback and failure behavior

The live job uses the image **only by immutable digest**. A revoked/inaccessible
GHCR package or image mismatch fails before starting OpenCode/Telegram;
it must not emit READY or open a second Telegram poller. Revert the activation
PR to restore the earlier `ubuntu-latest` job; the dependency installation
cold path remains available. Existing sessions cannot have their image
changed mid-flight.

The image build needs `packages: write`, and the runtime needs
`packages: read` plus GHCR credentials. To generate a CI-triggering
promotion PR, the workflow needs `TAP_PAT`; if absent it must fail clearly,
not pretend that the image is activated.

## Measurement and completion

A published image digest and successful dependency canary prove only
infrastructure readiness, not Product Contract Gate C/F PASS. Record
the complete elapsed time from `/run` to an authoritative `READY` marker
for both legacy and digest-based runtime, including image pull, public model
cache restore and corpus preparation. State measured values and cache hit/miss
conditions; do not invent performance improvements.

An owner can invoke `AA runtime image build/publish` manually to rebuild
from trusted main. Rebuild on dependency/tool/lock changes; ordinary edits
to AA conversation code are loaded by `checkout`, not by rebuilding
the image.
