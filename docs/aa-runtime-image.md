# AA prebuilt runtime image (issue #298)

## What it is

`docker/Dockerfile.aa-runtime` builds a reproducible Linux amd64 /
Python 3.12 job-runtime image published to `ghcr.io/kodmial/aa-runtime`
and consumed by `.github/workflows/aa-runtime.yml` **only by immutable
digest** (`docker/aa-runtime.digest`). It replaces repeated installation
of ffmpeg/Opus, the full Python dependency resolve (Torch, sherpa-onnx,
ONNXRuntime, retrieval stack), and the OpenCode curl install on every
bounded bot session.

The exact AA application source is never frozen into the image: the
runtime workflow checks out exact main and installs the project with
`pip install -e . --no-deps` (seconds) on top of the pinned dependency
layer.

## Model distribution decision (measured)

Public model weights are intentionally **not baked** into the image.
They keep flowing through the locked `actions/cache` + checksum-verified
prefetch path (`corpus/embedding.lock.json`, `corpus/voice.lock.json`):

- E5 embedding snapshot (~1.1 GB), GigaAM large INT8 (~1-2 GB), Silero
  TTS (~0.1 GB), and the presentation classifier would bloat every image
  pull by gigabytes, dwarfing the dependency/tooling layers (~2-3 GB
  total) and slowing cold start more than the cache path on hosted
  runners (registry bandwidth, disk/IO, GHCR pull limits).
- `actions/cache` restores are content-addressed by exact revision locks
  with `--check-only` prune-and-redownload semantics, so the validated
  hot path performs **zero downloads** while a cold miss rebuilds
  deterministically within bounds.
- Baking weights into independent Docker layers was evaluated and
  rejected for now: it couples large binary churn to the dependency
  layer cache, complicates license redistribution review per asset, and
  risks caching private material. Revisit only with measured cold-pull
  numbers showing a net win.

Before/after timing is reported by `scripts/verify_runtime_image.py`
(timings_ms), the image workflow step summary (image size, digest,
provenance), and the existing voice/prefetch summaries
(downloaded_bytes vs hit). No speed claim is valid without those three.

## Rebuild

Push to `main` touching the Dockerfile, `docker/opencode.version`,
`pyproject.toml`, model locks, or the image workflow triggers
`.github/workflows/aa-runtime-image.yml`, which builds with BuildKit
registry-layer caching, validates (ffmpeg/libopus, native imports,
pinned OpenCode, secret scan, provenance), pushes by digest, and runs a
fresh-runner canary that never starts a Telegram poller. Pull requests
build and validate without pushing. Owner-only `workflow_dispatch`
is available for prewarming.

## Digest promotion

1. Wait for the image workflow canary on exact main to pass.
2. Copy the published digest from the step summary or the
   `aa-runtime-image-digest` artifact.
3. Run `python scripts/promote_runtime_image.py --digest sha256:<hex> --apply`
   to update `docker/aa-runtime.digest` and the `aa-runtime.yml`
   container pin together.
4. Commit as one change; CI must pass.

## Rollback

One commit: delete the `container:` block from
`.github/workflows/aa-runtime.yml` to return to the legacy
`ubuntu-latest` runner. The apt-get, full pip install, and OpenCode
curl branches remain as the bounded cold-miss path, so rollback needs
no other edits. A missing/inaccessible image or digest mismatch fails
the job at container creation before any poller starts (never READY,
never a duplicate poller).

## Safety

No Telegram token, PAT, provider token, age identity, canonical EN/RU
plaintext, decrypted retrieval index/corpus, chats, voice/PCM/OGG,
transcripts, generated responses, or user/session data enters the build
context, layers, logs, or public caches (`docker/.dockerignore` plus
the workflow secret scan). Encrypted derived retrieval keeps its
verified content-addressed cache/restore path.
