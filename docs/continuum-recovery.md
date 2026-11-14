# AA Continuum/OpenCode recovery policy (issue #515)

This note records how the AA callers of the Continuum reusable workflows
behave on current Continuum `@main`, and the audit that produced the
platform reconciliation in issue #515. It supersedes the four separate
proposals #505–#508.

## One pinned runtime image and index

- `docker/aa-runtime.digest` is the only runtime-image promotion source.
  Every prepared-environment consumer (`aa-runtime.yml`, both jobs in
  `aa-self-proving-qualification.yml`, `aa-derived-index.yml`,
  `aa-bootstrap-metrics.yml`, and `aa-container-git-trust-regression.yml`)
  must carry exactly that digest. `scripts/check_runtime_image_consumers.py`
  and the repository tests fail closed on any mixed or retired digest.
- The active image is the canary-verified
  `ghcr.io/kodmial/aa-runtime@sha256:72802fe8025dc7343e1a4dca3c156b20d7f79bd3bbab52a874628fd248e8c066`
  from build run [42837651521](https://github.com/kodmial/aa/actions/runs/42837651521)
  (fresh-runner canary job green). Its baked `/opt/aa/provenance.txt`
  records Python `3.12.16`, OpenCode `v2.2.6`, pytest `9.1.1`,
  ruff `0.17.2`, mypy `2.4.0`, the `ru_core_news_sm` model and successful
  native imports. That matches `docker/opencode.version` (`2.2.6`) and the
  Dockerfile pins on current main; the Dockerfile has not changed since the
  image build.
- The previously pinned image
  `sha256:c2d49cfc8f89f2e2caf8463f771fecb6db2a4e26977787867bcb6eb8c4603ffd`
  was built with OpenCode `1.19.3`, so it is a retired artifact. The tests
  keep that digest named as a regression guard and reject its return.
- `docker/aa-index.digest` remains the validated L2 artifact it already
  pointed at. The proposed #507 digest
  `sha256:6a537d9d19e7…` also exists (GHCR version created
  2026-11-10, tag `idx-f29fdd587beb0ba2…`), but its metadata has the same
  content-addressed cache digest, embedding revision
  `d128750597153bb5987e10b1c3493a34e5a4502a`, EN/RU artifact hashes and
  `structure_sha256` (`23e0791d…`, equal to the committed
  `corpus/structure.json`). It is an equivalent rebuild, not a corpus
  change. Promoting it would churn the pin without changing the corpus,
  so #507 is discarded; the existing pin is not stale.

## Scheduler and OpenCode pause policy

- The AA issue scheduler is pinned to non-terminal convergence
  (`pause_on_failure: "false"`), so a failed dispatch must return the issue
  to the scheduler lease/backoff instead of entering a human-gated pause.
- The OpenCode caller now detects a scheduler dispatch marker on an
  `issue_comment` run and forwards `pause_on_failure: "false"` for that run.
  Marker resolution matches the scheduler exactly: explicit dispatch input,
  then the `AUTOMATION_DISPATCH_MARKER` repository variable, then the
  upstream `<!-- issue-scheduler-dispatch -->` default. Manual owner
  `/oc`/`/opencode` comments, review comments and explicit
  `workflow_dispatch` calls keep the bounded terminal-pause default, and an
  explicit dispatch input still overrides it. Without this, the OpenCode
  recovery job added `automation:paused` while the scheduler kept clearing
  it and re-dispatching the issue: a pause/unpause/re-dispatch loop (#506).
- `max_dispatch_attempts` stays at the reviewed caller default (`5`); the
  change does not reset dispatch counters or weaken the repair budget.

## Watchdog recovery and genuine 429 handling

- There is exactly one AA caller of the Continuum OpenCode watchdog
  (`continuum-opencode-watchdog.yml`) and it watches the `OpenCode agent`
  workflow name. The scheduler also wakes on that workflow completion, but
  only to reconcile lease/labels; it does not dispatch a competing
  recovery, and the watchdog stands down when a newer scheduler dispatch
  exists. This prevents duplicate watcher loops.
- The watchdog keeps a finite retry budget: `max_recovery_retries: "3"`.
  The OpenCode runner-restart 429 budget is explicitly pinned to the
  reviewed upstream default `max_429_runner_restarts: "3"` so a mutable
  repository variable cannot silently disable or unboundedly widen genuine
  fresh-VM 429 recovery.
- `pause_on_failure: "false"` makes watchdog exhaustion hand control back
  to the scheduler's bounded lease/backoff instead of adding a human pause.
- The 429 path is the Continuum checkpoint contract
  (`opencode/429-checkpoint-*` plus `.continuum-429-recovery.json` and
  `.continuum-429-stage` in the interrupted worktree). It is resolved by
  the workflow-owned recovery identity; AA has no dependency on the
  discarded #454 checkpoint. The ignored `.continuum-429-stage` is never
  committed by the implementation, and `.continuum-verification.json`
  remains workflow-owned baseline/verifier/final evidence.
- The watchdog recovery job keeps the current upstream 5-minute default.
  Audit of the most recent 100 AA watchdog runs found no timeout
  cancellation: accepted runs completed in seconds, and no current job hit
  a concrete 5-minute cap. The proposal in #505 to raise the timeout was
  therefore not ported; raising it would only hide a future hang.

## What must not change

Ruff/format/mypy/pytest coverage, the canonical `bash scripts/verify.sh`,
Gate A/B/C/E/F book grounding, qualification evidence requirements and the
three-cycle repair limit are unchanged. This reconciliation only aligns
artifact pins, scheduler pause policy and bounded recovery budgets.
