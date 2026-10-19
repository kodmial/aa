# AA live model routing — Work Lock #66

The running AA repository is the **consumer**, not the model-policy author.
Only the repository variable `OPENCODE_MODELS` is authoritative in AA
production. The permitted ordered values are owner-approved in
`tests/fixtures/aa-approved-models.json` for E2E/provisioning evidence,
but **no production workflow embeds a model identifier**.

## Owner-controlled provisioning

The `AA owner-approved model policy provisioning` workflow automatically
runs on the **reviewed merge to main** that introduces the model policy
and its approved fixture, and also supports the owner's exact
`/sync-opencode-models` comment on AA issue #391 or a manual dispatch.
It uses the existing `TAP_PAT` credential to **set one repository
Actions variable** and verifies it with GitHub's API. This does not invoke
Telegram. A missing admin scope fails visibly and is never a successful
rollout. Do not run a release qualification before provisioning succeeds.
Changing the order of the repository variable changes routing without
rebuilding Continuum or AA.

## Actual execution

The self-proving qualification, manual bot runtime, and scheduled canary
install their own compatible OpenCode binary, then use Continuum's one
canonical `model-route@main` selector, repository-scoped `github.token`
and separate `continuum-model-state` Git ref. The selected model is
frozen through `GITHUB_ENV` before any AA production side effects.
AA's old in-turn primary/fallback retry is deliberately disabled when
managed routing is active; it cannot substitute a different model after
a turn's non-idempotent actions. Failures must go to the bounded upstream
recovery on a new runner. Gate A rejects a missing/mismatched pool and
does not silently mark unapproved models qualified. Product Contract
#110, real book retrieval, external Telegram requirements, judge/verifier
and performance thresholds remain unchanged.

The independent `aa-model-routing-e2e.yml` exercises the **actual AA
config module**, official anonymous OpenCode file tools, durable state
attestation after successful acceptance, and rediscovery on a fresh
runner, at the AA runtime-pinned OpenCode version. This is independent
of Telegram and never claims the whole AA Product Contract passes.

## Acceptance links and rollback

Record the exact-run E2E, CI and live PR-Agent review evidence in Work
Lock #66 before declaring implementation complete. If any live
qualification fails, retain the failing gate and do not label PASS.
Rollback is by reverting the PR and re-running owner-approved variable
provisioning for the last qualified policy. Do not promote failing
model probes, and never use an unlisted provider or paid model.
