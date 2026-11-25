# One-shot retrieval A/B validation trigger (kodmial/aa#553).
#
# A push to main that changes this file runs the bounded real-book
# one-shot retrieval A/B diagnostic exactly once for the resulting
# exact-main SHA. It is validation evidence only: it never changes the
# production flag default, never admits #538/#514/#7 and never resets a
# 429/recovery/repair/convergence cap.
#
# Explicit owner re-runs use the existing workflow_dispatch with
# mode=one-shot-ab. The same exact SHA + configuration is deduplicated by
# the script's durable COMPLETE marker on issue #553.
entry=one-shot-ab
issue=553
