"""Regression: live AA agent provider access denial must retire its route."""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/aa-opencode-model-route-e2e.yml"


class ProviderDenialRouteTests(unittest.TestCase):
    def test_provider_denial_fails_closed_without_unreviewed_workaround(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        # The free-tier denial workaround still returned HTTP 403 and must not
        # be classified as a retryable provider failure.
        self.assertNotIn("kind=unavailable", text)
        self.assertIn("provider_failure=$kind", text)
        self.assertIn("steps.live_aa.outputs.provider_failure != ''", text)
        self.assertIn("operation: fail", text)
        self.assertIn("failure-reason: ${{ steps.live_aa.outputs.provider_failure }}", text)

    def test_denial_is_not_reclassified_as_qualified_success(self) -> None:
        text = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("AA agent live call failed; provider_error=", text)
        self.assertIn("exit 1", text)
        self.assertIn("Attest real AA task completion", text)
        self.assertNotIn("continue-on-error: true", text)


if __name__ == "__main__":
    unittest.main()
