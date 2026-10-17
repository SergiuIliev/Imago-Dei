"""
Unit and Integration Tests for Local-Only Operation and Privacy Security Invariants.
Verifies loopback binding, zero cloud telemetry, and absence of external network calls.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig


class TestPrivacyAudit(unittest.TestCase):

    def setUp(self):
        self.config = CapsuleConfig()

    def test_default_security_invariants(self):
        """Verifies default configuration enforces strict local-only execution and zero telemetry."""
        self.assertTrue(self.config.security.local_only)
        self.assertTrue(self.config.security.disable_telemetry)
        self.assertEqual(self.config.security.bind_host, "127.0.0.1")
        self.assertFalse(self.config.security.allow_remote_tunnels)

    def test_provider_endpoint_invariants(self):
        """Verifies that all AI provider endpoints default strictly to loopback addresses."""
        self.assertTrue(
            self.config.llm.endpoint.startswith("http://127.0.0.1") or
            self.config.llm.endpoint.startswith("http://localhost"),
            f"LLM endpoint must be local loopback, got {self.config.llm.endpoint}"
        )

    def test_no_hardcoded_cloud_endpoints(self):
        """Scans codebase for forbidden telemetry and cloud inference domains."""
        forbidden_strings = [
            "api.openai.com",
            "api.anthropic.com",
            "telemetry.posthog.com",
            "google-analytics.com",
            "segment.io"
        ]

        src_dirs = [PROJECT_ROOT / "core", PROJECT_ROOT / "memory", PROJECT_ROOT / "app", PROJECT_ROOT / "providers"]
        for d in src_dirs:
            for p in d.rglob("*.py"):
                with open(p, "r", encoding="utf-8") as f:
                    content = f.read()
                    for forbidden in forbidden_strings:
                        self.assertNotIn(
                            forbidden,
                            content,
                            f"Forbidden cloud/telemetry endpoint '{forbidden}' found in {p}"
                        )


if __name__ == "__main__":
    unittest.main()
