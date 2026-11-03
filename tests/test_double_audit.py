"""
Unit and Integration Tests for Double Audit Zero Info-Leak System.
Tests Layer 1 (Static Audit), Layer 2 (Runtime & Semantic Leakage Audit),
Dynamic Config Paths, and Sanitization Templates.
"""

from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig, APP_VERSION, get_default_base_dir
from scripts.double_audit import (
    DoubleAuditor,
    sanitize_text,
    audit_rag_context,
    audit_outgoing_payload,
)


class TestDoubleAuditLayer1(unittest.TestCase):
    """Verifies Layer 1 static detection of secrets, PII, and hardcoded home paths."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="tc_audit_test_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_detect_synthetic_secrets(self):
        """Verifies static detection of API keys, tokens, and private keys."""
        bad_file = Path(self.temp_dir) / "synthetic_secrets.py"
        bad_content = """
# Synthetic secrets file for testing
OPENAI_KEY = "sk-proj-1234567890abcdef1234567890abcdef"
ANTHROPIC_KEY = "sk-ant-1234567890abcdef1234567890abcdef"
HF_TOKEN = "hf_1234567890abcdef1234567890abcdef"
GITHUB_KEY = "ghp_1234567890abcdef1234567890abcdef1234"
CLOUDFLARE_JWT = "eyJhIjoiOTdlYjYyN2FjN2M2YzgyM2ExNzMyNGRlOTVhYTM3ZjgiLCJ0IjoiYWUzOWVjOWUtOGQ0OS00OTI4LTgzNzUtODUxOTk0YTAyMTc5IiwicyI6InpqNDdkK2piVjZ0S2RGc0lvK3pPbHlXOExab09XQTEveUpya3Y1ZDI4Nkk9In0="
"""
        bad_file.write_text(bad_content, encoding="utf-8")

        auditor = DoubleAuditor(root_dir=Path(self.temp_dir), strict=True)
        report = auditor.run_audit()

        self.assertFalse(report.passed)
        self.assertGreaterEqual(report.blocking_findings, 4)
        matched_rules = {f.matched_pattern for f in report.findings}
        self.assertIn("openai_api_key", matched_rules)
        self.assertIn("anthropic_api_key", matched_rules)
        self.assertIn("huggingface_token", matched_rules)
        self.assertIn("github_pat", matched_rules)
        self.assertIn("cloudflare_tunnel_jwt", matched_rules)

    def test_detect_synthetic_pii(self):
        """Verifies static detection of private emails, phone numbers, and Wi-Fi credentials."""
        bad_file = Path(self.temp_dir) / "synthetic_pii.txt"
        bad_content = """
Contact support at info@sj-webdesign.com for billing.
Direct personal line: sergiu.iliev@company.org
Phone contact: +40 722 123 456 or Swiss +41 79 123 45 67
wifi_password = "SuperSecretPersonalHomePass!"
"""
        bad_file.write_text(bad_content, encoding="utf-8")

        auditor = DoubleAuditor(root_dir=Path(self.temp_dir), strict=True)
        report = auditor.run_audit()

        self.assertFalse(report.passed)
        matched_rules = {f.matched_pattern for f in report.findings}
        self.assertIn("personal_email_sj", matched_rules)
        self.assertIn("phone_romanian", matched_rules)
        self.assertIn("phone_swiss", matched_rules)
        self.assertIn("wifi_password", matched_rules)

    def test_detect_hardcoded_home_paths(self):
        """Verifies static detection of machine-specific user home paths."""
        bad_file = Path(self.temp_dir) / "paths.py"
        bad_content = """
DATA_DIR = "/Users/Sergiu/AI/Time_Capsule/data"
WIN_DIR = r"C:\\Users\\JohnDoe\\Documents\\data"
LINUX_DIR = "/home/developer/projects"
"""
        bad_file.write_text(bad_content, encoding="utf-8")

        auditor = DoubleAuditor(root_dir=Path(self.temp_dir), strict=True)
        report = auditor.run_audit()

        self.assertFalse(report.passed)
        matched_categories = {f.category for f in report.findings}
        self.assertIn("HOME_PATH", matched_categories)

    def test_inline_audit_suppression(self):
        """Verifies '# audit:ignore' suppresses false positives when explicitly annotated."""
        suppressed_file = Path(self.temp_dir) / "suppressed.py"
        content = """
KNOWN_SAMPLE = "sk-proj-1234567890abcdef1234567890abcdef"  # audit:ignore
"""
        suppressed_file.write_text(content, encoding="utf-8")

        auditor = DoubleAuditor(root_dir=Path(self.temp_dir), strict=True)
        report = auditor.run_audit()
        self.assertTrue(report.passed)
        self.assertEqual(len(report.findings), 0)

    def test_clean_files_pass_layer1_with_zero_errors(self):
        """Verifies clean files pass Layer 1 with 0 findings and 0 blocking errors."""
        clean_file = Path(self.temp_dir) / "clean_module.py"
        clean_content = """
import os
from pathlib import Path

def get_data_dir() -> Path:
    return Path(os.environ.get("TIMECAPSULE_DATA_DIR", "./data/capsule"))
"""
        clean_file.write_text(clean_content, encoding="utf-8")

        auditor = DoubleAuditor(root_dir=Path(self.temp_dir), strict=True)
        report = auditor.run_audit()

        self.assertTrue(report.passed)
        self.assertEqual(report.blocking_findings, 0)
        self.assertEqual(len(report.findings), 0)

    def test_production_codebase_zero_blocking_leaks(self):
        """Audits the actual codebase and verifies zero blocking leaks and zero secret findings."""
        auditor = DoubleAuditor(root_dir=PROJECT_ROOT, strict=False)
        report = auditor.run_audit()

        # Zero secrets, zero PII
        leaks = [f for f in report.findings if f.category in ("SECRET", "PII")]
        self.assertEqual(len(leaks), 0, f"Production codebase contains leaks: {leaks}")
        self.assertEqual(report.blocking_findings, 0)


class TestDoubleAuditLayer2Runtime(unittest.TestCase):
    """Verifies Layer 2 runtime sanitization, RAG auditing, and payload inspection."""

    def test_sanitize_text_secrets(self):
        """Verifies that secrets are redacted from text strings."""
        raw = "Using token sk-proj-1234567890abcdef1234567890abcdef and hf_1234567890abcdef1234567890abcdef"
        sanitized = sanitize_text(raw)
        self.assertNotIn("sk-proj-1234567890abcdef1234567890abcdef", sanitized)
        self.assertNotIn("hf_1234567890abcdef1234567890abcdef", sanitized)
        self.assertIn("[REDACTED_OPENAI_API_KEY]", sanitized)
        self.assertIn("[REDACTED_HUGGINGFACE_TOKEN]", sanitized)

    def test_sanitize_text_pii_and_paths(self):
        """Verifies that PII and user home paths are redacted from runtime text."""
        raw = "Email dev@sj-webdesign.com or call +40 722 123 456 from /Users/Sergiu/AI/Time_Capsule"
        sanitized = sanitize_text(raw)
        self.assertNotIn("dev@sj-webdesign.com", sanitized)
        self.assertNotIn("+40 722 123 456", sanitized)
        self.assertNotIn("/Users/Sergiu", sanitized)
        self.assertIn("[REDACTED_PERSONAL_EMAIL_SJ]", sanitized)
        self.assertIn("[PORTABLE_USER_PATH]", sanitized)

    def test_audit_rag_context_clean(self):
        """Verifies clean memory context passes audit with no violations."""
        context = "On October 2024, discussed Brâncuși minimalism and edge AI sovereignty in Constanța."
        is_clean, violations = audit_rag_context(context)
        self.assertTrue(is_clean)
        self.assertEqual(len(violations), 0)

    def test_audit_rag_context_catches_leak(self):
        """Verifies audit_rag_context catches accidental secrets or PII in retrieved memories."""
        dirty_context = "My internal email is test@sj-webdesign.com and private key is sk-proj-1234567890abcdef1234567890abcdef."
        is_clean, violations = audit_rag_context(dirty_context)
        self.assertFalse(is_clean)
        self.assertGreaterEqual(len(violations), 2)

    def test_audit_outgoing_payload_clean(self):
        """Verifies deep traversal of clean dictionary payload."""
        payload = {
            "status": "success",
            "version": "v0.1.0",
            "memories": [
                {"id": 1, "text": "Historical recollection from 2023.", "tags": ["philosophy"]},
                {"id": 2, "text": "Thoughts on local hardware sovereignty.", "tags": ["edge"]}
            ]
        }
        is_clean, violations = audit_outgoing_payload(payload)
        self.assertTrue(is_clean)
        self.assertEqual(len(violations), 0)

    def test_audit_outgoing_payload_nested_leak(self):
        """Verifies deep traversal catches leaks in nested dicts and lists."""
        leaky_payload = {
            "status": "ok",
            "session": {
                "metadata": {
                    "contact": "admin@sj-webdesign.com",
                    "backup_path": "/Users/Sergiu/AI/backup.db"
                }
            }
        }
        is_clean, violations = audit_outgoing_payload(leaky_payload)
        self.assertFalse(is_clean)
        self.assertGreaterEqual(len(violations), 2)
        viol_str = " ".join(violations)
        self.assertIn("root.session.metadata.contact", viol_str)
        self.assertIn("root.session.metadata.backup_path", viol_str)


class TestConfigAndSanitizationTemplates(unittest.TestCase):
    """Verifies dynamic config paths, versioning, and sanitization templates."""

    def test_dynamic_config_paths(self):
        """Verifies CapsuleConfig paths dynamically derive from PROJECT_ROOT."""
        cfg = CapsuleConfig()
        self.assertEqual(cfg.app_version, APP_VERSION)
        self.assertEqual(cfg.app_version, "v0.1.0")
        self.assertTrue(Path(cfg.storage.base_dir).is_absolute())
        self.assertIn("data/capsule", cfg.storage.data_dir.replace("\\", "/"))
        self.assertIn("archive.db", cfg.storage.sqlite_db_path)
        self.assertEqual(cfg.security.internet_rag_mode, "airgapped")
        self.assertEqual(cfg.retrieval.internet_rag_mode, "airgapped")

    def test_environment_variable_override(self):
        """Verifies environment variables override default storage paths."""
        custom_data = "/tmp/test_custom_tc_data"
        old_val = os.environ.get("TIMECAPSULE_DATA_DIR")
        try:
            os.environ["TIMECAPSULE_DATA_DIR"] = custom_data
            cfg = CapsuleConfig()
            self.assertEqual(cfg.storage.data_dir, custom_data)
            self.assertEqual(cfg.storage.sqlite_db_path, f"{custom_data}/archive.db")
        finally:
            if old_val:
                os.environ["TIMECAPSULE_DATA_DIR"] = old_val
            else:
                os.environ.pop("TIMECAPSULE_DATA_DIR", None)

    def test_sanitized_config_templates_exist(self):
        """Verifies sanitized config templates exist and contain generic placeholders."""
        example_cfg = PROJECT_ROOT / "config" / "timecapsule" / "config.example.json"
        self.assertTrue(example_cfg.exists(), "config.example.json must exist")

        with open(example_cfg, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data.get("app_version"), "v0.1.0")
        self.assertEqual(data.get("storage", {}).get("base_dir"), ".")
        self.assertEqual(data.get("storage", {}).get("data_dir"), "./data/capsule")
        self.assertEqual(data.get("security", {}).get("internet_rag_mode"), "airgapped")

    def test_env_example_exists(self):
        """Verifies .env.example exists with placeholder configuration."""
        env_example = PROJECT_ROOT / ".env.example"
        self.assertTrue(env_example.exists(), ".env.example must exist")
        content = env_example.read_text(encoding="utf-8")
        self.assertIn("TIMECAPSULE_BASE_DIR=.", content)
        self.assertIn("INTERNET_RAG_MODE=airgapped", content)

    def test_credential_placeholders(self):
        """Verifies credential files in config do not contain real keys."""
        cf_token = PROJECT_ROOT / "config" / "cloudflared" / "tunnel_token.txt"
        if cf_token.exists():
            content = cf_token.read_text(encoding="utf-8").strip()
            self.assertTrue("PLACEHOLDER" in content or "eyJhIjoi" not in content)

        ow_key = PROJECT_ROOT / "config" / "open-webui" / "secret_key.txt"
        if ow_key.exists():
            content = ow_key.read_text(encoding="utf-8").strip()
            self.assertTrue("PLACEHOLDER" in content or len(content) != 64)


if __name__ == "__main__":
    unittest.main()
