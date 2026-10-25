"""
Unit and Integration Tests for Memory Storage, Database, and FTS5 Indexing.
Verifies data persistence across restarts, schema migrations, and user corrections.
"""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import MemoryItem, TemporalBelief, MemoryCorrection, VerificationStatus
from memory.database import DatabaseManager


class TestMemoryPersistence(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_archive.db")
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_database_initialization_and_migration(self):
        """Verifies schema tables, migrations, and FTS5 virtual table are created."""
        with self.db.get_connection() as conn:
            cur = conn.execute("SELECT version FROM schema_migrations")
            self.assertEqual(cur.fetchone()[0], 1)

            # Check FTS5 virtual table existence
            cur_fts = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='memories_fts'")
            self.assertIsNotNone(cur_fts.fetchone())

    def test_memory_crud_and_persistence_across_restart(self):
        """Verifies memories persist in SQLite after closing connection and re-instantiating."""
        mem = MemoryItem(
            id="test-mem-1",
            content="Testing persistence across process restarts.",
            summary="Test persistence memory",
            category="event",
            date_occurred="2024-01-01",
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        )
        self.db.add_memory(mem)

        # Simulate application restart with new DatabaseManager instance
        new_db = DatabaseManager(self.db_path)
        retrieved = new_db.get_memory("test-mem-1")

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "test-mem-1")
        self.assertEqual(retrieved.content, "Testing persistence across process restarts.")
        self.assertEqual(retrieved.verification_status, VerificationStatus.PERSONAL_VERIFIED)

    def test_fts5_keyword_search(self):
        """Verifies BM25 full-text indexing and retrieval."""
        mem1 = MemoryItem(
            id="fts-1",
            content="Autonomous orchestration using specialized agents.",
            summary="Autonomous orchestration",
            category="reflection",
            verification_status=VerificationStatus.SYNTHETIC_FIXTURE
        )
        mem2 = MemoryItem(
            id="fts-2",
            content="Cooking pasta with garlic and olive oil.",
            summary="Cooking recipe",
            category="lifestyle",
            verification_status=VerificationStatus.SYNTHETIC_FIXTURE
        )
        self.db.add_memory(mem1)
        self.db.add_memory(mem2)

        results = self.db.search_keyword("orchestration")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].id, "fts-1")

    def test_memory_correction_workflow(self):
        """Verifies that editing a memory preserves an immutable audit log and updates content."""
        mem = MemoryItem(
            id="mem-to-correct",
            content="Original unverified observation.",
            summary="Original summary",
            verification_status=VerificationStatus.EXTRACTED_UNVERIFIED
        )
        self.db.add_memory(mem)

        correction = MemoryCorrection(
            id="corr-1",
            memory_id="mem-to-correct",
            original_content="Original unverified observation.",
            corrected_content="Corrected and verified statement directly from personal notes.",
            correction_reason="Clarifying exact phrasing",
            corrected_by="user",
            created_at="2026-10-13T12:00:00"
        )
        success = self.db.correct_memory(correction)
        self.assertTrue(success)

        updated_mem = self.db.get_memory("mem-to-correct")
        self.assertEqual(updated_mem.content, "Corrected and verified statement directly from personal notes.")
        self.assertEqual(updated_mem.verification_status, VerificationStatus.PERSONAL_VERIFIED)

        # Audit history check
        corrections = self.db.get_corrections("mem-to-correct")
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0].correction_reason, "Clarifying exact phrasing")

    def test_direct_verify_memory(self):
        """Verifies directly promoting an unverified memory to personal_verified without modifying content."""
        mem = MemoryItem(
            id="mem-to-verify-direct",
            content="Direct verification test content.",
            summary="Direct verification summary",
            verification_status=VerificationStatus.EXTRACTED_UNVERIFIED
        )
        self.db.add_memory(mem)

        # Direct verify
        success = self.db.verify_memory("mem-to-verify-direct")
        self.assertTrue(success)

        updated = self.db.get_memory("mem-to-verify-direct")
        self.assertEqual(updated.verification_status, VerificationStatus.PERSONAL_VERIFIED)
        self.assertEqual(updated.content, "Direct verification test content.")

        # Nonexistent returns False
        self.assertFalse(self.db.verify_memory("nonexistent-id"))


if __name__ == "__main__":
    unittest.main()
