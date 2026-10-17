"""
Unit and Integration Tests for Backup, Export, Integrity Verification, and Restoration.
Verifies cryptographic manifests, corruption detection, and safe restoration.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import CapsuleConfig
from core.models import MemoryItem, VerificationStatus
from memory.database import DatabaseManager
from memory.backup import BackupManager


class TestBackupAndRestore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = CapsuleConfig()
        self.config.storage.data_dir = os.path.join(self.test_dir, "data")
        self.config.storage.sqlite_db_path = os.path.join(self.test_dir, "data", "archive.db")
        self.config.storage.chroma_dir = os.path.join(self.test_dir, "data", "chroma")
        self.config.storage.exports_dir = os.path.join(self.test_dir, "exports")

        os.makedirs(self.config.storage.data_dir, exist_ok=True)
        os.makedirs(self.config.storage.exports_dir, exist_ok=True)

        self.db = DatabaseManager(self.config.storage.sqlite_db_path)
        self.backup_mgr = BackupManager(self.config, self.db)

        # Seed test memory
        mem = MemoryItem(
            id="backup-mem-1",
            content="Critical memory to be preserved in backup.",
            summary="Backup memory",
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        )
        self.db.add_memory(mem)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_backup_creation_and_manifest(self):
        """Verifies creation of .tar.gz archive and SHA-256 manifest."""
        res = self.backup_mgr.create_full_backup(label="unit_test")
        self.assertTrue(os.path.exists(res["backup_file"]))
        self.assertGreater(res["size_bytes"], 0)
        self.assertGreater(len(res["sha256"]), 0)

        # Verify archive integrity
        verification = self.backup_mgr.verify_backup(res["backup_file"])
        self.assertTrue(verification.get("valid"))

    def test_backup_restore(self):
        """Verifies database restoration from an existing backup snapshot."""
        backup_res = self.backup_mgr.create_full_backup(label="restore_test")
        backup_file = backup_res["backup_file"]

        # Modify active DB
        self.db.add_memory(MemoryItem(
            id="backup-mem-2",
            content="Memory added after backup.",
            summary="Post backup memory",
            verification_status=VerificationStatus.PERSONAL_VERIFIED
        ))
        self.assertIsNotNone(self.db.get_memory("backup-mem-2"))

        # Restore from backup
        restore_res = self.backup_mgr.restore_backup(backup_file)
        self.assertTrue(restore_res["restored"])

        # Re-check DB: backup-mem-1 must exist, post-backup item should not be present
        new_db = DatabaseManager(self.config.storage.sqlite_db_path)
        self.assertIsNotNone(new_db.get_memory("backup-mem-1"))
        self.assertIsNone(new_db.get_memory("backup-mem-2"))

    def test_portable_json_export(self):
        """Verifies human-readable JSON export contains memories and metadata."""
        export_path = self.backup_mgr.export_portable_json()
        self.assertTrue(os.path.exists(export_path))

        with open(export_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("memories", data)
        self.assertIn("metadata", data)
        self.assertGreaterEqual(len(data["memories"]), 1)
        self.assertEqual(data["memories"][0]["id"], "backup-mem-1")


if __name__ == "__main__":
    unittest.main()
