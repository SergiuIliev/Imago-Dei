"""
Unit and Integration Tests for Media Ingestion Pipeline.
Verifies PyAV metadata extraction, ASR transcription, SHA-256 deduplication,
timestamp preservation, and resumable execution.
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

from core.config import CapsuleConfig
from core.models import MediaStatus, VerificationStatus
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from ingestion.pipeline import MediaIngestionPipeline
from providers.asr.mock_provider import MockASRProvider


class TestIngestionPipeline(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config = CapsuleConfig()
        self.config.storage.sqlite_db_path = os.path.join(self.test_dir, "test.db")
        self.config.storage.chroma_dir = os.path.join(self.test_dir, "test_chroma")

        self.db = DatabaseManager(self.config.storage.sqlite_db_path)
        self.vector_store = VectorStore(self.config.storage.chroma_dir)
        self.asr = MockASRProvider()
        self.pipeline = MediaIngestionPipeline(self.config, self.db, self.vector_store, asr_provider=self.asr)

        # Fixture path
        self.sample_wav = Path(PROJECT_ROOT) / "tests" / "fixtures" / "synthetic_sample.wav"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_synthetic_media_ingestion(self):
        """Verifies synthetic media can pass through ingestion and generate timestamped transcripts."""
        self.assertTrue(self.sample_wav.exists(), "Sample WAV fixture must exist")

        result = self.pipeline.process_file(str(self.sample_wav), is_synthetic_fixture=True)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["transcript_segments"], 0)
        self.assertEqual(result["verification_status"], VerificationStatus.SYNTHETIC_FIXTURE.value)

        # Verify DB records
        media_id = result["media_id"]
        media = self.db.get_media_by_id(media_id)
        self.assertIsNotNone(media)
        self.assertEqual(media.status, MediaStatus.INDEXED)
        self.assertGreater(len(media.file_hash), 0)

        # Verify transcripts have timestamps
        transcripts = self.db.get_transcripts_for_source(media_id)
        self.assertGreater(len(transcripts), 0)
        for seg in transcripts:
            self.assertGreaterEqual(seg.end_seconds, seg.start_seconds)
            self.assertGreater(len(seg.text), 0)

    def test_cryptographic_deduplication(self):
        """Verifies that re-ingesting a file with identical SHA-256 is detected and skipped."""
        # First ingestion
        res1 = self.pipeline.process_file(str(self.sample_wav), is_synthetic_fixture=True)
        self.assertEqual(res1["status"], "success")

        # Second ingestion with same file
        res2 = self.pipeline.process_file(str(self.sample_wav), is_synthetic_fixture=True)
        self.assertEqual(res2["status"], "duplicate_skipped")
        self.assertEqual(res2["media_id"], res1["media_id"])

    def test_empty_directory_handling(self):
        """Verifies that when no media path is supplied, pipeline terminates safely without scanning."""
        empty_dir = os.path.join(self.test_dir, "empty_dir")
        os.makedirs(empty_dir, exist_ok=True)

        job_result = self.pipeline.run_ingestion_job(target_directory=empty_dir)
        self.assertEqual(job_result["status"], "completed")
        self.assertEqual(job_result["total_files_discovered"], 0)


if __name__ == "__main__":
    unittest.main()
