"""
Integration Tests for FastAPI REST Endpoints and UI Serving.
Verifies all routes: health, chat, memories, beliefs, TTS, avatar, and backups.
"""

import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from app.main import app


class TestAPIEndpoints(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_serve_index_html(self):
        """Verifies root endpoint serves HTML dashboard with v0.1.0, theme toggle, and RAG mode."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Personal AI Time Capsule", resp.text)
        self.assertIn("Active Persona Snapshot", resp.text)
        self.assertIn("v0.1.0", resp.text)
        self.assertIn("themeToggleBtn", resp.text)
        self.assertIn("ragModeToggleBtn", resp.text)
        self.assertIn("ragModeModal", resp.text)
        self.assertIn("Homo Deus Imago Dei", resp.text)
        self.assertIn("data-theme", resp.text)

    def test_api_health(self):
        """Verifies health check returns subsystem statuses."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database"])
        self.assertTrue(data["security"]["local_only"])

    def test_api_eras(self):
        """Verifies listing chronological persona snapshots."""
        resp = self.client.get("/api/eras")
        self.assertEqual(resp.status_code, 200)
        eras = resp.json()
        self.assertGreaterEqual(len(eras), 3)
        era_ids = [e["era_id"] for e in eras]
        self.assertIn("early_20s", era_ids)
        self.assertIn("present", era_ids)

    def test_api_memories_list(self):
        """Verifies querying memory items."""
        resp = self.client.get("/api/memories")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

    def test_api_memory_verify_endpoint(self):
        """Verifies direct one-click verification of an unverified memory."""
        # Find or create a memory
        from core.models import MemoryItem, VerificationStatus
        from app.main import db
        mem_id = f"test-verify-{sys.platform}"
        test_mem = MemoryItem(
            id=mem_id,
            content="Testing direct verification endpoint.",
            summary="Direct verification test",
            verification_status=VerificationStatus.EXTRACTED_UNVERIFIED
        )
        db.add_memory(test_mem)

        # Call verify endpoint
        try:
            resp = self.client.post(f"/api/memories/{mem_id}/verify")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "success")
            self.assertEqual(data["memory"]["verification_status"], "personal_verified")

            # Nonexistent memory should 404
            resp_404 = self.client.post("/api/memories/nonexistent-id-xyz/verify")
            self.assertEqual(resp_404.status_code, 404)
        finally:
            with db.get_connection() as conn:
                conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
                conn.execute("DELETE FROM memories_fts WHERE id = ?", (mem_id,))
                conn.commit()

    def test_api_beliefs_list(self):
        """Verifies querying temporal beliefs."""
        resp = self.client.get("/api/beliefs")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)

    def test_api_chat_turn(self):
        """Verifies chat dialogue processing endpoint."""
        resp = self.client.post("/api/chat", json={
            "query": "What did you conclude about async event loops in 2016?",
            "era_id": "early_20s",
            "synthesize_audio": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("text", data)
        self.assertIn("evidence_used", data)
        self.assertIn("recommended_delivery", data)

    def test_api_tts_synthesize(self):
        """Verifies TTS synthesis endpoint produces audio URL."""
        resp = self.client.post("/api/tts/synthesize", json={
            "text": "Testing API speech synthesis.",
            "emotion": "neutral",
            "speech_rate": 1.0
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "success")
        self.assertIn("audio_url", data)

        # Test audio file retrieval
        audio_resp = self.client.get(data["audio_url"])
        self.assertEqual(audio_resp.status_code, 200)
        self.assertEqual(audio_resp.headers["content-type"], "audio/wav")

    def test_api_archive_stats(self):
        """Verifies archive statistics endpoint."""
        resp = self.client.get("/api/archive/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_memories", data)
        self.assertIn("verified_memories", data)

    def test_media_clip_streaming(self):
        """Verifies on-the-fly vlog audio slice streaming and bounds validation."""
        source_id = "3791e8c7-4c75-446a-82b3-641eebc98d3f"
        # 1. Valid range stream
        resp = self.client.get(f"/api/media/{source_id}/clip?start=244.9&end=282.9")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")
        self.assertGreater(len(resp.content), 1000)

        # 2. Negative start
        resp_neg = self.client.get(f"/api/media/{source_id}/clip?start=-5&end=10")
        self.assertEqual(resp_neg.status_code, 400)

        # 3. Start >= End
        resp_inv = self.client.get(f"/api/media/{source_id}/clip?start=50&end=10")
        self.assertEqual(resp_inv.status_code, 400)

        # 4. Start exceeds duration
        resp_ovr = self.client.get(f"/api/media/{source_id}/clip?start=9999&end=10000")
        self.assertEqual(resp_ovr.status_code, 400)

        # 5. Nonexistent source id
        resp_nf = self.client.get("/api/media/nonexistent-id/clip?start=0&end=5")
        self.assertEqual(resp_nf.status_code, 404)

    def test_api_asr_transcribe(self):
        """Verifies ASR transcription endpoint processes audio files."""
        ref_wav = PROJECT_ROOT / "data" / "capsule" / "voice_references" / "sergiu_ref_10.wav"
        if ref_wav.exists():
            with open(ref_wav, "rb") as f:
                resp = self.client.post(
                    "/api/asr/transcribe",
                    files={"audio": ("sergiu_ref_10.wav", f, "audio/wav")}
                )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "ok")
            self.assertIn("transcription", data)
            self.assertGreater(len(data["transcription"]), 10)

    def test_api_rag_mode(self):
        """Verifies GET and POST /api/settings/rag_mode."""
        # 1. Get current mode
        resp = self.client.get("/api/settings/rag_mode")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("mode", data)
        self.assertIn("allowed_modes", data)
        self.assertIn(data["mode"], ["airgapped", "jev_controlled", "open_internet"])

        # 2. Update to jev_controlled
        post_resp = self.client.post("/api/settings/rag_mode", json={"mode": "jev_controlled"})
        self.assertEqual(post_resp.status_code, 200)
        self.assertEqual(post_resp.json()["mode"], "jev_controlled")

        # 3. Verify it persisted
        get_resp = self.client.get("/api/settings/rag_mode")
        self.assertEqual(get_resp.json()["mode"], "jev_controlled")

        # 4. Reject invalid mode
        invalid_resp = self.client.post("/api/settings/rag_mode", json={"mode": "invalid_mode_xyz"})
        self.assertEqual(invalid_resp.status_code, 400)

        # 5. Restore to airgapped
        restore_resp = self.client.post("/api/settings/rag_mode", json={"mode": "airgapped"})
        self.assertEqual(restore_resp.status_code, 200)
        self.assertEqual(restore_resp.json()["mode"], "airgapped")



if __name__ == "__main__":
    unittest.main()
