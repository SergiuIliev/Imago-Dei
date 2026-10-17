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
        """Verifies root endpoint serves HTML dashboard."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("Personal AI Time Capsule", resp.text)
        self.assertIn("Active Persona Snapshot", resp.text)

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


if __name__ == "__main__":
    unittest.main()
