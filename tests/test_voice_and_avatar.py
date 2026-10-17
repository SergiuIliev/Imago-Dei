"""
Unit and Integration Tests for Voice Synthesis and Video Avatar Subsystems.
Verifies expressive prosody control, WAV synthesis, neural fallback,
and avatar render job lifecycle.
"""

import os
import sys
import unittest
import tempfile
import shutil
import wave
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.models import ExpressiveDeliveryParams, EmotionCategory
from core.voice import ExpressiveDeliveryController, VoiceProfile
from providers.tts.macos_tts import MacOSNativeTTSProvider
from providers.tts.neural_tts import NeuralTTSProvider
from providers.avatar.liveportrait_provider import LivePortraitAvatarProvider
from providers.avatar.base import AvatarJobStatus


class TestVoiceAndAvatar(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.controller = ExpressiveDeliveryController()
        self.tts = MacOSNativeTTSProvider(default_voice="Daniel", delivery_controller=self.controller)
        self.avatar = LivePortraitAvatarProvider(output_dir=os.path.join(self.test_dir, "avatar"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_expressive_delivery_controller(self):
        """Verifies text preparation and prosody parameter calculation for different emotions."""
        # Test Reflective delivery (should have slower rate and longer pauses)
        params_reflective = ExpressiveDeliveryParams(
            emotion=EmotionCategory.REFLECTIVE,
            intensity=0.8,
            speech_rate=0.9
        )
        prep_ref = self.controller.prepare_text_and_timing("This is a thoughtful memory. We paused.", params_reflective)
        self.assertLess(prep_ref["effective_wpm"], self.controller.profile.average_speaking_rate_wpm)
        self.assertGreater(prep_ref["pause_multiplier"], 1.0)
        self.assertIn("...", prep_ref["processed_text"])

        # Test Lively delivery (should have higher rate)
        params_lively = ExpressiveDeliveryParams(
            emotion=EmotionCategory.LIVELY,
            intensity=0.7,
            speech_rate=1.1
        )
        prep_live = self.controller.prepare_text_and_timing("Exciting breakthrough in the lab!", params_lively)
        self.assertGreater(prep_live["effective_wpm"], self.controller.profile.average_speaking_rate_wpm)

    def test_macos_tts_wav_synthesis(self):
        """Verifies actual synthesis of a 24kHz 16-bit WAV file on the host Mac."""
        out_wav = os.path.join(self.test_dir, "test_speech.wav")
        params = ExpressiveDeliveryParams(emotion=EmotionCategory.NEUTRAL, speech_rate=1.0)

        result_path = self.tts.synthesize(
            text="Testing local voice synthesis pipeline.",
            delivery=params,
            output_file_path=out_wav
        )
        self.assertTrue(os.path.exists(result_path))

        # Check audio properties using standard wave module
        with wave.open(result_path, "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getsampwidth(), 2) # 16-bit
            self.assertEqual(wf.getframerate(), 24000)
            self.assertGreater(wf.getnframes(), 0)

    def test_neural_tts_fallback(self):
        """Verifies neural TTS provider falls back to macOS native speech when weights are uninitialized."""
        neural_tts = NeuralTTSProvider(
            model_name="kokoro-82m",
            models_dir=os.path.join(self.test_dir, "nonexistent_models"),
            fallback_provider=self.tts
        )
        self.assertTrue(neural_tts.is_healthy())

        out_wav = os.path.join(self.test_dir, "fallback_test.wav")
        params = ExpressiveDeliveryParams(emotion=EmotionCategory.SERIOUS)
        res = neural_tts.synthesize("Fallback synthesis test.", params, out_wav)
        self.assertTrue(os.path.exists(res))

    def test_avatar_job_lifecycle(self):
        """Verifies avatar render job creation, progression, and video output creation."""
        # Synthesize source audio
        audio_file = os.path.join(self.test_dir, "avatar_audio.wav")
        self.tts.synthesize("Avatar driving audio.", ExpressiveDeliveryParams(), audio_file)

        # Enqueue avatar render job
        job = self.avatar.create_render_job(audio_path=audio_file)
        self.assertIsNotNone(job.job_id)

        # Wait for background rendering worker
        max_wait = 10
        start = time.time()
        completed = False
        while time.time() - start < max_wait:
            status = self.avatar.get_job_status(job.job_id)
            if status and status.status == AvatarJobStatus.COMPLETED:
                completed = True
                break
            time.sleep(0.1)

        self.assertTrue(completed, "Avatar render job should complete within timeout")
        self.assertIsNotNone(job.output_video_path)
        self.assertTrue(os.path.exists(job.output_video_path))
        self.assertGreater(os.path.getsize(job.output_video_path), 0)


if __name__ == "__main__":
    unittest.main()
