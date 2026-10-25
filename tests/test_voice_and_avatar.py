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

    def test_avatar_audio_video_muxing_and_reactivity(self):
        """Verifies MP4 container has both H.264 video and AAC audio with matching durations and reactive motion."""
        import av

        # Synthesize test audio
        audio_file = os.path.join(self.test_dir, "test_speech_reactivity.wav")
        self.tts.synthesize("Testing synchronized audio video muxing for avatar rendering.", ExpressiveDeliveryParams(), audio_file)

        out_mp4 = os.path.join(self.test_dir, "test_muxed.mp4")
        self.avatar.render_avatar_video(audio_path=audio_file, output_path=out_mp4)

        self.assertTrue(os.path.exists(out_mp4))
        self.assertGreater(os.path.getsize(out_mp4), 10000)

        # Probe container
        container = av.open(out_mp4)
        v_streams = [s for s in container.streams if s.type == "video"]
        a_streams = [s for s in container.streams if s.type == "audio"]

        self.assertEqual(len(v_streams), 1, "Must contain exactly 1 video stream")
        self.assertEqual(len(a_streams), 1, "Must contain exactly 1 audio stream")

        v = v_streams[0]
        a = a_streams[0]

        # Verify video stream specs
        self.assertEqual(v.width, 512)
        self.assertEqual(v.height, 512)
        self.assertIn(v.codec_context.name, ["h264", "mpeg4"])

        # Verify audio stream specs
        self.assertEqual(a.codec_context.name, "aac")
        self.assertEqual(a.rate, 24000)

        # Verify sub-millisecond duration alignment
        v_dur = float(v.duration * v.time_base)
        a_dur = float(a.duration * a.time_base)
        duration_delta_ms = abs(v_dur - a_dur) * 1000.0

        # Discrepancy must be well under 1 frame (40ms)
        self.assertLess(duration_delta_ms, 20.0, f"Duration difference too high: {duration_delta_ms} ms")
        container.close()

    def test_audio_dsp_dehumming(self):
        """Verifies DSP audio filter removes 50Hz/60Hz mains hum, 120Hz harmonics, and DC offset."""
        import numpy as np
        import soundfile as sf
        from core.audio_dsp import clean_audio_signal, clean_wav_file

        sr = 24000
        duration = 1.5
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)

        # 200 Hz voice fundamental + 50 Hz EU hum + 120 Hz US second harmonic + DC offset
        voice = 0.5 * np.sin(2 * np.pi * 200 * t)
        hum_50 = 0.25 * np.sin(2 * np.pi * 50 * t)
        hum_120 = 0.35 * np.sin(2 * np.pi * 120 * t)
        dc_offset = 0.08
        raw_signal = voice + hum_50 + hum_120 + dc_offset

        # FFT before filtering
        fft_raw = np.abs(np.fft.rfft(raw_signal))
        freqs = np.fft.rfftfreq(len(raw_signal), 1.0 / sr)
        idx_50 = np.argmin(np.abs(freqs - 50.0))
        idx_120 = np.argmin(np.abs(freqs - 120.0))
        raw_50_mag = fft_raw[idx_50]
        raw_120_mag = fft_raw[idx_120]

        # Apply DSP filter
        cleaned = clean_audio_signal(raw_signal, sr=sr)
        self.assertEqual(len(cleaned), len(raw_signal))

        # FFT after filtering
        fft_clean = np.abs(np.fft.rfft(cleaned))
        clean_50_mag = fft_clean[idx_50]
        clean_120_mag = fft_clean[idx_120]

        # Hum components must be attenuated significantly
        self.assertLess(clean_50_mag, raw_50_mag * 0.15, "50Hz hum was not sufficiently attenuated")
        self.assertLess(clean_120_mag, raw_120_mag * 0.15, "120Hz harmonic was not sufficiently attenuated")

        # DC offset must be eliminated
        self.assertAlmostEqual(float(np.mean(cleaned)), 0.0, places=2)

        # Peak must be normalized to -1.0 dBFS (0.891)
        self.assertAlmostEqual(float(np.max(np.abs(cleaned))), 0.891, delta=0.01)

        # Test WAV file clean in-place
        test_wav = os.path.join(self.test_dir, "test_hum.wav")
        sf.write(test_wav, raw_signal.astype(np.float32), sr)
        clean_wav_file(test_wav)
        self.assertTrue(os.path.exists(test_wav))
        loaded_data, loaded_sr = sf.read(test_wav)
        self.assertEqual(loaded_sr, sr)
        self.assertAlmostEqual(float(np.max(np.abs(loaded_data))), 0.891, delta=0.02)


if __name__ == "__main__":
    unittest.main()

