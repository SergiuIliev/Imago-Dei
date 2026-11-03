"""
Neural Voice Synthesis and Reference-Conditioned Voice Cloning Provider.
Provides high-fidelity flow matching TTS conditioned on Sergiu's authentic recordings (F5-TTS),
with fast-path routing to a warm Metal MPS daemon on 127.0.0.1:7861,
clean stage-direction sanitization, sentence-level synthesis with natural pause padding,
and automatic fallback to Kokoro-82M neural TTS or local macOS native speech.
"""

from __future__ import annotations
import os
import sys
import time
import uuid
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Dict, Any, Optional
import soundfile as sf
import numpy as np

from core.models import ExpressiveDeliveryParams, EmotionCategory
from core.voice import VoiceProfile, ExpressiveDeliveryController, get_reference_clip_for_emotion
from core.audio_dsp import clean_audio_signal, clean_wav_file
from providers.tts.base import TTSProvider
from providers.tts.macos_tts import MacOSNativeTTSProvider


class NeuralTTSProvider(TTSProvider):
    """
    Neural speech synthesizer with zero-shot reference audio conditioning and
    empirical speaker tensors. Designed for local execution on Apple Silicon M4.
    """

    def __init__(
        self,
        model_name: str = "sergiu_authentic_f5",
        models_dir: Optional[str] = None,
        voice_profile: Optional[VoiceProfile] = None,
        fallback_provider: Optional[TTSProvider] = None,
        delivery_controller: Optional[ExpressiveDeliveryController] = None
    ):
        project_root = Path(__file__).resolve().parent.parent.parent
        self.model_name = model_name
        self.models_dir = Path(models_dir or os.environ.get("TIMECAPSULE_TTS_MODELS_DIR", str(project_root / "models" / "tts")))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.profile = voice_profile or VoiceProfile()
        self.fallback = fallback_provider or MacOSNativeTTSProvider()
        self.delivery_controller = delivery_controller or ExpressiveDeliveryController(self.profile)

        # Pipeline instance cache
        self._pipeline = None
        self._custom_voice_tensor = None
        self.neural_engine_ready = False

    def _ensure_engine_ready(self) -> bool:
        """Lazily initializes Kokoro neural TTS only when needed as fallback."""
        if self._pipeline is not None:
            return True
        self.neural_engine_ready = self._initialize_engine()
        return self.neural_engine_ready

    def _initialize_engine(self) -> bool:
        """Initializes Kokoro neural TTS pipeline and loads custom voice tensor if available."""
        project_root = Path(__file__).resolve().parent.parent.parent
        if "VIRTUAL_ENV" not in os.environ:
            os.environ["VIRTUAL_ENV"] = str(project_root / "venv")

        try:
            from kokoro import KPipeline
            import torch

            # Lazy initialize pipeline
            self._pipeline = KPipeline(lang_code="a")

            # Check for custom calibrated voice tensor
            custom_voice_path = project_root / "data" / "capsule" / "voice_references" / "sergiu_voice.pt"
            if custom_voice_path.exists():
                self._custom_voice_tensor = torch.load(custom_voice_path, map_location="cpu", weights_only=False)
                self.profile.is_cloned_from_personal_media = True

            return True
        except Exception as e:
            print(f"[NeuralTTSProvider] Notice: Kokoro neural fallback initialization deferred: {e}")
            return False

    def is_daemon_responsive(self) -> bool:
        """Checks if the persistent F5-TTS Metal MPS daemon is ready on 127.0.0.1:7861."""
        try:
            req = urllib.request.Request(
                "http://127.0.0.1:7861/health",
                headers={"User-Agent": "TimeCapsule-Voice/1.0"}
            )
            with urllib.request.urlopen(req, timeout=0.6) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("status") == "ready"
        except Exception:
            pass
        return False

    def is_healthy(self) -> bool:
        """Provider is considered healthy if daemon is ready, Kokoro is ready, or fallback is ready."""
        if self.is_daemon_responsive():
            return True
        if self.neural_engine_ready and self._pipeline is not None:
            return True
        return self.fallback.is_healthy()

    def list_available_voices(self) -> List[str]:
        voices = ["sergiu_authentic_f5", "sergiu_voice_cloned"]
        voices.extend(["kokoro_am_adam", "kokoro_bm_george", "kokoro_bm_lewis"])
        voices.extend(self.fallback.list_available_voices())
        return voices

    def prepare_reference_conditioning(self, audio_paths: List[str]) -> Dict[str, Any]:
        """Validates and registers personal reference audio files for voice conditioning."""
        valid_clips = []
        for p_str in audio_paths:
            p = Path(p_str)
            if p.exists() and p.is_file():
                valid_clips.append(str(p))

        self.profile.reference_audio_clips = valid_clips
        self.profile.is_cloned_from_personal_media = len(valid_clips) > 0

        return {
            "registered_clips": len(valid_clips),
            "profile_active": self.profile.is_cloned_from_personal_media,
            "status": "ready_for_cloning" if valid_clips else "awaiting_personal_media"
        }

    def _call_daemon(
        self,
        text: str,
        output_path: str,
        speed: float,
        ref_id: str,
        steps: int = 8
    ) -> bool:
        """Sends synthesis request to warm F5-TTS daemon."""
        try:
            payload = json.dumps({
                "text": text,
                "output_path": output_path,
                "speed": speed,
                "ref_id": ref_id,
                "steps": steps
            }).encode("utf-8")
            req = urllib.request.Request(
                "http://127.0.0.1:7861/synthesize",
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "TimeCapsule-Voice/1.0"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                if resp.status == 200:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data.get("status") == "ok" and os.path.exists(output_path) and os.path.getsize(output_path) > 1000
        except Exception as e:
            print(f"[NeuralTTSProvider] Notice: Daemon synthesis call failed: {e}")
        return False

    def synthesize(
        self,
        text: str,
        delivery: ExpressiveDeliveryParams,
        output_file_path: str,
        voice_override: Optional[str] = None
    ) -> str:
        """
        Synthesizes speech using authentic neural conditioning from Sergiu's recordings (F5-TTS),
        with fast-path routing to warm daemon on 127.0.0.1:7861, stage-direction sanitization,
        sentence-level synthesis, and graceful fallback to Kokoro neural TTS or macOS native speech.
        """
        # 1. Prosodic Preparation & Sanitization through ExpressiveDeliveryController
        timing_info = self.delivery_controller.prepare_text_and_timing(text, delivery)
        clean_text = timing_info["clean_text"]
        segments = timing_info.get("segments", [])
        has_explicit_pauses = timing_info.get("has_explicit_pauses", False)
        ref_id = timing_info.get("reference_clip_id", "ref_02")
        effective_speed = max(0.75, min(1.3, 1.0 * delivery.speech_rate))

        # 16 flow-matching steps guarantees smooth, natural vocoder convergence without robotic artifacts; 24 steps for strong emphasis
        steps = 24 if delivery.emphasis_level == "strong" else 16

        if not clean_text:
            clean_text = timing_info.get("processed_text", text)

        out_path = Path(output_file_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        use_f5 = voice_override in [None, "sergiu_authentic_f5", "sergiu_voice_cloned"]

        # 2. Fast-Path: Persistent Warm Metal MPS Daemon on Port 7861
        if use_f5 and self.is_daemon_responsive():
            try:
                # If text contains explicit pause hints, do segment-level stitching (max 3 segments)
                if has_explicit_pauses and 1 < len(segments) <= 4:
                    audio_blocks = []
                    sr = 24000
                    temp_files = []
                    success = True

                    for i, seg in enumerate(segments[:4]):
                        seg_text = seg["text"].strip()
                        if not seg_text:
                            continue
                        seg_pause = seg.get("pause_after_seconds", 0.25)
                        temp_seg_file = f"/tmp/tc_seg_{uuid.uuid4().hex[:8]}_{i}.wav"
                        temp_files.append(temp_seg_file)

                        seg_ok = self._call_daemon(
                            text=seg_text,
                            output_path=temp_seg_file,
                            speed=effective_speed,
                            ref_id=ref_id,
                            steps=steps
                        )
                        if not seg_ok:
                            success = False
                            break

                        seg_audio, file_sr = sf.read(temp_seg_file)
                        sr = file_sr
                        audio_blocks.append(seg_audio)

                        # Clean pause padding between sentences
                        if i < len(segments[:4]) - 1:
                            silence_frames = int(sr * seg_pause)
                            audio_blocks.append(np.zeros(silence_frames, dtype=np.float32))

                    for tf in temp_files:
                        try:
                            if os.path.exists(tf):
                                os.unlink(tf)
                        except Exception:
                            pass

                    if success and audio_blocks:
                        combined_audio = np.concatenate(audio_blocks)
                        cleaned_audio = clean_audio_signal(combined_audio, sr)
                        sf.write(str(out_path), cleaned_audio, sr)
                        return str(out_path)

                # Direct single-pass synthesis: F5-TTS synthesizes coherent paragraphs in < 5s
                # For long text responses, limit spoken voice to first 2-3 sentences (~350 chars) for snappy conversational UX
                synthesis_text = clean_text
                if len(synthesis_text) > 400:
                    sents = timing_info.get("sentences", [])
                    concise_sents = []
                    curr_len = 0
                    for s in sents:
                        concise_sents.append(s)
                        curr_len += len(s)
                        if curr_len >= 250:
                            break
                    synthesis_text = " ".join(concise_sents) if concise_sents else clean_text[:350]

                direct_ok = self._call_daemon(
                    text=synthesis_text,
                    output_path=str(out_path),
                    speed=effective_speed,
                    ref_id=ref_id,
                    steps=steps
                )
                if direct_ok:
                    clean_wav_file(str(out_path))
                    return str(out_path)

            except Exception as e:
                print(f"[NeuralTTSProvider] Notice: Daemon synthesis failed ({e}), falling back to subprocess.")

        # 3. Subprocess Fallback with f5_synthesize.py
        project_root = Path(__file__).resolve().parent.parent.parent
        f5_script = project_root / "scripts" / "f5_synthesize.py"
        f5_python = Path(os.environ.get("VENV_VOICE_PYTHON", sys.executable))

        if f5_script.exists() and f5_python.exists() and use_f5:
            try:
                import subprocess
                cmd = [
                    str(f5_python),
                    str(f5_script),
                    "--text", clean_text,
                    "--output", output_file_path,
                    "--speed", str(round(effective_speed, 2)),
                    "--ref_id", ref_id,
                    "--steps", str(steps)
                ]
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = "0"
                res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
                if res.returncode == 0 and os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 1000:
                    clean_wav_file(output_file_path)
                    return output_file_path
                else:
                    print(f"[NeuralTTSProvider] Notice: F5-TTS subprocess returned code {res.returncode}, falling back to Kokoro: {res.stderr[:200]}")
            except Exception as e:
                print(f"[NeuralTTSProvider] F5-TTS invocation error: {e}, falling back to Kokoro.")

        # 4. Kokoro Neural Engine Fallback
        if self.neural_engine_ready and self._pipeline is not None:
            try:
                voice = self._custom_voice_tensor
                if voice_override:
                    if voice_override.startswith("kokoro_"):
                        voice = voice_override.replace("kokoro_", "")
                    elif voice_override == "sergiu_voice_cloned" and self._custom_voice_tensor is not None:
                        voice = self._custom_voice_tensor

                if voice is None:
                    voice = "bm_george"

                preset = self.profile.emotion_presets.get(delivery.emotion.value, {})
                rate_mult = preset.get("rate_multiplier", 1.0)
                base_speed = (self.profile.average_speaking_rate_wpm / 145.0)
                eff_speed = base_speed * delivery.speech_rate * rate_mult
                eff_speed = max(0.65, min(1.35, eff_speed))

                generator = self._pipeline(clean_text, voice=voice, speed=eff_speed)
                all_audio = []
                for _, _, audio in generator:
                    all_audio.append(audio)

                if all_audio:
                    combined_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                    cleaned_audio = clean_audio_signal(combined_audio, 24000)
                    sf.write(str(out_path), cleaned_audio, 24000)
                    return str(out_path)

            except Exception as e:
                print(f"[NeuralTTSProvider] Neural Kokoro fallback failed ({e}), falling back to MacOS native speech.")

        # 5. macOS Native Linear PCM WAV Fallback
        return self.fallback.synthesize(clean_text, delivery, output_file_path, voice_override)
