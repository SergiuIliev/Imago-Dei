"""
Neural Voice Synthesis and Reference-Conditioned Voice Cloning Provider.
Provides an extensible architecture for open-weight neural TTS (Kokoro-82M).
Supports reference audio conditioning and custom speaker tensors (sergiu_voice.pt)
empirically calibrated from personal vlogs, with automatic fallback to local macOS native speech.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import soundfile as sf

from core.models import ExpressiveDeliveryParams
from core.voice import VoiceProfile, ExpressiveDeliveryController
from providers.tts.base import TTSProvider
from providers.tts.macos_tts import MacOSNativeTTSProvider


class NeuralTTSProvider(TTSProvider):
    """
    Neural speech synthesizer with zero-shot reference audio conditioning and
    empirical speaker tensors. Designed for local execution on Apple Silicon M4.
    """

    def __init__(
        self,
        model_name: str = "kokoro-82m",
        models_dir: str = "/Users/Sergiu/AI/models/tts",
        voice_profile: Optional[VoiceProfile] = None,
        fallback_provider: Optional[TTSProvider] = None
    ):
        self.model_name = model_name
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.profile = voice_profile or VoiceProfile()
        self.fallback = fallback_provider or MacOSNativeTTSProvider()
        
        # Pipeline instance cache
        self._pipeline = None
        self._custom_voice_tensor = None
        self.neural_engine_ready = self._initialize_engine()

    def _initialize_engine(self) -> bool:
        """Initializes Kokoro neural TTS pipeline and loads custom voice tensor if available."""
        # Ensure VIRTUAL_ENV is visible to subprocesses / libraries
        if "VIRTUAL_ENV" not in os.environ:
            os.environ["VIRTUAL_ENV"] = "/Users/Sergiu/AI/venv-webui"

        try:
            from kokoro import KPipeline
            import torch
            
            # Lazy initialize pipeline
            self._pipeline = KPipeline(lang_code="a")

            # Check for custom calibrated voice tensor
            custom_voice_path = Path("/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_references/sergiu_voice.pt")
            if custom_voice_path.exists():
                self._custom_voice_tensor = torch.load(custom_voice_path, map_location="cpu", weights_only=False)
                self.profile.is_cloned_from_personal_media = True
            
            return True
        except Exception as e:
            print(f"[NeuralTTSProvider] Notice: Neural engine initialization deferred or unavailable: {e}")
            return False

    def is_healthy(self) -> bool:
        if self.neural_engine_ready and self._pipeline is not None:
            return True
        return self.fallback.is_healthy()

    def list_available_voices(self) -> List[str]:
        voices = ["sergiu_authentic_f5", "sergiu_voice_cloned"]
        voices.extend(["kokoro_am_adam", "kokoro_bm_george", "kokoro_bm_lewis"])
        voices.extend(self.fallback.list_available_voices())
        return voices

    def prepare_reference_conditioning(self, audio_paths: List[str]) -> Dict[str, Any]:
        """
        Validates and registers personal reference audio files for voice conditioning.
        """
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

    def synthesize(
        self,
        text: str,
        delivery: ExpressiveDeliveryParams,
        output_file_path: str,
        voice_override: Optional[str] = None
    ) -> str:
        """
        Synthesizes speech using authentic neural conditioning from Sergiu's recordings (F5-TTS),
        with fallback to Kokoro neural synthesis or macOS native speech.
        """
        # 1. Primary High-Fidelity Flow Matching with F5-TTS conditioned on Sergiu's authentic voice
        f5_script = Path("/Users/Sergiu/AI/Time_Capsule/scripts/f5_synthesize.py")
        f5_python = Path("/Users/Sergiu/AI/venv-voice/bin/python")

        use_f5 = voice_override in [None, "sergiu_authentic_f5", "sergiu_voice_cloned"]
        if f5_script.exists() and f5_python.exists() and use_f5:
            try:
                import subprocess
                effective_speed = max(0.7, min(1.2, 0.88 * delivery.speech_rate))
                cmd = [
                    str(f5_python),
                    str(f5_script),
                    "--text", text,
                    "--output", output_file_path,
                    "--speed", str(round(effective_speed, 2)),
                    "--ref_id", "ref_02",
                    "--steps", "16"
                ]
                env = os.environ.copy()
                env["PYTHONHASHSEED"] = "0"
                res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
                if res.returncode == 0 and os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 1000:
                    return output_file_path
                else:
                    print(f"[NeuralTTSProvider] Notice: F5-TTS returned code {res.returncode}, falling back to Kokoro: {res.stderr[:200]}")
            except Exception as e:
                print(f"[NeuralTTSProvider] F5-TTS invocation error: {e}, falling back to Kokoro.")

        # 2. Kokoro Neural Engine Fallback
        if self.neural_engine_ready and self._pipeline is not None:
            try:
                # Determine voice style
                voice = self._custom_voice_tensor
                if voice_override:
                    if voice_override.startswith("kokoro_"):
                        voice = voice_override.replace("kokoro_", "")
                    elif voice_override == "sergiu_voice_cloned" and self._custom_voice_tensor is not None:
                        voice = self._custom_voice_tensor
                
                if voice is None:
                    voice = "bm_george" # Default grounded male timbre

                # Map delivery speech rate and emotion presets to Kokoro speed factor
                preset = self.profile.emotion_presets.get(delivery.emotion.value, {})
                rate_mult = preset.get("rate_multiplier", 1.0)
                base_speed = (self.profile.average_speaking_rate_wpm / 145.0) # ~0.835
                effective_speed = base_speed * delivery.speech_rate * rate_mult
                effective_speed = max(0.65, min(1.35, effective_speed))

                generator = self._pipeline(text, voice=voice, speed=effective_speed)
                all_audio = []
                for _, _, audio in generator:
                    all_audio.append(audio)

                if all_audio:
                    import numpy as np
                    combined_audio = np.concatenate(all_audio) if len(all_audio) > 1 else all_audio[0]
                    
                    out_path = Path(output_file_path)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    sf.write(str(out_path), combined_audio, 24000)
                    return str(out_path)

            except Exception as e:
                print(f"[NeuralTTSProvider] Neural synthesis failed ({e}), falling back to MacOS native speech.")

        # 3. macOS Native Linear PCM WAV Fallback
        return self.fallback.synthesize(text, delivery, output_file_path, voice_override)
