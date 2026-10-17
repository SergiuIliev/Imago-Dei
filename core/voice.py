"""
Expressive Voice Architecture and Voice Profile Management.
Handles prosody modulation, emotional intensity, pauses, speech rate,
and preparation for future reference-conditioned voice cloning.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from core.models import ExpressiveDeliveryParams, EmotionCategory


@dataclass
class VoiceProfile:
    """
    Profile capturing speaker identity, acoustic characteristics,
    and voice conditioning reference paths.
    """
    profile_id: str = "sergiu_primary"
    speaker_name: str = "Sergiu"
    is_cloned_from_personal_media: bool = False # Set to True when personal reference media is loaded
    reference_audio_clips: List[str] = field(default_factory=list)
    average_speaking_rate_wpm: int = 121 # Empirically measured from 22 authentic vlogs (121.3 WPM)
    baseline_pitch_hz: float = 136.8 # Empirically measured median F0 from authentic speech
    pitch_range_hz: tuple = (119.4, 188.2) # 10th to 90th percentile measured pitch
    characteristic_mannerisms: List[str] = field(default_factory=lambda: [
        "Thoughtful, articulate cadence with natural pauses",
        "Calm baritone resonance with subtle melodic rise on analytical insights",
        "Crisp, clear technical pronunciation (engineering and architectural precision)",
        "Gentle dry humor with slightly relaxed pacing"
    ])
    emotion_presets: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "neutral": {"rate_multiplier": 1.0, "pitch_shift": 0.0, "pause_mult": 1.0},
        "reflective": {"rate_multiplier": 0.92, "pitch_shift": -0.05, "pause_mult": 1.3},
        "lively": {"rate_multiplier": 1.08, "pitch_shift": 0.08, "pause_mult": 0.85},
        "curious": {"rate_multiplier": 1.02, "pitch_shift": 0.06, "pause_mult": 1.05},
        "serious": {"rate_multiplier": 0.95, "pitch_shift": -0.04, "pause_mult": 1.1},
        "humorous": {"rate_multiplier": 1.04, "pitch_shift": 0.03, "pause_mult": 0.9},
        "melancholic": {"rate_multiplier": 0.88, "pitch_shift": -0.08, "pause_mult": 1.4}
    })

    def __post_init__(self):
        # Auto-detect empirical voice reference profile if present
        ref_manifest = Path("/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_references/voice_references.json")
        if ref_manifest.exists() and not self.reference_audio_clips:
            try:
                with open(ref_manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prof = data.get("profile", {})
                if prof:
                    self.average_speaking_rate_wpm = int(round(prof.get("average_wpm", self.average_speaking_rate_wpm)))
                    self.baseline_pitch_hz = float(prof.get("median_pitch_hz", self.baseline_pitch_hz))
                clips = data.get("clips", [])
                valid_paths = [c["path"] for c in clips if Path(c.get("path", "")).exists()]
                if valid_paths:
                    self.reference_audio_clips = valid_paths
                    self.is_cloned_from_personal_media = True
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> VoiceProfile:
        return cls(**data)


class ExpressiveDeliveryController:
    """
    Adapts text and timing instructions into prosodic guidance.
    Injects punctuation pauses, rates, and emotion-specific shaping.
    """

    def __init__(self, voice_profile: Optional[VoiceProfile] = None):
        self.profile = voice_profile or VoiceProfile()

    def prepare_text_and_timing(
        self,
        text: str,
        params: ExpressiveDeliveryParams
    ) -> Dict[str, Any]:
        """
        Transforms raw text into expressive speech script with pause timing,
        calculating words-per-minute and pitch adjustments.
        """
        clean_text = text.strip()
        # Clean markdown code blocks and asterisks for natural vocalization
        clean_text = re.sub(r'```.*?```', '', clean_text, flags=re.DOTALL)
        clean_text = re.sub(r'\*+(.*?)\*+', r'\1', clean_text)
        clean_text = re.sub(r'\[.*?\]', '', clean_text) # strip bracket citations from audio

        preset = self.profile.emotion_presets.get(params.emotion.value, {})
        rate_multiplier = params.speech_rate * preset.get("rate_multiplier", 1.0)
        pitch_shift = params.pitch_shift + preset.get("pitch_shift", 0.0)
        pause_mult = params.pause_duration_multiplier * preset.get("pause_mult", 1.0)

        # Apply thoughtful punctuation pauses based on pause_mult
        processed_text = clean_text
        if pause_mult > 1.2:
            # Lengthen sentence pauses and commas
            processed_text = processed_text.replace(". ", "... ")
            processed_text = processed_text.replace(", ", ", ")

        effective_wpm = int(self.profile.average_speaking_rate_wpm * rate_multiplier)
        effective_wpm = max(90, min(240, effective_wpm))

        return {
            "processed_text": processed_text,
            "effective_wpm": effective_wpm,
            "pitch_shift": round(pitch_shift, 3),
            "rate_multiplier": round(rate_multiplier, 3),
            "pause_multiplier": round(pause_mult, 3),
            "emotion": params.emotion.value,
            "intensity": params.intensity
        }
