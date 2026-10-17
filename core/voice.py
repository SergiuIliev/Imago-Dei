"""
Expressive Voice Architecture and Voice Profile Management.
Handles prosody modulation, emotional intensity, pauses, speech rate,
dynamic reference clip selection, and preparation for reference-conditioned voice cloning.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import re
from core.models import ExpressiveDeliveryParams, EmotionCategory


# Dynamic emotion-to-reference clip mapping based on authentic vlog acoustic profiles
# Evaluated from /Users/Sergiu/AI/Time_Capsule/data/capsule/voice_references/voice_references.json
EMOTION_REFERENCE_MAP: Dict[str, str] = {
    EmotionCategory.REFLECTIVE.value: "ref_02",  # 9.04s, reflective, philosophical cadence (F0: 135.6 Hz)
    EmotionCategory.HUMOROUS.value: "ref_10",    # 4.00s, playful, light British accent remark (F0: 145.5 Hz)
    EmotionCategory.LIVELY.value: "ref_10",      # 4.00s, energetic cadence (165 WPM), crisp articulation
    EmotionCategory.CURIOUS.value: "ref_04",     # 5.00s, "Another thing that I found quite interesting..."
    EmotionCategory.SERIOUS.value: "ref_01",     # 8.48s, resolute, grounded determination ("important is that if I fail...")
    EmotionCategory.MELANCHOLIC.value: "ref_03", # 7.00s, subdued, low pitch (F0: 117.6 Hz), reflective
    EmotionCategory.NEUTRAL.value: "ref_09",     # 5.00s, steady, calm, objective cadence ("guaranteed price...")
}


def get_reference_clip_for_emotion(emotion: Union[EmotionCategory, str]) -> str:
    """Returns the most acoustically appropriate reference clip ID for an emotion."""
    key = emotion.value if isinstance(emotion, EmotionCategory) else str(emotion).lower()
    return EMOTION_REFERENCE_MAP.get(key, "ref_02")


def sanitize_spoken_text(
    text: str,
    pause_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Sanitizes raw text for neural synthesis by:
    1. Stripping ALL bracketed citations like [PERSONAL_VERIFIED ...], [1], [source: ...]
    2. Stripping ALL parenthetical and asterisk stage directions like
       (pauses for a moment), (smirks slightly), (chuckles), (laughs), (smiles),
       *chuckles*, *laughs*, *pauses briefly*
    3. Translating pause hints (e.g. (pauses for a moment), *pauses briefly*, (beat))
       into clean sentence breaks and returning segment specifications with pause durations
       so silence padding (250-500ms) can be inserted cleanly into audio rather than spoken.
    4. Producing clean sentences, structured segments, and unified processed text.
    """
    clean = text.strip()

    # 1. Clean markdown code blocks
    clean = re.sub(r'```.*?```', '', clean, flags=re.DOTALL)

    # 2. Strip bracketed citations / metadata ([PERSONAL_VERIFIED ...], [1], etc.)
    clean = re.sub(
        r'\[(?:PERSONAL_VERIFIED|EXTRACTED_UNVERIFIED|SYNTHETIC_FIXTURE|source:|ref:|citation:|\d+)[^\]]*\]',
        '',
        clean,
        flags=re.IGNORECASE
    )
    clean = re.sub(r'\[.*?\]', '', clean)

    # 3. Detect and replace pause hints with internal marker __TC_PAUSE__
    pause_cue_pattern = (
        r'\(\s*(?:[^)]*?\b(?:pause|pauses|paused|pausing|beat|silence|moment of silence|brief silence)\b[^)]*?)\)'
        r'|\*\s*(?:[^*]*?\b(?:pause|pauses|paused|pausing|beat|silence)\b[^*]*?)\*'
    )
    clean = re.sub(pause_cue_pattern, ' __TC_PAUSE__ ', clean, flags=re.IGNORECASE)

    # 4. Strip ALL other parenthetical stage directions / actions
    action_cue_words = (
        r'smirk|smirks|smirking|chuckle|chuckles|chuckling|laugh|laughs|laughing|laughter|'
        r'smile|smiles|smiling|grin|grins|grinning|wink|winks|winking|sigh|sighs|sighing|'
        r'giggle|giggles|giggling|clears? throat|cough|coughs|shrug|shrugs|shrugging|'
        r'whisper|whispers|whispering|looks? away|looking away|looks? thoughtful|looking thoughtful|'
        r'softly|quietly|nod|nods|nodding|gasp|gasps|gasping|snicker|snickers|snickering'
    )
    action_cue_pattern = (
        rf'\(\s*(?:[^)]*?\b(?:{action_cue_words})\b[^)]*?)\)'
        rf'|\*\s*(?:[^*]*?\b(?:{action_cue_words})\b[^*]*?)\*'
    )
    clean = re.sub(action_cue_pattern, '', clean, flags=re.IGNORECASE)

    # Strip any remaining short parenthetical action expressions: e.g. (chuckles), (smirks)
    clean = re.sub(
        r'\(\s*[a-zA-Z\s,]{1,35}\s*\)',
        lambda m: '' if any(w in m.group(0).lower() for w in [
            'pause', 'smirk', 'chuckle', 'laugh', 'smile', 'sigh', 'wink', 'grin', 'giggle', 'shrug', 'whisper'
        ]) else m.group(0),
        clean
    )

    # Clean residual markdown asterisks while preserving emphasized text content
    clean = re.sub(r'\*+([^*]+?)\*+', r'\1', clean)

    # Clean whitespace and orphaned punctuation
    clean = re.sub(r'\s+([,.:;?!])', r'\1', clean)
    clean = re.sub(r'([.?!])\s*([.?!])+', r'\1', clean)

    # 5. Extract structured segments and sentences
    segments: List[Dict[str, Any]] = []
    pause_parts = clean.split('__TC_PAUSE__')

    for p_idx, part in enumerate(pause_parts):
        # Split on sentence boundaries
        raw_sents = re.split(r'(?<=[.!?])\s+', part.strip())
        for s_idx, raw_s in enumerate(raw_sents):
            s = re.sub(r'[ \t]+', ' ', raw_s).strip()
            # Remove any orphan punctuation at the start of sentence
            s = re.sub(r'^[,.:;?!]+\s*', '', s).strip()
            if not s:
                continue
            is_last_in_pause_part = (s_idx == len(raw_sents) - 1)
            if is_last_in_pause_part and p_idx < len(pause_parts) - 1:
                # Pause hint was explicitly requested here -> 450ms * pause_multiplier
                pause_sec = round(max(0.2, min(1.0, 0.45 * pause_multiplier)), 3)
            else:
                # Natural sentence boundary pause -> 250ms * pause_multiplier
                pause_sec = round(max(0.15, min(0.8, 0.25 * pause_multiplier)), 3)
            segments.append({
                "text": s,
                "pause_after_seconds": pause_sec
            })

    sentences = [seg["text"] for seg in segments]

    # Reconstruct single clean text
    clean_unified = " ".join(sentences)
    clean_unified = re.sub(r'[ \t]+', ' ', clean_unified).strip()

    return {
        "clean_text": clean_unified,
        "sentences": sentences,
        "segments": segments,
        "has_explicit_pauses": len(pause_parts) > 1
    }


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
    Injects punctuation pauses, rates, emotion-specific shaping, and dynamic reference clip selection.
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
        sanitizing stage directions, calculating words-per-minute, pitch adjustments,
        and dynamically selecting the reference clip for the emotion.
        """
        preset = self.profile.emotion_presets.get(params.emotion.value, {})
        rate_multiplier = params.speech_rate * preset.get("rate_multiplier", 1.0)
        pitch_shift = params.pitch_shift + preset.get("pitch_shift", 0.0)
        pause_mult = params.pause_duration_multiplier * preset.get("pause_mult", 1.0)

        # Sanitize stage directions, brackets, and citations
        sanitized = sanitize_spoken_text(text, pause_multiplier=pause_mult)
        clean_text = sanitized["clean_text"]
        sentences = sanitized["sentences"]
        segments = sanitized["segments"]

        # Apply thoughtful punctuation pauses based on pause_mult
        processed_text = clean_text
        if pause_mult > 1.2:
            # Lengthen sentence pauses and commas
            processed_text = processed_text.replace(". ", "... ")
            processed_text = processed_text.replace(", ", ", ")

        effective_wpm = int(self.profile.average_speaking_rate_wpm * rate_multiplier)
        effective_wpm = max(90, min(240, effective_wpm))

        # Dynamically map emotion to reference clip
        ref_clip_id = get_reference_clip_for_emotion(params.emotion)

        return {
            "processed_text": processed_text,
            "clean_text": clean_text,
            "sentences": sentences,
            "segments": segments,
            "effective_wpm": effective_wpm,
            "pitch_shift": round(pitch_shift, 3),
            "rate_multiplier": round(rate_multiplier, 3),
            "pause_multiplier": round(pause_mult, 3),
            "pause_seconds": round(max(0.2, min(0.8, 0.35 * pause_mult)), 3),
            "reference_clip_id": ref_clip_id,
            "emotion": params.emotion.value,
            "intensity": params.intensity
        }
