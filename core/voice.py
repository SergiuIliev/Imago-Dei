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
# Evaluated from data/capsule/voice_references/voice_references.json
EMOTION_REFERENCE_MAP: Dict[str, str] = {
    EmotionCategory.REFLECTIVE.value: "ref_02",  # 9.04s, reflective, philosophical cadence (F0: 135.6 Hz)
    EmotionCategory.HUMOROUS.value: "ref_10",    # 4.00s, playful, light British accent remark (F0: 145.5 Hz)
    EmotionCategory.LIVELY.value: "ref_10",      # 4.00s, energetic cadence (165 WPM), crisp articulation (canonical ref_05/ref_10)
    EmotionCategory.CURIOUS.value: "ref_04",     # 5.00s, "Another thing that I found quite interesting..."
    EmotionCategory.SERIOUS.value: "ref_01",     # 8.48s, resolute, grounded determination ("important is that if I fail...")
    EmotionCategory.MELANCHOLIC.value: "ref_03", # 7.00s, subdued, low pitch (F0: 117.6 Hz), reflective
    EmotionCategory.NEUTRAL.value: "ref_09",     # 5.00s, steady, calm, objective cadence ("guaranteed price...")
}

# 35-clip Emotion x Era Reference Matrix (7 emotions across 2023-2026 + canonical)
EMOTION_ERA_REFERENCE_MAP: Dict[str, Dict[str, str]] = {
    EmotionCategory.REFLECTIVE.value: {"canonical": "ref_02", "2023": "ref_06", "2024": "ref_07", "2025": "ref_08", "2026": "ref_11"},
    EmotionCategory.HUMOROUS.value: {"canonical": "ref_10", "2023": "ref_12", "2024": "ref_13", "2025": "ref_14", "2026": "ref_15"},
    EmotionCategory.LIVELY.value: {"canonical": "ref_05", "2023": "ref_16", "2024": "ref_17", "2025": "ref_18", "2026": "ref_19"},
    EmotionCategory.CURIOUS.value: {"canonical": "ref_04", "2023": "ref_20", "2024": "ref_21", "2025": "ref_22", "2026": "ref_23"},
    EmotionCategory.SERIOUS.value: {"canonical": "ref_01", "2023": "ref_24", "2024": "ref_25", "2025": "ref_26", "2026": "ref_27"},
    EmotionCategory.MELANCHOLIC.value: {"canonical": "ref_03", "2023": "ref_28", "2024": "ref_29", "2025": "ref_30", "2026": "ref_31"},
    EmotionCategory.NEUTRAL.value: {"canonical": "ref_09", "2023": "ref_32", "2024": "ref_33", "2025": "ref_34", "2026": "ref_35"},
}


def get_reference_clip_for_emotion(
    emotion: Union[EmotionCategory, str],
    era: Optional[str] = None
) -> str:
    """
    Returns the most acoustically appropriate reference clip ID for an emotion and optional era.
    Supports era-specific matching across 2023-2026 from the 35-clip reference matrix.
    """
    key = emotion.value if isinstance(emotion, EmotionCategory) else str(emotion).lower()
    if era:
        era_str = str(era).lower().replace("era", "").strip()
        era_key = None
        for y in ["2023", "2024", "2025", "2026", "canonical"]:
            if y in era_str:
                era_key = y
                break
        if era_key and key in EMOTION_ERA_REFERENCE_MAP:
            clip = EMOTION_ERA_REFERENCE_MAP[key].get(era_key)
            if clip:
                return clip
    return EMOTION_REFERENCE_MAP.get(key, "ref_02")


EMOTION_CRITERIA: Dict[str, str] = {
    EmotionCategory.REFLECTIVE.value: "Contemplative, philosophical reflection, life lessons, long-term compounding vision",
    EmotionCategory.HUMOROUS.value: "Playful, humorous, witty, self-deprecating, amused, or smiling tone",
    EmotionCategory.LIVELY.value: "Lively, enthusiastic, high energy, fast cadence, dynamic articulation",
    EmotionCategory.CURIOUS.value: "Inquisitive, intrigued, exploring an interesting observation or puzzle",
    EmotionCategory.SERIOUS.value: "Grounded, resolute determination, engineering discipline, failure and resilience",
    EmotionCategory.MELANCHOLIC.value: "Subdued, quiet, low-pitch, somber, vulnerable, or nostalgic reflection",
    EmotionCategory.NEUTRAL.value: "Calm, objective, factual exposition, steady informational pacing",
}


def select_emotion_and_reference_jev(
    text: str,
    jev_engine: Optional[Any] = None,
    era: Optional[str] = None,
    **kwargs: Any
) -> Tuple[EmotionCategory, str, Dict[str, Any]]:
    """
    Selects the optimal EmotionCategory and reference clip using Jev Choice primitive.
    Supports era-aware reference selection across the 35-clip reference matrix.
    Falls back cleanly to get_reference_clip_for_emotion and heuristic detection if Jev engine is unavailable.
    """
    # Disambiguate if jev_engine was passed as a string era: e.g. select_emotion_and_reference_jev(text, "2024")
    effective_jev = jev_engine
    effective_era = era or kwargs.get("era")
    if isinstance(jev_engine, str) and effective_era is None:
        effective_era = jev_engine
        effective_jev = kwargs.get("jev_engine")

    if effective_jev is not None:
        try:
            is_avail = not hasattr(effective_jev, "is_available") or effective_jev.is_available()
            if is_avail:
                choice_res = effective_jev.Choice(
                    instructions="Select the most appropriate emotional tone and vocal delivery for this spoken text.",
                    criteria=EMOTION_CRITERIA,
                    context=text
                )
                chosen_key = choice_res.get("choice", EmotionCategory.NEUTRAL.value)
                try:
                    category = EmotionCategory(chosen_key)
                except ValueError:
                    category = EmotionCategory.NEUTRAL
                clip_id = get_reference_clip_for_emotion(category, era=effective_era)
                return (category, clip_id, dict(choice_res))
        except Exception:
            pass

    # Heuristic fallback
    text_lower = text.lower()
    if any(w in text_lower for w in ["laugh", "funny", "joke", "playful", "amused", "smile", "chuckle", "british"]):
        cat = EmotionCategory.HUMOROUS
    elif any(w in text_lower for w in ["curious", "wonder", "interesting", "puzzle", "fascinating", "why"]):
        cat = EmotionCategory.CURIOUS
    elif any(w in text_lower for w in ["fail", "resilience", "discipline", "focus", "duty", "responsibility", "critical"]):
        cat = EmotionCategory.SERIOUS
    elif any(w in text_lower for w in ["remember", "vision", "compounding", "reflect", "future", "past", "growth", "mindset"]):
        cat = EmotionCategory.REFLECTIVE
    elif any(w in text_lower for w in ["excited", "great", "awesome", "amazing", "energy", "love"]):
        cat = EmotionCategory.LIVELY
    elif any(w in text_lower for w in ["sad", "difficult", "hard", "pain", "struggle", "loss", "tired"]):
        cat = EmotionCategory.MELANCHOLIC
    else:
        cat = EmotionCategory.NEUTRAL

    clip_id = get_reference_clip_for_emotion(cat, era=effective_era)
    return (cat, clip_id, {"choice": cat.value, "confidence": 0.85, "fallback": True, "era": effective_era})


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

    # 1b. Strip standardized metadata disclaimers so voice speaks pure authentic answer
    disclaimer_pattern = (
        r'^(?:Memory not found\.?\s*)?'
        r'(?:The closest memories are below\.?\s*)?'
        r'(?:Based on fine-tuning,?\s*(?:I can provide the following answer:?|my answer is:?)\s*)'
    )
    clean = re.sub(disclaimer_pattern, '', clean, flags=re.IGNORECASE).strip()
    clean = re.sub(r'^(?:Memory not found\.?\s*The closest memories are below\.?\s*)', '', clean, flags=re.IGNORECASE).strip()

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
    average_speaking_rate_wpm: int = 136 # Empirically measured from 4-year vlog corpus (136.3 WPM)
    baseline_pitch_hz: float = 132.6 # Empirically measured median F0 from authentic speech corpus
    pitch_range_hz: tuple = (117.5, 143.4) # 10th to 90th percentile measured pitch across corpus
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
        project_root = Path(__file__).resolve().parent.parent
        ref_manifest = project_root / "data" / "capsule" / "voice_references" / "voice_references.json"
        if ref_manifest.exists() and not self.reference_audio_clips:
            try:
                with open(ref_manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                prof = data.get("profile", {})
                if prof:
                    self.average_speaking_rate_wpm = int(round(prof.get("average_wpm", self.average_speaking_rate_wpm)))
                    self.baseline_pitch_hz = float(prof.get("median_pitch_hz", self.baseline_pitch_hz))
                    p10 = prof.get("pitch_10th_percentile_hz")
                    p90 = prof.get("pitch_90th_percentile_hz")
                    if p10 is not None and p90 is not None:
                        self.pitch_range_hz = (float(p10), float(p90))
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
        params: ExpressiveDeliveryParams,
        era: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transforms raw text into expressive speech script with pause timing,
        sanitizing stage directions, calculating words-per-minute, pitch adjustments,
        and dynamically selecting the reference clip for the emotion and optional era.
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

        # Dynamically map emotion and optional era to reference clip
        target_era = era or getattr(params, "era", None)
        ref_clip_id = get_reference_clip_for_emotion(params.emotion, era=target_era)

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
