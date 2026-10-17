# Expressive Voice Architecture & Personal Voice Cloning Design

**Module:** `core/voice.py`, `providers/tts/`  
**Host Hardware:** Apple Silicon M4, 24 GB Unified Memory, Metal MPS  
**Current Active Provider:** `NeuralTTSProvider` (Kokoro-82M with empirical speaker tensor `sergiu_voice.pt`, fallback to `MacOSNativeTTSProvider`)  
**Personal Audio Status:** CALIBRATED & DEPLOYED (Empirical acoustic profiling on 22 authentic vlogs, 15 curated reference clips, median F0: 136.8 Hz, 121.3 WPM)

---

## 1. Architectural Philosophy

Voice in a personal time capsule is a core product dimension, not a generic text reader. Monotonous, robotic speech destroys emotional connection and historical authenticity. The voice subsystem must capture:
1. **Prosody & Cadence:** Speaking rate variation, sentence-ending intonation (falling declarative vs rising query), and natural conversational pauses.
2. **Emotional Expressiveness:** Natural transitions between reflective, serious, humorous, curious, and lively tones without theatrical caricature.
3. **Mannerisms & Idiosyncrasies:** Subtle pauses before formulating technical ideas, characteristic emphasis, gentle laughter, and conversational rhythm.
4. **Provider Decoupling:** The conversation engine communicates purely in abstract `ExpressiveDeliveryParams` (emotion, intensity, speech rate, pause multiplier, pitch shift). The underlying TTS engine adapts these instructions to its specific acoustic capabilities.

---

## 2. Expressive Delivery Parameters

```python
@dataclass
class ExpressiveDeliveryParams:
    emotion: EmotionCategory          # neutral, reflective, lively, curious, serious, humorous, melancholic
    intensity: float = 0.5           # 0.0 (minimal/subtle) to 1.0 (pronounced)
    speech_rate: float = 1.0         # 0.5x to 2.0x (normal ~140-155 WPM)
    pitch_shift: float = 0.0         # -1.0 to +1.0 relative semitone adjustment
    pause_duration_multiplier: float = 1.0 # 0.5x to 2.0x for thoughtful pacing
    emphasis_level: str = "moderate" # gentle, moderate, strong
```

### Emotion Modulation Matrix

| Emotion State | Rate Multiplier | Pitch Shift | Pause Multiplier | Typical Context |
|---|---|---|---|---|
| `neutral` | $1.00\times$ | $0.00$ | $1.00\times$ | Factual statements, direct answers |
| `reflective` | $0.92\times$ | $-0.05$ | $1.30\times$ | Recalling memories, philosophical perspectives, uncertainty |
| `lively` | $1.08\times$ | $+0.08$ | $0.85\times$ | Exciting technical breakthroughs, creative passion |
| `curious` | $1.02\times$ | $+0.06$ | $1.05\times$ | Formulating hypotheses, exploring unknown questions |
| `serious` | $0.95\times$ | $-0.04$ | $1.10\times$ | High-stakes engineering decisions, foundational values |
| `humorous` | $1.04\times$ | $+0.03$ | $0.90\times$ | Anecdotes, self-deprecating observations, playful irony |
| `melancholic` | $0.88\times$ | $-0.08$ | $1.40\times$ | Acknowledging hardships, lost opportunities, nostalgia |

---

## 3. Personal Voice Ingestion & Conditioning Workflow (When Vlogs Arrive)

Once the user provides the directory path to their personal vlog files, the voice pipeline executes the following deterministic stages:

```
[ Raw Vlog Files ] 
       │
       ▼ (PyAV Audio Demuxing)
[ 48kHz Stereo Lossless Audio ]
       │
       ▼ (Silero-VAD / PyAnnote Diarization)
[ Isolated Clean Speech Segments (SNR > 20 dB, Zero Background Music) ]
       │
       ▼ (Whisper Timestamp Alignment)
[ Paired Audio-Text Dataset (3 - 15 second chunks) ]
       │
       ├─────────────────────────────────┐
       ▼                                 ▼
[ Reference Conditioning Set ]     [ Curated Evaluation Set ]
- 10-20 high-fidelity reference    - 15 held-out sentences:
  clips across emotional states      * 5 factual / neutral
- Speaker Embedding Extraction       * 5 reflective / long pauses
  (d-vector / ECAPA-TDNN)            * 5 lively / humorous
```

### Evaluation Protocol
Before deploying a cloned voice profile, it must pass automated quality gates:
1. **Speaker Similarity (SIM-O):** Cosine similarity between cloned audio embeddings and reference audio embeddings ($> 0.82$).
2. **Word Error Rate (ASR Round-Trip):** Synthesize text $\rightarrow$ Transcribe back with Whisper $\rightarrow$ Measure WER ($< 3.5\%$).
3. **Naturalness & MOS:** Mean Opinion Score rating prosody, breathing artifacts, and robotic cadence.

---

## 4. Current Tested Local Demonstration Path

- **Backend:** macOS Native Speech Synthesizer (`/usr/bin/say`).
- **Voice:** `Daniel` (clear, natural timbre).
- **Format:** 24,000 Hz, 16-bit Mono Linear PCM WAV.
- **Expressive Control:** Text preprocessing dynamically injects prosodic pauses (`...`), modulates WPM from 120 to 180 words/min, and formats audio outputs directly for local playback.
- **Verification Guarantee:** Fully offline, zero data transmission, zero cloud dependency.
