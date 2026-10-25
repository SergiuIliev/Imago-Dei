#!/usr/bin/env python3
"""
F5-TTS Voice Synthesis and Persona Reproduction Engine for Sergiu.
Executes flow-matching diffusion on Apple Silicon Metal MPS, conditioned directly
on Sergiu's authentic reference speech recordings and fine-tuned acoustic weights.
Fast-routes to persistent daemon on 127.0.0.1:7861 if available, otherwise executes local fallback.
"""

import os
import sys
import argparse
import json
import time
import re
import urllib.request
import urllib.error
from pathlib import Path

# Ensure environment
if not os.environ.get("PYTHONHASHSEED") or not os.environ.get("PYTHONHASHSEED", "").isdigit():
    os.environ["PYTHONHASHSEED"] = "0"
os.environ["PATH"] = f"/Users/Sergiu/AI/venv-voice/bin:{os.environ.get('PATH', '')}"

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
try:
    from core.audio_dsp import clean_wav_file
except ImportError:
    clean_wav_file = None

REF_MANIFEST = BASE_DIR / "data" / "capsule" / "voice_references" / "voice_references.json"
FINETUNED_CKPT = BASE_DIR / "data" / "capsule" / "voice_training" / "sergiu_f5_finetuned.pt"
CKPT_DIR = BASE_DIR / "data" / "capsule" / "voice_training" / "checkpoints"

# Emotion to reference clip mapping
EMOTION_MAP = {
    "reflective": "ref_02",  # 9.04s, reflective, philosophical cadence
    "humorous": "ref_10",    # 4.00s, playful, light British accent remark
    "lively": "ref_10",      # 4.00s, energetic cadence (165 WPM), crisp articulation
    "curious": "ref_04",     # 5.00s, "Another thing that I found quite interesting..."
    "serious": "ref_01",     # 8.48s, resolute, grounded determination
    "melancholic": "ref_03", # 7.00s, subdued, low pitch (F0: 117.6 Hz), reflective
    "neutral": "ref_09",     # 5.00s, steady, calm, objective cadence
}

def sanitize_text(text: str) -> str:
    """Strips stage directions, asterisks, and citations from text."""
    clean = text.strip()
    clean = re.sub(r'```.*?```', '', clean, flags=re.DOTALL)
    clean = re.sub(r'\[(?:PERSONAL_VERIFIED|EXTRACTED_UNVERIFIED|SYNTHETIC_FIXTURE|source:|ref:|citation:|\d+)[^\]]*\]', '', clean, flags=re.I)
    clean = re.sub(r'\[.*?\]', '', clean)
    cues = (
        r'pause|pauses|paused|pausing|beat|silence|moment of silence|'
        r'smirk|smirks|smirking|chuckle|chuckles|chuckling|laugh|laughs|laughing|laughter|'
        r'smile|smiles|smiling|grin|grins|grinning|wink|winks|winking|sigh|sighs|sighing|'
        r'giggle|giggles|giggling|clears? throat|cough|coughs|shrug|shrugs|shrugging|'
        r'whisper|whispers|whispering|looks? away|looking away|looks? thoughtful|looking thoughtful|'
        r'softly|quietly|nod|nods|nodding'
    )
    clean = re.sub(rf'\(\s*(?:[^)]*?\b(?:{cues})\b[^)]*?)\)|\*\s*(?:[^*]*?\b(?:{cues})\b[^*]*?)\*', '', clean, flags=re.I)
    clean = re.sub(r'\(\s*[a-zA-Z\s,]{1,35}\s*\)', lambda m: '' if any(w in m.group(0).lower() for w in [
        'pause', 'smirk', 'chuckle', 'laugh', 'smile', 'sigh', 'wink', 'grin'
    ]) else m.group(0), clean)
    clean = re.sub(r'\*+([^*]+?)\*+', r'\1', clean)
    clean = re.sub(r'\s+([,.:;?!])', r'\1', clean)
    clean = re.sub(r'([.?!])\s*([.?!])+', r'\1', clean)
    clean = re.sub(r'[ \t]+', ' ', clean).strip()
    return clean

def try_daemon_synthesize(text: str, output_path: str, speed: float, ref_id: str, steps: int) -> bool:
    """Attempts fast synthesis through the persistent warm daemon on port 7861."""
    try:
        # Check health
        health_req = urllib.request.Request("http://127.0.0.1:7861/health", headers={"User-Agent": "F5Synthesize/1.0"})
        with urllib.request.urlopen(health_req, timeout=0.5) as resp:
            if resp.status != 200:
                return False
            data = json.loads(resp.read().decode("utf-8"))
            if data.get("status") != "ready":
                return False

        # Send synthesize POST
        payload = json.dumps({
            "text": text,
            "output_path": output_path,
            "speed": speed,
            "ref_id": ref_id,
            "steps": steps
        }).encode("utf-8")
        synth_req = urllib.request.Request(
            "http://127.0.0.1:7861/synthesize",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "F5Synthesize/1.0"},
            method="POST"
        )
        t0 = time.time()
        with urllib.request.urlopen(synth_req, timeout=120) as resp:
            if resp.status == 200:
                res = json.loads(resp.read().decode("utf-8"))
                elapsed = time.time() - t0
                print(f"[F5-TTS] Warm daemon synthesized audio in {elapsed:.2f}s -> {output_path} (duration: {res.get('duration_seconds')}s)")
                return True
    except Exception:
        pass
    return False

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech in Sergiu's authentic voice.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output WAV path")
    parser.add_argument("--speed", type=float, default=0.9, help="Speech rate multiplier")
    parser.add_argument("--ref_id", type=str, default="ref_02", help="Reference clip ID or emotion name")
    parser.add_argument("--steps", type=int, default=16, help="Flow matching diffusion steps")
    args = parser.parse_args()

    clean_text = sanitize_text(args.text)
    if not clean_text:
        clean_text = args.text.strip()

    # Resolve emotion alias to clip ID
    ref_id = args.ref_id.lower().strip()
    if ref_id in EMOTION_MAP:
        ref_id = EMOTION_MAP[ref_id]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Try warm persistent daemon first
    if try_daemon_synthesize(clean_text, str(out_path), args.speed, ref_id, args.steps):
        return

    # 2. Local fallback if daemon is offline
    print("[F5-TTS] Daemon offline or unreachable. Initializing local flow matching engine...")
    from f5_tts.api import F5TTS

    ref_audio = None
    ref_text = None

    if REF_MANIFEST.exists():
        with open(REF_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        clips = data.get("clips", [])
        for c in clips:
            if c.get("clip_id") == ref_id:
                ref_audio = c.get("path")
                ref_text = c.get("text")
                break
        if not ref_audio and clips:
            ref_audio = clips[0].get("path")
            ref_text = clips[0].get("text")

    if not ref_audio or not os.path.exists(ref_audio):
        ref_audio = str(BASE_DIR / "data" / "capsule" / "voice_references" / "sergiu_ref_02.wav")
        ref_text = "What is possible in 20 years and how much better I can be at what I do if I continue to improve"

    device = "mps"

    ckpt_arg = ""
    if FINETUNED_CKPT.exists():
        ckpt_arg = str(FINETUNED_CKPT)
    elif CKPT_DIR.exists():
        checkpoints = sorted(
            [p for p in CKPT_DIR.glob("model_*.pt") if not p.name.startswith("pretrained_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if checkpoints:
            ckpt_arg = str(checkpoints[0])

    f5 = F5TTS(device=device, ckpt_file=ckpt_arg)

    t0 = time.time()
    wav, sr, spec = f5.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=clean_text,
        file_wave=str(out_path),
        speed=args.speed,
        nfe_step=args.steps
    )
    elapsed = time.time() - t0
    if clean_wav_file and out_path.exists():
        try:
            clean_wav_file(str(out_path))
        except Exception as e:
            print(f"[F5-TTS] Warning: DSP de-hum failed: {e}", file=sys.stderr)
    print(f"[F5-TTS] Cloned speech generated locally in {elapsed:.2f}s -> {out_path} ({sr}Hz)")

if __name__ == "__main__":
    main()
