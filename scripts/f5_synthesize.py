#!/usr/bin/env python3
"""
F5-TTS Voice Synthesis and Persona Reproduction Engine for Sergiu.
Executes flow-matching diffusion on Apple Silicon Metal MPS, conditioned directly
on Sergiu's authentic reference speech recordings and fine-tuned acoustic weights.
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path

# Ensure environment
if not os.environ.get("PYTHONHASHSEED") or not os.environ.get("PYTHONHASHSEED", "").isdigit():
    os.environ["PYTHONHASHSEED"] = "0"
os.environ["PATH"] = f"/Users/Sergiu/AI/venv-voice/bin:{os.environ.get('PATH', '')}"

from f5_tts.api import F5TTS

BASE_DIR = Path(__file__).resolve().parent.parent
REF_MANIFEST = BASE_DIR / "data" / "capsule" / "voice_references" / "voice_references.json"
FINETUNED_CKPT = BASE_DIR / "data" / "capsule" / "voice_training" / "sergiu_f5_finetuned.pt"
CKPT_DIR = BASE_DIR / "data" / "capsule" / "voice_training" / "checkpoints"

def main():
    parser = argparse.ArgumentParser(description="Synthesize speech in Sergiu's authentic voice.")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--output", type=str, required=True, help="Output WAV path")
    parser.add_argument("--speed", type=float, default=0.9, help="Speech rate multiplier")
    parser.add_argument("--ref_id", type=str, default="ref_02", help="Reference clip ID from voice_references.json")
    parser.add_argument("--steps", type=int, default=16, help="Flow matching diffusion steps")
    args = parser.parse_args()

    # Load reference clip metadata
    ref_audio = None
    ref_text = None

    if REF_MANIFEST.exists():
        with open(REF_MANIFEST, "r", encoding="utf-8") as f:
            data = json.load(f)
        clips = data.get("clips", [])
        for c in clips:
            if c.get("clip_id") == args.ref_id:
                ref_audio = c.get("path")
                ref_text = c.get("text")
                break
        if not ref_audio and clips:
            ref_audio = clips[0].get("path")
            ref_text = clips[0].get("text")

    if not ref_audio or not os.path.exists(ref_audio):
        ref_audio = str(BASE_DIR / "data" / "capsule" / "voice_references" / "sergiu_ref_02.wav")
        ref_text = "What is possible in 20 years and how much better I can be at what I do if I continue to improve"

    device = "mps" # Apple Silicon Metal GPU acceleration

    # Check for fine-tuned weights: final model or latest training checkpoint
    ckpt_arg = ""
    if FINETUNED_CKPT.exists():
        ckpt_arg = str(FINETUNED_CKPT)
        print(f"[F5-TTS] Using final fine-tuned voice weights: {ckpt_arg}")
    elif CKPT_DIR.exists():
        checkpoints = sorted(
            [p for p in CKPT_DIR.glob("model_*.pt") if not p.name.startswith("pretrained_")],
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if checkpoints:
            ckpt_arg = str(checkpoints[0])
            print(f"[F5-TTS] Using latest fine-tuned checkpoint: {ckpt_arg}")

    print(f"[F5-TTS] Initializing flow matching engine on {device}...")
    f5 = F5TTS(device=device, ckpt_file=ckpt_arg)

    print(f"[F5-TTS] Conditioning on Sergiu reference audio: {os.path.basename(ref_audio)}")
    print(f"[F5-TTS] Generating speech: \"{args.text[:80]}...\"")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    wav, sr, spec = f5.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=args.text,
        file_wave=str(out_path),
        speed=args.speed,
        nfe_step=args.steps
    )
    elapsed = time.time() - t0
    print(f"[F5-TTS] Cloned speech generated in {elapsed:.2f}s -> {out_path} ({sr}Hz)")

if __name__ == "__main__":
    main()
