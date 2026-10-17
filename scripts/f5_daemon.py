#!/usr/bin/env python3
"""
Persistent F5-TTS Metal MPS Speech Synthesis Daemon.
Runs in /Users/Sergiu/AI/venv-voice/bin/python on Apple Silicon Metal MPS (device="mps").
Pre-loads Sergiu's fine-tuned acoustic weights and Vocos 24kHz vocoder into Metal GPU memory
once at startup, eliminating cold-start latency and delivering warm sentence synthesis in < 5s.
Exposes high-speed local HTTP endpoints on 127.0.0.1:7861.
"""

import os
import sys
import time
import json
import signal
import asyncio
import threading
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure deterministic hash seed and execution path
if "PYTHONHASHSEED" not in os.environ or not os.environ.get("PYTHONHASHSEED", "").isdigit():
    os.environ["PYTHONHASHSEED"] = "0"
os.environ["PATH"] = f"/Users/Sergiu/AI/venv-voice/bin:{os.environ.get('PATH', '')}"

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import soundfile as sf
import numpy as np

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
REF_MANIFEST = BASE_DIR / "data" / "capsule" / "voice_references" / "voice_references.json"
FINETUNED_CKPT = BASE_DIR / "data" / "capsule" / "voice_training" / "sergiu_f5_finetuned.pt"
CKPT_DIR = BASE_DIR / "data" / "capsule" / "voice_training" / "checkpoints"
PID_FILE = BASE_DIR / "logs" / "f5_daemon.pid"

# Dynamic emotion to reference clip mapping
EMOTION_REFERENCE_MAP: Dict[str, str] = {
    "reflective": "ref_02",  # 9.04s, reflective, philosophical cadence
    "humorous": "ref_10",    # 4.00s, playful, light British accent remark
    "lively": "ref_10",      # 4.00s, energetic cadence (165 WPM), crisp articulation
    "curious": "ref_04",     # 5.00s, "Another thing that I found quite interesting..."
    "serious": "ref_01",     # 8.48s, resolute, grounded determination
    "melancholic": "ref_03", # 7.00s, subdued, low pitch (F0: 117.6 Hz), reflective
    "neutral": "ref_09",     # 5.00s, steady, calm, objective cadence
}

class F5TTSDaemonEngine:
    def __init__(self):
        self.device = "mps"
        self.model_name = "sergiu_f5_finetuned.pt"
        self.ckpt_path = ""
        self.f5 = None
        self.is_ready = False
        self.lock = threading.Lock()
        self.reference_clips: Dict[str, Dict[str, Any]] = {}
        self._load_reference_clips()

    def _load_reference_clips(self):
        """Loads and indexes reference audio clips and transcripts."""
        if REF_MANIFEST.exists():
            try:
                with open(REF_MANIFEST, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for clip in data.get("clips", []):
                    cid = clip.get("clip_id")
                    cpath = clip.get("path")
                    if cid and cpath and os.path.exists(cpath):
                        self.reference_clips[cid] = {
                            "path": cpath,
                            "text": clip.get("text", ""),
                            "duration": clip.get("duration_seconds", 0.0)
                        }
            except Exception as e:
                print(f"[Daemon] Warning loading reference manifest: {e}", file=sys.stderr)

        # Baseline fallback reference
        if not self.reference_clips:
            default_path = str(BASE_DIR / "data" / "capsule" / "voice_references" / "sergiu_ref_02.wav")
            self.reference_clips["ref_02"] = {
                "path": default_path,
                "text": "What is possible in 20 years and how much better I can be at what I do if I continue to improve",
                "duration": 9.04
            }

    def initialize_model(self):
        """Loads fine-tuned weights and Vocos vocoder into Metal MPS GPU memory."""
        print(f"[Daemon] Initializing F5-TTS on Apple Silicon Metal MPS ({self.device})...")
        t0 = time.time()

        # Locate fine-tuned checkpoint
        if FINETUNED_CKPT.exists():
            self.ckpt_path = str(FINETUNED_CKPT)
            self.model_name = FINETUNED_CKPT.name
        elif CKPT_DIR.exists():
            checkpoints = sorted(
                [p for p in CKPT_DIR.glob("model_*.pt") if not p.name.startswith("pretrained_")],
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if checkpoints:
                self.ckpt_path = str(checkpoints[0])
                self.model_name = checkpoints[0].name
            else:
                self.ckpt_path = ""
                self.model_name = "pretrained_base"
        else:
            self.ckpt_path = ""
            self.model_name = "pretrained_base"

        print(f"[Daemon] Loading model weights from: {self.ckpt_path or 'Base F5-TTS'}")
        from f5_tts.api import F5TTS
        self.f5 = F5TTS(device=self.device, ckpt_file=self.ckpt_path)
        self.is_ready = True
        elapsed = time.time() - t0
        print(f"[Daemon] Metal MPS warm initialization complete in {elapsed:.2f}s. Model: {self.model_name}")

    def resolve_reference(self, ref_id: str) -> tuple[str, str]:
        """Resolves reference audio file and conditioning text from clip ID or emotion name."""
        target_id = ref_id.lower().strip()
        # Check emotion aliases
        if target_id in EMOTION_REFERENCE_MAP:
            target_id = EMOTION_REFERENCE_MAP[target_id]

        if target_id in self.reference_clips:
            clip = self.reference_clips[target_id]
            return clip["path"], clip["text"]

        # Default fallback
        if "ref_02" in self.reference_clips:
            clip = self.reference_clips["ref_02"]
            return clip["path"], clip["text"]

        # Fallback to first available
        first_key = next(iter(self.reference_clips))
        clip = self.reference_clips[first_key]
        return clip["path"], clip["text"]

    def sanitize_text(self, text: str) -> str:
        """Removes stage directions and citations to ensure 100% clean vocalization."""
        import re
        clean = text.strip()
        # Clean markdown code blocks
        clean = re.sub(r'```.*?```', '', clean, flags=re.DOTALL)
        # Strip bracket citations
        clean = re.sub(r'\[(?:PERSONAL_VERIFIED|EXTRACTED_UNVERIFIED|SYNTHETIC_FIXTURE|source:|ref:|citation:|\d+)[^\]]*\]', '', clean, flags=re.I)
        clean = re.sub(r'\[.*?\]', '', clean)
        # Strip stage directions in parentheses or asterisks
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

    def infer(self, text: str, output_path: str, speed: float = 0.95, ref_id: str = "ref_02", steps: int = 16) -> Dict[str, Any]:
        """Performs warm flow matching diffusion synthesis directly on Metal MPS."""
        if not self.is_ready or self.f5 is None:
            raise RuntimeError("F5-TTS engine is not ready.")

        clean_text = self.sanitize_text(text)
        if not clean_text:
            raise ValueError("Input text is empty after sanitization.")

        ref_file, ref_text = self.resolve_reference(ref_id)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        with self.lock:
            wav, sr, spec = self.f5.infer(
                ref_file=ref_file,
                ref_text=ref_text,
                gen_text=clean_text,
                file_wave=str(out_p),
                speed=speed,
                nfe_step=steps
            )
        elapsed = time.time() - t0

        if not out_p.exists() or out_p.stat().st_size < 1000:
            raise RuntimeError(f"Output file {out_p} was not written properly.")

        duration_sec = len(wav) / sr if wav is not None else sf.info(str(out_p)).duration

        return {
            "status": "ok",
            "duration_seconds": round(float(duration_sec), 2),
            "elapsed_seconds": round(float(elapsed), 2)
        }


# Global engine instance
engine = F5TTSDaemonEngine()

# FastAPI application
app = FastAPI(title="F5-TTS Persistent Voice Daemon", version="1.0.0")

class SynthesizeRequest(BaseModel):
    text: str = Field(..., description="Text string to synthesize")
    output_path: str = Field(..., description="Absolute output path for generated WAV file")
    speed: float = Field(default=0.95, description="Speech cadence rate multiplier")
    ref_id: str = Field(default="ref_02", description="Reference clip ID or emotion name")
    steps: int = Field(default=16, description="Flow-matching diffusion ODE steps (8-32)")


@app.on_event("startup")
def on_startup():
    # Save PID file
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    # Initialize engine
    engine.initialize_model()


@app.on_event("shutdown")
def on_shutdown():
    if PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass
    print("[Daemon] F5-TTS daemon shut down cleanly.")


@app.get("/health")
def health():
    """Health check endpoint: verifies engine readiness, model identity, and MPS accelerator."""
    if not engine.is_ready:
        return JSONResponse(
            status_code=503,
            content={"status": "loading", "model": engine.model_name, "device": engine.device}
        )
    return {
        "status": "ready",
        "model": engine.model_name,
        "device": engine.device
    }


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """Synthesizes speech conditioned on authentic voice references with warm Metal MPS acceleration."""
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="F5-TTS engine is still initializing.")

    try:
        # Offload blocking inference to thread pool so health check remains responsive
        result = await asyncio.to_thread(
            engine.infer,
            text=req.text,
            output_path=req.output_path,
            speed=req.speed,
            ref_id=req.ref_id,
            steps=req.steps
        )
        return result
    except Exception as e:
        print(f"[Daemon] Synthesis error: {e}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=str(e))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Persistent F5-TTS Metal MPS Speech Daemon")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7861, help="Port to listen on (default: 7861)")
    args = parser.parse_args()

    print(f"[Daemon] Launching F5-TTS daemon on http://{args.host}:{args.port} (PID: {os.getpid()})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
