#!/usr/bin/env python3
"""
Restore Pristine Voice Reference Audio Clips.
Re-extracts the 35 canonical reference audio clips directly from untouched source audio cache
and SQLite archive transcripts, applying clean 80Hz sub-bass rumble highpass and -18 LUFS normalization.
Eliminates all destructive notch filters, flutter gates, and phase distortion.
"""

import os
import sys
import json
import sqlite3
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal

TARGET_SR = 24000
TARGET_LUFS = -18.0
MAX_PEAK = 0.95

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "capsule"
AUDIO_CACHE_DIR = DATA_DIR / "audio_cache"
VOICE_REF_DIR = DATA_DIR / "voice_references"
ARCHIVE_DB = DATA_DIR / "archive.db"
MANIFEST_PATH = VOICE_REF_DIR / "voice_references.json"


def highpass_80hz(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """Applies a clean 4th-order Butterworth high-pass filter at 80Hz to eliminate sub-bass rumble."""
    sos = scipy.signal.butter(4, 80, btype="highpass", fs=sr, output="sos")
    return scipy.signal.sosfilt(sos, audio)


def measure_lufs(audio: np.ndarray, sr: int = TARGET_SR) -> float:
    """Measures integrated loudness using simplified ITU-R BS.1770 high-pass weighting."""
    if len(audio) == 0:
        return -70.0
    b_hp, a_hp = scipy.signal.butter(2, 100 / (sr / 2), btype="highpass")
    filtered = scipy.signal.lfilter(b_hp, a_hp, audio)
    ms = np.mean(filtered ** 2)
    if ms < 1e-12:
        return -70.0
    return float(-0.691 + 10.0 * np.log10(ms))


def normalize_lufs(audio: np.ndarray, target_lufs: float = TARGET_LUFS, sr: int = TARGET_SR) -> np.ndarray:
    """Normalizes audio to target LUFS and clamps peak to < MAX_PEAK to avoid digital clipping."""
    cur_lufs = measure_lufs(audio, sr)
    if cur_lufs <= -70.0:
        return audio
    gain_db = target_lufs - cur_lufs
    gain_lin = 10.0 ** (gain_db / 20.0)
    out = audio * gain_lin
    peak = np.max(np.abs(out))
    if peak > MAX_PEAK:
        out = out * (MAX_PEAK / peak)
    return out.astype(np.float32)


def main():
    print("=" * 70)
    print("RESTORE PRISTINE VOICE REFERENCES FROM SOURCE CACHE")
    print("=" * 70)

    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    conn = sqlite3.connect(f"file:{ARCHIVE_DB.resolve()}?mode=ro", uri=True)
    cur = conn.cursor()

    clips = manifest.get("clips", [])
    print(f"Processing {len(clips)} reference clips...")

    restored = 0
    for clip in clips:
        clip_id = clip["clip_id"]
        filename = clip["filename"]
        sid = clip["source_media_id"]
        target_text = clip["text"]
        target_path = VOICE_REF_DIR / filename

        # Find matching transcript row
        cur.execute(
            """
            SELECT start_seconds, end_seconds, text
            FROM transcripts
            WHERE source_id = ? AND text LIKE ?
            """,
            (sid, f"%{target_text[:20]}%")
        )
        row = cur.fetchone()
        if not row:
            # Fallback search by text only if source_id differed
            cur.execute(
                """
                SELECT start_seconds, end_seconds, text
                FROM transcripts
                WHERE text LIKE ?
                """,
                (f"%{target_text[:25]}%",)
            )
            row = cur.fetchone()

        if not row:
            print(f"[!] Could not locate transcript in DB for clip {clip_id}: '{target_text[:30]}...'")
            continue

        start_sec, end_sec, db_text = row
        cache_wav = AUDIO_CACHE_DIR / f"{sid}.wav"

        if not cache_wav.exists():
            print(f"[!] Source audio cache missing: {cache_wav}")
            continue

        # Extract audio slice
        with sf.SoundFile(str(cache_wav)) as sf_file:
            sr = sf_file.samplerate
            start_f = int(start_sec * sr)
            num_f = int((end_sec - start_sec) * sr)
            sf_file.seek(start_f)
            data = sf_file.read(num_f)
            if data.ndim > 1:
                data = np.mean(data, axis=1)
            if sr != TARGET_SR:
                data = scipy.signal.resample_poly(data, TARGET_SR, sr)

        # Apply gentle highpass (80Hz Butterworth) to eliminate DC & sub-bass rumble
        cleaned = highpass_80hz(data, TARGET_SR)

        # Smooth 10ms Hann fade-in and 15ms Hann fade-out to prevent boundary clicks
        fade_in_len = int(TARGET_SR * 0.010)
        fade_out_len = int(TARGET_SR * 0.015)
        if len(cleaned) > fade_in_len + fade_out_len:
            cleaned[:fade_in_len] *= np.sin(np.linspace(0, np.pi / 2, fade_in_len)) ** 2
            cleaned[-fade_out_len:] *= np.cos(np.linspace(0, np.pi / 2, fade_out_len)) ** 2

        # -18 LUFS normalization with true-peak safety limit
        pristine = normalize_lufs(cleaned, TARGET_LUFS, TARGET_SR)

        # Write clean WAV
        sf.write(str(target_path), pristine, TARGET_SR, subtype="PCM_16")
        restored += 1
        peak = float(np.max(np.abs(pristine)))
        lufs = measure_lufs(pristine, TARGET_SR)
        dur = len(pristine) / TARGET_SR
        print(f"  [✓] Restored {filename} ({clip_id}) | Dur: {dur:5.2f}s | Peak: {peak:.3f} | LUFS: {lufs:5.1f} dBFS")

    print("=" * 70)
    print(f"Successfully restored {restored} / {len(clips)} reference clips to pristine quality.")
    print("=" * 70)


if __name__ == "__main__":
    main()
