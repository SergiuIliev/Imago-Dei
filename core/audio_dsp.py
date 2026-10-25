"""
High-Precision DSP Audio De-Humming and Noise Suppression Module.
Eliminates 50Hz/60Hz electrical mains hum, 100Hz/120Hz harmonics, DC offset,
and sub-bass acoustic rumble while preserving vocal warmth, formants, and natural cadence.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional, Union
import numpy as np
import soundfile as sf
from scipy import signal


def clean_audio_signal(
    data: np.ndarray,
    sr: int = 24000,
    hp_cutoff: float = 75.0,
    notch_freqs: Optional[list[float]] = None,
    noise_gate_enabled: bool = True
) -> np.ndarray:
    """
    Applies high-pass Butterworth filtering, surgical IIR notch filters for AC hum harmonics,
    and adaptive soft downward expansion (noise gate) for silent intervals.
    """
    if data is None or len(data) == 0:
        return data

    is_1d = data.ndim == 1
    if not is_1d and data.shape[1] == 1:
        data = data.squeeze(axis=1)
        is_1d = True

    # Process channel-by-channel if multichannel
    if not is_1d:
        channels = [clean_audio_signal(data[:, ch], sr, hp_cutoff, notch_freqs, noise_gate_enabled) for ch in range(data.shape[1])]
        return np.stack(channels, axis=1)

    # 1. Eliminate DC offset
    mean_offset = float(np.mean(data))
    filtered = data - mean_offset

    # 2. Butterworth 4th-order High-Pass Filter (removes sub-bass rumble & fundamental AC hum)
    if hp_cutoff > 0 and hp_cutoff < sr / 2:
        sos_hp = signal.butter(4, hp_cutoff, btype='highpass', fs=sr, output='sos')
        filtered = signal.sosfilt(sos_hp, filtered)

    # 3. Precision High-Q IIR Notch Filters for Mains Hum and Rectified Harmonics
    if notch_freqs is None:
        notch_freqs = [50.0, 60.0, 100.0, 120.0, 150.0, 180.0]

    for f0 in notch_freqs:
        if 10.0 < f0 < (sr / 2 - 20.0):
            # Q = 35 gives narrow 3dB bandwidth of ~1.5 - 3.5 Hz
            b_notch, a_notch = signal.iirnotch(f0, Q=35.0, fs=sr)
            filtered = signal.lfilter(b_notch, a_notch, filtered)

    # 4. Soft Downward Expander / Noise Gate for Pauses
    if noise_gate_enabled and len(filtered) > sr * 0.1:
        frame_len = max(128, int(sr * 0.020)) # 20ms frames
        hop_len = frame_len // 2
        pad_len = hop_len - (len(filtered) % hop_len)
        padded = np.pad(filtered, (0, pad_len), mode='reflect')

        # Sliding window RMS calculation
        frames = np.lib.stride_tricks.sliding_window_view(padded, frame_len)[::hop_len]
        rms = np.sqrt(np.mean(frames**2, axis=-1) + 1e-12)

        # Silence threshold at -44 dBFS (~0.006)
        silence_thresh = 0.006
        # Smooth quadratic expansion down to -30 dB attenuation during pauses
        gain = np.clip((rms / silence_thresh)**2, 0.03, 1.0)

        # Interpolate gain curve back to full audio sample length
        gain_interp = np.repeat(gain, hop_len)[:len(filtered)]
        if len(gain_interp) < len(filtered):
            gain_interp = np.pad(gain_interp, (0, len(filtered) - len(gain_interp)), mode='edge')

        filtered = filtered * gain_interp

    # 5. True Peak Normalization to -1.0 dBFS (0.891)
    peak = float(np.max(np.abs(filtered)))
    if peak > 0.005:
        target_peak = 0.891
        filtered = filtered * (target_peak / peak)

    return filtered.astype(np.float32)


def clean_wav_file(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    hp_cutoff: float = 75.0,
    noise_gate: bool = True
) -> str:
    """
    Reads a WAV file, cleans electrical hum and background noise, and saves the cleaned audio.
    If output_path is None, overwrites input_path atomically.
    """
    in_p = Path(input_path)
    if not in_p.exists():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    out_p = Path(output_path) if output_path else in_p
    out_p.parent.mkdir(parents=True, exist_ok=True)

    data, sr = sf.read(str(in_p))
    cleaned = clean_audio_signal(data, sr=sr, hp_cutoff=hp_cutoff, noise_gate_enabled=noise_gate)

    # Use atomic write via temp file if in-place overwrite
    if out_p == in_p:
        temp_out = in_p.with_suffix(".tmp.wav")
        sf.write(str(temp_out), cleaned, sr, subtype="PCM_16")
        temp_out.replace(in_p)
    else:
        sf.write(str(out_p), cleaned, sr, subtype="PCM_16")

    return str(out_p)
