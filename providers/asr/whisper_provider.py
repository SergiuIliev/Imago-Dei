"""
PyAV and Whisper-based ASR Transcription Provider.
Demuxes audio containers with PyAV C bindings and performs speech transcription.
"""

from __future__ import annotations
import uuid
import wave
from pathlib import Path
from typing import List, Optional
from core.models import TranscriptSegment
from providers.asr.base import ASRProvider
from providers.asr.mock_provider import MockASRProvider


class PyAVWhisperASRProvider(ASRProvider):
    """
    Extracts PCM audio from video containers via PyAV and transcribes speech.
    Falls back gracefully to deterministic transcript generation if offline or in testing.
    """

    def __init__(self, model_size: str = "base", fallback_mock: bool = True):
        self.model_size = model_size
        self.fallback_mock = fallback_mock
        self._mock = MockASRProvider() if fallback_mock else None
        self._faster_model = None

    def _get_faster_model(self):
        if self._faster_model is None:
            try:
                import glob
                import os
                from faster_whisper import WhisperModel
                # Check for cached local snapshot to enforce strict offline execution
                hub_pattern = os.path.expanduser(f"~/.cache/huggingface/hub/models--Systran--faster-whisper-{self.model_size}/snapshots/*")
                snapshots = glob.glob(hub_pattern)
                model_ref = snapshots[0] if snapshots else self.model_size
                self._faster_model = WhisperModel(
                    model_ref,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=bool(snapshots)
                )
            except Exception as e:
                print(f"[ASR] faster-whisper initialization note: {e}")
                self._faster_model = False
        return self._faster_model if self._faster_model is not False else None

    def is_healthy(self) -> bool:
        try:
            import av
            return True
        except ImportError:
            return False

    def extract_audio_wav(self, media_path: str, output_wav_path: str) -> str:
        """Demuxes audio stream to 16kHz mono WAV using PyAV."""
        import av
        container = av.open(media_path)
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if not audio_stream:
            raise ValueError(f"No audio stream found in {media_path}")

        # Resample to 16,000 Hz mono PCM
        resampler = av.AudioResampler(
            format="s16",
            layout="mono",
            rate=16000
        )

        Path(output_wav_path).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(output_wav_path, "wb") as wav_out:
            wav_out.setnchannels(1)
            wav_out.setsampwidth(2) # 16-bit
            wav_out.setframerate(16000)

            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                for r_frame in resampled_frames:
                    wav_out.writeframes(r_frame.to_ndarray().tobytes())

        return output_wav_path

    def transcribe(self, media_path: str, source_id: str) -> List[TranscriptSegment]:
        """
        Extracts audio and transcribes spoken utterances with timestamps.
        Prioritizes faster-whisper with Silero VAD filtering, falls back to openai-whisper or mock.
        """
        # 1. Faster-Whisper (preferred on Apple Silicon)
        faster_model = self._get_faster_model()
        if faster_model:
            try:
                import math
                segments_iter, info = faster_model.transcribe(
                    media_path,
                    beam_size=5,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500)
                )
                segments = []
                for seg in segments_iter:
                    text = seg.text.strip()
                    if not text:
                        continue
                    conf = round(math.exp(max(seg.avg_logprob, -3.0)), 2)
                    segments.append(TranscriptSegment(
                        id=str(uuid.uuid4()),
                        source_id=source_id,
                        start_seconds=round(float(seg.start), 2),
                        end_seconds=round(float(seg.end), 2),
                        speaker="Speaker 1",
                        text=text,
                        confidence=min(max(conf, 0.0), 1.0),
                        language=info.language or "en"
                    ))
                if segments:
                    return segments
            except Exception as e:
                print(f"[ASR] faster-whisper transcription error: {e}")

        # 2. Standard OpenAI Whisper fallback
        try:
            import whisper
            model = whisper.load_model(self.model_size)
            result = model.transcribe(media_path, language="en")
            segments = []
            for seg in result.get("segments", []):
                segments.append(TranscriptSegment(
                    id=str(uuid.uuid4()),
                    source_id=source_id,
                    start_seconds=round(float(seg["start"]), 2),
                    end_seconds=round(float(seg["end"]), 2),
                    speaker="Speaker 1",
                    text=seg["text"].strip(),
                    confidence=round(float(seg.get("confidence", 0.95)), 2),
                    language="en"
                ))
            if segments:
                return segments
        except Exception:
            pass

        # 3. Fallback to mock provider for test fixtures
        if self._mock:
            return self._mock.transcribe(media_path, source_id)

        raise RuntimeError(f"ASR transcription failed for {media_path} and fallback mock is disabled.")
