"""
macOS Native Speech Synthesis Provider.
Leverages macOS native high-quality speech engine (e.g. Daniel, Samantha, Alex)
with rate, pitch, pause modulation, and direct 24kHz 16-bit PCM WAV export.
Zero external cloud communication.
"""

from __future__ import annotations
import subprocess
import os
from pathlib import Path
from typing import List, Optional
from core.models import ExpressiveDeliveryParams
from core.voice import ExpressiveDeliveryController
from providers.tts.base import TTSProvider


class MacOSNativeTTSProvider(TTSProvider):
    """Local macOS TTS engine utilizing `/usr/bin/say` with precise audio formatting."""

    def __init__(
        self,
        default_voice: str = "Daniel",
        sample_rate: int = 24000,
        delivery_controller: Optional[ExpressiveDeliveryController] = None
    ):
        self.default_voice = default_voice
        self.sample_rate = sample_rate
        self.controller = delivery_controller or ExpressiveDeliveryController()

    def is_healthy(self) -> bool:
        return os.path.exists("/usr/bin/say")

    def list_available_voices(self) -> List[str]:
        try:
            res = subprocess.run(
                ["/usr/bin/say", "-v", "?"],
                capture_output=True,
                text=True,
                check=True
            )
            voices = []
            for line in res.stdout.strip().split("\n"):
                if line.strip():
                    parts = line.split()
                    if parts:
                        voices.append(parts[0])
            return voices
        except Exception:
            return ["Daniel", "Samantha", "Alex"]

    def synthesize(
        self,
        text: str,
        delivery: ExpressiveDeliveryParams,
        output_file_path: str,
        voice_override: Optional[str] = None
    ) -> str:
        """
        Synthesizes text to 24000Hz 16-bit mono PCM WAV file.
        """
        voice = voice_override or self.default_voice
        prep = self.controller.prepare_text_and_timing(text, delivery)
        processed_text = prep["processed_text"]
        wpm = prep["effective_wpm"]

        Path(output_file_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "/usr/bin/say",
            "--file-format=WAVE",
            f"--data-format=LEI16@{self.sample_rate}",
            "-v", voice,
            "-r", str(wpm),
            "-o", output_file_path,
            processed_text
        ]

        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if not os.path.exists(output_file_path):
                raise RuntimeError(f"TTS synthesis output file was not created: {output_file_path}")
            return output_file_path
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"macOS say synthesis failed: {e.stderr.decode('utf-8') if e.stderr else str(e)}")
