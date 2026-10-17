"""
Abstract Base Interface for Automated Speech Recognition (ASR).
Defines standard segment extraction with start/end timestamps and confidence.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List
from core.models import TranscriptSegment


class ASRProvider(ABC):
    """Abstract ASR engine for transcribing audio/video streams."""

    @abstractmethod
    def transcribe(self, media_path: str, source_id: str) -> List[TranscriptSegment]:
        """
        Transcribes the given media file into timestamped TranscriptSegments.
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Returns True if the transcription engine is ready."""
        pass
