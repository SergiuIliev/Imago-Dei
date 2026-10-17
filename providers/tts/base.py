"""
Abstract Base Class for Text-to-Speech Providers.
Enforces modularity, expressive parameters, and audio file export.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.models import ExpressiveDeliveryParams


class TTSProvider(ABC):
    """Abstract interface for speech synthesis engines."""

    @abstractmethod
    def synthesize(
        self,
        text: str,
        delivery: ExpressiveDeliveryParams,
        output_file_path: str,
        voice_override: Optional[str] = None
    ) -> str:
        """
        Synthesizes text into an audio file (e.g. WAV/AIFF) matching expressive delivery parameters.
        Returns the absolute path to the generated audio file.
        """
        pass

    @abstractmethod
    def list_available_voices(self) -> List[str]:
        """Returns list of selectable voice identifiers."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Checks if the synthesis backend is functional on the current host."""
        pass
