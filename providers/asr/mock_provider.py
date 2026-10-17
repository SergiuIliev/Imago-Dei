"""
Deterministic Mock ASR Provider for automated testing and fixture ingestion.
"""

from __future__ import annotations
import uuid
from typing import List, Optional
from core.models import TranscriptSegment
from providers.asr.base import ASRProvider


class MockASRProvider(ASRProvider):
    """Generates deterministic timestamped transcript segments for testing."""

    def __init__(self, predefined_segments: Optional[List[TranscriptSegment]] = None):
        self.predefined_segments = predefined_segments

    def is_healthy(self) -> bool:
        return True

    def transcribe(self, media_path: str, source_id: str) -> List[TranscriptSegment]:
        if self.predefined_segments:
            for s in self.predefined_segments:
                s.source_id = source_id
            return self.predefined_segments

        # Default synthetic segments
        return [
            TranscriptSegment(
                id=str(uuid.uuid4()),
                source_id=source_id,
                start_seconds=0.0,
                end_seconds=14.5,
                speaker="Speaker 1",
                text="In this project, I want to preserve my foundational thoughts on software architecture and autonomous systems.",
                confidence=0.98,
                language="en"
            ),
            TranscriptSegment(
                id=str(uuid.uuid4()),
                source_id=source_id,
                start_seconds=15.0,
                end_seconds=32.0,
                speaker="Speaker 1",
                text="Looking back to my early twenties, I used to value rapid hacking above all. Now, I prioritize durability and verified tests.",
                confidence=0.96,
                language="en"
            )
        ]
