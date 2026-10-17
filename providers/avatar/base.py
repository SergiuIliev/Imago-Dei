"""
Abstract Base Interface for Future Audio-Driven Video Avatar Generation.
Defines asynchronous job management, rendering parameters, and status querying.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class AvatarJobStatus(str, Enum):
    QUEUED = "queued"
    EXTRACTING_AUDIO = "extracting_audio"
    DRIVING_LANDMARKS = "driving_landmarks"
    RENDERING_FRAMES = "rendering_frames"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class AvatarRenderJob:
    job_id: str
    audio_path: str
    reference_portrait: Optional[str]
    status: AvatarJobStatus
    progress: float = 0.0 # 0.0 to 1.0
    output_video_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "audio_path": self.audio_path,
            "reference_portrait": self.reference_portrait,
            "status": self.status.value,
            "progress": round(self.progress, 2),
            "output_video_path": self.output_video_path,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "completed_at": self.completed_at
        }


class AvatarProvider(ABC):
    """Abstract interface for audio-driven visual avatar generation."""

    @abstractmethod
    def create_render_job(
        self,
        audio_path: str,
        reference_portrait: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> AvatarRenderJob:
        """Enqueues an audio-to-video avatar rendering job."""
        pass

    @abstractmethod
    def get_job_status(self, job_id: str) -> Optional[AvatarRenderJob]:
        """Queries status and progress of a rendering job."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancels an active rendering job."""
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Returns True if the avatar rendering environment is operational."""
        pass
