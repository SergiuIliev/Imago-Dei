"""
Core Data Models for Personal AI Time Capsule.
Defines schemas for Media, Transcripts, Memories, Temporal Beliefs, Corrections,
Persona Snapshots, and Expressive Voice Delivery.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import json
import uuid


class VerificationStatus(str, Enum):
    """Integrity and provenance tier of a memory record."""
    SYNTHETIC_FIXTURE = "synthetic_fixture"     # Synthetic test data for testing pipeline
    EXTRACTED_UNVERIFIED = "extracted_unverified" # Extracted automatically from media, pending user review
    PERSONAL_VERIFIED = "personal_verified"       # Authenticated and confirmed personal memory


class EmotionCategory(str, Enum):
    """Expressive emotional delivery categories."""
    NEUTRAL = "neutral"
    REFLECTIVE = "reflective"
    LIVELY = "lively"
    CURIOUS = "curious"
    SERIOUS = "serious"
    HUMOROUS = "humorous"
    MELANCHOLIC = "melancholic"


class MediaStatus(str, Enum):
    """Ingestion pipeline lifecycle status for a media item."""
    DISCOVERED = "discovered"
    PROCESSING = "processing"
    TRANSCRIBED = "transcribed"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass
class MediaSource:
    """Represents an ingested video or audio file."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str = ""
    file_hash: str = "" # SHA-256 checksum
    file_size_bytes: int = 0
    mime_type: str = ""
    duration_seconds: float = 0.0
    recorded_at: Optional[str] = None # ISO format string
    imported_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: MediaStatus = MediaStatus.DISCOVERED
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MediaSource:
        data_copy = dict(data)
        if isinstance(data_copy.get('status'), str):
            data_copy['status'] = MediaStatus(data_copy['status'])
        if isinstance(data_copy.get('metadata'), str):
            try:
                data_copy['metadata'] = json.loads(data_copy['metadata'])
            except Exception:
                data_copy['metadata'] = {}
        return cls(**data_copy)


@dataclass
class TranscriptSegment:
    """A timestamped spoken segment from an ASR transcription."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    start_seconds: float = 0.0
    end_seconds: float = 0.0
    speaker: str = "Unknown"
    text: str = ""
    confidence: float = 1.0
    language: str = "en"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TranscriptSegment:
        return cls(**data)


@dataclass
class MemoryItem:
    """
    An atomic unit of remembered experience, belief, or observation.
    Must maintain provenance and verification status.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    summary: str = ""
    category: str = "reflection" # event, belief, reflection, relationship, career, lifestyle
    date_occurred: Optional[str] = None # YYYY-MM-DD or era
    source_id: Optional[str] = None
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None
    verification_status: VerificationStatus = VerificationStatus.EXTRACTED_UNVERIFIED
    confidence: float = 1.0
    emotional_tone: str = EmotionCategory.NEUTRAL.value
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['verification_status'] = self.verification_status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryItem:
        data_copy = dict(data)
        if isinstance(data_copy.get('verification_status'), str):
            data_copy['verification_status'] = VerificationStatus(data_copy['verification_status'])
        if isinstance(data_copy.get('tags'), str):
            try:
                data_copy['tags'] = json.loads(data_copy['tags'])
            except Exception:
                data_copy['tags'] = [t.strip() for t in data_copy['tags'].split(',') if t.strip()]
        return cls(**data_copy)


@dataclass
class TemporalBelief:
    """
    A specific belief, value, or preference held during a defined time window.
    Crucial for preserving historical viewpoints even when superseded later.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic: str = ""
    belief_statement: str = ""
    era_label: str = "" # e.g. "Early 20s", "2018-2020", "Present"
    approx_year_start: Optional[int] = None
    approx_year_end: Optional[int] = None
    evidence_memory_ids: List[str] = field(default_factory=list)
    superseded_by_id: Optional[str] = None # Points to newer belief that replaced this
    supersedes_id: Optional[str] = None    # Points to older belief this replaced
    active: bool = True
    reason_for_change: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TemporalBelief:
        data_copy = dict(data)
        if isinstance(data_copy.get('evidence_memory_ids'), str):
            try:
                data_copy['evidence_memory_ids'] = json.loads(data_copy['evidence_memory_ids'])
            except Exception:
                data_copy['evidence_memory_ids'] = []
        if isinstance(data_copy.get('active'), int):
            data_copy['active'] = bool(data_copy['active'])
        return cls(**data_copy)


@dataclass
class MemoryCorrection:
    """Audit record when user manually edits or refines an extracted memory."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str = ""
    original_content: str = ""
    corrected_content: str = ""
    correction_reason: str = ""
    corrected_by: str = "user"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryCorrection:
        return cls(**data)


@dataclass
class ExpressiveDeliveryParams:
    """Fine-grained delivery controls for expressive voice synthesis."""
    emotion: EmotionCategory = EmotionCategory.NEUTRAL
    intensity: float = 0.5 # 0.0 to 1.0
    speech_rate: float = 1.0 # 0.5 (slower) to 2.0 (faster)
    pitch_shift: float = 0.0 # -1.0 to 1.0
    pause_duration_multiplier: float = 1.0 # 0.5 to 2.0
    emphasis_level: str = "moderate" # gentle, moderate, strong

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['emotion'] = self.emotion.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExpressiveDeliveryParams:
        data_copy = dict(data)
        if isinstance(data_copy.get('emotion'), str):
            data_copy['emotion'] = EmotionCategory(data_copy['emotion'])
        return cls(**data_copy)


@dataclass
class PersonaSnapshot:
    """A chronological snapshot representing beliefs and tone at a specific life stage."""
    era_id: str = ""
    era_name: str = ""
    target_age_range: str = ""
    time_period: str = ""
    worldview_summary: str = ""
    key_priorities: List[str] = field(default_factory=list)
    speaking_style_notes: str = ""
    default_delivery: ExpressiveDeliveryParams = field(default_factory=ExpressiveDeliveryParams)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['default_delivery'] = self.default_delivery.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PersonaSnapshot:
        data_copy = dict(data)
        if isinstance(data_copy.get('default_delivery'), dict):
            data_copy['default_delivery'] = ExpressiveDeliveryParams.from_dict(data_copy['default_delivery'])
        return cls(**data_copy)


@dataclass
class ChatSession:
    """A conversational session containing multi-turn dialogue history."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = "New Conversation"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatSession:
        return cls(**data)


@dataclass
class ChatMessage:
    """An individual conversational turn within a session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    role: str = "user"  # "user" or "assistant"
    content: str = ""
    evidence_json: Optional[str] = None
    audio_url: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChatMessage:
        return cls(**data)

