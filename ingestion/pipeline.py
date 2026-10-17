"""
Media Ingestion and Processing Pipeline for Personal AI Time Capsule.
Features cryptographic SHA-256 deduplication, PyAV stream metadata extraction,
ASR transcription, atomic memory extraction, and resumable job execution.
"""

from __future__ import annotations
import os
import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from core.models import (
    MediaSource,
    MediaStatus,
    TranscriptSegment,
    MemoryItem,
    VerificationStatus
)
from core.config import CapsuleConfig
from memory.database import DatabaseManager
from memory.vector_store import VectorStore
from providers.asr.base import ASRProvider
from providers.asr.whisper_provider import PyAVWhisperASRProvider


SUPPORTED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".wav", ".m4a", ".mp3", ".flac"}


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_recording_timestamp(file_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parses timestamps from filename patterns like PXL_YYYYMMDD_HHMMSSxxx.mp4.
    Returns (iso_timestamp, date_string).
    """
    stem = Path(file_path).stem
    parts = stem.split("_")
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit() and (part.startswith("202") or part.startswith("201")):
            year = int(part[0:4])
            month = int(part[4:6])
            day = int(part[6:8])
            date_str = f"{year:04d}-{month:02d}-{day:02d}"
            if i + 1 < len(parts) and len(parts[i+1]) >= 6 and parts[i+1][:6].isdigit():
                tpart = parts[i+1][:6]
                hour = int(tpart[0:2])
                minute = int(tpart[2:4])
                second = int(tpart[4:6])
                return f"{date_str}T{hour:02d}:{minute:02d}:{second:02d}Z", date_str
            return f"{date_str}T00:00:00Z", date_str
    return None, None


class MediaIngestionPipeline:
    """Manages discovery, deduplication, transcription, and indexing of media assets."""

    def __init__(
        self,
        config: CapsuleConfig,
        db: DatabaseManager,
        vector_store: VectorStore,
        asr_provider: Optional[ASRProvider] = None
    ):
        self.config = config
        self.db = db
        self.vector_store = vector_store
        self.asr = asr_provider or PyAVWhisperASRProvider()

    def discover_media_files(self, directory_path: str) -> List[Path]:
        """
        Scans ONLY the explicitly provided directory.
        Strict invariant: Does NOT recursively scan outside the target folder.
        """
        if not directory_path or not os.path.exists(directory_path):
            return []

        target = Path(directory_path)
        if not target.is_dir():
            return []

        found = []
        for root, _, files in os.walk(target):
            for file in sorted(files):
                if file.startswith("."):
                    continue
                p = Path(root) / file
                if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                    found.append(p)
        return sorted(found)

    def extract_metadata_pyav(self, file_path: str) -> Dict[str, Any]:
        """Extracts container duration, bitrate, video/audio stream properties using PyAV."""
        metadata = {
            "duration_seconds": 0.0,
            "has_video": False,
            "has_audio": False,
            "audio_sample_rate": 0,
            "video_codec": None,
            "audio_codec": None
        }
        try:
            import av
            with av.open(file_path) as container:
                if container.duration:
                    metadata["duration_seconds"] = float(container.duration) / 1000000.0

                video_stream = next((s for s in container.streams if s.type == "video"), None)
                if video_stream:
                    metadata["has_video"] = True
                    metadata["video_codec"] = video_stream.name

                audio_stream = next((s for s in container.streams if s.type == "audio"), None)
                if audio_stream:
                    metadata["has_audio"] = True
                    metadata["audio_codec"] = audio_stream.name
                    metadata["audio_sample_rate"] = audio_stream.rate
        except Exception as e:
            metadata["error"] = str(e)
            metadata["duration_seconds"] = 1.0 # fallback

        return metadata

    def process_file(
        self,
        file_path: str,
        is_synthetic_fixture: bool = False
    ) -> Dict[str, Any]:
        """
        Executes end-to-end ingestion on a single media file:
        1. Checksum & Deduplication check
        2. Metadata extraction & timestamp parsing
        3. Local audio extraction (persisted for offline TTS reference/playback)
        4. Neural ASR transcription
        5. Semantic memory chunking & Vector indexing
        """
        p = Path(file_path)
        if not p.exists():
            return {"status": "error", "error": f"File not found: {file_path}"}

        file_hash = compute_sha256(file_path)
        existing_media = self.db.get_media_by_hash(file_hash)

        if existing_media and existing_media.status == MediaStatus.INDEXED:
            return {
                "status": "duplicate_skipped",
                "media_id": existing_media.id,
                "file_name": p.name,
                "message": "File with identical SHA-256 hash already processed and indexed in local database."
            }

        # Create or update media record
        media_id = existing_media.id if existing_media else str(uuid.uuid4())
        file_size = os.path.getsize(file_path)
        meta = self.extract_metadata_pyav(file_path)

        # Parse recording timestamp from filename
        recorded_at_iso, date_str = parse_recording_timestamp(file_path)
        meta["date_occurred"] = date_str

        # Demux and extract 16kHz mono WAV locally into audio_cache so training,
        # voice reference conditioning, and audio playback stay available offline
        try:
            audio_dir = Path(self.config.storage.audio_output_dir)
            audio_dir.mkdir(parents=True, exist_ok=True)
            cached_audio_path = audio_dir / f"{media_id}.wav"
            if not cached_audio_path.exists() and meta.get("has_audio"):
                self.asr.extract_audio_wav(file_path, str(cached_audio_path))
            if cached_audio_path.exists():
                meta["local_audio_path"] = str(cached_audio_path)
        except Exception as e:
            meta["audio_extract_note"] = str(e)

        media = MediaSource(
            id=media_id,
            file_path=str(p.resolve()),
            file_hash=file_hash,
            file_size_bytes=file_size,
            mime_type=f"video/{p.suffix.lstrip('.')}" if p.suffix.lower() in [".mp4", ".mov", ".mkv"] else f"audio/{p.suffix.lstrip('.')}",
            duration_seconds=meta.get("duration_seconds", 0.0),
            recorded_at=recorded_at_iso,
            imported_at=datetime.utcnow().isoformat(),
            status=MediaStatus.PROCESSING,
            metadata=meta
        )
        self.db.upsert_media(media)

        try:
            # 2. Transcription
            segments = self.asr.transcribe(file_path, media_id)
            if not segments:
                raise ValueError(f"No speech utterances transcribed from {p.name}")

            self.db.insert_transcript_segments(segments)

            media.status = MediaStatus.TRANSCRIBED
            self.db.upsert_media(media)

            # 3. Memory Extraction & Indexing
            # Verified personal recording tier
            v_status = VerificationStatus.SYNTHETIC_FIXTURE if is_synthetic_fixture else VerificationStatus.PERSONAL_VERIFIED

            memories_created: List[MemoryItem] = []
            
            # Aggregate consecutive utterances into coherent narrative reflection paragraphs (30-50s)
            current_chunk_text = []
            current_start = None
            current_end = None
            current_confs = []

            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue

                if current_start is None:
                    current_start = seg.start_seconds
                current_end = seg.end_seconds
                current_chunk_text.append(text)
                current_confs.append(seg.confidence)

                # Split on natural length or time span
                combined = " ".join(current_chunk_text)
                span = current_end - current_start
                if span >= 35.0 or len(combined.split()) >= 65:
                    avg_conf = sum(current_confs) / max(len(current_confs), 1)
                    summary = combined[:90].strip() + ("..." if len(combined) > 90 else "")
                    mem = MemoryItem(
                        id=str(uuid.uuid4()),
                        content=combined,
                        summary=summary,
                        category="personal_reflection",
                        date_occurred=date_str or p.stem,
                        source_id=media_id,
                        start_seconds=round(current_start, 2),
                        end_seconds=round(current_end, 2),
                        verification_status=v_status,
                        confidence=round(avg_conf, 2),
                        emotional_tone="reflective",
                        tags=["personal_vlog", "pixel_recording", f"date:{date_str}" if date_str else "vlog"]
                    )
                    self.db.add_memory(mem)
                    self.vector_store.add_memory_embedding(mem)
                    memories_created.append(mem)
                    current_chunk_text = []
                    current_start = None
                    current_end = None
                    current_confs = []

            # Final tail chunk
            if current_chunk_text and current_start is not None:
                combined = " ".join(current_chunk_text)
                if len(combined.strip()) >= 10:
                    avg_conf = sum(current_confs) / max(len(current_confs), 1)
                    summary = combined[:90].strip() + ("..." if len(combined) > 90 else "")
                    mem = MemoryItem(
                        id=str(uuid.uuid4()),
                        content=combined,
                        summary=summary,
                        category="personal_reflection",
                        date_occurred=date_str or p.stem,
                        source_id=media_id,
                        start_seconds=round(current_start, 2),
                        end_seconds=round(current_end, 2),
                        verification_status=v_status,
                        confidence=round(avg_conf, 2),
                        emotional_tone="reflective",
                        tags=["personal_vlog", "pixel_recording", f"date:{date_str}" if date_str else "vlog"]
                    )
                    self.db.add_memory(mem)
                    self.vector_store.add_memory_embedding(mem)
                    memories_created.append(mem)

            media.status = MediaStatus.INDEXED
            self.db.upsert_media(media)

            return {
                "status": "success",
                "media_id": media_id,
                "file_name": p.name,
                "recorded_at": recorded_at_iso,
                "transcript_segments": len(segments),
                "memories_created": len(memories_created),
                "verification_status": v_status.value
            }

        except Exception as e:
            media.status = MediaStatus.FAILED
            media.error_message = str(e)
            self.db.upsert_media(media)
            return {
                "status": "failed",
                "media_id": media_id,
                "file_name": p.name,
                "error": str(e)
            }

    def run_ingestion_job(
        self,
        target_directory: Optional[str] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Runs batch ingestion against the configured or specified target directory."""
        dir_to_scan = target_directory or self.config.storage.media_input_dir
        if not dir_to_scan:
            return {
                "status": "idle",
                "message": "Media input directory is not set. Awaiting user vlog path configuration.",
                "processed": 0
            }

        files = self.discover_media_files(dir_to_scan)
        results = []
        total = len(files)
        for i, f in enumerate(files):
            if progress_callback:
                progress_callback(i + 1, total, str(f))
            res = self.process_file(str(f))
            results.append(res)

        return {
            "status": "completed",
            "directory": dir_to_scan,
            "total_files_discovered": total,
            "results": results
        }
