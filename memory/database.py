"""
SQLite Database Layer with FTS5 Full-Text Search and WAL Mode.
Provides durable persistence, transaction isolation, deduplication by file hash,
belief timeline tracking, and full audit trail for memory corrections.
"""

from __future__ import annotations
import sqlite3
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from core.models import (
    MediaSource,
    TranscriptSegment,
    MemoryItem,
    TemporalBelief,
    MemoryCorrection,
    VerificationStatus,
    MediaStatus
)


SCHEMA_VERSION = 1

INIT_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS media_sources (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    file_size_bytes INTEGER NOT NULL,
    mime_type TEXT,
    duration_seconds REAL DEFAULT 0.0,
    recorded_at TEXT,
    imported_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_media_hash ON media_sources(file_hash);
CREATE INDEX IF NOT EXISTS idx_media_status ON media_sources(status);

CREATE TABLE IF NOT EXISTS transcripts (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL,
    start_seconds REAL NOT NULL,
    end_seconds REAL NOT NULL,
    speaker TEXT DEFAULT 'Unknown',
    text TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    language TEXT DEFAULT 'en',
    FOREIGN KEY(source_id) REFERENCES media_sources(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_transcripts_source ON transcripts(source_id);
CREATE INDEX IF NOT EXISTS idx_transcripts_timestamps ON transcripts(source_id, start_seconds);

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    summary TEXT,
    category TEXT DEFAULT 'reflection',
    date_occurred TEXT,
    source_id TEXT,
    start_seconds REAL,
    end_seconds REAL,
    verification_status TEXT NOT NULL,
    confidence REAL DEFAULT 1.0,
    emotional_tone TEXT DEFAULT 'neutral',
    tags TEXT DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES media_sources(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(verification_status);
CREATE INDEX IF NOT EXISTS idx_memories_date ON memories(date_occurred);
CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);

-- Virtual Table for Full-Text Search (FTS5)
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    id UNINDEXED,
    content,
    summary,
    category,
    tags
);

CREATE TABLE IF NOT EXISTS temporal_beliefs (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    belief_statement TEXT NOT NULL,
    era_label TEXT NOT NULL,
    approx_year_start INTEGER,
    approx_year_end INTEGER,
    evidence_memory_ids TEXT DEFAULT '[]',
    superseded_by_id TEXT,
    supersedes_id TEXT,
    active INTEGER NOT NULL DEFAULT 1,
    reason_for_change TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_beliefs_topic ON temporal_beliefs(topic);
CREATE INDEX IF NOT EXISTS idx_beliefs_era ON temporal_beliefs(era_label);

CREATE TABLE IF NOT EXISTS memory_corrections (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    original_content TEXT NOT NULL,
    corrected_content TEXT NOT NULL,
    correction_reason TEXT NOT NULL,
    corrected_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_corrections_memory ON memory_corrections(memory_id);
"""


from contextlib import contextmanager

class DatabaseManager:
    """Manages SQLite connections, migrations, CRUD, and FTS5 search."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self):
        with self.get_connection() as conn:
            conn.executescript(INIT_SQL)
            cur = conn.execute("SELECT version FROM schema_migrations WHERE version = ?", (SCHEMA_VERSION,))
            if not cur.fetchone():
                conn.execute("INSERT INTO schema_migrations (version) VALUES (?)", (SCHEMA_VERSION,))
            conn.commit()

    # --- Media Source Operations ---

    def get_media_by_hash(self, file_hash: str) -> Optional[MediaSource]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM media_sources WHERE file_hash = ?", (file_hash,)).fetchone()
            if row:
                return MediaSource.from_dict(dict(row))
        return None

    def get_media_by_id(self, media_id: str) -> Optional[MediaSource]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM media_sources WHERE id = ?", (media_id,)).fetchone()
            if row:
                return MediaSource.from_dict(dict(row))
        return None

    def upsert_media(self, media: MediaSource) -> str:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO media_sources (
                    id, file_path, file_hash, file_size_bytes, mime_type,
                    duration_seconds, recorded_at, imported_at, status, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    file_path = excluded.file_path,
                    status = excluded.status,
                    duration_seconds = excluded.duration_seconds,
                    error_message = excluded.error_message,
                    metadata = excluded.metadata
            """, (
                media.id, media.file_path, media.file_hash, media.file_size_bytes,
                media.mime_type, media.duration_seconds, media.recorded_at, media.imported_at,
                media.status.value, media.error_message, json.dumps(media.metadata)
            ))
            conn.commit()
        return media.id

    def list_all_media(self) -> List[MediaSource]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM media_sources ORDER BY imported_at DESC").fetchall()
            return [MediaSource.from_dict(dict(r)) for r in rows]

    # --- Transcript Operations ---

    def insert_transcript_segments(self, segments: List[TranscriptSegment]):
        if not segments:
            return
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO transcripts (
                    id, source_id, start_seconds, end_seconds, speaker, text, confidence, language
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (s.id, s.source_id, s.start_seconds, s.end_seconds, s.speaker, s.text, s.confidence, s.language)
                for s in segments
            ])
            conn.commit()

    def get_transcripts_for_source(self, source_id: str) -> List[TranscriptSegment]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM transcripts WHERE source_id = ? ORDER BY start_seconds ASC", (source_id,)).fetchall()
            return [TranscriptSegment.from_dict(dict(r)) for r in rows]

    # --- Memory Operations ---

    def add_memory(self, memory: MemoryItem) -> str:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO memories (
                    id, content, summary, category, date_occurred, source_id,
                    start_seconds, end_seconds, verification_status, confidence,
                    emotional_tone, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory.id, memory.content, memory.summary, memory.category, memory.date_occurred,
                memory.source_id, memory.start_seconds, memory.end_seconds, memory.verification_status.value,
                memory.confidence, memory.emotional_tone, json.dumps(memory.tags), memory.created_at, memory.updated_at
            ))
            # Keep FTS5 table synchronized
            conn.execute("DELETE FROM memories_fts WHERE id = ?", (memory.id,))
            conn.execute("""
                INSERT INTO memories_fts (id, content, summary, category, tags)
                VALUES (?, ?, ?, ?, ?)
            """, (memory.id, memory.content, memory.summary, memory.category, " ".join(memory.tags)))
            conn.commit()
        return memory.id

    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
            if row:
                return MemoryItem.from_dict(dict(row))
        return None

    def list_memories(self, limit: int = 100, status_filter: Optional[str] = None) -> List[MemoryItem]:
        query = "SELECT * FROM memories"
        params = []
        if status_filter:
            query += " WHERE verification_status = ?"
            params.append(status_filter)
        query += " ORDER BY date_occurred DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [MemoryItem.from_dict(dict(r)) for r in rows]

    def correct_memory(self, correction: MemoryCorrection) -> bool:
        """Applies a user correction to an existing memory and records audit trail."""
        with self.get_connection() as conn:
            mem_row = conn.execute("SELECT * FROM memories WHERE id = ?", (correction.memory_id,)).fetchone()
            if not mem_row:
                return False
            # Insert audit record
            conn.execute("""
                INSERT INTO memory_corrections (
                    id, memory_id, original_content, corrected_content,
                    correction_reason, corrected_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                correction.id, correction.memory_id, correction.original_content,
                correction.corrected_content, correction.correction_reason,
                correction.corrected_by, correction.created_at
            ))
            # Update memory and mark as PERSONAL_VERIFIED
            conn.execute("""
                UPDATE memories SET
                    content = ?,
                    verification_status = ?,
                    updated_at = ?
                WHERE id = ?
            """, (
                correction.corrected_content,
                VerificationStatus.PERSONAL_VERIFIED.value,
                correction.created_at,
                correction.memory_id
            ))
            # Update FTS5
            conn.execute("UPDATE memories_fts SET content = ? WHERE id = ?", (correction.corrected_content, correction.memory_id))
            conn.commit()
        return True

    def get_corrections(self, memory_id: str) -> List[MemoryCorrection]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM memory_corrections WHERE memory_id = ? ORDER BY created_at DESC", (memory_id,)).fetchall()
            return [MemoryCorrection.from_dict(dict(r)) for r in rows]

    # --- Full-Text Search (FTS5) ---

    def search_keyword(self, query_string: str, limit: int = 20) -> List[Tuple[MemoryItem, float]]:
        """Executes BM25 ranking full-text search against the FTS5 virtual table."""
        # Sanitize query for FTS5 syntax
        clean_terms = [t.replace('"', '').replace("'", "") for t in query_string.split() if t.isalnum()]
        if not clean_terms:
            return []
        match_query = " ".join(clean_terms)

        sql = """
            SELECT m.*, bm25(memories_fts) as rank_score
            FROM memories_fts fts
            JOIN memories m ON fts.id = m.id
            WHERE memories_fts MATCH ?
            ORDER BY rank_score ASC
            LIMIT ?
        """
        results = []
        with self.get_connection() as conn:
            try:
                rows = conn.execute(sql, (match_query, limit)).fetchall()
                for r in rows:
                    item_dict = dict(r)
                    rank = abs(float(item_dict.pop('rank_score', 1.0))) # BM25 in SQLite is negative where lower is better
                    results.append((MemoryItem.from_dict(item_dict), rank))
            except sqlite3.OperationalError:
                # If MATCH syntax fails, fallback to LIKE
                like_sql = "SELECT * FROM memories WHERE content LIKE ? LIMIT ?"
                rows = conn.execute(like_sql, (f"%{clean_terms[0]}%", limit)).fetchall()
                for r in rows:
                    results.append((MemoryItem.from_dict(dict(r)), 1.0))
        return results

    # --- Temporal Beliefs Operations ---

    def add_belief(self, belief: TemporalBelief) -> str:
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO temporal_beliefs (
                    id, topic, belief_statement, era_label, approx_year_start,
                    approx_year_end, evidence_memory_ids, superseded_by_id,
                    supersedes_id, active, reason_for_change, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                belief.id, belief.topic, belief.belief_statement, belief.era_label,
                belief.approx_year_start, belief.approx_year_end, json.dumps(belief.evidence_memory_ids),
                belief.superseded_by_id, belief.supersedes_id, 1 if belief.active else 0,
                belief.reason_for_change, belief.created_at
            ))
            conn.commit()
        return belief.id

    def get_beliefs_for_topic(self, topic: str) -> List[TemporalBelief]:
        """Returns all historical and current beliefs on a topic to demonstrate evolution."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_beliefs WHERE topic LIKE ? ORDER BY approx_year_start ASC", (f"%{topic}%",)).fetchall()
            return [TemporalBelief.from_dict(dict(r)) for r in rows]

    def get_beliefs_for_era(self, era_label: str) -> List[TemporalBelief]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_beliefs WHERE era_label LIKE ? ORDER BY approx_year_start ASC", (f"%{era_label}%",)).fetchall()
            return [TemporalBelief.from_dict(dict(r)) for r in rows]

    def list_all_beliefs(self) -> List[TemporalBelief]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_beliefs ORDER BY approx_year_start ASC, created_at ASC").fetchall()
            return [TemporalBelief.from_dict(dict(r)) for r in rows]

    def get_archive_stats(self) -> Dict[str, Any]:
        with self.get_connection() as conn:
            media_count = conn.execute("SELECT COUNT(*) FROM media_sources").fetchone()[0]
            transcript_count = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
            memory_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            verified_count = conn.execute("SELECT COUNT(*) FROM memories WHERE verification_status = ?", (VerificationStatus.PERSONAL_VERIFIED.value,)).fetchone()[0]
            synthetic_count = conn.execute("SELECT COUNT(*) FROM memories WHERE verification_status = ?", (VerificationStatus.SYNTHETIC_FIXTURE.value,)).fetchone()[0]
            unverified_count = conn.execute("SELECT COUNT(*) FROM memories WHERE verification_status = ?", (VerificationStatus.EXTRACTED_UNVERIFIED.value,)).fetchone()[0]
            belief_count = conn.execute("SELECT COUNT(*) FROM temporal_beliefs").fetchone()[0]
            correction_count = conn.execute("SELECT COUNT(*) FROM memory_corrections").fetchone()[0]

            return {
                "media_files": media_count,
                "transcripts": transcript_count,
                "total_memories": memory_count,
                "verified_memories": verified_count,
                "synthetic_fixtures": synthetic_count,
                "unverified_memories": unverified_count,
                "temporal_beliefs": belief_count,
                "memory_corrections": correction_count
            }
