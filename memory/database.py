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
    MediaStatus,
    ChatSession,
    ChatMessage
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

CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    evidence_json TEXT,
    audio_url TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created ON chat_messages(created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS temporal_beliefs_fts USING fts5(
    id UNINDEXED,
    topic,
    belief_statement,
    era_label,
    reason_for_change
);
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
            # Sync temporal_beliefs_fts with existing beliefs if table has missing entries
            try:
                conn.execute("""
                    INSERT INTO temporal_beliefs_fts (id, topic, belief_statement, era_label, reason_for_change)
                    SELECT id, topic, belief_statement, era_label, coalesce(reason_for_change, '')
                    FROM temporal_beliefs
                    WHERE id NOT IN (SELECT id FROM temporal_beliefs_fts)
                """)
            except Exception:
                pass
            conn.commit()

    # --- Media Source Operations ---

    def get_media_by_hash(self, file_hash: str) -> Optional[MediaSource]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM media_sources WHERE file_hash = ?", (file_hash,)).fetchone()
            if row:
                return MediaSource.from_dict(dict(row))
        return None

    def get_media_by_path_and_size(self, file_path: str, file_size_bytes: int) -> Optional[MediaSource]:
        with self.get_connection() as conn:
            file_name = Path(file_path).name
            row = conn.execute(
                "SELECT * FROM media_sources WHERE (file_path = ? OR file_path LIKE ?) AND file_size_bytes = ?",
                (file_path, f"%/{file_name}", file_size_bytes)
            ).fetchone()
            if row:
                media = MediaSource.from_dict(dict(row))
                if media.file_path != file_path:
                    conn.execute("UPDATE media_sources SET file_path = ? WHERE id = ?", (file_path, media.id))
                    conn.commit()
                    media.file_path = file_path
                return media
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

    def get_transcript_segment_for_range(
        self,
        source_id: str,
        start_seconds: Optional[float] = None,
        end_seconds: Optional[float] = None
    ) -> Optional[TranscriptSegment]:
        """Finds or constructs a transcript segment covering the given time range."""
        if not source_id:
            return None
        with self.get_connection() as conn:
            if start_seconds is not None and end_seconds is not None:
                rows = conn.execute("""
                    SELECT * FROM transcripts
                    WHERE source_id = ?
                      AND end_seconds > ? AND start_seconds < ?
                    ORDER BY start_seconds ASC
                """, (source_id, start_seconds, end_seconds)).fetchall()
                if not rows:
                    rows = conn.execute("""
                        SELECT * FROM transcripts
                        WHERE source_id = ?
                          AND NOT (end_seconds <= ? OR start_seconds >= ?)
                        ORDER BY start_seconds ASC
                    """, (source_id, start_seconds, end_seconds)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM transcripts
                    WHERE source_id = ?
                    ORDER BY start_seconds ASC
                    LIMIT 1
                """, (source_id,)).fetchall()

            if not rows:
                return None

            if len(rows) == 1:
                return TranscriptSegment.from_dict(dict(rows[0]))

            seg_dicts = [dict(r) for r in rows]
            combined_text = " ".join(r['text'] for r in seg_dicts if r.get('text'))
            first = seg_dicts[0]
            last = seg_dicts[-1]
            return TranscriptSegment(
                id=first['id'],
                source_id=source_id,
                start_seconds=first['start_seconds'],
                end_seconds=last['end_seconds'],
                speaker=first.get('speaker', 'Unknown'),
                text=combined_text,
                confidence=first.get('confidence', 1.0),
                language=first.get('language', 'en')
            )

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
        clean_terms = [t.replace('"', '').replace("'", "") for t in query_string.split() if t.isalnum()]
        if not clean_terms:
            return []

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
            # 1. Try exact / AND match for all terms
            match_query = " ".join(clean_terms)
            try:
                rows = conn.execute(sql, (match_query, limit)).fetchall()
                if rows:
                    for r in rows:
                        item_dict = dict(r)
                        rank = abs(float(item_dict.pop('rank_score', 1.0)))
                        results.append((MemoryItem.from_dict(item_dict), rank))
                    return results
            except sqlite3.OperationalError:
                pass

            # 2. If multiple terms and AND returned nothing, filter stopwords and try OR across content terms
            from memory.retrieval import STOP_WORDS
            content_terms = [
                t.lower() for t in clean_terms
                if t.lower() not in STOP_WORDS
                and len(t) > 1
                and not (t.isdigit() and len(t) == 4)
            ]
            if content_terms and len(clean_terms) > 1:
                or_query = " OR ".join(content_terms)
                try:
                    rows = conn.execute(sql, (or_query, limit)).fetchall()
                    if rows:
                        for r in rows:
                            item_dict = dict(r)
                            rank = abs(float(item_dict.pop('rank_score', 1.0)))
                            results.append((MemoryItem.from_dict(item_dict), rank))
                        return results
                except sqlite3.OperationalError:
                    pass

            # 3. Fallback to LIKE if FTS fails
            try:
                like_sql = "SELECT * FROM memories WHERE content LIKE ? LIMIT ?"
                rows = conn.execute(like_sql, (f"%{clean_terms[0]}%", limit)).fetchall()
                for r in rows:
                    results.append((MemoryItem.from_dict(dict(r)), 1.0))
            except Exception:
                pass
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
            # Keep temporal_beliefs_fts synchronized
            conn.execute("DELETE FROM temporal_beliefs_fts WHERE id = ?", (belief.id,))
            conn.execute("""
                INSERT INTO temporal_beliefs_fts (id, topic, belief_statement, era_label, reason_for_change)
                VALUES (?, ?, ?, ?, ?)
            """, (belief.id, belief.topic, belief.belief_statement, belief.era_label, belief.reason_for_change or ""))
            conn.commit()
        return belief.id

    def get_beliefs_for_topic(self, topic: str) -> List[TemporalBelief]:
        """
        Returns all historical and current beliefs on a topic to demonstrate evolution.
        Uses FTS5 keyword matching and topic normalization rather than exact string equality.
        """
        import re
        if not topic or not topic.strip():
            return []

        # 1. Topic normalization: lowercase, strip punctuation, extract alphanumeric terms
        raw_terms = re.findall(r'[a-zA-Z0-9]+', topic.lower())
        if not raw_terms:
            return []

        from memory.retrieval import STOP_WORDS
        extra_stopwords = {"year", "years", "day", "days", "about", "what", "how", "your", "mine"}
        content_terms = [t for t in raw_terms if t not in STOP_WORDS and t not in extra_stopwords and not t.isdigit() and len(t) > 2]
        if not content_terms:
            content_terms = [t for t in raw_terms if not t.isdigit() and len(t) > 1]
        if not content_terms:
            content_terms = raw_terms

        results_map: Dict[str, Tuple[TemporalBelief, float]] = {}

        with self.get_connection() as conn:
            # 2. Try FTS5 keyword search against virtual table
            fts_candidates = []
            if len(content_terms) > 1:
                # 2a. Try AND query on topic or entire row
                topic_and = " AND ".join([f"topic: {t}*" for t in content_terms])
                fts_candidates.append(topic_and)
                general_and = " AND ".join([f"{t}*" for t in content_terms])
                fts_candidates.append(general_and)

            # 2b. Try OR query on topic
            topic_or = " OR ".join([f"topic: {t}*" for t in content_terms])
            fts_candidates.append(topic_or)

            # 2c. General OR
            general_or = " OR ".join([f"{t}*" for t in content_terms])
            fts_candidates.append(general_or)

            sql_fts = """
                SELECT b.*, bm25(temporal_beliefs_fts) as rank_score
                FROM temporal_beliefs_fts fts
                JOIN temporal_beliefs b ON fts.id = b.id
                WHERE temporal_beliefs_fts MATCH ?
                ORDER BY rank_score ASC
            """
            for fts_q in fts_candidates:
                try:
                    rows = conn.execute(sql_fts, (fts_q,)).fetchall()
                    if rows:
                        for r in rows:
                            item_dict = dict(r)
                            rank = abs(float(item_dict.pop('rank_score', 1.0)))
                            belief = TemporalBelief.from_dict(item_dict)
                            if belief.id not in results_map or rank < results_map[belief.id][1]:
                                results_map[belief.id] = (belief, rank)
                        break  # Found high-quality matches, stop cascade
                except (sqlite3.OperationalError, Exception):
                    continue

            # 3. Fallback / supplement with LIKE query for each content term if nothing matched
            if not results_map:
                like_sql = """
                    SELECT * FROM temporal_beliefs
                    WHERE topic LIKE ? OR belief_statement LIKE ?
                """
                for term in content_terms:
                    pattern = f"%{term}%"
                    rows = conn.execute(like_sql, (pattern, pattern)).fetchall()
                    for r in rows:
                        belief = TemporalBelief.from_dict(dict(r))
                        if belief.id not in results_map:
                            results_map[belief.id] = (belief, 10.0)

            # Sort results chronologically to show belief evolution
            sorted_beliefs = sorted(
                [b for b, _ in results_map.values()],
                key=lambda b: (b.approx_year_start if b.approx_year_start is not None else 9999, b.created_at)
            )
            return sorted_beliefs

    def get_beliefs_for_era(self, era_label: str) -> List[TemporalBelief]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_beliefs WHERE era_label LIKE ? ORDER BY approx_year_start ASC", (f"%{era_label}%",)).fetchall()
            return [TemporalBelief.from_dict(dict(r)) for r in rows]

    def list_all_beliefs(self) -> List[TemporalBelief]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM temporal_beliefs ORDER BY approx_year_start ASC, created_at ASC").fetchall()
            return [TemporalBelief.from_dict(dict(r)) for r in rows]

    # --- Chat Session & Multi-Turn Message Operations ---

    def create_session(self, title: str = "New Conversation", session_id: Optional[str] = None) -> str:
        import uuid
        from datetime import datetime
        sid = session_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO chat_sessions (id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (sid, title, now, now))
            conn.commit()
        return sid

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            row = conn.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if row:
                return dict(row)
        return None

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM chat_sessions ORDER BY updated_at DESC").fetchall()
            return [dict(r) for r in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        evidence: Optional[Any] = None,
        audio_url: Optional[str] = None,
        message_id: Optional[str] = None
    ) -> str:
        import uuid
        from datetime import datetime
        mid = message_id or str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        evidence_json = None
        if evidence is not None:
            if isinstance(evidence, str):
                evidence_json = evidence
            else:
                evidence_json = json.dumps(evidence)

        with self.get_connection() as conn:
            # Ensure session exists
            sess = conn.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,)).fetchone()
            if not sess:
                title = content[:60] if role == "user" else "Conversation"
                conn.execute("""
                    INSERT INTO chat_sessions (id, title, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, title, now, now))
            else:
                conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE id = ?", (now, session_id))

            conn.execute("""
                INSERT INTO chat_messages (id, session_id, role, content, evidence_json, audio_url, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (mid, session_id, role, content, evidence_json, audio_url, now))
            conn.commit()
        return mid

    def get_session_messages(self, session_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            sql = """
                SELECT * FROM (
                    SELECT * FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                ) ORDER BY created_at ASC
            """
            rows = conn.execute(sql, (session_id, limit)).fetchall()
            messages = []
            for r in rows:
                d = dict(r)
                if d.get("evidence_json"):
                    try:
                        d["evidence"] = json.loads(d["evidence_json"])
                    except Exception:
                        d["evidence"] = d["evidence_json"]
                else:
                    d["evidence"] = None
                messages.append(d)
            return messages

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
            session_count = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
            message_count = conn.execute("SELECT COUNT(*) FROM chat_messages").fetchone()[0]

            return {
                "media_files": media_count,
                "transcripts": transcript_count,
                "total_memories": memory_count,
                "verified_memories": verified_count,
                "synthetic_fixtures": synthetic_count,
                "unverified_memories": unverified_count,
                "temporal_beliefs": belief_count,
                "memory_corrections": correction_count,
                "chat_sessions": session_count,
                "chat_messages": message_count
            }

