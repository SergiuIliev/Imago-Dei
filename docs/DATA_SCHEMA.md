# Personal AI Time Capsule: Data Schema & Storage Architecture

**Database Engine:** SQLite 3.53+ with WAL Mode & FTS5  
**Vector Engine:** ChromaDB Persistent Client (ONNX `all-MiniLM-L6-v2`, 384-dimensional cosine distance)  
**Schema Version:** 1  
**Default Storage Location:** `data/capsule/archive.db`

---

## 1. Provenance & Verification Tiers

To prevent hallucination, simulation traps, or confusing test fixtures with real life experiences, every memory entity is strictly tagged with one of three immutable verification levels:

| Status Tier | Description | Source Invariant |
|---|---|---|
| `synthetic_fixture` | Synthetic test fixture generated specifically for pipeline testing. | **MUST NEVER** be presented as real memories. Automated tests verify isolation. |
| `extracted_unverified` | Extracted automatically from uploaded personal audio/video transcripts. | Machine-generated summary/observation. Awaits personal human review. |
| `personal_verified` | Directly authenticated, written, or edited by the user. | Ground truth personal memory. Takes precedence in conflicting persona queries. |

---

## 2. Relational Schemas (SQLite)

### Table: `schema_migrations`
Tracks database schema versioning for zero-downtime upgrades.
```sql
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Table: `media_sources`
Catalog of ingested audio/video files with cryptographic deduplication.
```sql
CREATE TABLE media_sources (
    id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,      -- SHA-256 Checksum for deduplication
    file_size_bytes INTEGER NOT NULL,
    mime_type TEXT,
    duration_seconds REAL DEFAULT 0.0,
    recorded_at TEXT,                    -- ISO-8601 Timestamp of recording
    imported_at TEXT NOT NULL,           -- ISO-8601 Timestamp of ingestion
    status TEXT NOT NULL,                -- discovered, processing, transcribed, indexed, failed
    error_message TEXT,
    metadata TEXT                        -- JSON dictionary of codec, bitrate, resolution
);
CREATE INDEX idx_media_hash ON media_sources(file_hash);
CREATE INDEX idx_media_status ON media_sources(status);
```

### Table: `transcripts`
Timestamped speech segments produced by ASR (Whisper / faster-whisper / PyAV).
```sql
CREATE TABLE transcripts (
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
CREATE INDEX idx_transcripts_source ON transcripts(source_id);
CREATE INDEX idx_transcripts_timestamps ON transcripts(source_id, start_seconds);
```

### Table: `memories`
Atomic units of remembered events, reflections, and life experiences.
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,               -- The complete narrative memory
    summary TEXT,                        -- Concise synopsis
    category TEXT DEFAULT 'reflection',  -- event, belief, reflection, relationship, career, lifestyle
    date_occurred TEXT,                  -- YYYY-MM-DD or life era
    source_id TEXT,                      -- Link to media_sources(id) if extracted
    start_seconds REAL,                  -- Timestamp offset in source recording
    end_seconds REAL,
    verification_status TEXT NOT NULL,   -- synthetic_fixture, extracted_unverified, personal_verified
    confidence REAL DEFAULT 1.0,
    emotional_tone TEXT DEFAULT 'neutral', -- neutral, reflective, lively, curious, serious, humorous, melancholic
    tags TEXT DEFAULT '[]',              -- JSON array of strings
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(source_id) REFERENCES media_sources(id) ON DELETE SET NULL
);
CREATE INDEX idx_memories_status ON memories(verification_status);
CREATE INDEX idx_memories_date ON memories(date_occurred);
CREATE INDEX idx_memories_category ON memories(category);
```

### Virtual Table: `memories_fts` (SQLite FTS5)
Full-text search index utilizing BM25 ranking for instantaneous keyword discovery.
```sql
CREATE VIRTUAL TABLE memories_fts USING fts5(
    id UNINDEXED,
    content,
    summary,
    category,
    tags
);
```

### Table: `temporal_beliefs`
Preserves historical beliefs, principles, and preferences across distinct life stages.
When a belief changes over time, earlier beliefs are **NEVER overwritten or erased**; instead, they remain linked via `superseded_by_id`.
```sql
CREATE TABLE temporal_beliefs (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,                 -- e.g. "Work-Life Balance", "Technology Philosophy"
    belief_statement TEXT NOT NULL,      -- What was believed
    era_label TEXT NOT NULL,             -- e.g. "Early 20s", "2018-2020", "Present Day"
    approx_year_start INTEGER,
    approx_year_end INTEGER,
    evidence_memory_ids TEXT DEFAULT '[]', -- JSON array of memory IDs justifying this belief
    superseded_by_id TEXT,               -- Foreign key to newer belief if evolved
    supersedes_id TEXT,                  -- Foreign key to previous belief
    active INTEGER NOT NULL DEFAULT 1,   -- 1 if active in this era, 0 if superseded
    reason_for_change TEXT,              -- Why the perspective evolved
    created_at TEXT NOT NULL
);
CREATE INDEX idx_beliefs_topic ON temporal_beliefs(topic);
CREATE INDEX idx_beliefs_era ON temporal_beliefs(era_label);
```

### Table: `memory_corrections`
Immutable audit log recording every user edit, correction, or clarification.
```sql
CREATE TABLE memory_corrections (
    id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    original_content TEXT NOT NULL,
    corrected_content TEXT NOT NULL,
    correction_reason TEXT NOT NULL,
    corrected_by TEXT NOT NULL DEFAULT 'user',
    created_at TEXT NOT NULL,
    FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE
);
CREATE INDEX idx_corrections_memory ON memory_corrections(memory_id);
```

---

## 3. Dense Vector Embeddings (ChromaDB)

- **Collection:** `time_capsule_memories`
- **Model:** `all-MiniLM-L6-v2` (ONNX local runtime)
- **Vector Dimension:** 384
- **Distance Metric:** Cosine Distance ($d = 1 - \frac{u \cdot v}{\|u\| \|v\|}$)
- **Indexed Document Text:** `"{summary}\n{content}"`
- **Metadata Fields:** `category`, `verification_status`, `date_occurred`, `emotional_tone`, `source_id`
- **Decoupling Guarantee:** The vector store is completely disposable. Replacing the embedding model or dimension only requires executing `vector_store.reindex_all(db.list_memories())` without impacting the canonical SQLite archive.

---

## 4. Hybrid Retrieval Architecture (Reciprocal Rank Fusion)

Queries are simultaneously dispatched to both the BM25 full-text engine and the dense vector engine. Results are fused using Reciprocal Rank Fusion ($k=60$):
$$RRF\_score(d) = \frac{w_{FTS}}{60 + rank_{FTS}(d)} + \frac{w_{Vector}}{60 + rank_{Vector}(d)}$$

Evidence thresholds: If the maximum candidate score is below $0.05$ (or if no results match), the system flags **insufficient evidence**, preventing hallucinated or fabricated answers.
