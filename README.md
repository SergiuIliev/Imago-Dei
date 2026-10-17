# Personal AI Time Capsule (Project 30)

An autonomous, private, evidence-grounded AI representation of Sergiu preserving memories, values, reasoning, chronological identity evolution, and voice, with an extensible path to audio-driven video avatars.

Built strictly for **local-only execution** on Apple Silicon with zero external cloud telemetry, cryptographic provenance, and modular model replacements.

---

## 1. System Overview & Key Capabilities

- **Chronological Persona Snapshots:** Interact with distinct historical life stages (Early 20s Formative Years, Mid/Late 20s Builder Phase, Present Day) preserving genuine evolution in beliefs and reasoning without flattening past contradictions.
- **Evidence-Grounded Retrieval:** Hybrid search fusing SQLite FTS5 (BM25 keyword search) and ChromaDB ONNX (`all-MiniLM-L6-v2` dense vectors) via Reciprocal Rank Fusion ($k=60$). Every answer cites specific timestamps and source IDs.
- **Honest Uncertainty:** Strict guardrails prevent hallucination. When the personal archive lacks direct evidence, the system explicitly states its uncertainty rather than fabricating experiences.
- **Verification Hierarchy:** Immutable provenance tiers distinguishing synthetic test fixtures (`[SYNTHETIC_FIXTURE]`), machine-extracted media statements (`[EXTRACTED_UNVERIFIED]`), and user-confirmed memories (`[PERSONAL_VERIFIED]`).
- **Expressive Voice Subsystem:** Decoupled acoustic controller modulating emotional states (reflective, lively, curious, serious, humorous, melancholic), speaking rate (WPM), pitch, and thoughtful pause timing. Includes a verified 24,000 Hz 16-bit PCM WAV exporter on macOS native speech (`Daniel`) and a neural cloning interface.
- **Audio-Driven Video Avatar Pipeline:** Modular integration boundary for LivePortrait with asynchronous job management, landmark driving, and PyAV MP4 container synthesis.
- **Disaster Recovery & Portability:** Timestamped `.tar.gz` archive snapshots with cryptographic SHA-256 manifests, bit-exact verification, and human-readable JSON exports.

---

## 2. Host Environment & Sizing

- **Machine:** Apple Silicon M4 Mac (Mac16,12), 10-core CPU, 10-core GPU (Metal 4, MPS active).
- **Operating System:** macOS 27.0 (Darwin arm64).
- **Unified Memory:** 24 GB LPDDR5X-7500 (120 GB/s bandwidth).
- **Local AI Runtimes:**
  - **LLM Engine:** Ollama running `llama3.2:3b` (Q4_K_M, 131k context) and `gemma2:2b`.
  - **Vector Engine:** ChromaDB Persistent Client (384-dim ONNX `all-MiniLM-L6-v2`).
  - **Audio/Video Engine:** PyAV (FFmpeg 18.1.0 C-bindings), CTranslate2, and macOS Speech Services.
- **Memory Consumption:** Interactive multi-modal runtime uses ~11.2 GB (46.7% of total pool), leaving >12 GB headroom for system cache with zero SSD swap.

---

## 3. Quickstart & Operational Commands

### 3.1 Launching the Web Application
```bash
/Users/Sergiu/AI/scripts/launch.sh
```
- Starts the FastAPI application at **`http://127.0.0.1:7860`**.
- Binds strictly to loopback (`127.0.0.1`) with zero LAN or public ingress.
- If the archive is empty, automatically seeds baseline test fixtures.

### 3.2 Running the Full Automated Test Suite
```bash
/Users/Sergiu/AI/scripts/run_tests.sh
```
- Executes all 29 automated unit and integration tests across 7 workstreams.
- Verifies database migrations, FTS5 search, vector embeddings, deduplication, timestamped ASR, temporal beliefs, honest uncertainty, expressive TTS WAV export, avatar job queues, and privacy invariants.

### 3.3 Creating an Archive Backup
```bash
/Users/Sergiu/AI/scripts/backup.sh manual_snapshot
```
- Creates a timestamped `.tar.gz` archive in `/Users/Sergiu/AI/exports/` sealed with a cryptographic SHA-256 `manifest.json`.

### 3.4 Restoring an Archive Backup
```bash
/Users/Sergiu/AI/scripts/restore.sh /Users/Sergiu/AI/exports/capsule_backup_manual_snapshot_YYYYMMDD_HHMMSS.tar.gz
```
- Verifies SHA-256 checksums of all internal files against the manifest before restoring.
- Automatically saves a safety copy of the existing database and cleans WAL/SHM sidecars.

### 3.5 Ingesting Media (When Vlog Path is Ready)
```bash
/Users/Sergiu/AI/scripts/ingest_media.sh /path/to/personal/vlogs
```
- Discovers supported media files (`.mp4`, `.mov`, `.mkv`, `.wav`).
- Extracts PyAV container metadata and performs SHA-256 deduplication.
- Extracts speech segments with timestamps and indexes memories into SQLite and ChromaDB.

---

## 4. Repository Structure

```
/Users/Sergiu/AI/
├── app/
│   └── main.py                     # FastAPI REST server & UI controller
├── core/
│   ├── config.py                   # Modular application configuration
│   ├── conversation.py             # Conversation Engine & grounded prompt synthesis
│   ├── models.py                   # Pydantic data models & verification enums
│   ├── personality.py              # Chronological persona snapshots & worldview prompts
│   └── voice.py                    # ExpressiveDeliveryController & VoiceProfile
├── providers/
│   ├── llm/
│   │   ├── base.py                 # Abstract LLMProvider interface
│   │   ├── ollama_provider.py      # Ollama local inference (llama3.2:3b / gemma2:2b)
│   │   └── mock_provider.py        # Deterministic offline test provider
│   ├── asr/
│   │   ├── base.py                 # Abstract ASRProvider interface
│   │   ├── whisper_provider.py     # PyAV + Whisper transcription
│   │   └── mock_provider.py        # Deterministic ASR test provider
│   ├── tts/
│   │   ├── base.py                 # Abstract TTSProvider interface
│   │   ├── macos_tts.py            # macOS native 24kHz WAV speech engine
│   │   └── neural_tts.py           # Neural reference conditioning interface
│   └── avatar/
│       ├── base.py                 # Abstract AvatarProvider interface & job dataclass
│       └── liveportrait_provider.py# Audio-driven avatar queue & video container generator
├── memory/
│   ├── backup.py                   # Backup, export, SHA-256 manifest & restore manager
│   ├── database.py                 # SQLite 3 WAL persistence & FTS5 full-text index
│   ├── retrieval.py                # Hybrid Reciprocal Rank Fusion retrieval
│   └── vector_store.py             # ChromaDB persistent ONNX embeddings
├── ingestion/
│   └── pipeline.py                 # Media discovery, deduplication, and indexing pipeline
├── ui/
│   ├── static/
│   │   ├── app.js                  # Frontend client application logic
│   │   └── styles.css              # Responsive dark theme (Apple HIG inspired)
│   └── templates/
│       └── index.html              # Single-page dashboard template
├── scripts/
│   ├── launch.sh                   # Application launcher
│   ├── run_tests.sh                # Test suite runner
│   ├── seed_fixtures.py            # Baseline fixture seeder
│   ├── backup.sh                   # CLI backup generator
│   ├── restore.sh                  # CLI archive restorer
│   └── ingest_media.sh             # CLI batch media ingestion tool
├── tests/
│   ├── fixtures/
│   │   ├── synthetic_fixtures.json # Tagged synthetic memories & temporal beliefs
│   │   └── synthetic_sample.wav    # 24kHz synthetic audio fixture
│   ├── test_api_endpoints.py       # FastAPI REST & HTML route tests
│   ├── test_backup_restore.py      # Backup, manifest, and restore tests
│   ├── test_ingestion_pipeline.py  # Deduplication, PyAV, and ASR tests
│   ├── test_persistence.py         # SQLite WAL, FTS5, and audit log tests
│   ├── test_privacy_audit.py       # Loopback and zero-telemetry tests
│   ├── test_retrieval_and_beliefs.py# Hybrid search, honest uncertainty, belief evolution tests
│   └── test_voice_and_avatar.py    # Expressive prosody, WAV synthesis, and avatar tests
├── docs/
│   ├── ARCHITECTURE.md             # Detailed component boundaries and data flows
│   ├── RESEARCH.md                 # State-of-the-art model research & benchmarks
│   ├── MODEL_REGISTRY.md           # Model versions, licenses, storage, and status
│   ├── ROADMAP.md                  # Milestone status and future upgrade plan
│   ├── VOICE_DESIGN.md             # Voice conditioning, prosody, and cloning workflow
│   ├── PRIVACY.md                  # Security assumptions and local invariants
│   ├── DATA_SCHEMA.md              # Relational schemas, FTS5, and vector indexing
│   ├── TEST_REPORT.md              # Test execution log verifying all 15 acceptance criteria
│   └── TASK_BOARD.md               # Multi-agent workstream board and deliverable tracking
├── data/capsule/                   # Canonical database & vector storage (git-ignored)
└── exports/                        # Compressed backup snapshots (git-ignored)
```

---

## 5. Technical Documentation Index

- **[System Architecture](docs/ARCHITECTURE.md):** Deep breakdown of subsystem topology, data flows, and hardware roofline analysis.
- **[Model Research](docs/RESEARCH.md):** Comprehensive Apple Silicon M4 research on LLMs, Whisper ASR, dense embeddings, neural TTS (Kokoro/F5-TTS), and LivePortrait video avatars.
- **[Model Registry](docs/MODEL_REGISTRY.md):** Detailed catalog with versions, code vs weight licenses, memory sizes, and health checks.
- **[Data Schema](docs/DATA_SCHEMA.md):** Complete SQLite schemas, FTS5 virtual tables, temporal belief linking, and verification rules.
- **[Voice Design](docs/VOICE_DESIGN.md):** Prosody modulation matrix, expressive delivery controllers, and reference conditioning protocol.
- **[Privacy Architecture](docs/PRIVACY.md):** Threat model, loopback invariants, cryptographic manifests, and network audit specifications.
- **[Test Execution Report](docs/TEST_REPORT.md):** Full log of test execution and 15-point acceptance criteria verification.
- **[Roadmap & Next Steps](docs/ROADMAP.md):** Milestones for personal media ingestion, neural voice cloning, and avatar rendering.
- **[Task Board](docs/TASK_BOARD.md):** Status tracking across all 7 autonomous engineering workstreams.

---

## 6. What Happens When Personal Vlogs Are Supplied

The system currently operates with verified synthetic test fixtures. Once Sergiu provides the path to his vlog directory:
1. Configure the directory path via UI or run `/Users/Sergiu/AI/scripts/ingest_media.sh /path/to/vlogs`.
2. PyAV demuxes video containers into 16kHz audio, and Whisper transcribes speech with microsecond timestamps.
3. Extracted memories appear in the **Memory Vault** with badge `[EXTRACTED_UNVERIFIED]`.
4. High-SNR audio clips are registered into `VoiceProfile` for neural zero-shot voice cloning.
5. Reference portrait keyframes are registered for `LivePortrait` avatar facial driving.
6. User reviews, clarifies, or verifies memories in the Web UI, promoting them to `[PERSONAL_VERIFIED]`.
