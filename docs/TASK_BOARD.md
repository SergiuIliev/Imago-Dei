# Engineering Task Board: Personal AI Time Capsule (Project 30)

**Host Platform:** Apple Silicon M4 (macOS 27.0, 24 GB Unified Memory, Metal MPS)  
**Root Working Directory:** `~/AI`  
**Current Status:** **DONE (MILESTONE 1 VERIFIED)**  

---

## Workstream & Subagent Assignments

| Agent | Role | Domain / Responsibilities | Status | Assigned Deliverables | Key Dependencies |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Agent 1** | Technical Lead & Integration Owner | Overall architecture, directory structure, module interfaces, task board, system integration, end-to-end acceptance | **DONE** | `README.md`, `docs/ARCHITECTURE.md`, `docs/ROADMAP.md`, `docs/TASK_BOARD.md`, `scripts/*` | All Workstreams |
| **Agent 2** | Local AI & Model Researcher | State-of-the-art model research, Apple Silicon MPS benchmarks, registry of vetted models, licensing & memory footprints | **DONE** | `docs/RESEARCH.md`, `docs/MODEL_REGISTRY.md` | None (Async) |
| **Agent 3** | Personal Memory & Retrieval Engine | SQLite + FTS5 persistence, ChromaDB embeddings, schema migrations, belief evolution, provenance & evidence tracking, backup/restore | **DONE** | `memory/`, `data/capsule/`, `docs/DATA_SCHEMA.md` | Schema specs |
| **Agent 4** | Personality & Conversation Engine | Persona snapshots (chronological eras), grounded prompt synthesis, honest uncertainty logic, source excerpt citation formatting | **DONE** | `core/personality.py`, `core/conversation.py`, `providers/llm/` | Memory Retrieval API |
| **Agent 5** | Expressive Voice & Audio Engineering | TTSProvider interface, macOS native `say` provider, neural TTS integration path (Kokoro/Qwen/XTTS), expressive delivery controller, voice profile | **DONE** | `core/voice.py`, `providers/tts/`, `docs/VOICE_DESIGN.md` | Audio drivers |
| **Agent 6** | User Interface & Product Experience | Local Web interface (FastAPI + HTML5/JS dashboard), chat screen, era selector, memory inspector/editor, archive status, audio player, avatar stub | **DONE** | `ui/`, `app/` | Conversation & Memory APIs |
| **Agent 7** | Quality Assurance, Privacy & Reliability | Comprehensive test suite, privacy network audit (local-only), synthetic fixture pipeline, backup/restore verification | **DONE** | `tests/`, `tests/fixtures/`, `docs/PRIVACY.md`, `docs/TEST_REPORT.md` | All Modules |

---

## Detailed Task Checklist

### Agent 1: Lead Architecture & Integration
- [x] Phase Zero inspection (M4 Apple Silicon, 24GB RAM, Ollama running llama3.2 & gemma2, PyAV, PyTorch MPS).
- [x] Establish modular repository structure (`app/`, `core/`, `providers/`, `memory/`, `ingestion/`, `ui/`, `tests/`, `docs/`, `scripts/`).
- [x] Initialize Task Board and Component Interfaces.
- [x] Implement command line lifecycle scripts (`scripts/launch.sh`, `scripts/run_tests.sh`, `scripts/backup.sh`, `scripts/restore.sh`, `scripts/ingest_media.sh`).
- [x] End-to-end integration and verification against all 15 acceptance criteria (29/29 tests passing).

### Agent 2: Model Research & Registry
- [x] Launched subagent research task.
- [x] Complete `docs/RESEARCH.md` (35.7 KB) covering LLM, ASR, Embeddings, TTS, and Avatar candidates on Apple M4.
- [x] Complete `docs/MODEL_REGISTRY.md` (25.6 KB) with explicit licenses, sizes, memory bounds, and current local status.

### Agent 3: Personal Memory & Retrieval Engine
- [x] Implement data models (`core/models.py`) with Pydantic / dataclasses (MediaSource, TranscriptSegment, MemoryItem, TemporalBelief, MemoryCorrection).
- [x] Implement SQLite database with WAL mode and FTS5 full-text indexing (`memory/database.py`).
- [x] Implement Vector embedding store (`memory/vector_store.py`) using local ChromaDB ONNX (`all-MiniLM-L6-v2`).
- [x] Implement hybrid retrieval engine (`memory/retrieval.py`) combining BM25 keyword search and cosine semantic similarity via RRF ($k=60$).
- [x] Implement backup, export, and restoration with SHA-256 integrity verification (`memory/backup.py`).
- [x] Generate `docs/DATA_SCHEMA.md`.

### Agent 4: Personality & Conversation Engine
- [x] Implement `PersonalityProfile` (`core/personality.py`) with customizable traits, core values, and historical snapshots (Early 20s, Builder Phase, Present Day).
- [x] Implement `LLMProvider` abstraction (`providers/llm/base.py`) with Ollama provider (`providers/llm/ollama_provider.py`) and mock test provider (`providers/llm/mock_provider.py`).
- [x] Implement `ConversationEngine` (`core/conversation.py`) enforcing:
  - Temporal belief alignment for chosen era/age.
  - Strict grounding in retrieved evidence with exact source citations.
  - Honest uncertainty response when archive lacks evidence.
  - Simulated perspective labeling when extrapolating.

### Agent 5: Expressive Voice Subsystem
- [x] Implement `TTSProvider` base interface (`providers/tts/base.py`).
- [x] Implement `MacOSNativeTTSProvider` (`providers/tts/macos_tts.py`) utilizing macOS speech synthesis with WAV export and rate/pitch modulation.
- [x] Implement `ExpressiveDeliveryController` (`core/voice.py`) managing prosody, emotional state, intensity, speech rate, and pauses.
- [x] Implement `VoiceProfile` structure and personal reference voice conditioning workflow (`providers/tts/neural_tts.py`).
- [x] Generate `docs/VOICE_DESIGN.md`.

### Agent 6: User Interface & Product Experience
- [x] Implement FastAPI web server (`app/main.py`).
- [x] Build responsive modern UI (`ui/templates/index.html`, `ui/static/app.js`, `ui/static/styles.css`):
  - Chat interface with chronological era selector.
  - Source evidence drawer showing timestamps and verification status badges.
  - Memory manager & correction workflow.
  - Media ingestion & archive status monitor.
  - Expressive audio player & voice controls.
  - Video avatar display container.

### Agent 7: Quality Assurance, Privacy & Verification
- [x] Build synthetic media ingestion test fixtures (`tests/fixtures/synthetic_fixtures.json`, synthetic WAV).
- [x] Build automated test suite covering all 15 acceptance criteria (29 passing tests):
  - Test SQLite persistence & migrations.
  - Test deduplication & interrupted job handling.
  - Test temporal belief evolution & conflicting views over time.
  - Test retrieval accuracy & missing evidence uncertainty.
  - Test provider swapping via configuration.
  - Test expressive TTS synthesis and fallback.
  - Test backup export and restoration integrity.
  - Test privacy audit (zero outbound cloud requests).
- [x] Generate `docs/PRIVACY.md` and `docs/TEST_REPORT.md`.
