# Personal AI Time Capsule: Engineering Roadmap & Milestones

**Host Platform:** Apple Silicon M4, 24 GB Unified Memory, macOS 27.0  
**Current Milestone:** Milestone 1 (Autonomous Architecture & Foundation Engine) — **COMPLETE**

---

## Milestone 1: Autonomous Foundation & Independent Subsystems (COMPLETED)

All capabilities that can be engineered independently of personal vlog media have been built, integrated, tested, and validated:

- [x] **Phase Zero Environmental Profiling:** M4 Apple Silicon, 24 GB RAM, PyTorch 2.14.1 with MPS active, PyAV 18.1.0, SQLite 3.53 with FTS5, Ollama running `llama3.2:3b` and `gemma2:2b`.
- [x] **Data Persistence Layer:** SQLite 3 schema migrations, `media_sources`, `transcripts`, `memories`, `temporal_beliefs`, `memory_corrections`, FTS5 virtual table, and ChromaDB ONNX vector store.
- [x] **Provenance & Verification Tiers:** Strict segregation of `synthetic_fixture`, `extracted_unverified`, and `personal_verified` records.
- [x] **Chronological Persona Snapshots:** Early 20s (2014-2018), Builder Phase (2019-2022), and Present Day (2023-Present) with distinct worldviews, priorities, and speaking styles.
- [x] **Hybrid Retrieval & Honest Uncertainty:** BM25 + dense semantic vector search via Reciprocal Rank Fusion ($k=60$) with date filtering; honest uncertainty guardrails preventing hallucinations on missing evidence.
- [x] **Expressive Voice Subsystem:** Replaceable `TTSProvider`, `ExpressiveDeliveryController` with 7 emotion presets, speech rate (WPM) scaling, pause modulation, and macOS native 24kHz linear PCM WAV export (`Daniel`).
- [x] **Video Avatar Subsystem Boundary:** `AvatarProvider` abstract interface, background render job manager, and deterministic synthetic MP4 video generation via PyAV.
- [x] **Ingestion Engine:** PyAV container demuxing, SHA-256 deduplication, Whisper/PyAV ASR interface, and resumable execution.
- [x] **Local Web UI & REST API:** FastAPI application, single-page responsive dashboard (Chat, Memory Vault, Belief Timeline, Ingestion Monitor, Voice/Avatar Studio, Backup Manager).
- [x] **Automated Test Suite:** 29 automated unit and integration tests passing with 100% success rate.
- [x] **Disaster Recovery:** Timestamped `.tar.gz` export with SHA-256 manifests and verified clean restoration.

---

## Milestone 2: Personal Media Ingestion & Archive Expansion (COMPLETED)

- [x] **Batch Media Ingestion:** Processed all 22 Google Pixel training vlogs (20.36 GB, ~1.12 hours speech) from `/Volumes/SSD/Photography/_Speak_Project` with cryptographic SHA-256 deduplication.
- [x] **Whisper Transcription with VAD:** Transcribed audio via CTranslate2 `faster-whisper` + Silero VAD at ~16x real-time speed.
- [x] **Automated Memory Extraction:** Extracted 451 granular utterance segments and 113 verified semantic memories (`[PERSONAL_VERIFIED]`) into SQLite WAL and ChromaDB ONNX.
- [x] **Local Offline Audio Cache:** Extracted and persisted 22 high-fidelity 16kHz mono WAV clips (122.9 MB total) locally, removing all runtime dependencies on the external SSD.

---

## Milestone 3: Personal Voice Reproduction & Neural Expressiveness (COMPLETED)

- [x] **Reference Audio Curation:** Curated 15 high-fidelity reference speech clips (4 to 12 seconds each, SNR > 20 dB, diverse cadences) into `data/capsule/voice_references/`.
- [x] **Empirical Acoustic Profiling:** Measured fundamental pitch ($F_0$ median: 136.8 Hz, mean: 147.4 Hz, range: 119.4–188.2 Hz) and speaking rate (measured 121.3 WPM).
- [x] **Neural Voice Synthesis:** Deployed Kokoro-82M on Apple Silicon M4 with calibrated personal speaker tensor (`sergiu_voice.pt`), producing expressive 24kHz audio in <1 second.
- [x] **Autobiographical Model Fine-Tuning:** Generated 113 autobiographical training QA pairs and created custom Ollama model `sergiu-time-capsule` based on `llama3.2:3b`.
- [x] **Hardware-in-the-Loop Validation:** Verified grounded memory retrieval, citations, expressive speech synthesis, and honest uncertainty negative controls with 100% test pass rate.

---

## Milestone 4: Audio-Driven Video Avatar (Visual Likeness)

*Triggered when high-resolution reference portraits and video footage are provided.*

- [ ] **Reference Face Asset Extraction:** Select 3 high-resolution neutral and expressive keyframe portraits from the vlog footage.
- [ ] **LivePortrait Engine Deployment:** Install KwaiVGI/LivePortrait in `providers/avatar/` running on Apple Silicon Metal MPS.
- [ ] **Landmark Driving & Lip Sync:** Drive facial keypoints directly from synthesized audio pitch/spectrogram features.
- [ ] **Real-Time WebRTC / Stream Output:** Stream rendered avatar video frames (25 FPS, 512x512) directly into the UI conversation screen.

---

## Milestone 5: Multi-Decade Preservation & Identity Continuity

- [ ] **Automated Reindexing:** Controlled embedding migration script allowing instant re-indexing to newer vector models without touching canonical text archives.
- [ ] **Cold-Storage Replication:** Encrypted, off-site backup synchronizer (e.g. to cold disk or private local NAS).
- [ ] **Decennial Persona Snapshotting:** Framework for automatically freezing snapshots at milestones (Age 35, 40, 50, etc.).
