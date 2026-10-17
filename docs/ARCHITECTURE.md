# Personal AI Time Capsule: Technical Architecture & System Design

**Project:** Project 30 (Autonomous Development of My Personal AI Time Capsule)  
**Host Environment:** Apple Silicon M4 Mac (macOS 27.0, 24 GB Unified Memory)  
**Target Directory:** `/Users/Sergiu/AI`  
**Security Invariant:** 100% Local Inference · Loopback Only (`127.0.0.1`) · Zero Cloud Telemetry

---

## 1. High-Level Architecture Topology

```
                                   ┌────────────────────────────────────────┐
                                   │           Local Web Browser            │
                                   │      http://127.0.0.1:7860 (UI)        │
                                   └──────────────────┬─────────────────────┘
                                                      │ (HTTP / JSON / Media)
                                                      ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ FASTAPI APPLICATION SERVER (`app/main.py`)                                                       │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Conversation Engine (`core/conversation.py`)                                             │   │
│   │ - Chronological Era Resolution (`PersonalityProfile`)                                    │   │
│   │ - Grounded Prompt Synthesis (Source citations, timestamps, confidence)                   │   │
│   │ - Honest Uncertainty Guardrails (Zero hallucination on absent evidence)                  │   │
│   │ - Expressive Delivery Instruction Generator                                              │   │
│   └──────────────────────────┬─────────────────────────────┬─────────────────────────────────┘   │
│                              │                             │                                     │
│                              ▼                             ▼                                     │
│   ┌────────────────────────────────────────┐   ┌─────────────────────────────────────────────┐   │
│   │ Hybrid Retrieval (`memory/retrieval.py`)│   │ LLM Provider Interface (`providers/llm/`)   │   │
│   │ - BM25 Full-Text Search (SQLite FTS5)  │   │ - Primary: Ollama Local (llama3.2:3b)       │   │
│   │ - Dense Vector Search (Chroma ONNX)    │   │ - Fallback: Ollama (gemma2:2b)              │   │
│   │ - Reciprocal Rank Fusion (RRF k=60)    │   │ - Testing: Mock Deterministic Provider      │   │
│   └────────────────────┬───────────────────┘   └─────────────────────────────────────────────┘   │
│                        │                                                                         │
│         ┌──────────────┴──────────────┐                                                          │
│         ▼                             ▼                                                          │
│   ┌──────────────────────────┐  ┌──────────────────────────┐                                     │
│   │ SQLite 3 Archive DB      │  │ ChromaDB Vector Store    │                                     │
│   │ (`memory/database.py`)   │  │ (`memory/vector_store.py`)│                                     │
│   │ - WAL Mode + Transactions│  │ - ONNX all-MiniLM-L6-v2  │                                     │
│   │ - Media & Transcripts    │  │ - 384-dimensional cosine │                                     │
│   │ - Memories & Audit Logs  │  │ - Disposable & Reindexable│                                     │
│   │ - Temporal Belief Graph  │  └──────────────────────────┘                                     │
│   └──────────────────────────┘                                                                   │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Voice & Audio Subsystem (`core/voice.py`, `providers/tts/`)                              │   │
│   │ - `ExpressiveDeliveryController`: Emotion presets, speaking rate (WPM), pause cadence     │   │
│   │ - `MacOSNativeTTSProvider`: 24,000 Hz 16-bit PCM Linear WAV export (`/usr/bin/say`)      │   │
│   │ - `NeuralTTSProvider`: Modular interface for Kokoro-82M / F5-TTS / Qwen3-TTS             │   │
│   │ - `VoiceProfile`: Personal reference conditioning & mannerism configuration              │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Future Video Avatar Subsystem (`providers/avatar/`)                                      │   │
│   │ - `AvatarProvider` Abstract Boundary & Render Job Manager                                │   │
│   │ - `LivePortraitAvatarProvider`: Audio feature driving -> Video frame rendering           │   │
│   │ - Deterministic synthetic integration validation (MP4 container with PyAV)               │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Media Ingestion Pipeline (`ingestion/pipeline.py`)                                       │   │
│   │ - PyAV Audio/Video demuxing, duration, and stream metadata extraction                    │   │
│   │ - SHA-256 cryptographic deduplication                                                    │   │
│   │ - Resumable batch execution                                                              │   │
│   │ - ASR Provider: PyAV + Whisper timestamped segment extraction                            │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────────────────────────────┐   │
│   │ Backup & Disaster Recovery (`memory/backup.py`)                                          │   │
│   │ - SQLite WAL checkpointing + Staging directory                                           │   │
│   │ - Timestamped `.tar.gz` export with SHA-256 cryptographic manifest                        │   │
│   │ - Safe restore with automatic pre-restore safety copy & WAL/SHM reset                    │   │
│   │ - Portable JSON / Markdown archive export                                                │   │
│   └──────────────────────────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Boundaries & Interfaces

| Component | Interface / Base Class | Implementations | Key Responsibilities |
|---|---|---|---|
| **Language Model** | `LLMProvider` | `OllamaLLMProvider`, `MockLLMProvider` | Local prompt generation, system constraints, token limits. |
| **Speech-to-Text** | `ASRProvider` | `PyAVWhisperASRProvider`, `MockASRProvider` | Audio demuxing, resampling to 16kHz mono, timestamped segments. |
| **Speech Synthesis** | `TTSProvider` | `MacOSNativeTTSProvider`, `NeuralTTSProvider` | Expressive acoustic synthesis, rate/pitch modulation, 24kHz WAV export. |
| **Video Avatar** | `AvatarProvider` | `LivePortraitAvatarProvider` | Asynchronous render job management, audio-to-video container generation. |
| **Memory Storage** | `DatabaseManager` | SQLite 3.53+ with WAL mode | ACID persistence, FTS5 virtual tables, temporal belief evolution, audit logs. |
| **Vector Store** | `VectorStore` | ChromaDB PersistentClient | Dense semantic embeddings (`all-MiniLM-L6-v2`), cosine similarity, batch reindexing. |
| **Retrieval** | `HybridRetrievalProvider` | Reciprocal Rank Fusion ($k=60$) | Fusing keyword and dense vector ranks, era date filtering, evidence thresholding. |
| **Personality** | `PersonalityProfile` | Default chronological snapshots | Persona snapshots (Early 20s, Builder Phase, Present Day), worldview prompts. |
| **Disaster Recovery**| `BackupManager` | Tarfile + SHA-256 Manifest | Full archive snapshotting, manifest verification, zero-loss restoration. |

---

## 3. Data Flow Pathways

### 3.1 Conversational Query Flow
1. **User input:** User submits query via Web UI or REST API (`POST /api/chat`) with selected chronological era (e.g. `early_20s` or `present`).
2. **Era resolution:** `ConversationEngine` queries `PersonalityProfile` for the target era's worldview, style notes, and default delivery parameters.
3. **Hybrid Retrieval:** `HybridRetrievalProvider` executes simultaneous BM25 keyword matching (`memories_fts`) and dense vector similarity (`time_capsule_memories`). Results are fused via Reciprocal Rank Fusion ($k=60$) with date filtering.
4. **Evidence Thresholding:** If top score $< 0.005$ or no documents match, `has_sufficient_evidence` is set to `False`.
5. **Prompt Assembly:** System prompt injects:
   - Chronological persona constraints.
   - Relevant temporal beliefs showing historical evolution on the topic.
   - Grounded memory citations with timestamps and verification tiers.
   - Mandatory honest uncertainty directive if evidence is absent.
6. **LLM Inference:** `OllamaLLMProvider` generates response via local loopback API using `llama3.2:3b`.
7. **Prosody Adaptation:** `ExpressiveDeliveryController` calculates speaking rate (WPM), emotion category, pitch shift, and pause multipliers.
8. **Optional Voice Generation:** `TTSProvider` generates a 24kHz WAV file served via `/api/audio/{filename}` for browser playback.

### 3.2 Media Ingestion Flow (When Vlogs Arrive)
1. **Discovery:** Target directory is scanned for supported extensions (`.mp4`, `.mov`, `.mkv`, `.wav`).
2. **Deduplication:** SHA-256 hash is computed. If matching hash exists in `media_sources` with status `indexed`, file is skipped.
3. **PyAV Stream Demuxing:** Container metadata (duration, codecs, sample rate, channels) extracted without external FFmpeg CLI.
4. **ASR Transcription:** Audio stream converted to 16kHz mono PCM and transcribed into timestamped segments (`start_seconds`, `end_seconds`, `text`, `confidence`).
5. **Memory Extraction:** Atomic memory statements are categorized, tagged, and persisted in SQLite `memories` and indexed in `memories_fts`.
6. **Vector Indexing:** Dense embeddings generated and indexed in ChromaDB `time_capsule_memories`.
7. **Status Finalization:** Media record marked `indexed`.

---

## 4. Hardware Sizing & Memory Map (Apple Silicon M4)

| Subsystem | Resident Model | Memory Footprint (RAM) | Execution Engine |
|---|---|---|---|
| **Operating System** | macOS 27.0 & Display Server | ~4.8 GB | Darwin / Metal |
| **LLM Inference** | `llama3.2:3b` (Q4_K_M) | ~2.2 GB | Ollama (Metal GPU / MPS) |
| **Embedding Model** | `all-MiniLM-L6-v2` | ~0.1 GB | ChromaDB (ONNX CPU/MPS) |
| **Speech Synthesis** | macOS Native Speech Engine | ~0.2 GB | System Audio Services |
| **Web Server & DB** | FastAPI + SQLite + Python 3.11 | ~0.3 GB | CPython 3.11 |
| **Future Neural TTS**| `Kokoro-82M` (or `F5-TTS`) | ~0.4 GB (or 1.8 GB) | PyTorch MPS |
| **Future Avatar** | `LivePortrait` | ~3.2 GB | PyTorch MPS (Metal) |
| **Total AI Runtime**| **Interactive Multi-Modal** | **~11.2 GB** | **46.7% of 24 GB Pool** |
| **Remaining Headroom**| **Free / Unified Buffer Cache**| **~12.8 GB** | **Zero SSD Swapping** |
