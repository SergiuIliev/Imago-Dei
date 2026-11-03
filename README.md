# Home Deus Imago Dei

> *"In Homo Deus, 'gods' are merely super-powered biological or digital entities who possess immense strength, intelligence, and lifespan, but not necessarily superior morality, love, or holiness.*
> 
> *This tool ultimately aims to empower people to create an image of their own cognition and leverage it to assist them in better understanding themselves."*

[![Release](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Local-First](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](docs/PRIVACY.md)
[![Double Audit](https://img.shields.io/badge/Security-Double%20Audit%20Verified-brightgreen.svg)](scripts/double_audit.py)
[![Apple Silicon Metal](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-black.svg)](docs/ARCHITECTURE.md)

---

## 1. The Manifesto: Cognitive Mirror Sovereignty

In an era dominated by centralized hyperscaler cloud models that monetize human correspondence and flatten distinct human consciousness into generic corporate prose, **Time Capsule** offers a sovereign, private alternative. 

Human identity is neither static nor monolithic. It compounds across decades—shaped by youthful idealism, builder grit, crucible failures, and mature synthesis. Rather than surrendering your personal journals, audio, thoughts, and memories to external servers, Time Capsule enables you to build an autonomous, mathematically grounded **Imago Dei** (an image of your own cognition) operating entirely on your personal machine.

### Core Philosophical Invariants
1. **Local Sovereignty Over Cloud Delegation**: Your memories, voice prints, likeness, and reasoning models run 100% locally on your own hardware (`127.0.0.1`). Zero outbound telemetry, zero third-party data broker transmission.
2. **Honest Epistemic Uncertainty**: When the archive lacks direct evidence for a query, the model refuses to hallucinate synthetic biographical facts. It states its epistemic bounds transparently.
3. **Chronological Dialectics**: Historical eras (Formative Years, Builder Phase, Present Day, Panoramic View) remain distinct. Past perspectives are preserved in their authentic context without being retroactively rewritten by present hindsight.
4. **Simplicity is Complexity Resolved**: Complex multi-modal models (Whisper ASR, F5-TTS flow matching, LivePortrait video avatars, Hybrid FTS5 + ChromaDB RAG, and TypeSafe AI Jev verification) are unified under clean, deterministic, scriptable interfaces.

---

## 2. Key Capabilities & Architectural Pillars

### 🛡️ Public Secure Double Audit System
Time Capsule incorporates a rigorous **two-layer privacy audit engine** (`scripts/double_audit.py`) designed to prevent accidental information leaks:
- **Layer 1 (Static Audit)**: Scans files, git commits, configuration dictionaries, and ASTs for API secrets, private bearer tokens, personal email addresses, phone numbers, Wi-Fi credentials, and hardcoded home directory paths.
- **Layer 2 (Runtime & Semantic Leak Interception)**: An active gateway intercepting RAG context chunks, prompt synthesis, and API output payloads. If synthetic canaries or unredacted PII are detected, the payload is sanitized before returning to the user or writing to disk.

```bash
# Run strict double privacy audit across the entire codebase
python scripts/double_audit.py --strict
```

### 🌐 Tri-Mode Internet RAG Engine
Dynamic RAG mode switching directly from the UI header or REST API:
- 🔒 **Airgapped Local (Default)**: Strictly offline. Zero external network requests. Answers are grounded purely in local SQLite FTS5 and ChromaDB vector embeddings.
- 🛡️ **Jev-Controlled Internet**: Uses TypeSafe AI / Jev single-token verification gating:
  1. Jev evaluates proposition `requires_external_factual_grounding`. If false (e.g. personal inquiry), web access is blocked.
  2. If true, synthesizes concise search queries.
  3. Sanitizes search snippets against prompt injection, tracking scripts, and malicious instructions.
  4. Verifies epistemic relevance before passing facts into the generation context.
- 🌐 **Open Web**: Directly augments memory retrieval with real-time web search for queries requiring global contemporary context.

### 🧠 Latest Open-Weights Model Ingestion Pipeline
An automated ingestion and benchmarking pipeline (`ingestion/model_pipeline.py`, `scripts/ingest_models.py`) supporting the latest open-weights model families:
- **Supported Model Families**:
  - **LLaMA 3.3 (70B) & LLaMA 3.2 (1B, 3B)**
  - **Qwen 2.5 (0.5B to 72B) & Qwen 2.5-Coder**
  - **DeepSeek-R1 (Distilled Qwen & LLaMA variants)**
  - **Mistral Nemo (12B) & Mistral Small (24B)**
  - **Gemma 2 (2B, 9B, 27B)**
  - **Phi-3.5 / Phi-4**
- **Hardware Profiler & Memory Headroom**: Automatically detects Apple Silicon Unified Memory or discrete GPU VRAM, calculates memory footprints across 4-bit, 8-bit, and 16-bit quantizations, and flags whether the model fits comfortably without SSD swapping.
- **Benchmark Suite**: Evaluates real generation throughput (tokens/sec), time-to-first-token (TTFT), and autobiographical epistemic alignment.

```bash
# Inspect system memory and recommended models
python scripts/ingest_models.py profile

# List supported model configurations and memory fit
python scripts/ingest_models.py list

# Onboard a modern model via Ollama or Apple MLX
python scripts/ingest_models.py pull llama3.2:3b --backend ollama
```

### 🎙️ Studio-Grade Neural Voice & Live Push-to-Talk
- **Live Microphone Push-to-Talk ASR**: Powered by Whisper with two-stage voice activity detection (VAD), standardizing incoming browser audio to 16kHz mono Linear PCM WAV via FFmpeg.
- **Persistent F5-TTS Metal MPS Daemon**: Fast, expressive flow-matching neural voice synthesis running on Apple Silicon GPU (`:7861`), delivering 24kHz broadcast-quality speech.
- **Parametric DSP De-Humming**: Real-time notch filtering (50 Hz / 60 Hz hum elimination) and dynamic reference clip isolation.

### 👤 Photorealistic Era-Aware Avatar Reenactment
- **Optical-Flow Talking-Head Synthesis**: Modular LivePortrait pipeline with asynchronous job management, spontaneous eye-blink synthesis (~3.2s cycle), natural dental highlights, and 3D head kinematics muxed at 25 FPS sample-locked H.264/AAC.

### 🌓 Minimalist Dual-Theme UI
- **Obsidian Dark Mode**: Low-fatigue slate/obsidian palette with emerald accents and crisp typography.
- **Editorial Paper White / Light Mode**: High-contrast, clean minimalist paper aesthetic with warm charcoal typography and crisp borders.
- Instant toggle button with automatic `prefers-color-scheme` synchronization and persistent local storage.

---

## 3. Git Branching Model: Local Dev vs Clean Production

To ensure private personal vlogs, local calibrations, and personal memory dumps are never leaked to public repositories, Time Capsule enforces a strict two-branch architecture:

```
                  ┌───────────────────────────────┐
                  │      dev (Local Sovereign)    │
                  │  - Personal media & vlogs     │
                  │  - Local hardware configs     │
                  │  - Active development & tests │
                  └──────────────┬────────────────┘
                                 │
                 Double Audit & Sanitization Gate
             (scripts/sanitize_for_prod.py --verify)
                                 │
                                 ▼
                  ┌───────────────────────────────┐
                  │      prod / main (Public)     │
                  │  - 100% PII-sanitized code    │
                  │  - Clean generic config.json  │
                  │  - Sample synthetic fixtures  │
                  │  - Public release v0.1.0      │
                  └───────────────────────────────┘
```

- **`dev` Branch**: Where you ingest your personal vlogs, train your custom voice clones, and experiment. Local only.
- **`prod` Branch**: The public, open-source branch. Completely sanitized of user paths, private credentials, and personal recordings. Verified by `scripts/double_audit.py` before every merge.

---

## 4. Quickstart Guide

### 4.1 Prerequisites
- **macOS** (Apple Silicon M-series recommended for Metal MPS acceleration) or **Linux** (NVIDIA CUDA or CPU).
- Python 3.11+.
- Ollama or Apple MLX for local LLM inference.
- FFmpeg installed (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu).

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/your-username/time-capsule.git
cd time-capsule

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Configuration
Time Capsule automatically reads configuration from environment variables or `config/timecapsule/config.json`. All paths default dynamically to relative project directories:

```bash
# Optional environment overrides:
export TIMECAPSULE_DATA_DIR="./data/capsule"
export TIMECAPSULE_LLM_PROVIDER="ollama"
export TIMECAPSULE_RAG_MODE="airgapped" # airgapped | jev_controlled | open_internet
```

### 4.4 Launching Services
```bash
# Start the server (FastAPI REST backend and web interface on http://127.0.0.1:7860)
./scripts/start_server.sh

# Start the optional F5-TTS Metal Voice Daemon (port 7861)
./scripts/start_f5_daemon.sh
```

Navigate your browser to **`http://127.0.0.1:7860`**.

### 4.5 Running Verification & Double Audit
```bash
# Run full automated unit and integration tests
./scripts/run_tests.sh

# Run the strict double privacy audit
python scripts/double_audit.py --strict
```

---

## 5. Directory Layout

```
Time_Capsule/
├── app/
│   └── main.py                     # FastAPI server, REST routes, web UI controller
├── core/
│   ├── config.py                   # Portable configuration & hardware dataclasses
│   ├── conversation.py             # Conversation Engine & grounded prompt synthesis
│   ├── jev_engine.py               # TypeSafe AI Jev decision primitives & dual-model verifier
│   ├── models.py                   # Core Pydantic schemas, verification tiers, enums
│   ├── personality.py              # Chronological persona snapshots & dialectical beliefs
│   └── voice.py                    # Expressive acoustic controller & emotion mapping
├── ingestion/
│   ├── pipeline.py                 # Media discovery, ASR transcription & vectorization
│   └── model_pipeline.py           # Open-weights model onboarding, quantizer & profiler
├── memory/
│   ├── database.py                 # SQLite storage (WAL mode, FTS5 full-text indexing)
│   ├── internet_retrieval.py       # Tri-mode Internet RAG engine (Airgapped / Jev / Open)
│   ├── retrieval.py                # Hybrid Reciprocal Rank Fusion (BM25 + ChromaDB ONNX)
│   └── vector_store.py             # Dense semantic vector persistence client
├── providers/
│   ├── asr/                        # Whisper & PyAV audio transcription providers
│   ├── avatar/                     # LivePortrait talking-head video synthesis providers
│   ├── llm/                        # Ollama, Apple MLX Metal, and Mock LLM providers
│   └── tts/                        # F5-TTS Metal MPS, macOS native, and Kokoro TTS providers
├── scripts/
│   ├── double_audit.py             # Layer 1 & Layer 2 zero-leak privacy audit engine
│   ├── ingest_models.py            # CLI for open-weights model discovery & benchmarks
│   ├── sanitize_for_prod.py        # Automated cleaner & branch sanitizer for public release
│   ├── start_server.sh             # Production daemon launcher
│   └── run_tests.sh                # End-to-end test execution suite
├── tests/                          # Automated unit, integration, and security test suites
└── ui/
    ├── static/                     # CSS (light/dark dual theme), Vanilla JS, icons
    └── templates/                  # Jinja2 minimalist web templates & architecture portal
```

---

## 6. Security, Privacy & Contributing

- **Zero-Telemetry Guarantee**: This repository contains zero third-party telemetry, tracking pixels, or phone-home requests.
- **Reporting Vulnerabilities**: To report any potential information disclosure or security issue, please open a private GitHub advisory.
- **Contributing**: Please review the [Branching Protocol](#3-git-branching-model-local-dev-vs-clean-production). All pull requests must pass `python scripts/double_audit.py --strict` with zero warnings before merging.

---

## 7. License

Released under the **MIT License**. Build freely, stay sovereign.
