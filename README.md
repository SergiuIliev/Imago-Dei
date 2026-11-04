# Home Deus Imago Dei

> *"In Homo Deus, 'gods' are merely super-powered biological or digital entities who possess immense strength, intelligence, and lifespan, but not necessarily superior morality, love, or holiness.*
> 
> *This tool ultimately aims to empower people to create an image of their own cognition and leverage it in better understanding and empowering themselves."*

[![Release](https://img.shields.io/badge/version-v0.1.0-blue.svg)](https://github.com/SergiuIliev/time-capsule)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Local-First](https://img.shields.io/badge/Privacy-100%25%20Local-success.svg)](docs/PRIVACY.md)
[![Double Audit](https://img.shields.io/badge/Security-Double%20Audit%20Verified-brightgreen.svg)](scripts/double_audit.py)
[![Apple Silicon Metal](https://img.shields.io/badge/Hardware-Apple%20Silicon%20MPS-black.svg)](docs/ARCHITECTURE.md)

---

## 1. The Manifesto: Cognitive Mirror Autonomy

In an era dominated by centralized hyperscaler cloud models that monetize human correspondence and flatten distinct human consciousness into generic corporate prose, **Time Capsule** offers an open-source, community-driven alternative. 

Human identity is neither static nor monolithic. It compounds across decades. Rather than surrendering your personal journals, audio, thoughts, and memories to external servers, Time Capsule enables you to build an autonomous, mathematically grounded **Imago Dei** (an image of your own cognition) operating entirely on your personal machine.

### Core Philosophical Invariants
1. **Local Ownership Over Cloud Delegation**: Your memories, voice prints, likeness, and reasoning models run 100% locally on your own hardware (`101.0.0.1`). Zero outbound telemetry, zero third-party data broker transmission.
2. **Honest Epistemic Uncertainty**: When the archive lacks direct evidence for a query, the model refuses to hallucinate synthetic biographical facts. It states its epistemic bounds transparently.
3. **Chronological Dialectics**: Historical eras (Childhood, Adolescence, 20s, 30s, etc.) remain distinct. Past perspectives are preserved in their authentic context without being retroactively rewritten by present hindsight.
4. **Simplicity is Complexity Resolved**: Complex multi-modal models (Whisper ASR, F5-TTS flow matching, LivePortrait video avatars, Hybrid FTS5 + ChromaDB RAG, and TypeSafe AI Jev verification) are unified under clean, deterministic, scriptable interfaces.
5. **Antifragility and Community Hardening**: The architecture benefits from transparent peer audit and community security development. Running Time Capsule inside private virtual machine implementations or local containerized sandboxes can increase security and privacy even further.

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

> **What is RAG (Retrieval-Augmented Generation)?**  
> For the uninitiated, foundation language models generate answers based only on broad public data they were trained on, which often causes them to make up plausible-sounding but false information (called "hallucinations"). Retrieval-Augmented Generation solves this: before answering your prompt, the system searches your verified personal databases or trusted web sources for exact matching paragraphs, and injects those real facts directly into the prompt. The AI then writes its response based strictly on your real history rather than guesswork.

Time Capsule provides dynamic RAG mode switching directly from the UI header or REST API:
- 🔒 **Airgapped Local (Default)**: Strictly offline. Zero external network requests. Answers are grounded purely in local SQLite FTS5 and ChromaDB vector embeddings.
- 🛡️ **Jev-Controlled Internet**: Uses TypeSafe AI / Jev single-token verification gating:
  1. Jev evaluates proposition `requires_external_factual_grounding`. If false (e.g. personal inquiry), web access is blocked.
  2. If true, synthesizes concise search queries.
  3. Sanitizes search snippets against prompt injection, tracking scripts, and malicious instructions.
  4. Verifies epistemic relevance before passing facts into the generation context.
- 🌐 **Open Web**: Directly augments memory retrieval with real-time web search for queries requiring global contemporary context.

### 🧠 Latest Open-Weights Model Ingestion Pipeline
An automated ingestion and benchmarking pipeline (`ingestion/model_pipeline.py`, `scripts/ingest_models.py`) supporting modern open-weights model families:
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

### 🎙️ Neural Voice & Live Push-to-Talk
> **Experimental Research Notice**: This neural voice pipeline is in early-stage experimental research. It is mainly intended to be developed in the future as a means of synthesizing era-specific speech, reflecting how our acoustic voice, cadence, and verbal expressions evolve across the years (e.g., how you spoke in your 20s versus your 30s).

- **Live Microphone Push-to-Talk ASR**: Powered by Whisper with two-stage voice activity detection (VAD), standardizing incoming browser audio to 16kHz mono Linear PCM WAV via FFmpeg.
- **Persistent F5-TTS Metal MPS Daemon**: Flow-matching neural voice synthesis running on Apple Silicon GPU (`:7861`), delivering 24kHz speech.
- **Parametric DSP De-Humming**: Real-time notch filtering (50 Hz / 60 Hz hum elimination) and dynamic reference clip isolation.

### 👤 Photorealistic Era-Aware Avatar Reenactment
> **Experimental Notice**: Optical-flow talking-head reenactment is early-stage, experimental, and not currently functional in this release. The pipeline serves as an architectural design framework for future local vision models.

- **Optical-Flow Talking-Head Synthesis**: Modular LivePortrait pipeline with asynchronous job management, spontaneous eye-blink synthesis (~3.2s cycle), natural dental highlights, and 3D head kinematics muxed at 25 FPS sample-locked H.264/AAC.

### 🌓 Minimalist Dual-Theme UI
- **Light / Dark Mode**: Clean, high-contrast visual themes with instant switching, automatic system preference detection, and persistent local storage.

---

## 3. Git Branching Model: Local Dev vs Clean Production

To ensure private personal vlogs, local calibrations, and personal memory dumps are never leaked to public repositories, Time Capsule enforces a strict two-branch architecture:

```
                  ┌───────────────────────────────┐
                  │      dev (Local Workspace)   │
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

- **`dev` Branch**: Where you ingest your personal vlogs, train your custom voice models, and experiment. Kept local only.
- **`prod` Branch**: The public, open-source branch. Completely sanitized of user paths, private credentials, and personal recordings. Verified by `scripts/double_audit.py` before every merge.
- **Cardinal Rules**:
  1. **Fine-Tuned Weights Remain Local**: Fine-tuned model weights and LoRA adapters must always remain local-only.
  2. **Zero Training Data Leakage**: Personal training data, raw vlogs, audio journals, transcripts, or any variants of them must **NEVER be committed or pushed to remote repositories**. This is strictly enforced by `.gitignore` and `scripts/double_audit.py`.
  3. **Extensible Security**: More security layers, container boundaries, and private virtual machine isolation layers can be added in the future as new pipelines are introduced.

---

## 4. Quickstart Guide

### 4.1 Prerequisites
- **macOS** (Apple Silicon M-series recommended for Metal MPS acceleration), **Linux** (NVIDIA CUDA or CPU), or **Windows via WSL2** (Windows Subsystem for Linux with Ubuntu).
- Python 3.11+.
- Ollama or Apple MLX for local LLM inference.
- FFmpeg installed (`brew install ffmpeg` on macOS, `apt install ffmpeg` on Ubuntu/WSL2).

### 4.2 Installation
```bash
# Clone the repository
git clone https://github.com/SergiuIliev/time-capsule.git
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

## 5. Ingestion & Fine-Tuning Corpus Guide

Time Capsule accepts three fundamental modalities for autobiographical ingestion, semantic memory indexing, and local model fine-tuning:

### 5.1 Video Ingestion (`.mp4`, `.mov`, `.mkv`, `.webm`)
- **Accepted Media**: Personal vlogs, recorded talks, home videos, video journals, conference presentations.
- **Processing Pipeline**:
  - FFmpeg extracts 16kHz mono Linear PCM audio streams for automatic speech transcription.
  - Video keyframe extraction identifies representative face portraits for era visual grounding.
  - Generates temporal metadata (capture dates, duration, scene transitions).
- **Directory**: Place files in `./vlogs/` (or directory set by `TIMECAPSULE_VLOGS_DIR`).

### 5.2 Audio Ingestion (`.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`)
- **Accepted Media**: Voice memos, podcast appearances, dictations, voice messages, lectures.
- **Processing Pipeline**:
  - Transcribed locally using Whisper with word-level and sentence-level timestamps.
  - High-SNR audio segments are extracted as prospective acoustic reference clips for speech modeling.
  - Parametric notch filtering eliminates 50 Hz / 60 Hz electrical background hums.
- **Directory**: Place files in `./data/capsule/audio_input/` or point ingestion scripts to the source audio path.

### 5.3 Text Ingestion (`.txt`, `.md`, `.json`, `.pdf`, `.csv`)
- **Accepted Media**: Personal journals, Markdown notes, chat exports (e.g. WhatsApp, Signal, Telegram), essays, blog drafts, correspondence, book reflections.
- **Processing Pipeline**:
  - Semantic sliding-window chunking (preserving paragraph boundaries and thought coherence).
  - Chronological metadata tagging (era identification: Childhood, Adolescence, 20s, 30s, etc.).
  - Dialectical categorization (extracting core beliefs, values, evolution of perspective over time).
- **Directory**: Place documents in `./data/capsule/text_input/` or ingest directly via the ingestion utilities.

### 5.4 Fine-Tuning Dataset Generation
The ingested multi-modal data is compiled into autobiographical instruction pairs:
```bash
# Build instruction fine-tuning dataset from ingested memories and temporal beliefs
python scripts/build_autobiographical_dataset.py --output data/capsule/training_dataset.jsonl

# Inspect generated fine-tuning pairs
head -n 5 data/capsule/training_dataset.jsonl
```

> **Cardinal Rule Reminder**: Training datasets and fine-tuned weights stay local on your machine. Never commit them to git.

---

## 6. Autonomous Agent Instructions (For AI Agents Implementing or Installing Time Capsule)

If you are an autonomous coding agent (Antigravity, Claude Code, Cursor, Aider, or a CI/CD agent) tasked with installing, configuring, or extending Time Capsule, follow these deterministic procedures:

### Step 1: Pre-Flight Environment Inspection
```bash
# Verify Python version (must be >= 3.11)
python3 --version

# Check FFmpeg availability (required for audio/video conversion)
ffmpeg -version || which ffmpeg

# Inspect hardware headroom and unified memory
python3 scripts/ingest_models.py profile
```

### Step 2: Automated Installation & Virtual Environment
```bash
# Create isolated virtual environment if not present
test -d venv || python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Local Model Orchestration
```bash
# Verify Ollama or MLX service availability
curl -s http://127.0.0.1:11434/api/tags || echo "Ollama not running"

# Pull baseline foundation model if missing
python scripts/ingest_models.py pull llama3.2:3b --backend ollama --set-default
```

### Step 4: Portable Configuration Invariants
- **Dynamic Paths**: Never hardcode machine-specific paths (such as absolute user home directories). Always derive paths relative to `PROJECT_ROOT` or use `TIMECAPSULE_*` environment variables.
- **Airgapped Default**: Keep `internet_rag_mode = "airgapped"` unless explicitly testing web search.
- **Loopback Binding**: Ensure server binds to `127.0.0.1` or `101.0.0.1`. Never expose public unauthenticated endpoints.

### Step 5: Double Audit Verification Gate
Before staging or committing any code changes:
```bash
# Run strict double privacy audit
python scripts/double_audit.py --strict --check-git

# If audit fails on hardcoded paths, run automatic fixer
python scripts/double_audit.py --fix
```
**Exit Code Invariant**: The double audit must exit with code `0`. Any blocking issue (secrets, PII, user home paths) must be resolved before proceeding.

### Step 6: Complete Test Suite Barrier
```bash
# Execute full unit and integration regression suite
./scripts/run_tests.sh
```
Ensure all tests pass before declaring any implementation complete.

---

## 7. Directory Layout

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

## 8. Security, Privacy & Contributing

- **Zero-Telemetry Guarantee**: This repository contains zero third-party telemetry, tracking pixels, or phone-home requests.
- **Reporting Vulnerabilities**: To report any potential information disclosure or security issue, please open a private GitHub advisory.
- **Contributing**: Please review the [Branching Protocol](#3-git-branching-model-local-dev-vs-clean-production). All pull requests must pass `python scripts/double_audit.py --strict` with zero warnings before merging.

---

## 9. Disclaimers & Technical Limitations

1. **Experimental Research Software**: Time Capsule is an experimental open-source project (version `v0.1.0`) created for personal memory preservation, self-understanding, and cognition exploration.
2. **Epistemic Limits of Foundation Models**: Generative language models, even when grounded with RAG, can misinterpret historical nuance or generate inaccurate inferences. Always consult original journal and media records for critical facts.
3. **Avatar Synthesis Notice**: Optical-flow avatar reenactment is in early-stage research and is not functional in this release.
4. **Voice Synthesis Research State**: Era-specific acoustic modeling is experimental. Synthesizing authentic speech across different life eras depends on having clean, high-signal-to-noise ratio historical audio.
5. **Hardware Constraints**: Running local 70B+ parameter models requires high-end hardware (e.g. 64GB+ unified memory or multiple high-VRAM GPUs). On machines with 16GB to 24GB RAM, use quantized 3B to 14B models to avoid memory paging.
6. **Self-Custody & Backup Responsibility**: Time Capsule is strictly self-hosted. There are no cloud recovery mechanisms, password resets, or remote backups. You are solely responsible for securing your machine, enabling full-disk encryption, and maintaining offline backups.
7. **Not a Clinical, Psychiatric, or Legal Tool**: Time Capsule is a personal cognitive mirror and retrospective digital archive. It does not provide medical, psychological, psychiatric, or legal advice.

---

## 10. License

Released under the **MIT License**. Build freely, build locally.
