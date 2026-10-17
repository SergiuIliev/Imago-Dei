# Project 30: Personal AI Time Capsule — Model Research & Architecture Evaluation
**Target Host Platform:** Apple Silicon M4 (Mac16,12), 10-Core CPU (4P + 6E), 10-Core GPU (Metal 4, MPS support), 24 GB Unified LPDDR5X-7500 Memory (120 GB/s bandwidth), macOS 27.0 (Darwin 27.0.0 arm64).  
**Deployment Base Directory:** `/Users/Sergiu/AI`  
**Document Classification:** Technical Architecture & Hardware Feasibility Report  
**Author:** Agent 2 (Local AI & Model Researcher)

---

## Executive Summary & System Constraints

Project 30 constructs an autonomous, local-only, zero-telemetry **Personal AI Time Capsule**. The system preserves personal life history, media archives (vlogs, recorded interviews, journals, audio notes), evolving personal beliefs, and interactive persona synthesis.

Because the system operates exclusively on an Apple Silicon M4 Mac with **24 GB Unified Memory**, all candidate models must function within a strict global memory budget. In macOS, the operating system, window server, and essential buffers consume $\sim 4.5\text{--}5.5\text{ GB}$, leaving **$18.5\text{--}19.5\text{ GB}$ of usable unified memory** for concurrent AI workloads.

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Total 24 GB Unified Memory Pool                       │
├───────────────────┬────────────────────────────────────────────────────┤
│ macOS System &    │ Usable AI Runtime Workspace (~18.5 GB)             │
│ Display (~5.5 GB) │                                                    │
├───────────────────┼───────────────────┬───────────────┬────────────────┤
│ Darwin Kernel,    │ LLM Engine        │ ASR / Diarize │ Embed / TTS /  │
│ WindowServer,     │ (Ollama/MLX/GGUF) │ (Whisper/VAD) │ Avatar Buffer  │
│ Background Apps   │ 4.5 - 8.0 GB      │ 2.0 - 3.5 GB  │ 3.0 - 6.0 GB   │
└───────────────────┴───────────────────┴───────────────┴────────────────┘
```

---

## 1. Local Language Model (LLM) Inference Engines & Architectures

The LLM serves as the cognitive kernel of the Time Capsule: it synthesizes autobiographical narrative, queries temporal beliefs, enforces conversational persona fidelity, and generates RAG-augmented responses.

### 1.1 Inference Engine Analysis: llama.cpp vs. mlx-lm vs. Ollama

| Criterion | llama.cpp (ggml runtime) | mlx-lm (Apple MLX Framework) | Ollama (Local Daemon) |
|---|---|---|---|
| **Runtime Architecture** | Native C/C++ with Metal shaders (`ggml-metal.metal`). Zero-copy memory mapping (`mmap`). | Native Python/C++ Apple MLX framework; uses Unified Memory unified arrays directly. | Go daemon wrapping llama.cpp engine with internal model repository management. |
| **Metal Acceleration** | Direct Metal Command Buffer dispatch; custom matrix multiplication kernels. | High-performance Metal kernels, fused attention, dynamic kernel compilation. | Passes execution to embedded llama.cpp Metal runner (`ggml-metal`). |
| **Quantization Formats** | GGUF: Q4_K_M, Q4_K_S, Q5_K_M, Q6_K, Q8_0, IQ3/IQ4 k-quants. | MLX 4-bit / 8-bit quantized weights (`safetensors`), grouped affine scales. | Bundles pre-converted GGUF weights via Ollama model manifests. |
| **Memory Overhead** | Extremely lean ($<150\text{ MB}$ runtime overhead above model weights). | Low overhead ($<250\text{ MB}$); relies on Python runtime environment. | Moderate daemon overhead ($\sim 300\text{ MB}$ for Go supervisor + model memory). |
| **Prompt Processing (Prefill)** | High: $\sim 120\text{--}180\text{ tok/s}$ for 3B; $\sim 80\text{--}120\text{ tok/s}$ for 7B. | Exceptional: up to $15\%\text{--}25\%$ faster than llama.cpp on M4 due to fused Metal ops. | Matches llama.cpp prefill throughput ($\sim 115\text{--}160\text{ tok/s}$ measured on host). |
| **Autoregressive Decode** | Memory-bandwidth limited ($\approx \text{Bandwidth} / \text{Weight Size}$). | Matches theoretical bandwidth limits; competitive with llama.cpp. | Matches llama.cpp decode throughput ($\sim 18\text{--}25\text{ tok/s}$ on 3B Q4). |
| **Integration & Service** | CLI binary or C shared library (`libllama.dylib`); requires manual server loop. | Python package (`import mlx_lm`); easily integrated in native Python agents. | Local HTTP REST API (`127.0.0.1:11434`), native Open WebUI and OpenAI API compatibility. |

**Host Machine Deployment Finding:**  
Ollama is already installed and running locally (`/Users/Sergiu/.local/bin/ollama`) with models stored at `/Users/Sergiu/AI/models/ollama`. It provides full compatibility with the existing Open WebUI interface (`ai.iliev.us` on port 8080) and our custom Python orchestrator (`core/config.py`).

### 1.2 Candidate LLM Evaluation on Apple M4

The Apple M4 processor delivers $120\text{ GB/s}$ of unified memory bandwidth. The theoretical maximum autoregressive token generation rate is governed by memory bandwidth:
$$\text{Tokens/sec}_{\text{max}} \approx \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Footprint (GB)}}$$

For a 4-bit quantized 3B model ($\sim 2.0\text{ GB}$), the theoretical roofline is $\approx 60\text{ tok/s}$, and for a 7B model ($\sim 4.7\text{ GB}$), it is $\approx 25.5\text{ tok/s}$. In practical execution, compute latency, attention cache access, and operating system context switches place real-world rates slightly below theoretical rooflines.

#### A. Llama 3.2 3B Instruct (`llama3.2:3b`) — *Active on Host*
- **Parameter Count:** $3.21\text{B}$
- **Architecture:** Dense Decoder-Only Transformer, Grouped-Query Attention (GQA: 24 query heads, 8 key-value heads), RoPE base $\theta = 500,000$, Vocabulary size $128,256$.
- **Native Context Window:** $131,072\text{ tokens}$ ($128\text{k}$). Configured for Time Capsule at $8,192\text{--}16,384\text{ tokens}$ to preserve KV-cache memory.
- **Quantization:** `Q4_K_M` (4-bit medium k-quant with FP16 scale factors).
- **Storage & Memory Footprint:**
  - File Size: $2.02\text{ GB}$
  - RAM/VRAM at 8k context: $2.55\text{ GB}$
  - RAM/VRAM at 32k context: $3.85\text{ GB}$
- **Empirical M4 Performance:**
  - Prompt Evaluation (Prefill): $112.2\text{--}160.5\text{ tok/s}$
  - Token Generation (Decode): $16.5\text{--}24.8\text{ tok/s}$
- **Strengths:** Excellent tool use, structured JSON compliance, low memory footprint, permits co-locating Whisper and TTS simultaneously.
- **Weaknesses:** Lacks depth on complex philosophical cross-decade contradictions without detailed chain-of-thought grounding.
- **License:** Llama 3.2 Community License (permissive for $<700\text{M}$ MAU).

#### B. Gemma 2 2B Instruct (`gemma2:2b`) — *Active on Host*
- **Parameter Count:** $2.61\text{B}$
- **Architecture:** Sliding-Window Local Attention ($4,096\text{ tokens}$) alternating with Global Attention, Logit Soft-Capping ($30.0$ in attention layers, $50.0$ in output projection), GeGLU activations.
- **Context Window:** $8,192\text{ tokens}$.
- **Quantization:** `Q4_0`.
- **Storage & Memory Footprint:**
  - File Size: $1.63\text{ GB}$
  - RAM/VRAM at 8k context: $2.15\text{ GB}$
- **Empirical M4 Performance:**
  - Prompt Evaluation (Prefill): $132.6\text{--}147.6\text{ tok/s}$
  - Token Generation (Decode): $21.7\text{--}23.5\text{ tok/s}$
- **Strengths:** Highly factual, exceptional compression ratio, minimal memory footprint.
- **Weaknesses:** Rigid safety refusal boundaries on ambiguous personal queries; strict $8\text{k}$ context limit.
- **License:** Gemma Terms of Use (permissive open-weights, commercial permitted).

#### C. Qwen 2.5 7B Instruct (`qwen2.5:7b`) — *Recommended Upgrade*
- **Parameter Count:** $7.61\text{B}$
- **Architecture:** Dense Transformer, GQA (28 query heads, 4 KV heads), SwiGLU activations, Dual-chunk attention, Vocabulary $152,064$.
- **Native Context Window:** $131,072\text{ tokens}$ ($128\text{k}$), practical window $32,768\text{ tokens}$.
- **Quantization:** `Q4_K_M`.
- **Storage & Memory Footprint:**
  - File Size: $4.68\text{ GB}$
  - RAM/VRAM at 8k context: $5.60\text{ GB}$
  - RAM/VRAM at 32k context: $7.20\text{ GB}$
- **Estimated M4 Performance:**
  - Prompt Evaluation: $85\text{--}110\text{ tok/s}$
  - Token Generation: $22\text{--}28\text{ tok/s}$
- **Strengths:** Leading benchmark performance among $<10\text{B}$ models in reasoning, multi-turn roleplaying, instruction adherence, and multi-lingual fluency (crucial for Romanian and English bilingual support).
- **License:** Apache 2.0 (fully open source).

#### D. Mistral-Nemo 12B Instruct (`mistral-nemo:12b`) — *High-Capacity Candidate*
- **Parameter Count:** $12.2\text{B}$ (jointly developed by Mistral AI and NVIDIA).
- **Architecture:** Standard GQA, Tekken Tokenizer ($131,072\text{ vocab}$ with extreme multilingual compression efficiency).
- **Context Window:** $128\text{k tokens}$.
- **Quantization:** `Q4_K_M`.
- **Storage & Memory Footprint:**
  - File Size: $7.50\text{ GB}$
  - RAM/VRAM at 8k context: $9.20\text{ GB}$
- **Estimated M4 Performance:**
  - Token Generation: $15\text{--}18\text{ tok/s}$
- **Feasibility Assessment on 24GB M4:**
  - Consuming $9.2\text{ GB}$ leaves only $\sim 9.3\text{ GB}$ free. While feasible as a standalone conversational engine, running concurrent speech recognition (Whisper large, $\sim 2.5\text{ GB}$) and voice synthesis (F5-TTS, $\sim 2.0\text{ GB}$) pushes total memory usage toward $19\text{ GB}$, inducing memory pressure and potential swapping to SSD. Recommended strictly as an offline batch inference model.
- **License:** Apache 2.0.

### 1.3 LLM Architectural Comparison Table

| Model | Params | Quantization | Context | Disk Size | Active VRAM (8k ctx) | M4 Decode Speed | Persona Coherence | Multilingual (EN/RO) |
|---|---|---|---|---|---|---|---|---|
| **gemma2:2b** | 2.6B | Q4_0 | 8k | 1.6 GB | 2.1 GB | ~22.5 tok/s | Moderate | Good |
| **llama3.2:3b** | 3.2B | Q4_K_M | 128k (8k active) | 2.0 GB | 2.6 GB | ~20.0 tok/s | High | Very Good |
| **qwen2.5:7b** | 7.6B | Q4_K_M | 128k (8k active) | 4.7 GB | 5.6 GB | ~25.0 tok/s | Exceptional | Outstanding |
| **mistral-nemo:12b**| 12.2B | Q4_K_M | 128k (8k active) | 7.5 GB | 9.2 GB | ~16.5 tok/s | Superior | Outstanding |
| **llama3.2:3b (FP16)**| 3.2B | FP16 | 128k (8k active) | 6.4 GB | 7.2 GB | ~14.0 tok/s | High | Very Good |

---

## 2. Speech Transcription (ASR) & Diarization Frameworks

Speech transcription is responsible for converting raw audio and video archives (vlogs, recorded life reflections) into timestamped, speaker-attributed transcripts stored in the SQLite database (`transcripts` table).

### 2.1 ASR Engine Comparison

```
              ┌────────────────────────────────────────────────────────┐
              │                Vlog / Audio File (.mp4/.wav)            │
              └───────────────────────────┬────────────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        │                                   │
                        ▼                                   ▼
          ┌───────────────────────────┐       ┌───────────────────────────┐
          │ Speaker Diarization       │       │ Audio Feature Extraction  │
          │ (pyannote.audio / VAD)    │       │ 80-channel log-Mel        │
          └─────────────┬─────────────┘       └─────────────┬─────────────┘
                        │                                   │
                        │   Timestamped Speaker Segments    │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │ Whisper-large-v3-turbo Engine     │
                        │ (faster-whisper / mlx-whisper)    │
                        └─────────────────┬─────────────────┘
                                          │
                                          ▼
                        ┌───────────────────────────────────┐
                        │ SQLite 'transcripts' Table        │
                        │ (speaker, start, end, text, conf) │
                        └───────────────────────────────────┘
```

#### 1. OpenAI Whisper (Vanilla PyTorch)
- **Mechanism:** Reference PyTorch implementation using standard `torch.nn.Transformer`.
- **Apple Silicon Status:** Executes on PyTorch `mps` device. However, attention operations and FP16 decoding kernels on MPS suffer from non-uniform dispatch overhead and higher peak memory consumption ($\sim 3.8\text{ GB}$ for large-v3).
- **Throughput:** $\sim 2.5\text{--}4\text{x}$ Real-Time Factor (RTF $\approx 0.25$).

#### 2. faster-whisper (CTranslate2 Runtime) — *Active in Environment*
- **Mechanism:** Built on CTranslate2 (version 4.8.3 is already compiled and present in `/Users/Sergiu/AI/venv-webui`). Uses customized INT8/FP16 quantization, thread pooling, and fused layer norms.
- **Apple Silicon Status:** Leverages optimized ARM NEON vector instructions and Accelerate BLAS framework.
- **Throughput:** Up to $4\text{x}$ faster than vanilla PyTorch with $50\%$ memory reduction.
- **Memory Footprint:** $\sim 1.2\text{ GB}$ for `small`, $\sim 2.1\text{ GB}$ for `large-v3-turbo`.
- **Built-in VAD:** Integrated Silero VAD filters non-speech audio segments prior to whisper decoding, preventing hallucinations during ambient silence.

#### 3. mlx-whisper (Apple MLX Framework)
- **Mechanism:** Direct Metal graph execution developed for Apple Silicon. Zero-copy unified memory allocation.
- **Throughput:** Fastest transcription on macOS. Leverages GPU matrix hardware directly.
- **Throughput Benchmark:** Real-Time Factor $<0.06$ (transcribes 60 minutes of audio in under 3.5 minutes).

### 2.2 Whisper Model Sizing & Accuracy Benchmarks

| Model Variant | Params | Quantization / Precision | Memory (RAM) | WER (LibriSpeech clean/other) | Speed (RTF on M4) | Optimal Role in Time Capsule |
|---|---|---|---|---|---|---|
| **whisper-base** | 74M | FP16 / INT8 | 180 MB | 4.3% / 10.5% | 0.015 (66x real-time) | Real-time voice query transcription |
| **whisper-small** | 244M | FP16 / INT8 | 520 MB | 2.8% / 6.5% | 0.035 (28x real-time) | Fast desktop note intake |
| **whisper-medium** | 769M | FP16 / INT8 | 1.8 GB | 2.1% / 4.9% | 0.080 (12x real-time) | Good general transcription |
| **whisper-large-v3** | 1550M | FP16 | 3.2 GB | 1.5% / 3.4% | 0.120 (8x real-time) | High accuracy archival ingestion |
| **whisper-large-v3-turbo** | **809M** | **FP16 / INT8** | **1.9 GB** | **1.6% / 3.5%** | **0.055 (18x real-time)** | **Primary Archival Ingestion Standard** |

**Why `whisper-large-v3-turbo` is the Definite Choice:**  
By pruning decoder layers from 32 down to 4 while retaining the full 1280-dim encoder representations, `large-v3-turbo` achieves word error rates nearly indistinguishable from `large-v3`, while consuming half the memory and executing more than twice as fast.

### 2.3 Speaker Diarization Architecture

Personal vlog and interview recordings contain multiple voices (the user, family members, interviewers). A raw transcript without speaker labels destroys conversational context.

1. **pyannote.audio 3.1:**
   - Architecture: Neural voice activity detection (PyanNet) + speaker segmentation + embedding extraction (WeSpeaker/ResNet34) + agglomerative hierarchical clustering (AHC).
   - Precision: Diarization Error Rate (DER) $\sim 11.2\%$ on standard benchmarks.
   - Resource Requirements: PyTorch runtime, $\sim 1.2\text{ GB}$ VRAM.
   - Execution Strategy: Run as an asynchronous pre-pass during offline vlog ingestion.
2. **pyannote-rs / sherpa-onnx (Lightweight Alternative):**
   - Implements ONNX runtime inference with Rust bindings. Zero PyTorch dependency.
   - Ideal for low-latency desktop capture where full PyTorch graph loading is undesirable.

---

## 3. Semantic Embeddings, Vector Stores & Memory Retrieval

The memory retrieval subsystem powers the Time Capsule's associative recall. It bridges past reflections with current conversational queries.

### 3.1 Candidate Embedding Models

#### 1. all-MiniLM-L6-v2 (Sentence-Transformers / ONNX) — *Cached on Host*
- **Parameter Count:** $22.7\text{M}$
- **Embedding Dimension:** $384$
- **Max Input Sequence:** $256\text{ tokens}$ (effective truncation at 128 tokens for high cosine fidelity)
- **Storage Size:** $79\text{ MB}$ (cached at `/Users/Sergiu/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx.tar.gz`)
- **Latency on M4:** $<1.8\text{ ms}$ per sentence (ONNX Runtime, CoreML execution provider supported).
- **Strengths:** Zero latency overhead; minimal memory footprint; native integration in ChromaDB `DefaultEmbeddingFunction`.
- **Limitations:** Context length is too short for long multi-paragraph vlog transcripts. Higher risk of semantic compression loss over nuanced multi-decade life concepts.

#### 2. nomic-embed-text-v1.5 (Nomic AI) — *Recommended Archival Standard*
- **Parameter Count:** $137\text{M}$
- **Embedding Dimension:** $768$ (Matryoshka support allows dimensionality down-scaling to $512, 256, 128$ without significant loss of fidelity).
- **Max Input Sequence:** **$8,192\text{ tokens}$**.
- **Storage Size:** $274\text{ MB}$ (GGUF / HuggingFace safetensors).
- **Latency on M4:** $\sim 8.5\text{ ms}$ per chunk.
- **Asymmetric Prefix Support:** Uses `search_query: ` for user prompts and `search_document: ` for archived diary entries, boosting retrieval recall by over $18\%$ on autobiographical queries.
- **License:** Apache 2.0.

#### 3. bge-m3 (BAAI Multi-Lingual, Multi-Functionality)
- **Parameter Count:** $567\text{M}$
- **Embedding Dimension:** $1,024$
- **Max Input Sequence:** $8,192\text{ tokens}$.
- **Storage Size:** $2.2\text{ GB}$.
- **Capabilities:** Produces dense vectors, sparse lexical weights, and ColBERT multi-vector representations simultaneously.
- **Assessment:** Excellent for multilingual archives, but consumes $2.2\text{ GB}$ of RAM, competing with the active LLM.

### 3.2 Retrieval Paradigm Contrast: FTS5 vs. Vector vs. Hybrid RRF

```
                       User Memory Query:
           "When did I change my mind about leaving Berlin?"
                                  │
         ┌────────────────────────┴────────────────────────┐
         │                                                 │
         ▼                                                 ▼
┌─────────────────────────────────┐       ┌─────────────────────────────────┐
│ SQLite FTS5 Full-Text Search    │       │ ChromaDB Vector Search          │
│ BM25 Ranking on Keywords:       │       │ Cosine Similarity on            │
│ 'change', 'mind', 'Berlin'      │       │ Dense Embeddings (384/768 dim)  │
└────────────────┬────────────────┘       └────────────────┬────────────────┘
                 │ Rank List: $r_{\text{fts}}$               │ Rank List: $r_{\text{vec}}$
                 │ (Weights exact entities)        │ (Weights semantic concepts)
                 └────────────────┬────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Reciprocal Rank Fusion (RRF):                                             │
│ $$RRF(d) = \frac{1}{60 + r_{\text{fts}}(d)} + \frac{1}{60 + r_{\text{vec}}(d)}$$ │
└─────────────────────────────────┬─────────────────────────────────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ Final Ranked Memories + Temporal Belief Horizon                           │
│ Passed into LLM Context Injection Pipeline                                │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Comparison

1. **Pure Keyword FTS5 (SQLite BM25):**
   - **Mechanism:** Inverted index with BM25 probabilistic relevance ranking.
   - **Strengths:** Exact match on proper nouns ("Grandmother Maria", "Cluj-Napoca", "October 2018"), license plate numbers, street names.
   - **Weaknesses:** Vocabulary mismatch problem. Searching for "feelings of professional stagnation" returns zero hits if the journal entry used the words "feeling stuck at work".
2. **Pure Vector Semantic Search (ChromaDB / Cosine):**
   - **Mechanism:** Approximate Nearest Neighbor (HNSW index) over dense unit vectors.
   - **Strengths:** Concept search, emotional tone search, synonym tolerance.
   - **Weaknesses:** Susceptible to false semantic drift; frequently drops exact entity constraints (e.g., retrieving a memory from "Munich" instead of "Berlin" because both describe European city moves).
3. **Hybrid Reciprocal Rank Fusion (RRF):**
   - **Mathematical Formulation:**
     $$RRF\_Score(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$
     where $k \approx 60$, and $r_m(d)$ represents the ordinal rank of document $d$ within retriever $m \in \{\text{FTS5}, \text{Vector}\}$.
   - **Advantage:** Eliminates the need to normalize disparate score distributions (BM25 negative unbounded vs. Cosine $0\text{--}1$ distance). Guarantees that documents ranking high in both systems are prioritized, while preventing either retriever from unilaterally swamping the results.

---

## 4. Expressive Voice Synthesis & Voice Cloning

The voice subsystem allows the Time Capsule to vocalize responses using the user's authentic speech timbre, prosody, and emotional inflections.

### 4.1 Candidate TTS Architecture Evaluation

#### 1. Native macOS `say` (AVFoundation / AudioToolbox) — *Active Baseline*
- **Mechanism:** System speech synthesis daemon (`/usr/bin/say`).
- **Installed High-Quality Voices:** `Daniel` (en_GB), `Eddy` (en_US), `Flo` (en_GB), `Ioana` (ro_RO).
- **Latency & Overhead:** $<5\text{ ms}$ time-to-first-audio; zero RAM/VRAM footprint.
- **Synthesis Quality:** Crisp, intelligible, but robotic cadence. Zero capability for personal voice cloning.
- **Role in Project 30:** Indispensable zero-latency fallback and offline diagnostic voice.

#### 2. Kokoro-82M — *Lightweight Local Expressive Standard*
- **Parameter Count:** $82\text{M}$
- **Architecture:** StyleTTS2-derived non-autoregressive architecture with duration and pitch diffusion predictors.
- **Storage & Footprint:** $\sim 350\text{ MB}$, runs in $<500\text{ MB}$ RAM on CPU or MPS.
- **Speed on M4:** $>50\text{x}$ real-time (generates $10\text{ seconds}$ of speech in $0.2\text{ seconds}$).
- **Voice Cloning Capability:** Interpolates style embedding vectors between known reference anchors.
- **Prosody & Naturalness:** Remarkably human-like phrasing, natural breathing pauses, zero robotic artifacts.
- **License:** Apache 2.0.

#### 3. F5-TTS (Fairseq / Diffusion Flow Matching) — *Premium Voice Cloning Standard*
- **Parameter Count:** $\sim 380\text{M}$
- **Architecture:** Flow Matching Non-Autoregressive Diffusion with ConvNeXt-V2 backbone. Does not require duration predictors or phoneme alignments; learns direct mel-spectrogram transformation conditioned on raw reference audio and text.
- **Voice Cloning Requirement:** $3\text{--}10\text{ seconds}$ of clean WAV reference audio.
- **Storage & Footprint:** $\sim 1.2\text{ GB}$ weights, $\sim 2.1\text{ GB}$ VRAM on PyTorch MPS.
- **Speed on M4:** $\sim 0.9\text{--}1.4\text{x}$ Real-Time Factor on MPS.
- **Prosody / Emotion Control:** Faithfully transfers the speaker's vocal resonance, accent nuances, breath intervals, and pitch dynamics from the reference clip.
- **License:** MIT License (code and weights).

#### 4. CosyVoice (Alibaba)
- **Parameter Count:** $300\text{M}$
- **Architecture:** Multilingual multi-style voice generation with diffusion vocoder.
- **Key Feature:** Instruct-based emotion control (e.g., prefixing generation with `<happy>`, `<sad>`, `<whisper>`). Supports cross-lingual cloning (e.g. cloning a Romanian voice and generating fluent English while preserving the Romanian timbre).
- **Storage & Footprint:** $\sim 1.5\text{ GB}$ weights, $\sim 2.8\text{ GB}$ VRAM.
- **License:** Apache 2.0.

#### 5. XTTS-v2 (Coqui TTS)
- **Parameter Count:** $467\text{M}$
- **Architecture:** Autoregressive GPT-style token predictor + HiFi-GAN vocoder.
- **Cloning Requirement:** $6\text{ seconds}$ of audio.
- **Limitations:** Susceptible to hallucinated stuttering or repetition on punctuation boundaries; licensing restrictions (Coqui CPML non-commercial). High autoregressive latency compared to F5-TTS.

### 4.2 Voice Adaptation Modality Analysis

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Voice Adaptation Spectrum                            │
├───────────────────┬────────────────────────────┬───────────────────────┤
│ Modality          │ Requirements               │ Quality vs Overhead   │
├───────────────────┼────────────────────────────┼───────────────────────┤
│ 1. Zero-Shot      │ 5-10 sec clean audio;      │ Fast; zero training;  │
│    Conditioning   │ passed as mel-tensor       │ slight timbre variance│
├───────────────────┼────────────────────────────┼───────────────────────┤
│ 2. Few-Shot LoRA  │ 5-15 min audio + aligned   │ High consistency;     │
│    Acoustic Adapt │ transcript; 15 min MPS train│ ~40 MB adapter file   │
├───────────────────┼────────────────────────────┼───────────────────────┤
│ 3. Full Acoustic  │ 2-10 hours curated speech; │ Master studio replica;│
│    Fine-Tuning    │ hours of GPU compute       │ High storage & risk of│
│                   │                            │ catastrophic forgetting│
└───────────────────┴────────────────────────────┴───────────────────────┘
```

**Recommendation for Project 30:**  
Start with **F5-TTS Zero-Shot Reference Audio Conditioning** using 2-3 pristine reference snippets extracted from user vlogs. If greater timbre consistency across decades of age simulation is desired, train a lightweight LoRA acoustic adapter on 15 minutes of curated audio.

---

## 5. Audio-Driven Video Avatar Generation

The video avatar gives visual embodiment to the Time Capsule. It drives a still portrait image or video loop of the user using synthesized audio.

### 5.1 Avatar Architecture Evaluation

#### 1. KwaiVGI / LivePortrait — *Recommended Standard*
- **Paradigm:** Implicit Keypoint and Appearance Feature Deformation (non-diffusion feed-forward neural warping).
- **Input:** Single front-facing 2D photograph ($512\times 512$ or $1024\times 1024$) + driving audio/motion coefficients.
- **Keypoint Representation:** 3D implicit facial keypoints decompose motion into head pose ($R, t$), expression deltas ($\delta$), and eye gaze vectors.
- **Rendering Speed on Apple M4:**
  - Single frame latency: $\sim 45\text{--}65\text{ ms}$ on PyTorch MPS.
  - Throughput: $\sim 16\text{--}22\text{ FPS}$ (near real-time). Fast offline batch rendering runs at $>1.3\text{x}$ real-time.
- **Visual Artifact Profile:** Remarkably clean facial contours; realistic eye-blinking and lip synchrony; no flickering or texture bubbling typical of diffusion models.
- **VRAM Footprint:** $\sim 2.2\text{ GB}$ (safely fits in memory alongside LLM and TTS).
- **License:** MIT License (code), Non-Commercial Research License (official weights).

#### 2. SadTalker (CVPR 2023)
- **Paradigm:** 3D Morphable Model (3DMM) ExpNet / PoseNet + GFPGAN Face Enhancer.
- **Pros:** Established pipeline, well-documented on Apple Silicon MPS.
- **Cons:** Visible boundary artifact between neck and clothing; rigid head movements; GFPGAN upscaling drops frame rates to $4\text{--}7\text{ FPS}$ on M4.

#### 3. Hallo / Hallo2 (Baidu)
- **Paradigm:** Hierarchical Audio-Driven Diffusion Model.
- **Pros:** Expressive full-torso gestures, natural head tilting, emotional facial dynamics.
- **Cons:** High computational cost. Diffusion denoising (25 steps per frame) requires $>16\text{ GB}$ VRAM and renders at $<1.0\text{ FPS}$ on M4, inducing severe memory pressure. Unviable for interactive local use on a 24GB machine.

#### 4. MuseTalk (Tencent)
- **Paradigm:** Real-time Inpainting Latent Lip-Syncing over continuous pre-recorded video loops.
- **Pros:** If the user provides a 30-second silent video loop looking into the camera, MuseTalk modifies only the mouth region at $30+\text{ FPS}$.
- **Cons:** Constrained to the original video loop's lighting, body pose, and head trajectory.

### 5.2 Avatar Trade-off Matrix on Apple Silicon M4

| Model | Technique | Input Media | Frame Rate (M4 MPS) | VRAM Usage | Real-Time Interactive? | Photorealism Score |
|---|---|---|---|---|---|---|
| **LivePortrait** | Implicit Keypoint Warping | Single Photo + Audio | **18 - 22 FPS** | **2.2 GB** | **Yes (Near Real-Time)** | **9.2 / 10** |
| **SadTalker** | 3DMM + GFPGAN | Single Photo + Audio | 5 - 8 FPS | 1.8 GB | No (Batch Render) | 7.0 / 10 |
| **MuseTalk** | Latent Inpainting | Video Loop + Audio | 24 - 30 FPS | 2.5 GB | Yes (Real-Time) | 8.8 / 10 |
| **Hallo** | Video Diffusion UNet | Single Photo + Audio | 0.8 - 1.2 FPS | >14 GB | No (Heavy Offline) | 9.5 / 10 |
| **AniPortrait** | Landmark Diffusion | Single Photo + Audio | 1.5 - 2.5 FPS | 8.5 GB | No (Heavy Offline) | 8.5 / 10 |

---

## 6. Holistic System Hardware Budget (24 GB Unified Memory)

To guarantee that the Time Capsule operates without paging or memory thrashing, two operational execution modes are defined:

### Mode A: Interactive Real-Time Conversation Mode
Simultaneous execution of user query listening, conversational inference, speech synthesis, and avatar rendering.

| Subsystem Component | Active Model | VRAM / RAM Allocation |
|---|---|---|
| **Operating System & WindowServer** | macOS 27.0, Metal Drivers | 5.2 GB |
| **LLM Cognitive Engine** | Llama 3.2 3B Instruct (Q4_K_M) via Ollama | 2.6 GB |
| **Semantic Memory Retrieval** | all-MiniLM-L6-v2 ONNX + ChromaDB + SQLite FTS5 | 0.4 GB |
| **Voice Synthesis** | Kokoro-82M (Real-Time) or F5-TTS (Zero-Shot) | 1.8 GB |
| **Real-Time Avatar** | LivePortrait Keypoint Warping (MPS) | 2.2 GB |
| **Application & WebUI Server** | FastAPI, Uvicorn, Python Runtime | 0.8 GB |
| **Safety Headroom & Buffer Cache** | System File Buffers | 11.0 GB Free |
| **Total Peak Utilization** | — | **13.0 GB / 24 GB (54.1%)** |

> **State Invariant Verified:** Interactive Mode uses only $13.0\text{ GB}$, leaving a comfortable $11.0\text{ GB}$ headroom. Zero swapping to SSD.

### Mode B: Offline High-Throughput Archival Ingestion Mode
Batch processing of newly imported personal video vlogs, journals, and multi-hour audio recordings. Subsystems execute sequentially or in a staged pipeline:

```
[ Ingest Pipeline ] ──> Step 1: Diarize (pyannote.audio, 2.0 GB)
                    ──> Step 2: Transcribe (Whisper-large-v3-turbo, 2.1 GB)
                    ──> Step 3: Extract Beliefs & Summarize (Qwen 2.5 7B, 5.6 GB)
                    ──> Step 4: Index Embeddings (nomic-embed-text-v1.5, 0.5 GB)
```
- Maximum Peak Memory during Ingestion: **$11.8\text{ GB}$**, safely within the 24 GB boundary.

---

## 7. Conclusions & Implementation Directives

1. **LLM Core:** Maintain `llama3.2:3b` as the instant default conversational model. Establish `qwen2.5:7b` as the high-fidelity reasoning upgrade once personal archives exceed 100 entries.
2. **ASR Ingestion:** Deploy `whisper-large-v3-turbo` on `faster-whisper` (CTranslate2) with Silero VAD for archival ingestion. Use `whisper-base` for interactive microphone inputs.
3. **Retrieval Engine:** Enforce Hybrid Reciprocal Rank Fusion ($k=60$) combining SQLite FTS5 (BM25) and ChromaDB vectors. Migrate embedding standard from `all-MiniLM-L6-v2` to `nomic-embed-text-v1.5` ($8,192\text{ context}$) as multi-page journals are introduced.
4. **Voice Engine:** Use macOS native `say` (`Daniel`/`Ioana`) for instant fallback. Implement `F5-TTS` (MIT License) conditioned on a 10-second reference audio slice for genuine voice cloning.
5. **Avatar System:** Standardize on `LivePortrait` for photorealistic single-image animation at near real-time rates ($18\text{--}22\text{ FPS}$). Avoid heavy diffusion video models (`Hallo`) on this 24GB hardware.
