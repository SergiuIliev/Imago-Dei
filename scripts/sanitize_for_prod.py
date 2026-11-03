#!/usr/bin/env python3
"""
Sanitization & Production Branch Preparer for Personal AI Time Capsule.
Prepares the codebase for zero-leak public open-source release.

Actions:
1. Creates `config/timecapsule/config.example.json` with generic paths and placeholder endpoints.
2. Creates `.env.example` with environment variable overrides.
3. Sanitizes `config/cloudflared/tunnel_token.txt` and `config/open-webui/secret_key.txt` with dummy values.
4. Creates `.example` files for tunnel tokens and secret keys.
5. Updates `.gitignore` to guarantee secret tokens, private vlogs, credentials, and local SQLite databases are NEVER tracked in production git.
6. Sanitizes hardcoded user paths across config templates.

Usage:
    python scripts/sanitize_for_prod.py
    python scripts/sanitize_for_prod.py --untrack-secrets
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_EXAMPLE_DATA = {
    "app_version": "v0.1.0",
    "storage": {
        "base_dir": ".",
        "data_dir": "./data/capsule",
        "sqlite_db_path": "./data/capsule/archive.db",
        "chroma_dir": "./data/capsule/vector_store",
        "media_input_dir": "./vlogs",
        "exports_dir": "./exports",
        "audio_output_dir": "./data/capsule/audio_cache",
        "models_dir": "./models"
    },
    "llm": {
        "provider": "ollama",
        "endpoint": "http://127.0.0.1:11434",
        "model_name": "llama3.2:3b",
        "fallback_model": "gemma2:2b",
        "critic_model": "qwen3:4b",
        "mlx_endpoint": "http://127.0.0.1:8081",
        "enable_jev_verification": True,
        "temperature": 0.35,
        "context_window": 8192,
        "timeout_seconds": 45.0
    },
    "embedding": {
        "provider": "chroma_onnx",
        "model_name": "all-MiniLM-L6-v2",
        "dimension": 384,
        "batch_size": 32
    },
    "asr": {
        "provider": "pyav_whisper",
        "model_size": "base",
        "compute_type": "float32",
        "device": "cpu"
    },
    "tts": {
        "provider": "kokoro",
        "default_voice": "default_voice",
        "sample_rate": 24000,
        "allow_reference_cloning": True,
        "reference_audio_dir": "./data/capsule/voice_references"
    },
    "avatar": {
        "provider": "liveportrait_stub",
        "target_fps": 25,
        "resolution": "512x512",
        "reference_portrait_image": ""
    },
    "security": {
        "local_only": True,
        "disable_telemetry": True,
        "bind_host": "127.0.0.1",
        "bind_port": 7860,
        "allow_remote_tunnels": False,
        "internet_rag_mode": "airgapped"
    },
    "retrieval": {
        "internet_rag_mode": "airgapped",
        "top_k_memories": 5,
        "top_k_beliefs": 3,
        "similarity_threshold": 0.65
    }
}

ENV_EXAMPLE_CONTENT = """# ==============================================================================
# Personal AI Time Capsule - Environment Configuration Template
# Copy this file to .env and adjust variables as needed for your local setup.
# ==============================================================================

# Application paths
TIMECAPSULE_BASE_DIR=.
TIMECAPSULE_DATA_DIR=./data/capsule
TIMECAPSULE_SQLITE_PATH=./data/capsule/archive.db
TIMECAPSULE_CHROMA_DIR=./data/capsule/vector_store
TIMECAPSULE_VLOGS_DIR=./vlogs
TIMECAPSULE_EXPORTS_DIR=./exports
TIMECAPSULE_AUDIO_CACHE_DIR=./data/capsule/audio_cache
TIMECAPSULE_MODELS_DIR=./models
TIMECAPSULE_VOICE_REF_DIR=./data/capsule/voice_references
TIMECAPSULE_CONFIG=./config/timecapsule/config.json

# Local AI Endpoints (All Loopback Airgapped)
OLLAMA_ENDPOINT=http://127.0.0.1:11434
MLX_ENDPOINT=http://127.0.0.1:8081
F5_TTS_ENDPOINT=http://127.0.0.1:7861

# Security & Retrieval
INTERNET_RAG_MODE=airgapped
TIMECAPSULE_HOST=127.0.0.1
TIMECAPSULE_PORT=7860
"""

GITIGNORE_CONTENT = """# ==============================================================================
# Personal AI Time Capsule - Strict Production Git Ignore
# Enforces Zero Info-Leak: No private vlogs, credentials, sqlite DBs, or weights
# ==============================================================================

# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class
*.so
.DS_Store

# Virtual Environments
venv/
.venv/
venv-webui/
venv-voice/
venv-mlx/

# Secrets, Credentials, Tokens & Keys
config/cloudflared/tunnel_token.txt
config/cloudflared/tunnel_id.txt
config/open-webui/secret_key.txt
*.key
*.pem
*.token
*.cert
*.crt
*.pfx
*.p12
.env
.env.local

# Databases & Vector Stores
*.db
*.db-journal
*.sqlite
*.sqlite3
data/
data/capsule/
data/capsule/archive.db*
data/capsule/vector_store/

# Model Weights, Tensors & Checkpoints
*.pt
*.safetensors
*.gguf
*.bin
*.onnx
models/

# Media Files, Private Vlogs, Audio Caches
*.wav
*.mp3
*.m4a
*.flac
*.mp4
*.mov
*.mkv
*.webm
vlogs/*
!vlogs/README.md
data/capsule/audio_cache/
data/capsule/voice_references/*
!data/capsule/voice_references/.gitkeep
exports/*
!exports/.gitkeep

# Process IDs & Log files
*.pid
logs/
*.log

# Build Artifacts & macOS App Bundles
Toggle Time Capsule.app/
"""


def create_sanitized_templates():
    print("[Sanitizer] Creating sanitized templates...")

    # 1. config/timecapsule/config.example.json
    cfg_example_path = PROJECT_ROOT / "config" / "timecapsule" / "config.example.json"
    cfg_example_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_example_path, "w", encoding="utf-8") as f:
        json.dump(CONFIG_EXAMPLE_DATA, f, indent=2)
    print(f"  Created: {cfg_example_path.relative_to(PROJECT_ROOT)}")

    # 2. .env.example
    env_example_path = PROJECT_ROOT / ".env.example"
    with open(env_example_path, "w", encoding="utf-8") as f:
        f.write(ENV_EXAMPLE_CONTENT)
    print(f"  Created: {env_example_path.relative_to(PROJECT_ROOT)}")

    # 3. Cloudflare tunnel token placeholder and example
    cf_dir = PROJECT_ROOT / "config" / "cloudflared"
    cf_dir.mkdir(parents=True, exist_ok=True)

    token_example_path = cf_dir / "tunnel_token.txt.example"
    with open(token_example_path, "w", encoding="utf-8") as f:
        f.write("PLACEHOLDER_CLOUDFLARE_TUNNEL_TOKEN_REPLACE_ME\n")
    print(f"  Created: {token_example_path.relative_to(PROJECT_ROOT)}")

    id_example_path = cf_dir / "tunnel_id.txt.example"
    with open(id_example_path, "w", encoding="utf-8") as f:
        f.write("00000000-0000-0000-0000-000000000000\n")
    print(f"  Created: {id_example_path.relative_to(PROJECT_ROOT)}")

    # Sanitize current tunnel_token.txt and tunnel_id.txt
    token_path = cf_dir / "tunnel_token.txt"
    with open(token_path, "w", encoding="utf-8") as f:
        f.write("PLACEHOLDER_CLOUDFLARE_TUNNEL_TOKEN_REPLACE_ME\n")
    print(f"  Sanitized: {token_path.relative_to(PROJECT_ROOT)}")

    id_path = cf_dir / "tunnel_id.txt"
    with open(id_path, "w", encoding="utf-8") as f:
        f.write("00000000-0000-0000-0000-000000000000\n")
    print(f"  Sanitized: {id_path.relative_to(PROJECT_ROOT)}")

    # 4. Open-WebUI secret key placeholder and example
    ow_dir = PROJECT_ROOT / "config" / "open-webui"
    ow_dir.mkdir(parents=True, exist_ok=True)

    sk_example_path = ow_dir / "secret_key.txt.example"
    with open(sk_example_path, "w", encoding="utf-8") as f:
        f.write("PLACEHOLDER_OPEN_WEBUI_SECRET_KEY_REPLACE_ME\n")
    print(f"  Created: {sk_example_path.relative_to(PROJECT_ROOT)}")

    sk_path = ow_dir / "secret_key.txt"
    with open(sk_path, "w", encoding="utf-8") as f:
        f.write("PLACEHOLDER_OPEN_WEBUI_SECRET_KEY_REPLACE_ME\n")
    print(f"  Sanitized: {sk_path.relative_to(PROJECT_ROOT)}")

    # 5. Sanitize config/timecapsule/config.json with generic portable paths
    cfg_path = PROJECT_ROOT / "config" / "timecapsule" / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(CONFIG_EXAMPLE_DATA, f, indent=2)
        print(f"  Sanitized: {cfg_path.relative_to(PROJECT_ROOT)}")

    # 6. Update .gitignore
    gitignore_path = PROJECT_ROOT / ".gitignore"
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(GITIGNORE_CONTENT)
    print(f"  Updated: {gitignore_path.relative_to(PROJECT_ROOT)}")


def untrack_secrets_from_git():
    """Removes sensitive credentials from git index if they were tracked."""
    print("[Sanitizer] Untracking sensitive credentials from git cache...")
    sensitive_targets = [
        "config/cloudflared/tunnel_token.txt",
        "config/cloudflared/tunnel_id.txt",
        "config/open-webui/secret_key.txt",
    ]
    for target in sensitive_targets:
        target_path = PROJECT_ROOT / target
        if target_path.exists():
            try:
                subprocess.run(
                    ["git", "rm", "--cached", target],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True
                )
                print(f"  Untracked from git: {target}")
            except Exception as e:
                print(f"  Notice: could not untrack {target}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sanitization & Production Branch Preparer for Personal AI Time Capsule"
    )
    parser.add_argument(
        "--untrack-secrets",
        action="store_true",
        help="Remove sensitive credential files from git cache"
    )
    args = parser.parse_args()

    print("========================================================")
    print(" Personal AI Time Capsule - Sanitization Pipeline")
    print("========================================================")

    create_sanitized_templates()

    if args.untrack_secrets:
        untrack_secrets_from_git()

    print("\nSanitization complete! Codebase is prepared for zero-leak public release.")


if __name__ == "__main__":
    main()
