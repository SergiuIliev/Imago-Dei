"""
Application Configuration and Provider Registry for Personal AI Time Capsule.
Enforces local-only execution, zero telemetry, and modular provider configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
from typing import Dict, Any, Optional


@dataclass
class StorageConfig:
    base_dir: str = "/Users/Sergiu/AI/Time_Capsule"
    data_dir: str = "/Users/Sergiu/AI/Time_Capsule/data/capsule"
    sqlite_db_path: str = "/Users/Sergiu/AI/Time_Capsule/data/capsule/archive.db"
    chroma_dir: str = "/Users/Sergiu/AI/Time_Capsule/data/capsule/vector_store"
    media_input_dir: str = "/Users/Sergiu/AI/Time_Capsule/vlogs"
    exports_dir: str = "/Users/Sergiu/AI/Time_Capsule/exports"
    audio_output_dir: str = "/Users/Sergiu/AI/Time_Capsule/data/capsule/audio_cache"
    models_dir: str = "/Users/Sergiu/AI/models"


@dataclass
class LLMConfig:
    provider: str = "ollama" # ollama, mlx, mock
    endpoint: str = "http://127.0.0.1:11434"
    model_name: str = "sergiu-time-capsule" # Fine-tuned autobiographical Ollama model
    fallback_model: str = "llama3.2:3b"
    critic_model: str = "qwen3:4b-heretic"
    mlx_endpoint: str = "http://127.0.0.1:8081"
    enable_jev_verification: bool = True
    temperature: float = 0.35
    context_window: int = 8192
    timeout_seconds: float = 45.0


@dataclass
class EmbeddingConfig:
    provider: str = "chroma_onnx" # chroma_onnx, sentence_transformers, mock
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class ASRConfig:
    provider: str = "pyav_whisper" # pyav_whisper, faster_whisper, mock
    model_size: str = "base" # tiny, base, small, medium, large-v3-turbo
    compute_type: str = "float32"
    device: str = "cpu" # cpu, mps


@dataclass
class TTSConfig:
    provider: str = "kokoro" # kokoro, macos_native, mock
    default_voice: str = "sergiu_voice_cloned" # Empirically calibrated neural voice clone
    sample_rate: int = 24000
    allow_reference_cloning: bool = True
    reference_audio_dir: str = "/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_references"


@dataclass
class AvatarConfig:
    provider: str = "liveportrait_stub" # liveportrait_stub, liveportrait, sadtalker
    target_fps: int = 25
    resolution: str = "512x512"
    reference_portrait_image: str = "" # Future path to high-res face portrait


@dataclass
class SecurityConfig:
    local_only: bool = True
    disable_telemetry: bool = True
    bind_host: str = "127.0.0.1"
    bind_port: int = 7860
    allow_remote_tunnels: bool = False


@dataclass
class CapsuleConfig:
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CapsuleConfig:
        config = cls()
        if "storage" in data:
            s_keys = {f for f in StorageConfig.__dataclass_fields__}
            config.storage = StorageConfig(**{k: v for k, v in data["storage"].items() if k in s_keys})
        if "llm" in data:
            l_keys = {f for f in LLMConfig.__dataclass_fields__}
            config.llm = LLMConfig(**{k: v for k, v in data["llm"].items() if k in l_keys})
        if "embedding" in data:
            e_keys = {f for f in EmbeddingConfig.__dataclass_fields__}
            config.embedding = EmbeddingConfig(**{k: v for k, v in data["embedding"].items() if k in e_keys})
        if "asr" in data:
            a_keys = {f for f in ASRConfig.__dataclass_fields__}
            config.asr = ASRConfig(**{k: v for k, v in data["asr"].items() if k in a_keys})
        if "tts" in data:
            t_keys = {f for f in TTSConfig.__dataclass_fields__}
            config.tts = TTSConfig(**{k: v for k, v in data["tts"].items() if k in t_keys})
        if "avatar" in data:
            av_keys = {f for f in AvatarConfig.__dataclass_fields__}
            config.avatar = AvatarConfig(**{k: v for k, v in data["avatar"].items() if k in av_keys})
        if "security" in data:
            sec_keys = {f for f in SecurityConfig.__dataclass_fields__}
            config.security = SecurityConfig(**{k: v for k, v in data["security"].items() if k in sec_keys})
        return config

    def save_to_file(self, config_path: str = "/Users/Sergiu/AI/config/timecapsule/config.json"):
        p = Path(config_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_or_create(cls, config_path: str = "/Users/Sergiu/AI/config/timecapsule/config.json") -> CapsuleConfig:
        p = Path(config_path)
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                print(f"Error loading config from {config_path}: {e}. Using defaults.")
        cfg = cls()
        cfg.save_to_file(config_path)
        return cfg


# Ensure root directories exist upon module load
def ensure_directories(config: CapsuleConfig):
    for dir_path in [
        config.storage.data_dir,
        config.storage.chroma_dir,
        config.storage.exports_dir,
        config.storage.audio_output_dir,
        config.tts.reference_audio_dir,
        config.storage.models_dir
    ]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
