"""
Application Configuration and Provider Registry for Personal AI Time Capsule.
Enforces local-only execution, zero telemetry, dynamic portable paths, and modular provider configuration.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import os
from typing import Dict, Any, Optional

# Project root dynamically resolved from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_VERSION = "v0.1.0"


def get_default_base_dir() -> str:
    return os.environ.get("TIMECAPSULE_BASE_DIR", str(PROJECT_ROOT))


def get_default_data_dir() -> str:
    base = Path(get_default_base_dir())
    return os.environ.get("TIMECAPSULE_DATA_DIR", str(base / "data" / "capsule"))


def get_default_sqlite_path() -> str:
    return os.environ.get(
        "TIMECAPSULE_SQLITE_PATH",
        str(Path(get_default_data_dir()) / "archive.db")
    )


def get_default_chroma_dir() -> str:
    return os.environ.get(
        "TIMECAPSULE_CHROMA_DIR",
        str(Path(get_default_data_dir()) / "vector_store")
    )


def get_default_media_input_dir() -> str:
    base = Path(get_default_base_dir())
    return os.environ.get("TIMECAPSULE_MEDIA_INPUT_DIR", str(base / "vlogs"))


def get_default_exports_dir() -> str:
    base = Path(get_default_base_dir())
    return os.environ.get("TIMECAPSULE_EXPORTS_DIR", str(base / "exports"))


def get_default_audio_output_dir() -> str:
    return os.environ.get(
        "TIMECAPSULE_AUDIO_OUTPUT_DIR",
        str(Path(get_default_data_dir()) / "audio_cache")
    )


def get_default_models_dir() -> str:
    return os.environ.get(
        "TIMECAPSULE_MODELS_DIR",
        str(PROJECT_ROOT / "models")
    )


def get_default_reference_audio_dir() -> str:
    return os.environ.get(
        "TIMECAPSULE_VOICE_REF_DIR",
        str(Path(get_default_data_dir()) / "voice_references")
    )


def get_default_config_path() -> str:
    return os.environ.get(
        "TIMECAPSULE_CONFIG",
        str(PROJECT_ROOT / "config" / "timecapsule" / "config.json")
    )


@dataclass
class StorageConfig:
    base_dir: str = field(default_factory=get_default_base_dir)
    data_dir: str = field(default_factory=get_default_data_dir)
    sqlite_db_path: str = field(default_factory=get_default_sqlite_path)
    chroma_dir: str = field(default_factory=get_default_chroma_dir)
    media_input_dir: str = field(default_factory=get_default_media_input_dir)
    exports_dir: str = field(default_factory=get_default_exports_dir)
    audio_output_dir: str = field(default_factory=get_default_audio_output_dir)
    models_dir: str = field(default_factory=get_default_models_dir)


@dataclass
class LLMConfig:
    provider: str = "ollama"  # ollama, mlx, mock
    endpoint: str = "http://127.0.0.1:11434"
    model_name: str = "sergiu-time-capsule:latest"  # Fine-tuned autobiographical Ollama model
    fallback_model: str = "sergiu-time-capsule:latest"
    critic_model: str = "qwen3:4b-heretic"
    mlx_endpoint: str = "http://127.0.0.1:8081"
    enable_jev_verification: bool = True
    temperature: float = 0.35
    context_window: int = 8192
    timeout_seconds: float = 45.0


@dataclass
class EmbeddingConfig:
    provider: str = "chroma_onnx"  # chroma_onnx, sentence_transformers, mock
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


@dataclass
class ASRConfig:
    provider: str = "pyav_whisper"  # pyav_whisper, faster_whisper, mock
    model_size: str = "base"  # tiny, base, small, medium, large-v3-turbo
    compute_type: str = "float32"
    device: str = "cpu"  # cpu, mps


@dataclass
class TTSConfig:
    provider: str = "kokoro"  # kokoro, macos_native, mock
    default_voice: str = "sergiu_voice_cloned"  # Empirically calibrated neural voice clone
    sample_rate: int = 24000
    allow_reference_cloning: bool = True
    reference_audio_dir: str = field(default_factory=get_default_reference_audio_dir)


@dataclass
class AvatarConfig:
    provider: str = "liveportrait_stub"  # liveportrait_stub, liveportrait, sadtalker
    target_fps: int = 25
    resolution: str = "512x512"
    reference_portrait_image: str = ""  # Future path to high-res face portrait


@dataclass
class SecurityConfig:
    local_only: bool = True
    disable_telemetry: bool = True
    bind_host: str = "127.0.0.1"
    bind_port: int = 7860
    allow_remote_tunnels: bool = False
    internet_rag_mode: str = "airgapped"  # "airgapped", "jev_controlled", "open_internet"


@dataclass
class RetrievalConfig:
    internet_rag_mode: str = "airgapped"  # "airgapped", "jev_controlled", "open_internet"
    top_k_memories: int = 5
    top_k_beliefs: int = 3
    similarity_threshold: float = 0.65


@dataclass
class CapsuleConfig:
    app_version: str = APP_VERSION
    storage: StorageConfig = field(default_factory=StorageConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @property
    def internet_rag_mode(self) -> str:
        return self.retrieval.internet_rag_mode

    @internet_rag_mode.setter
    def internet_rag_mode(self, val: str):
        if val not in ["airgapped", "jev_controlled", "open_internet"]:
            raise ValueError(f"Invalid internet_rag_mode '{val}'. Must be airgapped, jev_controlled, or open_internet.")
        self.retrieval.internet_rag_mode = val
        self.security.internet_rag_mode = val

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["internet_rag_mode"] = self.internet_rag_mode
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CapsuleConfig:
        config = cls()
        if "app_version" in data:
            config.app_version = data["app_version"]
        elif "version" in data:
            config.app_version = data["version"]

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
        if "retrieval" in data:
            r_keys = {f for f in RetrievalConfig.__dataclass_fields__}
            config.retrieval = RetrievalConfig(**{k: v for k, v in data["retrieval"].items() if k in r_keys})

        # Ensure security and retrieval stay in sync for internet_rag_mode
        if "internet_rag_mode" in data:
            config.internet_rag_mode = data["internet_rag_mode"]
        elif "security" in data and "internet_rag_mode" in data["security"]:
            config.retrieval.internet_rag_mode = data["security"]["internet_rag_mode"]
            config.security.internet_rag_mode = data["security"]["internet_rag_mode"]
        elif "retrieval" in data and "internet_rag_mode" in data["retrieval"]:
            config.security.internet_rag_mode = data["retrieval"]["internet_rag_mode"]
            config.retrieval.internet_rag_mode = data["retrieval"]["internet_rag_mode"]

        return config

    def save_to_file(self, config_path: Optional[str] = None):
        target = Path(config_path) if config_path else Path(get_default_config_path())
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_or_create(cls, config_path: Optional[str] = None) -> CapsuleConfig:
        target = Path(config_path) if config_path else Path(get_default_config_path())
        if target.exists():
            try:
                with open(target, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls.from_dict(data)
            except Exception as e:
                print(f"Error loading config from {target}: {e}. Using defaults.")
        cfg = cls()
        try:
            cfg.save_to_file(str(target))
        except Exception:
            pass
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
