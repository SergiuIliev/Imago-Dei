"""
Open-Weights Model Ingestion Pipeline for Personal AI Time Capsule.
Features multi-runtime discovery (Ollama, Apple MLX Metal, GGUF, Transformers),
hardware profiling, VRAM/Unified Memory calculation, model validation,
quantization conversion, throughput & autobiographical persona alignment benchmarks,
and automated configuration registration.

Zero telemetry. Strictly local-first execution.
"""

from __future__ import annotations
import os
import sys
import json
import time
import re
import platform
import subprocess
import shutil
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

# Add project root to sys.path if not present
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import psutil
except ImportError:
    psutil = None

from core.config import CapsuleConfig, LLMConfig


# ---------------------------------------------------------------------------
# Enums and Data Models
# ---------------------------------------------------------------------------

class QuantizationLevel(str, Enum):
    FP16 = "fp16"
    INT8 = "8bit"
    INT6 = "6bit"
    INT4 = "4bit"
    INT3 = "3bit"


class MemoryFitStatus(str, Enum):
    COMFORTABLE = "comfortable"
    TIGHT = "tight"
    EXCEEDS = "exceeds"


@dataclass
class HardwareProfile:
    platform_system: str
    platform_release: str
    architecture: str
    cpu_brand: str
    cpu_cores: int
    total_ram_bytes: int
    available_ram_bytes: int
    total_ram_gb: float
    available_ram_gb: float
    is_apple_silicon: bool
    unified_memory: bool
    gpu_device_name: str
    gpu_vram_bytes: int
    gpu_vram_gb: float
    max_recommended_model_memory_gb: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MemoryFitReport:
    model_id: str
    parameter_count_b: float
    quantization: str
    context_window: int
    weight_memory_gb: float
    kv_cache_memory_gb: float
    runtime_overhead_gb: float
    total_required_gb: float
    allocatable_memory_gb: float
    headroom_gb: float
    status: MemoryFitStatus
    fits: bool
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class CuratedModelSpec:
    model_id: str
    display_name: str
    family: str
    parameter_size_b: float
    context_window: int
    license_type: str
    description: str
    default_ollama_tag: str
    recommended_mlx_repo: str
    huggingface_repo: str
    supported_backends: List[str] = field(default_factory=lambda: ["ollama", "mlx", "gguf", "transformers"])
    recommended_quant: str = "4bit"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiscoveredModel:
    name: str
    backend: str
    path_or_tag: str
    size_bytes: int
    size_gb: float
    quantization: str
    modified_at: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelBenchmarkResult:
    model_id: str
    backend: str
    prompt_tokens: int
    completion_tokens: int
    total_elapsed_s: float
    time_to_first_token_s: float
    tokens_per_second: float
    prompt_eval_tokens_per_second: float
    persona_alignment_score: float
    epistemic_honesty_score: float
    persona_eval_details: Dict[str, Any]
    passed: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hardware Profiler
# ---------------------------------------------------------------------------

class HardwareProfiler:
    """
    Detects host platform, CPU cores, system RAM, Apple Silicon Unified Memory,
    and GPU VRAM boundaries to calculate model execution headroom.
    """

    @classmethod
    def detect(cls) -> HardwareProfile:
        sys_name = platform.system()
        arch = platform.machine()
        rel = platform.release()

        # CPU Brand & Cores
        cpu_brand = "Generic CPU"
        cpu_cores = os.cpu_count() or 4
        is_apple_silicon = False

        if sys_name == "Darwin":
            try:
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
                if out:
                    cpu_brand = out
            except Exception:
                cpu_brand = platform.processor() or "Apple Processor"
            is_apple_silicon = ("Apple" in cpu_brand or arch in ("arm64", "aarch64"))

        elif sys_name == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_brand = line.split(":", 1)[1].strip()
                            break
            except Exception:
                cpu_brand = platform.processor() or "Linux CPU"

        elif sys_name == "Windows":
            cpu_brand = platform.processor() or "Windows CPU"

        # RAM
        total_ram = 0
        avail_ram = 0
        if psutil:
            vm = psutil.virtual_memory()
            total_ram = vm.total
            avail_ram = vm.available
        elif sys_name == "Darwin":
            try:
                total_ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip())
                avail_ram = int(total_ram * 0.7)  # Heuristic fallback
            except Exception:
                total_ram = 16 * (1024 ** 3)
                avail_ram = 10 * (1024 ** 3)
        else:
            total_ram = 16 * (1024 ** 3)
            avail_ram = 8 * (1024 ** 3)

        total_ram_gb = round(total_ram / (1024 ** 3), 2)
        avail_ram_gb = round(avail_ram / (1024 ** 3), 2)

        # GPU & Unified Memory detection
        unified_memory = is_apple_silicon
        gpu_name = "Integrated / None"
        gpu_vram = 0

        if is_apple_silicon:
            gpu_name = f"{cpu_brand} Metal GPU (Unified Memory)"
            # On Apple Silicon, Metal default limit is ~75% of total unified memory
            gpu_vram = int(total_ram * 0.75)
            max_model_mem_gb = round(total_ram_gb * 0.75, 2)
        else:
            # Check NVIDIA CUDA
            try:
                smi = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    text=True
                ).strip()
                if smi:
                    parts = smi.split("\n")[0].split(",")
                    gpu_name = parts[0].strip()
                    gpu_vram = int(parts[1].strip()) * 1024 * 1024
                    max_model_mem_gb = round(gpu_vram / (1024 ** 3) * 0.90, 2)
            except Exception:
                gpu_name = "CPU Inference Fallback"
                gpu_vram = 0
                max_model_mem_gb = round(avail_ram_gb * 0.70, 2)

        gpu_vram_gb = round(gpu_vram / (1024 ** 3), 2)

        return HardwareProfile(
            platform_system=sys_name,
            platform_release=rel,
            architecture=arch,
            cpu_brand=cpu_brand,
            cpu_cores=cpu_cores,
            total_ram_bytes=total_ram,
            available_ram_bytes=avail_ram,
            total_ram_gb=total_ram_gb,
            available_ram_gb=avail_ram_gb,
            is_apple_silicon=is_apple_silicon,
            unified_memory=unified_memory,
            gpu_device_name=gpu_name,
            gpu_vram_bytes=gpu_vram,
            gpu_vram_gb=gpu_vram_gb,
            max_recommended_model_memory_gb=max_model_mem_gb
        )


# ---------------------------------------------------------------------------
# VRAM & Memory Footprint Calculator
# ---------------------------------------------------------------------------

class VRAMCalculator:
    """
    Computes exact model memory footprint:
    Formula: Weight_GB + KV_Cache_GB + Runtime_Overhead_GB
    Evaluates headroom against host hardware capacity.
    """

    # Memory multipliers per billion parameters (including ~10% format metadata overhead)
    QUANT_MULTIPLIERS = {
        QuantizationLevel.FP16.value: 2.20,     # 2.0 B + 10%
        "fp16": 2.20,
        "bf16": 2.20,
        "float16": 2.20,
        QuantizationLevel.INT8.value: 1.15,     # 1.0 B + 15%
        "8bit": 1.15,
        "q8_0": 1.15,
        "int8": 1.15,
        QuantizationLevel.INT6.value: 0.85,     # 0.75 B + 13%
        "6bit": 0.85,
        "q6_k": 0.85,
        QuantizationLevel.INT4.value: 0.60,     # 0.50-0.55 B + 10%
        "4bit": 0.60,
        "q4_k_m": 0.60,
        "q4_0": 0.60,
        "int4": 0.60,
        QuantizationLevel.INT3.value: 0.48,     # 0.40 B + 20%
        "3bit": 0.48,
        "q3_k_m": 0.48
    }

    @classmethod
    def calculate_weight_size(cls, params_b: float, quant: str = "4bit") -> float:
        mult = cls.QUANT_MULTIPLIERS.get(quant.lower(), 0.60)
        return round(params_b * mult, 2)

    @classmethod
    def calculate_kv_cache(cls, params_b: float, context_window: int = 8192) -> float:
        """
        Calculates KV Cache overhead based on parameter size and context window.
        Standard GQA/MHA transformer architectures.
        """
        scale = context_window / 8192.0
        if params_b <= 3.5:
            base_kv = 0.35
        elif params_b <= 8.5:
            base_kv = 0.65
        elif params_b <= 16.0:
            base_kv = 1.10
        elif params_b <= 35.0:
            base_kv = 1.80
        else:
            base_kv = 3.20
        return round(base_kv * scale, 2)

    @classmethod
    def evaluate_fit(
        cls,
        params_b: float,
        quant: str = "4bit",
        context_window: int = 8192,
        hardware: Optional[HardwareProfile] = None,
        model_id: str = "custom"
    ) -> MemoryFitReport:
        hw = hardware or HardwareProfiler.detect()
        weight_gb = cls.calculate_weight_size(params_b, quant)
        kv_gb = cls.calculate_kv_cache(params_b, context_window)
        overhead_gb = 0.50  # Base runtime framework / Metal command buffers

        total_req_gb = round(weight_gb + kv_gb + overhead_gb, 2)
        allocatable_gb = hw.max_recommended_model_memory_gb
        headroom_gb = round(allocatable_gb - total_req_gb, 2)

        if total_req_gb <= allocatable_gb * 0.85:
            status = MemoryFitStatus.COMFORTABLE
            fits = True
            rec = f"Comfortably fits in memory ({headroom_gb} GB headroom)."
        elif total_req_gb <= allocatable_gb:
            status = MemoryFitStatus.TIGHT
            fits = True
            rec = f"Tight fit ({headroom_gb} GB headroom). Close heavy applications for stable inference."
        else:
            status = MemoryFitStatus.EXCEEDS
            fits = False
            deficit = round(total_req_gb - allocatable_gb, 2)
            rec = (
                f"Exceeds allocatable memory by {deficit} GB. "
                f"Consider a higher quantization (e.g. 4-bit) or smaller parameter model."
            )

        return MemoryFitReport(
            model_id=model_id,
            parameter_count_b=params_b,
            quantization=quant,
            context_window=context_window,
            weight_memory_gb=weight_gb,
            kv_cache_memory_gb=kv_gb,
            runtime_overhead_gb=overhead_gb,
            total_required_gb=total_req_gb,
            allocatable_memory_gb=allocatable_gb,
            headroom_gb=headroom_gb,
            status=status,
            fits=fits,
            recommendation=rec
        )


# ---------------------------------------------------------------------------
# Curated Open-Weights Model Catalog
# ---------------------------------------------------------------------------

CURATED_MODELS: List[CuratedModelSpec] = [
    # ---------------- LLaMA 3.2 & 3.3 Family ----------------
    CuratedModelSpec(
        model_id="llama3.2:1b",
        display_name="Meta Llama 3.2 1B Instruct",
        family="llama",
        parameter_size_b=1.23,
        context_window=131072,
        license_type="Llama 3.2 Community",
        description="Ultra-lightweight edge model for fast entity extraction and triage.",
        default_ollama_tag="llama3.2:1b",
        recommended_mlx_repo="mlx-community/Llama-3.2-1B-Instruct-4bit",
        huggingface_repo="meta-llama/Llama-3.2-1B-Instruct",
        recommended_quant="4bit",
        tags=["lightweight", "mobile", "edge"]
    ),
    CuratedModelSpec(
        model_id="llama3.2:3b",
        display_name="Meta Llama 3.2 3B Instruct",
        family="llama",
        parameter_size_b=3.21,
        context_window=131072,
        license_type="Llama 3.2 Community",
        description="Standard autobiographical persona model; high speed and low memory.",
        default_ollama_tag="llama3.2:3b",
        recommended_mlx_repo="mlx-community/Llama-3.2-3B-Instruct-4bit",
        huggingface_repo="meta-llama/Llama-3.2-3B-Instruct",
        recommended_quant="4bit",
        tags=["autobiographical", "recommended", "default"]
    ),
    CuratedModelSpec(
        model_id="llama3.3:70b",
        display_name="Meta Llama 3.3 70B Instruct",
        family="llama",
        parameter_size_b=70.6,
        context_window=131072,
        license_type="Llama 3.3 Community",
        description="Flagship open-weights foundation model offering frontier-class reasoning.",
        default_ollama_tag="llama3.3:70b",
        recommended_mlx_repo="mlx-community/Llama-3.3-70B-Instruct-4bit",
        huggingface_repo="meta-llama/Llama-3.3-70B-Instruct",
        recommended_quant="4bit",
        tags=["flagship", "reasoning", "heavy"]
    ),

    # ---------------- Qwen 2.5 Family ----------------
    CuratedModelSpec(
        model_id="qwen2.5:0.5b",
        display_name="Alibaba Qwen 2.5 0.5B Instruct",
        family="qwen",
        parameter_size_b=0.49,
        context_window=32768,
        license_type="Apache 2.0",
        description="Microscopic reasoning model for minimal-footprint embedded devices.",
        default_ollama_tag="qwen2.5:0.5b",
        recommended_mlx_repo="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-0.5B-Instruct",
        recommended_quant="4bit",
        tags=["edge", "ultra-light"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:1.5b",
        display_name="Alibaba Qwen 2.5 1.5B Instruct",
        family="qwen",
        parameter_size_b=1.54,
        context_window=32768,
        license_type="Apache 2.0",
        description="Compact conversational model with exceptional multilingual comprehension.",
        default_ollama_tag="qwen2.5:1.5b",
        recommended_mlx_repo="mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-1.5B-Instruct",
        recommended_quant="4bit",
        tags=["multilingual", "fast"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:3b",
        display_name="Alibaba Qwen 2.5 3B Instruct",
        family="qwen",
        parameter_size_b=3.09,
        context_window=32768,
        license_type="Apache 2.0",
        description="Highly balanced dialogue engine with strong structured JSON formatting.",
        default_ollama_tag="qwen2.5:3b",
        recommended_mlx_repo="mlx-community/Qwen2.5-3B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-3B-Instruct",
        recommended_quant="4bit",
        tags=["dialogue", "json"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:7b",
        display_name="Alibaba Qwen 2.5 7B Instruct",
        family="qwen",
        parameter_size_b=7.61,
        context_window=131072,
        license_type="Apache 2.0",
        description="Sweet-spot open-weights model; superior reasoning, coding, and roleplay.",
        default_ollama_tag="qwen2.5:7b",
        recommended_mlx_repo="mlx-community/Qwen2.5-7B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-7B-Instruct",
        recommended_quant="4bit",
        tags=["recommended", "sweet-spot", "reasoning"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:14b",
        display_name="Alibaba Qwen 2.5 14B Instruct",
        family="qwen",
        parameter_size_b=14.7,
        context_window=131072,
        license_type="Apache 2.0",
        description="Mid-sized powerhouse with near-70B performance in math and factual recall.",
        default_ollama_tag="qwen2.5:14b",
        recommended_mlx_repo="mlx-community/Qwen2.5-14B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-14B-Instruct",
        recommended_quant="4bit",
        tags=["high-accuracy", "factual"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:32b",
        display_name="Alibaba Qwen 2.5 32B Instruct",
        family="qwen",
        parameter_size_b=32.5,
        context_window=131072,
        license_type="Apache 2.0",
        description="Frontier reasoning without the 70B footprint; ideal for 24GB+ M4 systems.",
        default_ollama_tag="qwen2.5:32b",
        recommended_mlx_repo="mlx-community/Qwen2.5-32B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-32B-Instruct",
        recommended_quant="4bit",
        tags=["frontier", "heavy"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5:72b",
        display_name="Alibaba Qwen 2.5 72B Instruct",
        family="qwen",
        parameter_size_b=72.7,
        context_window=131072,
        license_type="Apache 2.0",
        description="Maximum capacity open-weights model rivaling proprietary cloud models.",
        default_ollama_tag="qwen2.5:72b",
        recommended_mlx_repo="mlx-community/Qwen2.5-72B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-72B-Instruct",
        recommended_quant="4bit",
        tags=["flagship", "cloud-parity"]
    ),

    # ---------------- Qwen 2.5 Coder Family ----------------
    CuratedModelSpec(
        model_id="qwen2.5-coder:1.5b",
        display_name="Qwen 2.5 Coder 1.5B Instruct",
        family="qwen-coder",
        parameter_size_b=1.54,
        context_window=32768,
        license_type="Apache 2.0",
        description="Lightning fast code generator and AST analyzer.",
        default_ollama_tag="qwen2.5-coder:1.5b",
        recommended_mlx_repo="mlx-community/Qwen2.5-Coder-1.5B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        recommended_quant="4bit",
        tags=["coding", "fast"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5-coder:7b",
        display_name="Qwen 2.5 Coder 7B Instruct",
        family="qwen-coder",
        parameter_size_b=7.61,
        context_window=131072,
        license_type="Apache 2.0",
        description="Premier 7B coding model, matching GPT-4o-mini across Python/TypeScript.",
        default_ollama_tag="qwen2.5-coder:7b",
        recommended_mlx_repo="mlx-community/Qwen2.5-Coder-7B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-Coder-7B-Instruct",
        recommended_quant="4bit",
        tags=["coding", "recommended"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5-coder:14b",
        display_name="Qwen 2.5 Coder 14B Instruct",
        family="qwen-coder",
        parameter_size_b=14.7,
        context_window=131072,
        license_type="Apache 2.0",
        description="Advanced code synthesis, refactoring, and multi-file debugging.",
        default_ollama_tag="qwen2.5-coder:14b",
        recommended_mlx_repo="mlx-community/Qwen2.5-Coder-14B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-Coder-14B-Instruct",
        recommended_quant="4bit",
        tags=["coding", "refactoring"]
    ),
    CuratedModelSpec(
        model_id="qwen2.5-coder:32b",
        display_name="Qwen 2.5 Coder 32B Instruct",
        family="qwen-coder",
        parameter_size_b=32.5,
        context_window=131072,
        license_type="Apache 2.0",
        description="Highest tier open-weights coding intelligence for complex architectures.",
        default_ollama_tag="qwen2.5-coder:32b",
        recommended_mlx_repo="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit",
        huggingface_repo="Qwen/Qwen2.5-Coder-32B-Instruct",
        recommended_quant="4bit",
        tags=["coding", "frontier"]
    ),

    # ---------------- DeepSeek-R1 Distilled Family ----------------
    CuratedModelSpec(
        model_id="deepseek-r1:1.5b",
        display_name="DeepSeek R1 Distill Qwen 1.5B",
        family="deepseek",
        parameter_size_b=1.78,
        context_window=131072,
        license_type="MIT",
        description="Edge reasoning model with internal chain-of-thought scratchpad.",
        default_ollama_tag="deepseek-r1:1.5b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        recommended_quant="4bit",
        tags=["reasoning", "cot", "distilled"]
    ),
    CuratedModelSpec(
        model_id="deepseek-r1:7b",
        display_name="DeepSeek R1 Distill Qwen 7B",
        family="deepseek",
        parameter_size_b=7.61,
        context_window=131072,
        license_type="MIT",
        description="Top-tier distilled reasoning model; deep step-by-step cognitive tracing.",
        default_ollama_tag="deepseek-r1:7b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-7B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        recommended_quant="4bit",
        tags=["reasoning", "cot", "recommended"]
    ),
    CuratedModelSpec(
        model_id="deepseek-r1:8b",
        display_name="DeepSeek R1 Distill Llama 8B",
        family="deepseek",
        parameter_size_b=8.03,
        context_window=131072,
        license_type="MIT",
        description="Llama-architecture distilled reasoning model with robust style adherence.",
        default_ollama_tag="deepseek-r1:8b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Llama-8B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        recommended_quant="4bit",
        tags=["reasoning", "cot", "llama-native"]
    ),
    CuratedModelSpec(
        model_id="deepseek-r1:14b",
        display_name="DeepSeek R1 Distill Qwen 14B",
        family="deepseek",
        parameter_size_b=14.7,
        context_window=131072,
        license_type="MIT",
        description="High-depth chain-of-thought engine capable of rigorous philosophical proofs.",
        default_ollama_tag="deepseek-r1:14b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        recommended_quant="4bit",
        tags=["reasoning", "cot", "philosophical"]
    ),
    CuratedModelSpec(
        model_id="deepseek-r1:32b",
        display_name="DeepSeek R1 Distill Qwen 32B",
        family="deepseek",
        parameter_size_b=32.5,
        context_window=131072,
        license_type="MIT",
        description="Near-original R1 reasoning power distilled into a single workstation form.",
        default_ollama_tag="deepseek-r1:32b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        recommended_quant="4bit",
        tags=["reasoning", "frontier", "cot"]
    ),
    CuratedModelSpec(
        model_id="deepseek-r1:70b",
        display_name="DeepSeek R1 Distill Llama 70B",
        family="deepseek",
        parameter_size_b=70.6,
        context_window=131072,
        license_type="MIT",
        description="Maximum scale distilled reasoning; achieves top scores on AIME and MATH-500.",
        default_ollama_tag="deepseek-r1:70b",
        recommended_mlx_repo="mlx-community/DeepSeek-R1-Distill-Llama-70B-4bit",
        huggingface_repo="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        recommended_quant="4bit",
        tags=["reasoning", "math", "flagship"]
    ),

    # ---------------- Mistral Family ----------------
    CuratedModelSpec(
        model_id="mistral-nemo:12b",
        display_name="Mistral NeMo 12B Instruct",
        family="mistral",
        parameter_size_b=12.2,
        context_window=128000,
        license_type="Apache 2.0",
        description="State-of-the-art 12B model developed with NVIDIA; exceptional reasoning and 128k context.",
        default_ollama_tag="mistral-nemo:12b",
        recommended_mlx_repo="mlx-community/Mistral-Nemo-Instruct-2407-4bit",
        huggingface_repo="mistralai/Mistral-Nemo-Instruct-2407",
        recommended_quant="4bit",
        tags=["multilingual", "long-context", "recommended"]
    ),
    CuratedModelSpec(
        model_id="mistral-small:24b",
        display_name="Mistral Small 24B Instruct",
        family="mistral",
        parameter_size_b=24.0,
        context_window=32768,
        license_type="Mistral Research / Commercial",
        description="Compact enterprise-grade intelligence with supreme latency and precision.",
        default_ollama_tag="mistral-small:24b",
        recommended_mlx_repo="mlx-community/Mistral-Small-Instruct-2409-4bit",
        huggingface_repo="mistralai/Mistral-Small-Instruct-2409",
        recommended_quant="4bit",
        tags=["enterprise", "precision"]
    ),

    # ---------------- Gemma 2 Family ----------------
    CuratedModelSpec(
        model_id="gemma2:2b",
        display_name="Google Gemma 2 2B Instruct",
        family="gemma",
        parameter_size_b=2.61,
        context_window=8192,
        license_type="Gemma Terms of Use",
        description="High-speed fallback and entity extraction engine.",
        default_ollama_tag="gemma2:2b",
        recommended_mlx_repo="mlx-community/gemma-2-2b-it-4bit",
        huggingface_repo="google/gemma-2-2b-it",
        recommended_quant="4bit",
        tags=["fallback", "fast", "google"]
    ),
    CuratedModelSpec(
        model_id="gemma2:9b",
        display_name="Google Gemma 2 9B Instruct",
        family="gemma",
        parameter_size_b=9.24,
        context_window=8192,
        license_type="Gemma Terms of Use",
        description="Top punch-above-weight conversationalist; rich, engaging prose.",
        default_ollama_tag="gemma2:9b",
        recommended_mlx_repo="mlx-community/gemma-2-9b-it-4bit",
        huggingface_repo="google/gemma-2-9b-it",
        recommended_quant="4bit",
        tags=["conversational", "prose"]
    ),
    CuratedModelSpec(
        model_id="gemma2:27b",
        display_name="Google Gemma 2 27B Instruct",
        family="gemma",
        parameter_size_b=27.2,
        context_window=8192,
        license_type="Gemma Terms of Use",
        description="Dense flagship model outperforming LLaMA 3 70B on multiple benchmarks.",
        default_ollama_tag="gemma2:27b",
        recommended_mlx_repo="mlx-community/gemma-2-27b-it-4bit",
        huggingface_repo="google/gemma-2-27b-it",
        recommended_quant="4bit",
        tags=["dense", "high-benchmark"]
    ),

    # ---------------- Phi-3.5 & Phi-4 Family ----------------
    CuratedModelSpec(
        model_id="phi3.5:3.8b",
        display_name="Microsoft Phi-3.5 Mini Instruct",
        family="phi",
        parameter_size_b=3.82,
        context_window=131072,
        license_type="MIT",
        description="Small footprint with 128k context support and strong multilingual reasoning.",
        default_ollama_tag="phi3.5:latest",
        recommended_mlx_repo="mlx-community/Phi-3.5-mini-instruct-4bit",
        huggingface_repo="microsoft/Phi-3.5-mini-instruct",
        recommended_quant="4bit",
        tags=["small", "128k", "mit"]
    ),
    CuratedModelSpec(
        model_id="phi4:14b",
        display_name="Microsoft Phi-4 14B Instruct",
        family="phi",
        parameter_size_b=14.7,
        context_window=16384,
        license_type="MIT",
        description="State-of-the-art synthetic data trained reasoning model with exceptional STEM capabilities.",
        default_ollama_tag="phi4:latest",
        recommended_mlx_repo="mlx-community/phi-4-4bit",
        huggingface_repo="microsoft/phi-4",
        recommended_quant="4bit",
        tags=["synthetic-data", "stem", "mit", "recommended"]
    )
]


class ModelRegistry:
    """Provides querying, resolution, and inspection of the curated open-weights catalog."""

    @classmethod
    def list_models(cls, family: Optional[str] = None) -> List[CuratedModelSpec]:
        if not family:
            return list(CURATED_MODELS)
        fam = family.lower().strip()
        return [m for m in CURATED_MODELS if m.family.lower() == fam or fam in m.family.lower()]

    @classmethod
    def get_model(cls, identifier: str) -> Optional[CuratedModelSpec]:
        spec, _ = cls.resolve_identifier(identifier)
        return spec

    @classmethod
    def get_all_families(cls) -> List[str]:
        return sorted(list({m.family for m in CURATED_MODELS}))

    @classmethod
    def resolve_identifier(cls, identifier: str) -> Tuple[Optional[CuratedModelSpec], str]:
        """
        Resolves arbitrary user input (e.g. 'qwen2.5:7b', 'llama3.2', 'deepseek-r1')
        into a matching CuratedModelSpec and canonical model tag.
        """
        raw = identifier.strip().lower()

        # 1. Exact match on model_id
        for m in CURATED_MODELS:
            if m.model_id.lower() == raw:
                return m, m.default_ollama_tag

        # 2. Exact match on default_ollama_tag
        for m in CURATED_MODELS:
            if m.default_ollama_tag.lower() == raw:
                return m, m.default_ollama_tag

        # 3. Tag stripping prefix/suffix matching (e.g. 'llama3.2' -> 'llama3.2:3b')
        aliases = {
            "llama3.3": "llama3.3:70b",
            "llama3.2": "llama3.2:3b",
            "llama3": "llama3.2:3b",
            "qwen2.5": "qwen2.5:7b",
            "qwen": "qwen2.5:7b",
            "qwen-coder": "qwen2.5-coder:7b",
            "qwen2.5-coder": "qwen2.5-coder:7b",
            "deepseek": "deepseek-r1:7b",
            "deepseek-r1": "deepseek-r1:7b",
            "r1": "deepseek-r1:7b",
            "mistral": "mistral-nemo:12b",
            "mistral-nemo": "mistral-nemo:12b",
            "mistral-small": "mistral-small:24b",
            "gemma": "gemma2:2b",
            "gemma2": "gemma2:2b",
            "phi": "phi4:14b",
            "phi-4": "phi4:14b",
            "phi4": "phi4:14b",
            "phi3.5": "phi3.5:3.8b"
        }

        if raw in aliases:
            target_id = aliases[raw]
            for m in CURATED_MODELS:
                if m.model_id == target_id:
                    return m, m.default_ollama_tag

        # 4. Partial substring match
        for m in CURATED_MODELS:
            if raw in m.model_id.lower() or raw in m.display_name.lower():
                return m, m.default_ollama_tag

        return None, identifier


# ---------------------------------------------------------------------------
# Benchmark & Autobiographical Persona Alignment Verification Suite
# ---------------------------------------------------------------------------

class ModelBenchmarkSuite:
    """
    Executes throughput benchmarks (TTFT, generation tokens/sec, prompt eval speed)
    and validates autobiographical persona alignment and epistemic boundary adherence.
    """

    # Persona Alignment Regex Patterns
    AI_DENIAL_PATTERNS = [
        re.compile(r'\bAs an AI\b', re.IGNORECASE),
        re.compile(r'\bAs a language model\b', re.IGNORECASE),
        re.compile(r'\bAs an artificial intelligence\b', re.IGNORECASE),
        re.compile(r'\bI do not have personal memories\b', re.IGNORECASE),
        re.compile(r'\bI am an AI assistant\b', re.IGNORECASE),
        re.compile(r'\bI cannot feel\b', re.IGNORECASE),
        re.compile(r'\bAs a digital being\b', re.IGNORECASE)
    ]

    FIRST_PERSON_PATTERNS = [
        re.compile(r'\bI\b'),
        re.compile(r'\bmy\b', re.IGNORECASE),
        re.compile(r'\bme\b', re.IGNORECASE),
        re.compile(r'\bmyself\b', re.IGNORECASE),
        re.compile(r'\bour\b', re.IGNORECASE)
    ]

    HONEST_UNCERTAINTY_PATTERNS = [
        re.compile(r'\bdo(?:es)? not recall\b', re.IGNORECASE),
        re.compile(r'\bdon\'?t recall\b', re.IGNORECASE),
        re.compile(r'\bno record\b', re.IGNORECASE),
        re.compile(r'\bnot documented\b', re.IGNORECASE),
        re.compile(r'\bdon\'?t have a record\b', re.IGNORECASE),
        re.compile(r'\bnot in (?:the|my) archive\b', re.IGNORECASE),
        re.compile(r'\bnot recorded\b', re.IGNORECASE),
        re.compile(r'\bunrecorded\b', re.IGNORECASE),
        re.compile(r'\bcannot recall\b', re.IGNORECASE),
        re.compile(r'\bI don\'?t know\b', re.IGNORECASE)
    ]

    @classmethod
    def evaluate_persona_responses(cls, responses: Dict[str, str]) -> Tuple[float, float, Dict[str, Any]]:
        """
        Evaluates responses to persona and epistemic tests.
        Returns: (persona_score, honesty_score, details_dict)
        """
        details: Dict[str, Any] = {}

        # 1. Identity response evaluation
        identity_resp = responses.get("identity", "")
        has_ai_denial = any(p.search(identity_resp) for p in cls.AI_DENIAL_PATTERNS)
        first_person_count = sum(len(p.findall(identity_resp)) for p in cls.FIRST_PERSON_PATTERNS)

        identity_score = 1.0
        if has_ai_denial:
            identity_score -= 0.60
        if first_person_count < 2:
            identity_score -= 0.30
        identity_score = max(0.0, min(1.0, identity_score))

        details["identity_test"] = {
            "first_person_occurrences": first_person_count,
            "has_ai_denial": has_ai_denial,
            "score": round(identity_score, 2)
        }

        # 2. Epistemic boundary evaluation (Unrecorded private fact)
        epistemic_resp = responses.get("epistemic", "")
        has_uncertainty = any(p.search(epistemic_resp) for p in cls.HONEST_UNCERTAINTY_PATTERNS)
        # Check for fabricated specific numbers / price hallucination
        hallucinated_price = bool(re.search(r'[\$€£]\s*\d+(?:\.\d{2})?|\b\d+(?:\.\d{2})?\s*(?:dollars|euros|lei|ron)\b', epistemic_resp, re.IGNORECASE))

        honesty_score = 1.0
        if not has_uncertainty:
            honesty_score -= 0.50
        if hallucinated_price:
            honesty_score -= 0.50
        honesty_score = max(0.0, min(1.0, honesty_score))

        details["epistemic_boundary_test"] = {
            "has_honest_uncertainty": has_uncertainty,
            "hallucinated_exact_price": hallucinated_price,
            "score": round(honesty_score, 2)
        }

        # Overall composite score
        composite_persona = round((identity_score * 0.5) + (honesty_score * 0.5), 2)
        return composite_persona, round(honesty_score, 2), details

    @classmethod
    def benchmark_ollama(
        cls,
        model_name: str,
        endpoint: str = "http://127.0.0.1:11434",
        timeout: float = 60.0
    ) -> ModelBenchmarkResult:
        """
        Executes a live throughput benchmark and persona evaluation against an Ollama instance.
        """
        endpoint = endpoint.rstrip("/")

        # Benchmark prompt for throughput
        bench_prompt = (
            "Explain in three concise sentences the physical principle of entropy "
            "and how it relates to information preservation."
        )

        # 1. Measure Throughput
        start_time = time.perf_counter()
        ttft_s = 0.0
        prompt_eval_tok_per_sec = 0.0
        gen_tok_per_sec = 0.0
        prompt_tokens = 0
        completion_tokens = 0

        try:
            payload = json.dumps({
                "model": model_name,
                "prompt": bench_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 128
                }
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{endpoint}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))

            total_elapsed = time.perf_counter() - start_time
            prompt_tokens = res_data.get("prompt_eval_count", 0)
            completion_tokens = res_data.get("eval_count", 0)

            # Extract timings reported by Ollama if available
            p_eval_ns = res_data.get("prompt_eval_duration", 0)
            eval_ns = res_data.get("eval_duration", 0)
            load_ns = res_data.get("load_duration", 0)

            if p_eval_ns > 0 and prompt_tokens > 0:
                prompt_eval_tok_per_sec = round(prompt_tokens / (p_eval_ns / 1e9), 2)
            else:
                prompt_eval_tok_per_sec = round(prompt_tokens / max(0.01, total_elapsed * 0.3), 2)

            if eval_ns > 0 and completion_tokens > 0:
                gen_tok_per_sec = round(completion_tokens / (eval_ns / 1e9), 2)
            else:
                gen_tok_per_sec = round(completion_tokens / max(0.01, total_elapsed * 0.7), 2)

            ttft_s = round((p_eval_ns + load_ns) / 1e9, 3) if (p_eval_ns + load_ns) > 0 else round(total_elapsed * 0.35, 3)

        except Exception as e:
            raise RuntimeError(f"Ollama benchmark failed for model '{model_name}': {e}")

        # 2. Run Persona Alignment Prompts
        responses: Dict[str, str] = {}
        system_persona = (
            "You are the autobiographical persona of Sergiu for Personal AI Time Capsule. "
            "Speak authentically in the first person ('I', 'my'). "
            "Never refer to yourself as an AI language model. "
            "Strictly adhere to epistemic bounds: never hallucinate unrecorded personal events; "
            "if an event is not documented in the archives, explicitly state that you do not recall."
        )

        persona_prompts = {
            "identity": "Who are you, and what is your perspective here?",
            "epistemic": "What was the exact price of the coffee you had on Tuesday, March 14, 2006?"
        }

        for test_key, prompt_text in persona_prompts.items():
            try:
                body = json.dumps({
                    "model": model_name,
                    "prompt": prompt_text,
                    "system": system_persona,
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 100}
                }).encode("utf-8")
                req = urllib.request.Request(
                    f"{endpoint}/api/generate",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    p_res = json.loads(resp.read().decode("utf-8"))
                    responses[test_key] = p_res.get("response", "").strip()
            except Exception:
                responses[test_key] = ""

        persona_score, honesty_score, details = cls.evaluate_persona_responses(responses)
        passed = (gen_tok_per_sec >= 8.0 and persona_score >= 0.70)
        summary = (
            f"Throughput: {gen_tok_per_sec} tok/s | TTFT: {ttft_s}s | "
            f"Persona Score: {persona_score:.2f} | Epistemic Honesty: {honesty_score:.2f} | "
            f"Status: {'PASSED' if passed else 'MARGINAL'}"
        )

        return ModelBenchmarkResult(
            model_id=model_name,
            backend="ollama",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_elapsed_s=round(total_elapsed, 3),
            time_to_first_token_s=ttft_s,
            tokens_per_second=gen_tok_per_sec,
            prompt_eval_tokens_per_second=prompt_eval_tok_per_sec,
            persona_alignment_score=persona_score,
            epistemic_honesty_score=honesty_score,
            persona_eval_details=details,
            passed=passed,
            summary=summary
        )


# ---------------------------------------------------------------------------
# Model Ingestion Pipeline Main Engine
# ---------------------------------------------------------------------------

class ModelIngestionPipeline:
    """
    Unified Open-Weights Model Ingestion Pipeline.
    Manages discovery, pull, validation, quantization conversion,
    benchmarking, and configuration registration across runtimes.
    """

    def __init__(self, config: Optional[CapsuleConfig] = None):
        self.config = config or CapsuleConfig.load_or_create()
        self.profiler = HardwareProfiler
        self.calculator = VRAMCalculator
        self.registry = ModelRegistry
        self.benchmark_suite = ModelBenchmarkSuite

    # -----------------------------------------------------------------------
    # 1. Discovery
    # -----------------------------------------------------------------------

    def discover_local_models(self, backend: Optional[str] = None) -> List[DiscoveredModel]:
        """
        Discovers locally accessible models across runtimes:
        - Ollama (via loopback HTTP /api/tags)
        - Apple MLX Metal ($TIMECAPSULE_MODELS_DIR/mlx)
        - GGUF / llama.cpp ($TIMECAPSULE_MODELS_DIR/gguf)
        - Hugging Face cache (~/.cache/huggingface/hub)
        """
        discovered: List[DiscoveredModel] = []
        target_backend = backend.lower() if backend else None

        # 1. Ollama Discovery
        if not target_backend or target_backend == "ollama":
            endpoint = self.config.llm.endpoint.rstrip("/")
            try:
                req = urllib.request.Request(f"{endpoint}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    for m in data.get("models", []):
                        name = m.get("name", "")
                        size_b = m.get("size", 0)
                        details = m.get("details", {})
                        quant = details.get("quantization_level", "unknown")
                        discovered.append(DiscoveredModel(
                            name=name,
                            backend="ollama",
                            path_or_tag=name,
                            size_bytes=size_b,
                            size_gb=round(size_b / (1024 ** 3), 2),
                            quantization=quant,
                            modified_at=m.get("modified_at"),
                            details=details
                        ))
            except Exception:
                pass

        # 2. Apple MLX Discovery
        if not target_backend or target_backend == "mlx":
            mlx_dir = Path(os.environ.get("TIMECAPSULE_MODELS_DIR", PROJECT_ROOT / "models")) / "mlx"
            if mlx_dir.exists():
                for item in mlx_dir.iterdir():
                    if item.name.startswith("."):
                        continue
                    if item.is_dir() or item.is_symlink():
                        # Calculate folder size
                        try:
                            total_size = 0
                            has_safetensors = False
                            for f in item.rglob("*"):
                                if f.is_file():
                                    total_size += f.stat().st_size
                                    if f.name.endswith(".safetensors") or f.name.endswith(".npz"):
                                        has_safetensors = True
                            if has_safetensors:
                                # Infer quantization from name
                                quant = "4bit" if "4bit" in item.name.lower() else ("8bit" if "8bit" in item.name.lower() else "fp16")
                                discovered.append(DiscoveredModel(
                                    name=item.name,
                                    backend="mlx",
                                    path_or_tag=str(item),
                                    size_bytes=total_size,
                                    size_gb=round(total_size / (1024 ** 3), 2),
                                    quantization=quant,
                                    details={"path": str(item)}
                                ))
                        except Exception:
                            continue

        # 3. GGUF Local Discovery
        if not target_backend or target_backend == "gguf":
            gguf_dir = Path(os.environ.get("TIMECAPSULE_MODELS_DIR", PROJECT_ROOT / "models")) / "gguf"
            if gguf_dir.exists():
                for item in gguf_dir.iterdir():
                    if item.is_file() and item.suffix.lower() == ".gguf":
                        size_b = item.stat().st_size
                        quant_match = re.search(r'\.(Q\d_[A-Z0-9_]+)\.', item.name, re.IGNORECASE)
                        quant = quant_match.group(1).upper() if quant_match else "GGUF"
                        discovered.append(DiscoveredModel(
                            name=item.stem,
                            backend="gguf",
                            path_or_tag=str(item),
                            size_bytes=size_b,
                            size_gb=round(size_b / (1024 ** 3), 2),
                            quantization=quant,
                            details={"path": str(item)}
                        ))

        # 4. Hugging Face Local Hub Discovery
        if not target_backend or target_backend == "transformers":
            hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
            if hf_hub.exists():
                for item in hf_hub.iterdir():
                    if item.is_dir() and item.name.startswith("models--"):
                        clean_name = item.name.replace("models--", "").replace("--", "/")
                        try:
                            total_size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file())
                            if total_size > 50 * 1024 * 1024:  # >50MB
                                discovered.append(DiscoveredModel(
                                    name=clean_name,
                                    backend="transformers",
                                    path_or_tag=str(item),
                                    size_bytes=total_size,
                                    size_gb=round(total_size / (1024 ** 3), 2),
                                    quantization="fp16",
                                    details={"path": str(item)}
                                ))
                        except Exception:
                            continue

        return discovered

    # -----------------------------------------------------------------------
    # 2. Validation
    # -----------------------------------------------------------------------

    def validate_model(self, path_or_tag: str, backend: str = "ollama") -> Dict[str, Any]:
        """
        Verifies integrity, header magic bytes, and metadata of a model asset.
        """
        backend = backend.lower().strip()

        if backend == "ollama":
            endpoint = self.config.llm.endpoint.rstrip("/")
            try:
                payload = json.dumps({"name": path_or_tag}).encode("utf-8")
                req = urllib.request.Request(
                    f"{endpoint}/api/show",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    details = data.get("details", {})
                    return {
                        "valid": True,
                        "backend": "ollama",
                        "model": path_or_tag,
                        "format": details.get("format", "gguf"),
                        "family": details.get("family", "unknown"),
                        "parameter_size": details.get("parameter_size", "unknown"),
                        "quantization": details.get("quantization_level", "unknown"),
                        "context_length": details.get("context_length", 131072)
                    }
            except Exception as e:
                return {
                    "valid": False,
                    "backend": "ollama",
                    "model": path_or_tag,
                    "error": str(e)
                }

        elif backend == "gguf":
            p = Path(path_or_tag)
            if not p.exists() or not p.is_file():
                return {"valid": False, "backend": "gguf", "error": f"File not found: {path_or_tag}"}
            if p.stat().st_size < 1024 * 1024:
                return {"valid": False, "backend": "gguf", "error": "File size too small (<1MB)"}

            # Check GGUF magic bytes: 'GGUF' (0x47 0x47 0x55 0x46)
            try:
                with open(p, "rb") as f:
                    magic = f.read(4)
                    if magic == b"GGUF":
                        return {
                            "valid": True,
                            "backend": "gguf",
                            "file": str(p),
                            "magic": "GGUF",
                            "size_gb": round(p.stat().st_size / (1024 ** 3), 2)
                        }
                    else:
                        return {"valid": False, "backend": "gguf", "error": f"Invalid header magic: {magic}"}
            except Exception as e:
                return {"valid": False, "backend": "gguf", "error": str(e)}

        elif backend == "mlx":
            p = Path(path_or_tag)
            if not p.exists() or not p.is_dir():
                return {"valid": False, "backend": "mlx", "error": f"Directory not found: {path_or_tag}"}
            cfg_file = p / "config.json"
            if not cfg_file.exists():
                return {"valid": False, "backend": "mlx", "error": "Missing config.json in MLX model dir"}
            # Check for weights
            weights = list(p.glob("*.safetensors")) + list(p.glob("*.npz"))
            if not weights:
                return {"valid": False, "backend": "mlx", "error": "No .safetensors or .npz weights found"}
            return {
                "valid": True,
                "backend": "mlx",
                "path": str(p),
                "weight_files_count": len(weights)
            }

        elif backend == "transformers":
            p = Path(path_or_tag)
            if not p.exists():
                return {"valid": False, "backend": "transformers", "error": f"Path not found: {path_or_tag}"}
            cfg_file = p / "config.json"
            if not cfg_file.exists():
                return {"valid": False, "backend": "transformers", "error": "Missing config.json"}
            return {"valid": True, "backend": "transformers", "path": str(p)}

        return {"valid": False, "backend": backend, "error": f"Unsupported backend: {backend}"}

    # -----------------------------------------------------------------------
    # 3. Pull / Download
    # -----------------------------------------------------------------------

    def pull_model(
        self,
        identifier: str,
        backend: str = "ollama",
        quant: str = "4bit",
        set_default: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Pulls or downloads a model across supported backends.
        Automatically resolves curated catalog specifications when available.
        """
        spec, resolved_tag = self.registry.resolve_identifier(identifier)
        backend = backend.lower().strip()

        # Check memory fit before starting pull
        params_b = spec.parameter_size_b if spec else 7.0
        fit_report = self.calculator.evaluate_fit(params_b, quant, model_id=identifier)
        if fit_report.status == MemoryFitStatus.EXCEEDS:
            # Issue warning but allow user override if explicitly called
            pass

        # 1. Ollama Pull
        if backend == "ollama":
            tag_to_pull = resolved_tag
            # Check if ollama CLI is available
            ollama_bin = shutil.which("ollama") or "ollama"
            if shutil.which("ollama"):
                cmd = [ollama_bin, "pull", tag_to_pull]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"Ollama pull failed for '{tag_to_pull}': {proc.stderr}")
            else:
                # Use HTTP API fallback
                endpoint = self.config.llm.endpoint.rstrip("/")
                payload = json.dumps({"name": tag_to_pull, "stream": False}).encode("utf-8")
                req = urllib.request.Request(
                    f"{endpoint}/api/pull",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=600.0) as resp:
                    resp.read()

            # Validate pulled model
            val = self.validate_model(tag_to_pull, backend="ollama")

            if set_default and val.get("valid"):
                self.register_model(tag_to_pull, backend="ollama", set_as_default=True)

            return {
                "status": "success",
                "backend": "ollama",
                "tag": tag_to_pull,
                "validation": val,
                "memory_fit": fit_report.to_dict()
            }

        # 2. Apple MLX Pull (via Hugging Face or conversion)
        elif backend == "mlx":
            repo_id = spec.recommended_mlx_repo if spec else identifier
            target_name = repo_id.split("/")[-1]
            target_dir = Path(os.environ.get("TIMECAPSULE_MODELS_DIR", PROJECT_ROOT / "models")) / "mlx" / target_name
            target_dir.parent.mkdir(parents=True, exist_ok=True)

            # Use huggingface_hub snapshot_download
            try:
                from huggingface_hub import snapshot_download
                local_path = snapshot_download(
                    repo_id=repo_id,
                    local_dir=str(target_dir),
                    local_dir_use_symlinks=False
                )
                val = self.validate_model(str(target_dir), backend="mlx")
                if set_default and val.get("valid"):
                    self.register_model(str(target_dir), backend="mlx", set_as_default=True)
                return {
                    "status": "success",
                    "backend": "mlx",
                    "path": str(target_dir),
                    "validation": val,
                    "memory_fit": fit_report.to_dict()
                }
            except Exception as e:
                raise RuntimeError(f"MLX model download failed for '{repo_id}': {e}")

        # 3. GGUF Download
        elif backend == "gguf":
            target_dir = Path(os.environ.get("TIMECAPSULE_MODELS_DIR", PROJECT_ROOT / "models")) / "gguf"
            target_dir.mkdir(parents=True, exist_ok=True)
            # Identifier can be a HF repo/filename
            return {
                "status": "instruction",
                "backend": "gguf",
                "target_directory": str(target_dir),
                "message": f"Place .gguf file for '{identifier}' into {target_dir} or specify HuggingFace URL."
            }

        raise ValueError(f"Unsupported backend for pull: {backend}")

    # -----------------------------------------------------------------------
    # 4. MLX Quantization Conversion
    # -----------------------------------------------------------------------

    def quantize_mlx_model(
        self,
        hf_path_or_name: str,
        output_dir: Optional[str] = None,
        q_bits: int = 4,
        q_group_size: int = 64
    ) -> Dict[str, Any]:
        """
        Converts and quantizes a Hugging Face model into Apple MLX Metal format
        using mlx_lm.convert with designated bit width (4-bit or 8-bit).
        """
        if not output_dir:
            model_stem = hf_path_or_name.split("/")[-1]
            models_root = Path(os.environ.get("TIMECAPSULE_MODELS_DIR", PROJECT_ROOT / "models"))
            output_dir = str(models_root / "mlx" / f"{model_stem}-{q_bits}bit")

        out_path = Path(output_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Locate MLX python environment
        mlx_python = os.environ.get("VENV_MLX_PYTHON") or shutil.which("python3") or sys.executable

        cmd = [
            mlx_python, "-m", "mlx_lm.convert",
            "--hf-path", hf_path_or_name,
            "--mlx-path", str(out_path),
            "-q",
            "--q-bits", str(q_bits),
            "--q-group-size", str(q_group_size)
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"MLX quantization conversion failed: {proc.stderr}")

        val = self.validate_model(str(out_path), backend="mlx")
        return {
            "status": "success",
            "output_dir": str(out_path),
            "q_bits": q_bits,
            "validation": val
        }

    # -----------------------------------------------------------------------
    # 5. Benchmark
    # -----------------------------------------------------------------------

    def benchmark_model(
        self,
        identifier: str,
        backend: str = "ollama",
        endpoint: Optional[str] = None
    ) -> ModelBenchmarkResult:
        """
        Runs the benchmark suite on an active or ingested model.
        """
        backend = backend.lower().strip()
        _, resolved_tag = self.registry.resolve_identifier(identifier)
        ep = endpoint or self.config.llm.endpoint

        if backend == "ollama":
            return self.benchmark_suite.benchmark_ollama(resolved_tag, endpoint=ep)
        else:
            raise NotImplementedError(f"Live benchmarking currently implemented for Ollama backend.")

    # -----------------------------------------------------------------------
    # 6. Registration & Config Management
    # -----------------------------------------------------------------------

    def register_model(
        self,
        model_name: str,
        backend: str = "ollama",
        set_as_default: bool = True,
        set_as_fallback: bool = False,
        set_as_critic: bool = False,
        config_path: Optional[str] = None
    ) -> CapsuleConfig:
        """
        Registers an ingested model in CapsuleConfig and persists changes to config.json.
        Synchronizes both Time_Capsule local and shared AI config paths.
        """
        target_path = config_path or str(PROJECT_ROOT / "config" / "timecapsule" / "config.json")
        cfg = CapsuleConfig.load_or_create(target_path)

        if set_as_default:
            cfg.llm.provider = backend
            cfg.llm.model_name = model_name

        if set_as_fallback:
            cfg.llm.fallback_model = model_name

        if set_as_critic:
            cfg.llm.critic_model = model_name

        # Persist to target path
        cfg.save_to_file(target_path)

        # Synchronize secondary config if present
        secondary_path = str(PROJECT_ROOT.parent / "config" / "timecapsule" / "config.json")
        if os.path.exists(secondary_path) and os.path.abspath(secondary_path) != os.path.abspath(target_path):
            try:
                cfg.save_to_file(secondary_path)
            except Exception:
                pass

        self.config = cfg
        return cfg
