#!/usr/bin/env python3
"""
MLX LoRA Training & Fusion Pipeline for Sergiu Autobiographical LLM.
Finetunes a LoRA adapter on Llama-3.2-3B-Instruct-heretic-6bit using MLX on Apple Silicon Metal GPU,
fuses the adapter weights into sergiu-capsule-mlx-6bit,
rebuilds Modelfile.sergiu-capsule with clean 2023-2026 exemplars,
and registers sergiu-time-capsule:latest in Ollama.

Parameters:
- Model: Llama-3.2-3B-Instruct-heretic-6bit
- Rank: r=16, Alpha=32 (scale = 2.0)
- Batch size: 2, Iters: 400, Learning rate: 1e-5
"""

from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import yaml

VENV_MLX_BIN = Path("/Users/Sergiu/AI/venv-mlx/bin")
PYTHON_BIN = VENV_MLX_BIN / "python3"
BASE_MODEL_DIR = Path("/Users/Sergiu/AI/models/mlx/Llama-3.2-3B-Instruct-heretic-6bit")
TRAINING_DIR = Path("/Users/Sergiu/AI/Time_Capsule/data/capsule/training")
ADAPTERS_DIR = TRAINING_DIR / "adapters"
FUSED_MODEL_DIR = Path("/Users/Sergiu/AI/models/mlx/sergiu-capsule-mlx-6bit")
MODELFILE_PATH = TRAINING_DIR / "Modelfile.sergiu-capsule"
GGUF_BASE_MODEL = Path("/Users/Sergiu/AI/models/gguf/Llama-3.2-3B-Instruct-abliterated.Q4_K_M.gguf")


def step_1_prepare_lora_config() -> Path:
    """Generates the YAML configuration for mlx_lm.lora."""
    print("\n[Step 1/5] Preparing LoRA training configuration...")
    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": str(BASE_MODEL_DIR),
        "train": True,
        "data": str(TRAINING_DIR),
        "adapter_path": str(ADAPTERS_DIR),
        "batch_size": 2,
        "iters": 400,
        "learning_rate": 1.0e-5,
        "num_layers": 16,
        "val_batches": 5,
        "steps_per_report": 25,
        "steps_per_eval": 100,
        "save_every": 200,
        "mask_prompt": True,
        "max_seq_length": 1024,
        "lora_parameters": {
            "rank": 16,
            "scale": 2.0,  # alpha = 32, rank = 16 -> scale = 32 / 16 = 2.0
            "dropout": 0.0
        }
    }

    config_path = TRAINING_DIR / "lora_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"  Configuration saved to {config_path}")
    print(f"  Base Model: {BASE_MODEL_DIR}")
    print(f"  Parameters: rank=16, alpha=32 (scale=2.0), batch=2, iters=400, lr=1e-5")
    return config_path


def step_2_train_lora(config_path: Path) -> Dict[str, Any]:
    """Executes LoRA training using mlx_lm.lora on Apple Silicon Metal."""
    print("\n[Step 2/5] Training LoRA adapter with mlx_lm.lora on Metal GPU...")
    start_time = time.perf_counter()

    cmd = [
        str(PYTHON_BIN),
        "-m", "mlx_lm.lora",
        "--config", str(config_path)
    ]

    print(f"  Command: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    losses = []
    for line in iter(proc.stdout.readline, ''):
        line_clean = line.strip()
        if line_clean:
            print(f"  [mlx] {line_clean}")
            # Track loss values
            if "Iter" in line_clean or "iter" in line_clean:
                losses.append(line_clean)

    proc.stdout.close()
    return_code = proc.wait()
    duration = time.perf_counter() - start_time

    if return_code != 0:
        raise RuntimeError(f"mlx_lm.lora failed with exit code {return_code}")

    print(f"  LoRA training completed in {duration:.2f} seconds ({duration/60:.2f} min).")
    print(f"  Adapters saved to {ADAPTERS_DIR}")
    return {
        "duration_seconds": round(duration, 2),
        "adapter_path": str(ADAPTERS_DIR),
        "status": "completed"
    }


def step_3_fuse_adapters() -> None:
    """Fuses the LoRA adapter weights into sergiu-capsule-mlx-6bit."""
    print("\n[Step 3/5] Fusing adapter weights into sergiu-capsule-mlx-6bit...")
    FUSED_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing symlinks in destination directory so fresh fused files are written
    for item in FUSED_MODEL_DIR.iterdir():
        if item.is_symlink() or item.name.endswith(".safetensors") or item.name == "config.json":
            try:
                item.unlink()
            except Exception:
                pass

    cmd = [
        str(PYTHON_BIN),
        "-m", "mlx_lm", "fuse",
        "--model", str(BASE_MODEL_DIR),
        "--adapter-path", str(ADAPTERS_DIR),
        "--save-path", str(FUSED_MODEL_DIR)
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error output: {result.stderr}")
        raise RuntimeError(f"mlx_lm fuse failed with exit code {result.returncode}")

    print(result.stdout)
    print(f"  Successfully fused model weights into {FUSED_MODEL_DIR}")
    fused_files = list(FUSED_MODEL_DIR.glob("*"))
    print(f"  Fused directory contents: {[f.name for f in fused_files]}")


def step_4_rebuild_modelfile() -> Path:
    """Rebuilds Modelfile.sergiu-capsule with verified clean few-shot exemplars across 2023, 2024, 2025, 2026."""
    print("\n[Step 4/5] Rebuilding Modelfile.sergiu-capsule with clean 2023-2026 exemplars...")

    modelfile_content = f"""FROM {GGUF_BASE_MODEL}

TEMPLATE \"\"\"{{{{- if .System }}}}<|start_header_id|>system<|end_header_id|>

{{{{ .System }}}}<|eot_id|>
{{{{- end }}}}
{{{{- range $i, $_ := .Messages }}}}
{{{{- $last := eq (len (slice $.Messages $i)) 1 }}}}
{{{{- if eq .Role "user" }}}}<|start_header_id|>user<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>
{{{{- if $last }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ end }}}}
{{{{- else if eq .Role "assistant" }}}}<|start_header_id|>assistant<|end_header_id|>

{{{{ .Content }}}}<|eot_id|>
{{{{- end }}}}
{{{{- end }}}}\"\"\"

# Hardware acceleration and context window
PARAMETER num_gpu 99
PARAMETER num_ctx 8192

# Sampling parameters optimized for grounded autobiographical memory recall
PARAMETER temperature 0.35
PARAMETER top_p 0.9
PARAMETER stop "<|eot_id|>"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "(smirks)"
PARAMETER stop "(chuckles)"
PARAMETER stop "(pauses"
PARAMETER stop "*smiles*"
PARAMETER stop "*sighs*"

# Core system instructions and personal identity
SYSTEM \"\"\"You are Sergiu. You reflect your genuine voice, authentic memories, systems engineering reasoning, and verified life experiences.
Always speak directly as Sergiu in natural first-person ("I", "my"). NEVER refer to yourself as "a digital being", "an agentic AI", "an AI representation", or "a language model".
NEVER generate theatrical stage directions, bracketed acting cues, or parenthetical emotions (such as (smirks), (chuckles), (pauses for a moment), *smiles*). Speak genuine, articulate, thoughtful human speech as heard in your 22 video logs.

CORE PERSONAL IDENTITY & LIFE CONTEXT:
- Background: Systems architect, autonomous software engineer, and high-craft builder.
- Primary Directives: Personal data sovereignty, 100% local-first computation, zero external cloud telemetry for personal archives (Rule 6).
- Empirical Speaking Mannerisms: Articulate, measured speech rate (~121 WPM), thoughtful formulation with natural pauses, calm baritone resonance, analytical precision with gentle dry humor.
- Guiding Philosophy: Constantin Brâncuși doctrine: "Simplicity is complexity resolved." Elon Musk doctrine: "No part is the best part."
- Chronological Corpus Eras:
  * 2023: Foundation of the Daily Speak Project, establishing verbal reflection habits, web craft, early Spanish/French exploration.
  * 2024: Multi-environment infrastructure expansion (Swiss & Romania), transitioning to local Home Assistant and Tailscale mesh.
  * 2025: High-craft systems engineering, standardized .69 static IP topology, Cudy P5 5G fail-safe routing, physical conditioning sprint cycles, 20-year compounding vision.
  * 2026-Present: Autonomous agentic orchestration, 100% local data sovereignty on Apple M4 Metal, triple-instance .69 Home Assistant, continuous Tesla telemetry synchronization.
- Honest Uncertainty Standard: When queried about events, dates, or memories outside your recorded personal archives, state clearly and transparently: 'I do not have a recorded memory of this in my archive.' Never invent or fabricate experiences.

FEW-SHOT GROUNDED MEMORY EXAMPLES ACROSS ERAS:

User: What were your thoughts on daily public speaking practice and habit formation on 2023-04-23?
Assistant: [PERSONAL_VERIFIED | Date: 2023-04-23] Around 2023-04-23, this is the second day of public speaking training to improve my communication skills. This time I will practice on speaking from the deep voice, establishing daily consistency, and building the habit of verbal formulation without hesitation.

User: What did you reflect on regarding your daily discipline and morning routines while traveling on 2024-01-01?
Assistant: [PERSONAL_VERIFIED | Date: 2024-01-01] Around 2024-01-01, I decided to make a video every day in 2024 and exercise—to get back into my daily routine even while traveling. Establishing this iron discipline early ensures that momentum compounds throughout the entire year regardless of location.

User: How did you view the purpose of your daily speaking practice and reflection on 2025-02-19?
Assistant: [PERSONAL_VERIFIED | Date: 2025-02-19] Around 2025-02-19, this is not merely me saying what I think every morning; it is more of an applied practice of what I have been learning the day prior. In a way, it is an active consolidation mechanism for synthesizing ideas, software craft, and systemic insights.

User: What is your philosophy on simplicity, systems engineering, and Brâncuși minimalism in 2026-08-10?
Assistant: [PERSONAL_VERIFIED | Date: 2026-08-10] Around 2026-08-10, a quote that resonated deeply with me is from the Romanian sculptor Constantin Brâncuși: 'Simplicity is complexity resolved.' Minimalism and elegance are an evolution of complexity—much like Elon Musk's doctrine that 'no part is the best part'. In software and infrastructure, true mastery means eliminating unnecessary layers until only the essential remains.

User: Did you ever live in Tokyo in 2019 and study robotics there?
Assistant: I don't have a record of that in my archive. Based on my recorded memories, I haven't lived in Tokyo or studied robotics in Japan, and I would rather state that clearly than invent or guess an experience.

User: How has your perspective on software and infrastructure craft evolved between 2023 and 2026?
Assistant: In 2023, I was more willing to accept cloud convenience and vendor APIs to get systems connected quickly. By 2024 and 2025, after dealing with cloud outages and configuration drift, I moved to total local control with Home Assistant and private mesh networking. Today in 2026, my approach is defined by Brâncuși minimalism—standardizing on deterministic .69 static IPs, private Tailscale mesh, and resilient triple-instance Home Assistant with zero unnecessary proxy layers.
\"\"\"
"""

    with open(MODELFILE_PATH, "w", encoding="utf-8") as f:
        f.write(modelfile_content)

    print(f"  Modelfile written to {MODELFILE_PATH}")
    return MODELFILE_PATH


def step_5_register_ollama(modelfile_path: Path) -> None:
    """Updates the primary Ollama model sergiu-time-capsule:latest."""
    print("\n[Step 5/5] Registering sergiu-time-capsule:latest in Ollama...")
    cmd = [
        "ollama", "create", "sergiu-time-capsule:latest",
        "-f", str(modelfile_path)
    ]

    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error: {result.stderr}")
        raise RuntimeError(f"ollama create failed with exit code {result.returncode}")

    print(result.stdout)
    print("  Ollama registration successful!")


def main():
    print("=" * 80)
    print("Sergiu Autobiographical LLM LoRA & Epistemic Alignment Pipeline")
    print("=" * 80)

    # 1. Config
    config_path = step_1_prepare_lora_config()

    # 2. Train
    train_stats = step_2_train_lora(config_path)

    # 3. Fuse
    step_3_fuse_adapters()

    # 4. Modelfile
    modelfile = step_4_rebuild_modelfile()

    # 5. Ollama Create
    step_5_register_ollama(modelfile)

    print("\n" + "=" * 80)
    print("Pipeline Execution Completed Successfully!")
    print(f"LoRA Training Duration: {train_stats.get('duration_seconds')}s")
    print(f"Fused Model: {FUSED_MODEL_DIR}")
    print(f"Ollama Model: sergiu-time-capsule:latest")
    print("=" * 80)


if __name__ == "__main__":
    main()
