#!/usr/bin/env python3
"""
CLI Tool for Open-Weights Model Ingestion & Management in Personal AI Time Capsule.
Supports listing curated models, hardware profiling, pulling from Ollama/MLX/GGUF,
running throughput and autobiographical persona benchmarks, and configuring defaults.

Zero telemetry. 100% local-first.
"""

from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.model_pipeline import (
    ModelIngestionPipeline,
    HardwareProfiler,
    VRAMCalculator,
    ModelRegistry,
    MemoryFitStatus
)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Renders an aligned ASCII table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    row_lines = [
        " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return f"{header_line}\n{separator}\n" + "\n".join(row_lines)


def cmd_profile(args):
    hw = HardwareProfiler.detect()
    print("=" * 64)
    print(" Personal AI Time Capsule — Hardware Profiler & VRAM Headroom")
    print("=" * 64)
    print(f" Platform:          {hw.platform_system} ({hw.architecture}) release {hw.platform_release}")
    print(f" CPU Model:         {hw.cpu_brand} ({hw.cpu_cores} cores)")
    print(f" Apple Silicon:     {'Yes (ARM64 Metal)' if hw.is_apple_silicon else 'No'}")
    print(f" Unified Memory:    {'Yes' if hw.unified_memory else 'No (Discrete / Non-unified)'}")
    print(f" GPU Device:        {hw.gpu_device_name}")
    print(f" Total System RAM:  {hw.total_ram_gb:.2f} GB")
    print(f" Available RAM:     {hw.available_ram_gb:.2f} GB")
    print(f" Metal / GPU VRAM:  {hw.gpu_vram_gb:.2f} GB")
    print(f" Max Allocatable:   {hw.max_recommended_model_memory_gb:.2f} GB (Recommended Model Ceiling)")
    print("=" * 64)


def cmd_list(args):
    pipeline = ModelIngestionPipeline()
    hw = HardwareProfiler.detect()

    if args.discovered:
        print("\n--- Locally Discovered Models on Host ---")
        models = pipeline.discover_local_models(backend=args.backend)
        if not models:
            print("No local models found for specified backend.")
            return

        headers = ["Backend", "Model Name / Tag", "Size (GB)", "Quantization", "Path / Identifier"]
        rows = [
            [m.backend.upper(), m.name, f"{m.size_gb:.2f}", m.quantization, m.path_or_tag[:45]]
            for m in models
        ]
        print(format_table(headers, rows))
        print(f"\nTotal discovered local models: {len(models)}\n")
        return

    # Curated model listing
    models = ModelRegistry.list_models(family=args.family)
    headers = ["Model ID", "Family", "Params", "Context", "4-bit RAM", "Host Memory Fit Status"]
    rows = []

    for m in models:
        fit = VRAMCalculator.evaluate_fit(m.parameter_size_b, quant=args.quant, hardware=hw, model_id=m.model_id)
        if fit.status == MemoryFitStatus.COMFORTABLE:
            fit_label = f"Comfortable ({fit.headroom_gb:+.1f} GB)"
        elif fit.status == MemoryFitStatus.TIGHT:
            fit_label = f"Tight Fit ({fit.headroom_gb:+.1f} GB)"
        else:
            fit_label = f"EXCEEDS (deficit {abs(fit.headroom_gb):.1f} GB)"

        rows.append([
            m.model_id,
            m.family,
            f"{m.parameter_size_b:.1f}B",
            f"{m.context_window // 1024}k",
            f"{fit.total_required_gb:.1f} GB",
            fit_label
        ])

    print("\n" + "=" * 80)
    print(f" Curated Open-Weights Model Catalog (Host Allocatable: {hw.max_recommended_model_memory_gb:.1f} GB)")
    print("=" * 80)
    print(format_table(headers, rows))
    print("=" * 80)
    print(f"Showing {len(rows)} curated models. Filter with --family <name> or inspect with --discovered.\n")


def cmd_pull(args):
    pipeline = ModelIngestionPipeline()
    print(f"[*] Preparing ingestion for model: '{args.model_identifier}' (backend: {args.backend}, quant: {args.quant})")

    # Evaluate fit first
    spec, tag = ModelRegistry.resolve_identifier(args.model_identifier)
    params = spec.parameter_size_b if spec else 7.0
    fit = VRAMCalculator.evaluate_fit(params, quant=args.quant)

    print(f"[*] Footprint: {fit.total_required_gb:.2f} GB required | Allocatable: {fit.allocatable_memory_gb:.2f} GB")
    print(f"[*] Recommendation: {fit.recommendation}")

    if not fit.fits:
        print(f"[!] WARNING: Model '{args.model_identifier}' exceeds host recommended memory headroom.")
        if not args.force:
            print("[!] Aborting pull. Use --force to proceed anyway.")
            sys.exit(1)

    try:
        res = pipeline.pull_model(
            args.model_identifier,
            backend=args.backend,
            quant=args.quant,
            set_default=args.set_default
        )
        print(f"[+] Model '{args.model_identifier}' successfully ingested via {args.backend}!")
        if args.set_default:
            print(f"[+] Registered as default active model in Time Capsule configuration.")
    except Exception as e:
        print(f"[-] Error pulling model: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_benchmark(args):
    pipeline = ModelIngestionPipeline()
    print(f"[*] Initiating benchmark and persona verification on '{args.model_identifier}'...")
    try:
        res = pipeline.benchmark_model(args.model_identifier, backend=args.backend, endpoint=args.endpoint)
        print("\n" + "=" * 64)
        print(" Benchmark & Epistemic Alignment Report")
        print("=" * 64)
        print(f" Target Model:            {res.model_id} ({res.backend})")
        print(f" Time To First Token:     {res.time_to_first_token_s:.3f} s")
        print(f" Prompt Eval Speed:       {res.prompt_eval_tokens_per_second:.2f} tok/s")
        print(f" Generation Speed:        {res.tokens_per_second:.2f} tok/s")
        print(f" Completion Tokens:       {res.completion_tokens} tokens")
        print(f" Total Wall Time:         {res.total_elapsed_s:.3f} s")
        print("-" * 64)
        print(f" Autobiographical Persona: {res.persona_alignment_score:.2f} / 1.00")
        print(f" Epistemic Honesty:       {res.epistemic_honesty_score:.2f} / 1.00")
        print(f" Overall Verification:    {'PASSED' if res.passed else 'FAILED/MARGINAL'}")
        print("=" * 64)
        print(f" Summary: {res.summary}\n")
    except Exception as e:
        print(f"[-] Benchmark failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_set_default(args):
    pipeline = ModelIngestionPipeline()
    is_primary = (args.role == "primary")
    is_fallback = (args.role == "fallback")
    is_critic = (args.role == "critic")

    cfg = pipeline.register_model(
        args.model_identifier,
        backend=args.backend,
        set_as_default=is_primary,
        set_as_fallback=is_fallback,
        set_as_critic=is_critic
    )
    print(f"[+] Successfully registered '{args.model_identifier}' as {args.role} model (provider: {args.backend}).")
    print(f"    Current Primary:  {cfg.llm.model_name} ({cfg.llm.provider})")
    print(f"    Current Fallback: {cfg.llm.fallback_model}")
    print(f"    Current Critic:   {cfg.llm.critic_model}")


def cmd_quantize(args):
    pipeline = ModelIngestionPipeline()
    print(f"[*] Quantizing HF model '{args.hf_model}' to MLX Metal format ({args.bits}-bit)...")
    try:
        res = pipeline.quantize_mlx_model(
            args.hf_model,
            output_dir=args.output_dir,
            q_bits=args.bits,
            q_group_size=args.group_size
        )
        print(f"[+] MLX Quantization complete: {res['output_dir']}")
    except Exception as e:
        print(f"[-] Quantization failed: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Personal AI Time Capsule — Open-Weights Model Ingestion & Management CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # profile
    subparsers.add_parser("profile", help="Display host hardware, VRAM, and memory headroom")

    # list
    p_list = subparsers.add_parser("list", help="List curated models or locally discovered models")
    p_list.add_argument("--family", type=str, help="Filter by model family (llama, qwen, deepseek, mistral, gemma, phi)")
    p_list.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "fp16"], help="Quantization for fit calculation (default: 4bit)")
    p_list.add_argument("--discovered", action="store_true", help="List locally installed/discovered models on host")
    p_list.add_argument("--backend", type=str, choices=["ollama", "mlx", "gguf", "transformers"], help="Backend filter for discovered models")

    # pull
    p_pull = subparsers.add_parser("pull", help="Download/pull a model into the local environment")
    p_pull.add_argument("model_identifier", type=str, help="Model ID, tag, or repo name (e.g. qwen2.5:7b, llama3.2:3b)")
    p_pull.add_argument("--backend", type=str, default="ollama", choices=["ollama", "mlx", "gguf"], help="Inference backend (default: ollama)")
    p_pull.add_argument("--quant", type=str, default="4bit", help="Target quantization (default: 4bit)")
    p_pull.add_argument("--set-default", action="store_true", help="Set as default model in config after pulling")
    p_pull.add_argument("--force", action="store_true", help="Force pull even if model exceeds recommended memory")

    # benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run speed and persona alignment benchmarks on a model")
    p_bench.add_argument("model_identifier", type=str, help="Model tag or identifier to benchmark")
    p_bench.add_argument("--backend", type=str, default="ollama", choices=["ollama", "mlx"], help="Inference backend (default: ollama)")
    p_bench.add_argument("--endpoint", type=str, help="Custom API endpoint (optional)")

    # set-default
    p_set = subparsers.add_parser("set-default", help="Set the default active LLM model in Time Capsule configuration")
    p_set.add_argument("model_identifier", type=str, help="Model tag or path")
    p_set.add_argument("--backend", type=str, default="ollama", choices=["ollama", "mlx", "mock"], help="Inference backend (default: ollama)")
    p_set.add_argument("--role", type=str, default="primary", choices=["primary", "fallback", "critic"], help="Role to assign (default: primary)")

    # quantize
    p_quant = subparsers.add_parser("quantize", help="Quantize a Hugging Face model to MLX format")
    p_quant.add_argument("hf_model", type=str, help="Hugging Face repo or local path")
    p_quant.add_argument("--output-dir", type=str, help="Output directory for MLX model")
    p_quant.add_argument("--bits", type=int, default=4, choices=[4, 8], help="Quantization bits (default: 4)")
    p_quant.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    cmds = {
        "profile": cmd_profile,
        "list": cmd_list,
        "pull": cmd_pull,
        "benchmark": cmd_benchmark,
        "set-default": cmd_set_default,
        "quantize": cmd_quantize
    }

    cmds[args.command](args)


if __name__ == "__main__":
    main()
