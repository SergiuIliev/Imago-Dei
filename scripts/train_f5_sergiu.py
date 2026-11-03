#!/usr/bin/env python3
"""
scripts/train_f5_sergiu.py
Fine-tuning F5-TTS on Sergiu's Authentic Voice Dataset.

Directives:
- Rule 6: 100% on-device local execution on loopback. Zero external network calls.
- Device: Apple Silicon M4 CPU ('cpu') for DiT flow-matching backprop
  (prevents Metal MPS autograd view/stride issues during backward passes).
- Trains on the 450 curated 24kHz speech segments extracted from personal vlogs.
- Learning rate: 2e-5, epochs: 3-5, batch_size: 2, grad_accumulation: 2.
- Exports fine-tuned weights to data/capsule/voice_training/sergiu_f5_finetuned.pt.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import torch
from safetensors.torch import load_file

# Set environment
os.environ["PYTHONHASHSEED"] = "0"
try:
    import imageio_ffmpeg
    ffmpeg_dir = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())
    os.environ["PATH"] = f"{ffmpeg_dir}:{os.environ.get('PATH', '')}"
except Exception:
    pass

# Use 8 threads on Apple Silicon M4 for optimal CPU performance
torch.set_num_threads(8)

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "capsule" / "voice_training" / "f5_processed"
CKPT_DIR = PROJECT_ROOT / "data" / "capsule" / "voice_training" / "checkpoints"
FINAL_CKPT = PROJECT_ROOT / "data" / "capsule" / "voice_training" / "sergiu_f5_finetuned.pt"
hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
BASE_CKPT = Path(os.environ.get("F5TTS_BASE_CKPT", str(hf_home / "hub" / "models--SWivid--F5-TTS" / "snapshots" / "84e5a410d9cead4de2f847e7c9369a6440bdfaca" / "F5TTS_v1_Base" / "model_1250000.safetensors")))


def main():
    parser = argparse.ArgumentParser(description="F5-TTS Flow-Matching Fine-Tuning on Sergiu's Authentic Voice")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per step (default: 2)")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps (default: 2)")
    parser.add_argument("--max-updates", type=int, default=None, help="Optional maximum gradient updates ceiling")
    parser.add_argument("--save-per-updates", type=int, default=25, help="Save frequency (default: 25)")
    args = parser.parse_args()

    print("=" * 72)
    print(" PROJECT 30: F5-TTS NEURAL VOICE FINE-TUNING ON SERGIU GOLD DATASET")
    print("=" * 72)
    print(f"Target Compute Device:  cpu (Apple Silicon M4 Multi-Threaded, threads: {torch.get_num_threads()})")
    print(f"Hyperparameters:        LR={args.learning_rate}, Epochs={args.epochs}, Batch={args.batch_size}, GradAccum={args.grad_accum}")
    print(f"Dataset Directory:      {DATASET_DIR}")
    print(f"Checkpoint Directory:   {CKPT_DIR}")
    print(f"Final Export Path:      {FINAL_CKPT}")
    print("=" * 72)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    vocab_path = DATASET_DIR / "vocab.txt"
    if not vocab_path.exists():
        raise FileNotFoundError(f"vocab.txt not found at {vocab_path}")

    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    print(f"\nVocabulary Size: {vocab_size} tokens")

    # Audio Spectrogram Hyperparameters (24kHz, matching Vocos vocoder)
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )

    # Model Configuration matching F5TTS_v1_Base (Diffusion Transformer)
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )

    print("\nInstantiating Diffusion Transformer (DiT) and CFM flow-matching model...")
    transformer = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Load Weights: Prefer continuing from existing fine-tuned weights, or start from base F5-TTS
    loaded_source = None
    if FINAL_CKPT.exists():
        print(f"Initializing from existing fine-tuned checkpoint: {FINAL_CKPT}")
        ckpt = torch.load(FINAL_CKPT, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "ema_model_state_dict" in ckpt:
            weights = {k.replace("ema_model.", ""): v for k, v in ckpt["ema_model_state_dict"].items() if k not in ["initted", "step"]}
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            weights = ckpt["model_state_dict"]
        else:
            weights = ckpt
        model_state = model.state_dict()
        filtered = {k: v for k, v in weights.items() if k in model_state and model_state[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} layers from existing fine-tuned model.")
        loaded_source = "existing_finetuned"
    elif BASE_CKPT.exists():
        print(f"Loading pretrained base F5-TTS weights from: {BASE_CKPT}")
        weights = load_file(str(BASE_CKPT))
        model_state = model.state_dict()
        filtered = {}
        for k, v in weights.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            elif "ema_model" in k:
                subk = k.replace("ema_model.", "")
                if subk in model_state and model_state[subk].shape == v.shape:
                    filtered[subk] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)} layers from pretrained base.")
        loaded_source = "base_pretrained"
    else:
        print("⚠️ Warning: Neither fine-tuned checkpoint nor base checkpoint found. Training from scratch.")
        loaded_source = "scratch"

    # Load Processed Gold Dataset
    print(f"\nLoading Gold Dataset from: {DATASET_DIR}...")
    train_dataset = load_dataset(
        dataset_name=str(DATASET_DIR),
        tokenizer="custom",
        dataset_type="CustomDatasetPath",
        audio_type="raw",
        mel_spec_kwargs=mel_spec_kwargs,
    )
    print(f"Loaded {len(train_dataset)} authentic speech segments.")

    trainer = Trainer(
        model=model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_warmup_updates=10,
        save_per_updates=args.save_per_updates,
        checkpoint_path=str(CKPT_DIR),
        batch_size_per_gpu=args.batch_size,
        batch_size_type="sample",
        max_samples=args.batch_size,
        grad_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        logger=None,
        log_samples=False,
        accelerate_kwargs={"cpu": True},
    )

    print(f"\nBeginning flow-matching fine-tuning for {args.epochs} epochs...")
    t0 = time.time()
    try:
        trainer.train(train_dataset, num_workers=2, resumable_with_seed=666)
    except KeyboardInterrupt:
        print("\n[Training] Interrupted by user. Saving checkpoint...")

    elapsed = time.time() - t0
    print(f"\nFine-tuning completed in {elapsed:.1f}s ({elapsed / 60.0:.2f} minutes).")

    # Save final fine-tuned model checkpoint
    print(f"Exporting final model checkpoint to: {FINAL_CKPT}...")
    save_dict = {
        "ema_model_state_dict": trainer.ema_model.state_dict(),
        "model_state_dict": trainer.accelerator.unwrap_model(trainer.model).state_dict(),
    }
    torch.save(save_dict, FINAL_CKPT)
    ckpt_size_mb = FINAL_CKPT.stat().st_size / (1024 * 1024)
    print(f"✔ Successfully saved {ckpt_size_mb:.1f} MB fine-tuned voice model to: {FINAL_CKPT}")


if __name__ == "__main__":
    main()
