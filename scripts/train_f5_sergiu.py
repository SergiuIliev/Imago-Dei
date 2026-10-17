"""
Fine-tuning F5-TTS on Sergiu's Authentic Voice Dataset.
Trains the Diffusion Transformer (DiT) flow-matching model directly on
the 218 24kHz speech segments extracted from personal vlogs.
"""

import os
import sys
import torch
from pathlib import Path
from safetensors.torch import load_file

# Set environment
os.environ["PYTHONHASHSEED"] = "0"
os.environ["PATH"] = f"/Users/Sergiu/AI/venv-voice/bin:{os.environ.get('PATH', '')}"

from f5_tts.model import CFM, DiT, Trainer
from f5_tts.model.dataset import load_dataset
from f5_tts.model.utils import get_tokenizer

def main():
    print("======================================================================")
    print(" PROJECT 30: F5-TTS NEURAL VOICE FINE-TUNING ON SERGIU RECORDINGS")
    print("======================================================================")
    
    device = "cpu" # CPU avoids Metal MPS backward stride autograd view issues
    print(f"Target Compute Device: {device}")
    
    dataset_dir = "/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_training/f5_processed"
    ckpt_dir = "/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_training/checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    vocab_path = os.path.join(dataset_dir, "vocab.txt")
    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")
    print(f"Vocabulary Size: {vocab_size} tokens")

    # Audio Spectrogram Hyperparameters (24kHz)
    mel_spec_kwargs = dict(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24000,
        mel_spec_type="vocos",
    )

    # Model Configuration matching F5TTS_v1_Base
    model_cfg = dict(
        dim=1024,
        depth=22,
        heads=16,
        ff_mult=2,
        text_dim=512,
        conv_layers=4,
    )

    transformer = DiT(**model_cfg, text_num_embeds=vocab_size, mel_dim=100)
    model = CFM(
        transformer=transformer,
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    # Load Base Pretrained Weights
    base_ckpt = "/Users/Sergiu/.cache/huggingface/hub/models--SWivid--F5-TTS/snapshots/84e5a410d9cead4de2f847e7c9369a6440bdfaca/F5TTS_v1_Base/model_1250000.safetensors"
    if os.path.exists(base_ckpt):
        print(f"Loading pretrained base weights from: {base_ckpt}")
        weights = load_file(base_ckpt)
        # Filter matching weights (vocabs might slightly differ in size)
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
        print(f"Loaded {len(filtered)} layers successfully.")

    # Load Processed Dataset
    print(f"Loading prepped dataset from {dataset_dir}...")
    train_dataset = load_dataset(
        dataset_name=dataset_dir,
        tokenizer="custom",
        dataset_type="CustomDatasetPath",
        audio_type="raw",
        mel_spec_kwargs=mel_spec_kwargs
    )
    print(f"Training dataset size: {len(train_dataset)} speech segments.")

    trainer = Trainer(
        model=model,
        epochs=3,
        learning_rate=2e-5,
        num_warmup_updates=10,
        save_per_updates=25,
        checkpoint_path=ckpt_dir,
        batch_size_per_gpu=2,
        batch_size_type="sample",
        max_samples=2,
        grad_accumulation_steps=2,
        max_grad_norm=1.0,
        logger=None,
        log_samples=False,
        accelerate_kwargs={"cpu": True},
    )

    print("Beginning fine-tuning updates on Sergiu's vocal tract and speech timbre...")
    trainer.train(train_dataset, num_workers=2, resumable_with_seed=666)

    # Export final fine-tuned model checkpoint
    final_ckpt_path = "/Users/Sergiu/AI/Time_Capsule/data/capsule/voice_training/sergiu_f5_finetuned.pt"
    torch.save(model.state_dict(), final_ckpt_path)
    print(f"\n✔ Successfully exported fine-tuned voice model to: {final_ckpt_path}")

if __name__ == "__main__":
    main()
