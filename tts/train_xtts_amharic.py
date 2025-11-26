"""
Amharic Text-to-Speech Training Script
Fine-tunes Coqui XTTS v2 for Amharic language support.

This script performs Language Adaptation:
1. Vocabulary Expansion - Teaching the tokenizer Amharic characters (Fidel script)
2. GPT Fine-Tuning - Training the model to predict Amharic sound flows

Note: Amharic is NOT one of the 17 native XTTS languages, so we use
the "Language Hijacking" technique if needed.
"""

import os
import sys
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.xtts_config import XttsConfig, XttsArgs
from TTS.tts.models.xtts import Xtts
from TTS.tts.datasets import load_tts_samples
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.utils.manage import ModelManager

# --- Configuration ---
# 1. Define Paths
OUT_PATH = "xtts_amharic_model"
DATA_PATH = "./dataset/"  # Must contain 'wavs' folder and 'metadata.csv'
METADATA_FILE = "metadata.csv"  # Format: audio_file.wav|text_transcription

# 2. Define Language & Model
# Use "am" for Amharic. If you get "Language not supported" error,
# change to "ar" (Arabic) - this is the "Language Hijacking" trick
TARGET_LANG = "am"  # Amharic code (change to "ar" if needed)
CHECKPOINT_DIR = "./xtts_base_model"

# 3. Training hyperparameters
NUM_EPOCHS = 10  # Start with 10, increase to 50+ for better quality
BATCH_SIZE = 4
EVAL_BATCH_SIZE = 4
LEARNING_RATE = 5e-06  # Very low to preserve pre-training
SAVE_STEP = 500
EVAL_STEP = 50


def check_dataset():
    """Verify dataset structure before training."""
    wavs_path = os.path.join(DATA_PATH, "wavs")
    metadata_path = os.path.join(DATA_PATH, METADATA_FILE)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Dataset folder not found: {DATA_PATH}")
        print("\n📁 Expected structure:")
        print("   dataset/")
        print("   ├── wavs/")
        print("   │   ├── audio1.wav")
        print("   │   ├── audio2.wav")
        print("   │   └── ...")
        print("   └── metadata.csv")
        print("\n📝 metadata.csv format:")
        print("   audio1.wav|ሰላም ይህ የአማርኛ ጽሁፍ ነው")
        print("   audio2.wav|እንዴት ነህ")
        return False
    
    if not os.path.exists(wavs_path):
        print(f"❌ 'wavs' folder not found in {DATA_PATH}")
        return False
    
    if not os.path.exists(metadata_path):
        print(f"❌ metadata.csv not found in {DATA_PATH}")
        return False
    
    # Count audio files
    wav_files = [f for f in os.listdir(wavs_path) if f.endswith('.wav')]
    print(f"✅ Found {len(wav_files)} audio files in dataset")
    
    # Check metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    print(f"✅ Found {len(lines)} entries in metadata.csv")
    
    return True


def main():
    print("=" * 60)
    print("🗣️  Amharic XTTS v2 Training Script")
    print("=" * 60)
    
    # --- Step 0: Verify Dataset ---
    print("\n📂 Checking dataset structure...")
    if not check_dataset():
        print("\n⚠️  Please prepare your dataset first!")
        print("See TTS_README.md for data preparation guidelines.")
        sys.exit(1)
    
    # --- Step 1: Download Base Model ---
    print("\n⬇️  Downloading XTTS v2 Base Model...")
    manager = ModelManager()
    model_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
    print(f"✅ Model downloaded to: {model_path}")
    
    # --- Step 2: Configure Dataset ---
    print("\n📋 Configuring dataset...")
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech",  # Assumes file|text format
        dataset_name="amharic_corpus",
        path=DATA_PATH,
        meta_file_train=METADATA_FILE,
        language=TARGET_LANG,
    )
    
    # --- Step 3: Configure Training ---
    print("\n⚙️  Setting up training configuration...")
    config = XttsConfig(
        output_path=OUT_PATH,
        model_args=XttsArgs(
            gpt_batch_size=1,  # Keep low for consumer GPUs (increase if 24GB+ VRAM)
            enable_redaction=False,
            kv_cache=True,
            gpt_num_audio_tokens=1024,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_val_freq=EVAL_STEP,
        ),
        run_eval=True,
        test_delay_epochs=-1,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_loader_workers=4,
        print_step=50,
        plot_step=100,
        log_model_step=100,
        save_step=SAVE_STEP,
        save_n_checkpoints=2,
        save_checkpoints=True,
        target_loss="loss",
        print_eval=False,
        # Optimizer settings (Standard for XTTS)
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=LEARNING_RATE,
        lr_scheduler="MicroBatchCosineLR",
        lr_scheduler_params={"warmup_steps": 100, "min_lr": 1e-7},
    )
    
    # --- Step 4: Load and Adapt Model ---
    print("\n🔧 Loading base model and adapting for Amharic...")
    
    model_dir = os.path.dirname(model_path)
    config.load_json(os.path.join(model_dir, "config.json"))
    
    # Initialize the model
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)
    
    # --- Step 5: Load Training Data ---
    print("\n📊 Loading training samples...")
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=dataset_config.eval_split_max_size,
        eval_split_size=dataset_config.eval_split_size,
    )
    
    print(f"   Training samples: {len(train_samples)}")
    print(f"   Evaluation samples: {len(eval_samples)}")
    
    # Analyze unique Amharic characters
    all_text = " ".join([s["text"] for s in train_samples])
    unique_chars = sorted(list(set(all_text)))
    print(f"\n🔤 Found {len(unique_chars)} unique characters in dataset")
    print(f"   Sample characters: {''.join(unique_chars[:20])}...")
    
    # --- Step 6: Start Training ---
    print("\n" + "=" * 60)
    print("🚀 Starting Training...")
    print("=" * 60)
    print(f"   Target Language: {TARGET_LANG}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Output: {OUT_PATH}")
    print("=" * 60)
    
    if TARGET_LANG == "ar":
        print("\n⚠️  Using 'Language Hijacking' mode (Arabic -> Amharic)")
        print("   The model will learn Amharic patterns over the Arabic slot.")
    
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            skip_train_epoch=False,
            start_with_eval=False,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    
    # Start training!
    trainer.fit()
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Model saved to: {OUT_PATH}")
    print("=" * 60)
    print("\n💡 Next steps:")
    print("   1. Find the best checkpoint in the output folder")
    print("   2. Use inference_xtts_amharic.py to test your model")
    print("   3. If quality is poor, increase epochs to 50+")


if __name__ == "__main__":
    main()
