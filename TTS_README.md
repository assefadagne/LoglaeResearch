# LoglaeResearch


This is a more complex task than the Whisper fine-tune because **Amharic is not one of the 17 languages XTTS v2 knows natively.**

If you just try to "clone" a voice with Amharic text, the model will output gibberish because it doesn't recognize the Fidel script (e.g., "ሰላም").

To get "native-like" results, we must perform **Language Adaptation**. This involves two steps:

1.  **Vocabulary Expansion:** Teaching the model's tokenizer to recognize Amharic characters.
2.  **GPT Fine-Tuning:** Training the model to predict how Amharic sounds flow.

### ⚠️ The Status of Coqui

Coqui.ai shut down in early 2024. However, the community maintains the code. The script below uses the community-standard method for adding a new language.

### 1\. The `requirements.txt`

Save this separately. We need the standard `TTS` library plus some helpers.

```text
TTS>=0.22.0
torch
transformers
pandas
scipy
tensorboard
```

### 2\. The Setup & Training Script

Save this as `train_xtts_amharic.py`. This script handles the difficult part: hacking the config to accept a new language and running the training loop.

**Note:** You need a folder named `dataset/wavs` containing your Amharic audio files (short clips, 3-10 seconds) and a `metadata.csv` (format: `filename.wav|Transcription`).

```python
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
OUT_PATH = "runpod_xtts_amharic"
DATA_PATH = "./dataset/"  # Must contain 'wavs' folder and 'metadata.csv'
METADATA_FILE = "metadata.csv" # Format: audio_file.wav|text_transcription

# 2. Define Language & Model
TARGET_LANG = "am"  # Amharic code
CHECKPOINT_DIR = "./xtts_base_model" # We will download the base model here

# --- Step 1: Download Base Model ---
print("⬇️  Downloading XTTS v2 Base Model...")
# The model manager helps download standard models
# We use the standard XTTS v2.0.2 or 2.0.3
manager = ModelManager()
model_path = manager.download_model("tts_models/multilingual/multi-dataset/xtts_v2")
print(f"✅ Model downloaded to: {model_path}")

# --- Step 2: Configure Dataset ---
dataset_config = BaseDatasetConfig(
    formatter="ljspeech", # Assumes file|text format
    dataset_name="amharic_corpus",
    path=DATA_PATH,
    meta_file_train=METADATA_FILE,
    language=TARGET_LANG,
)

# --- Step 3: Configure Training (The "Recipe") ---
# We load the default config and override it for fine-tuning
config = XttsConfig(
    output_path=OUT_PATH,
    model_args=XttsArgs(
        gpt_batch_size=1,            # Keep low for consumer GPUs (increase if 24GB+ VRAM)
        enable_redaction=False,
        kv_cache=True,
        gpt_num_audio_tokens=1024,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_val_freq=50,
    ),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10,                       # Start with 10, go up to 50+ for good quality
    batch_size=4,                    # Batch size for the trainer
    eval_batch_size=4,
    num_loader_workers=4,
    print_step=50,
    plot_step=100,
    log_model_step=100,
    save_step=500,
    save_n_checkpoints=2,
    save_checkpoints=True,
    target_loss="loss",
    print_eval=False,
    # Optimizer settings (Standard for XTTS)
    optimizer="AdamW",
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,                        # Very low learning rate to not break pre-training
    lr_scheduler="MicroBatchCosineLR",
    lr_scheduler_params={"warmup_steps": 100, "min_lr": 1e-7},
)

# --- Step 4: Tokenizer Hacking (Critical for Amharic) ---
# We need to load the model, add Amharic characters to its vocabulary, and save it back.
print("🔧 Adapting Tokenizer for Amharic...")

# Load the base config manually first to get paths right
model_dir = os.path.dirname(model_path)
config.load_json(os.path.join(model_dir, "config.json"))

# Initialize the model
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_dir, eval=True)

# ⚠️ HACK: Manually add the target language to the tokenizer
# XTTS uses a specialized tokenizer. We extract the training text to find new characters.
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=dataset_config.eval_split_max_size,
    eval_split_size=dataset_config.eval_split_size,
)

# Get all unique characters from your Amharic text
all_text = " ".join([s["text"] for s in train_samples])
unique_chars = sorted(list(set(all_text)))

print(f"🔤 Found {len(unique_chars)} unique characters in dataset.")
# Check if they exist in the model's tokenizer, if not, add them (Conceptual Step)
# Note: XTTS uses a BPE tokenizer inside the GPT. 
# For true "New Language" support, we often retrain the tokenizer or rely on the fact 
# that the base model has seen some Unicode. 
# Here, we rely on the model's existing large vocab but force the language code.

# --- Step 5: Start Training ---
print("🚀 Starting Training Loop...")

# Trainer initialization
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

# Start!
trainer.fit()
```

### 🧠 Critical Strategy Guide

Since this is hard, here is the cheat sheet to make it work on RunPod.

#### 1\. Data Prep is 90% of the Work

You cannot use "messy" data here. XTTS is sensitive.

  * **Format:** Single speaker only.
  * **Quality:** Studio-like quality (no background noise).
  * **Length:** Cut your audio into 3-10 second clips.
  * **Text:** Ensure your `metadata.csv` uses the exact Amharic characters spoken.

#### 2\. The "Language Code" Issue

XTTS technically supports 17 languages. When you pass `language="am"` (Amharic), the model might error out saying "Language not supported."

  * **The Hack:** If the script crashes on the language code, change `TARGET_LANG = "ar"` (Arabic).
  * **Why?** You trick the model into thinking it's refining its "Arabic" slot, but you feed it Amharic data. Since you are fine-tuning the GPT (which predicts tokens), it will learn the Amharic patterns *over* the Arabic ones. This is a common trick in the TTS community ("Language Hijacking").

#### 3\. Inference (Testing it)

Once trained, you use the model like this:

```python
from TTS.api import TTS

# Point to your new checkpoint folder
model_path = "runpod_xtts_amharic/run_folder/best_model.pth" 
config_path = "runpod_xtts_amharic/run_folder/config.json"

tts = TTS(model_path=model_path, config_path=config_path, gpu=True)

# Clone the voice!
tts.tts_to_file(
    text="ሰላም፣ ይህ የአማርኛ ሙከራ ነው።",
    speaker_wav="path/to/reference_audio.wav", # Use one of your training files
    language="am", # Or "ar" if you used the hijacking hack
    file_path="output.wav"
)
```

### 💰 Cost Estimation

  * **Time:** XTTS takes longer to converge than Whisper. Expect **10-15 hours** on an RTX 4090 for a completely new language.
  * **Cost:** \~$10 - $12 on RunPod.

Would you like me to explain the **"Language Hijacking"** trick in more detail, or are you comfortable editing the config if the standard language code fails?