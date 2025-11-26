# Amharic Text-to-Speech (TTS)

Fine-tune Coqui XTTS v2 for Amharic language support using Language Adaptation.

## ⚠️ Important Notes

- **Amharic is NOT natively supported** by XTTS v2 (only 17 languages)
- We use **Language Adaptation** to teach the model Amharic
- Requires **Python 3.9-3.11** (not 3.12+)
- Training takes **10-15 hours** on RTX 4090

## Features

- **Model**: XTTS v2 with Language Adaptation
- **Techniques**: Vocabulary Expansion + GPT Fine-Tuning
- **Fallback**: "Language Hijacking" trick if native code fails

## Files

| File | Description |
|------|-------------|
| `train_xtts_amharic.py` | Main training script |
| `inference_xtts_amharic.py` | Generate speech from text |
| `prepare_dataset.py` | Dataset preparation helper |
| `requirements.txt` | Python dependencies |

## Dataset Preparation

### Required Structure

```
dataset/
├── wavs/
│   ├── audio_001.wav
│   ├── audio_002.wav
│   └── ...
└── metadata.csv
```

### metadata.csv Format

```
audio_001.wav|ሰላም ይህ የአማርኛ ድምጽ ናሙና ነው
audio_002.wav|እንዴት ነህ
audio_003.wav|ደህና ነኝ እናመሰግናለሁ
```

### Audio Requirements

| Requirement | Specification |
|-------------|---------------|
| Format | WAV (16-bit PCM) |
| Sample Rate | 22050 Hz |
| Channels | Mono |
| Duration | 3-10 seconds per clip |
| Quality | Studio-like (no background noise) |
| Speaker | Single speaker only |

### Using the Preparation Tool

```bash
# Initialize empty dataset structure
python prepare_dataset.py init

# Convert audio files from various formats
python prepare_dataset.py convert /path/to/audio/files

# Split long audio into segments
python prepare_dataset.py convert /path/to/audio --segment

# Validate your dataset before training
python prepare_dataset.py validate
```

## Quick Start (RunPod)

### 1. Setup Environment

```bash
# Install system dependencies
apt-get update && apt-get install -y ffmpeg

# Install Python packages (requires Python 3.9-3.11)
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
# Create dataset structure
python prepare_dataset.py init

# Add your audio files to dataset/wavs/
# Edit dataset/metadata.csv with transcriptions

# Validate dataset
python prepare_dataset.py validate
```

### 3. Run Training

```bash
python train_xtts_amharic.py
```

### 4. Run Inference

```bash
# Single text
python inference_xtts_amharic.py -t "ሰላም፣ እንዴት ነህ?" -s reference.wav

# From text file
python inference_xtts_amharic.py -f texts.txt -s reference.wav -o outputs/

# Interactive mode
python inference_xtts_amharic.py -i -s reference.wav
```

## Configuration

Edit `train_xtts_amharic.py` to customize:

```python
# Paths
OUT_PATH = "xtts_amharic_model"
DATA_PATH = "./dataset/"

# Language (change to "ar" if "am" fails)
TARGET_LANG = "am"

# Training
NUM_EPOCHS = 10          # Start with 10, go to 50+ for quality
BATCH_SIZE = 4
LEARNING_RATE = 5e-06    # Very low to preserve pre-training
```

## Language Hijacking Trick

If you get "Language not supported" error for Amharic (`am`):

1. Change `TARGET_LANG = "ar"` (Arabic) in the training script
2. The model learns Amharic patterns over the Arabic slot
3. Use `--language ar` during inference

This is a common trick in the TTS community for unsupported languages.

## Troubleshooting

### "Language not supported" Error
Change `TARGET_LANG = "ar"` in `train_xtts_amharic.py`

### Out of Memory
- Reduce `gpt_batch_size` to 1
- Reduce `batch_size` to 2
- Use a GPU with more VRAM

### Poor Audio Quality
- Increase training epochs (50+)
- Improve dataset quality (cleaner audio)
- Use more training data (2+ hours)

### Python Version Error
XTTS requires Python 3.9-3.11:
```bash
# Check Python version
python --version

# On RunPod, the PyTorch template uses Python 3.10
```

## Output

After training:
```
xtts_amharic_model/
└── run_YYYYMMDD-HHMMSS/
    ├── best_model.pth
    ├── config.json
    ├── checkpoint_XXXXX.pth
    └── ...
```

## Inference Examples

```python
from TTS.api import TTS

# Load model
tts = TTS(
    model_path="xtts_amharic_model/run_.../best_model.pth",
    config_path="xtts_amharic_model/run_.../config.json",
    gpu=True
)

# Generate speech
tts.tts_to_file(
    text="ሰላም፣ ይህ የአማርኛ ሙከራ ነው።",
    speaker_wav="reference.wav",
    language="am",  # or "ar" if using hijacking
    file_path="output.wav"
)
```

## Estimated Resources

| Resource | Estimate |
|----------|----------|
| Training Time | 10-15 hours |
| GPU VRAM | ~16-20GB |
| Dataset Size | 1-3 hours of audio |
| RunPod Cost | ~$10-12 (RTX 4090) |

## Tips for Best Results

1. **Data Quality**: Clean, studio-quality audio is essential
2. **Single Speaker**: Use only one speaker in your dataset
3. **Consistent Volume**: Normalize audio levels
4. **Accurate Transcriptions**: Ensure text matches exactly what's spoken
5. **More Data**: 2+ hours of audio improves quality significantly
