# Amharic Speech-to-Text (STT)

Fine-tune OpenAI's Whisper Large V3 on Amharic speech using LoRA adapters for efficient training.

## Features

- **Model**: Whisper Large V3 with LoRA adapters
- **Dataset**: Mozilla Common Voice Amharic
- **Techniques**: 8-bit quantization + LoRA for memory efficiency
- **Hardware**: Optimized for RTX 4090 (24GB VRAM)

## Files

| File | Description |
|------|-------------|
| `finetune_whisper_amharic.py` | Main training script |
| `inference_whisper_amharic.py` | Transcribe audio files |
| `requirements.txt` | Python dependencies |

## Quick Start (RunPod)

### 1. Setup Environment

```bash
# Install system dependencies
apt-get update && apt-get install -y ffmpeg

# Install Python packages
pip install -r requirements.txt
```

### 2. Run Training

```bash
python finetune_whisper_amharic.py
```

The script will:
1. Download Mozilla Common Voice Amharic dataset
2. Load Whisper Large V3 with 8-bit quantization
3. Apply LoRA adapters for efficient fine-tuning
4. Train and save to `./whisper-large-v3-amharic-lora`

### 3. Run Inference

```bash
# Single file
python inference_whisper_amharic.py audio.wav

# Multiple files
python inference_whisper_amharic.py file1.wav file2.wav file3.wav

# Save to output file
python inference_whisper_amharic.py audio.wav -o transcription.txt

# Use CPU (slower)
python inference_whisper_amharic.py audio.wav --cpu
```

## Configuration

Edit `finetune_whisper_amharic.py` to customize:

```python
# Training hyperparameters
BATCH_SIZE = 8                    # Reduce if OOM errors
GRADIENT_ACCUMULATION_STEPS = 2   # Effective batch = 16
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
MAX_STEPS = 4000                  # Set to -1 for full epochs

# LoRA configuration
LORA_R = 32                       # LoRA rank
LORA_ALPHA = 64                   # LoRA alpha
LORA_DROPOUT = 0.05
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `BATCH_SIZE` to 4 or 2
- Increase `GRADIENT_ACCUMULATION_STEPS` to compensate

### Dataset Download Fails
Download manually from [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets):
1. Download Amharic dataset
2. Extract to project directory
3. Modify script to use local path

### CUDA Errors
Ensure you're using a compatible PyTorch + CUDA version:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Output

After training:
```
whisper-large-v3-amharic-lora/
├── adapter_config.json
├── adapter_model.bin
├── preprocessor_config.json
├── tokenizer.json
└── ...
```

## Estimated Resources

| Resource | Estimate |
|----------|----------|
| Training Time | 3-5 hours |
| GPU VRAM | ~20GB |
| RunPod Cost | ~$3-4 (RTX 4090) |
