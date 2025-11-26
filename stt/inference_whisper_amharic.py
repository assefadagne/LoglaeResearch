"""
Amharic Speech-to-Text Inference Script
Use this script to transcribe Amharic audio using your fine-tuned Whisper model.
"""

import os
import sys
import argparse
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Default paths
DEFAULT_MODEL_PATH = "./whisper-large-v3-amharic-lora"
BASE_MODEL_NAME = "openai/whisper-large-v3"


def load_model(model_path: str = DEFAULT_MODEL_PATH, use_gpu: bool = True):
    """Load the fine-tuned Whisper model for inference."""
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading model on {device}...")
    
    # Load processor
    processor = WhisperProcessor.from_pretrained(model_path)
    
    # Check if this is a LoRA model or merged model
    if os.path.exists(os.path.join(model_path, "adapter_config.json")):
        # LoRA adapter - need to load base model and merge
        print("   Loading base model + LoRA adapters...")
        base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.merge_and_unload()
    else:
        # Already merged model
        print("   Loading merged model...")
        model = WhisperForConditionalGeneration.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully!")
    return model, processor, device


def transcribe_audio(
    audio_path: str,
    model,
    processor,
    device: str,
    language: str = "amharic",
    task: str = "transcribe"
) -> str:
    """Transcribe an audio file to text."""
    
    # Load and preprocess audio
    print(f"🎵 Processing: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=16000)
    
    # Extract features
    input_features = processor(
        audio, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_features.to(device)
    
    # Generate transcription
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language=language,
            task=task,
            max_length=448,
        )
    
    # Decode
    transcription = processor.batch_decode(
        predicted_ids, 
        skip_special_tokens=True
    )[0]
    
    return transcription


def transcribe_batch(
    audio_paths: list,
    model,
    processor,
    device: str,
    language: str = "amharic"
) -> list:
    """Transcribe multiple audio files."""
    results = []
    for path in audio_paths:
        try:
            text = transcribe_audio(path, model, processor, device, language)
            results.append({"file": path, "transcription": text, "error": None})
            print(f"   ✅ {os.path.basename(path)}: {text[:50]}...")
        except Exception as e:
            results.append({"file": path, "transcription": None, "error": str(e)})
            print(f"   ❌ {os.path.basename(path)}: Error - {e}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe Amharic audio using fine-tuned Whisper model"
    )
    parser.add_argument(
        "audio_files",
        nargs="+",
        help="Path(s) to audio file(s) to transcribe"
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the fine-tuned model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--language",
        default="amharic",
        help="Language for transcription (default: amharic)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (slower but works without GPU)"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file for transcriptions (optional)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  Amharic Speech-to-Text Inference")
    print("=" * 60)
    
    # Load model
    model, processor, device = load_model(
        model_path=args.model_path,
        use_gpu=not args.cpu
    )
    
    # Transcribe
    print(f"\n📝 Transcribing {len(args.audio_files)} file(s)...\n")
    results = transcribe_batch(
        args.audio_files,
        model,
        processor,
        device,
        args.language
    )
    
    # Output results
    print("\n" + "=" * 60)
    print("📋 Results:")
    print("=" * 60)
    
    for result in results:
        filename = os.path.basename(result["file"])
        if result["error"]:
            print(f"\n❌ {filename}")
            print(f"   Error: {result['error']}")
        else:
            print(f"\n✅ {filename}")
            print(f"   {result['transcription']}")
    
    # Save to file if requested
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            for result in results:
                filename = os.path.basename(result["file"])
                text = result["transcription"] or f"ERROR: {result['error']}"
                f.write(f"{filename}\t{text}\n")
        print(f"\n💾 Results saved to: {args.output}")
    
    print("\n🎉 Done!")


if __name__ == "__main__":
    main()
