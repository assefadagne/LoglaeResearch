"""
Amharic Text-to-Speech Inference Script
Use this script to generate Amharic speech from text using your fine-tuned XTTS model.
"""

import os
import sys
import argparse
import torch
from TTS.api import TTS

# Default paths
DEFAULT_MODEL_PATH = "./xtts_amharic_model"
DEFAULT_LANGUAGE = "am"  # Change to "ar" if you used Language Hijacking


def find_best_model(model_dir: str) -> tuple:
    """Find the best model checkpoint in the output directory."""
    
    # Look for run folders
    run_folders = [d for d in os.listdir(model_dir) if d.startswith("run")]
    
    if not run_folders:
        # Check if model files are directly in the directory
        if os.path.exists(os.path.join(model_dir, "best_model.pth")):
            return (
                os.path.join(model_dir, "best_model.pth"),
                os.path.join(model_dir, "config.json")
            )
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Get the latest run folder
    latest_run = sorted(run_folders)[-1]
    run_path = os.path.join(model_dir, latest_run)
    
    # Look for best_model.pth
    best_model = os.path.join(run_path, "best_model.pth")
    config_file = os.path.join(run_path, "config.json")
    
    if os.path.exists(best_model):
        return best_model, config_file
    
    # Fallback to checkpoint files
    checkpoints = [f for f in os.listdir(run_path) if f.startswith("checkpoint_")]
    if checkpoints:
        latest_checkpoint = sorted(checkpoints)[-1]
        return (
            os.path.join(run_path, latest_checkpoint),
            config_file
        )
    
    raise FileNotFoundError(f"No model checkpoint found in {run_path}")


def load_model(model_path: str = None, config_path: str = None, use_gpu: bool = True):
    """Load the fine-tuned XTTS model for inference."""
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"🔧 Loading model on {device}...")
    
    if model_path is None:
        # Auto-detect model path
        model_path, config_path = find_best_model(DEFAULT_MODEL_PATH)
    
    print(f"   Model: {model_path}")
    print(f"   Config: {config_path}")
    
    tts = TTS(model_path=model_path, config_path=config_path, gpu=(device == "cuda"))
    
    print("✅ Model loaded successfully!")
    return tts


def synthesize_speech(
    tts,
    text: str,
    speaker_wav: str,
    output_path: str = "output.wav",
    language: str = DEFAULT_LANGUAGE
) -> str:
    """Generate speech from text."""
    
    print(f"\n🎵 Synthesizing speech...")
    print(f"   Text: {text[:50]}{'...' if len(text) > 50 else ''}")
    print(f"   Speaker reference: {speaker_wav}")
    print(f"   Language: {language}")
    
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path
    )
    
    print(f"✅ Audio saved to: {output_path}")
    return output_path


def synthesize_batch(
    tts,
    texts: list,
    speaker_wav: str,
    output_dir: str = "./outputs",
    language: str = DEFAULT_LANGUAGE
) -> list:
    """Generate speech for multiple texts."""
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i, text in enumerate(texts, 1):
        output_path = os.path.join(output_dir, f"output_{i:03d}.wav")
        try:
            synthesize_speech(tts, text, speaker_wav, output_path, language)
            results.append({"text": text, "audio": output_path, "error": None})
        except Exception as e:
            print(f"❌ Error synthesizing text {i}: {e}")
            results.append({"text": text, "audio": None, "error": str(e)})
    
    return results


def interactive_mode(tts, speaker_wav: str, language: str):
    """Interactive text-to-speech mode."""
    
    print("\n" + "=" * 60)
    print("🎤 Interactive Mode")
    print("=" * 60)
    print("Type Amharic text and press Enter to synthesize.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    counter = 1
    while True:
        try:
            text = input("📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not text:
                continue
            
            output_path = f"interactive_output_{counter:03d}.wav"
            synthesize_speech(tts, text, speaker_wav, output_path, language)
            counter += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Generate Amharic speech using fine-tuned XTTS model"
    )
    
    parser.add_argument(
        "--text", "-t",
        help="Text to synthesize (Amharic)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Text file with lines to synthesize"
    )
    parser.add_argument(
        "--speaker-wav", "-s",
        required=True,
        help="Path to speaker reference audio (3-10 seconds)"
    )
    parser.add_argument(
        "--output", "-o",
        default="output.wav",
        help="Output audio file path (default: output.wav)"
    )
    parser.add_argument(
        "--model-path", "-m",
        help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--config-path", "-c",
        help="Path to config.json"
    )
    parser.add_argument(
        "--language", "-l",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE}, use 'ar' if Language Hijacking was used)"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive mode"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🗣️  Amharic Text-to-Speech Inference")
    print("=" * 60)
    
    # Load model
    tts = load_model(
        model_path=args.model_path,
        config_path=args.config_path,
        use_gpu=not args.cpu
    )
    
    # Interactive mode
    if args.interactive:
        interactive_mode(tts, args.speaker_wav, args.language)
        return
    
    # Batch mode from file
    if args.file:
        print(f"\n📄 Reading texts from: {args.file}")
        with open(args.file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        output_dir = os.path.splitext(args.output)[0] + "_outputs"
        results = synthesize_batch(tts, texts, args.speaker_wav, output_dir, args.language)
        
        print(f"\n📊 Results: {sum(1 for r in results if r['audio'])} / {len(results)} succeeded")
        return
    
    # Single text mode
    if args.text:
        synthesize_speech(tts, args.text, args.speaker_wav, args.output, args.language)
        return
    
    # No input provided - show help
    print("\n⚠️  No input provided!")
    print("\nExamples:")
    print(f"  python {sys.argv[0]} -t 'ሰላም፣ እንዴት ነህ?' -s reference.wav")
    print(f"  python {sys.argv[0]} -f texts.txt -s reference.wav")
    print(f"  python {sys.argv[0]} -i -s reference.wav  # Interactive mode")


if __name__ == "__main__":
    main()
