"""
Dataset Preparation Script for Amharic TTS Training

This script helps you prepare your audio data for XTTS training.
It handles:
1. Audio resampling to required format
2. Audio segmentation into short clips
3. Metadata file generation

Requirements:
- Audio files (any format: mp3, wav, flac, etc.)
- Transcriptions
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is installed."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_audio(input_path: str, output_path: str, sample_rate: int = 22050):
    """Convert audio to required format using ffmpeg."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        "-acodec", "pcm_s16le",  # 16-bit PCM
        output_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def segment_audio(
    input_path: str, 
    output_dir: str, 
    segment_length: float = 8.0,
    min_length: float = 3.0
) -> list:
    """Split long audio into segments."""
    
    duration = get_audio_duration(input_path)
    if duration <= segment_length:
        return [input_path]  # No need to segment
    
    segments = []
    base_name = Path(input_path).stem
    
    start = 0
    segment_num = 1
    
    while start < duration:
        end = min(start + segment_length, duration)
        
        # Skip if remaining segment is too short
        if end - start < min_length:
            break
        
        output_path = os.path.join(output_dir, f"{base_name}_{segment_num:03d}.wav")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ss", str(start),
            "-t", str(segment_length),
            "-ar", "22050",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            segments.append(output_path)
            segment_num += 1
        
        start = end
    
    return segments


def create_dataset_structure(output_dir: str):
    """Create the required dataset folder structure."""
    wavs_dir = os.path.join(output_dir, "wavs")
    os.makedirs(wavs_dir, exist_ok=True)
    return wavs_dir


def validate_dataset(dataset_dir: str):
    """Validate the dataset structure and contents."""
    
    wavs_dir = os.path.join(dataset_dir, "wavs")
    metadata_path = os.path.join(dataset_dir, "metadata.csv")
    
    issues = []
    
    # Check structure
    if not os.path.exists(wavs_dir):
        issues.append("❌ Missing 'wavs' folder")
    
    if not os.path.exists(metadata_path):
        issues.append("❌ Missing 'metadata.csv'")
        return issues
    
    # Check metadata entries
    with open(metadata_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    if len(lines) == 0:
        issues.append("❌ metadata.csv is empty")
        return issues
    
    # Validate each entry
    missing_files = 0
    invalid_format = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('|')
        if len(parts) != 2:
            invalid_format += 1
            continue
        
        audio_file, text = parts
        audio_path = os.path.join(wavs_dir, audio_file)
        
        if not os.path.exists(audio_path):
            missing_files += 1
    
    if invalid_format > 0:
        issues.append(f"⚠️  {invalid_format} lines have invalid format (expected: filename.wav|text)")
    
    if missing_files > 0:
        issues.append(f"⚠️  {missing_files} audio files referenced in metadata are missing")
    
    # Audio stats
    wav_files = [f for f in os.listdir(wavs_dir) if f.endswith('.wav')] if os.path.exists(wavs_dir) else []
    
    total_duration = 0
    short_files = 0
    long_files = 0
    
    for wav_file in wav_files:
        duration = get_audio_duration(os.path.join(wavs_dir, wav_file))
        total_duration += duration
        if duration < 3:
            short_files += 1
        if duration > 10:
            long_files += 1
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Audio files: {len(wav_files)}")
    print(f"   Metadata entries: {len(lines)}")
    print(f"   Total duration: {total_duration/60:.1f} minutes")
    
    if short_files > 0:
        issues.append(f"⚠️  {short_files} files are shorter than 3 seconds (may cause issues)")
    
    if long_files > 0:
        issues.append(f"⚠️  {long_files} files are longer than 10 seconds (consider splitting)")
    
    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio dataset for Amharic TTS training"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Convert command
    convert_parser = subparsers.add_parser("convert", help="Convert audio files to required format")
    convert_parser.add_argument("input_dir", help="Directory with source audio files")
    convert_parser.add_argument("--output", "-o", default="./dataset", help="Output dataset directory")
    convert_parser.add_argument("--segment", action="store_true", help="Split long audio into segments")
    convert_parser.add_argument("--segment-length", type=float, default=8.0, help="Segment length in seconds")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset structure")
    validate_parser.add_argument("dataset_dir", nargs="?", default="./dataset", help="Dataset directory to validate")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize empty dataset structure")
    init_parser.add_argument("--output", "-o", default="./dataset", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎙️  TTS Dataset Preparation Tool")
    print("=" * 60)
    
    if args.command == "init":
        print(f"\n📁 Creating dataset structure in: {args.output}")
        wavs_dir = create_dataset_structure(args.output)
        
        # Create empty metadata file with example
        metadata_path = os.path.join(args.output, "metadata.csv")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            f.write("# Format: filename.wav|Amharic transcription\n")
            f.write("# Example:\n")
            f.write("# sample_001.wav|ሰላም ይህ የአማርኛ ድምጽ ናሙና ነው\n")
        
        print(f"✅ Created: {wavs_dir}")
        print(f"✅ Created: {metadata_path}")
        print("\n📝 Next steps:")
        print("   1. Add your .wav files to the 'wavs' folder")
        print("   2. Edit metadata.csv with transcriptions")
        print("   3. Run: python prepare_dataset.py validate")
        
    elif args.command == "convert":
        if not check_ffmpeg():
            print("❌ ffmpeg not found! Please install it:")
            print("   apt-get install -y ffmpeg")
            sys.exit(1)
        
        print(f"\n📂 Input: {args.input_dir}")
        print(f"📂 Output: {args.output}")
        
        wavs_dir = create_dataset_structure(args.output)
        
        # Find audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
        audio_files = []
        for f in os.listdir(args.input_dir):
            if Path(f).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(args.input_dir, f))
        
        print(f"\n🔄 Processing {len(audio_files)} audio files...")
        
        processed = 0
        for audio_file in audio_files:
            base_name = Path(audio_file).stem
            output_path = os.path.join(wavs_dir, f"{base_name}.wav")
            
            if args.segment:
                segments = segment_audio(audio_file, wavs_dir, args.segment_length)
                processed += len(segments)
                print(f"   ✅ {base_name} -> {len(segments)} segments")
            else:
                if convert_audio(audio_file, output_path):
                    processed += 1
                    print(f"   ✅ {base_name}.wav")
                else:
                    print(f"   ❌ {base_name} (conversion failed)")
        
        print(f"\n✅ Processed {processed} audio files")
        print("\n📝 Next steps:")
        print(f"   1. Create {os.path.join(args.output, 'metadata.csv')}")
        print("   2. Add transcriptions in format: filename.wav|text")
        print("   3. Run: python prepare_dataset.py validate")
        
    elif args.command == "validate":
        print(f"\n🔍 Validating dataset: {args.dataset_dir}")
        issues = validate_dataset(args.dataset_dir)
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ Dataset is valid and ready for training!")
            
    else:
        parser.print_help()
        print("\n📋 Quick Start:")
        print("   1. python prepare_dataset.py init")
        print("   2. Add audio files and transcriptions")
        print("   3. python prepare_dataset.py validate")


if __name__ == "__main__":
    main()
