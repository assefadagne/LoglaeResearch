"""
Amharic Speech-to-Text Fine-tuning Script
Fine-tunes OpenAI's Whisper Large V3 on Mozilla Common Voice Amharic dataset
using LoRA (Low-Rank Adaptation) for efficient training on consumer GPUs.
"""

import os
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from datasets import load_dataset, Audio, DatasetDict
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate

# --- Configuration ---
MODEL_NAME = "openai/whisper-large-v3"
LANGUAGE = "amharic"
LANGUAGE_CODE = "am"
TASK = "transcribe"
OUTPUT_DIR = "./whisper-large-v3-amharic-lora"

# Training hyperparameters (optimized for RTX 4090 with 24GB VRAM)
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
WARMUP_STEPS = 500
MAX_STEPS = 4000  # Set to -1 for full training based on epochs

# LoRA configuration
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05


def main():
    print("=" * 60)
    print("🎙️  Amharic Whisper Fine-tuning Script")
    print("=" * 60)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n📊 Device: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # --- Step 1: Load Dataset ---
    print("\n⬇️  Loading Mozilla Common Voice Amharic dataset...")
    try:
        # Try loading from Hugging Face Hub
        common_voice = DatasetDict()
        common_voice["train"] = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            LANGUAGE_CODE,
            split="train+validation",
            trust_remote_code=True,
        )
        common_voice["test"] = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            LANGUAGE_CODE,
            split="test",
            trust_remote_code=True,
        )
        print(f"✅ Dataset loaded: {len(common_voice['train'])} train, {len(common_voice['test'])} test samples")
    except Exception as e:
        print(f"⚠️  Error loading from HuggingFace: {e}")
        print("   Attempting to load from local 'data/' directory...")
        # Fallback to local data if available
        raise RuntimeError(
            "Dataset not found. Please download the Amharic Common Voice dataset manually.\n"
            "Visit: https://commonvoice.mozilla.org/en/datasets"
        )
    
    # --- Step 2: Load Processor and Model ---
    print(f"\n📦 Loading {MODEL_NAME}...")
    
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, 
        language=LANGUAGE, 
        task=TASK
    )
    
    # Load model with 8-bit quantization for memory efficiency
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Disable cache for training (incompatible with gradient checkpointing)
    model.config.use_cache = False
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    # --- Step 3: Configure LoRA ---
    print("\n🔧 Configuring LoRA adapters...")
    
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
        bias="none",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # --- Step 4: Prepare Dataset ---
    print("\n🔄 Preprocessing audio data...")
    
    # Remove unnecessary columns
    common_voice = common_voice.remove_columns(
        ["accent", "age", "client_id", "down_votes", "gender", 
         "locale", "path", "segment", "up_votes", "variant"]
    )
    
    # Resample audio to 16kHz (Whisper's expected sample rate)
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    
    def prepare_dataset(batch):
        """Prepare a single batch for training."""
        audio = batch["audio"]
        
        # Extract features from audio
        batch["input_features"] = processor.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        # Encode target text
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
        
        return batch
    
    # Process datasets
    common_voice = common_voice.map(
        prepare_dataset,
        remove_columns=common_voice.column_names["train"],
        num_proc=4,  # Parallel processing
    )
    
    print(f"✅ Preprocessing complete!")
    
    # --- Step 5: Data Collator ---
    @dataclass
    class DataCollatorSpeechSeq2SeqWithPadding:
        """Custom data collator for Whisper training."""
        processor: Any
        
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            # Split inputs and labels
            input_features = [{"input_features": f["input_features"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            
            # Pad input features
            batch = self.processor.feature_extractor.pad(
                input_features, 
                return_tensors="pt"
            )
            
            # Pad labels
            labels_batch = self.processor.tokenizer.pad(
                label_features, 
                return_tensors="pt"
            )
            
            # Replace padding with -100 for loss computation
            labels = labels_batch["input_ids"].masked_fill(
                labels_batch.attention_mask.ne(1), -100
            )
            
            # Remove BOS token if present (will be added by model)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            
            batch["labels"] = labels
            return batch
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    # --- Step 6: Evaluation Metric ---
    print("\n📊 Setting up evaluation metrics...")
    
    wer_metric = evaluate.load("wer")
    
    def compute_metrics(pred):
        """Compute Word Error Rate (WER) for evaluation."""
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        
        # Decode predictions and references
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        
        return {"wer": wer}
    
    # --- Step 7: Training Arguments ---
    print("\n⚙️  Configuring training...")
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        gradient_checkpointing=True,
        fp16=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        logging_steps=25,
        report_to=["tensorboard"],
        push_to_hub=False,
        predict_with_generate=True,
        generation_max_length=225,
    )
    
    # --- Step 8: Initialize Trainer ---
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    
    # --- Step 9: Train! ---
    print("\n" + "=" * 60)
    print("🚀 Starting Training...")
    print("=" * 60)
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE} (effective: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    trainer.train()
    
    # --- Step 10: Save Final Model ---
    print("\n💾 Saving final model...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("=" * 60)
    
    # --- Step 11: Quick Test ---
    print("\n🧪 Running quick inference test...")
    
    # Load the fine-tuned model for inference
    from peft import PeftModel
    
    base_model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model = model.merge_and_unload()
    model.to(device)
    
    # Test with a sample from test set
    sample = common_voice["test"][0]
    input_features = processor(
        sample["input_features"], 
        return_tensors="pt"
    ).input_features.to(device)
    
    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"   Sample transcription: {transcription}")
    
    print("\n🎉 All done! Your Amharic STT model is ready.")


if __name__ == "__main__":
    main()
