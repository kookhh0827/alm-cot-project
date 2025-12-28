import os
import json
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from pathlib import Path

from datasets import load_dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


# ==========================
# Configuration
# ==========================

MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_DIR = "/ocean/projects/cis220031p/hkook/finetuned/Qwen2-Audio-7B-TEARS-CoT-V3"
DATA_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
# Use the recovered/final datasets
TRAIN_FILE = str(DATA_ROOT / "train_v3_final_recovered.json")
TEST_FILE = str(DATA_ROOT / "test_v3_final_recovered.json")

# WandB setup
os.environ["WANDB_PROJECT"] = "ALM-CoT-Finetuning-V3"
os.environ["WANDB_RESUME"] = "allow" # Resume run if ID matches, or start new if not
os.environ["WANDB_RUN_ID"] = "oadveyqe" # wandb run_id

# Training hyperparameters
BATCH_SIZE = 2            # A100 80GB -> 2~4 possible with LoRA
GRAD_ACCUMULATION = 8     # Effective BS = 16
LEARNING_RATE = 1e-4      # LoRA standard
NUM_EPOCHS = 3
MAX_LENGTH = 2048         # Context length


# ==========================
# Dataset loader
# ==========================

def load_tears_dataset():
    """Load TEARS_V2 recovered dataset from local JSON files."""
    data_files = {"train": TRAIN_FILE, "test": TEST_FILE}
    dataset = load_dataset("json", data_files=data_files)
    
    # Filter out failures (just in case)
    dataset = dataset.filter(lambda x: x["validation_status"] == "pass")
    print(f"Loaded dataset. Train: {len(dataset['train'])}, Test: {len(dataset['test'])}")
    return dataset


# ==========================
# Data collator
# ==========================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = [f["input_ids"] for f in features]
        labels = [f["labels"] for f in features]

        # 1) Pad input_ids
        text_batch = self.processor.tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        )
        batch_input_ids = text_batch["input_ids"]
        batch_attention_mask = text_batch["attention_mask"]

        # 2) Pad labels separately
        labels_batch = self.processor.tokenizer.pad(
            {"input_ids": labels},
            padding=True,
            return_tensors="pt",
        )["input_ids"]

        # Convert pad_token_id in labels to -100
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels_batch[labels_batch == pad_token_id] = -100

        # 3) Audio features
        input_features = [f["input_features"] for f in features]
        input_features = [torch.tensor(f) if isinstance(f, list) else f for f in input_features]

        max_audio_len = max(feat.shape[-1] for feat in input_features)
        padded_features = []
        feature_attention_masks = []

        for feat in input_features:
            C, T = feat.shape
            pad_len = max_audio_len - T
            padded = torch.nn.functional.pad(feat, (0, pad_len), value=0.0)
            padded_features.append(padded)
            
            mask = torch.ones(T, dtype=torch.long)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
            feature_attention_masks.append(mask)

        batch_input_features = torch.stack(padded_features)             # [B, C, T_max]
        batch_feature_attention_mask = torch.stack(feature_attention_masks)  # [B, T_max]

        batch = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": labels_batch,
            "input_features": batch_input_features,
            "feature_attention_mask": batch_feature_attention_mask,
        }
        return batch


# ==========================
# Helper: Flatten Latent
# ==========================
def flatten_latent_to_string(latent_json):
    flat_str = []
    # 1. Acoustic
    if "acoustic" in latent_json:
        for k, v in latent_json["acoustic"].items():
            flat_str.append(f"{k}={v}")
    # 2. Phonology
    if "phonology" in latent_json:
        for k, v in latent_json["phonology"].items():
            flat_str.append(f"{k}={v}")
            
    return ", ".join(flat_str)


# ==========================
# Training function
# ==========================

def train():
    print("Loading model and processor...")
    
    device_map = "auto"
    
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Enable gradient checkpointing
    model.enable_input_require_grads()
    
    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False

    print("Loading dataset...")
    dataset = load_tears_dataset()

    # Preprocessing
    def prepare_dataset(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        import librosa

        audio_paths = [DATA_ROOT / p for p in batch["audio_path"]]
        prompts = batch["prompt"]    
        responses = batch["generated_reasoning"] # The CoT + Answer block
        latents = batch["latent_json"]
        
        input_features_list = []
        input_ids_list = []
        labels_list = []
        
        # Define System Instruction
        SYSTEM_INSTRUCTION = (
            "You are a grounded speaker profiling assistant. "
            "First, analyze the audio to extract latent acoustic and phonological features. "
            "Output these features in a <LATENT> block. "
            "Then, use these features to THINK step-by-step and provide a final RESPONSE."
        )

        for audio_path, question, answer, latent_json in zip(audio_paths, prompts, responses, latents):
            # 1. Audio Feature Extraction
            try:
                y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
                audio_inputs = processor.feature_extractor(
                    y,
                    sampling_rate=processor.feature_extractor.sampling_rate,
                    return_tensors="pt",
                )
                input_features = audio_inputs.input_features[0]
                input_features_list.append(input_features)
            except Exception as e:
                print(f"Skipping bad audio: {audio_path} ({e})")
                continue

            # 2. Construct FULL TARGET RESPONSE (Latent + Think + Response)
            latent_str = flatten_latent_to_string(latent_json)
            full_assistant_response = f"<LATENT>\n{latent_str}\n</LATENT>\n\n{answer}"
            
            # 3. User Input (Audio + Instruction + Question)
            user_content = (
                f"{SYSTEM_INSTRUCTION}\n\n"
                f"Question: {question}"
            )
            
            # Chat Templates
            conv_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "file://dummy"},
                        {"type": "text", "text": user_content},
                    ],
                }
            ]
            
            conv_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "file://dummy"},
                        {"type": "text", "text": user_content},
                    ],
                },
                {
                    "role": "assistant",
                    "content": full_assistant_response, # Model learns to generate LATENT -> THINK -> RESPONSE
                },
            ]

            # Tokenize Prompt (Input)
            prompt_text = processor.apply_chat_template(conv_prompt, add_generation_prompt=True, tokenize=False)
            full_text = processor.apply_chat_template(conv_full, add_generation_prompt=False, tokenize=False)

            prompt_ids = processor.tokenizer(prompt_text, add_special_tokens=False).input_ids
            full_ids = processor.tokenizer(full_text, add_special_tokens=False).input_ids

            response_ids = full_ids[len(prompt_ids):]
            
            # Construct Labels (Mask prompt)
            labels = [-100] * len(prompt_ids) + response_ids
            
            # Truncate
            if len(full_ids) > MAX_LENGTH:
                full_ids = full_ids[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]

            input_ids_list.append(full_ids)
            labels_list.append(labels)

        return {
            "input_features": input_features_list,
            "input_ids": input_ids_list,
            "labels": labels_list,
        }

    print("Preprocessing train dataset...")
    train_dataset = dataset["train"].map(
        prepare_dataset,
        batched=True,
        batch_size=4,
        remove_columns=dataset["train"].column_names,
    )

    print("Preprocessing test dataset...")
    test_dataset = dataset["test"].map(
        prepare_dataset,
        batched=True,
        batch_size=4,
        remove_columns=dataset["test"].column_names,
    )
    
    # Subset for fast eval
    if len(test_dataset) > 500:
        test_dataset = test_dataset.select(range(500))

    # Training Args
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=False,
        bf16=True,
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=200,
        save_total_limit=10,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    print("Starting training...")
    
    # Auto-resume from checkpoint if exists
    resume_from_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        # Look for folders like "checkpoint-500"
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            resume_from_checkpoint = True
            print(f"Found existing checkpoints in {OUTPUT_DIR}. Resuming from latest checkpoint.")
            
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
