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
OUTPUT_DIR = "/ocean/projects/cis220031p/hkook/finetuned/Qwen2-Audio-7B-TEARS-CoT"
DATA_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
TRAIN_FILE = str(DATA_ROOT / "train_v3.json")
TEST_FILE = str(DATA_ROOT / "test_v3.json")

# WandB setup
os.environ["WANDB_PROJECT"] = "ALM-CoT-Finetuning"

# Training hyperparameters
BATCH_SIZE = 2            # Reduced from 8 to fix OOM
GRAD_ACCUMULATION = 8     # Increased to keep effective batch size ~16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 3
MAX_LENGTH = 4096         # Max number of text tokens (prompt + answer)


# ==========================
# Dataset loader
# ==========================

def load_tears_dataset():
    """Load TEARS_V2 dataset from local JSON files."""
    data_files = {"train": TRAIN_FILE, "test": TEST_FILE}
    dataset = load_dataset("json", data_files=data_files)
    return dataset


# ==========================
# Data collator
# ==========================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator that:
      - Pads text input_ids and labels using the tokenizer.
      - Converts padded label positions (pad_token_id) into -100 so they are ignored by the loss.
      - Pads audio features to the same temporal length and creates a feature_attention_mask.
    """
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Text fields: input_ids and labels are lists of variable-length token IDs
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

        # Convert pad_token_id in labels to -100 so they are ignored by cross-entropy loss
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels_batch[labels_batch == pad_token_id] = -100

        # 3) Audio features: list of [C, T] tensors
        input_features = [f["input_features"] for f in features]
        
        # Convert to tensor if list (happens when dataset is saved/loaded)
        input_features = [torch.tensor(f) if isinstance(f, list) else f for f in input_features]

        max_audio_len = max(feat.shape[-1] for feat in input_features)

        padded_features = []
        feature_attention_masks = []

        for feat in input_features:
            # feat shape: [C, T]
            C, T = feat.shape
            pad_len = max_audio_len - T

            # Zero-pad along the temporal dimension
            padded = torch.nn.functional.pad(feat, (0, pad_len), value=0.0)
            padded_features.append(padded)

            # Feature attention mask: 1 for valid frames, 0 for padding
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
# Training function
# ==========================

def train():
    # 1. Load model and processor
    print("Loading model and processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # 2. Configure LoRA (PEFT)
    #    Target common attention and MLP projection layers
    
    # Enable gradient checkpointing compatability by enabling input gradients
    model.enable_input_require_grads()
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=128,
        lora_alpha=256,       # Typically 2 * r
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Enable gradient checkpointing -> disable cache
    model.config.use_cache = False

    # 3. Load dataset
    dataset = load_tears_dataset()

    # 4. Preprocessing function
    def prepare_dataset(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        For each example:
          - Load audio and extract features with processor.feature_extractor.
          - Build a conversation in Qwen2-Audio's chat format using apply_chat_template.
          - Use two templates:
              conv_prompt: only user turn (with audio+question), with add_generation_prompt=True
              conv_full:   user + assistant(answer), with add_generation_prompt=False
          - Tokenize both texts; the prefix length of conv_prompt defines which tokens are prompt.
          - Labels:
              - prompt tokens -> -100
              - answer tokens -> actual token IDs
        """
        import librosa

        audio_paths = [DATA_ROOT / p for p in batch["audio_path"]]
        prompts = batch["prompt"]    # text question
        responses = batch["response"]  # CoT + answer

        input_features_list = []
        input_ids_list = []
        labels_list = []

        for audio_path, question, answer in zip(audio_paths, prompts, responses):
            # ---- Audio loading ----
            y, sr = librosa.load(
                audio_path,
                sr=processor.feature_extractor.sampling_rate,
            )

            # Extract audio features
            audio_inputs = processor.feature_extractor(
                y,
                sampling_rate=processor.feature_extractor.sampling_rate,
                return_tensors="pt",
            )
            # audio_inputs.input_features shape: [1, C, T]
            input_features = audio_inputs.input_features[0]  # [C, T]
            input_features_list.append(input_features)

            # ---- Chat template construction (text only) ----
            # We use a dummy "audio_url" because we are not reusing the multimodal processor here.
            # The important part is that the text template (special tokens) matches inference format.
            conv_prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "file://dummy"},
                        {"type": "text", "text": question},
                    ],
                }
            ]

            conv_full = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": "file://dummy"},
                        {"type": "text", "text": question},
                    ],
                },
                {
                    "role": "assistant",
                    "content": answer,
                },
            ]

            # Text for prompt-only (with generation prompt)
            prompt_text = processor.apply_chat_template(
                conv_prompt,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Text for full conversation (user + assistant answer)
            full_text = processor.apply_chat_template(
                conv_full,
                add_generation_prompt=False,
                tokenize=False,
            )

            # ---- Tokenize both ----
            prompt_ids = processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
            ).input_ids

            full_ids = processor.tokenizer(
                full_text,
                add_special_tokens=False,
            ).input_ids

            # Answer tokens are the suffix after the prompt tokens
            response_ids = full_ids[len(prompt_ids):]

            input_ids = full_ids
            # Labels: ignore prompt tokens (-100), train on answer tokens
            labels = [-100] * len(prompt_ids) + response_ids

            # Truncate if longer than MAX_LENGTH
            if len(input_ids) > MAX_LENGTH:
                input_ids = input_ids[:MAX_LENGTH]
                labels = labels[:MAX_LENGTH]

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        return {
            "input_features": input_features_list,
            "input_ids": input_ids_list,
            "labels": labels_list,
        }

    # 5. Apply preprocessing to train/test splits
    num_proc = 4  # Adjust based on CPU cores

    train_dataset = dataset["train"].map(
        prepare_dataset,
        batched=True,
        batch_size=8,
        remove_columns=dataset["train"].column_names,
        # num_proc=num_proc,
    )

    test_dataset = dataset["test"].map(
        prepare_dataset,
        batched=True,
        batch_size=8,
        remove_columns=dataset["test"].column_names,
        # num_proc=num_proc,
    )

    # Use a smaller subset for evaluation to speed it up (e.g., 1000 samples)
    if len(test_dataset) > 1000:
        test_dataset = test_dataset.select(range(1000))

    # 6. TrainingArguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        fp16=False,
        bf16=True,  # A100 supports bf16
        gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,  # Important for custom collator with audio
        dataloader_num_workers=4,
        report_to="wandb",
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=processor),
    )

    # 8. Train
    print("Starting training...")
    trainer.train()

    # 9. Save final model and processor
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    train()
