import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
from peft import PeftModel
from datasets import load_dataset
import librosa
import random
from pathlib import Path
import transformers

from transformers.models.qwen2_audio.processing_qwen2_audio import Qwen2AudioProcessor
# Configuration
CHECKPOINT_DIR = "/ocean/projects/cis220031p/hkook/finetuned/Qwen2-Audio-7B-TEARS-CoT/checkpoint-8500"
BASE_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"
DATA_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
TEST_FILE = str(DATA_ROOT / "test_v3.json")

def main():
    print(f"Loading model from {CHECKPOINT_DIR}...")
    
    # Load Processor
    # Use BASE_MODEL_ID for processor since checkpoint might not have all config files for auto processor
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    
    # Load Base Model
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        BASE_MODEL_ID,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # Load Adapters
    model = PeftModel.from_pretrained(model, CHECKPOINT_DIR)
    model.eval()
    
    print("Loading test dataset...")
    data_files = {"test": TEST_FILE}
    dataset = load_dataset("json", data_files=data_files)["test"]
    
    # Select 5 random samples
    indices = random.sample(range(len(dataset)), 50)
    samples = dataset.select(indices)
    
    print(f"Selected {len(samples)} random samples for inference.\n")
    
    for i, item in enumerate(samples):
        print(f"=== Sample {i+1} ===")
        audio_path_rel = item["audio_path"]
        full_audio_path = DATA_ROOT / audio_path_rel
        question = item["prompt"]
        ground_truth = item["response"] # Or "answer" depending on what we want to compare. "response" is CoT + Answer
        answer = item["answer"]

        print(f"Audio: {audio_path_rel}")
        print(f"Question: {question}")
        
        # Load audio
        y, sr = librosa.load(full_audio_path, sr=processor.feature_extractor.sampling_rate)
        
        # Prepare inputs
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "file://dummy"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        
        audio_inputs = processor(
            text=text,
            audio=y,
            sampling_rate=sr,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in audio_inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            
        # Decode
        # The generated_ids include input_ids, so we might want to decode only the new tokens
        # But usually decode(generated_ids) gives the full text. 
        # Qwen2Audio might repeat the prompt? Let's decode everything and strip prompt if needed.
        # Actually Qwen usually returns full sequence.
        
        generated_ids = generated_ids[:, inputs["input_ids"].size(1):] # Slice off prompt
        response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Ground Truth: {ground_truth}")
        print(f"Generated: {response}")
        print(f"Answer: {answer}")
        print("-" * 50 + "\n")

if __name__ == "__main__":
    main()

