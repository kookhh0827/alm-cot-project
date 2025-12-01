import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, List, Dict, Tuple, Optional
import textgrid
from tqdm import tqdm

# Add parent directory to path to import from parallel modules if needed
# (Assuming this script runs from alm-cot-project root or data-generation folder)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Rename import from 'abc' to avoid conflict with standard library 'abc' module
# Renamed local file to 'data_gen_abc.py'
from data_gen_abc import (
    AbstractDatasetProcessor,
    AbstractAudioFeatExtractor,
    AbstractAlignedPhonemesExtractor,
    AbstractTrainingDataGenerator,
    AbstractDataValidator,
)
from schema import (
    Audio,
    Question,
    Answer,
    Transcript,
    DatasetItem,
    AudioFeatures,
    AlignedPhoneme,
    AlignedPhonemes,
    ProcessedSample,
    CoTReasoningTrace,
    ValidationResult,
)
from pipeline import DataGenPipeline, PipelineConfig

# --- Constants ---
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
TEARS_MFA_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_MFA")

# --- Implementations ---

class TearsDatasetProcessor(AbstractDatasetProcessor):
    def __init__(self, split: str = "train", limit: Optional[int] = None) -> None:
        self.split = split
        self.limit = limit
        self.json_path = TEARS_V2_ROOT / f"{split}.json"
        
    def iterate(self, dataset: Any = None) -> Iterator[DatasetItem]:
        if not self.json_path.exists():
            raise FileNotFoundError(f"Dataset JSON not found: {self.json_path}")
            
        with open(self.json_path, "r") as f:
            data = json.load(f)
            
        count = 0
        # Pre-filter or process prompt/responses properly
        # User request: handle age, ethnicity, dialect_region, gender only.
        # If null, skip.
        
        target_attributes = {
            "gender": "What is the speaker's gender?",
            "age": "What is the speaker's age?",
            "dialect_region": "What is the speaker's dialect?",
            "ethnicity": "What is the speaker's ethnicity?" 
        }
        
        for entry in data:
            if self.limit and count >= self.limit:
                break
                
            # Speaker info
            speaker = entry.get("speaker", {})
            if not speaker:
                continue
                
            # Path in JSON is relative
            rel_path = entry.get("audio_path", "")
            full_audio_path = TEARS_V2_ROOT / rel_path
            
            # Transcript
            transcript_text = entry.get("transcript", "")
            
            # Generate items for each target attribute if metadata exists
            for attr, prompt_text in target_attributes.items():
                val = speaker.get(attr)
                    
                if val is None:
                    continue # Skip if metadata is missing
                
                # Create DatasetItem
                # We use the metadata value as the ground truth 'answer' for context
                # The generation model will use this 'answer' in the prompt to generate the CoT trace.
                
                yield DatasetItem(
                    audio=Audio(path=full_audio_path),
                    question=Question(text=prompt_text),
                    answer=Answer(text=str(val)),
                    transcript=Transcript(text=transcript_text),
                )
                count += 1

class TearsAudioFeatExtractor(AbstractAudioFeatExtractor):
    def extract(self, audio: Audio) -> AudioFeatures:
        # Expecting .AF file in the same directory as .wav
        wav_path = audio.path
        af_path = wav_path.with_suffix(".AF")
        
        features = {}
        if af_path.exists():
            try:
                with open(af_path, "r") as f:
                    raw_features = json.load(f)
                    # Round features to 2 decimal places
                    features = {k: round(float(v), 2) for k, v in raw_features.items()}
            except Exception as e:
                print(f"Error loading AF {af_path}: {e}")
        else:
            pass 
            # print(f"Warning: AF file missing for {wav_path}") # Too noisy for full dataset
            
        return AudioFeatures(values=features)

class TearsAlignedPhonemesExtractor(AbstractAlignedPhonemesExtractor):
    def align(self, audio: Audio, transcript: Transcript) -> AlignedPhonemes:
        # TextGrid is in TEARS_MFA_ROOT
        # Structure: TEARS_MFA_ROOT / <rel_path_without_filename> / <filename>.TextGrid
        # But wait, TEARS_V2 has "ears_dataset" and "timit_dataset".
        # TEARS_MFA also has "ears_dataset" and "timit_dataset" as per `ls` output.
        # So relative structure should match.
        
        # audio.path: .../TEARS_V2/ears_dataset/train/p001/file.wav
        # rel_path from TEARS_V2_ROOT: ears_dataset/train/p001/file.wav
        
        try:
            rel_path = audio.path.relative_to(TEARS_V2_ROOT)
        except ValueError:
            # Fallback if path is not absolute or wrong root
            # Assuming audio.path is absolute
            return []

        # Construct TextGrid path
        # TEARS_MFA / ears_dataset / train / p001 / file.TextGrid
        tg_path = TEARS_MFA_ROOT / rel_path.with_suffix(".TextGrid")
        
        phonemes = []
        if tg_path.exists():
            try:
                # Using textgrid library correctly
                tg = textgrid.TextGrid.fromFile(str(tg_path))
                
                # Find phone tier
                phone_tier = None
                for tier in tg.tiers:
                    if "phone" in tier.name.lower():
                        phone_tier = tier
                        break
                
                if phone_tier:
                    for interval in phone_tier:
                        # interval has minTime, maxTime, mark
                        duration = interval.maxTime - interval.minTime
                        # Filter out empty phonemes if needed, but MFA usually outputs "sil" or similar
                        # Round duration to 2 decimal places
                        phonemes.append(AlignedPhoneme(phoneme=interval.mark, duration=round(duration, 2)))
                else:
                    pass 
                    # print(f"Warning: No phone tier found in {tg_path}")
                    
            except Exception as e:
                print(f"Error parsing TextGrid {tg_path}: {e}")
        else:
            # Try fallback search? 
            # Maybe MFA output structure is slightly different?
            # `ls` showed: TEARS_MFA/ears_dataset/train/p001/...
            # So it should match.
            pass 
            # print(f"Warning: TextGrid missing: {tg_path}")
            
        return phonemes

def process_dataset_split(split_name: str, limit: Optional[int] = None):
    print(f"Processing split: {split_name}")
    
    dataset_processor = TearsDatasetProcessor(split=split_name, limit=limit)
    audio_feat_extractor = TearsAudioFeatExtractor()
    aligned_phonemes_extractor = TearsAlignedPhonemesExtractor()
    
    output_json_path = TEARS_V2_ROOT / f"{split_name}_v2.json"
    
    # Resume logic
    processed_data = []
    if output_json_path.exists():
        try:
            with open(output_json_path, "r") as f:
                processed_data = json.load(f)
            print(f"Resuming {split_name}: Found {len(processed_data)} existing entries.")
        except Exception as e:
            print(f"Error loading existing {output_json_path}: {e}")
            
    # Create a set of (audio_path, question) to identify duplicates for resume
    processed_keys = set()
    for item in processed_data:
        # Key structure matches new schema: prompt instead of question
        p = item.get("prompt") or item.get("question")
        key = (item["audio_path"], p)
        processed_keys.add(key)
        
    # --- Data Loading Phase ---
    print("Loading and preparing samples...")
    
    items_to_process = []
    
    # Wrap in tqdm
    iterator = dataset_processor.iterate()
    
    # tqdm might fail if imported as module in error? No, error was `item in tqdm(iterator)`
    # but import was `import tqdm`. Standard is `from tqdm import tqdm`.
    # I fixed import above.
    
    for item in tqdm(iterator, desc="Preparing input data"):
        # Check resume
        try:
            rel_path = str(item.audio.path.relative_to(TEARS_V2_ROOT))
        except ValueError:
            rel_path = str(item.audio.path)
            
        key = (rel_path, item.question.text)
        if key in processed_keys:
            continue
            
        # Feature Extraction & Phoneme Alignment
        # This is CPU bound, do it here.
        audio_features = audio_feat_extractor.extract(item.audio)
        transcript = item.transcript # Assuming transcript is already in item from processor
        aligned_phonemes = aligned_phonemes_extractor.align(item.audio, transcript)
        
        processed_sample = ProcessedSample(
            audio=item.audio,
            question=item.question,
            answer=item.answer,
            transcript=transcript,
            audio_features=audio_features,
            aligned_phonemes=aligned_phonemes,
        )
        items_to_process.append(processed_sample)
    
    print(f"Total items to process: {len(items_to_process)}")
    
    if not items_to_process:
        print("Nothing to process.")
        return

    # --- Batch Generation Phase ---
    # Initialize vLLM
    from vllm import LLM, SamplingParams
    from main_qwen import QwenModelManager
    
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
    llm = QwenModelManager.get_model(model_name)
    tokenizer = llm.get_tokenizer()
    
    # Prepare Prompts
    all_prompts = []
    sys_prompt = (
      "You are a speech-and-text reasoning assistant. Given audio features, "
      "aligned phonemes, question and answer, produce a concise but explicit "
      "step-by-step reasoning trace that could teach a model how to solve it. "
      "Return plain text, no JSON."
    )
    
    # Reuse formatting helpers
    def _format_features_block(d: Dict[str, float]) -> str:
        lines = ["{"]
        for k, v in sorted(d.items()):
            lines.append(f"  '{k}': {float(v):.2f},")
        lines.append("}")
        return "\n".join(lines)

    def _format_phonemes_block(items: List[AlignedPhoneme]) -> str:
        pairs = ", ".join(f"('{ap.phoneme}', {ap.duration:.2f})" for ap in items)
        return f"[{pairs}]"
    
    # Load template once
    template_path = Path(__file__).resolve().parent / "data_gen_prompt3.txt"
    template_content = None
    if template_path.exists():
        template_content = template_path.read_text(encoding="utf-8")
    
    print("Constructing prompts...")
    for sample in tqdm(items_to_process, desc="Formatting prompts"):
        if template_content:
            tpl = template_content
            tpl = tpl.replace("[transcript]", sample.transcript.text)
            tpl = tpl.replace("[audio_features]", _format_features_block(sample.audio_features.values))
            tpl = tpl.replace("[aligned_phonemes]", _format_phonemes_block(sample.aligned_phonemes))
            tpl = tpl.replace("[Question]", sample.question.text)
            tpl = tpl.replace("[Answer]", sample.answer.text)
            user_text = tpl
        else:
            user_text = (
                "We are now designing a system to generate structured audio-based chain-of-thought reasoning data.\n\n"
                f"Transcript: {sample.transcript.text}\n\n"
                f"audio_featuers: {_format_features_block(sample.audio_features.values)}\n\n"
                f"aligned_phonemes: {_format_phonemes_block(sample.aligned_phonemes)}\n\n"
                f"Question: {sample.question.text}\n"
                f"Answer: {sample.answer.text}\n"
                "Please respond with <THINK>...</THINK><RESPONSE>...</RESPONSE> as specified."
            )
            
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_text},
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(full_prompt)
        
    # Generate in Batches (Chunking)
    batch_size = 500
    total_prompts = len(all_prompts)
    print(f"Generating responses for {total_prompts} items in batches of {batch_size}...")
    
    sampling_params = SamplingParams(temperature=0.3, max_tokens=2000)
    
    for i in range(0, total_prompts, batch_size):
        batch_end = min(i + batch_size, total_prompts)
        batch_prompts = all_prompts[i:batch_end]
        batch_items = items_to_process[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}/{(total_prompts + batch_size - 1)//batch_size} (Indices {i}-{batch_end})...")
        
        # vLLM generate
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process Results for this batch
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text.strip()
            sample = batch_items[j]
            
            # Extract <FINAL_ANSWER> content
            final_answer_tag = "<FINAL_ANSWER>"
            response_content = generated_text
            
            if final_answer_tag in generated_text:
                parts = generated_text.split(final_answer_tag)
                if len(parts) > 1:
                    response_content = parts[-1].strip()
            
            try:
                rel_path = str(sample.audio.path.relative_to(TEARS_V2_ROOT))
            except ValueError:
                rel_path = str(sample.audio.path)
                
            new_entry = {
                "audio_path": rel_path,
                "prompt": sample.question.text,
                "response": response_content, # Extracted Final Answer
                "answer": sample.answer.text, # Ground Truth Metadata
            }
            processed_data.append(new_entry)
            
        # Save checkpoint after each batch
        print(f"Saving checkpoint... Total items: {len(processed_data)}")
        with open(output_json_path, "w") as f:
            json.dump(processed_data, f, indent=2)
    
    print(f"Finished {split_name}. Total: {len(processed_data)}")

def main():
    # Process test (limit=None for full)
    # process_dataset_split("test", limit=None)
    
    # Process train
    process_dataset_split("train", limit=None)

if __name__ == "__main__":
    main()

