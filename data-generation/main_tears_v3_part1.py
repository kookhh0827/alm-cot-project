import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator, List, Dict, Tuple, Optional
import textgrid
from tqdm import tqdm

# Add parent directory to path to import from parallel modules if needed
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_gen_abc import (
    AbstractDatasetProcessor,
    AbstractAudioFeatExtractor,
    AbstractAlignedPhonemesExtractor,
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
)
from latent_utils import LatentCalculator

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
        
        target_attributes = {
            "gender": "What is the speaker's gender?",
            "age": "What is the speaker's age?",
            "dialect_region": "What is the speaker's dialect?",
            "ethnicity": "What is the speaker's ethnicity?" 
        }
        
        for entry in data:
            if self.limit and count >= self.limit:
                break
                
            speaker = entry.get("speaker", {})
            if not speaker:
                continue
                
            rel_path = entry.get("audio_path", "")
            full_audio_path = TEARS_V2_ROOT / rel_path
            
            transcript_text = entry.get("transcript", "")
            
            for attr, prompt_text in target_attributes.items():
                val = speaker.get(attr)
                    
                if val is None:
                    continue
                
                yield DatasetItem(
                    audio=Audio(path=full_audio_path),
                    question=Question(text=prompt_text),
                    answer=Answer(text=str(val)),
                    transcript=Transcript(text=transcript_text),
                )
                count += 1

class TearsAudioFeatExtractor(AbstractAudioFeatExtractor):
    def extract(self, audio: Audio) -> AudioFeatures:
        wav_path = audio.path
        af_path = wav_path.with_suffix(".AF")
        
        features = {}
        if af_path.exists():
            try:
                with open(af_path, "r") as f:
                    raw_features = json.load(f)
                    features = {k: float(v) for k, v in raw_features.items()}
            except Exception as e:
                print(f"Error loading AF {af_path}: {e}")
        else:
            pass 
            
        return AudioFeatures(values=features)

class TearsAlignedPhonemesExtractor(AbstractAlignedPhonemesExtractor):
    def align(self, audio: Audio, transcript: Transcript) -> AlignedPhonemes:
        try:
            rel_path = audio.path.relative_to(TEARS_V2_ROOT)
        except ValueError:
            return []

        tg_path = TEARS_MFA_ROOT / rel_path.with_suffix(".TextGrid")
        
        phonemes = []
        if tg_path.exists():
            try:
                tg = textgrid.TextGrid.fromFile(str(tg_path))
                
                phone_tier = None
                for tier in tg.tiers:
                    if "phone" in tier.name.lower():
                        phone_tier = tier
                        break
                
                if phone_tier:
                    for interval in phone_tier:
                        duration = interval.maxTime - interval.minTime
                        phonemes.append(AlignedPhoneme(phoneme=interval.mark, duration=duration))
                    
            except Exception as e:
                print(f"Error parsing TextGrid {tg_path}: {e}")
            
        return phonemes

def run_part1(split_name: str, limit: Optional[int] = None):
    print(f"=== Part 1: Latent Generation for {split_name} ===")
    
    dataset_processor = TearsDatasetProcessor(split=split_name, limit=limit)
    audio_feat_extractor = TearsAudioFeatExtractor()
    aligned_phonemes_extractor = TearsAlignedPhonemesExtractor()
    
    stats_path = TEARS_V2_ROOT / "latent_stats_v2.json"
    latent_calculator = LatentCalculator(stats_path)
    
    output_path = TEARS_V2_ROOT / f"{split_name}_v3_intermediate.json"
    
    # --- 1. Load Existing Progress ---
    processed_data = []
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                processed_data = json.load(f)
            print(f"Resuming: Found {len(processed_data)} existing entries.")
        except Exception as e:
            print(f"Error loading existing {output_path}: {e}")
            
    processed_keys = set()
    for item in processed_data:
        key = (item["audio_path"], item["prompt"])
        processed_keys.add(key)
        
    # --- 2. Collect Data (Pass 1) ---
    # We need to buffer data to (a) fit stats if needed, or (b) ensure we have features before generating latents
    # Since we want to save incrementally, we should FIT first (if needed), then LOOP again or iterate.
    
    need_fit = (split_name == "train" and not stats_path.exists())
    
    buffer_for_fit = []
    items_to_process = [] # List of ProcessedSample
    
    print("Scanning dataset...")
    # We iterate everything first. If we need to fit, we store in memory. 
    # If memory is an issue (dataset > 100k items), we might need 2 passes on disk.
    # Assuming it fits in memory (TEARS is likely < 50k?).
    
    iterator = dataset_processor.iterate()
    for item in tqdm(iterator, desc="Loading Data"):
        # Check if already processed
        try:
            rel_path = str(item.audio.path.relative_to(TEARS_V2_ROOT))
        except ValueError:
            rel_path = str(item.audio.path)
            
        key = (rel_path, item.question.text)
        
        # If we need fit, we must process even if in processed_keys (to ensure stats are consistent with full data)
        # BUT if stats already exist, we can skip processed items.
        
        if not need_fit and key in processed_keys:
            continue
            
        try:
            audio_features = audio_feat_extractor.extract(item.audio)
            aligned_phonemes = aligned_phonemes_extractor.align(item.audio, item.transcript)
            
            sample = ProcessedSample(
                audio=item.audio,
                question=item.question,
                answer=item.answer,
                transcript=item.transcript,
                audio_features=audio_features,
                aligned_phonemes=aligned_phonemes,
            )
            
            if need_fit:
                buffer_for_fit.append(sample)
            
            # If not in processed keys, add to items_to_process
            if key not in processed_keys:
                items_to_process.append(sample)
                
        except Exception as e:
            # print(f"Error processing {item.audio.path}: {e}")
            pass

    # --- 3. Fit Stats (if needed) ---
    if need_fit:
        if not buffer_for_fit:
            print("No data found to fit stats!")
            return
        print(f"Fitting stats on {len(buffer_for_fit)} items...")
        latent_calculator.fit(buffer_for_fit)
        # Clear buffer to free memory
        buffer_for_fit = []
    else:
        latent_calculator.load_stats()
        
    # --- 4. Generate Latents and Save incrementally ---
    print(f"Generating latents for {len(items_to_process)} new items...")
    
    save_interval = 100
    
    for i, sample in enumerate(tqdm(items_to_process, desc="Computing Latents")):
        try:
            rel_path = str(sample.audio.path.relative_to(TEARS_V2_ROOT))
        except ValueError:
            rel_path = str(sample.audio.path)
            
        latent_json = latent_calculator.process_item(
            sample.audio_features.values, 
            sample.aligned_phonemes
        )
        
        entry = {
            "audio_path": rel_path,
            "prompt": sample.question.text,
            "transcript": sample.transcript.text,
            "ground_truth_answer": sample.answer.text,
            "latent_json": latent_json
            # We don't save raw features to keep JSON size smaller, 
            # unless needed for debugging. Latent + Transcript + GT is enough for Part 2.
        }
        processed_data.append(entry)
        
        if (i + 1) % save_interval == 0:
            with open(output_path, "w") as f:
                json.dump(processed_data, f, indent=2)
                
    # Final save
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
        
    print(f"Part 1 Complete. Saved {len(processed_data)} items to {output_path}")

def main():
    run_part1("test")
    # run_part1("train", limit=None)

if __name__ == "__main__":
    main()

