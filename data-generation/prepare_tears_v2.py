import os
import re
import json
import torch
import librosa
import soundfile as sf
import datasets
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import numpy as np

# --- Configuration ---
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
TIMIT_SRC_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TIMIT/data")
EARS_SRC_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/EARS")

# Output directories
TIMIT_DST_DIR = TEARS_V2_ROOT / "timit_dataset"
EARS_DST_DIR = TEARS_V2_ROOT / "ears_dataset"

# Create directories
TIMIT_DST_DIR.mkdir(parents=True, exist_ok=True)
EARS_DST_DIR.mkdir(parents=True, exist_ok=True)

# Load EARS transcripts
TRANSCRIPTS_PATH = EARS_SRC_ROOT / "ears_dataset/transcripts.json"
try:
    with open(TRANSCRIPTS_PATH, "r") as f:
        EARS_TRANSCRIPTS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load transcripts.json: {e}")
    EARS_TRANSCRIPTS = {}

# Initialize ASR Pipeline (Whisper)
# Using 'openai/whisper-base.en' for better speed while maintaining decent accuracy
# Increased chunk_length_s and added batch_size for throughput
print("Loading Whisper model for Freeform ASR...")
device = "cuda" if torch.cuda.is_available() else "cpu"
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-base.en",
    device=device,
    chunk_length_s=30,
    batch_size=16
)

# EARS slice regex
# matches: rainbow_03_highpitch_0_153653.wav
# Group 1: base name, Group 2: start, Group 3: end
EARS_SLICE_PATTERN = re.compile(r'^(.*)_(\d+)_(\d+)\.wav$', re.IGNORECASE)

def ensure_16k(audio, sr):
    if sr != 16000:
        # Librosa resampling is high quality but can be slow. 
        # soundfile doesn't resample.
        # Using librosa.resample
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio

def process_split(split_name):
    print(f"Processing split: {split_name}")
    try:
        ds = datasets.load_dataset("cmu-mlsp/TEARS", split=split_name)
    except Exception as e:
        print(f"Could not load split {split_name}: {e}")
        return []

    output_json_path = TEARS_V2_ROOT / f"{split_name}.json"
    processed_samples = []
    existing_paths = set()

    # Resume functionality
    if output_json_path.exists():
        try:
            with open(output_json_path, "r") as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    processed_samples = existing_data
                    existing_paths = {s["audio_path"] for s in processed_samples}
            print(f"Resuming {split_name}: Found {len(processed_samples)} existing samples.")
        except Exception as e:
            print(f"Error reading existing JSON, starting fresh: {e}")

    # Buffer for incremental saving
    save_interval = 100
    
    for i, sample in enumerate(tqdm(ds)):
        original_path_str = sample.get('audio_path', '')
        if not original_path_str:
            continue
        
        # Normalize path
        path_str = original_path_str.replace('\\', '/')
        path_parts = path_str.split('/')
        
        # Determine destination path early for skip check
        rel_path = None
        if 'timit' in path_str.lower() and len(path_parts) >= 3:
             rel_path = Path(*path_parts)
        elif 'ears' in path_str.lower() and len(path_parts) >= 4:
             new_rel_parts = list(path_parts)
             new_rel_parts[0] = "ears_dataset"
             rel_path = Path(*new_rel_parts)
        
        # Check if already processed
        if rel_path and str(rel_path) in existing_paths:
            continue
        
        # --- Processing Logic ---
        if 'timit' in path_str.lower():
            # Logic: timit_dataset/train/DR1/MRWS0/SI1732.WAV -> TIMIT/data/TRAIN/DR1/MRWS0/SI1732.WAV
            if len(path_parts) < 3: continue
            
            # Mapping: timit_dataset/<split>/... -> TIMIT_SRC_ROOT/<SPLIT>/...
            # TEARS split in path might be 'train' or 'test'
            t_split = path_parts[1].upper() # TRAIN or TEST
            t_rest = path_parts[2:]
            
            src_wav_path = TIMIT_SRC_ROOT / t_split / "/".join(t_rest)
            
            if not src_wav_path.exists():
                continue # Skip missing WAV
            
            # Check Transcript (.TXT or .txt)
            src_txt_path = src_wav_path.with_suffix('.TXT')
            if not src_txt_path.exists():
                src_txt_path = src_wav_path.with_suffix('.txt')
            
            if not src_txt_path.exists():
                continue # Skip missing Transcript
            
            # Read Transcript
            try:
                with open(src_txt_path, 'r') as f:
                    line = f.readline().strip()
                    # TIMIT txt format: "0 6190 She had your dark suit in greasy wash water all year."
                    # We want the text part.
                    parts = line.split(' ', 2)
                    if len(parts) == 3:
                        transcript_text = parts[2]
                    else:
                        transcript_text = line # Fallback
            except:
                continue
            
            # Prepare Destination
            # Keep similar structure: timit_dataset/<split>/DRX/SPK/FILE
            # rel_path_parts = ['timit_dataset', path_parts[1], ... ]
            # We will start from path_parts[1] to maintain "timit_dataset/train/..."
            # Actually, user said put in "timit_dataset" folder.
            # path_parts[0] is 'timit_dataset'.
            
            rel_path = Path(*path_parts) # timit_dataset/train/...
            dst_wav_path = TEARS_V2_ROOT / rel_path
            
            # Process Audio
            try:
                y, sr = sf.read(src_wav_path)
                y_16k = ensure_16k(y, sr)
                
                # Write Audio
                dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst_wav_path, y_16k, 16000)
                
                # Write Transcript to .txt
                dst_txt_path = dst_wav_path.with_suffix('.txt')
                with open(dst_txt_path, 'w') as f:
                    f.write(transcript_text)
                
            except Exception as e:
                print(f"Error processing audio {src_wav_path}: {e}")
                continue
                
            # Add to dataset
            new_sample = {
                "audio_path": str(rel_path),
                "transcript": transcript_text,
                "duration": len(y_16k) / 16000.0,
                "dataset": "timit",
                "speaker_id": path_parts[-2] if len(path_parts) >= 2 else "unknown"
            }
            # Copy other relevant fields if needed (e.g. responses/prompts from original?)
            # User asked to create "train.json". The structure usually implies the fields needed for training.
            # The user mentioned "TEARS dataset... response... output <FINAL_ANSWER>... acoustic features...".
            # The data generation script `main_qwen.py` expects `Audio`, `Question`, `Answer`, `Transcript`.
            # In TEARS:
            # `prompts` (list), `responses` (list).
            # We should probably keep these.
            new_sample['prompts'] = sample.get('prompts')
            new_sample['responses'] = sample.get('responses')
            new_sample['speaker'] = sample.get('speaker')
            
            processed_samples.append(new_sample)

        elif 'ears' in path_str.lower():
            # Logic: ears_dataset_processed/train/p101/rainbow_03_highpitch_0_153653.wav
            if len(path_parts) < 4: continue
            
            speaker = path_parts[2]
            filename = path_parts[3]
            
            # Parse Slice Info
            match = EARS_SLICE_PATTERN.match(filename)
            if not match:
                # Maybe unsliced file? TEARS usually has sliced.
                continue
            
            base_name = match.group(1) # rainbow_03_highpitch
            start_sample = int(match.group(2))
            end_sample = int(match.group(3))
            
            original_filename = base_name + ".wav"
            src_wav_path = EARS_SRC_ROOT / speaker / original_filename
            
            if not src_wav_path.exists():
                continue # Skip missing WAV
            
            # Transcript Strategy
            # User reported that using full transcripts for sliced files results in text 
            # that includes content after the cut.
            # Since almost all EARS files in TEARS are sliced (or at least potentially),
            # and we want high-quality transcripts for the specific audio segment,
            # we will use ASR (Whisper Large V3) for ALL EARS samples.
            use_asr = True
            transcript_text = ""
            
            if not use_asr:
                 # We have a transcript from JSON.
                 pass
            
            # Prepare Destination
            # TEARS_V2/ears_dataset/train/p101/rainbow...
            # Map 'ears_dataset_processed' -> 'ears_dataset' to match user instruction folder name
            # But wait, if I change the folder name, I should update the path in JSON.
            # path_parts[0] is 'ears_dataset_processed'.
            
            new_rel_parts = list(path_parts)
            new_rel_parts[0] = "ears_dataset" # User requested folder name
            rel_path = Path(*new_rel_parts)
            dst_wav_path = TEARS_V2_ROOT / rel_path
            
            # Process Audio
            try:
                # Load FULL audio
                # sf.read allows reading slices if we know frame offset, but resampling complicates it.
                # Better to load full (or sufficient chunk), resample, then slice.
                # OR: Load, resample, slice.
                # EARS files can be long (freeform).
                # Optimization: Load with sf.read using start/stop frames IF sr is already 16k.
                # But we don't know SR for sure without checking. EARS is usually 48k.
                # So we must load, resample to 16k, then slice.
                # This is heavy.
                
                # Let's load full file first (streaming if possible, but librosa.load is easiest).
                # librosa.load(..., sr=16000) does resampling.
                y_16k, _ = librosa.load(src_wav_path, sr=16000)
                
                # Slice
                if end_sample > len(y_16k):
                    # Padding or clamp? Clamp.
                    slice_audio = y_16k[start_sample:]
                else:
                    slice_audio = y_16k[start_sample:end_sample]
                
                # If slice is too short?
                if len(slice_audio) < 1600: # < 0.1s
                    continue
                
                # ASR (Always for EARS now)
                if use_asr:
                    # Run ASR on this slice
                    # Input to pipeline can be numpy array
                    result = asr_pipeline(slice_audio)
                    transcript_text = result.get("text", "").strip()
                    if not transcript_text:
                        continue # Skip empty transcript
                
                # Write Audio
                dst_wav_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(dst_wav_path, slice_audio, 16000)
                
                # Write Transcript to .txt
                dst_txt_path = dst_wav_path.with_suffix('.txt')
                with open(dst_txt_path, 'w') as f:
                    f.write(transcript_text)
                
            except Exception as e:
                print(f"Error processing EARS audio {src_wav_path}: {e}")
                continue

            new_sample = {
                "audio_path": str(rel_path),
                "transcript": transcript_text,
                "duration": len(slice_audio) / 16000.0,
                "dataset": "ears",
                "speaker_id": speaker
            }
            new_sample['prompts'] = sample.get('prompts')
            new_sample['responses'] = sample.get('responses')
            new_sample['speaker'] = sample.get('speaker')
            
            processed_samples.append(new_sample)
            existing_paths.add(str(rel_path))
            
            # Incremental Save
            if len(processed_samples) % save_interval == 0:
                with open(output_json_path, "w") as f:
                    json.dump(processed_samples, f, indent=2)
            
    return processed_samples

def main():
    # Process Test
    # TEARS usually has 'test' split? "test" is valid split name in HuggingFace dataset.
    test_samples = process_split("test")
    print(f"Processed {len(test_samples)} test samples.")
    
    with open(TEARS_V2_ROOT / "test.json", "w") as f:
        json.dump(test_samples, f, indent=2)

    # Process Train
    train_samples = process_split("train")
    print(f"Processed {len(train_samples)} train samples.")
    
    with open(TEARS_V2_ROOT / "train.json", "w") as f:
        json.dump(train_samples, f, indent=2)
        
    print("Done.")

if __name__ == "__main__":
    main()

