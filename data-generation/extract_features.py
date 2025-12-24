import os
import json
import numpy as np
import librosa
import parselmouth
from pathlib import Path
from tqdm import tqdm
import concurrent.futures

# Configuration
TEARS_V2_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/TEARS_V2")
TARGET_SR = 16000

def extract_comprehensive_features(audio_path, target_sr=16000):
    """
    Extract comprehensive acoustic features including Jitter and Shimmer.
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        sound = parselmouth.Sound(y, sampling_frequency=target_sr)

        features = {}
        features['duration'] = sound.duration
        features['sampling_frequency'] = float(target_sr)

        # ===== 1. Pitch features (5 features) =====
        try:
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]

            if len(pitch_values) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values))
                features['pitch_std'] = float(np.std(pitch_values))
                features['pitch_median'] = float(np.median(pitch_values))
                features['pitch_range'] = float(np.ptp(pitch_values))
                features['pitch_variance'] = float(np.var(pitch_values))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_median'] = 0.0
                features['pitch_range'] = 0.0
                features['pitch_variance'] = 0.0
        except Exception:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_median'] = 0.0
            features['pitch_range'] = 0.0
            features['pitch_variance'] = 0.0

        # ===== 2. Energy features (5 features) =====
        try:
            intensity = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[~np.isnan(intensity_values)]

            if len(intensity_values) > 0:
                features['energy_mean'] = float(np.mean(intensity_values))
                features['energy_std'] = float(np.std(intensity_values))
                features['energy_max'] = float(np.max(intensity_values))
                features['energy_min'] = float(np.min(intensity_values))
                features['energy_dynamic_range'] = float(features['energy_max'] - features['energy_min'])
            else:
                features['energy_mean'] = 0.0
                features['energy_std'] = 0.0
                features['energy_max'] = 0.0
                features['energy_min'] = 0.0
                features['energy_dynamic_range'] = 0.0
        except Exception:
            features['energy_mean'] = 0.0
            features['energy_std'] = 0.0
            features['energy_max'] = 0.0
            features['energy_min'] = 0.0
            features['energy_dynamic_range'] = 0.0

        # ===== 3. Formant features (9 features) =====
        try:
            formant = sound.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5500)
            formant_times = formant.ts()

            for i in range(1, 4):
                f_values = []
                for t in formant_times:
                    f = formant.get_value_at_time(i, t)
                    if not np.isnan(f) and f > 0:
                        f_values.append(f)

                if len(f_values) > 0:
                    features[f'F{i}_mean'] = float(np.mean(f_values))
                    features[f'F{i}_std'] = float(np.std(f_values))
                    features[f'F{i}_median'] = float(np.median(f_values))
                else:
                    features[f'F{i}_mean'] = 0.0
                    features[f'F{i}_std'] = 0.0
                    features[f'F{i}_median'] = 0.0
        except Exception:
            for i in range(1, 4):
                features[f'F{i}_mean'] = 0.0
                features[f'F{i}_std'] = 0.0
                features[f'F{i}_median'] = 0.0

        # ===== 4. HNR feature =====
        try:
            harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)
            hnr_values = harmonicity.values[0]
            hnr_values = hnr_values[~np.isnan(hnr_values)]
            hnr_values = hnr_values[hnr_values != -200]

            if len(hnr_values) > 0:
                features['hnr_mean'] = float(np.mean(hnr_values))
            else:
                features['hnr_mean'] = 0.0
        except Exception:
            features['hnr_mean'] = 0.0

        # ===== 5. Voice Quality (Jitter & Shimmer) [NEW] =====
        try:
            # Need Pitch object for PointProcess
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
            
            # Jitter (local)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            features['jitter_local'] = float(jitter) if not np.isnan(jitter) else 0.0
            
            # Shimmer (local)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_local'] = float(shimmer) if not np.isnan(shimmer) else 0.0
            
        except Exception as e:
            features['jitter_local'] = 0.0
            features['shimmer_local'] = 0.0

        return features

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_file(file_path):
    file_path = Path(file_path)
    
    # Check if .AF exists
    af_path = file_path.with_suffix('.AF')
    
    # Logic: Read existing AF, check if jitter/shimmer exists. If so, skip. If not, re-compute.
    if af_path.exists():
        try:
            with open(af_path, "r") as f:
                existing = json.load(f)
            if 'jitter_local' in existing and 'shimmer_local' in existing:
                return # Already updated
        except:
            pass # corrupted, re-do
        
    features = extract_comprehensive_features(str(file_path), target_sr=TARGET_SR)
    
    if features:
        with open(af_path, "w") as f:
            json.dump(features, f, indent=2)

def main():
    print("Starting Audio Feature Extraction (Update Mode)...")
    
    # Collect all wav files
    print("Scanning files...")
    wav_files = []
    wav_files.extend(list(TEARS_V2_ROOT.rglob("*.wav")))
    wav_files.extend(list(TEARS_V2_ROOT.rglob("*.WAV")))
    
    print(f"Found {len(wav_files)} audio files.")
    
    # Parallel processing
    max_workers = os.cpu_count() or 4
    print(f"Processing with {max_workers} workers...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        list(tqdm(executor.map(process_file, wav_files), total=len(wav_files)))
        
    print("Feature extraction complete.")

if __name__ == "__main__":
    main()
