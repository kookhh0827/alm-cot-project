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
    Extract robust acoustic features optimized for 10s clips and Qwen2-Audio.
    Dropped: Formants (F1, F2), Skewness
    Added: Pitch Slope (Intonation)
    """
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        sound = parselmouth.Sound(y, sampling_frequency=target_sr)

        features = {}
        features['duration'] = sound.duration
        features['sampling_frequency'] = float(target_sr)

        # ===== 1. Pitch features (Basic + Contour) =====
        try:
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
            pitch_values = pitch.selected_array['frequency']
            # Filter unvoiced (0)
            pitch_values_voiced = pitch_values[pitch_values > 0]

            if len(pitch_values_voiced) > 0:
                features['pitch_mean'] = float(np.mean(pitch_values_voiced))
                features['pitch_std'] = float(np.std(pitch_values_voiced))
                features['pitch_range'] = float(np.ptp(pitch_values_voiced)) # Range is better than Variance
                
                # --- NEW: Pitch Slope (Intonation) ---
                # Get time points for voiced frames
                xs = pitch.xs()
                # Create mask for voiced frames (pitch > 0)
                mask = pitch_values > 0
                if np.sum(mask) > 10: # Need enough points for regression
                    voiced_times = xs[mask]
                    voiced_pitch = pitch_values[mask]
                    # Linear regression: pitch = a * time + b
                    # Slope 'a' indicates overall rising/falling trend
                    slope, _ = np.polyfit(voiced_times, voiced_pitch, 1)
                    features['pitch_slope'] = float(slope)
                else:
                    features['pitch_slope'] = 0.0
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
                features['pitch_range'] = 0.0
                features['pitch_slope'] = 0.0
                
        except Exception:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
            features['pitch_slope'] = 0.0

        # ===== 2. Energy features (Dynamics) =====
        try:
            intensity = sound.to_intensity(minimum_pitch=75.0, time_step=0.01)
            intensity_values = intensity.values[0]
            intensity_values = intensity_values[~np.isnan(intensity_values)]

            if len(intensity_values) > 0:
                features['energy_mean'] = float(np.mean(intensity_values))
                e_max = float(np.max(intensity_values))
                e_min = float(np.min(intensity_values))
                features['energy_dynamic_range'] = e_max - e_min
            else:
                features['energy_mean'] = 0.0
                features['energy_dynamic_range'] = 0.0
        except Exception:
            features['energy_mean'] = 0.0
            features['energy_dynamic_range'] = 0.0

        # ===== 3. Voice Quality (HNR + Jitter/Shimmer) =====
        try:
            # HNR (Harmonics-to-Noise Ratio) - Good for Clear vs Hoarse
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

        try:
            # Need Pitch object for PointProcess
            pitch_for_pulses = sound.to_pitch(time_step=0.01, pitch_floor=75.0, pitch_ceiling=600.0)
            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75.0, 600.0)
            
            # Jitter (local)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
            features['jitter_local'] = float(jitter) if not np.isnan(jitter) else 0.0
            
            # Shimmer (local)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
            features['shimmer_local'] = float(shimmer) if not np.isnan(shimmer) else 0.0
            
        except Exception:
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
    
    # Logic: Force update for new features (pitch_slope)
    # Or check if pitch_slope exists in current AF
    # if af_path.exists():
    #     try:
    #         with open(af_path, "r") as f:
    #             existing = json.load(f)
    #         # If pitch_slope is missing, we need to re-compute
    #         if 'pitch_slope' in existing and 'hnr_mean' in existing:
    #             return # Already has new features
    #     except:
    #         pass # corrupted, re-do
        
    features = extract_comprehensive_features(str(file_path), target_sr=TARGET_SR)
    
    if features:
        with open(af_path, "w") as f:
            json.dump(features, f, indent=2)

def main():
    print("Starting Optimized Feature Extraction...")
    
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
