import os, csv
from pathlib import Path
import torchaudio, torch, random
import numpy as np
import librosa
import parselmouth

EARS_ROOT = Path("/ocean/projects/cis220031p/hkook/dataset/EARS")
AUG_ROOT  = Path("/ocean/projects/cis220031p/hkook/aug_wav")
OUT_CSV   = Path("/ocean/projects/cis220031p/hkook/hailey/ears_aug_features.csv")

AUG_ROOT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)

# reuse CoLmbo's augmentation function
def augment_audio(waveform, sample_rate, policy):
    if policy == "noise":
        noise = torch.randn_like(waveform) * random.uniform(0.001, 0.015)
        waveform = waveform + noise
        aug_tag = "noise"

    elif policy == "reverb":
        effect = [['reverb', '-w', '20']]
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate, effects=effect
        )
        aug_tag = "reverb"

    else:
        aug_tag = "none"

    return waveform, aug_tag

def augment_and_save(
    audio_path: Path,
    out_dir: Path,
    aug_id: int,
    seed: int,
    policy: str,
    sample_rate=16000
):
    random.seed(seed)
    torch.manual_seed(seed)

    wav, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

    wav, aug_tag = augment_audio(wav, sample_rate, policy)

    out_path = out_dir / f"{audio_path.stem}_{policy}_aug{aug_id}.wav"
    torchaudio.save(out_path, wav, sample_rate)

    return out_path, aug_tag


def extract_comprehensive_features(audio_path, target_sr=16000):
    """
    Extract comprehensive acoustic features from audio file

    Args:
        audio_path: Path to audio file (wav/flac/mp3)
        target_sr: Target sampling rate (default: 16000 Hz)

    Returns:
        Dictionary of acoustic features (20 features + metadata)
    """
    try:
        # already uniformed to 16kHz during augmentation
        # y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        # sound = parselmouth.Sound(y, sampling_frequency=target_sr)
        sound = parselmouth.Sound(str(audio_path))

        features = {}

        # Basic audio information
        features['duration'] = sound.duration
        features['sampling_frequency'] = sound.sampling_frequency

        # ===== 1. Pitch features (5 features) =====
        pitch = sound.to_pitch(
            time_step=0.01,      # Add time_step for consistency
            pitch_floor=75.0,
            pitch_ceiling=600.0
        )

        pitch_values = pitch.selected_array['frequency']
        pitch_values = pitch_values[pitch_values > 0]  # Filter unvoiced frames

        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_median'] = np.median(pitch_values)
            features['pitch_range'] = np.ptp(pitch_values)  # max - min
            features['pitch_variance'] = np.var(pitch_values)
        else:
            features['pitch_mean'] = np.nan
            features['pitch_std'] = np.nan
            features['pitch_median'] = np.nan
            features['pitch_range'] = np.nan
            features['pitch_variance'] = np.nan

        # ===== 2. Energy features (5 features) =====
        intensity = sound.to_intensity(
            minimum_pitch=75.0,
            time_step=0.01      # Add time_step for consistency
        )

        intensity_values = intensity.values[0]
        intensity_values = intensity_values[~np.isnan(intensity_values)]

        if len(intensity_values) > 0:
            features['energy_mean'] = np.mean(intensity_values)
            features['energy_std'] = np.std(intensity_values)
            features['energy_max'] = np.max(intensity_values)
            features['energy_min'] = np.min(intensity_values)
            features['energy_dynamic_range'] = features['energy_max'] - features['energy_min']
        else:
            features['energy_mean'] = np.nan
            features['energy_std'] = np.nan
            features['energy_max'] = np.nan
            features['energy_min'] = np.nan
            features['energy_dynamic_range'] = np.nan

        # ===== 3. Formant features (9 features: F1/F2/F3 x 3 stats) =====
        formant = sound.to_formant_burg(
            time_step=0.01,              # Add time_step
            max_number_of_formants=5,
            maximum_formant=5500,        # Uniform parameter
            window_length=0.025,         # 25ms window
            pre_emphasis_from=50.0
        )

        formant_times = formant.ts()

        # Extract F1, F2, F3
        for i in range(1, 4):  # F1, F2, F3
            formant_values = []

            for t in formant_times:
                f = formant.get_value_at_time(i, t)
                if not np.isnan(f) and f > 0:
                    formant_values.append(f)

            if len(formant_values) > 0:
                features[f'F{i}_mean'] = np.mean(formant_values)
                features[f'F{i}_std'] = np.std(formant_values)
                features[f'F{i}_median'] = np.median(formant_values)
            else:
                features[f'F{i}_mean'] = np.nan
                features[f'F{i}_std'] = np.nan
                features[f'F{i}_median'] = np.nan

        # ===== 4. HNR feature (1 feature) =====
        try:
            harmonicity = parselmouth.praat.call(
                sound,
                "To Harmonicity (cc)",
                0.01,   # time_step
                75.0,   # minimum pitch
                0.1,    # silence threshold
                1.0     # periods per window
            )

            hnr_values = harmonicity.values[0]
            hnr_values = hnr_values[~np.isnan(hnr_values)]
            hnr_values = hnr_values[hnr_values != -200]  # Remove undefined values

            if len(hnr_values) > 0:
                features['hnr_mean'] = np.mean(hnr_values)
            else:
                features['hnr_mean'] = np.nan
        except Exception as e:
            print(f"Warning: HNR extraction failed for {audio_path}: {e}")
            features['hnr_mean'] = np.nan

        return features

    except Exception as e:
        print(f"Error processing file {audio_path}: {e}")
        return None
    
ears_wavs = list(EARS_ROOT.rglob("*.wav"))
K = 1   # augmentations per audio file
base_seed = 42
aug_root = AUG_ROOT
dataset = "EARS"

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "utt_id","speaker_id","dataset","duration","aug_id", "aug_tag", "aug_wav_path",
        "sampling_frequency","pitch_mean","pitch_std","pitch_median",
        "pitch_range","pitch_variance",
        "energy_mean","energy_std","energy_max","energy_min","energy_dynamic_range",
        "F1_mean","F1_std","F1_median",
        "F2_mean","F2_std","F2_median",
        "F3_mean","F3_std","F3_median",
        "hnr_mean"
    ])

    # test several speakers
    for idx, wav_path in enumerate(ears_wavs):
        for policy in ["noise", "reverb"]:
            for aug_id in range(K):
                aug_wav, aug_tag = augment_and_save(
                    wav_path,
                    out_dir=aug_root,
                    aug_id=aug_id,
                    seed=base_seed + aug_id,
                    policy=policy
                )

                speaker_id = wav_path.parent.name    # EARS/p101/xxx.wav

                features = extract_comprehensive_features(aug_wav)

                utt_id = wav_path.stem

                if features is None:
                    continue

                writer.writerow([
                    utt_id,
                    speaker_id,
                    dataset,
                    features["duration"],
                    aug_id,
                    aug_tag,
                    str(aug_wav),
                    features["sampling_frequency"],
                    features["pitch_mean"],
                    features["pitch_std"],
                    features["pitch_median"],
                    features["pitch_range"],
                    features["pitch_variance"],
                    features["energy_mean"],
                    features["energy_std"],
                    features["energy_max"],
                    features["energy_min"],
                    features["energy_dynamic_range"],
                    features["F1_mean"],
                    features["F1_std"],
                    features["F1_median"],
                    features["F2_mean"],
                    features["F2_std"],
                    features["F2_median"],
                    features["F3_mean"],
                    features["F3_std"],
                    features["F3_median"],
                    features["hnr_mean"],
                ])

            if idx % 50 == 0:
                f.flush()
                os.fsync(f.fileno())

