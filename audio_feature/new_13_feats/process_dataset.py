# process EARS and TIMIT datasets without augmentation
# for EARS: output one .csv file with all speakers 001-107
# for TIMIT: output two .csv file, each for TRAIN and TEST speakers

import os
import time
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd
import parselmouth
from tqdm import tqdm
import json
from pathlib import Path

# TEST
# EARS_ROOT   = Path("/ocean/projects/cis220031p/hkook/hailey/new_13_feats/EARS_test")
# TIMIT_TRAIN = Path("/ocean/projects/cis220031p/hkook/hailey/new_13_feats/TIMIT_test/data/TRAIN")
# TIMIT_TEST  = Path("/ocean/projects/cis220031p/hkook/hailey/new_13_feats/TIMIT_test/data/TEST")

# OUT_DIR = Path("/ocean/projects/cis220031p/hkook/hailey/new_13_feats/outputs_test")
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# TODO
EARS_ROOT  = Path("/ocean/projects/cis220031p/hkook/dataset/EARS")
TIMIT_TRAIN = Path("/ocean/projects/cis220031p/hkook/dataset/TIMIT/data/TRAIN")
TIMIT_TEST  = Path("/ocean/projects/cis220031p/hkook/dataset/TIMIT/data/TEST")

OUT_DIR = Path("/ocean/projects/cis220031p/hkook/hailey/new_13_feats/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMIT_DIALECT_MAP = {
    'dr1': 'new_england',
    'dr2': 'northern',
    'dr3': 'north_midland',
    'dr4': 'south_midland',
    'dr5': 'southern',
    'dr6': 'new_york_city',
    'dr7': 'western',
    'dr8': 'army_brat'
}

META_PATH = Path(__file__).parent / "speaker_statistics.json"
with open(META_PATH, "r") as f:
    ears_meta = json.load(f)

ears_native_lang = {
    # use native language as a proxy for dialect in EARS, rather than ethnicity, because ethnicity does not indicate a speaker’s place of birth or linguistic background.
    spk: info.get("native language", "")
    for spk, info in ears_meta.items()
}

def extract_comprehensive_features(wav_path):
    """Extract comprehensive acoustic features from wav audio without timing information"""
    target_sr = 16000
    sound = parselmouth.Sound(str(wav_path))
    if int(sound.sampling_frequency) != target_sr:
        sound = sound.resample(target_sr)
    try:
        feats = {}

        feats["duration"] = sound.duration
        feats["sampling_frequency"] = sound.sampling_frequency

        # Pitch: 1 feature, pitch_median
        pitch = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
        pitch_vals = pitch.selected_array["frequency"]
        pitch_vals = pitch_vals[pitch_vals > 0]
        feats["pitch_median"] = np.median(pitch_vals) if len(pitch_vals) else np.nan

        # Energy: 2 features, energy_min, energy_dynamic_range
        intensity = sound.to_intensity()
        iv = intensity.values[0]
        feats["energy_min"] = np.min(iv)
        feats["energy_dynamic_range"] = np.max(iv) - feats["energy_min"]

        # Formants: 7 features, F1_mean, F1_median, F1_std, F2_mean, F2_median, F3_mean, F3_median
        formant = sound.to_formant_burg(max_number_of_formants=5, maximum_formant=5500)
        times = formant.xs()

        for i in [1, 2, 3]:
            vals = [formant.get_value_at_time(i, t) for t in times]
            vals = [v for v in vals if v > 0]
            feats[f"F{i}_mean"] = np.mean(vals) if vals else np.nan
            feats[f"F{i}_median"] = np.median(vals) if vals else np.nan
            if i == 1:
                feats["F1_std"] = np.std(vals) if vals else np.nan

        # HNR: 1 feature, hnr_mean
        try:
            hnr = sound.to_harmonicity()
            hv = hnr.values[hnr.values != -200]
            feats["hnr_mean"] = np.mean(hv) if len(hv) else np.nan
        except:
            feats["hnr_mean"] = np.nan

        # Voice quality: 2 features, jitter_local, shimmer_apq5
        try:
            pp = parselmouth.praat.call(
                sound, "To PointProcess (periodic, cc)", 75, 600
            )
            feats["jitter_local"] = parselmouth.praat.call(
                pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
            )
            feats["shimmer_apq5"] = parselmouth.praat.call(
                [sound, pp], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
            )
        except:
            feats["jitter_local"] = np.nan
            feats["shimmer_apq5"] = np.nan

        return feats

    except Exception as e:
        print(f"ERROR {wav_path}: {e}")
        return None


def process_ears():
    rows = []
    wavs = sorted(EARS_ROOT.rglob("*.wav"))

    for wav in tqdm(wavs, desc="EARS"):
        spk = wav.parent.name
        feats = extract_comprehensive_features(wav)
        if feats is None:
            continue

        rows.append({
            "filename": wav.name,
            "speaker_id": wav.parent.name,
            "dataset": "EARS",
            "file_path": str(wav),
            "accent": "",
            "native_language": ears_native_lang.get(spk, ""),
            **feats
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "ears_features.csv", index=False)


def parse_timit(path: Path):
    p = str(path).lower().replace("\\", "/")
    m = re.search(r"/(dr[1-8])/([^/]+)/", p)
    dialect = m.group(1) if m else ""
    speaker = m.group(2) if m else path.parent.name
    return speaker, dialect


def process_timit(root, split):
    rows = []
    wavs = sorted(root.rglob("*.wav"))

    for wav in tqdm(wavs, desc=f"TIMIT-{split}"):
        feats = extract_comprehensive_features(wav)
        if feats is None:
            continue

        speaker, dialect = parse_timit(wav)

        rows.append({
            "filename": wav.name,
            "speaker_id": speaker,
            "dataset": "TIMIT",
            "file_path": str(wav),
            "accent": dialect,
            "dialect": TIMIT_DIALECT_MAP.get(dialect, ""),
            "native_language": "american_english",
            **feats
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / f"timit_{split}_features.csv", index=False)


if __name__ == "__main__":
    process_ears()
    process_timit(TIMIT_TRAIN, "train")
    process_timit(TIMIT_TEST, "test")
