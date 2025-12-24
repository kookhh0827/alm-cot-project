import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# --- Constants: MFA v3.1.0 English IPA Phone Set ---
# Note: MFA output often attaches stress numbers (0, 1, 2) to vowels (e.g., "æ1").
# The logic below strips numbers before checking membership.

# 1. Monophthongs (Simple Vowels)
MONOPHTHONGS = {
    "a", "aː", "e", "eː", "i", "iː", "o", "oː", "u", "uː", 
    "æ", "ɐ", "ɑ", "ɑː", "ɒ", "ɒː", "ɔ", "ə", "ɚ", "ɛ", "ɛː", 
    "ɜ", "ɜː", "ɝ", "ɪ", "ʉ", "ʉː", "ʊ"
}

# 2. Diphthongs (Complex Vowels) - Key for dialect
DIPHTHONGS = {
    "aj", "aw", "ej", "ow", "ɔj", "əw"
}

VOWELS = MONOPHTHONGS | DIPHTHONGS

# R-colored vowels + Approximant r + Flap (optional, but often acts rhotic-like)
RHOTICS = {
    "ɹ", "ɚ", "ɝ", "ɾ" 
}

# Syllabic consonants appearing in the list (m̩)
# Note: 'ɫ' (velarized l) is often syllabic but listed as consonant. 
# We include 'm̩' explicitly.
SYLLABIC_CONSONANTS = {"m̩"}

class LatentCalculator:
    def __init__(self, stats_path: Path):
        self.stats_path = stats_path
        self.stats = {}
        self.loaded = False

    def load_stats(self):
        if self.stats_path.exists():
            with open(self.stats_path, "r") as f:
                self.stats = json.load(f)
            self.loaded = True
            print(f"Loaded latent statistics from {self.stats_path}")
        else:
            print(f"No stats found at {self.stats_path}. Run fit() first.")

    def _safe_log(self, val: float) -> float:
        return np.log(val) if val > 1e-6 else 0.0

    def _compute_scalars(self, features: Dict[str, float], phonemes: List[Any]) -> Dict[str, float]:
        """Raw feature dict -> Intermediate Scalar dict for binning"""
        scalars = {}
        
        # --- Acoustic (V2.1 Enhanced) ---
        scalars['f0_mean'] = features.get('pitch_mean', 0.0)
        scalars['f0_std'] = features.get('pitch_std', 0.0)
        
        p_mean = features.get('pitch_mean', 0.0)
        p_med = features.get('pitch_median', 0.0)
        scalars['f0_skew_proxy'] = p_mean - p_med
        
        scalars['energy_mean'] = features.get('energy_mean', 0.0)
        
        e_max = features.get('energy_max', 0.0)
        e_min = features.get('energy_min', 0.0)
        if 'energy_dynamic_range' in features and features['energy_dynamic_range'] > 0:
            scalars['energy_var_proxy'] = features['energy_dynamic_range']
        else:
            scalars['energy_var_proxy'] = e_max - e_min
            
        scalars['snr_proxy'] = features.get('hnr_mean', 0.0)
        
        # New Acoustic: Voice Quality
        scalars['jitter'] = features.get('jitter_local', 0.0)
        scalars['shimmer'] = features.get('shimmer_local', 0.0)
        
        # Formants
        f1 = features.get('F1_mean', 0.0)
        f2 = features.get('F2_mean', 0.0)
        
        if f1 > 0 and f2 > 0:
            scalars['vowel_space_scale'] = self._safe_log(f2) - self._safe_log(f1)
        else:
            scalars['vowel_space_scale'] = 0.0
            
        scalars['front_back_tilt'] = f2
        
        # --- Phonology (V2.1 Enhanced) ---
        clean_phones = []
        total_dur = 0.0
        silence_count = 0
        silence_dur = 0.0
        
        for p in phonemes:
            # p.phoneme might be "æ1" or "tʃ"
            # Strip stress numbers (0-9)
            ph_raw = p.phoneme
            ph_str = re.sub(r'[0-9]+', '', ph_raw)
            
            # Detect silence for pause rate
            if ph_str in ["", "SIL", "sil", "sp", "spn", "<eps>"]: 
                silence_count += 1
                silence_dur += p.duration
                continue
            
            clean_phones.append((ph_str, p.duration))
            total_dur += p.duration
            
        n_phones = len(clean_phones)
        if n_phones > 0 and total_dur > 0:
            # phones per sec (excluding pauses) = articulation rate
            scalars['articulation_rate'] = n_phones / total_dur
            # sec per phone
            scalars['mean_phone_dur'] = total_dur / n_phones
        else:
            scalars['articulation_rate'] = 0.0
            scalars['mean_phone_dur'] = 0.0

        # Ratios
        n_vowels = sum(1 for p, _ in clean_phones if p in VOWELS)
        n_rhotics = sum(1 for p, _ in clean_phones if p in RHOTICS)
        n_diphthongs = sum(1 for p, _ in clean_phones if p in DIPHTHONGS)
        
        scalars['vowel_ratio'] = n_vowels / n_phones if n_phones > 0 else 0.0
        scalars['rhotic_ratio'] = n_rhotics / n_phones if n_phones > 0 else 0.0
        
        # Diphthong ratio: proportion of vowels that are diphthongs
        scalars['diphthong_ratio'] = n_diphthongs / n_vowels if n_vowels > 0 else 0.0
        
        # Pause Rate: pauses per second (of total time)
        full_duration = total_dur + silence_dur
        scalars['pause_rate'] = silence_count / full_duration if full_duration > 0 else 0.0
        
        # Syllabic check
        has_syllabic = any(p in SYLLABIC_CONSONANTS for p, _ in clean_phones)
        scalars['has_syllabic'] = 1.0 if has_syllabic else 0.0
        
        return scalars

    def fit(self, iterator):
        """Pass 1: Collect scalars and compute quantiles"""
        print("Collecting stats for binning...")
        accumulator = {}
        
        for item in iterator:
            scalars = self._compute_scalars(item.audio_features.values, item.aligned_phonemes)
            for k, v in scalars.items():
                if k not in accumulator:
                    accumulator[k] = []
                accumulator[k].append(v)
                
        stats = {}
        for k, vals in accumulator.items():
            arr = np.array(vals)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                stats[k] = {"q33": 0, "q66": 0}
                continue
                
            stats[k] = {
                "q33": float(np.percentile(arr, 33)),
                "q66": float(np.percentile(arr, 66)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr))
            }
            
        self.stats = stats
        self.loaded = True
        
        with open(self.stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print("Stats saved.")

    def get_bin_label(self, key: str, val: float, labels: List[str] = ["low", "mid", "high"]) -> str:
        if key not in self.stats:
            return labels[1]
        
        q33 = self.stats[key]["q33"]
        q66 = self.stats[key]["q66"]
        
        if val <= q33: return labels[0]
        elif val <= q66: return labels[1]
        else: return labels[2]

    def process_item(self, audio_features: Dict[str, float], aligned_phonemes: List[Any]) -> Dict[str, Any]:
        if not self.loaded:
            self.load_stats()
            
        scalars = self._compute_scalars(audio_features, aligned_phonemes)
        
        latent = {
            "latent_version": "v2.1",
            "acoustic": {
                "f0_level": self.get_bin_label("f0_mean", scalars["f0_mean"]),
                "f0_var": self.get_bin_label("f0_std", scalars["f0_std"]),
                "f0_skew": self.get_bin_label("f0_skew_proxy", scalars["f0_skew_proxy"]),
                "energy_level": self.get_bin_label("energy_mean", scalars["energy_mean"]),
                "energy_var": self.get_bin_label("energy_var_proxy", scalars["energy_var_proxy"]),
                "snr_proxy": self.get_bin_label("snr_proxy", scalars["snr_proxy"]),
                "voice_stability": self.get_bin_label("jitter", scalars["jitter"], ["stable", "mid", "unstable"]), # New
                "breathiness_proxy": self.get_bin_label("shimmer", scalars["shimmer"], ["clear", "mid", "breathy"]), # New
                "vowel_space_scale": self.get_bin_label("vowel_space_scale", scalars["vowel_space_scale"], ["small", "mid", "large"]),
                "front_back_tilt": self.get_bin_label("front_back_tilt", scalars["front_back_tilt"], ["front", "neutral", "back"]),
            },
            "phonology": {
                "speaking_rate": self.get_bin_label("articulation_rate", scalars["articulation_rate"], ["slow", "mid", "fast"]), # Renamed metric
                "mean_phone_dur": self.get_bin_label("mean_phone_dur", scalars["mean_phone_dur"], ["short", "mid", "long"]),
                "pause_frequency": self.get_bin_label("pause_rate", scalars["pause_rate"], ["rare", "mid", "frequent"]), # New
                "vowel_ratio": self.get_bin_label("vowel_ratio", scalars["vowel_ratio"]),
                "rhotic_phone_ratio": self.get_bin_label("rhotic_ratio", scalars["rhotic_ratio"]),
                "diphthong_index": self.get_bin_label("diphthong_ratio", scalars["diphthong_ratio"], ["low", "mid", "high"]), # New
                "syllabic_consonants": "present" if scalars["has_syllabic"] > 0.5 else "absent"
            }
        }
        return latent
