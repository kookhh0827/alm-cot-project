import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# --- Constants: MFA v3.1.0 English IPA Phone Set ---
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

# R-colored vowels + Approximant r + Flap
RHOTICS = {
    "ɹ", "ɚ", "ɝ", "ɾ" 
}

# Syllabic consonants
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
        
        # --- Acoustic Optimized ---
        scalars['f0_mean'] = features.get('pitch_mean', 0.0)
        scalars['f0_range'] = features.get('pitch_range', 0.0)
        scalars['pitch_slope'] = features.get('pitch_slope', 0.0) # Intonation
        
        scalars['energy_mean'] = features.get('energy_mean', 0.0)
        scalars['energy_dynamic_range'] = features.get('energy_dynamic_range', 0.0)
            
        # Voice Quality
        # Combined "Tone" metric logic will be handled in binning, but we need raw values
        scalars['hnr'] = features.get('hnr_mean', 0.0)
        scalars['jitter'] = features.get('jitter_local', 0.0)
        scalars['shimmer'] = features.get('shimmer_local', 0.0)
        
        # --- Phonology Optimized ---
        clean_phones = []
        total_dur = 0.0
        silence_count = 0
        silence_dur = 0.0
        
        for p in phonemes:
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
        full_duration = total_dur + silence_dur

        if n_phones > 0 and total_dur > 0:
            # Speaking Rate: phones per second (excluding pauses)
            scalars['speaking_rate'] = n_phones / total_dur
        else:
            scalars['speaking_rate'] = 0.0

        # Pause Frequency: pauses per second (of total time)
        scalars['pause_rate'] = silence_count / full_duration if full_duration > 0 else 0.0

        # Ratios (Critical for Dialect/Ethnicity)
        n_vowels = sum(1 for p, _ in clean_phones if p in VOWELS)
        n_rhotics = sum(1 for p, _ in clean_phones if p in RHOTICS)
        n_diphthongs = sum(1 for p, _ in clean_phones if p in DIPHTHONGS)
        
        # Vowel Ratio: How "vowel-heavy" is the speech?
        scalars['vowel_ratio'] = n_vowels / n_phones if n_phones > 0 else 0.0
        
        # Rhoticity: Post-vocalic R-dropping proxy
        scalars['rhotic_ratio'] = n_rhotics / n_phones if n_phones > 0 else 0.0
        
        # Diphthong Index: Gliding vowels
        scalars['diphthong_ratio'] = n_diphthongs / n_vowels if n_vowels > 0 else 0.0
        
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
                
            # Use percentiles for robust binning
            stats[k] = {
                "q33": float(np.percentile(arr, 33)),
                "q66": float(np.percentile(arr, 66)),
                "median": float(np.median(arr)),
                "std": float(np.std(arr))
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

    def get_pitch_contour_label(self, slope: float) -> str:
        # Simple thresholding for slope
        # Slope units: Hz per frame (approx). Needs empirical check, but usually:
        # > 0.5: rising, < -0.5: falling, else: flat
        # We can use distribution stats if available, but absolute slope is often better for meaning.
        # Let's use stats-based approach for consistency first.
        
        if 'pitch_slope' not in self.stats:
            return "flat"
            
        # Slope distribution is often centered around 0.
        # Use std dev to determine significant rise/fall.
        std = self.stats['pitch_slope']['std']
        median = self.stats['pitch_slope']['median']
        
        # If slope is > median + 0.5*std -> rising
        if slope > median + 0.5 * std: return "rising"
        elif slope < median - 0.5 * std: return "falling"
        else: return "flat"

    def get_voice_tone_label(self, hnr: float, jitter: float) -> str:
        # Complex logic combining HNR (cleanliness) and Jitter (roughness)
        # 1. Check HNR first. Low HNR = Noisy/Breathy/Hoarse
        if 'hnr' in self.stats:
            hnr_q33 = self.stats['hnr']['q33']
            if hnr < hnr_q33:
                return "hoarse" # or breathy
        
        # 2. Check Jitter. High Jitter = Creaky/Rough
        if 'jitter' in self.stats:
            jit_q66 = self.stats['jitter']['q66']
            if jitter > jit_q66:
                return "creaky"
                
        # Default
        return "modal" # Normal/Clear

    def process_item(self, audio_features: Dict[str, float], aligned_phonemes: List[Any]) -> Dict[str, Any]:
        if not self.loaded:
            self.load_stats()
            
        scalars = self._compute_scalars(audio_features, aligned_phonemes)
        
        latent = {
            "latent_version": "v3.0_optimized",
            "acoustic": {
                "f0_level": self.get_bin_label("f0_mean", scalars["f0_mean"]),
                "f0_range": self.get_bin_label("f0_range", scalars["f0_range"]), # Replaces f0_var
                "pitch_contour": self.get_pitch_contour_label(scalars["pitch_slope"]), # New: Intonation
                "energy_level": self.get_bin_label("energy_mean", scalars["energy_mean"]),
                "energy_dynamics": self.get_bin_label("energy_dynamic_range", scalars["energy_dynamic_range"]), # Replaces energy_var
                "voice_tone": self.get_voice_tone_label(scalars["hnr"], scalars["jitter"]), # New: Replaces stability/breathiness
            },
            "phonology": {
                "speaking_rate": self.get_bin_label("speaking_rate", scalars["speaking_rate"], ["slow", "mid", "fast"]),
                "pause_frequency": self.get_bin_label("pause_rate", scalars["pause_rate"], ["rare", "mid", "frequent"]),
                "vowel_ratio": self.get_bin_label("vowel_ratio", scalars["vowel_ratio"]),
                "rhotic_ratio": self.get_bin_label("rhotic_ratio", scalars["rhotic_ratio"]),
                "diphthong_index": self.get_bin_label("diphthong_ratio", scalars["diphthong_ratio"], ["low", "mid", "high"]),
                "syllabic_consonants": "present" if scalars["has_syllabic"] > 0.5 else "absent"
            }
        }
        return latent
