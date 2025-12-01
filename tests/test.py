import numpy as np
from pathlib import Path
from praatio import textgrid

from data-generation.types import Audio

meta_data = {
  "age": "46-55",
  "comments": None,
  "dialect_region": None,
  "education": None,
  "ethnicity": "black or african american",
  "gender": "male",
  "height": "6' - 6'3",
  "id": "p101",
  "nationality": None,
  "native_language": "american english",
  "race": None,
  "weight": "160 - 180 lbs"
}

question = "What is the dialect of the speaker?"
answer = "Northern"


audio = Audio(path=Path("data-generation/freeform_speech_06_16k_chunk001.wav"))

transcript = Transcript(text="series called Spartacus. That was the only series I really follow. As far as any other series, no. I I don't have time to sit there and and and then I'll start on a series")


audio_featuers = {
  'duration': 10.0, 
  'sampling_frequency': 16000.0, 
  'pitch_mean': 118.77,
  'pitch_std': 100.37,
  'pitch_median': 91.83,
  'pitch_range': 518.23,
  'pitch_variance': 10075.40,
  'energy_mean': 48.58,
  'energy_std': 9.64,
  'energy_max': 69.48,
  'energy_min': 19.57,
  'energy_dynamic_range': 49.91,
  'F1_mean': 843.11,
  'F1_std': 560.48,
  'F1_median': 562.11,
  'F2_mean': 1852.03,
  'F2_std': 495.92,
  'F2_median': 1812.94,
  'F3_mean': 2793.71,
  'F3_std': 663.36,
  'F3_median': 2502.64,
  'hnr_mean': 9.87,
}

aligned_phonemes = [('s', 0.09), ('ɪ', 0.07), ('ɹ', 0.07), ('i', 0.04), ('z', 0.07), ('kʰ', 0.08), ('ɒː', 0.06), ('ɫ', 0.03), ('d', 0.03), ('s', 0.07), ('p', 0.07), ('ɑ', 0.07), ('ɹ', 0.04), ('t', 0.03), ('ə', 0.04), ('k', 0.1), ('ə', 0.07), ('s', 0.17), ('d̪', 0.12), ('æ', 0.1), ('w', 0.06), ('ɐ', 0.02), ('z', 0.08), ('d̪', 0.06), ('ə', 0.03), ('ow', 0.07), ('ɲ', 0.03), ('ʎ', 0.04), ('i', 0.04), ('s', 0.15), ('ɪ', 0.04), ('ɹ', 0.09), ('i', 0.08), ('z', 0.05), ('aj', 0.1), ('ɹ', 0.09), ('ɪ', 0.03), ('ʎ', 0.03), ('i', 0.05), ('f', 0.13), ('ɑ', 0.13), ('l', 0.05), ('ow', 0.14), ('ej', 0.08), ('z', 0.08), ('f', 0.06), ('ɑ', 0.09), ('ɹ', 0.04), ('ej', 0.03), ('z', 0.05), ('ɛ', 0.04), ('ɲ', 0.04), ('i', 0.1), ('ɐ', 0.05), ('ð', 0.04), ('ɚ', 0.05), ('s', 0.14), ('ɪ', 0.07), ('ɹ', 0.08), ('i', 0.1), ('z', 0.03), ('n', 0.07), ('ow', 0.23), ('aj', 0.16), ('aj', 0.16), ('d', 0.03), ('ow', 0.07), ('h', 0.07), ('æ', 0.08), ('v', 0.07), ('tʰ', 0.08), ('aj', 0.13), ('m', 0.03), ('ə', 0.05), ('s', 0.11), ('ɪ', 0.06), ('t', 0.05), ('d̪', 0.05), ('ɛ', 0.06), ('ɹ', 0.24), ('n̩', 0.05), ('ɛ', 0.37), ('n', 0.03), ('æ', 0.08), ('n', 0.03), ('d̪', 0.03), ('ɛ', 0.08), ('n', 0.15), ('aj', 0.11), ('ɫ̩', 0.04), ('s', 0.1), ('t', 0.07), ('ɑ', 0.09), ('ɹ', 0.03), ('t', 0.03), ('ɒ', 0.06), ('n', 0.04), ('ə', 0.06), ('s', 0.13), ('ɪ', 0.09), ('ɹ', 0.09), ('i', 0.09), ('z', 0.03)]