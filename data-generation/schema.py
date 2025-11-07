from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class Audio:
  path: Path
  sampling_rate: Optional[int] = None


@dataclass
class Transcript:
  text: str


@dataclass
class Question:
  text: str


@dataclass
class Answer:
  text: str


@dataclass
class AudioFeatures:
  values: Dict[str, float] = field(default_factory=dict)


@dataclass
class AlignedPhoneme:
  phoneme: str
  duration: float


AlignedPhonemes = List[AlignedPhoneme]


@dataclass
class DatasetItem:
  audio: Audio
  question: Question
  answer: Answer
  transcript: Optional[Transcript] = None


@dataclass
class ProcessedSample:
  audio: Audio
  question: Question
  answer: Answer
  transcript: Transcript
  audio_features: AudioFeatures
  aligned_phonemes: AlignedPhonemes


@dataclass
class CoTReasoningTrace:
  text: str


