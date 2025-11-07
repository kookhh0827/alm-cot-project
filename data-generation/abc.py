from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator, Iterable, Iterator, Optional

from .schema import (
  AlignedPhonemes,
  Answer,
  Audio,
  AudioFeatures,
  CoTReasoningTrace,
  ValidationResult,
  DatasetItem,
  ProcessedSample,
  Question,
  Transcript,
)


class AbstractDatasetProcessor(ABC):
  @abstractmethod
  def iterate(self, dataset: Any) -> Iterator[DatasetItem]:
    pass


class AbstractSTTExtractor(ABC):
  @abstractmethod
  def transcribe(self, audio: Audio) -> Transcript:
    pass


class AbstractAudioFeatExtractor(ABC):
  @abstractmethod
  def extract(self, audio: Audio) -> AudioFeatures:
    pass


class AbstractAlignedPhonemesExtractor(ABC):
  @abstractmethod
  def align(self, audio: Audio, transcript: Transcript) -> AlignedPhonemes:
    pass


class AbstractTrainingDataGenerator(ABC):
  @abstractmethod
  def generate(self, sample: ProcessedSample) -> CoTReasoningTrace:
    pass


class AbstractDataValidator(ABC):
  @abstractmethod
  def validate(self, sample: ProcessedSample, trace: CoTReasoningTrace) -> ValidationResult:
    pass


