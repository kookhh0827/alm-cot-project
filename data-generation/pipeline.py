from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Tuple

from .abc import (
  AbstractAlignedPhonemesExtractor,
  AbstractAudioFeatExtractor,
  AbstractDatasetProcessor,
  AbstractSTTExtractor,
  AbstractTrainingDataGenerator,
)
from .schema import (
  AlignedPhonemes,
  Audio,
  AudioFeatures,
  CoTReasoningTrace,
  DatasetItem,
  ProcessedSample,
  Transcript,
)


@dataclass
class PipelineConfig:
  force_stt: bool = False
  use_stt_if_missing: bool = True


class DataGenPipeline:
  def __init__(
    self,
    dataset_processor: AbstractDatasetProcessor,
    audio_feat_extractor: AbstractAudioFeatExtractor,
    aligned_phonemes_extractor: AbstractAlignedPhonemesExtractor,
    training_data_generator: AbstractTrainingDataGenerator,
    stt_extractor: Optional[AbstractSTTExtractor] = None,
    config: Optional[PipelineConfig] = None,
  ) -> None:
    self.dataset_processor = dataset_processor
    self.audio_feat_extractor = audio_feat_extractor
    self.aligned_phonemes_extractor = aligned_phonemes_extractor
    self.training_data_generator = training_data_generator
    self.stt_extractor = stt_extractor
    self.config = config or PipelineConfig()

  def _resolve_transcript(self, item: DatasetItem) -> Transcript:
    if self.config.force_stt:
      if self.stt_extractor is None:
        raise RuntimeError("STT is required by config.force_stt but not provided")
      return self.stt_extractor.transcribe(item.audio)

    if item.transcript is not None:
      return item.transcript

    if self.config.use_stt_if_missing:
      if self.stt_extractor is None:
        raise RuntimeError("Transcript missing and STT not available")
      return self.stt_extractor.transcribe(item.audio)

    raise RuntimeError("Transcript missing and STT disabled by config")

  def iter_processed(self, dataset: Any) -> Iterator[ProcessedSample]:
    for item in self.dataset_processor.iterate(dataset):
      transcript = self._resolve_transcript(item)
      audio_features = self.audio_feat_extractor.extract(item.audio)
      aligned_phonemes = self.aligned_phonemes_extractor.align(item.audio, transcript)
      yield ProcessedSample(
        audio=item.audio,
        question=item.question,
        answer=item.answer,
        transcript=transcript,
        audio_features=audio_features,
        aligned_phonemes=aligned_phonemes,
      )

  def run(self, dataset: Any) -> List[Tuple[ProcessedSample, CoTReasoningTrace]]:
    results: List[Tuple[ProcessedSample, CoTReasoningTrace]] = []
    for sample in self.iter_processed(dataset):
      trace = self.training_data_generator.generate(sample)
      results.append((sample, trace))
    return results


