from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import os
import base64
import numpy as np  # type: ignore

from data_gen_abc import (
  AbstractAlignedPhonemesExtractor,
  AbstractAudioFeatExtractor,
  AbstractDatasetProcessor,
  AbstractSTTExtractor,
  AbstractTrainingDataGenerator,
  AbstractDataValidator,
)
from pipeline import DataGenPipeline, PipelineConfig
from schema import (
  AlignedPhoneme,
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


def _load_tests_values() -> Dict[str, Any]:
  """Execute tests/test.py with our types injected to capture fixture vars.

  The file contains an invalid import (hyphen in package name). We inject
  symbols and execute the file text to extract variables without modifying it.
  """
  repo_root = Path(__file__).resolve().parent.parent
  tests_py = repo_root / "tests" / "test.py"
  code_text = tests_py.read_text(encoding="utf-8")
  # Strip the problematic import line if present
  code_text = code_text.replace("from data-generation.types import Audio", "")
  # Prepare globals with required symbols
  g: Dict[str, Any] = {
    "Audio": Audio,
    "Transcript": Transcript,
    "Path": Path,
    "np": np,
  }
  # Execute in isolated globals dict
  exec(compile(code_text, str(tests_py), "exec"), g)
  return g


# Concrete test-driven implementations using fixtures in tests/test.py

class TestDatasetProcessor(AbstractDatasetProcessor):
  def __init__(self, audio: Audio, question: str, answer: str, transcript: Transcript) -> None:
    self.audio = audio
    self.question = question
    self.answer = answer
    self.transcript = transcript

  def iterate(self, dataset: Any) -> Iterator[DatasetItem]:
    yield DatasetItem(
      audio=self.audio,
      question=Question(self.question),
      answer=Answer(self.answer),
      transcript=self.transcript,
    )


class TestAudioFeatExtractor(AbstractAudioFeatExtractor):
  def __init__(self, features_dict: dict) -> None:
    self.features_dict = {str(k): float(v) for k, v in features_dict.items()}

  def extract(self, audio: Audio) -> AudioFeatures:
    return AudioFeatures(values=self.features_dict)


class TestAlignedPhonemesExtractor(AbstractAlignedPhonemesExtractor):
  def __init__(self, phoneme_list: List[tuple]) -> None:
    self.phoneme_list = [(str(p), float(d)) for p, d in phoneme_list]

  def align(self, audio: Audio, transcript: Transcript) -> AlignedPhonemes:
    return [AlignedPhoneme(phoneme=p, duration=d) for p, d in self.phoneme_list]


class OpenAITrainingDataGenerator(AbstractTrainingDataGenerator):
  def __init__(self, model: str = "gpt-5.1-2025-11-13") -> None:
    # Lazy import to avoid hard dependency when not used
    from openai import OpenAI  # type: ignore
    self._client = OpenAI(api_key="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")
    self._model = model

  def generate(self, sample: ProcessedSample) -> CoTReasoningTrace:
    sys_prompt = (
      "You are a speech-and-text reasoning assistant. Given audio features, "
      "aligned phonemes, question and answer, produce a concise but explicit "
      "step-by-step reasoning trace that could teach a model how to solve it. "
      "Return plain text, no JSON."
    )

    features_str = ", ".join(f"{k}={v:.4f}" for k, v in sorted(sample.audio_features.values.items()))
    phonemes_preview = ", ".join(f"{ap.phoneme}:{ap.duration:.2f}" for ap in sample.aligned_phonemes)

    # Resolve audio path and base64-encode content for multimodal input
    audio_path = sample.audio.path
    if not audio_path.exists():
      repo_root = Path(__file__).resolve().parent.parent
      candidate = repo_root / "tests" / audio_path.name
      if candidate.exists():
        audio_path = candidate
    audio_b64 = ""
    audio_fmt = audio_path.suffix.lstrip(".").lower() or "wav"
    try:
      audio_bytes = audio_path.read_bytes()
      audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
      # If audio load fails, proceed without audio (still return a trace)
      audio_b64 = ""

    # Build user prompt following the structure in example_prompt.txt
    def _format_features_block(d: Dict[str, float]) -> str:
      lines = ["{"]  # keep key order stable
      for k, v in sorted(d.items()):
        lines.append(f"  '{k}': {float(v)},")
      lines.append("}")
      return "\n".join(lines)

    def _format_phonemes_block(items: List[AlignedPhoneme]) -> str:
      pairs = ", ".join(f"('{ap.phoneme}', {ap.duration})" for ap in items)
      return f"[{pairs}]"

    template_path = Path(__file__).resolve().parent / "data_gen_prompt3.txt"
    if template_path.exists():
      tpl = template_path.read_text(encoding="utf-8")
      tpl = tpl.replace("[transcript]", sample.transcript.text)
      tpl = tpl.replace("[audio_features]", _format_features_block(sample.audio_features.values))
      tpl = tpl.replace("[aligned_phonemes]", _format_phonemes_block(sample.aligned_phonemes))
      tpl = tpl.replace("[Question]", sample.question.text)
      tpl = tpl.replace("[Answer]", sample.answer.text)
      user_text = tpl
    else:
      # Fallback: concise prompt if template missing
      user_text = (
        "We are now designing a system to generate structured audio-based chain-of-thought reasoning data.\n\n"
        f"Transcript: {sample.transcript.text}\n\n"
        f"audio_featuers: {_format_features_block(sample.audio_features.values)}\n\n"
        f"aligned_phonemes: {_format_phonemes_block(sample.aligned_phonemes)}\n\n"
        f"Question: {sample.question.text}\n"
        f"Answer: {sample.answer.text}\n"
        "Please respond with <THINK>...</THINK><RESPONSE>...</RESPONSE> as specified."
      )

    # Compose multimodal message content with text and (optional) input_audio
    user_content: List[Dict[str, Any]] = [
      {"type": "text", "text": user_text}
    ]
    # if audio_b64:
    #   user_content.append({
    #     "type": "input_audio",
    #     "input_audio": {"data": audio_b64, "format": audio_fmt},
    #   })

    resp = self._client.chat.completions.create(
      model=self._model,
      messages=[  # type: ignore[dict-item]
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
      ],  # type: ignore[list-item]
      temperature=0.3,
      max_completion_tokens=2000,
    )
    text = (resp.choices[0].message.content or "").strip()
    return CoTReasoningTrace(text=text)

class OpenAIDataValidator(AbstractDataValidator):
  def __init__(self, model: str = "gpt-5.1-2025-11-13") -> None:
    from openai import OpenAI  # type: ignore
    self._client = OpenAI(api_key="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")
    self._model = model

  def validate(self, sample: ProcessedSample, trace: CoTReasoningTrace) -> ValidationResult:
    sys_prompt = (
      "You are a strict data validator. Given the input data and a proposed CoT, "
      "judge whether the CoT is faithful, non-hallucinatory, and useful for SFT. "
      "Return a concise verdict and rationale."
    )

    # Helpers (duplicate of training generator's formatters to keep file local)
    def _format_features_block(d: Dict[str, float]) -> str:
      lines = ["{"]
      for k, v in sorted(d.items()):
        lines.append(f"  '{k}': {float(v)},")
      lines.append("}")
      return "\n".join(lines)

    def _format_phonemes_block(items: List[AlignedPhoneme]) -> str:
      pairs = ", ".join(f"('{ap.phoneme}', {ap.duration})" for ap in items)
      return f"[{pairs}]"

    # Attach audio if available
    audio_path = sample.audio.path
    if not audio_path.exists():
      repo_root = Path(__file__).resolve().parent.parent
      candidate = repo_root / "tests" / audio_path.name
      if candidate.exists():
        audio_path = candidate
    audio_b64 = ""
    audio_fmt = audio_path.suffix.lstrip(".").lower() or "wav"
    try:
      audio_bytes = audio_path.read_bytes()
      audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
      audio_b64 = ""

    template_path = Path(__file__).resolve().parent / "data_val_prompt.txt"
    if template_path.exists():
      tpl = template_path.read_text(encoding="utf-8")
      tpl = tpl.replace("[transcript]", sample.transcript.text)
      tpl = tpl.replace("[audio_features]", _format_features_block(sample.audio_features.values))
      tpl = tpl.replace("[aligned_phonemes]", _format_phonemes_block(sample.aligned_phonemes))
      tpl = tpl.replace("[Question]", sample.question.text)
      tpl = tpl.replace("[Answer]", sample.answer.text)
      tpl = tpl.replace("[CoT]", trace.text)
      user_text = tpl
    else:
      user_text = (
        "Validate the following data and CoT. Provide pass/fail and rationale.\n\n"
        f"Transcript: {sample.transcript.text}\n\n"
        f"audio_featuers: {_format_features_block(sample.audio_features.values)}\n\n"
        f"aligned_phonemes: {_format_phonemes_block(sample.aligned_phonemes)}\n\n"
        f"Question: {sample.question.text}\nAnswer: {sample.answer.text}\n\n"
        f"CoT: {trace.text}\n"
      )

    user_content: List[Dict[str, Any]] = [
      {"type": "text", "text": user_text}
    ]
    # if audio_b64:
    #   user_content.append({
    #     "type": "input_audio",
    #     "input_audio": {"data": audio_b64, "format": audio_fmt},
    #   })

    resp = self._client.chat.completions.create(
      model=self._model,
      messages=[
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_content},
      ],
      # temperature=0.2,
      max_completion_tokens=1000,
    )
    text = (resp.choices[0].message.content or "").strip()
    lowered = text.lower()
    ok: Optional[bool] = None
    if "pass" in lowered and "fail" not in lowered:
      ok = True
    if "fail" in lowered and "pass" not in lowered:
      ok = False
    return ValidationResult(text=text, is_valid=ok)

def build_pipeline() -> DataGenPipeline:
  tests_vals = _load_tests_values()

  dataset_processor = TestDatasetProcessor(
    audio=tests_vals.get("audio") or Audio(path=Path("/dev/null")),
    question=tests_vals.get("question", ""),
    answer=tests_vals.get("answer", ""),
    transcript=tests_vals.get("transcript") or Transcript(""),
  )
  audio_feat_extractor = TestAudioFeatExtractor(tests_vals.get("audio_featuers", {}))
  aligned_phonemes_extractor = TestAlignedPhonemesExtractor(tests_vals.get("aligned_phonemes", []))
  training_data_generator = OpenAITrainingDataGenerator(model=os.getenv("OPENAI_MODEL", "gpt-5.1-2025-11-13"))
  validator = OpenAIDataValidator(model=os.getenv("OPENAI_VALIDATION_MODEL", "gpt-5.1-2025-11-13"))
  config = PipelineConfig(force_stt=False, use_stt_if_missing=False)

  return DataGenPipeline(
    dataset_processor=dataset_processor,
    audio_feat_extractor=audio_feat_extractor,
    aligned_phonemes_extractor=aligned_phonemes_extractor,
    training_data_generator=training_data_generator,
    validator=validator,
    stt_extractor=None,
    config=config,
  )


if __name__ == "__main__":
  pipeline = build_pipeline()
  results = pipeline.run(dataset=None)
  for sample, trace, validation in results:
    print("=== Sample ===")
    print(f"Audio: {sample.audio.path}")
    print(f"Question: {sample.question.text}")
    print(f"Answer: {sample.answer.text}")
    print(f"AudioFeatures: {len(sample.audio_features.values)} features")
    print(f"AlignedPhonemes: {len(sample.aligned_phonemes)} items")
    print("--- CoT Reasoning Trace ---")
    print(trace.text)
    if validation is not None:
      print("--- Validation ---")
      print(validation.text)


