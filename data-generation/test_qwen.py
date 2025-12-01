from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import numpy as np

# Dependencies for vLLM
from vllm import LLM, SamplingParams

from data_gen_abc import (
  AbstractAlignedPhonemesExtractor,
  AbstractAudioFeatExtractor,
  AbstractDatasetProcessor,
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

# -------------------------------------------------------------------------
# Test/Fixture Loading (reused from main.py)
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Reused Mock Implementations from main.py
# -------------------------------------------------------------------------

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

# -------------------------------------------------------------------------
# Qwen-based Implementations (vLLM Version)
# -------------------------------------------------------------------------

class QwenModelManager:
    """Singleton-like manager to hold the vLLM engine."""
    _llm = None
    _model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

    @classmethod
    def get_model(cls, model_name: Optional[str] = None):
        if model_name:
            cls._model_name = model_name
            
        if cls._llm is None:
            print(f"Loading Qwen model with vLLM: {cls._model_name}...")
            # gpu_memory_utilization: Adjust if OOM occurs (default 0.9)
            # max_model_len: Adjust based on context length requirements
            cls._llm = LLM(
                model=cls._model_name,
                trust_remote_code=True,
                gpu_memory_utilization=0.9,
                tensor_parallel_size=1, # Increase if using multiple GPUs
            )
        return cls._llm

class QwenTrainingDataGenerator(AbstractTrainingDataGenerator):
  def __init__(self, model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8") -> None:
    self.model_path = model_path
    self.llm = QwenModelManager.get_model(self.model_path)

  def generate(self, sample: ProcessedSample) -> CoTReasoningTrace:
    sys_prompt = (
      "You are a speech-and-text reasoning assistant. Given audio features, "
      "aligned phonemes, question and answer, produce a concise but explicit "
      "step-by-step reasoning trace that could teach a model how to solve it. "
      "Return plain text, no JSON."
    )

    def _format_features_block(d: Dict[str, float]) -> str:
      lines = ["{"]
      for k, v in sorted(d.items()):
        lines.append(f"  '{k}': {float(v)},")
      lines.append("}")
      return "\n".join(lines)

    def _format_phonemes_block(items: List[AlignedPhoneme]) -> str:
      pairs = ", ".join(f"('{ap.phoneme}', {ap.duration})" for ap in items)
      return f"[{pairs}]"

    # Load templates from the same directory
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
      user_text = (
        "We are now designing a system to generate structured audio-based chain-of-thought reasoning data.\n\n"
        f"Transcript: {sample.transcript.text}\n\n"
        f"audio_featuers: {_format_features_block(sample.audio_features.values)}\n\n"
        f"aligned_phonemes: {_format_phonemes_block(sample.aligned_phonemes)}\n\n"
        f"Question: {sample.question.text}\n"
        f"Answer: {sample.answer.text}\n"
        "Please respond with <THINK>...</THINK><RESPONSE>...</RESPONSE> as specified."
      )

    # Use the tokenizer from vLLM engine
    tokenizer = self.llm.get_tokenizer()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]
    # Apply chat template to get the full prompt string
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    sampling_params = SamplingParams(temperature=0.3, max_tokens=2000)
    
    # Generate
    outputs = self.llm.generate([prompt], sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return CoTReasoningTrace(text=generated_text.strip())


class QwenDataValidator(AbstractDataValidator):
  def __init__(self, model_path: str = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8") -> None:
    self.model_path = model_path
    self.llm = QwenModelManager.get_model(self.model_path)

  def validate(self, sample: ProcessedSample, trace: CoTReasoningTrace) -> ValidationResult:
    sys_prompt = (
      "You are a strict data validator. Given the input data and a proposed CoT, "
      "judge whether the CoT is faithful, non-hallucinatory, and useful for SFT. "
      "Return a concise verdict and rationale."
    )

    def _format_features_block(d: Dict[str, float]) -> str:
      lines = ["{"]
      for k, v in sorted(d.items()):
        lines.append(f"  '{k}': {float(v)},")
      lines.append("}")
      return "\n".join(lines)

    def _format_phonemes_block(items: List[AlignedPhoneme]) -> str:
      pairs = ", ".join(f"('{ap.phoneme}', {ap.duration})" for ap in items)
      return f"[{pairs}]"

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

    tokenizer = self.llm.get_tokenizer()
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_text},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    sampling_params = SamplingParams(max_tokens=1000) # temperature default 1.0
    
    outputs = self.llm.generate([prompt], sampling_params)
    text = outputs[0].outputs[0].text.strip()
    
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
  
  # Use the Qwen classes with vLLM
  model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"
  training_data_generator = QwenTrainingDataGenerator(model_path=model_name)
  validator = QwenDataValidator(model_path=model_name)
  
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
