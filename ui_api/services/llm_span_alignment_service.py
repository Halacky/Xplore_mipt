# /home/kirill/projects_2/folium/Xplore/ui_api/services/llm_span_alignment_service.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import logging
from dataclasses import dataclass

from nlp.llm.base import LLMCallTrace, LLMClient
from nlp.llm.ensemble import EnsembleTaskRunner
from ui_api.services.patient_feature_service import get_patient_feature_extractor
from utils.json_final_validator import FinalJSONValidator

logger = logging.getLogger(__name__)


@dataclass
class SpanGuess:
    start_char: int
    end_char: int
    text: str


def _build_span_prompt(text: str, quote: str) -> str:
    return f"""
You are a precise text span locator.

You are given:
1) A SOURCE_TEXT (the exact text; indices are 0-based, end is exclusive).
2) A QUOTE which should exist (possibly with minor formatting differences) in the source.

Your task:
- Find the BEST matching contiguous span [start_char, end_char) in SOURCE_TEXT that corresponds to QUOTE.
- Return STRICT JSON ONLY, with the following schema:

{{
  "start_char": <int>,
  "end_char": <int>,
  "text": "<exact substring SOURCE_TEXT[start_char:end_char]>"
}}

Rules:
- Indices MUST be valid for SOURCE_TEXT.
- The substring MUST reflect the meaning of QUOTE as closely as possible.
- If multiple matches exist, pick the best semantic match.
- NO extra keys, NO comments, NO Markdown fencing.

SOURCE_TEXT:
\"\"\"{text}\"\"\"

QUOTE:
\"\"\"{quote}\"\"\"
""".strip()


def _build_span_judge_prompt(text: str, quote: str, raw_outputs: List[str]) -> str:
    joined = "\n\n--- OUTPUT ---\n\n".join(raw_outputs)
    return f"""
You are an aggregator of JSON span guesses.

You are given:
- SOURCE_TEXT
- QUOTE
- Several JSON objects, each of form:
  {{"start_char": int, "end_char": int, "text": "..."}}

Your task:
1. Parse all JSONs.
2. Discard obviously invalid spans (out of range, end <= start).
3. Choose the BEST consensus span that matches QUOTE in SOURCE_TEXT.
4. Return STRICT JSON with the SAME SCHEMA:

{{
  "start_char": <int>,
  "end_char": <int>,
  "text": "<exact substring SOURCE_TEXT[start_char:end_char]>"
}}

SOURCE_TEXT:
\"\"\"{text}\"\"\"

QUOTE:
\"\"\"{quote}\"\"\"

CANDIDATE_JSONS:
{joined}
""".strip()


class LLMSpanAligner:
    """
    Heavy, but precise LLM-based span aligner, using the same ensemble runner infra.
    """

    def __init__(self, ensemble_runner: EnsembleTaskRunner, repeats: int = 3) -> None:
        self.base_models = ensemble_runner.base_models
        self.judge_model = ensemble_runner.judge_model
        self.repeats = repeats
        self.validator = FinalJSONValidator(expected_type="object")

    def align_one(self, text: str, quote: str) -> Optional[SpanGuess]:
        if not text or not quote:
            return None

        raw_outputs: List[str] = []
        traces: List[LLMCallTrace] = []

        base_prompt = _build_span_prompt(text, quote)
        for model in self.base_models:
            for _ in range(self.repeats):
                tr = model.generate(base_prompt, temperature=0.1)
                raw_outputs.append(tr.raw_response or "")
                traces.append(tr)

        judge_prompt = _build_span_judge_prompt(text, quote, raw_outputs)
        judge_trace = self.judge_model.generate(judge_prompt, temperature=0.0)

        parsed = self.validator.validate(judge_trace.raw_response)
        if not isinstance(parsed, dict):
            logger.error("LLMSpanAligner: judge JSON is not object")
            return None

        try:
            start = int(parsed.get("start_char", 0))
            end = int(parsed.get("end_char", 0))
        except Exception:
            logger.error("LLMSpanAligner: invalid indices")
            return None

        start = max(0, min(start, len(text)))
        end = max(start, min(end, len(text)))
        if end <= start:
            logger.error("LLMSpanAligner: empty span after clamp")
            return None

        substr = text[start:end]
        return SpanGuess(start_char=start, end_char=end, text=substr)


_span_aligner_instance: Optional[LLMSpanAligner] = None

def get_llm_span_aligner() -> Optional[LLMSpanAligner]:
    global _span_aligner_instance
    if _span_aligner_instance is not None:
        return _span_aligner_instance

    try:
        extractor = get_patient_feature_extractor()
        runner = extractor.ensemble_runner
        _span_aligner_instance = LLMSpanAligner(ensemble_runner=runner, repeats=2)
        return _span_aligner_instance
    except Exception as e:
        logger.error("get_llm_span_aligner: failed to init LLM span aligner: %s", e)
        return None