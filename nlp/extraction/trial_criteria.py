# nlp/extraction/trial_criteria.py
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass
import json
import re
import uuid
import logging

from nlp.llm.base import LLMCallTrace
from nlp.llm.ensemble import EnsembleTaskRunner
from knowledge_base.schemas.trial import (
    AtomicCriterion,
    DocumentSpan,
)
from utils.json_repair import JSONRepairer
from utils.json_final_validator import FinalJSONValidator  # NEW
from nlp.extraction.trial_criteria_prompts import (
    build_base_criteria_extraction_prompt,
    build_per_model_criteria_aggregation_prompt,
    build_final_criteria_aggregation_prompt,
)
logger = logging.getLogger(__name__)


@dataclass
class CriterionDecompositionTrace:
    composite_id: str
    atomic_ids: List[str]
    llm_traces: List[LLMCallTrace]


class TrialCriteriaExtractor:
    """
    Extracts and atomizes inclusion/exclusion criteria from trial protocols.
    Uses the same EnsembleTaskRunner pattern as patient feature extraction.
    """

    def __init__(self, ensemble_runner: EnsembleTaskRunner):
        self.prototype_runner = ensemble_runner
        self._final_validator = FinalJSONValidator(expected_type="object")

    def _make_criteria_runner(self) -> EnsembleTaskRunner:
        return EnsembleTaskRunner(
            base_models=self.prototype_runner.base_models,
            judge_model=self.prototype_runner.judge_model,
            n_repeats=self.prototype_runner.n_repeats,
            per_model_aggregation_prompt_builder=None,
            final_aggregation_prompt_builder=None,
        )
    
    def extract_atomic_criteria(self, trial_protocol) -> Dict[str, List[AtomicCriterion]]:
        trial_id = trial_protocol.trial_id
        json_repairer = JSONRepairer(enable_llm_fallback=False)
        md = trial_protocol.metadata or {}
        inclusion_text = (md.get("trial_inclusion") or "").strip()
        exclusion_text = (md.get("trial_exclusion") or "").strip()
        trial_title = trial_protocol.title or ""
        criteria_runner = self._make_criteria_runner()
        criteria_runner.per_model_aggregation_prompt_builder = (
            lambda input_text, repeated_outputs: build_per_model_criteria_aggregation_prompt(
                inclusion_text=inclusion_text,
                exclusion_text=exclusion_text,
                trial_id=trial_id,
                repeated_outputs=repeated_outputs,
            )
        )
        criteria_runner.final_aggregation_prompt_builder = (
            lambda input_text, per_model_aggregates: build_final_criteria_aggregation_prompt(
                inclusion_text=inclusion_text,
                exclusion_text=exclusion_text,
                trial_id=trial_id,
                per_model_aggregates=per_model_aggregates,
            )
        )

        result = criteria_runner.run_task_generic(
            task_name="trial_criteria",
            input_text="(see inclusion/exclusion in prompt)",
            base_prompt_builder=lambda _: build_base_criteria_extraction_prompt(
                inclusion_text=inclusion_text,
                exclusion_text=exclusion_text,
                trial_id=trial_id,
                trial_title=trial_title,
            ),
        )

        final_trace: LLMCallTrace = result["final_trace"]
        raw = final_trace.raw_response or ""

        # Полная проверка / ремонт результата
        parsed_any = self._final_validator.validate(raw)
        if parsed_any is None:
            parsed_any = json_repairer.repair(raw, expected_type="object")

        if parsed_any is None:
            logger.error(
                "TrialCriteriaExtractor: failed to parse final aggregated criteria JSON trial_id=%s; raw start: %r",
                trial_id,
                raw[:500],
            )
            return {"inclusion": [], "exclusion": []}

        # дальше — как было
        if isinstance(parsed_any, list):
            candidate = None
            for item in parsed_any:
                if isinstance(item, dict) and (
                    "inclusion_criteria" in item or "exclusion_criteria" in item
                ):
                    candidate = item
                    break
            if candidate is None:
                logger.error(
                    "TrialCriteriaExtractor: parsed JSON is list but no suitable object found; trial_id=%s",
                    trial_id,
                )
                return {"inclusion": [], "exclusion": []}
            parsed = candidate
        elif isinstance(parsed_any, dict):
            parsed = parsed_any
        else:
            logger.error(
                "TrialCriteriaExtractor: parsed JSON is unsupported type=%s trial_id=%s",
                type(parsed_any),
                trial_id,
            )
            return {"inclusion": [], "exclusion": []}

        inc_raw = parsed.get("inclusion_criteria", []) or []
        exc_raw = parsed.get("exclusion_criteria", []) or []

        def spans_to_docspans(source: str, spans: List[Dict[str, Any]]) -> List[DocumentSpan]:
            doc_id = f"{trial_id}:{source}"
            out = []
            for s in spans or []:
                try:
                    start = int(s.get("start_char", 0))
                    end = int(s.get("end_char", 0))
                    text = str(s.get("text", ""))
                    src_text = inclusion_text if source == "inclusion" else exclusion_text
                    start = max(0, min(start, len(src_text)))
                    end = max(start, min(end, len(src_text)))
                    extracted = src_text[start:end]
                    if not text or text != extracted:
                        text = extracted
                    out.append(DocumentSpan(document_id=doc_id, start_char=start, end_char=end, text=text))
                except Exception:
                    continue
            return out

        def norm_one(c: Dict[str, Any], type_: str) -> AtomicCriterion:
            cid = c.get("id") or str(uuid.uuid4())
            desc = str(c.get("description", "")).strip()
            imp = str(c.get("importance", "important")).strip().lower()
            if imp not in {"critical", "major", "minor"}:
                # маппинг в вашу схему importance=critical|important|optional
                # чтобы не ломать AtomicCriterion: сведем major->important, minor->optional
                if imp == "major":
                    imp2 = "important"
                elif imp == "minor":
                    imp2 = "optional"
                else:
                    imp2 = "important"
            else:
                imp2 = {"critical": "critical", "major": "important", "minor": "optional"}[imp]

            # dsl_expression пока не генерим здесь (можно позже расширить)
            dsl = None
            spans = spans_to_docspans("inclusion" if type_ == "inclusion" else "exclusion", c.get("evidence_spans", []))

            return AtomicCriterion(
                id=cid,
                trial_id=trial_id,
                type=type_,
                description=desc,
                dsl_expression=dsl,
                source_spans=spans,
                importance=imp2,
            )

        return {
            "inclusion": [norm_one(c, "inclusion") for c in inc_raw],
            "exclusion": [norm_one(c, "exclusion") for c in exc_raw],
        }


    def _extract_for_block(self, trial_id: str, type_: str, block_text: str) -> List[AtomicCriterion]:
        """
        Run ensemble task for a single block (inclusion or exclusion).
        """
        result = self.ensemble_runner.run_patient_feature_task(
            note_text=block_text,
            base_prompt_builder=lambda txt: self._build_base_criteria_prompt(trial_id, type_, txt),
        )

        final_trace: LLMCallTrace = result["final_trace"]
        raw_final = final_trace.raw_response

        parsed = self._safe_parse_json(raw_final)
        if parsed is None:
            logger.warning("TrialCriteriaExtractor: failed to parse judge JSON, trying repair")
            parsed = self._attempt_simple_json_repair(raw_final)

        if parsed is None:
            logger.error("TrialCriteriaExtractor: JSON parse+repair failed for trial_id=%s type=%s", trial_id, type_)
            return []

        raw_criteria = parsed.get("criteria", []) or []
        logger.info(
            "TrialCriteriaExtractor: parsed %d raw criteria for trial_id=%s type=%s",
            len(raw_criteria), trial_id, type_
        )

        # create one coarse source span that points to the whole block
        block_span = DocumentSpan(
            document_id=f"{trial_id}:{type_}",
            start_char=0,
            end_char=len(block_text),
            text=block_text,
        )

        criteria: List[AtomicCriterion] = []
        for rc in raw_criteria:
            try:
                cid = rc.get("id") or str(uuid.uuid4())
                desc = str(rc.get("description", "")).strip()
                importance = str(rc.get("importance", "important")).strip().lower()
                if importance not in {"critical", "important", "optional"}:
                    importance = "important"

                dsl = rc.get("dsl_expression", None)
                if dsl is not None:
                    dsl = str(dsl).strip() or None

                # IMPORTANT: AtomicCriterion schema doesn't have importance field now.
                # We store it in description or in dsl_expression? Better: extend schema later.
                # For now: embed in dsl_expression metadata-like string is bad.
                # So we attach importance into description prefix, and also include in metadata via dsl_expression=None.
                # (If you want clean storage, extend AtomicCriterion dataclass.)
                # We'll do the clean way: add "importance" into dsl_expression? No.
                # -> We'll extend AtomicCriterion dataclass in knowledge_base/schemas/trial.py (see below).
                criteria.append(
                    AtomicCriterion(
                        id=cid,
                        trial_id=trial_id,
                        type=type_,
                        description=desc,
                        dsl_expression=dsl,
                        source_spans=[block_span],
                        importance=importance,  # type: ignore (added in schema change below)
                    )
                )
            except Exception as e:
                logger.exception("TrialCriteriaExtractor: failed to normalize criterion: %s", e)

        return criteria

    def _build_base_criteria_prompt(self, trial_id: str, type_: str, text: str) -> str:
        return f"""
You are an expert clinical trial protocol criteria extraction system.

Task:
From the following {type_.upper()} criteria block for trial {trial_id}, extract ALL criteria
and rewrite them as ATOMIC criteria.

"Atomic" means:
- One indivisible requirement/condition only.
- If a line contains multiple requirements (AND/OR), split into multiple atomic criteria when possible.
- Keep the meaning faithful to the protocol.

For each atomic criterion, output an object with:
- "id": optional string (may be empty)
- "description": string describing the atomic criterion
- "importance": one of ["critical","important","optional"]
  Guidance:
  - "critical": core gatekeeper criterion (e.g., diagnosis required, major safety exclusion)
  - "important": typical inclusion/exclusion but not the single main gate
  - "optional": operational/administrative or weakly enforced / less central
- "dsl_expression": optional string (can be null). If you can confidently express it using
  simple comparison on a feature name (e.g., "LVEF <= 40"), provide it; otherwise null.

VERY IMPORTANT:
- Return STRICTLY VALID JSON only. No extra commentary.
- Top-level schema must be:
{{
  "criteria": [
    {{
      "id": "...",
      "description": "...",
      "importance": "critical|important|optional",
      "dsl_expression": null
    }}
  ]
}}

Here is the criteria block text (index starts at 0):

CRITERIA_TEXT:
\"\"\"{text}\"\"\"
"""

    def _safe_parse_json(self, raw: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Try to parse raw string as JSON.
        Accept both dict and list at the top level.
        """
        raw = (raw or "").strip()
        if not raw:
            return None

        try:
            return json.loads(raw)
        except Exception:
            pass

        if raw.startswith("```"):
            raw_no_fences = re.sub(r"^```[a-zA-Z0-9]*\s*", "", raw)
            raw_no_fences = re.sub(r"\s*```$", "", raw_no_fences)
            raw_no_fences = raw_no_fences.strip()
            try:
                return json.loads(raw_no_fences)
            except Exception:
                raw = raw_no_fences  # fallthrough

        # try substring from first '{' / '[' to last '}' / ']'
        first_brace = min(
            (i for i in (raw.find("{"), raw.find("[")) if i != -1),
            default=-1,
        )
        last_brace = max(raw.rfind("}"), raw.rfind("]"))
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            candidate = raw[first_brace : last_brace + 1]
            try:
                return json.loads(candidate)
            except Exception:
                return None
        return None

    def _attempt_simple_json_repair(self, raw: str) -> Optional[Union[Dict[str, Any], List[Any]]]:
        """
        Very simple deterministic "repairs": strip markdown fences.
        """
        text = (raw or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return self._safe_parse_json(text)