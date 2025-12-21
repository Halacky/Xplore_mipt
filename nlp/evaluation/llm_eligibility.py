from typing import Any, Dict, List, Optional
import json
import re
import logging
from dataclasses import dataclass
from utils.json_repair import JSONRepairer
from utils.json_final_validator import FinalJSONValidator  # NEW
from nlp.llm.base import LLMCallTrace
from nlp.llm.ensemble import EnsembleTaskRunner
from knowledge_base.schemas.trial import AtomicCriterion
from nlp.extraction.patient_features import AtomicPatientFeature

logger = logging.getLogger(__name__)

class LLMBasedEligibilityEvaluator:
    """
    ## LLM ensemble-based eligibility evaluation:
    base_models (3) produce evaluation JSON; judge aggregates final.
    """

    def __init__(self, ensemble_runner: EnsembleTaskRunner):
        self.prototype_runner = ensemble_runner
        self._final_validator = FinalJSONValidator(expected_type="object")

    def _make_eval_runner(self) -> EnsembleTaskRunner:
        return EnsembleTaskRunner(
            base_models=self.prototype_runner.base_models,
            judge_model=self.prototype_runner.judge_model,
            n_repeats=self.prototype_runner.n_repeats,
            per_model_aggregation_prompt_builder=self._build_per_model_agg_prompt,
            final_aggregation_prompt_builder=self._build_final_agg_prompt,
        )
    
    def evaluate(
        self,
        patient_id: str,
        trial_id: str,
        features: List[AtomicPatientFeature],
        criteria: Dict[str, List[AtomicCriterion]],
    ) -> Dict[str, Any]:
        features_payload = [
            {
                "id": f.id,
                "name": f.name,
                "value": f.value,
                "value_type": f.value_type,
                "unit": f.unit,
                "confidence": f.confidence,
                "evidence_spans": [
                    {
                        "document_id": s.document_id,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                        "text": s.text,
                    }
                    for s in (f.source_spans or [])
                ],
            }
            for f in features
        ]

        criteria_payload = {
            "inclusion": [
                {
                    "id": c.id,
                    "type": c.type,
                    "description": c.description,
                    "importance": getattr(c, "importance", "important"),
                    "dsl_expression": c.dsl_expression,
                    "source_spans": [
                        {
                            "document_id": s.document_id,
                            "start_char": s.start_char,
                            "end_char": s.end_char,
                            "text": s.text,
                        }
                        for s in (c.source_spans or [])
                    ],
                }
                for c in criteria.get("inclusion", [])
            ],
            "exclusion": [
                {
                    "id": c.id,
                    "type": c.type,
                    "description": c.description,
                    "importance": getattr(c, "importance", "important"),
                    "dsl_expression": c.dsl_expression,
                    "source_spans": [
                        {
                            "document_id": s.document_id,
                            "start_char": s.start_char,
                            "end_char": s.end_char,
                            "text": s.text,
                        }
                        for s in (c.source_spans or [])
                    ],
                }
                for c in criteria.get("exclusion", [])
            ],
        }
        eval_runner = self._make_eval_runner()
        json_repairer = JSONRepairer(enable_llm_fallback=False)

        base_prompt_builder = lambda _ignored: self._build_base_prompt(
            patient_id=patient_id,
            trial_id=trial_id,
            features_json=features_payload,
            criteria_json=criteria_payload,
        )
        result = eval_runner.run_task_generic(
            task_name="eligibility_llm",
            input_text="(payloads embedded in prompt)",
            base_prompt_builder=base_prompt_builder,
        )

        final_trace: LLMCallTrace = result["final_trace"]

        parsed = self._final_validator.validate(final_trace.raw_response)
        if parsed is None:
            parsed = json_repairer.repair(final_trace.raw_response, expected_type="object")

        if parsed is None:
            return {
                "verdict": "unknown",
                "score": None,
                "matches": [],
                "non_matches": [],
                "unknowns": [],
                "commentary": "Failed to parse judge JSON. See raw_judge_response.",
                "raw_judge_response": final_trace.raw_response,
                "traces": result.get("all_traces", []),
            }


        def _normalize_bucket(bucket: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            normalized = []
            for p in bucket or []:
                feature_id = p.get("feature_id")
                feature_name = p.get("feature_name")

                if "feature_value" in p:
                    feature_value = p.get("feature_value")
                else:
                    feature_value = p.get("value")

                value_type = (
                    p.get("value_type")
                    or p.get("feature_value_type")
                    or p.get("feature_type")
                )

                feature_spans = (
                    p.get("feature_evidence_spans")
                    or p.get("evidence_spans")
                    or []
                )

                crit_obj = p.get("criterion", {})
                criterion_id = p.get("criterion_id") or crit_obj.get("id")
                criterion_type = (
                    p.get("criterion_type")
                    or crit_obj.get("type")
                    or "inclusion"
                )
                criterion_description = (
                    p.get("criterion_description")
                    or crit_obj.get("description")
                    or ""
                )

                criterion_spans = (
                    p.get("criterion_evidence_spans")
                    or crit_obj.get("source_spans")
                    or crit_obj.get("evidence_spans")
                    or []
                )

                normalized.append(
                    {
                        "feature_id": feature_id,
                        "feature_name": feature_name,
                        "feature_value": feature_value,
                        "value_type": value_type,
                        "feature_evidence_spans": feature_spans,
                        "criterion_id": criterion_id,
                        "criterion_type": criterion_type,
                        "criterion_description": criterion_description,
                        "criterion_evidence_spans": criterion_spans,
                        "rationale": p.get("rationale"),
                    }
                )
            return normalized

        parsed["matches"] = _normalize_bucket(parsed.get("matches", []))
        parsed["non_matches"] = _normalize_bucket(parsed.get("non_matches", []))
        parsed["unknowns"] = _normalize_bucket(parsed.get("unknowns", []))

        feature_index: Dict[str, List[Dict[str, Any]]] = {}
        for f in features:
            spans = []
            for s in f.source_spans or []:
                spans.append(
                    {
                        "document_id": s.document_id,
                        "start_char": s.start_char,
                        "end_char": s.end_char,
                        "text": s.text,
                    }
                )
            feature_index[f.id] = spans

        def _merge_feature_spans(bucket: List[Dict[str, Any]]) -> None:

            for p in bucket:
                fid = p.get("feature_id")
                if p.get("feature_evidence_spans"):
                    continue
                if not fid:
                    continue
                if fid not in feature_index:
                    continue
                p["feature_evidence_spans"] = list(feature_index[fid])

        _merge_feature_spans(parsed["matches"])
        _merge_feature_spans(parsed["non_matches"])
        _merge_feature_spans(parsed["unknowns"])
        # ------------------------------------------------------------------

        parsed["traces"] = result.get("all_traces", [])
        return parsed

    def _build_base_prompt(
        self,
        patient_id: str,
        trial_id: str,
        features_json: List[Dict[str, Any]],
        criteria_json: Dict[str, Any],
    ) -> str:
        return f"""
You are a clinical trial eligibility evaluator.

## You are given:
1) ATOMIC patient features (already extracted), with evidence spans referencing the patient note
2) ATOMIC trial criteria (already extracted), with evidence spans referencing the protocol text

## Task:
## Evaluate eligibility. Produce:
- verdict: "included" or "excluded" or "unknown"
- score: 0..100 (integer)
- matches: list of matched (feature <-> criterion) pairs
- non_matches: list of conflicts (feature <-> criterion) pairs that fail eligibility
- unknowns: list of pairs where evidence is insufficient/ambiguous
- commentary: string with assumptions, missing info, caveats

## For each pair object you MUST include:
- feature_id, feature_name, feature_value, feature_evidence_spans (copy evidence spans)
- criterion_id, criterion_type, criterion_description, criterion_evidence_spans (copy source spans)
- rationale: short explanation

## VERY IMPORTANT:
- Output STRICT JSON only. No markdown.
- Use ONLY the provided JSON; do not invent new features/criteria.
- If a criterion is not addressable by any feature, put it into unknowns.

## Schema:
{{
  "verdict": "included|excluded|unknown",
  "score": 0,
  "matches": [{{"feature_id":"...","feature_name":"...","feature_value":...,"feature_evidence_spans":[...],
               "criterion_id":"...","criterion_type":"inclusion|exclusion","criterion_description":"...","criterion_evidence_spans":[...],
               "rationale":"..."}}],
  "non_matches": [ ... same shape ... ],
  "unknowns": [ ... same shape ... ],
  "commentary": "..."
}}

patient_id: {patient_id}
trial_id: {trial_id}

# PATIENT_FEATURES_JSON:
{json.dumps(features_json, ensure_ascii=False)}

## TRIAL_CRITERIA_JSON:
{json.dumps(criteria_json, ensure_ascii=False)}
"""

    def _build_per_model_agg_prompt(self, _input_text: str, repeated_outputs: List[str]) -> str:
        joined = "\n\n--- OUTPUT ---\n\n".join(repeated_outputs)
        return f"""
You are a strict JSON aggregator for eligibility evaluations.

You are given multiple JSON outputs from the SAME base model (same task, same inputs).
Your task: merge them into ONE JSON of the same schema, and add confidence-like consistency in commentary if needed.
If fields disagree, choose the most supported across runs.

## Output STRICT JSON only with the SAME schema:
{{
  "verdict": "...",
  "score": 0,
  "matches": [...],
  "non_matches": [...],
  "unknowns": [...],
  "commentary": "..."
}}

## Here are the repeated outputs:
{joined}
"""

    def _build_final_agg_prompt(self, _input_text: str, per_model_aggregates: List[str]) -> str:
        joined = "\n\n--- PER-MODEL ---\n\n".join(per_model_aggregates)
        return f"""
You are the ultimate judge who evaluates multiple JSONs suitable for each model.

Combine them into one final JSON.
MANDATORY RULES (MUST NOT BE VIOLATED):

1. Any quantitative threshold (>, ≥, <, ≤, between, etc.) is a STRICT CRITERION.
   - If x < 40% is written and x = 60%, then:
     - THIS IS AN AUTOMATIC NON-COMPLIANCE WITH THE CRITERION.
     - It CANNOT be interpreted as a match because of the "context" or other signs.
   - Only an error of up to 5% of the threshold is allowed.
     Example: the threshold is less than 40, 41-42 is acceptable, BUT NOT 60.

2. If at least one quantitative criterion is definitely violated (taking into account the 5% tolerance),
then:
- this criterion SHOULD be marked as "not_met" or similar;
   - if there is such a violation, you need to be careful.:
     - according to this criterion, it is IMPOSSIBLE to write that it is fulfilled;
     - in the final verdict, this SHOULD affect the rejection or reduction of the assessment.

3. You CANNOT "overdo" or "soften" the wording of numerical criteria.
   Prohibited:
- "the clinical context supports inclusion, despite..."
- "the criterion is not formally fulfilled, but in general the patient is suitable"
   If the criterion is not met, it is NOT fulfilled. Point.

4. Final decision:
- We CAN give "exclude" only if ≥ 50% of private answers give "exclude"
     OR if at least one STRICT quantitative criterion is violated WITHOUT admission.
   - "included" is allowed ONLY if:
     - no strict quantitative criteria have been violated (taking into account the 5% tolerance);
     - and most of the private answers do not give "exclude".
{{
"verdict": "included|excluded|unknown",
"score": 0,
"matches": [...],
"inconsistencies": [...],
"unknown": [...],
"comment": "..."
}}

Aggregates for each model:
{joined}
"""
