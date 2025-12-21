# rules_engine/scoring.py
from typing import Dict, Any, List
from dataclasses import dataclass

from storage.models.patient import PatientRecord
from storage.models.trial import TrialProtocol
from nlp.extraction.patient_features import AtomicPatientFeature
from knowledge_base.schemas.trial import AtomicCriterion
from rules_engine.evaluators.criteria import (
    CriterionEvaluator,
    CriterionEvaluationResult,
)


@dataclass
class EligibilityDecision:
    """
    Final eligibility decision for a patient-trial pair, with explanation.
    """
    patient_id: str
    trial_id: str
    included: bool
    main_reason: str
    inclusion_results: List[CriterionEvaluationResult]
    exclusion_results: List[CriterionEvaluationResult]
    # Can be used to store scores, confidence, etc.
    metadata: Dict[str, Any]


class EligibilityScoringEngine:
    """
    High-level engine that uses CriterionEvaluator to calculate
    eligibility and build an explainable decision.
    """

    def __init__(self, criterion_evaluator: CriterionEvaluator):
        self.criterion_evaluator = criterion_evaluator

    def evaluate(
        self,
        patient: PatientRecord,
        atomic_features: List[AtomicPatientFeature],
        trial: TrialProtocol,
        atomic_criteria: Dict[str, List[AtomicCriterion]],
    ) -> Dict[str, Any]:
        """
        Orchestrate inclusions/exclusions evaluation and build final decision.

        atomic_criteria:
            {
                "inclusion": [...],
                "exclusion": [...]
            }

        Returns:
            dictionary serializable for API/DB:
            {
                "decision": EligibilityDecision.asdict(),
                "trace": {...}
            }
        """
        features_by_name = {
            f.name: f for f in atomic_features
        }

        inclusion_results: List[CriterionEvaluationResult] = []
        for c in atomic_criteria.get("inclusion", []):
            inclusion_results.append(
                self.criterion_evaluator.evaluate(c, features_by_name)
            )

        exclusion_results: List[CriterionEvaluationResult] = []
        for c in atomic_criteria.get("exclusion", []):
            exclusion_results.append(
                self.criterion_evaluator.evaluate(c, features_by_name)
            )

        included, main_reason = self._aggregate_decision(
            inclusion_results, exclusion_results
        )

        decision = EligibilityDecision(
            patient_id=patient.patient_id,
            trial_id=trial.trial_id,
            included=included,
            main_reason=main_reason,
            inclusion_results=inclusion_results,
            exclusion_results=exclusion_results,
            metadata={},
        )

        trace = self._build_trace(
            patient, trial, decision, atomic_features, atomic_criteria
        )

        return {
            "decision": decision,
            "trace": trace,
        }

    def _aggregate_decision(
        self,
        inclusion_results: List[CriterionEvaluationResult],
        exclusion_results: List[CriterionEvaluationResult],
    ) -> (bool, str):
        """
        Aggregate atomic results into a final eligibility decision.

        Simple default logic:
          - Patient is eligible only if all inclusion criteria passed
            AND no exclusion criteria passed.
        """
        # TODO: implement actual aggregation logic, can be extended.
        raise NotImplementedError

    def _build_trace(
        self,
        patient: PatientRecord,
        trial: TrialProtocol,
        decision: EligibilityDecision,
        atomic_features: List[AtomicPatientFeature],
        atomic_criteria: Dict[str, List[AtomicCriterion]],
    ) -> Dict[str, Any]:
        """
        Build explainability trace structure linking:
         - Patient source text spans
         - Trial criteria source text spans
         - LLM calls and decisions
        """
        # TODO: implement trace building; this is where you maintain
        # full provenance for audit/explainability.
        raise NotImplementedError