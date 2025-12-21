from typing import Any, Dict, List
import concurrent.futures

from storage.db.session import DBSession
from storage.models.patient import PatientRecord
from storage.models.trial import TrialProtocol

from nlp.extraction.patient_features import PatientFeatureExtractor
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from nlp.evaluation.llm_eligibility import LLMBasedEligibilityEvaluator
from ui_api.services.span_refinement_service import refine_evidence_spans 

from rules_engine.decision_tree_stub import DecisionTreeEligibilityEvaluatorStub
from ml.ml_stub import MLEligibilityPredictorStub

def evaluate_patient_trial(
    db: DBSession,
    patient_id: str,
    trial_id: str,
    patient_extractor: PatientFeatureExtractor,
    trial_extractor: TrialCriteriaExtractor,
    llm_evaluator: LLMBasedEligibilityEvaluator,
) -> Dict[str, Any]:

    patient_data = db.get_patient(patient_id)
    if not patient_data:
        raise ValueError(f"Patient {patient_id} not found")

    trial_data = db.get_trial(trial_id)
    if not trial_data:
        raise ValueError(f"Trial {trial_id} not found")

    patient_record = PatientRecord(
        patient_id=patient_data["patient_id"],
        note_text=patient_data["note"],
        metadata=patient_data.get("meta_data") or {},
    )

    trial_protocol = TrialProtocol(
        trial_id=trial_data["trial_id"],
        title=trial_data.get("trial_title") or "",
        raw_protocol_document_id=trial_data["trial_id"],
        metadata={
            "trial_inclusion": trial_data.get("trial_inclusion") or "",
            "trial_exclusion": trial_data.get("trial_exclusion") or "",
            "db_meta": trial_data.get("meta_data") or {},
        },
    )

    # 1) parallel extraction
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        f_features = ex.submit(patient_extractor.extract_atomic_features, patient_record)
        f_criteria = ex.submit(trial_extractor.extract_atomic_criteria, trial_protocol)
        atomic_features = f_features.result()
        atomic_criteria = f_criteria.result()

    features_json = {
        "patient_id": patient_id,
        "features": [
            {
                "id": f.id,
                "name": f.name,
                "value": f.value,
                "value_type": f.value_type,
                "unit": f.unit,
                "confidence": f.confidence,
                "evidence_spans": [
                    {"document_id": s.document_id, "start_char": s.start_char, "end_char": s.end_char, "text": s.text}
                    for s in (f.source_spans or [])
                ],
            }
            for f in atomic_features
        ],
    }

    criteria_json = {
        "trial_id": trial_id,
        "inclusion": [
            {
                "id": c.id,
                "type": c.type,
                "description": c.description,
                "importance": getattr(c, "importance", "important"),
                "dsl_expression": c.dsl_expression,
                "source_spans": [
                    {"document_id": s.document_id, "start_char": s.start_char, "end_char": s.end_char, "text": s.text}
                    for s in (c.source_spans or [])
                ],
            }
            for c in atomic_criteria.get("inclusion", [])
        ],
        "exclusion": [
            {
                "id": c.id,
                "type": c.type,
                "description": c.description,
                "importance": getattr(c, "importance", "important"),
                "dsl_expression": c.dsl_expression,
                "source_spans": [
                    {"document_id": s.document_id, "start_char": s.start_char, "end_char": s.end_char, "text": s.text}
                    for s in (c.source_spans or [])
                ],
            }
            for c in atomic_criteria.get("exclusion", [])
        ],
        "metadata": {
            "trial_inclusion_text": trial_protocol.metadata.get("trial_inclusion", "") or "",
            "trial_exclusion_text": trial_protocol.metadata.get("trial_exclusion", "") or "",
        },
    }

    dt_eval = DecisionTreeEligibilityEvaluatorStub()
    ml_pred = MLEligibilityPredictorStub()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        f_llm = ex.submit(llm_evaluator.evaluate, patient_id, trial_id, atomic_features, atomic_criteria)
        f_dt = ex.submit(dt_eval.evaluate, patient_id, trial_id, features_json, criteria_json)
        f_ml = ex.submit(ml_pred.predict, patient_id, trial_id, features_json, criteria_json)

        llm_block = f_llm.result()
        dt_block = f_dt.result()
        ml_block = f_ml.result()

    raw_payload = {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "inputs": {
            "patient_features": features_json,
            "trial_criteria": criteria_json,
            "patient_note_text": patient_record.note_text,
        },
        "evaluation": {
            "llm": llm_block,
            "decision_tree": dt_block,
            "ml": ml_block,
        },
    }

    refined_payload = refine_evidence_spans(raw_payload)

    return refined_payload