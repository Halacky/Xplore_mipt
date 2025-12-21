# /home/kirill/projects_2/folium/Xplore/ui_api/routers/eligibility.py

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from typing import Any, Dict, List
import logging

from pipeline import EligibilityPipeline
from storage.db.session import DBSession
from storage.models.patient import PatientRecord
from ui_api.routers.data_management import get_db_session
from ui_api.services.llm_service import get_llm_ensemble
from ui_api.services.patient_feature_service import get_patient_feature_extractor
from nlp.extraction.patient_features import PatientFeatureExtractor
from ui_api.services.trial_criteria_service import get_trial_criteria_extractor
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from storage.models.trial import TrialProtocol
from ui_api.services.eligibility_evaluation_service import evaluate_patient_trial
from ui_api.services.eligibility_llm_service import get_llm_eligibility_evaluator
from nlp.evaluation.llm_eligibility import LLMBasedEligibilityEvaluator
import pandas as pd
import io
from ui_api.services.batch_evaluation_service import process_batch_csv
from utils.data_preprocessing import DatasetCleaner
from prometheus_client import Summary

logger = logging.getLogger(__name__)

router = APIRouter()

EVAL_V2_LATENCY = Summary(
    "evaluate_by_ids_v2_latency_seconds",
    "Latency for /eligibility/evaluate_by_ids_v2 endpoint"
)

def get_pipeline() -> EligibilityPipeline:
    # Placeholder for full pipeline (not yet implemented)
    raise NotImplementedError("Full eligibility pipeline not yet implemented")


@router.post("/evaluate", response_model=Dict[str, Any])
def evaluate_single(
    payload: Dict[str, Any],
    trial_id: str,
    pipeline: EligibilityPipeline = Depends(get_pipeline),
) -> Dict[str, Any]:
    """
    Evaluate single patient eligibility using full pipeline (NOT YET IMPLEMENTED)
    """
    return pipeline.run_single(payload, trial_id)


@router.post("/batch_evaluate", response_model=List[Dict[str, Any]])
def evaluate_batch(
    payloads: List[Dict[str, Any]],
    trial_id: str,
    pipeline: EligibilityPipeline = Depends(get_pipeline),
) -> List[Dict[str, Any]]:
    """
    Batch evaluate patient eligibility using full pipeline (NOT YET IMPLEMENTED)
    """
    return pipeline.run_batch(payloads, trial_id)


@router.post("/extract-features", response_model=Dict[str, Any])
def extract_patient_features(
    patient_id: str,
    db: DBSession = Depends(get_db_session),
    extractor: PatientFeatureExtractor = Depends(get_patient_feature_extractor),
) -> Dict[str, Any]:
    """
    Extract atomic patient features from clinical note using ensemble of LLMs.
    
    This endpoint:
    1. Loads patient from database
    2. Runs ensemble extraction (multiple models x multiple repeats)
    3. Returns structured features with evidence spans and confidence
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        {
            "patient_id": str,
            "features": [
                {
                    "id": str,
                    "name": str,
                    "value": Any,
                    "value_type": str,
                    "unit": str,
                    "confidence": float,  # if available from judge
                    "evidence_spans": [
                        {
                            "start_char": int,
                            "end_char": int,
                            "text": str
                        }
                    ]
                }
            ],
            "extraction_metadata": {
                "total_features": int,
                "models_used": List[str],
                "total_llm_calls": int
            }
        }
    """
    
    logger.info("Feature extraction requested for patient_id=%s", patient_id)
    
    # Load patient from database
    patient_data = db.get_patient(patient_id)
    if not patient_data:
        logger.error("Patient %s not found in database", patient_id)
        raise HTTPException(
            status_code=404,
            detail=f"Patient {patient_id} not found in database"
        )
    
    logger.info(
        "Patient loaded: patient_id=%s, note_length=%d chars",
        patient_id,
        len(patient_data.get("note", ""))
    )
    
    # Create PatientRecord object
    patient_record = PatientRecord(
        patient_id=patient_data["patient_id"],
        note_text=patient_data["note"],
        metadata=patient_data.get("meta_data", {})
    )
    
    try:
        # Extract features using ensemble
        logger.info("Starting ensemble feature extraction for patient_id=%s", patient_id)
        
        features = extractor.extract_atomic_features(patient_record)
        
        logger.info(
            "Feature extraction completed: patient_id=%s, features_extracted=%d",
            patient_id,
            len(features)
        )
        
        # Convert features to response format
        features_list = []
        for f in features:
            feature_dict = {
                "id": f.id,
                "name": f.name,
                "value": f.value,
                "value_type": f.value_type,
                "unit": f.unit,
                "confidence": getattr(f, "confidence", None),
                "evidence_spans": [
                    {
                        "start_char": span.start_char,
                        "end_char": span.end_char,
                        "text": span.text
                    }
                    for span in f.source_spans
                ]
            }
            features_list.append(feature_dict)
        
        # Build metadata about extraction process
        models_used = set()
        
        total_llm_calls = 0
        if features and features[0].extraction_traces:
            total_llm_calls = len(features[0].extraction_traces)
            for trace in features[0].extraction_traces:
                models_used.add(f"{trace.model_name} ({trace.provider})")
        
        response = {
            "patient_id": patient_id,
            "features": features_list,
            "extraction_metadata": {
                "total_features": len(features_list),
                "models_used": list(models_used),
                "total_llm_calls": total_llm_calls,
            }
        }
        
        logger.info(
            "Returning response: patient_id=%s, features=%d, llm_calls=%d",
            patient_id,
            len(features_list),
            total_llm_calls
        )
        
        return response
        
    except Exception as e:
        logger.exception(
            "Feature extraction failed for patient_id=%s: %s",
            patient_id,
            str(e)
        )
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )

@router.post("/batch-evaluate-csv", response_model=Dict[str, Any])
async def batch_evaluate_from_csv(
    csv_file: UploadFile = File(..., description="CSV file with patient-trial pairs"),
    max_workers: int = 3,
    db: DBSession = Depends(get_db_session),
    patient_extractor: PatientFeatureExtractor = Depends(get_patient_feature_extractor),
    trial_extractor: TrialCriteriaExtractor = Depends(get_trial_criteria_extractor),
    llm_eval: LLMBasedEligibilityEvaluator = Depends(get_llm_eligibility_evaluator),
) -> Dict[str, Any]:
    logger.info("Batch evaluation CSV upload started: filename=%s", csv_file.filename)
    
    try:
        contents = await csv_file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = DatasetCleaner.validate_and_clean(df)
        logger.info(
            "CSV loaded: %d rows, columns=%s",
            len(df),
            list(df.columns)
        )
        
        required_columns = [
            "patient_id",
            "note",
            "trial_id",
            "trial_title",
            "trial_inclusion",
            "trial_exclusion"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        print(df.columns)
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        csv_rows = df.to_dict('records')
        
        result = process_batch_csv(
            csv_rows=csv_rows,
            db=db,
            patient_extractor=patient_extractor,
            trial_extractor=trial_extractor,
            llm_evaluator=llm_eval,
            max_workers=max_workers,
        )
        
        logger.info(
            "Batch evaluation completed: total=%d, evaluated=%d, cached=%d, errors=%d",
            result["statistics"]["total_rows"],
            result["statistics"]["evaluated"],
            result["statistics"]["cached"],
            result["statistics"]["errors"]
        )
        
        return result
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="CSV file is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(status_code=400, detail=f"CSV parsing error: {str(e)}")
    except Exception as e:
        logger.exception("Batch evaluation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")
    
@router.post("/evaluate_by_ids_v2", response_model=Dict[str, Any])
@EVAL_V2_LATENCY.time()
def evaluate_by_ids_v2(
    patient_id: str,
    trial_id: str,
    db: DBSession = Depends(get_db_session),
    patient_extractor: PatientFeatureExtractor = Depends(get_patient_feature_extractor),
    trial_extractor: TrialCriteriaExtractor = Depends(get_trial_criteria_extractor),
    llm_eval: LLMBasedEligibilityEvaluator = Depends(get_llm_eligibility_evaluator),
) -> Dict[str, Any]:
    """
    ## V2: full deterministic pipeline:
    - load from DB
    - parallel extraction: patient atomic features + trial atomic criteria (ensemble: 3 models x 3 repeats)
    - parallel evaluation: LLM strategy + decision tree stub + ML stub
    - return combined JSON
    """
    try:
        return evaluate_patient_trial(
            db=db,
            patient_id=patient_id,
            trial_id=trial_id,
            patient_extractor=patient_extractor,
            trial_extractor=trial_extractor,
            llm_evaluator=llm_eval,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("evaluate_by_ids_v2 failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate_by_ids", response_model=Dict[str, Any])
def evaluate_by_ids(
    patient_id: str,
    trial_id: str,
    db: DBSession = Depends(get_db_session),
    ensemble = Depends(get_llm_ensemble),
) -> Dict[str, Any]:
    """
    Simple eligibility evaluation using basic LLM ensemble (legacy endpoint).
    
    NOTE: For production, use /extract-features instead, which provides:
    - Multi-model ensemble with proper aggregation
    - Structured atomic features
    - Evidence spans and traceability
    
    This endpoint is kept for backward compatibility.
    """
    
    logger.warning(
        "Legacy endpoint /evaluate_by_ids called. Consider using /extract-features instead."
    )

    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    trial = db.get_trial(trial_id)
    if not trial:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found")

    patient_note = patient.get("note", "")
    trial_title = trial.get("trial_title", "")
    trial_inclusion = trial.get("trial_inclusion", "")
    trial_exclusion = trial.get("trial_exclusion", "")

    prompt = f"""
You are a Clinical Trial Eligibility Assessment Assistant.

Project Overview
Given:
1) Patient's Clinical Record
2) Study Title
3) Inclusion Criteria
4) Exclusion Criteria

Determine whether the patient meets the study's eligibility criteria (included or excluded)
and provide a structured explanation.

The overall solution concept is the maximum atomicity of each module and the principle of determinism.

Your role and task:
We need to ensure that the patient's Clinical Record is broken down into the simplest elements that can be used as nodes in a decision tree.

Generate a response as a JSON file, where the keys are the selected features, and the values are an object that will store the found feature value, the index of the position from which the identified feature information was retrieved, and the context (as a text fragment).

Also, do not include anything before or after the generated JSON file in the response.

Patient Note:
{patient_note}

Trial Title:
{trial_title}

Inclusion Criteria:
{trial_inclusion}

Exclusion Criteria:
{trial_exclusion}
"""

    result = ensemble.run_task(prompt)
    final_answer = result.get("final_answer")
    traces = result.get("traces", [])

    return {
        "patient_id": patient_id,
        "trial_id": trial_id,
        "raw_llm_answer": final_answer,
        "llm_traces": traces,
    }


@router.post("/extract-criteria", response_model=Dict[str, Any])
def extract_trial_criteria(
    trial_id: str,
    db: DBSession = Depends(get_db_session),
    extractor: TrialCriteriaExtractor = Depends(get_trial_criteria_extractor),
 ) -> Dict[str, Any]:
    """
    Extract atomic inclusion/exclusion criteria from stored trial protocol using ensemble.
    Returns criteria with importance labels.
    """
    logger.info("Criteria extraction requested for trial_id=%s", trial_id)

    trial_data = db.get_trial(trial_id)
    if not trial_data:
        raise HTTPException(status_code=404, detail=f"Trial {trial_id} not found in database")

    inclusion = (trial_data.get("trial_inclusion") or "")
    exclusion = (trial_data.get("trial_exclusion") or "")

    trial_protocol = TrialProtocol(
        trial_id=trial_data["trial_id"],
        title=trial_data.get("trial_title") or "",
        raw_protocol_document_id=trial_data["trial_id"],
        metadata={
            "trial_inclusion": inclusion,
            "trial_exclusion": exclusion,
            "db_meta": trial_data.get("meta_data") or {},
        },
    )

    criteria = extractor.extract_atomic_criteria(trial_protocol)

    def _crit_to_dict(c):
        return {
            "id": c.id,
            "trial_id": c.trial_id,
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

    inc_list = [_crit_to_dict(c) for c in criteria.get("inclusion", [])]
    exc_list = [_crit_to_dict(c) for c in criteria.get("exclusion", [])]

    models_used = set()
    total_llm_calls = 0

    return {
        "trial_id": trial_id,
        "inclusion": inc_list,
        "exclusion": exc_list,
        "extraction_metadata": {
            "total_inclusion": len(inc_list),
            "total_exclusion": len(exc_list),
            "models_used": list(models_used),
            "total_llm_calls": total_llm_calls,
        },
    }