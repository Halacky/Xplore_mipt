# ui_api/services/batch_evaluation_service.py

from typing import Any, Dict, List, Optional
import logging
import concurrent.futures
from datetime import datetime

from storage.db.session import DBSession
from storage.models.patient import PatientRecord
from storage.models.trial import TrialProtocol

from nlp.extraction.patient_features import PatientFeatureExtractor
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from nlp.evaluation.llm_eligibility import LLMBasedEligibilityEvaluator

from ui_api.services.eligibility_evaluation_service import evaluate_patient_trial

logger = logging.getLogger(__name__)


def process_single_row(
    row_data: Dict[str, Any],
    db: DBSession,
    patient_extractor: PatientFeatureExtractor,
    trial_extractor: TrialCriteriaExtractor,
    llm_evaluator: LLMBasedEligibilityEvaluator,
    row_index: int,
) -> Dict[str, Any]:

    patient_id = row_data.get("patient_id")
    trial_id = row_data.get("trial_id")
    
    logger.info(
        "Processing row %d: patient_id=%s, trial_id=%s",
        row_index,
        patient_id,
        trial_id
    )
    
    try:
        patient_data = {
            "patient_id": patient_id,
            "note": row_data.get("note", ""),
            "metadata": {
                "source": "batch_csv",
                "row_index": row_index,
                "uploaded_at": datetime.now().isoformat()
            }
        }
        db.save_patient(patient_data)
        
        trial_data = {
            "trial_id": trial_id,
            "trial_title": row_data.get("trial_title", ""),
            "trial_inclusion": row_data.get("trial_inclusion", ""),
            "trial_exclusion": row_data.get("trial_exclusion", ""),
            "metadata": {
                "source": "batch_csv",
                "uploaded_at": datetime.now().isoformat()
            }
        }
        db.save_trial_protocol(trial_data)
        
        cached_result = db.get_eligibility_result(patient_id, trial_id)
        
        if cached_result:
            logger.info(
                "Row %d: Found cached result for patient_id=%s, trial_id=%s (created_at=%s)",
                row_index,
                patient_id,
                trial_id,
                cached_result["created_at"]
            )
            
            return {
                "row_index": row_index,
                "patient_id": patient_id,
                "trial_id": trial_id,
                "status": "cached",
                "result": cached_result["evaluation_json"],
                "cached_at": str(cached_result["created_at"])
            }
        
        logger.info(
            "Row %d: No cached result found, running evaluation for patient_id=%s, trial_id=%s",
            row_index,
            patient_id,
            trial_id
        )
        
        evaluation_result = evaluate_patient_trial(
            db=db,
            patient_id=patient_id,
            trial_id=trial_id,
            patient_extractor=patient_extractor,
            trial_extractor=trial_extractor,
            llm_evaluator=llm_evaluator,
        )
        
        llm_verdict = evaluation_result.get("evaluation", {}).get("llm", {}).get("verdict", "unknown")
        llm_score = evaluation_result.get("evaluation", {}).get("llm", {}).get("score")
        
        result_to_save = {
            "patient_id": patient_id,
            "trial_id": trial_id,
            "verdict": llm_verdict,
            "score": llm_score,
            "evaluation_json": evaluation_result,
            "expert_eligibility": row_data.get("expert_eligibility")  # ground truth
        }
        db.save_eligibility_result(result_to_save)
        
        logger.info(
            "Row %d: Evaluation completed and saved for patient_id=%s, trial_id=%s (verdict=%s, score=%s)",
            row_index,
            patient_id,
            trial_id,
            llm_verdict,
            llm_score
        )
        
        return {
            "row_index": row_index,
            "patient_id": patient_id,
            "trial_id": trial_id,
            "status": "evaluated",
            "result": evaluation_result,
            "verdict": llm_verdict,
            "score": llm_score,
            "expert_eligibility": row_data.get("expert_eligibility")
        }
        
    except Exception as e:
        logger.exception(
            "Row %d: Error processing patient_id=%s, trial_id=%s: %s",
            row_index,
            patient_id,
            trial_id,
            str(e)
        )
        
        return {
            "row_index": row_index,
            "patient_id": patient_id,
            "trial_id": trial_id,
            "status": "error",
            "error": str(e)
        }


def process_batch_csv(
    csv_rows: List[Dict[str, Any]],
    db: DBSession,
    patient_extractor: PatientFeatureExtractor,
    trial_extractor: TrialCriteriaExtractor,
    llm_evaluator: LLMBasedEligibilityEvaluator,
    max_workers: int = 3,
) -> Dict[str, Any]:

    logger.info(
        "Starting batch processing: %d rows, max_workers=%d",
        len(csv_rows),
        max_workers
    )
    
    start_time = datetime.now()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_row = {
            executor.submit(
                process_single_row,
                row_data=row,
                db=db,
                patient_extractor=patient_extractor,
                trial_extractor=trial_extractor,
                llm_evaluator=llm_evaluator,
                row_index=idx,
            ): idx
            for idx, row in enumerate(csv_rows)
        }
        
        for future in concurrent.futures.as_completed(future_to_row):
            row_idx = future_to_row[future]
            try:
                result = future.result()
                results.append(result)
                
                logger.info(
                    "Completed %d/%d rows",
                    len(results),
                    len(csv_rows)
                )
            except Exception as e:
                logger.exception(
                    "Unexpected error in future for row %d: %s",
                    row_idx,
                    str(e)
                )
                results.append({
                    "row_index": row_idx,
                    "status": "error",
                    "error": f"Unexpected error: {str(e)}"
                })
    
    results.sort(key=lambda x: x.get("row_index", 0))
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    stats = {
        "total_rows": len(csv_rows),
        "evaluated": sum(1 for r in results if r.get("status") == "evaluated"),
        "cached": sum(1 for r in results if r.get("status") == "cached"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "elapsed_seconds": elapsed,
        "avg_time_per_row": elapsed / len(csv_rows) if csv_rows else 0,
    }
    
    logger.info(
        "Batch processing completed: %d rows in %.2f seconds (%.2f sec/row)",
        len(csv_rows),
        elapsed,
        stats["avg_time_per_row"]
    )
    logger.info(
        "Results: evaluated=%d, cached=%d, errors=%d",
        stats["evaluated"],
        stats["cached"],
        stats["errors"]
    )
    
    return {
        "statistics": stats,
        "results": results,
        "started_at": str(start_time),
        "completed_at": str(end_time),
    }