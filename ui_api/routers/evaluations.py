# /home/kirill/projects_2/folium/Xplore/ui_api/routers/evaluations.py

from fastapi import APIRouter, Depends, HTTPException
from typing import Any, Dict, List, Optional
import io
import csv
import json
from fastapi.responses import StreamingResponse

from storage.db.session import DBSession
from ui_api.routers.data_management import get_db_session
from ui_api.services.eligibility_evaluation_service import evaluate_patient_trial
from nlp.extraction.patient_features import PatientFeatureExtractor
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from nlp.evaluation.llm_eligibility import LLMBasedEligibilityEvaluator
from ui_api.services.patient_feature_service import get_patient_feature_extractor
from ui_api.services.trial_criteria_service import get_trial_criteria_extractor
from ui_api.services.eligibility_llm_service import get_llm_eligibility_evaluator

router = APIRouter()


def _get_eval_blocks(evaluation_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Вспомогательная функция: извлекает llm-часть и основные поля.
    """
    llm = (evaluation_json.get("evaluation") or {}).get("llm") or {}
    verdict = llm.get("verdict")
    score = llm.get("score")
    return {
        "llm_verdict": verdict,
        "llm_score": score,
    }


@router.post("/save", response_model=Dict[str, Any])
def save_evaluation(
    payload: Dict[str, Any],
    validity: str,
    db: DBSession = Depends(get_db_session),
):
    """
    Сохранить итоговый отчёт (результат evaluate_by_ids_v2).
    validity: "valid" или "invalid".
    body: полный JSON, который вернул /eligibility/evaluate_by_ids_v2
    """
    if validity not in ("valid", "invalid"):
        raise HTTPException(status_code=400, detail="validity must be 'valid' or 'invalid'")

    patient_id = payload.get("patient_id")
    trial_id = payload.get("trial_id")
    if not patient_id or not trial_id:
        raise HTTPException(status_code=400, detail="patient_id and trial_id are required in payload")

    db.save_saved_evaluation(
        {
            "patient_id": patient_id,
            "trial_id": trial_id,
            "validity": validity,
            "evaluation_json": payload,
            "user_comment": None,
        }
    )
    return {"status": "ok"}


@router.get("/list", response_model=List[Dict[str, Any]])
def list_saved(
    validity: Optional[str] = None,
    db: DBSession = Depends(get_db_session),
):

    if validity is not None and validity not in ("valid", "invalid"):
        raise HTTPException(status_code=400, detail="validity must be 'valid' or 'invalid' or omitted")
    rows = db.list_saved_evaluations(validity=validity)
    out = []
    for r in rows:
        extra = _get_eval_blocks(r["evaluation_json"] or {})
        out.append(
            {
                "id": r["id"],
                "patient_id": r["patient_id"],
                "trial_id": r["trial_id"],
                "validity": r["validity"],
                "created_at": r["created_at"],
                "llm_verdict": extra["llm_verdict"],
                "llm_score": extra["llm_score"],
            }
        )
    return out


@router.post("/re-evaluate/{saved_id}", response_model=Dict[str, Any])
def re_evaluate_saved(
    saved_id: int,
    db: DBSession = Depends(get_db_session),
    patient_extractor: PatientFeatureExtractor = Depends(get_patient_feature_extractor),
    trial_extractor: TrialCriteriaExtractor = Depends(get_trial_criteria_extractor),
    llm_eval: LLMBasedEligibilityEvaluator = Depends(get_llm_eligibility_evaluator),
):

    row = db.get_saved_evaluation(saved_id)
    if not row:
        raise HTTPException(status_code=404, detail="Saved evaluation not found")

    patient_id = row["patient_id"]
    trial_id = row["trial_id"]

    new_eval = evaluate_patient_trial(
        db=db,
        patient_id=patient_id,
        trial_id=trial_id,
        patient_extractor=patient_extractor,
        trial_extractor=trial_extractor,
        llm_evaluator=llm_eval,
    )

    return new_eval


@router.get("/export/json", response_class=StreamingResponse)
def export_json(
    validity: str,
    db: DBSession = Depends(get_db_session),
):

    if validity not in ("valid", "invalid"):
        raise HTTPException(status_code=400, detail="validity must be 'valid' or 'invalid'")

    rows = db.list_saved_evaluations(validity=validity)
    data = []
    for r in rows:
        data.append(
            {
                "id": r["id"],
                "patient_id": r["patient_id"],
                "trial_id": r["trial_id"],
                "validity": r["validity"],
                "created_at": r["created_at"].isoformat() if r["created_at"] else None,
                "evaluation": r["evaluation_json"],
            }
        )
    content = json.dumps(data, ensure_ascii=False, indent=2)
    return StreamingResponse(
        io.BytesIO(content.encode("utf-8")),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="saved_{validity}.json"'
        },
    )


@router.get("/export/csv", response_class=StreamingResponse)
def export_csv(
    validity: str,
    db: DBSession = Depends(get_db_session),
):

    if validity not in ("valid", "invalid"):
        raise HTTPException(status_code=400, detail="validity must be 'valid' or 'invalid'")

    rows = db.list_saved_evaluations(validity=validity)

    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["id", "patient_id", "trial_id", "validity", "created_at", "llm_verdict", "llm_score"])

    for r in rows:
        extra = _get_eval_blocks(r["evaluation_json"] or {})
        w.writerow(
            [
                r["id"],
                r["patient_id"],
                r["trial_id"],
                r["validity"],
                r["created_at"].isoformat() if r["created_at"] else "",
                extra["llm_verdict"] or "",
                extra["llm_score"] if extra["llm_score"] is not None else "",
            ]
        )
    output.seek(0)

    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="saved_{validity}.csv"'
        },
    )

@router.get("/detail/{saved_id}", response_model=Dict[str, Any])
def get_saved_detail(
    saved_id: int,
    db: DBSession = Depends(get_db_session),
):

    row = db.get_saved_evaluation(saved_id)
    if not row:
        raise HTTPException(status_code=404, detail="Saved evaluation not found")
    return {
        "id": row["id"],
        "patient_id": row["patient_id"],
        "trial_id": row["trial_id"],
        "validity": row["validity"],
        "created_at": row["created_at"],
        "evaluation_json": row["evaluation_json"],
        "user_comment": row["user_comment"],
    }