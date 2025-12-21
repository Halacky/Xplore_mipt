from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Dict, Any, Optional, List
from contextlib import contextmanager


class DBSession:
    def __init__(self, connection_params: Dict[str, Any]):
        self.connection_params = connection_params
        db_url = connection_params.get(
            "url", 
            "sqlite:///./clinical_trials.db"
        )
        self.engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
            echo=connection_params.get("echo", False)
        )
        self.SessionLocal = sessionmaker(
            autocommit=False, 
            autoflush=False, 
            bind=self.engine
        )
    
    @contextmanager
    def get_session(self):
        """Context manager for database sessions"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()
    
    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """Get patient by ID, returns dict"""
        from storage.db.models import Patient
        with self.get_session() as session:
            patient = session.query(Patient).filter(
                Patient.patient_id == patient_id
            ).first()
            
            if patient:
                return {
                    "patient_id": patient.patient_id,
                    "note": patient.note,
                    "meta_data": patient.meta_data,
                    "created_at": patient.created_at,
                    "updated_at": patient.updated_at
                }
            return None
    
    def save_patient(self, patient_data: Dict[str, Any]) -> None:
        """Save or update patient"""
        from storage.db.models import Patient
        with self.get_session() as session:
            patient = session.query(Patient).filter(
                Patient.patient_id == patient_data["patient_id"]
            ).first()
            
            if patient:
                patient.note = patient_data.get("note", patient.note)
                patient.meta_data = patient_data.get("metadata", patient.meta_data)
            else:
                patient = Patient(
                    patient_id=patient_data["patient_id"],
                    note=patient_data["note"],
                    meta_data=patient_data.get("metadata", {})
                )
                session.add(patient)
    
    def get_trial(self, trial_id: str) -> Optional[Dict[str, Any]]:
        """Get trial protocol by ID, returns dict"""
        from storage.db.models import TrialProtocol
        with self.get_session() as session:
            trial = session.query(TrialProtocol).filter(
                TrialProtocol.trial_id == trial_id
            ).first()
            
            if trial:
                return {
                    "trial_id": trial.trial_id,
                    "trial_title": trial.trial_title,
                    "trial_inclusion": trial.trial_inclusion,
                    "trial_exclusion": trial.trial_exclusion,
                    "protocol_pdf_path": trial.protocol_pdf_path,
                    "meta_data": trial.meta_data,
                    "created_at": trial.created_at,
                    "updated_at": trial.updated_at
                }
            return None
    
    def save_trial_protocol(self, trial_data: Dict[str, Any]) -> None:
        """Save or update trial protocol"""
        from storage.db.models import TrialProtocol
        with self.get_session() as session:
            trial = session.query(TrialProtocol).filter(
                TrialProtocol.trial_id == trial_data["trial_id"]
            ).first()
            
            if trial:
                for key in ["trial_title", "trial_inclusion", "trial_exclusion", 
                           "protocol_pdf_path"]:
                    if key in trial_data:
                        setattr(trial, key, trial_data[key])
                if "metadata" in trial_data:
                    trial.meta_data = trial_data["metadata"]
            else:
                trial_data_copy = trial_data.copy()
                if "metadata" in trial_data_copy:
                    trial_data_copy["meta_data"] = trial_data_copy.pop("metadata")
                trial = TrialProtocol(**trial_data_copy)
                session.add(trial)
    
    def get_eligibility_result(
        self, 
        patient_id: str, 
        trial_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached eligibility result if exists"""
        from storage.db.models import EligibilityResult
        with self.get_session() as session:
            result = session.query(EligibilityResult).filter(
                EligibilityResult.patient_id == patient_id,
                EligibilityResult.trial_id == trial_id
            ).order_by(EligibilityResult.created_at.desc()).first()
            
            if result:
                return {
                    "id": result.id,
                    "patient_id": result.patient_id,
                    "trial_id": result.trial_id,
                    "verdict": result.verdict,
                    "score": result.score,
                    "evaluation_json": result.evaluation_json,
                    "expert_eligibility": result.expert_eligibility,
                    "created_at": result.created_at
                }
            return None
    
    def save_eligibility_result(self, result_data: Dict[str, Any]) -> None:
        """Save eligibility evaluation result"""
        from storage.db.models import EligibilityResult
        with self.get_session() as session:
            result = EligibilityResult(
                patient_id=result_data["patient_id"],
                trial_id=result_data["trial_id"],
                verdict=result_data["verdict"],
                score=result_data.get("score"),
                evaluation_json=result_data["evaluation_json"],
                expert_eligibility=result_data.get("expert_eligibility")
            )
            session.add(result)
    
    def init_db(self):
        """Initialize database tables"""
        from storage.db.models import Base
        Base.metadata.create_all(bind=self.engine)

    def save_saved_evaluation(self, data: Dict[str, Any]) -> None:
        from storage.db.models import SavedEvaluation
        with self.get_session() as session:
            obj = SavedEvaluation(
                patient_id=data["patient_id"],
                trial_id=data["trial_id"],
                validity=data["validity"],  # "valid" / "invalid"
                evaluation_json=data["evaluation_json"],
                user_comment=data.get("user_comment"),
            )
            session.add(obj)

    def list_saved_evaluations(
        self,
        validity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        from storage.db.models import SavedEvaluation
        with self.get_session() as session:
            q = session.query(SavedEvaluation)
            if validity in ("valid", "invalid"):
                q = q.filter(SavedEvaluation.validity == validity)
            q = q.order_by(SavedEvaluation.created_at.desc())
            rows = q.all()
            out: List[Dict[str, Any]] = []
            for r in rows:
                out.append(
                    {
                        "id": r.id,
                        "patient_id": r.patient_id,
                        "trial_id": r.trial_id,
                        "validity": r.validity,
                        "evaluation_json": r.evaluation_json,
                        "user_comment": r.user_comment,
                        "created_at": r.created_at,
                    }
                )
            return out

    def get_saved_evaluation(self, saved_id: int) -> Optional[Dict[str, Any]]:
        from storage.db.models import SavedEvaluation
        with self.get_session() as session:
            r = session.query(SavedEvaluation).filter(
                SavedEvaluation.id == saved_id
            ).first()
            if not r:
                return None
            return {
                "id": r.id,
                "patient_id": r.patient_id,
                "trial_id": r.trial_id,
                "validity": r.validity,
                "evaluation_json": r.evaluation_json,
                "user_comment": r.user_comment,
                "created_at": r.created_at,
            }