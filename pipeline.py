# pipeline.py
from typing import List, Dict, Any

from nlp.feature_extractor import PatientFeatureExtractor, TrialCriteriaExtractor
from rules_engine.scoring import EligibilityScoringEngine
from storage.models.patient import PatientRecord
from storage.models.trial import TrialProtocol
from storage.db.session import DBSession
# from utils.data_preprocessing import RawPatientNotePreprocessor


class EligibilityPipeline:
    """
    Orchestrates the full eligibility evaluation pipeline:
    1) Preprocess raw input
    2) Extract atomic patient features
    3) Extract atomic trial criteria
    4) Evaluate eligibility via rules/DSL
    5) Persist results with explainability traces
    """

    def __init__(
        self,
        db_session: DBSession,
        patient_extractor: PatientFeatureExtractor,
        trial_extractor: TrialCriteriaExtractor,
        scoring_engine: EligibilityScoringEngine,
    ):
        self.db_session = db_session
        self.patient_extractor = patient_extractor
        self.trial_extractor = trial_extractor
        self.scoring_engine = scoring_engine
        # self.preprocessor = RawPatientNotePreprocessor()

    def run_single(
        self, raw_patient_payload: Dict[str, Any], trial_id: str
    ) -> Dict[str, Any]:
        """
        Entry point for a single patient-trial evaluation.

        raw_patient_payload: dict with raw patient data (id, note, etc.)
        trial_id: trial identifier (e.g. NCT number)

        Returns:
            Dict with eligibility verdict, detailed reasoning, and traces.
        """
        # 1. Preprocess raw note
        preprocessed_data = self.preprocessor.preprocess(raw_patient_payload)

        # 2. Extract or load patient record
        patient = self._get_or_create_patient(preprocessed_data)

        # 3. Load trial protocol + extract atomic criteria
        trial_protocol = self._load_trial_protocol(trial_id)
        atomic_criteria = self.trial_extractor.extract_atomic_criteria(
            trial_protocol
        )

        # 4. Extract atomic patient features
        atomic_features = self.patient_extractor.extract_atomic_features(
            patient
        )

        # 5. Evaluate eligibility and collect explanation
        result = self.scoring_engine.evaluate(
            patient=patient,
            atomic_features=atomic_features,
            trial=trial_protocol,
            atomic_criteria=atomic_criteria,
        )

        # 6. Persist results (with explainability trace)
        self._persist_result(result)

        return result

    def run_batch(
        self, raw_patient_payloads: List[Dict[str, Any]], trial_id: str
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline for multiple patients against the same trial.
        """
        results: List[Dict[str, Any]] = []
        for payload in raw_patient_payloads:
            results.append(self.run_single(payload, trial_id))
        return results

    def _get_or_create_patient(
        self, preprocessed_data: Dict[str, Any]
    ) -> PatientRecord:
        """
        Lookup patient in DB or create a new record from preprocessed data.
        """
        # TODO: implement DB retrieval/creation
        raise NotImplementedError

    def _load_trial_protocol(self, trial_id: str) -> TrialProtocol:
        """
        Load trial protocol text + metadata from DB/knowledge base.
        """
        # TODO: implement DB / knowledge_base integration
        raise NotImplementedError

    def _persist_result(self, result: Dict[str, Any]) -> None:
        """
        Persist eligibility result, including full explanation trace.
        """
        # TODO: implement persistence
        raise NotImplementedError