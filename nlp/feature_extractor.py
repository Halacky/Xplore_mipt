# nlp/feature_extractor.py
from typing import List, Dict, Any

from nlp.extraction.patient_features import PatientFeatureExtractor, AtomicPatientFeature
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from storage.models.patient import PatientRecord
from storage.models.trial import TrialProtocol


# Re-export or define higher-level interfaces here if needed
__all__ = [
    "PatientFeatureExtractor",
    "AtomicPatientFeature",
    "TrialCriteriaExtractor",
]