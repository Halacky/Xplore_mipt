from functools import lru_cache
import logging

from nlp.llm.ensemble import EnsembleTaskRunner
from nlp.extraction.trial_criteria import TrialCriteriaExtractor
from ui_api.services.patient_feature_service import get_patient_feature_extractor
from nlp.extraction.patient_feature_prompts import (
    build_per_model_aggregation_prompt,
    build_final_aggregation_prompt,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_trial_criteria_extractor() -> TrialCriteriaExtractor:
    patient_extractor = get_patient_feature_extractor()
    ensemble_runner: EnsembleTaskRunner = patient_extractor.ensemble_runner

    logger.info("TrialCriteriaExtractor initialized (reusing EnsembleTaskRunner)")
    return TrialCriteriaExtractor(ensemble_runner=ensemble_runner)