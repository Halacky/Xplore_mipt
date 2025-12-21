from functools import lru_cache
from nlp.evaluation.llm_eligibility import LLMBasedEligibilityEvaluator
from ui_api.services.patient_feature_service import get_patient_feature_extractor

@lru_cache(maxsize=1)
def get_llm_eligibility_evaluator() -> LLMBasedEligibilityEvaluator:
    extractor = get_patient_feature_extractor()
    return LLMBasedEligibilityEvaluator(ensemble_runner=extractor.ensemble_runner)