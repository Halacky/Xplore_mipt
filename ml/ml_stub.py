from typing import Any, Dict

class MLEligibilityPredictorStub:
    def predict(self, patient_id: str, trial_id: str, features_json: Dict[str, Any], criteria_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "verdict": "unknown",
            "probability_included": None,
            "commentary": "ML predictor is a stub. Train and plug a model to enable."
        }