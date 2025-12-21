from typing import Any, Dict

class DecisionTreeEligibilityEvaluatorStub:
    def evaluate(self, patient_id: str, trial_id: str, features_json: Dict[str, Any], criteria_json: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "verdict": "unknown",
            "excluded_at_node": None,
            "commentary": "Decision tree evaluator is a stub. Provide decision tree JSON to enable."
        }