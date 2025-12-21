# nlp/llm/base.py
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class LLMCallTrace:
    """
    Metadata about a single LLM call for reproducibility and auditing.
    """
    model_name: str
    provider: str  # e.g. "openai", "local-llama"
    temperature: float
    seed: Optional[int]
    prompt: str
    raw_response: str
    parsed_output: Any


class LLMClient:
    """
    Abstract base class for LLM clients (API-based or self-hosted).
    """

    def __init__(self, model_name: str, provider: str):
        self.model_name = model_name
        self.provider = provider

    def generate(self, prompt: str, **kwargs) -> LLMCallTrace:
        """
        Run a single LLM generation and return trace.
        """
        raise NotImplementedError


class LLMEnsemble:
    """
    Runs the same task across multiple LLM models and/or multiple runs
    per model, then aggregates with a judge model.
    """

    def __init__(
        self,
        base_models: List[LLMClient],
        judge_model: LLMClient,
        n_repeats: int = 1,
    ):
        self.base_models = base_models
        self.judge_model = judge_model
        self.n_repeats = n_repeats

    def run_task(self, task_prompt: str) -> Dict[str, Any]:
        """
        Execute the same task across all models and repeats, then
        let the judge model aggregate responses.

        Returns:
            {
                "final_decision": ...,
                "individual_runs": [...],
                "judge_trace": ...
            }
        """
        # TODO: implement ensemble logic and judge aggregation
        raise NotImplementedError