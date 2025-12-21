# Xplore/ui_api/services/llm_service.py
from functools import lru_cache
from typing import List

from nlp.llm.clients.ollama_client import OllamaLLMClient
from nlp.llm.ensemble import SimpleLLMEnsemble

@lru_cache(maxsize=1)
def get_llm_ensemble() -> SimpleLLMEnsemble:

    base_model = OllamaLLMClient(model_name="llama3.1:8b")
    ensemble = SimpleLLMEnsemble(
        base_models=[base_model],
        judge_model=base_model,
        n_repeats=1,
    )
    return ensemble