# /home/kirill/projects_2/folium/Xplore/ui_api/services/patient_feature_service.py

from functools import lru_cache
import logging
import sys
import os

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from nlp.llm.clients.ollama_client import OllamaLLMClient
from nlp.llm.clients.openai_client import OpenAILLMClient
from nlp.llm.ensemble import EnsembleTaskRunner
from nlp.extraction.patient_features import PatientFeatureExtractor
from nlp.extraction.patient_feature_prompts import (
    build_per_model_aggregation_prompt,
    build_final_aggregation_prompt,
)
from config import LLMConfig

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_patient_feature_extractor() -> PatientFeatureExtractor:
    """
    Factory for PatientFeatureExtractor with mixed model setup:
    - 2 local Ollama models (GPU-accelerated)
    - 1 OpenAI model (for comparison/diversity)
    - Same judge model (can be changed)
    
    This setup provides:
    1. Speed from local GPU models
    2. Quality/diversity from OpenAI
    3. Redundancy if one fails
    """
    
    logger.info("Initializing PatientFeatureExtractor with mixed model ensemble")
    
    # Validate configuration
    config_issues = LLMConfig.validate()
    for issue in config_issues:
        logger.warning("Configuration issue: %s", issue)
    
    base_models = []
    
    # Add 2 Ollama models (same model, different instances for parallel execution)
    # try:
    #     ollama_model_1 = OllamaLLMClient(
    #         model_name=LLMConfig.OLLAMA_MODEL,
    #         timeout=LLMConfig.OLLAMA_TIMEOUT,
    #         gpu_layers=LLMConfig.OLLAMA_GPU_LAYERS,
    #     )
    #     base_models.append(ollama_model_1)
    #     logger.info("Added Ollama model #1: %s", LLMConfig.OLLAMA_MODEL)
        
    # except Exception as e:
    #     logger.error("Failed to initialize Ollama models: %s", e)
    #     logger.warning("Continuing without Ollama models")
    
    # Add OpenAI model if API key is available
    if LLMConfig.OPENAI_API_KEY:
        try:
            openai_model = OpenAILLMClient(
                model_name=LLMConfig.API_MODEL_4,
                timeout=LLMConfig.OPENAI_TIMEOUT,
            )
            base_models.append(openai_model)
            logger.info("Added OpenAI model: %s", LLMConfig.API_MODEL_4)

            nvidia_model = OpenAILLMClient(
                model_name=LLMConfig.API_MODEL_4,
                timeout=LLMConfig.OPENAI_TIMEOUT,
            )
            base_models.append(nvidia_model)
            logger.info("Added OpenAI model: %s", LLMConfig.API_MODEL_4)
            

            # nvidia_model = OpenAILLMClient(
            #     model_name=LLMConfig.API_MODEL_3,
            #     timeout=LLMConfig.OPENAI_TIMEOUT,
            # )
            # base_models.append(nvidia_model)
            # logger.info("Added OpenAI model: %s", LLMConfig.API_MODEL_3)
            
        except Exception as e:
            logger.error("Failed to initialize OpenAI model: %s", e)
            logger.warning("Continuing without OpenAI model")
    else:
        logger.warning(
            "OPENAI_API_KEY not set. OpenAI model will not be used. "
            "Set it in environment or .env file."
        )
    
    # Ensure we have at least one model
    if not base_models:
        raise RuntimeError(
            "No base models available! Check Ollama server and OpenAI API key."
        )
    
    logger.info("Total base models initialized: %d", len(base_models))
    
    # Judge model (use first available model, or can use a dedicated stronger model)
    # For production, consider using a more powerful judge like GPT-4
    try:
        judge_model = OpenAILLMClient(
            model_name=LLMConfig.API_MODEL_4,   
            timeout=LLMConfig.OPENAI_TIMEOUT,
        )
    except Exception as e:
        logger.error("Failed to initialize judge model, falling back to first base: %s", e)
        judge_model = base_models[0]
    
    # Create ensemble runner
    ensemble_runner = EnsembleTaskRunner(
        base_models=base_models,
        judge_model=judge_model,
        n_repeats=LLMConfig.ENSEMBLE_N_REPEATS,
        per_model_aggregation_prompt_builder=build_per_model_aggregation_prompt,
        final_aggregation_prompt_builder=build_final_aggregation_prompt,
    )
    
    logger.info(
        "EnsembleTaskRunner created: n_repeats=%d, total calls per extraction=%d",
        LLMConfig.ENSEMBLE_N_REPEATS,
        len(base_models) * LLMConfig.ENSEMBLE_N_REPEATS + len(base_models) + 1,
    )
    
    # Create extractor
    extractor = PatientFeatureExtractor(ensemble_runner=ensemble_runner)
    
    logger.info("PatientFeatureExtractor initialized successfully")
    
    return extractor