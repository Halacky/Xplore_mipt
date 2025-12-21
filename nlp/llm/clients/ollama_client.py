# /home/kirill/projects_2/folium/Xplore/nlp/llm/clients/ollama_client.py

import requests
from typing import Any, Dict, Optional
import logging
import time

from ..base import LLMClient, LLMCallTrace
from config import LLMConfig

logger = logging.getLogger(__name__)


class OllamaLLMClient(LLMClient):
    """Ollama local LLM client with GPU support and extended timeout"""
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = "ollama",
        base_url: str = None,
        timeout: int = None,
        gpu_layers: int = None,
    ):
        model_name = model_name or LLMConfig.OLLAMA_MODEL
        super().__init__(model_name=model_name, provider=provider)
        
        self.base_url = (base_url or LLMConfig.OLLAMA_BASE_URL).rstrip("/")
        self.timeout = timeout or LLMConfig.OLLAMA_TIMEOUT
        self.gpu_layers = gpu_layers if gpu_layers is not None else LLMConfig.OLLAMA_GPU_LAYERS
        
        logger.info(
            "OllamaLLMClient initialized: model=%s, base_url=%s, timeout=%ds, gpu_layers=%d",
            self.model_name,
            self.base_url,
            self.timeout,
            self.gpu_layers,
        )
        
        # Check Ollama server availability
        self._check_server()
        
        # Ensure model is loaded with GPU
        self._ensure_model_loaded()
    
    def _check_server(self) -> bool:
        """Check if Ollama server is running"""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            logger.info("Ollama server is running at %s", self.base_url)
            return True
        except Exception as e:
            logger.warning("Ollama server check failed: %s", e)
            logger.warning(
                "Make sure Ollama is running with: ollama serve"
            )
            return False
    
    def _ensure_model_loaded(self) -> None:
        """Ensure model is loaded into memory with GPU layers"""
        try:
            logger.info("Loading model %s with GPU layers=%d", self.model_name, self.gpu_layers)
            
            # Pull/load model with GPU configuration
            payload = {
                "model": self.model_name,
                "options": {
                    "num_gpu": self.gpu_layers,  # Number of layers to offload to GPU
                }
            }
            
            url = f"{self.base_url}/api/show"
            resp = requests.post(url, json={"name": self.model_name}, timeout=10)
            
            if resp.status_code == 200:
                logger.info("Model %s is ready", self.model_name)
            else:
                logger.warning(
                    "Model %s may not be available. Status: %d. "
                    "Run: ollama pull %s",
                    self.model_name,
                    resp.status_code,
                    self.model_name
                )
                
        except Exception as e:
            logger.warning("Could not verify model status: %s", e)
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCallTrace:
        """Generate response from Ollama with extended timeout and retry logic"""
        
        url = f"{self.base_url}/api/generate"
        
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": temperature,
                "num_gpu": self.gpu_layers,  # Ensure GPU is used
            },
        }
        
        if seed is not None:
            payload["options"]["seed"] = seed
        
        # Add any additional options from kwargs
        if kwargs:
            payload["options"].update(kwargs)
        
        max_retries = 2
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "OllamaLLMClient: sending request to %s with model=%s (attempt %d/%d)",
                    url,
                    self.model_name,
                    attempt + 1,
                    max_retries + 1,
                )
                
                start_time = time.time()
                
                resp = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                    headers={"Content-Type": "application/json"}
                )
                
                elapsed_time = time.time() - start_time
                
                resp.raise_for_status()
                data = resp.json()
                raw_text = data.get("response", "")
                
                logger.debug(
                    "OllamaLLMClient: received response in %.2fs, length=%d chars",
                    elapsed_time,
                    len(raw_text) if raw_text else 0
                )
                
                # Log GPU usage info if available
                if "eval_count" in data:
                    logger.debug(
                        "OllamaLLMClient: eval_count=%d, eval_duration=%.2fs",
                        data.get("eval_count", 0),
                        data.get("eval_duration", 0) / 1e9 if data.get("eval_duration") else 0
                    )
                
                trace = LLMCallTrace(
                    model_name=self.model_name,
                    provider=self.provider,
                    temperature=temperature,
                    seed=seed,
                    prompt=prompt,
                    raw_response=raw_text,
                    parsed_output=raw_text,
                )
                
                return trace
                
            except requests.exceptions.Timeout as e:
                logger.error(
                    "OllamaLLMClient timeout error (attempt %d/%d): %s",
                    attempt + 1,
                    max_retries + 1,
                    str(e)
                )
                
                if attempt < max_retries:
                    logger.info("Retrying after %ds...", retry_delay)
                    time.sleep(retry_delay)
                    continue
                else:
                    error_text = (
                        f"ERROR: Ollama timeout after {self.timeout}s "
                        f"({max_retries + 1} attempts): {str(e)}"
                    )
                    
            except requests.exceptions.ConnectionError as e:
                logger.error("OllamaLLMClient connection error: %s", str(e))
                error_text = (
                    f"ERROR: Cannot connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running with 'ollama serve': {str(e)}"
                )
                
            except requests.exceptions.RequestException as e:
                logger.error("OllamaLLMClient request error: %s", str(e))
                error_text = f"ERROR: Ollama request failed: {str(e)}"
                
            except Exception as e:
                logger.exception("OllamaLLMClient unexpected error: %s", str(e))
                error_text = f"ERROR: Unexpected error: {str(e)}"
            
            # If we got here, there was an error - break retry loop
            break
        
        # Return error trace
        return LLMCallTrace(
            model_name=self.model_name,
            provider=self.provider,
            temperature=temperature,
            seed=seed,
            prompt=prompt,
            raw_response=error_text,
            parsed_output=error_text,
        )