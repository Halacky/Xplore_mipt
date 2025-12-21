# /home/kirill/projects_2/folium/Xplore/nlp/llm/clients/openai_client.py

from typing import Any, Dict, Optional
import logging
import json
from openai import OpenAI
from openai import OpenAIError, APITimeoutError, RateLimitError

from ..base import LLMClient, LLMCallTrace
from config import LLMConfig

logger = logging.getLogger(__name__)


class OpenAILLMClient(LLMClient):
    """OpenAI API client with retry logic and error handling"""
    
    def __init__(
        self,
        model_name: str = None,
        provider: str = "openai",
        api_key: Optional[str] = None,
        timeout: int = None,
        max_retries: int = None,
    ):
        model_name = model_name or LLMConfig.OPENAI_MODEL
        super().__init__(model_name=model_name, provider=provider)
        
        self.api_key = api_key or LLMConfig.OPENAI_API_KEY
        self.base_url = LLMConfig.BASE_URL
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.timeout = timeout or LLMConfig.OPENAI_TIMEOUT
        self.max_retries = max_retries or LLMConfig.OPENAI_MAX_RETRIES
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        
        logger.info(
            "OpenAILLMClient initialized: model=%s, timeout=%ds, max_retries=%d",
            self.model_name,
            self.timeout,
            self.max_retries,
        )
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> LLMCallTrace:
        """Generate response from OpenAI API with retry logic"""
        
        logger.debug(
            "OpenAILLMClient: generating with model=%s, temperature=%.2f, seed=%s",
            self.model_name,
            temperature,
            seed,
        )
        
        try:
            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": "You are a clinical information extraction expert. "
                               "Always respond with valid JSON only, no additional text."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Prepare API call parameters
            call_params = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temperature,
                "response_format": {"type": "json_object"},
            }
            
            # Add seed if provided (supported in newer API versions)
            if seed is not None:
                call_params["seed"] = seed
            
            logger.debug("OpenAILLMClient: sending request to OpenAI API")
            
            # Make API call
            response = self.client.chat.completions.create(**call_params)
            
            # Extract response text
            raw_text = response.choices[0].message.content
            
            logger.debug(
                "OpenAILLMClient: received response, length=%d chars",
                len(raw_text) if raw_text else 0
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
            
        except APITimeoutError as e:
            logger.error("OpenAI API timeout error: %s", str(e))
            error_text = f"ERROR: OpenAI API timeout after {self.timeout}s: {str(e)}"
            
        except RateLimitError as e:
            logger.error("OpenAI API rate limit error: %s", str(e))
            error_text = f"ERROR: OpenAI API rate limit exceeded: {str(e)}"
            
        except OpenAIError as e:
            logger.error("OpenAI API error: %s", str(e))
            error_text = f"ERROR: OpenAI API error: {str(e)}"
            
        except Exception as e:
            logger.exception("Unexpected error in OpenAI client: %s", str(e))
            error_text = f"ERROR: Unexpected error: {str(e)}"
        
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