# /home/kirill/projects_2/folium/Xplore/config.py

import os
from typing import Optional

# LLM Configuration
class LLMConfig:
    """Configuration for LLM clients and ensemble"""
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT: int = int(os.getenv("OLLAMA_TIMEOUT", "600"))  # 10 minutes
    OLLAMA_GPU_LAYERS: int = int(os.getenv("OLLAMA_GPU_LAYERS", "-1"))  # -1 = all layers on GPU
    
    # OpenAI Configuration
    BASE_URL: str = os.getenv("BASE_URL", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    API_MODEL_1: str = os.getenv("API_MODEL_1", "ttt")
    API_MODEL_2: str = os.getenv("API_MODEL_2", "ttt")
    API_MODEL_3: str = os.getenv("API_MODEL_3", "ttt")
    API_MODEL_4: str = os.getenv("API_MODEL_4", "ttt")
    OPENAI_TIMEOUT: int = int(os.getenv("OPENAI_TIMEOUT", "300"))  # 5 minutes
    OPENAI_MAX_RETRIES: int = int(os.getenv("OPENAI_MAX_RETRIES", "3"))

    # JSON validation / repair
    JSON_VALIDATOR_MODEL: str = os.getenv("JSON_VALIDATOR_MODEL", API_MODEL_4 or API_MODEL_2)
    JSON_VALIDATOR_MAX_LEN: int = int(os.getenv("JSON_VALIDATOR_MAX_LEN", "8000"))
    
    # Ensemble Configuration
    ENSEMBLE_N_REPEATS: int = int(os.getenv("ENSEMBLE_N_REPEATS", "3"))
    ENSEMBLE_BASE_TEMPERATURE: float = float(os.getenv("ENSEMBLE_BASE_TEMPERATURE", "0.2"))
    ENSEMBLE_JUDGE_TEMPERATURE: float = float(os.getenv("ENSEMBLE_JUDGE_TEMPERATURE", "0.0"))
    
    # Feature Extraction Configuration
    EXTRACTION_MAX_WORKERS: int = int(os.getenv("EXTRACTION_MAX_WORKERS", "3"))
    
    @classmethod
    def validate(cls):
        """Validate configuration"""
        issues = []
        
        if not cls.OPENAI_API_KEY:
            issues.append("OPENAI_API_KEY is not set. OpenAI model will not be available.")
        
        if cls.OLLAMA_TIMEOUT < 300:
            issues.append(f"OLLAMA_TIMEOUT ({cls.OLLAMA_TIMEOUT}s) is quite low. Consider 600+ seconds.")
        
        return issues


# Database Configuration
class DBConfig:
    """Database configuration"""
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./clinical_trials.db")
    DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"


# Logging Configuration
class LogConfig:
    """Logging configuration"""
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")