# app/core/config.py - Updated version

from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Concierge"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"  # Change in production
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # LLM Settings
    OPENAI_API_KEY: Optional[str] = None
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    
    # Local model settings
    LOCAL_MODEL_PATH: Optional[str] = None  # Path to local GGUF model
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 30
    
    # Vector store
    KNOWLEDGE_BASE_PATH: str = "./data/knowledge_base.json"

    class Config:
        env_file = ".env"

settings = Settings()