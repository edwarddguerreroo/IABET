import os
from typing import List, Optional

# Para versiones modernas de Pydantic (v2+)
try:
    from pydantic_settings import BaseSettings
    from pydantic import AnyHttpUrl, field_validator
    PYDANTIC_V2 = True
# Para versiones antiguas de Pydantic (v1)
except ImportError:
    from pydantic import BaseSettings, AnyHttpUrl, validator
    PYDANTIC_V2 = False

class Settings(BaseSettings):
    # Configuración básica
    PROJECT_NAME: str = "FastAPI-IA"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "API para servir modelos de IA de predicción deportiva"
    
    # Configuración del entorno
    API_PREFIX: str = "/api"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = ENVIRONMENT == "development"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    ALLOW_ORIGINS: List[str] = ["*"]  # Configuración permisiva para desarrollo
    
    # Configuración de seguridad
    SECRET_KEY: Optional[str] = None
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 días
    
    # Rutas de recursos
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH", None)
    DATA_PATH: Optional[str] = os.getenv("DATA_PATH", "./data")
    
    # Configuración de base de datos (para futura implementación)
    DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL", None)
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    if PYDANTIC_V2:
        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "case_sensitive": True,
            "extra": "ignore",  # Ignorar variables extra del .env
        }
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            extra = "ignore"  # Ignorar variables extra del .env

settings = Settings()
