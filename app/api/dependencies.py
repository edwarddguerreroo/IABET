# app/api/dependencies.py

import logging
from typing import Callable
from app.services.model_service import ModelService

def get_model_service() -> ModelService:
    """
    Dependencia que te devuelve la instancia de tu servicio de IA.
    """
    return ModelService()

def get_logger() -> logging.Logger:
    """
    Dependencia que te devuelve un logger preconfigurado.
    """
    return logging.getLogger("app.api")
