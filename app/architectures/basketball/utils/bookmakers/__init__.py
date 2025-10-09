"""
Módulo de Integración con Casas de Apuestas NBA
=============================================

Este módulo proporciona integración completa con APIs de casas de apuestas
para obtener cuotas, analizar líneas y identificar oportunidades de value betting.

Características principales:
- Integración con Sportradar API para cuotas NBA
- Análisis de value betting 
- Detección de ineficiencias de mercado
- Análisis de arbitraje entre casas
- Gestión óptima de capital (Kelly Criterion)
- Simulación de cuotas para testing
"""

from .sportradar_api import SportradarAPI
from .bookmakers_data_fetcher import BookmakersDataFetcher
from .bookmakers_integration import BookmakersIntegration
from .config.config import BookmakersConfig
from .config.exceptions import (
    BookmakersAPIError,
    SportradarAPIError,
    InsufficientDataError,
    ConfigurationError
)

__version__ = "1.0.0"
__author__ = "NBA Prediction System"

__all__ = [
    # APIs principales
    'SportradarAPI',
    'BookmakersDataFetcher', 
    'BookmakersIntegration',
    
    # Configuración
    'BookmakersConfig',
    
    # Excepciones
    'BookmakersAPIError',
    'SportradarAPIError',
    'InsufficientDataError',
    'ConfigurationError',
    
    # Metadatos
    '__version__',
    '__author__'
]

# Configuración por defecto
DEFAULT_CONFIG = {
    'sportradar': {
        'base_url': 'https://api.sportradar.us/nba/trial/v8/en',
        'timeout': 30,
        'retry_attempts': 3,
        'retry_delay': 1.0
    },
    'betting': {
        'minimum_edge': 0.04,
        'confidence_threshold': 0.96,
        'max_kelly_fraction': 0.25,
        'min_odds': 1.5
    },
    'data': {
        'cache_enabled': True,
        'cache_duration_hours': 1,
        'simulate_when_no_data': False  # DESHABILITADO - Solo datos reales
    }
}