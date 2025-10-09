"""
Módulo de predicción de triples (3P) NBA
=======================================

Contiene modelos y features especializados para predicción de triples.
"""

from .model_triples import Stacking3PTModel
from .features_triples import ThreePointsFeatureEngineer

__all__ = [
    'Stacking3PTModel', 
    'ThreePointsFeatureEngineer'
] 