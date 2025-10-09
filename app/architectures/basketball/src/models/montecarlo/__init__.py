"""
Motor de Simulación Monte Carlo para Predicciones NBA
====================================================

Sistema avanzado de simulación Monte Carlo que genera predicciones coherentes
y correlacionadas para múltiples variables NBA en un partido.

Componentes:
- MonteCarloEngine: Motor principal de simulación
- CorrelationMatrix: Matriz de correlaciones entre estadísticas
- GameSimulator: Simulador de partidos completos
- ProbabilityCalculator: Calculador de probabilidades derivadas
"""

from .engine import MonteCarloEngine
from .correlations import CorrelationMatrix
from .simulator import NBAGameSimulator
from .probabilities import ProbabilityCalculator

__all__ = [
    'MonteCarloEngine',
    'CorrelationMatrix', 
    'GameSimulator',
    'ProbabilityCalculator'
] 