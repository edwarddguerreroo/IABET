"""
Configuración Optimizada del Sistema Monte Carlo NBA
===================================================

"""

from typing import Dict, List, Any
import numpy as np


class MonteCarloConfig:
    """Configuración principal del sistema Monte Carlo."""
    
    # Configuración de simulación
    DEFAULT_N_SIMULATIONS = 10000
    MAX_N_SIMULATIONS = 100000
    MIN_N_SIMULATIONS = 100000
    
    # Semilla por defecto
    DEFAULT_RANDOM_STATE = 42
    
    # Configuración de correlaciones
    MIN_GAMES_FOR_CORRELATION = 10
    CORRELATION_THRESHOLD = 0.95
    
    # Límites físicos NBA basados en temporadas 2023-24 y 2024-25
    NBA_LIMITS = {
        'PTS': (0, 50),     # Basado en Wembanyama 50 pts + margen de seguridad
        'TRB': (0, 15),     # Basado en performances reales de Jokic/Sabonis
        'AST': (0, 15),     # Basado en líderes actuales como Haliburton
        'STL': (0, 4),      # Basado en defensores elite actuales
        'BLK': (0, 4),      # Basado en Wembanyama y otros bloqueadores
        'TOV': (0, 9),     # Límite práctico para jugadores high-usage
        'MP': (0, 36),      # Límite práctico (algunos juegan ~42-43 max)
        'FG%': (0.0, 1.0),
        '3P%': (0.0, 1.0),
        'FT%': (0.0, 1.0)
    }
    
    # Correlaciones baseline NBA (basadas en análisis histórico)
    NBA_BASELINE_CORRELATIONS = {
        ('PTS', 'MP'): 0.75,
        ('PTS', 'FG%'): 0.45,
        ('PTS', 'AST'): 0.35,
        ('TRB', 'MP'): 0.65,
        ('TRB', 'BLK'): 0.40,
        ('AST', 'MP'): 0.60,
        ('AST', 'TOV'): 0.55,
        ('STL', 'AST'): 0.30,
        ('PTS', 'TRB'): 0.25,
        ('is_started', 'MP'): 0.80,
        ('is_home', 'PTS'): 0.08,
    }
    
    # Ajustes contextuales
    CONTEXTUAL_ADJUSTMENTS = {
        'home_boost': 1.05,        # 5% boost en casa
        'starter_boost': 1.15,     # 15% boost para titulares
        'pace_baseline': 100.0,    # Pace promedio NBA
        'def_rating_baseline': 110.0  # Rating defensivo promedio
    }
    
    # Contribución típica de la banca (% del total del equipo)
    BENCH_CONTRIBUTION_RATIOS = {
        'PTS': 0.25,  # 25% de puntos
        'TRB': 0.20,  # 20% de rebotes
        'AST': 0.15,  # 15% de asistencias
        'STL': 0.20,  # 20% de robos
        'BLK': 0.20,  # 20% de bloqueos
        'TOV': 0.25   # 25% de pérdidas
    }


class BettingLinesConfig:
    """Configuración de líneas de apuestas típicas basadas en sportsbooks 2024-25."""
    
    # Líneas comunes por estadística (basadas en DraftKings, FanDuel, BetMGM)
    COMMON_LINES = {
        'PTS': [10.5, 15.5, 20.5, 25.5, 30.5, 35.5, 40.5, 45.5],      # Líneas targets reales
        'TRB': [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],     # Targets rebotes
        'AST': [4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5],     # Targets asistencias  
        'total_points': [205.5, 210.5, 215.5, 220.5, 225.5, 230.5, 235.5, 240.5],  # Totales del juego
        'team_points': [90.5, 95.5, 100.5, 105.5, 110.5, 115.5, 120.5, 125.5, 130.5],  # Puntos por equipo
        'triples': [1, 2, 3, 4, 5]                          # Líneas de triples
    }
    
    # Líneas por posición (basadas en targets reales de casas de apuestas 2024-25)
    POSITION_LINES = {
        'Guard': {
            'PTS': [12.5, 16.5, 22.5, 26.5, 29.5],  # SGA/Edwards nivel
            'AST': [4.5, 6.5, 8.5, 10.5],           # Haliburton/Trae Young nivel
            'TRB': [3.5, 4.5, 5.5, 6.5]             # Guards típicos
        },
        'Forward': {
            'PTS': [14.5, 19.5, 24.5, 28.5],        # Tatum/Durant nivel
            'TRB': [6.5, 8.5, 10.5, 12.5],          # Forwards versátiles
            'AST': [3.5, 5.5, 7.5]                  # Facilitadores
        },
        'Center': {
            'PTS': [16.5, 20.5, 26.5],              # Jokic/Embiid nivel
            'TRB': [10.5, 12.5, 14.5, 16.5],        # Reboteadores elite
            'BLK': [0.5, 1.5, 2.5, 3.5]             # Wembanyama/Gobert nivel
        }
    }
    
    # Spreads típicos
    COMMON_SPREADS = [-15.5, -10.5, -7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5, 10.5, 15.5]
    
    # Líneas específicas para jugadores TARGET de temporadas 2023-24 y 2024-25
    TARGET_PLAYER_LINES = {
        # ELITE SCORERS
        'Shai Gilgeous-Alexander': {'PTS': 32.5, 'AST': 6.5, 'TRB': 5.5},
        'Anthony Edwards': {'PTS': 27.5, 'TRB': 5.5, '3P': 4.5},
        'Nikola Jokic': {'PTS': 29.5, 'TRB': 12.5, 'AST': 10.5},
        'Giannis Antetokounmpo': {'PTS': 30.5, 'TRB': 11.5, 'AST': 5.5},
        'Jayson Tatum': {'PTS': 26.5, 'TRB': 8.5, 'AST': 6.5},
        
        # ELITE PLAYMAKERS  
        'Tyrese Haliburton': {'AST': 9.5, 'PTS': 20.5, 'TRB': 4.5},
        'Trae Young': {'AST': 11.5, 'PTS': 24.5, 'TRB': 3.5},
        'Cade Cunningham': {'AST': 9.5, 'PTS': 26.5, 'TRB': 6.5},
        'James Harden': {'AST': 8.5, 'PTS': 22.5, 'TRB': 5.5},
        
        # ELITE REBOUNDERS/BIGS
        'Victor Wembanyama': {'PTS': 24.5, 'TRB': 11.5, 'BLK': 3.5},
        'Domantas Sabonis': {'TRB': 13.5, 'AST': 8.5, 'PTS': 19.5},
        'Alperen Sengun': {'TRB': 9.5, 'AST': 5.5, 'PTS': 21.5},
        'Bam Adebayo': {'TRB': 10.5, 'PTS': 15.5, 'AST': 3.5},
        
        # SPECIALISTS
        'Rudy Gobert': {'TRB': 12.5, 'BLK': 2.5, 'PTS': 14.5},
        'Stephen Curry': {'PTS': 22.5, '3P': 4.5, 'AST': 6.5},
        'Damian Lillard': {'PTS': 24.5, '3P': 3.5, 'AST': 7.5}
    }


class DistributionConfig:
    """Configuración de distribuciones estadísticas."""
    
    # Tipos de distribución por estadística
    STAT_DISTRIBUTIONS = {
        'PTS': 'gamma',
        'TRB': 'gamma', 
        'AST': 'gamma',
        'STL': 'gamma',
        'BLK': 'gamma',
        'TOV': 'gamma',
        'FG%': 'beta',
        '3P%': 'beta',
        'FT%': 'beta',
        'MP': 'truncnorm'
    }
    
    # Parámetros por defecto para distribuciones
    DEFAULT_DISTRIBUTION_PARAMS = {
        'gamma': {'shape': 2.0, 'scale': 1.0},
        'beta': {'alpha': 2.0, 'beta': 2.0},
        'truncnorm': {'a': 0, 'b': 48}
    }


class ValidationConfig:
    """Configuración para validación de datos."""
    
    # Mínimos requeridos
    MIN_PLAYER_GAMES = 5
    MIN_TEAM_GAMES = 10
    MIN_CORRELATION_SAMPLES = 20
    
    # Tolerancias
    CORRELATION_TOLERANCE = 0.01
    PROBABILITY_TOLERANCE = 0.001
    
    # Límites de validación
    MAX_CORRELATION = 0.99
    MIN_CORRELATION = -0.99
    MAX_PROBABILITY = 1.0
    MIN_PROBABILITY = 0.0


class OutputConfig:
    """Configuración de salidas y reportes."""
    
    # Directorios por defecto
    DEFAULT_OUTPUT_DIR = "results/montecarlo"
    DEFAULT_CACHE_DIR = "cache/montecarlo"
    DEFAULT_LOG_DIR = "logs"
    
    # Formatos de archivo
    SUPPORTED_FORMATS = ['json', 'csv', 'xlsx', 'parquet']
    DEFAULT_FORMAT = 'json'
    
    # Configuración de reportes
    REPORT_SECTIONS = [
        'metadata',
        'simulation_summary',
        'detailed_results',
        'statistical_analysis',
        'betting_insights'
    ]
    
    # Límites de exportación
    MAX_CSV_ROWS = 100000
    MAX_PLAYERS_PER_EXPORT = 20


def get_default_config() -> Dict[str, Any]:
    """Obtiene configuración por defecto completa."""
    return {
        'simulation': {
            'n_simulations': MonteCarloConfig.DEFAULT_N_SIMULATIONS,
            'random_state': MonteCarloConfig.DEFAULT_RANDOM_STATE,
            'nba_limits': MonteCarloConfig.NBA_LIMITS,
            'contextual_adjustments': MonteCarloConfig.CONTEXTUAL_ADJUSTMENTS
        },
        'correlations': {
            'min_games': MonteCarloConfig.MIN_GAMES_FOR_CORRELATION,
            'threshold': MonteCarloConfig.CORRELATION_THRESHOLD,
            'baseline': MonteCarloConfig.NBA_BASELINE_CORRELATIONS
        },
        'betting': {
            'common_lines': BettingLinesConfig.COMMON_LINES,
            'position_lines': BettingLinesConfig.POSITION_LINES,
            'spreads': BettingLinesConfig.COMMON_SPREADS
        },
        'distributions': {
            'types': DistributionConfig.STAT_DISTRIBUTIONS,
            'params': DistributionConfig.DEFAULT_DISTRIBUTION_PARAMS
        },
        'validation': {
            'min_games': ValidationConfig.MIN_PLAYER_GAMES,
            'tolerances': {
                'correlation': ValidationConfig.CORRELATION_TOLERANCE,
                'probability': ValidationConfig.PROBABILITY_TOLERANCE
            }
        },
        'output': {
            'directories': {
                'results': OutputConfig.DEFAULT_OUTPUT_DIR,
                'cache': OutputConfig.DEFAULT_CACHE_DIR,
                'logs': OutputConfig.DEFAULT_LOG_DIR
            },
            'formats': OutputConfig.SUPPORTED_FORMATS,
            'limits': {
                'csv_rows': OutputConfig.MAX_CSV_ROWS,
                'players_per_export': OutputConfig.MAX_PLAYERS_PER_EXPORT
            }
        }
    }


def get_position_config(position: str) -> Dict[str, Any]:
    """
    Obtiene configuración específica por posición.
    
    Args:
        position: Posición del jugador ('Guard', 'Forward', 'Center')
        
    Returns:
        Configuración específica de la posición
    """
    base_config = get_default_config()
    
    if position in BettingLinesConfig.POSITION_LINES:
        base_config['betting']['lines'] = BettingLinesConfig.POSITION_LINES[position]
    
    # Ajustes específicos por posición
    position_adjustments = {
        'Guard': {
            'ast_boost': 1.2,
            'reb_penalty': 0.8,
            'pace_sensitivity': 1.1
        },
        'Forward': {
            'versatility_boost': 1.1,
            'pace_sensitivity': 1.0
        },
        'Center': {
            'reb_boost': 1.3,
            'blk_boost': 1.5,
            'ast_penalty': 0.7,
            'pace_sensitivity': 0.9
        }
    }
    
    if position in position_adjustments:
        base_config['position_adjustments'] = position_adjustments[position]
    
    return base_config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Valida una configuración.
    
    Args:
        config: Configuración a validar
        
    Returns:
        True si la configuración es válida
    """
    required_sections = ['simulation', 'correlations', 'betting']
    
    for section in required_sections:
        if section not in config:
            return False
    
    # Validar límites de simulación
    n_sims = config.get('simulation', {}).get('n_simulations', 0)
    if not (MonteCarloConfig.MIN_N_SIMULATIONS <= n_sims <= MonteCarloConfig.MAX_N_SIMULATIONS):
        return False
    
    return True 


def get_target_player_line(player_name: str, stat: str) -> float:
    """
    Obtiene la línea de apuesta específica para un jugador target.
    
    Args:
        player_name: Nombre del jugador
        stat: Estadística (PTS, TRB, AST, etc.)
        
    Returns:
        float: Línea de apuesta para el jugador y estadística específica
    """
    target_lines = BettingLinesConfig.TARGET_PLAYER_LINES
    
    if player_name in target_lines:
        if stat in target_lines[player_name]:
            return target_lines[player_name][stat]
    
    # Fallback a líneas generales por posición si no hay específica
    common_lines = BettingLinesConfig.COMMON_LINES
    if stat in common_lines:
        # Retornar línea media para la estadística
        lines = common_lines[stat]
        return lines[len(lines) // 2] if lines else 0.0
    
    return 0.0


def is_target_player(player_name: str) -> bool:
    """
    Verifica si un jugador está en la lista de targets principales.
    
    Args:
        player_name: Nombre del jugador
        
    Returns:
        bool: True si es un jugador target
    """
    return player_name in BettingLinesConfig.TARGET_PLAYER_LINES


def get_target_players() -> List[str]:
    """
    Obtiene la lista de todos los jugadores target.
    
    Returns:
        List[str]: Lista de nombres de jugadores target
    """
    return list(BettingLinesConfig.TARGET_PLAYER_LINES.keys())


# PARÁMETROS PRINCIPALES DE SIMULACIÓN
SIMULATION_CONFIG = {
    # Número de simulaciones por partido (configurado para óptimo rendimiento)
    'num_simulations': 10000,  # 10K simulaciones para precisión óptima
    
    # Tolerancia para convergencia estadística
    'convergence_tolerance': 0.001,  # 0.1% cambio para convergencia
    'min_simulations': 10000,  # Mínimo absoluto
    'max_simulations': 100000,  # Máximo absoluto
    
    # Control de calidad
    'min_success_rate': 0.95,  # 95% simulaciones exitosas mínimo
    'retry_failed_simulations': True,
    'max_retries': 3
}

# CONFIGURACIÓN DE CORRELACIONES MEJORADA
CORRELATION_CONFIG = {
    # Correlaciones principales (basadas en análisis real del dataset)
    'primary_correlations': {
        ('PTS', 'MP'): 0.85,    # CRÍTICO: Más minutos = más puntos
        ('PTS', 'FGA'): 0.92,   # CRÍTICO: Más intentos = más puntos  
        ('TRB', 'MP'): 0.82,    # CRÍTICO: Más minutos = más rebotes
        ('AST', 'MP'): 0.75,    # CRÍTICO: Más minutos = más asistencias
        ('PTS', 'TS%'): 0.68,   # True shooting impacta puntos
        ('TRB', 'DRB'): 0.96,   # Rebotes defensivos dominan
        ('AST', 'TOV'): 0.58,   # Manejo de balón
    },
    
    # Factores de ajuste contextual
    'context_adjustments': {
        'home_advantage': 0.08,        # 8% boost en casa (NBA real)
        'starter_advantage': 1.15,     # 15% más oportunidades titulares
        'clutch_variance': 0.15,       # 15% variación en momentos clutch
        'injury_impact': 0.25,         # 25% reducción por lesión
        'fatigue_impact': 0.12,        # 12% reducción por fatiga
        'momentum_impact': 0.18,       # 18% boost por momentum
    },
    
    # Umbrales de confianza
    'confidence_thresholds': {
        'min_games_for_profile': 10,   # Mínimo 10 juegos para perfil confiable
        'high_confidence_games': 30,   # 30+ juegos = alta confianza
        'historical_weight': 0.7,      # 70% peso a datos históricos
        'recent_form_weight': 0.3,     # 30% peso a forma reciente
    }
}

# DISTRIBUCIONES ESTADÍSTICAS OPTIMIZADAS
DISTRIBUTION_CONFIG = {
    # Tipos de distribución por estadística
    'stat_distributions': {
        'PTS': 'gamma',      # Gamma para stats siempre positivas
        'TRB': 'gamma',      
        'AST': 'gamma',
        'STL': 'gamma',
        'BLK': 'gamma',
        '3P': 'gamma',
        'TOV': 'poisson',    # Poisson para conteos discretos
        'FG%': 'beta',       # Beta para porcentajes (0-1)
        '3P%': 'beta',
        'FT%': 'beta',
        'MP': 'truncnorm',   # Normal truncada (0-48 minutos)
        '+/-': 'normal',     # Normal para stats que pueden ser negativas
        'BPM': 'normal'
    },
    
    # Parámetros de límites NBA realistas basados en temporadas 2023-24 y 2024-25
    'realistic_limits': {
        'PTS': (0, 52),     # Wembanyama máximo 50 pts en 2024-25 + margen
        'TRB': (0, 21),     # Máximo práctico basado en stats actuales (Jokic ~19 max)
        'AST': (0, 17),     # Máximo práctico basado en líderes actuales (Haliburton ~15 max)
        'STL': (0, 6),      # Máximo práctico basado en defensores elite
        'BLK': (0, 7),      # Máximo práctico basado en Wembanyama y otros
        'TOV': (0, 12),     # Máximo práctico para jugadores con alto uso
        '3P': (0, 13),      # Máximo práctico basado en shooters elite (Edwards 320/82 = ~4 avg, max ~13)
        'MP': (0, 48),      # Límite absoluto NBA
        'FG%': (0.0, 1.0),  # Porcentaje válido
        '3P%': (0.0, 1.0),
        'FT%': (0.0, 1.0)
    },
    
    # Parámetros de suavizado para distribuciones
    'smoothing_params': {
        'min_variance': 0.1,      # Varianza mínima para evitar deltas
        'outlier_threshold': 3.0,  # 3 sigma para outliers
        'robust_estimation': True  # Usar estimación robusta
    }
}

# CONFIGURACIÓN DE CONTEXTO DEL JUEGO
GAME_CONTEXT_CONFIG = {
    # Factores de situación del juego
    'situational_factors': {
        'back_to_back': {
            'minutes_reduction': 0.85,    # 15% menos minutos
            'shooting_penalty': 0.92,     # 8% peor tiro
            'energy_factor': 0.88         # 12% menos energía
        },
        'well_rested': {
            'shooting_bonus': 1.05,       # 5% mejor tiro
            'energy_factor': 1.03         # 3% más energía
        },
        'rivalry_game': {
            'intensity_boost': 1.08,      # 8% más intensidad
            'defensive_focus': 0.96,      # 4% menos puntos
            'turnover_increase': 1.12     # 12% más pérdidas
        },
        'playoffs': {
            'minutes_increase': 1.12,     # 12% más minutos
            'clutch_boost': 1.15,         # 15% mejor clutch
            'defensive_intensity': 0.94   # 6% menos puntos
        },
        'nationally_televised': {
            'performance_boost': 1.03,    # 3% mejor rendimiento
            'pressure_factor': 1.02       # 2% más presión
        }
    },
    
    # Análisis de matchups
    'matchup_analysis': {
        'pace_impact': {
            'fast_pace': 1.15,           # 15% más stats en ritmo rápido
            'slow_pace': 0.88            # 12% menos stats en ritmo lento
        },
        'defensive_rating_impact': {
            'elite_defense': 0.85,       # 15% menos puntos vs elite
            'poor_defense': 1.18         # 18% más puntos vs pobre
        },
        'style_matchups': {
            'shooter_vs_good_defense': 0.88,
            'inside_scorer_vs_small_lineup': 1.12,
            'playmaker_vs_fast_pace': 1.08
        }
    }
}

# CONFIGURACIÓN DE ARQUETIPADO DE JUGADORES
PLAYER_ARCHETYPE_CONFIG = {
    # Umbrales para clasificación de arquetipo
    'archetype_thresholds': {
        'scorer': {
            'min_ppg': 18.0,
            'min_usage': 0.25
        },
        'elite_scorer': {
            'min_ppg': 25.0,
            'min_efficiency': 0.58
        },
        'playmaker': {
            'min_apg': 6.0,
            'min_ast_rate': 0.25
        },
        'rebounder': {
            'min_rpg': 9.0,
            'min_reb_rate': 0.15
        },
        'defender': {
            'min_stocks': 2.0,  # STL + BLK
            'min_def_rating': 110
        },
        'shooter': {
            'min_3pm': 2.0,
            'min_3p_pct': 0.35
        }
    },
    
    # Tendencias por arquetipo
    'archetype_tendencies': {
        'elite_scorer': {
            'clutch_bonus': 1.12,
            'usage_increase': 1.15,
            'consistency': 0.85
        },
        'playmaker': {
            'home_bonus': 1.08,
            'pace_sensitivity': 1.12,
            'turnover_correlation': 1.15
        },
        'shooter': {
            'variance': 1.25,
            'hot_streak_potential': 1.20,
            'cold_streak_risk': 0.75
        },
        'rebounder': {
            'pace_correlation': 1.18,
            'consistency': 1.15,
            'minutes_stability': 1.08
        }
    }
}

# CONFIGURACIÓN DE OPTIMIZACIÓN
OPTIMIZATION_CONFIG = {
    # Parámetros de búsqueda
    'parameter_search': {
        'correlation_range': (0.1, 0.95),    # Rango válido correlaciones
        'adjustment_range': (0.8, 1.25),     # Rango ajustes contextuales
        'learning_rate': 0.01,               # Tasa de aprendizaje
        'momentum': 0.9                      # Momentum para optimización
    },
    
    # Métricas de evaluación
    'evaluation_metrics': {
        'primary_metric': 'win_probability_accuracy',
        'secondary_metrics': [
            'score_prediction_mae',
            'individual_stat_correlation',
            'upset_prediction_rate'
        ],
        'weight_primary': 0.6,
        'weight_secondary': 0.4
    },
    
    # Calibración automática
    'auto_calibration': {
        'enabled': True,
        'calibration_games': 100,        # Últimos 100 juegos para calibrar
        'recalibration_frequency': 50,   # Recalibrar cada 50 predicciones
        'min_accuracy_threshold': 0.65   # Mínimo 65% precisión
    }
}

# CONFIGURACIÓN DE VALIDACIÓN
VALIDATION_CONFIG = {
    # Validación cruzada
    'cross_validation': {
        'k_folds': 10,                   # 10-fold CV
        'test_size': 0.2,                # 20% para testing
        'random_state': 42,              # Semilla reproducible
        'stratify_by_team': True         # Estratificar por equipo
    },
    
    # Métricas de validación
    'validation_metrics': {
        'accuracy_threshold': 0.70,      # 70% precisión mínima
        'calibration_error_max': 0.1,    # Error de calibración <10%
        'prediction_intervals': [0.8, 0.9, 0.95],  # Intervalos de confianza
        'backtesting_window': 365        # Días para backtesting
    }
}

# CONFIGURACIÓN DE LOGGING Y MONITOREO
MONITORING_CONFIG = {
    # Logging detallado
    'detailed_logging': {
        'log_individual_performances': False,  # Muy verboso
        'log_correlation_adjustments': True,
        'log_context_factors': True,
        'log_prediction_confidence': True
    },
    
    # Métricas de rendimiento
    'performance_tracking': {
        'track_simulation_time': True,
        'track_memory_usage': True,
        'track_convergence_rate': True,
        'save_detailed_results': True
    },
    
    # Alertas automáticas
    'alerts': {
        'low_accuracy_threshold': 0.60,    # Alerta si precisión <60%
        'high_error_rate_threshold': 0.1,  # Alerta si >10% errores
        'unusual_prediction_threshold': 3.0 # Alerta si predicción >3 sigma
    }
}

# CONFIGURACIÓN MAESTRA
MASTER_CONFIG = {
    'simulation': SIMULATION_CONFIG,
    'correlations': CORRELATION_CONFIG,
    'distributions': DISTRIBUTION_CONFIG,
    'game_context': GAME_CONTEXT_CONFIG,
    'player_archetypes': PLAYER_ARCHETYPE_CONFIG,
    'optimization': OPTIMIZATION_CONFIG,
    'validation': VALIDATION_CONFIG,
    'monitoring': MONITORING_CONFIG,
    
    # Metadatos
    'version': '2.0.0',
    'last_updated': '2024-12-15',
    'calibrated_on_games': 55901,
    'target_accuracy': 0.85,  # Objetivo 85% precisión
    'confidence_level': 0.95
} 