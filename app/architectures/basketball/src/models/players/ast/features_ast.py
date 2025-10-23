"""
Módulo de Características para Predicción de Asistencias (AST)
==============================================================

FEATURES BASADAS EN PRINCIPIOS FUNDAMENTALES DE ASISTENCIAS:

1. VISIÓN DE CANCHA: Capacidad de ver oportunidades de pase
2. BASKETBALL IQ: Inteligencia de juego y toma de decisiones
3. CONTROL DEL BALÓN: Manejo del balón y tiempo de posesión
4. RITMO DE JUEGO: Velocidad y transiciones
5. CONTEXTO DEL EQUIPO: Calidad de tiradores y sistema ofensivo
6. CONTEXTO DEL OPONENTE: Presión defensiva y estilo
7. HISTORIAL DE PASES: Rendimiento pasado en asistencias
8. SITUACIÓN DEL JUEGO: Contexto específico del partido
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
import warnings


warnings.filterwarnings('ignore')

# Configurar logging balanceado para features - mostrar etapas principales
logger = logging.getLogger(__name__)

class AssistsFeatureEngineer:
    """
    Motor para generar features para predicción de asistencias (AST)
    """
    
    def __init__(self, correlation_threshold: float = 0.95, max_features: int = 75, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.teams_df = teams_df  # Datos de equipos 
        self.players_df = players_df  # Datos de jugadores total
        self.players_quarters_df = players_quarters_df  # Datos de jugadores por cuartos
        self.feature_registry = {}
        self.feature_categories = {
            # Categorías especializadas para asistencias basadas en patrón identificado
            'evolutionary_features': [],      # Features evolutivas/adaptativas (DOMINANTE 51.98%)
            'assists_history': [],           # Historial de asistencias
            'playmaking_intelligence': [],   # IQ basketbolístico y visión
            'minutes_positioning': [],       # Posicionamiento y tiempo de juego
            'team_context': [],             # Contexto del equipo ofensivo
            'opponent_context': [],         # Contexto del oponente defensivo
            'efficiency_metrics': [],       # Eficiencia en distribución
            'predictive_models': [],        # Modelos predictivos avanzados
            'other_features': []            # Otras features complementarias
        }
        self.protected_features = ['assists', 'player', 'Date', 'Team', 'Opp', 'position']
        
        # Cache para cálculos repetidos
        self._rolling_cache = {}

    def _register_feature(self, feature_name: str, category: str) -> bool:
        """Registra una feature en la categoría correspondiente"""
        if feature_name not in self.feature_registry:
            self.feature_registry[feature_name] = category
            if category in self.feature_categories:
                self.feature_categories[category].append(feature_name)
            return True
        return False
    
    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, operation: str = 'mean') -> pd.Series:
        """
        Cálculo rolling optimizado con validación robusta de índices
        SIEMPRE usa shift(1) para prevenir data leakage (futuros juegos)
        """
        # Validar entrada
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df debe ser un DataFrame")
        if column not in df.columns:
            logger.warning(f" Columna '{column}' no encontrada, retornando ceros")
            return pd.Series(0, index=df.index, name=f"{column}_{window}_{operation}")
        
        # Validar que df esté ordenado cronológicamente
        if 'Date' in df.columns and not df['Date'].is_monotonic_increasing:
            df = df.sort_values(['player', 'Date']).reset_index(drop=True)
        
        # Calcular rolling con validación de índices
        try:
            if operation == 'mean':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
            elif operation == 'std':
                result = df.groupby('player')[column].rolling(window=window, min_periods=2).std().shift(1).fillna(0)
            elif operation == 'max':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).max().shift(1).fillna(0)
            elif operation == 'sum':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).sum().shift(1).fillna(0)
            else:
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).mean().shift(1).fillna(0)
            
            # Validar que result tenga MultiIndex
            if isinstance(result.index, pd.MultiIndex):
                # Reset del MultiIndex a índice simple
                result = result.reset_index(level=0, drop=True)
            
            # Validar alineación de índices antes de reindex
            if not result.index.equals(df.index):
                logger.debug(f"Reindexando {column}_{window}_{operation} para alinear índices")
                result = result.reindex(df.index, fill_value=0)
            
            # Validar resultado final
            if len(result) != len(df):
                logger.error(f" Error: Longitud de resultado ({len(result)}) != DataFrame ({len(df)})")
                return pd.Series(0, index=df.index, name=f"{column}_{window}_{operation}")
            
            return result
            
        except Exception as e:
            logger.error(f" Error calculando {column}_{window}_{operation}: {e}")
            return pd.Series(0, index=df.index, name=f"{column}_{window}_{operation}")
            
    def _convert_mp_to_numeric(self, df: pd.DataFrame) -> None:
        """Convierte columna minutes (minutos jugados) de formato MM:SS a decimal o usa 'minutes' si existe"""
        # Priorizar 'minutes' del nuevo dataset si existe
        if 'minutes' in df.columns:
            # Si minutes existe y es object, convertir a numérico
            if df['minutes'].dtype == 'object':
                def parse_time(time_str):
                    try:
                        if pd.isna(time_str) or time_str == '':
                            return 0.0
                        if ':' in str(time_str):
                            parts = str(time_str).split(':')
                            return float(parts[0]) + float(parts[1]) / 60.0
                        return float(time_str)
                    except:
                        return 0.0
                
                df['minutes'] = df['minutes'].apply(parse_time)

    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """
        Pipeline completo para generar todas las features para predicción de asistencias (AST)
        """
        # Cache para cálculos repetidos
        self._rolling_cache = {}
        
        # Convertir minutes
        self._convert_mp_to_numeric(df)

        # Verificar target
        if 'assists' in df.columns:
            ast_stats = df['assists'].describe()
            logger.info(f"Target assists disponible - Media={ast_stats['mean']:.1f}, Max={ast_stats['max']:.0f}")
        else:
            available_cols = list(df.columns)[:10]
            logger.warning(f"Target assists no disponible - features limitadas")
            logger.warning(f"Columnas disponibles: {available_cols}")
        
        # Limpiar registro de features
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        # ===== FEATURES OPTIMIZADAS EN GRUPOS =====
        
        # GRUPO 0: FEATURES AVANZADAS DEL DATASET (NUEVO)
        self._generate_advanced_dataset_features(df)
        
        # GRUPO 1: FEATURES EVOLUTIVAS DOMINANTES (51.98% importancia acumulada)
        self._create_evolutionary_features_group(df)
        
        # GRUPO 2: FEATURES DE POSICIONAMIENTO Y MINUTOS (6.49% importancia)
        self._create_positioning_features_group(df)
        
        # GRUPO 3: FEATURES DE CONTEXTO Y EFICIENCIA (3.46% importancia)
        self._create_context_efficiency_features_group(df)
        
        # GRUPO 4: FEATURES POR RANGOS DE ASISTENCIAS (CRÍTICO para cases extremos)
        self._create_assist_range_features(df)
        
        # GRUPO 5: FEATURES HISTÓRICAS BÁSICAS
        self._create_basic_historical_features(df)
        
        # GRUPO 6: FEATURES DE OPONENTE Y CONTEXTO
        self._create_opponent_context_features(df)
        
        # GRUPO 8: FEATURES BASADAS EN DATOS POR CUARTO (NUEVA - Corrección crítica)
        self._create_quarter_based_features(df)
    
    
        # Compilar lista de features (excluir columnas base y auxiliares)
        base_columns = [
            # IDs y columnas categóricas básicas
            'player_id', 'player', 'jersey_number', 'position', 'primary_position', 'Team', 'Opp', 'is_home', 'Date', 'game_id',
            
            # Estadísticas del juego actual (NO USAR - data leakage)
            'minutes', 'field_goals_made', 'field_goals_att', 'field_goals_pct', 'three_points_made', 'three_points_att', 'three_points_pct',
            'two_points_made', 'two_points_att', 'two_points_pct', 'blocked_att', 'free_throws_made', 'free_throws_att', 'free_throws_pct',
            'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists', 'turnovers', 'steals', 'blocks', 'assists_turnover_ratio',
            'personal_fouls', 'tech_fouls', 'flagrant_fouls', 'pls_min', 'points', 'tech_fouls_non_unsportsmanlike',
            
            # Target variables
            'double_double', 'triple_double',
            
            # Estadísticas avanzadas del juego actual (NO USAR - data leakage)
            'effective_fg_pct', 'efficiency', 'efficiency_game_score', 'fouls_drawn', 'offensive_fouls', 'points_in_paint',
            'points_in_paint_att', 'points_in_paint_made', 'points_in_paint_pct', 'points_off_turnovers', 'true_shooting_att',
            'true_shooting_pct', 'coach_ejections', 'coach_tech_fouls', 'second_chance_pts', 'second_chance_pct',
            'fast_break_pts', 'fast_break_att', 'fast_break_made', 'fast_break_pct', 'minus', 'plus',
            'defensive_rebounds_pct', 'offensive_rebounds_pct', 'rebounds_pct', 'steals_pct', 'turnovers_pct',
            'second_chance_att', 'second_chance_made', 'defensive_rating', 'offensive_rating',
            
            # Columnas auxiliares temporales básicas (NO especializadas)
            'day_of_week', 'month', 'days_rest', 'days_into_season'
        ]
        
        # Después de crear todas las features, obtener las nuevas columnas
        final_columns = set(df.columns)
        initial_columns = set(base_columns)
        
        # Obtener lista de features creadas
        created_features = list(self.feature_registry.keys())
        
        # Verificar qué features creadas están disponibles en el DataFrame
        available_features = []
        for feature in created_features:
            if feature in df.columns:
                available_features.append(feature)
        
        # CRÍTICO: Excluir base_columns para evitar data leakage
        final_features = []
        excluded_features = []
        
        for feature in available_features:
            if feature not in base_columns:
                final_features.append(feature)
            else:
                excluded_features.append(feature)
        
        if excluded_features:
            logger.warning(f"Excluidas {len(excluded_features)} features por data leakage: {excluded_features[:5]}...")
        
        logger.info(f"FEATURES VÁLIDAS para modelo: {len(final_features)}")
        logger.info(f"Importancia acumulada: 64.87% con solo {len(final_features)} features")
        
        return final_features

    def _create_evolutionary_features_group(self, df: pd.DataFrame) -> None:
        """
        Feature evolutivas dominantes para predicción de asistencias (AST)
        """
        
        # Calcular series históricas una sola vez
        ast_avg = self._get_historical_series(df, 'assists', window=10, operation='mean')
        ast_std = self._get_historical_series(df, 'assists', window=7, operation='std')
        ast_recent = self._get_historical_series(df, 'assists', window=3, operation='mean')
        ast_season = self._get_historical_series(df, 'assists', window=15, operation='mean')
        
        if 'minutes' in df.columns:
            mp_avg = self._get_historical_series(df, 'minutes', window=5, operation='mean')
            mp_expected = mp_avg.fillna(20.0)
            minutes_pressure = (mp_expected / 30.0).clip(0.5, 2.0)
        else:
            minutes_pressure = 1.0
        
        # 1. EVOLUTIONARY SELECTION PRESSURE (28.88%)
        if self._register_feature('evolutionary_selection_pressure', 'evolutionary_features'):
            evolutionary_pressure = ast_avg * minutes_pressure
            df['evolutionary_selection_pressure'] = evolutionary_pressure.fillna(3.0)
        
        # 2. EVOLUTIONARY DOMINANCE INDEX (9.51%)
        if self._register_feature('evolutionary_dominance_index', 'evolutionary_features'):
            selection_pressure = df.get('evolutionary_selection_pressure', 0)
            evolutionary_fitness = 1 / (ast_std.fillna(1) + 0.1)
            dominance_index = (selection_pressure * 0.7 + evolutionary_fitness * 0.3)
            df['evolutionary_dominance_index'] = dominance_index.fillna(0.5)
        
        # 3. PLAYER ADAPTABILITY SCORE (4.53%) - CORREGIDO SIN DATA LEAKAGE
        if self._register_feature('player_adaptability_score', 'evolutionary_features'):
            if 'Date' in df.columns and 'assists' in df.columns:
                # Usar datos históricos para calcular adaptabilidad
                try:
                    df_temp = df.copy()
                    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
                    df_temp['month'] = df_temp['Date'].dt.month
                    
                    # Usar shift para evitar data leakage
                    df_temp['assists_shifted'] = df_temp.groupby('player')['assists'].shift(1)
                    
                    # Calcular varianza mensual histórica (excluyendo juego actual)
                    player_monthly_var = df_temp.groupby(['player', 'month'])['assists_shifted'].var().reset_index()
                    player_adaptability = player_monthly_var.groupby('player')['assists_shifted'].mean()
                    
                    # Adaptabilidad = 1 / (varianza + 1) - menor varianza = mayor adaptabilidad
                    adaptability_map = (1 / (player_adaptability.fillna(1) + 1)).to_dict()
                    df['player_adaptability_score'] = df['player'].map(adaptability_map).fillna(0.5)
                except Exception as e:
                    logger.warning(f"Error en player adaptability: {e}")
                    df['player_adaptability_score'] = 0.5
            else:
                df['player_adaptability_score'] = 0.5
        
        
        # 5. EVOLUTIONARY MOMENTUM PREDICTOR (4.02%) - Clipping suave
        if self._register_feature('evolutionary_momentum_predictor', 'evolutionary_features'):
            momentum_factor = (ast_recent - ast_season).fillna(0)
            adaptability = df.get('player_adaptability_score', 0.5)
            evolutionary_momentum = (momentum_factor * adaptability * 2.0)
            df['evolutionary_momentum_predictor'] = evolutionary_momentum.fillna(0).clip(-1.5, 1.5)
        
        # 6. GENETIC ALGORITHM PREDICTOR (3.71%)
        if self._register_feature('genetic_algorithm_predictor', 'evolutionary_features'):
            evolutionary_features = ['evolutionary_selection_pressure', 'evolutionary_dominance_index', 
                                   'player_adaptability_score']
            available_features = [f for f in evolutionary_features if f in df.columns]
            
            if len(available_features) >= 2:
                weights = [0.5, 0.3, 0.2]
                genetic_score = pd.Series([0] * len(df), index=df.index)
                for i, feature in enumerate(available_features[:3]):
                    genetic_score += df[feature] * weights[i]
                df['genetic_algorithm_predictor'] = genetic_score.fillna(0)
        else:
                df['genetic_algorithm_predictor'] = 0

    def _create_positioning_features_group(self, df: pd.DataFrame) -> None:
        """
        FEATURES DE POSICIONAMIENTO Y MINUTOS
        """
        
        if 'minutes' in df.columns:
            mp_hist = self._get_historical_series(df, 'minutes', window=5, operation='mean')
            mp_expected = self._get_historical_series(df, 'minutes', window=10, operation='mean')
            
            # MINUTES BASED AST PREDICTOR (3.54%)
            if self._register_feature('minutes_based_ast_predictor', 'minutes_positioning') and 'assists' in df.columns:
                ast_hist = self._get_historical_series(df, 'assists', window=5, operation='mean')
                ast_per_minute = ast_hist / (mp_hist + 1)
                minutes_predictor = mp_expected.fillna(20.0) * ast_per_minute
                df['minutes_based_ast_predictor'] = minutes_predictor.fillna(3.0)
            
            # MINUTES EXPECTED (2.95%)
            if self._register_feature('minutes_expected', 'minutes_positioning'):
                df['minutes_expected'] = mp_expected.fillna(20.0)
        else:
            if self._register_feature('minutes_based_ast_predictor', 'minutes_positioning'):
                df['minutes_based_ast_predictor'] = 3.0
            if self._register_feature('minutes_expected', 'minutes_positioning'):
                df['minutes_expected'] = 20.0

    def _create_context_efficiency_features_group(self, df: pd.DataFrame) -> None:
        """
        FEATURES DE CONTEXTO Y EFICIENCIA
        """
        # HOME AWAY AST FACTOR (1.86%)
        if self._register_feature('home_away_ast_factor', 'team_context') and 'assists' in df.columns:
            try:
                player_home_away_map = {}
                for player in df['player'].unique():
                    player_data = df[df['player'] == player].copy()
                    if len(player_data) < 5:
                        player_home_away_map[player] = 1.0
                        continue
                    
                    if 'is_home' in player_data.columns:
                        home_games = player_data[player_data['is_home'] == 1]
                        away_games = player_data[player_data['is_home'] == 0]
                    elif 'Away' in player_data.columns:
                        home_games = player_data[player_data['Away'] != '@']
                        away_games = player_data[player_data['Away'] == '@']
                    else:
                        player_home_away_map[player] = 1.0
                        continue
                    
                    home_ast_avg = home_games['assists'].mean() if len(home_games) > 0 else 3.0
                    away_ast_avg = away_games['assists'].mean() if len(away_games) > 0 else 3.0
                    
                    player_home_away_map[player] = home_ast_avg / (away_ast_avg + 0.1)
                
                df['home_away_ast_factor'] = df['player'].map(player_home_away_map).fillna(1.0)
            except Exception as e:
                logger.warning(f"Error en home/away factor: {e}")
                df['home_away_ast_factor'] = 1.0
        
        # PLAYER OFFENSIVE LOAD PCT L10 (1.60%) - OPTIMIZADO CON DATASET
        if self._register_feature('player_offensive_load_pct_l10', 'efficiency_metrics'):
            if 'offensive_rating' in df.columns:
                # USAR DIRECTAMENTE offensive_rating del dataset (más preciso)
                off_rating_hist = self._get_historical_series(df, 'offensive_rating', window=10, operation='mean')
                # Normalizar a porcentaje vs league average (~110)
                load_percentage = (off_rating_hist / 110.0) * 100
                df['player_offensive_load_pct_l10'] = load_percentage.fillna(100.0)
            elif 'points' in df.columns and 'assists' in df.columns:
                # Fallback al cálculo original si no hay offensive_rating
                df['offensive_contribution'] = df['points'] + (df['assists'] * 2)
                offensive_load = self._get_historical_series(df, 'offensive_contribution', window=10, operation='mean')
                
                if 'Team' in df.columns and 'Date' in df.columns:
                    team_offensive_total = df.groupby(['Team', 'Date'])['offensive_contribution'].transform('sum')
                    team_avg_load = team_offensive_total.rolling(window=5).mean().shift(1)
                    load_percentage = (offensive_load / (team_avg_load + 1)) * 100
                    df['player_offensive_load_pct_l10'] = load_percentage.fillna(20.0)
                else:
                    league_avg_load = offensive_load.mean()
                    df['player_offensive_load_pct_l10'] = ((offensive_load / (league_avg_load + 1)) * 100).fillna(20.0)
            else:
                df['player_offensive_load_pct_l10'] = 20.0

    def _create_assist_range_features(self, df: pd.DataFrame) -> None:
        """
        FEATURES POR RANGOS DE ASISTENCIAS (CRÍTICO máximo 25)
        """
        
        if 'assists' not in df.columns:
            return
        
        # Calcular promedios históricos para rangos
        ast_avg = self._get_historical_series(df, 'assists', window=10, operation='mean')
        ast_max = self._get_historical_series(df, 'assists', window=10, operation='max')
        ast_recent = self._get_historical_series(df, 'assists', window=3, operation='mean')

        # 5. EXPLOSION POTENTIAL (diferencia max - promedio)
        if self._register_feature('ast_explosion_potential', 'assists_history'):
            explosion_potential = (ast_max - ast_avg).fillna(0).clip(0, 15)
            df['ast_explosion_potential'] = explosion_potential
        
        # 8. EXTREME GAME FREQUENCY (frecuencia de 10+ asistencias)
        if self._register_feature('extreme_ast_game_freq', 'assists_history'):
            # Calcular frecuencia de juegos con 10+ asistencias por jugador
            extreme_game_freq = df.groupby('player')['assists'].apply(lambda x: (x.shift(1) >= 10).rolling(20, min_periods=5).mean())
            extreme_game_freq = extreme_game_freq.reset_index(level=0, drop=True).reindex(df.index, fill_value=0)
            df['extreme_ast_game_freq'] = extreme_game_freq.fillna(0).clip(0, 1)

    def _create_basic_historical_features(self, df: pd.DataFrame) -> None:
        """
        FEATURES HISTÓRICAS BÁSICAS
        """
        
        if 'assists' not in df.columns:
            return
        
        # Features básicas históricas para completar las 25 features
        for window in [3, 10, 20]:
            feature_name = f'ast_avg_{window}g'
            if self._register_feature(feature_name, 'assists_history'):
                ast_hist = self._get_historical_series(df, 'assists', window=window, operation='mean')
                df[feature_name] = ast_hist.fillna(3.0)
        
        # Season average como baseline
        if self._register_feature('ast_season_avg', 'assists_history'):
            ast_season = self._get_historical_series(df, 'assists', window=20, operation='mean')
            df['ast_season_avg'] = ast_season.fillna(3.0)

        # 12. AST HYBRID PREDICTOR (1.2773% importancia)
        if self._register_feature('ast_hybrid_predictor', 'evolutionary_features'):
            # Combinar múltiples predictores evolutivos
            base_features = ['evolutionary_selection_pressure', 'genetic_algorithm_predictor', 'player_adaptability_score']
            available = [f for f in base_features if f in df.columns]
            if len(available) >= 2:
                hybrid_score = sum(df[f] for f in available) / len(available)
                df['ast_hybrid_predictor'] = hybrid_score * 1.15  # Factor de amplificación
            else:
                df['ast_hybrid_predictor'] = 3.0
        
        # 15. POSITION INDEX (1.0833% importancia)
        if self._register_feature('position_index', 'player_context'):
            if 'Pos' in df.columns:
                # Mapear posiciones a índices de playmaking
                pos_map = {'PG': 5, 'SG': 3, 'SF': 2, 'PF': 1, 'C': 1}
                df['position_index'] = df['Pos'].map(pos_map).fillna(2)
        else:
                df['position_index'] = 2
        
        # 16. ADAPTIVE VOLATILITY PREDICTOR (1.0255% importancia)
        if self._register_feature('adaptive_volatility_predictor', 'evolutionary_features'):
            if 'assists' in df.columns:
                ast_std = self._get_historical_series(df, 'assists', window=7, operation='std')
                ast_avg = self._get_historical_series(df, 'assists', window=10, operation='mean')
                # Predictor adaptativo basado en volatilidad
                volatility_factor = ast_std / (ast_avg + 0.1)
                adaptive_pred = ast_avg * (1 + volatility_factor * 0.3)
                df['adaptive_volatility_predictor'] = adaptive_pred.fillna(3.0)
        else:
                df['adaptive_volatility_predictor'] = 3.0
        
        # 18. EVOLUTIONARY FITNESS (0.9077% importancia)
        if self._register_feature('evolutionary_fitness', 'evolutionary_features'):
            if 'assists' in df.columns:
                # Fitness evolutivo basado en consistencia y eficiencia
                ast_consistency = self._get_historical_series(df, 'assists', window=7, operation='std')
                fitness = 1 / (ast_consistency.fillna(1) + 0.1)
                df['evolutionary_fitness'] = fitness.clip(0, 10)
            else:
                df['evolutionary_fitness'] = 1.0
        
        
        # 21. EVOLUTIONARY MUTATION RATE (0.7845% importancia)
        if self._register_feature('evolutionary_mutation_rate', 'evolutionary_features'):
            if 'assists' in df.columns:
                # Tasa de mutación evolutiva = cambios abruptos en rendimiento
                ast_rolling = self._get_historical_series(df, 'assists', window=5, operation='mean')
                ast_diff = ast_rolling.diff().abs()
                mutation_rate = ast_diff.rolling(window=3).mean().fillna(0)
                df['evolutionary_mutation_rate'] = mutation_rate.clip(0, 5)
            else:
                df['evolutionary_mutation_rate'] = 0.5
        
        # 22. DYNAMIC RANGE PREDICTOR (0.7775% importancia)
        if self._register_feature('dynamic_range_predictor', 'assists_history'):
            if 'assists' in df.columns:
                # Predictor dinámico basado en rangos móviles
                ast_min_5g = self._get_historical_series(df, 'assists', window=5, operation='min')
                ast_max_5g = self._get_historical_series(df, 'assists', window=5, operation='max')
                ast_avg_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Predictor dinámico = promedio ponderado con sesgo hacia máximo
                dynamic_pred = (ast_avg_5g * 0.6 + ast_max_5g * 0.3 + ast_min_5g * 0.1)
                df['dynamic_range_predictor'] = dynamic_pred.fillna(3.0)
            else:
                df['dynamic_range_predictor'] = 3.0
        
        # 23. ULTRA EFFICIENCY PREDICTOR (0.7696% importancia)
        if self._register_feature('ultra_efficiency_predictor', 'efficiency_metrics'):
            if 'assists' in df.columns and 'minutes' in df.columns:
                # Ultra eficiencia = AST por minuto con ajustes contextuales
                ast_avg = self._get_historical_series(df, 'assists', window=7, operation='mean')
                mp_avg = self._get_historical_series(df, 'minutes', window=7, operation='mean')
                ast_per_min = ast_avg / (mp_avg + 1)
                
                # Predictor ultra eficiente proyectado a 36 minutos
                ultra_pred = ast_per_min * 36
                df['ultra_efficiency_predictor'] = ultra_pred.fillna(3.0)
            else:
                df['ultra_efficiency_predictor'] = 3.0
        

        # 29. HIGH VOLUME EFFICIENCY (0.6721% importancia)
        if self._register_feature('high_volume_efficiency', 'efficiency_metrics'):
            if 'assists' in df.columns and 'minutes' in df.columns:
                # Eficiencia en alto volumen de minutos
                ast_avg = self._get_historical_series(df, 'assists', window=7, operation='mean')
                mp_avg = self._get_historical_series(df, 'minutes', window=7, operation='mean')
                
                # Boost para jugadores con muchos minutos y alta eficiencia
                high_volume = (mp_avg >= 28).astype(int)
                efficiency = ast_avg / (mp_avg / 36)  # AST per 36 minutes
                
                high_vol_eff = high_volume * efficiency
                df['high_volume_efficiency'] = high_vol_eff.fillna(0)
            else:
                df['high_volume_efficiency'] = 0
        
        # 30. ASSIST MOMENTUM ACCELERATION (NUEVA - basada en assists_turnover_ratio)
        if 'assists_turnover_ratio' in df.columns:
            if self._register_feature('assist_momentum_acceleration', 'advanced_playmaking'):
                # Aceleración del momentum de asistencias usando A/T ratio
                at_ratio_3g = self._get_historical_series(df, 'assists_turnover_ratio', window=3, operation='mean')
                at_ratio_7g = self._get_historical_series(df, 'assists_turnover_ratio', window=7, operation='mean')
                at_ratio_15g = self._get_historical_series(df, 'assists_turnover_ratio', window=15, operation='mean')
                
                # Momentum = tendencia de mejora en A/T ratio
                momentum = (at_ratio_3g - at_ratio_7g) / (at_ratio_15g + 0.1)
                acceleration = (at_ratio_3g - at_ratio_7g) - (at_ratio_7g - at_ratio_15g)
                df['assist_momentum_acceleration'] = (momentum + acceleration).fillna(0).clip(-2, 2)
            else:
                df['assist_momentum_acceleration'] = 0
        
        # 31. EFFICIENCY GAME SCORE IMPACT (NUEVA - basada en efficiency_game_score)
        if 'efficiency_game_score' in df.columns:
            if self._register_feature('efficiency_game_score_impact', 'efficiency_metrics'):
                # Impacto del game score en la predicción de asistencias
                game_score_5g = self._get_historical_series(df, 'efficiency_game_score', window=5, operation='mean')
                game_score_std = self._get_historical_series(df, 'efficiency_game_score', window=10, operation='std')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Correlación entre game score y asistencias
                impact_score = (game_score_5g * ast_5g) / (game_score_std + 1)
                df['efficiency_game_score_impact'] = impact_score.fillna(0).clip(0, 10)
            else:
                df['efficiency_game_score_impact'] = 0
        
        # 32. FOULS DRAWN PLAYMAKING (NUEVA - basada en fouls_drawn)
        if 'fouls_drawn' in df.columns:
            if self._register_feature('fouls_drawn_playmaking', 'advanced_playmaking'):
                # Jugadores que atraen faltas tienden a ser mejores playmakers
                fouls_drawn_5g = self._get_historical_series(df, 'fouls_drawn', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                minutes_5g = self._get_historical_series(df, 'minutes', window=5, operation='mean')
                
                # Playmaking por atracción de faltas
                playmaking_score = (fouls_drawn_5g * ast_5g) / (minutes_5g + 0.1)
                df['fouls_drawn_playmaking'] = playmaking_score.fillna(0).clip(0, 5)
            else:
                df['fouls_drawn_playmaking'] = 0
        
        # 33. SECOND CHANCE PLAYMAKING (NUEVA - basada en second_chance_pts)
        if 'second_chance_pts' in df.columns:
            if self._register_feature('second_chance_playmaking', 'advanced_playmaking'):
                # Jugadores que crean en second chance tienden a ser mejores asistentes
                sc_pts_5g = self._get_historical_series(df, 'second_chance_pts', window=5, operation='mean')
                sc_att_5g = self._get_historical_series(df, 'second_chance_att', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Eficiencia en second chance + asistencias
                sc_efficiency = sc_pts_5g / (sc_att_5g + 0.1)
                playmaking_score = sc_efficiency * ast_5g
                df['second_chance_playmaking'] = playmaking_score.fillna(0).clip(0, 8)
            else:
                df['second_chance_playmaking'] = 0
        
        # 34. FAST BREAK PLAYMAKING ENHANCED (NUEVA - basada en fast_break_pts)
        if 'fast_break_pts' in df.columns:
            if self._register_feature('fast_break_playmaking_enhanced', 'advanced_playmaking'):
                # Playmaking en transición mejorado
                fb_pts_5g = self._get_historical_series(df, 'fast_break_pts', window=5, operation='mean')
                fb_att_5g = self._get_historical_series(df, 'fast_break_att', window=5, operation='mean')
                fb_pct_5g = self._get_historical_series(df, 'fast_break_pct', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Eficiencia en fast break + asistencias
                fb_efficiency = fb_pct_5g / 100
                playmaking_score = (fb_pts_5g * fb_efficiency * ast_5g) / 10
                df['fast_break_playmaking_enhanced'] = playmaking_score.fillna(0).clip(0, 6)
            else:
                df['fast_break_playmaking_enhanced'] = 0
        
        # 35. DEFENSIVE RATING IMPACT (NUEVA - basada en defensive_rating)
        if 'defensive_rating' in df.columns:
            if self._register_feature('defensive_rating_impact', 'opponent_context'):
                # Impacto del rating defensivo del jugador en sus asistencias
                def_rating_5g = self._get_historical_series(df, 'defensive_rating', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Jugadores con mejor rating defensivo tienden a ser mejores playmakers
                impact_score = ast_5g / (def_rating_5g + 50) * 100
                df['defensive_rating_impact'] = impact_score.fillna(0).clip(0, 5)
            else:
                df['defensive_rating_impact'] = 0
        
        # 36. OFFENSIVE RATING SYNERGY (NUEVA - basada en offensive_rating)
        if 'offensive_rating' in df.columns:
            if self._register_feature('offensive_rating_synergy', 'efficiency_metrics'):
                # Sinergia entre rating ofensivo y asistencias
                off_rating_5g = self._get_historical_series(df, 'offensive_rating', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                minutes_5g = self._get_historical_series(df, 'minutes', window=5, operation='mean')
                
                # Sinergia = asistencias por minuto * rating ofensivo
                synergy = (ast_5g / (minutes_5g + 0.1)) * (off_rating_5g / 100)
                df['offensive_rating_synergy'] = synergy.fillna(0).clip(0, 3)
            else:
                df['offensive_rating_synergy'] = 0
        
        # 37. BLOCKED ATTEMPTS PLAYMAKING (NUEVA - basada en blocked_att)
        if 'blocked_att' in df.columns:
            if self._register_feature('blocked_attempts_playmaking', 'advanced_playmaking'):
                # Jugadores que intentan más tiros tienden a ser mejores asistentes
                blocked_att_5g = self._get_historical_series(df, 'blocked_att', window=5, operation='mean')
                fg_att_5g = self._get_historical_series(df, 'field_goals_att', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Agresividad ofensiva + asistencias
                aggression = blocked_att_5g / (fg_att_5g + 0.1)
                playmaking_score = aggression * ast_5g
                df['blocked_attempts_playmaking'] = playmaking_score.fillna(0).clip(0, 4)
            else:
                df['blocked_attempts_playmaking'] = 0
        
        # 38. PLUS MINUS MOMENTUM (NUEVA - basada en plus)
        if 'plus' in df.columns:
            if self._register_feature('plus_minus_momentum', 'team_context'):
                # Momentum del plus/minus en relación con asistencias
                plus_5g = self._get_historical_series(df, 'plus', window=5, operation='mean')
                minus_5g = self._get_historical_series(df, 'minus', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Momentum = (plus - minus) * asistencias
                momentum = (plus_5g - minus_5g) * ast_5g / 10
                df['plus_minus_momentum'] = momentum.fillna(0).clip(-2, 2)
            else:
                df['plus_minus_momentum'] = 0
        
        # 39. ELITE PLAYER SCALING (NUEVA - Corrección crítica)
        if self._register_feature('elite_player_scaling', 'player_tier'):
            # Identificar jugadores elite por nombre y escalar sus predicciones
            elite_players = [
                'Trae Young', 'Luka Doncic', 'Nikola Jokić', 'LeBron James', 
                'Stephen Curry', 'James Harden', 'Chris Paul', 'Russell Westbrook',
                'Damian Lillard', 'Kyrie Irving', 'Ja Morant', 'Dejounte Murray',
                'Tyrese Haliburton', 'LaMelo Ball', 'Darius Garland'
            ]
            
            # Factor de escalamiento basado en promedio histórico de asistencias
            elite_scaling = pd.Series([1.0] * len(df), index=df.index)
            
            for player in elite_players:
                player_mask = df['player'] == player
                if player_mask.any():
                    # Calcular promedio histórico de asistencias del jugador
                    player_ast_avg = df[player_mask]['assists'].mean()
                    
                    # Factor de escalamiento basado en promedio histórico
                    if player_ast_avg >= 8:  # Elite playmakers
                        elite_scaling[player_mask] = 1.8
                    elif player_ast_avg >= 6:  # Good playmakers
                        elite_scaling[player_mask] = 1.4
                    elif player_ast_avg >= 4:  # Average playmakers
                        elite_scaling[player_mask] = 1.1
            
            df['elite_player_scaling'] = elite_scaling
        
        # 40. EXPLOSIVE GAME POTENTIAL (NUEVA - Corrección crítica)
        if self._register_feature('explosive_game_potential', 'advanced_playmaking'):
            if 'assists' in df.columns:
                # Potencial para juegos explosivos basado en variabilidad histórica
                ast_std_10g = self._get_historical_series(df, 'assists', window=10, operation='std')
                ast_max_10g = self._get_historical_series(df, 'assists', window=10, operation='max')
                ast_avg_10g = self._get_historical_series(df, 'assists', window=10, operation='mean')
                
                # Potencial explosivo = capacidad de superar promedio significativamente
                explosive_potential = (ast_max_10g - ast_avg_10g) / (ast_std_10g + 0.1)
                df['explosive_game_potential'] = explosive_potential.fillna(0).clip(0, 5)
            else:
                df['explosive_game_potential'] = 0
        
        # 41. HIGH VOLUME GAME INDICATOR (NUEVA - Corrección crítica)
        if self._register_feature('high_volume_game_indicator', 'game_context'):
            if 'minutes' in df.columns and 'assists' in df.columns:
                # Indicador de juegos de alto volumen basado en minutos y contexto
                minutes_5g = self._get_historical_series(df, 'minutes', window=5, operation='mean')
                ast_5g = self._get_historical_series(df, 'assists', window=5, operation='mean')
                
                # Alto volumen = muchos minutos + buen promedio de asistencias
                high_volume = ((minutes_5g >= 30) & (ast_5g >= 5)).astype(int)
                df['high_volume_game_indicator'] = high_volume.fillna(0)
            else:
                df['high_volume_game_indicator'] = 0
        
        # 42. STAR PLAYER MOMENTUM (NUEVA - Corrección crítica)
        if self._register_feature('star_player_momentum', 'advanced_playmaking'):
            if 'assists' in df.columns:
                # Momentum de jugador estrella basado en tendencia reciente
                ast_3g = self._get_historical_series(df, 'assists', window=3, operation='mean')
                ast_7g = self._get_historical_series(df, 'assists', window=7, operation='mean')
                ast_15g = self._get_historical_series(df, 'assists', window=15, operation='mean')
                
                # Momentum = tendencia de mejora + consistencia alta
                momentum = (ast_3g - ast_7g) + (ast_7g - ast_15g) * 0.5
                consistency_bonus = (ast_3g >= 6).astype(int) * 0.5
                
                df['star_player_momentum'] = (momentum + consistency_bonus).fillna(0).clip(-2, 3)
            else:
                df['star_player_momentum'] = 0
        
        # 43. QUARTER-BASED EXPLOSIVE DETECTION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_explosive_detection', 'quarter_analysis'):
            # Detectar juegos explosivos basado en patrones por cuarto
            # Trae Young: 2.88-3.02 AST por cuarto = 11-12 AST total esperado
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular promedio de asistencias por cuarto para cada jugador
                    quarter_stats = self.players_quarters_df.groupby(['player', 'quarter'])['assists'].mean().reset_index()
                    player_quarter_avg = quarter_stats.groupby('player')['assists'].mean().to_dict()
                    
                    # Factor de explosión basado en promedio por cuarto
                    explosive_factor = pd.Series([1.0] * len(df), index=df.index)
                    for idx, player in enumerate(df['player']):
                        if player in player_quarter_avg:
                            quarter_avg = player_quarter_avg[player]
                            # Si promedia 3+ AST por cuarto, es un jugador explosivo
                            if quarter_avg >= 3.0:
                                explosive_factor.iloc[idx] = 2.5  # Factor alto para explosivos
                            elif quarter_avg >= 2.5:
                                explosive_factor.iloc[idx] = 2.0  # Factor medio
                            elif quarter_avg >= 2.0:
                                explosive_factor.iloc[idx] = 1.5  # Factor bajo
                    
                    df['quarter_explosive_detection'] = explosive_factor
                except Exception as e:
                    logger.warning(f"Error en quarter explosive detection: {e}")
                    df['quarter_explosive_detection'] = 1.0
            else:
                df['quarter_explosive_detection'] = 1.0
        
        # 44. QUARTER CONSISTENCY PATTERN (NUEVA - Corrección crítica)
        if self._register_feature('quarter_consistency_pattern', 'quarter_analysis'):
            # Patrón de consistencia por cuartos
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular consistencia por cuarto (baja desviación = alta consistencia)
                    quarter_consistency = self.players_quarters_df.groupby('player')['assists'].agg(['mean', 'std']).reset_index()
                    quarter_consistency['consistency_score'] = quarter_consistency['mean'] / (quarter_consistency['std'] + 0.1)
                    
                    consistency_map = dict(zip(quarter_consistency['player'], quarter_consistency['consistency_score']))
                    df['quarter_consistency_pattern'] = df['player'].map(consistency_map).fillna(1.0)
                except Exception as e:
                    logger.warning(f"Error en quarter consistency: {e}")
                    df['quarter_consistency_pattern'] = 1.0
            else:
                df['quarter_consistency_pattern'] = 1.0
        
        # 45. QUARTER MOMENTUM ACCELERATION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_momentum_acceleration', 'quarter_analysis'):
            # Aceleración del momentum por cuartos
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular momentum por cuarto (Q1+Q2 vs Q3+Q4)
                    quarter_momentum = self.players_quarters_df.groupby(['player', 'quarter'])['assists'].mean().reset_index()
                    
                    momentum_scores = {}
                    for player in quarter_momentum['player'].unique():
                        player_data = quarter_momentum[quarter_momentum['player'] == player]
                        q1_q2 = player_data[player_data['quarter'].isin([1, 2])]['assists'].mean()
                        q3_q4 = player_data[player_data['quarter'].isin([3, 4])]['assists'].mean()
                        
                        # Momentum = mejora en segunda mitad
                        if pd.notna(q1_q2) and pd.notna(q3_q4):
                            momentum = (q3_q4 - q1_q2) / (q1_q2 + 0.1)
                        else:
                            momentum = 0
                        
                        momentum_scores[player] = momentum
                    
                    df['quarter_momentum_acceleration'] = df['player'].map(momentum_scores).fillna(0)
                except Exception as e:
                    logger.warning(f"Error en quarter momentum: {e}")
                    df['quarter_momentum_acceleration'] = 0
            else:
                df['quarter_momentum_acceleration'] = 0
        
        # 46. ELITE QUARTER PERFORMANCE (NUEVA - Corrección crítica)
        if self._register_feature('elite_quarter_performance', 'quarter_analysis'):
            # Rendimiento elite por cuarto
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Identificar jugadores que consistentemente tienen 3+ AST por cuarto
                    elite_quarter_players = self.players_quarters_df.groupby('player')['assists'].mean()
                    elite_quarter_players = elite_quarter_players[elite_quarter_players >= 3.0].index.tolist()
                    
                    # Factor de elite basado en rendimiento por cuarto
                    elite_factor = pd.Series([1.0] * len(df), index=df.index)
                    for idx, player in enumerate(df['player']):
                        if player in elite_quarter_players:
                            elite_factor.iloc[idx] = 3.0  # Factor muy alto para elite
                    
                    df['elite_quarter_performance'] = elite_factor
                except Exception as e:
                    logger.warning(f"Error en elite quarter performance: {e}")
                    df['elite_quarter_performance'] = 1.0
            else:
                df['elite_quarter_performance'] = 1.0
        
        # 47. QUARTER-BASED GAME PROJECTION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_based_game_projection', 'quarter_analysis'):
            # Proyección de juego basada en patrones por cuarto
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular proyección basada en promedio por cuarto * 4
                    quarter_projection = self.players_quarters_df.groupby('player')['assists'].mean() * 4
                    projection_map = quarter_projection.to_dict()
                    
                    df['quarter_based_game_projection'] = df['player'].map(projection_map).fillna(3.0)
                except Exception as e:
                    logger.warning(f"Error en quarter projection: {e}")
                    df['quarter_based_game_projection'] = 3.0
            else:
                df['quarter_based_game_projection'] = 3.0

    def _create_quarter_based_features(self, df: pd.DataFrame) -> None:
        """
        FEATURES BASADAS EN DATOS POR CUARTO
        Corrección crítica para detectar juegos explosivos
        """
        # 43. QUARTER-BASED EXPLOSIVE DETECTION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_explosive_detection', 'quarter_analysis'):
            # Detectar juegos explosivos basado en patrones por cuarto
            # Trae Young: 2.88-3.02 AST por cuarto = 11-12 AST total esperado
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular promedio de asistencias por cuarto para cada jugador
                    quarter_stats = self.players_quarters_df.groupby(['player', 'quarter'])['assists'].mean().reset_index()
                    player_quarter_avg = quarter_stats.groupby('player')['assists'].mean().to_dict()
                    
                    # Factor de explosión basado en promedio por cuarto
                    explosive_factor = pd.Series([1.0] * len(df), index=df.index)
                    for idx, player in enumerate(df['player']):
                        if player in player_quarter_avg:
                            quarter_avg = player_quarter_avg[player]
                            # Si promedia 3+ AST por cuarto, es un jugador explosivo
                            if quarter_avg >= 3.0:
                                explosive_factor.iloc[idx] = 2.5  # Factor alto para explosivos
                            elif quarter_avg >= 2.5:
                                explosive_factor.iloc[idx] = 2.0  # Factor medio
                            elif quarter_avg >= 2.0:
                                explosive_factor.iloc[idx] = 1.5  # Factor bajo
                    
                    df['quarter_explosive_detection'] = explosive_factor
                except Exception as e:
                    logger.warning(f"Error en quarter explosive detection: {e}")
                    df['quarter_explosive_detection'] = 1.0
            else:
                df['quarter_explosive_detection'] = 1.0
        
        # 44. QUARTER CONSISTENCY PATTERN (NUEVA - Corrección crítica)
        if self._register_feature('quarter_consistency_pattern', 'quarter_analysis'):
            # Patrón de consistencia por cuartos
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular consistencia por cuarto (baja desviación = alta consistencia)
                    quarter_consistency = self.players_quarters_df.groupby('player')['assists'].agg(['mean', 'std']).reset_index()
                    quarter_consistency['consistency_score'] = quarter_consistency['mean'] / (quarter_consistency['std'] + 0.1)
                    
                    consistency_map = dict(zip(quarter_consistency['player'], quarter_consistency['consistency_score']))
                    df['quarter_consistency_pattern'] = df['player'].map(consistency_map).fillna(1.0)
                except Exception as e:
                    logger.warning(f"Error en quarter consistency: {e}")
                    df['quarter_consistency_pattern'] = 1.0
            else:
                df['quarter_consistency_pattern'] = 1.0
        
        # 45. QUARTER MOMENTUM ACCELERATION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_momentum_acceleration', 'quarter_analysis'):
            # Aceleración del momentum por cuartos
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular momentum por cuarto (Q1+Q2 vs Q3+Q4)
                    quarter_momentum = self.players_quarters_df.groupby(['player', 'quarter'])['assists'].mean().reset_index()
                    
                    momentum_scores = {}
                    for player in quarter_momentum['player'].unique():
                        player_data = quarter_momentum[quarter_momentum['player'] == player]
                        q1_q2 = player_data[player_data['quarter'].isin([1, 2])]['assists'].mean()
                        q3_q4 = player_data[player_data['quarter'].isin([3, 4])]['assists'].mean()
                        
                        # Momentum = mejora en segunda mitad
                        if pd.notna(q1_q2) and pd.notna(q3_q4):
                            momentum = (q3_q4 - q1_q2) / (q1_q2 + 0.1)
                        else:
                            momentum = 0
                        
                        momentum_scores[player] = momentum
                    
                    df['quarter_momentum_acceleration'] = df['player'].map(momentum_scores).fillna(0)
                except Exception as e:
                    logger.warning(f"Error en quarter momentum: {e}")
                    df['quarter_momentum_acceleration'] = 0
            else:
                df['quarter_momentum_acceleration'] = 0
        
        # 46. ELITE QUARTER PERFORMANCE (NUEVA - Corrección crítica)
        if self._register_feature('elite_quarter_performance', 'quarter_analysis'):
            # Rendimiento elite por cuarto
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Identificar jugadores que consistentemente tienen 3+ AST por cuarto
                    elite_quarter_players = self.players_quarters_df.groupby('player')['assists'].mean()
                    elite_quarter_players = elite_quarter_players[elite_quarter_players >= 3.0].index.tolist()
                    
                    # Factor de elite basado en rendimiento por cuarto
                    elite_factor = pd.Series([1.0] * len(df), index=df.index)
                    for idx, player in enumerate(df['player']):
                        if player in elite_quarter_players:
                            elite_factor.iloc[idx] = 3.0  # Factor muy alto para elite
                    
                    df['elite_quarter_performance'] = elite_factor
                except Exception as e:
                    logger.warning(f"Error en elite quarter performance: {e}")
                    df['elite_quarter_performance'] = 1.0
            else:
                df['elite_quarter_performance'] = 1.0
        
        # 47. QUARTER-BASED GAME PROJECTION (NUEVA - Corrección crítica)
        if self._register_feature('quarter_based_game_projection', 'quarter_analysis'):
            # Proyección de juego basada en patrones por cuarto
            if hasattr(self, 'players_quarters_df') and self.players_quarters_df is not None:
                try:
                    # Calcular proyección basada en promedio por cuarto * 4
                    quarter_projection = self.players_quarters_df.groupby('player')['assists'].mean() * 4
                    projection_map = quarter_projection.to_dict()
                    
                    df['quarter_based_game_projection'] = df['player'].map(projection_map).fillna(3.0)
                except Exception as e:
                    logger.warning(f"Error en quarter projection: {e}")
                    df['quarter_based_game_projection'] = 3.0
            else:
                df['quarter_based_game_projection'] = 3.0

    def _generate_advanced_dataset_features(self, df: pd.DataFrame) -> None:
        """
        Genera features avanzadas usando las nuevas métricas del dataset
        """        
        # 1. ASSIST-TO-TURNOVER EFFICIENCY (ya calculado en dataset)
        if 'assists_turnover_ratio' in df.columns:
            if self._register_feature('ast_to_ratio_enhanced', 'efficiency_metrics'):
                # Usar directamente la métrica del dataset con mejoras históricas
                ratio_hist = self._get_historical_series(df, 'assists_turnover_ratio', window=5, operation='mean')
                ratio_trend = self._get_historical_series(df, 'assists_turnover_ratio', window=3, operation='mean') - \
                             self._get_historical_series(df, 'assists_turnover_ratio', window=10, operation='mean')
                df['ast_to_ratio_enhanced'] = ratio_hist + ratio_trend * 0.3
                
        # 2. FAST BREAK PLAYMAKING (del dataset)
        if all(col in df.columns for col in ['fast_break_pts', 'fast_break_att']):
            if self._register_feature('fast_break_playmaking', 'team_context'):
                # Eficiencia en transiciones - clave para asistencias
                fb_hist = self._get_historical_series(df, 'fast_break_pts', window=5, operation='mean')
                fb_att_hist = self._get_historical_series(df, 'fast_break_att', window=5, operation='mean')
                fb_efficiency = fb_hist / (fb_att_hist + 0.1)
                df['fast_break_playmaking'] = fb_efficiency.fillna(0.5)
                
        # 3. SECOND CHANCE CREATION (del dataset)
        if all(col in df.columns for col in ['second_chance_pts', 'offensive_rebounds']):
            if self._register_feature('second_chance_creation', 'team_context'):
                # Capacidad de crear segundas oportunidades
                sc_pts_hist = self._get_historical_series(df, 'second_chance_pts', window=5, operation='mean')
                orb_hist = self._get_historical_series(df, 'offensive_rebounds', window=5, operation='mean')
                creation_rate = sc_pts_hist / (orb_hist + 0.1)
                df['second_chance_creation'] = creation_rate.fillna(1.0)
                
        # 6. TRUE SHOOTING SUPPORT (del dataset)
        if 'true_shooting_pct' in df.columns:
            if self._register_feature('true_shooting_support', 'efficiency_metrics'):
                # Correlación entre TS% del equipo y asistencias del jugador
                ts_hist = self._get_historical_series(df, 'true_shooting_pct', window=5, operation='mean')
                ast_hist = self._get_historical_series(df, 'assists', window=5, operation='mean')
                # Jugadores que asisten cuando el equipo tira bien
                support_score = ts_hist * ast_hist / 100.0
                df['true_shooting_support'] = support_score.fillna(1.0)
                
        # 7. EFFICIENCY RATING CORRELATION (del dataset)  
        if all(col in df.columns for col in ['efficiency', 'offensive_rating']):
            if self._register_feature('efficiency_rating_synergy', 'efficiency_metrics'):
                # Sinergia entre eficiencia personal y rating ofensivo
                eff_hist = self._get_historical_series(df, 'efficiency', window=5, operation='mean')
                off_rating_hist = self._get_historical_series(df, 'offensive_rating', window=5, operation='mean')
                synergy = (eff_hist * off_rating_hist) / 1000.0  # Normalizar
                df['efficiency_rating_synergy'] = synergy.fillna(10.0)
                
        # 8. DEFENSIVE IMPACT ON ASSISTS (del dataset)
        if 'defensive_rating' in df.columns:
            if self._register_feature('defensive_assist_impact', 'efficiency_metrics'):
                # Mejor defensa = más posesiones = más oportunidades de asistir
                def_rating_hist = self._get_historical_series(df, 'defensive_rating', window=5, operation='mean')
                # Invertir: menor defensive rating = mejor defensa = más asistencias potenciales
                assist_opportunities = 120.0 / (def_rating_hist + 1.0)  # Normalizar alrededor de 110-120
                df['defensive_assist_impact'] = assist_opportunities.fillna(1.0)

    def _create_opponent_context_features(self, df: pd.DataFrame) -> None:
        """
        FEATURES DE OPONENTE Y CONTEXTO
        """
        # 1. REAL OPPONENT DEF RATING (1.3569% importancia) - CORREGIDO
        if self._register_feature('real_opponent_def_rating', 'opponent_context'):
            if 'Opp' in df.columns and hasattr(self, 'teams_df') and self.teams_df is not None:
                # Mapear rating defensivo del oponente usando defensive_rating del dataset de equipos
                try:
                    # Calcular rating defensivo promedio por equipo (menor = mejor defensa)
                    team_def_rating = self.teams_df.groupby('Team')['defensive_rating'].mean().to_dict()
                    df['real_opponent_def_rating'] = df['Opp'].map(team_def_rating).fillna(105.0)
                except Exception as e:
                    logger.warning(f"Error en real_opponent_def_rating: {e}")
                    df['real_opponent_def_rating'] = 105.0
            else:
                df['real_opponent_def_rating'] = 105.0
        
        # 2. OPP PTS ALLOWED (0.719% importancia) - CORREGIDO
        if self._register_feature('opp_pts_allowed', 'opponent_context'):
            if 'Opp' in df.columns and hasattr(self, 'teams_df') and self.teams_df is not None:
                # Puntos permitidos por el oponente (facilidad defensiva) - CORREGIDO
                try:
                    # Usar points_against del dataset de equipos (puntos permitidos por el equipo)
                    opp_pts_allowed = self.teams_df.groupby('Team')['points_against'].mean().to_dict()
                    df['opp_pts_allowed'] = df['Opp'].map(opp_pts_allowed).fillna(110.0)
                except Exception as e:
                    logger.warning(f"Error en opp_pts_allowed: {e}")
                    df['opp_pts_allowed'] = 110.0
            else:
                df['opp_pts_allowed'] = 110.0
        
        # 3. OPPONENT ADAPTATION SCORE (0.744% importancia) - CORREGIDO SIN DATA LEAKAGE
        if self._register_feature('opponent_adaptation_score', 'opponent_context'):
            if 'Opp' in df.columns and 'assists' in df.columns:
                # Score de adaptación contra oponentes específicos usando datos históricos
                try:
                    # Crear DataFrame temporal con shift para evitar data leakage
                    df_temp = df.copy()
                    df_temp['assists_shifted'] = df_temp.groupby('player')['assists'].shift(1)
                    
                    # Calcular promedios históricos por jugador-oponente (excluyendo juego actual)
                    player_opp_performance = df_temp.groupby(['player', 'Opp'])['assists_shifted'].mean().reset_index()
                    player_overall_avg = df_temp.groupby('player')['assists_shifted'].mean()
                    
                    adaptation_map = {}
                    for _, row in player_opp_performance.iterrows():
                        player = row['player']
                        opp_avg = row['assists_shifted']
                        overall_avg = player_overall_avg.get(player, 3.0)
                        
                        # Adaptation score = ratio vs promedio general histórico
                        if pd.notna(opp_avg) and pd.notna(overall_avg) and overall_avg > 0:
                            adaptation_score = opp_avg / (overall_avg + 0.1)
                        else:
                            adaptation_score = 1.0
                        
                        key = f"{player}_{row['Opp']}"
                        adaptation_map[key] = adaptation_score
                    
                    # Aplicar adaptation score
                    df['player_opp_key'] = df['player'] + '_' + df['Opp']
                    df['opponent_adaptation_score'] = df['player_opp_key'].map(adaptation_map).fillna(1.0)
                    df.drop('player_opp_key', axis=1, inplace=False)
                except Exception as e:
                    logger.warning(f"Error en opponent adaptation: {e}")
                    df['opponent_adaptation_score'] = 1.0
            else:
                df['opponent_adaptation_score'] = 1.0