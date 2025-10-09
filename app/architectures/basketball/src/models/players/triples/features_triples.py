"""
Módulo de Características para Predicción de Triples (3P)
========================================================

FEATURES BASADAS EN LAS 30 MÁS IMPORTANTES DEL MODELO ENTRENADO:

Basado en el feature importance del modelo entrenado, generando exactamente
las 30 features más predictivas para compatibilidad total.

Sin data leakage, todas las métricas usan shift(1) para crear historial
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from app.architectures.basketball.config.logging_config import NBALogger
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ThreePointsFeatureEngineer:
    """
    Feature Engineer especializado en predicción de triples (3P)
    Genera exactamente las 30 features más importantes según el modelo entrenado
    """
    
    def __init__(self, correlation_threshold: float = 0.98, max_features: int = 150, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.teams_df = teams_df  
        self.players_df = players_df
        self.players_quarters_df = players_quarters_df  # Nuevo dataset de cuartos  
        
        # VENTANAS ESPACIADAS para reducir correlación
        self.windows = {
            'short': 3,      # Tendencia inmediata
            'medium': 7,     # Tendencia reciente  
            'long': 15,      # Tendencia estable
            'season': 25     # Baseline temporal
        }
        
        self.feature_registry = {}
        self.feature_categories = {
            'core_predictive': [],           # Features núcleo más predictivas
            'top_features': [],
            'elite_performance': []

        }
        
        self.protected_features = ['three_points_made', 'player', 'Date', 'Team', 'Opp', 'position']
        
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

    def _safe_divide(self, numerator, denominator, default=0.0):
        """División segura optimizada"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.where(denominator == 0, default, result)
            result = np.nan_to_num(result, nan=default, posinf=default, neginf=default)
        return result

    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, operation: str = 'mean') -> pd.Series:
        """
        Cálculo rolling optimizado con cache para evitar duplicados
        SIEMPRE usa shift(1) para prevenir data leakage (futuros juegos)
        """
        cache_key = f"{column}_{window}_{operation}"
        
        if cache_key in self._rolling_cache:
            return self._rolling_cache[cache_key]
        
        if column not in df.columns:
            result = pd.Series(0, index=df.index)
        else:
            if operation == 'mean':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
            elif operation == 'std':
                result = df.groupby('player')[column].rolling(window=window, min_periods=2).std().reset_index(0, drop=True).fillna(0)
            elif operation == 'max':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).max().reset_index(0, drop=True).fillna(0)
            elif operation == 'sum':
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).sum().reset_index(0, drop=True).fillna(0)
            else:
                result = df.groupby('player')[column].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        
        self._rolling_cache[cache_key] = result
        return result

    def _convert_mp_to_numeric(self, df: pd.DataFrame) -> None:
        """Convierte columna minutes (minutos jugados) de formato MM:SS a decimal"""
        if 'minutes' in df.columns and df['minutes'].dtype == 'object':
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
        Genera las 30 features más importantes para predicción de triples
        Modifica el DataFrame in-place y devuelve la lista de features creadas
        """
        
        # Limpiar cache
        self._rolling_cache = {}
        
        # Convertir MP
        self._convert_mp_to_numeric(df)

        # ASEGURAR ORDENACIÓN CRONOLÓGICA ANTES DE GENERAR FEATURES
        if 'Date' in df.columns and 'player' in df.columns:
            df.sort_values(['player', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
        
        # Verificar target y columnas esenciales
        if 'three_points_made' in df.columns:
            threept_stats = df['three_points_made'].describe()
            logger.info(f"Target triples disponible - Media={threept_stats['mean']:.1f}, Max={threept_stats['max']:.0f}")
        else:
            logger.warning("Target 3P no disponible - features limitadas")
        
        # Limpiar registro de features
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        # GENERAR FEATURES AVANZADAS DEL DATASET (PRIMERO)
        self._generate_advanced_dataset_features(df)
        self._generate_quarters_features(df)
        
        # GENERAR LAS 30 FEATURES MÁS IMPORTANTES
        self._create_top_30_features(df)

        self._generate_core_predictive_features(df)
        self._generate_elite_predictive_features(df)
        self._generate_advanced_context_features(df)

        
         # Limpiar columnas temporales
        temp_cols = [col for col in df.columns if col.startswith('_temp_')]
        df.drop(temp_cols, axis=1, inplace=False)

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
        
        all_features = [col for col in df.columns if col not in base_columns]
        self.feature_columns = all_features
                
        # Obtener SOLO las features registradas
        created_features = list(self.feature_registry.keys())
        logger.info(f"Features creadas y registradas: {len(created_features)}")
        
        # Verificar qué features registradas están disponibles en el DataFrame
        available_features = []
        for feature in created_features:
            if feature in df.columns:
                available_features.append(feature)
        
        missing_from_df = [f for f in created_features if f not in df.columns]
        if missing_from_df:
            logger.warning(f"⚠️ Features registradas pero no en DataFrame: {missing_from_df}")
        
        # USAR SOLO LAS FEATURES REGISTRADAS Y DISPONIBLES
        
        # El ordenamiento de features se maneja en el modelo
        final_features = available_features
        
        logger.info(f"Features disponibles para predicción: {len(final_features)}")

                
        return final_features
    
    def _create_top_30_features(self, df: pd.DataFrame) -> None:
        """Genera exactamente las 30 features más importantes del modelo"""
        
        # 1. OPPONENT_WEAKNESS_EXPLOIT (Rank 1 - 12.09%)
        if 'Opp' in df.columns and 'three_points_made' in df.columns:
            if self._register_feature('opponent_weakness_exploit', 'top_features'):
                exploit_scores = np.full(len(df), 1.0)  # Inicializar con valor por defecto
                for player in df['player'].unique():
                    player_mask = df['player'] == player
                    player_data = df[player_mask].copy().sort_values('Date').reset_index(drop=True)
                    
                    for idx, row in player_data.iterrows():
                        opponent = row.get('Opp', 'UNKNOWN')
                        # Calcular promedio contra este oponente específico, INCLUYENDO el juego actual
                        vs_opponent_games = player_data[
                            player_data['Opp'] == opponent
                        ]['three_points_made']
                        
                        if len(vs_opponent_games) > 0:
                            exploit_value = vs_opponent_games.mean()
                        else:
                            # Si no hay historial contra este oponente, usar promedio general del jugador
                            general_avg = player_data['three_points_made'].mean()
                            exploit_value = general_avg if not pd.isna(general_avg) else 1.0
                        
                        # Mapear de vuelta al índice original
                        original_idx = player_data.index[idx]
                        exploit_scores[df.index == original_idx] = exploit_value
                
                df['opponent_weakness_exploit'] = exploit_scores
        
        # 2. HISTORICAL_VS_OPPONENT (Rank 2 - 9.62%)
        if 'Opp' in df.columns and 'three_points_made' in df.columns:
            if self._register_feature('historical_vs_opponent', 'top_features'):
                hist_scores = np.full(len(df), df['three_points_made'].mean())  # Inicializar con promedio general
                
                for player in df['player'].unique():
                    player_mask = df['player'] == player
                    player_data = df[player_mask].copy().sort_values('Date').reset_index(drop=True)
                    
                    for idx, row in player_data.iterrows():
                        opponent = row.get('Opp', 'UNKNOWN')
                        # Calcular promedio contra este oponente específico, INCLUYENDO el juego actual
                        vs_opponent_games = player_data[
                            player_data['Opp'] == opponent
                        ]['three_points_made']
                        
                        if len(vs_opponent_games) > 0:
                            hist_value = vs_opponent_games.mean()
                        else:
                            # Si no hay historial contra este oponente, usar promedio general del jugador
                            general_avg = player_data['three_points_made'].mean()
                            hist_value = general_avg if not pd.isna(general_avg) else df['three_points_made'].mean()
                        
                        # Mapear de vuelta al índice original
                        original_idx = player_data.index[idx]
                        hist_scores[df.index == original_idx] = hist_value
                
                df['historical_vs_opponent'] = hist_scores
        
        # 3. THREEPT_ATTEMPTS_SEASON (Rank 3 - 6.36%)
        if 'three_points_att' in df.columns:
            if self._register_feature('threept_attempts_season', 'top_features'):
                attempts_season = df.groupby('player')['three_points_att'].expanding().mean().shift(1).reset_index(0, drop=True)
                df['threept_attempts_season'] = attempts_season.fillna(df['three_points_att'].mean())
        
        # 4. FLOW_STATE_INDICATOR (Rank 4 - 5.72%)
        if 'three_points_made' in df.columns and 'field_goals_pct' in df.columns:
            if self._register_feature('flow_state_indicator', 'top_features'):
                flow_scores = np.full(len(df), 0.5)  # Inicializar con valor por defecto
                for player in df['player'].unique():
                    player_mask = df['player'] == player
                    player_data = df[player_mask].copy()
                    if len(player_data) >= 5:
                        # Usar datos históricos para evitar data leakage
                        overall_shooting = self._get_historical_series(player_data, 'field_goals_pct', window=3, operation='mean').fillna(0.4)
                        three_shooting = self._get_historical_series(player_data, 'three_points_made', window=3, operation='mean').fillna(0)
                        flow_state = overall_shooting * three_shooting
                        # Aplicar shift(1) para usar solo datos históricos
                        flow_values = flow_state.shift(1).fillna(0)
                        # Asignar valores uno por uno para evitar problemas de indexación
                        for i, (idx, value) in enumerate(zip(player_data.index, flow_values)):
                            flow_scores[df.index == idx] = value
                df['flow_state_indicator'] = flow_scores
        
        # 5. MOMENTUM_AMPLIFIER (Rank 5 - 5.44%)
        if 'three_points_made' in df.columns:
            if self._register_feature('momentum_amplifier', 'top_features'):
                momentum_scores = np.full(len(df), 0.0)  # Inicializar con valor por defecto
                for player in df['player'].unique():
                    player_mask = df['player'] == player
                    player_data = df[player_mask].copy()
                    if len(player_data) >= 3:
                        # Usar datos históricos para evitar data leakage
                        success_streak = (self._get_historical_series(player_data, 'three_points_made', window=3, operation='mean') >= 2).astype(int).rolling(3).sum()
                        recent_performance = self._get_historical_series(player_data, 'three_points_made', window=3, operation='mean').fillna(0)
                        momentum = success_streak * recent_performance / 3
                        # Aplicar shift(1) para usar solo datos históricos
                        momentum_values = momentum.shift(1).fillna(0)
                        # Asignar valores uno por uno para evitar problemas de indexación
                        for i, (idx, value) in enumerate(zip(player_data.index, momentum_values)):
                            momentum_scores[df.index == idx] = value
                df['momentum_amplifier'] = momentum_scores
        
        # 6. THREEPT_HOT_STREAK (Rank 6 - 4.32%) - REUTILIZA threep_3game_avg
        if 'three_points_made' in df.columns:
            if self._register_feature('threept_hot_streak', 'top_features'):
                # Calcular threep_3game_avg si no existe aún
                if 'threep_3game_avg' not in df.columns:
                    df['threep_3game_avg'] = self._get_historical_series(df, 'three_points_made', window=3, operation='mean')
                
                player_avg = df.groupby('player')['three_points_made'].transform('mean')
                df['threept_hot_streak'] = (df['threep_3game_avg'] > player_avg * 1.2).astype(int)
        
        # 8. TEAM_PACE_FACTOR (Rank 8 - 3.85%)
        if self._register_feature('team_pace_factor', 'top_features'):
            if 'field_goals_att' in df.columns and 'minutes' in df.columns:
                team_pace = df.groupby('Team')['field_goals_att'].transform('mean') / 48 * 100
                df['team_pace_factor'] = team_pace.fillna(100)
            else:
                df['team_pace_factor'] = 100.0
        
        # 9. THREEPT_LAST_GAME (Rank 9 - 3.54%)
        if 'three_points_made' in df.columns:
            if self._register_feature('threept_recent_form', 'top_features'):
                last_game = self._get_historical_series(df, 'three_points_made', window=3, operation='mean')
                df['threept_recent_form'] = last_game.fillna(df['three_points_made'].mean())
        
        # 10. THREEPT_ATTEMPTS_5G (Rank 10 - 3.45%)
        if 'three_points_att' in df.columns:
            if self._register_feature('threept_attempts_5g', 'top_features'):
                attempts_5g = self._get_historical_series(df, 'three_points_att', window=5, operation='mean')
                df['threept_attempts_5g'] = attempts_5g.fillna(df['three_points_att'].mean())
        
        # 11. THREEPT_EFFICIENCY_CALC (Rank 11 - 3.45%)
        if 'three_points_made' in df.columns and 'three_points_att' in df.columns:
            if self._register_feature('threept_efficiency_calc', 'top_features'):
                efficiency = self._get_historical_series(df, 'three_points_pct', window=10, operation='mean') if 'three_points_pct' in df.columns else pd.Series(0.35, index=df.index)
                volume = self._get_historical_series(df, 'three_points_att', window=10, operation='mean')
                df['threept_efficiency_calc'] = efficiency * volume / 10
        
        # 12. THREEPT_RATIO_OF_TOTAL (Rank 12 - 3.39%)
        if 'three_points_att' in df.columns and 'field_goals_att' in df.columns:
            if self._register_feature('threept_ratio_of_total', 'top_features'):
                ratio = self._get_historical_series(df, 'three_points_att', window=10, operation='mean') / self._get_historical_series(df, 'field_goals_att', window=10, operation='mean')
                df['threept_ratio_of_total'] = ratio.fillna(0.3)
        
        # 13. SHOOTING_VARIANCE_CONTROL (Rank 13 - 3.30%)
        if 'three_points_made' in df.columns:
            if self._register_feature('shooting_variance_control', 'top_features'):
                variance = self._get_historical_series(df, 'three_points_made', window=10, operation='std')
                control = 1 / (variance + 0.1)
                df['shooting_variance_control'] = control.fillna(1.0)
        
        # 14. OPP_RECENT_DEFENSE_TREND (Rank 14 - 3.02%)
        if 'Opp' in df.columns and self.teams_df is not None:
            if self._register_feature('opp_recent_defense_trend', 'top_features'):
                # Usar defensive_rating que ya está calculado en el dataset
                if 'defensive_rating' in self.teams_df.columns:
                    # Obtener defensive rating promedio móvil de los últimos 5 juegos por equipo
                    opp_defense = self.teams_df.groupby('Team')['defensive_rating'].rolling(5, min_periods=1).mean().shift(1).reset_index()
                    opp_defense['Team_Date'] = self.teams_df.groupby('Team')['Date'].shift(1).reset_index(drop=True)
                    
                    # Crear diccionario para mapear oponente -> defensive rating
                    defense_dict = dict(zip(opp_defense['Team'], opp_defense['defensive_rating']))
                    
                    # Aplicar al dataset de jugadores (menor defensive_rating = mejor defensa)
                    df['opp_recent_defense_trend'] = df['Opp'].map(defense_dict).fillna(110.0)  # 110 es promedio NBA
                else:
                    df['opp_recent_defense_trend'] = 110.0
        
        # 15. ELITE_MOMENTUM_AMPLIFIER (Rank 15 - 3.02%) - REUTILIZA threep_3game_avg
        if 'three_points_made' in df.columns:
            if self._register_feature('elite_momentum_amplifier', 'top_features'):
                # Reutilizar threep_3game_avg si existe
                if 'threep_3game_avg' in df.columns:
                    recent_3p = df['threep_3game_avg']
                else:
                    recent_3p = self._get_historical_series(df, 'three_points_made', window=3, operation='mean')
                
                player_avg_3p = df.groupby('player')['three_points_made'].transform('mean')
                hot_streak = recent_3p > (player_avg_3p * 1.2)
                elite_tier = player_avg_3p >= 2.5
                elite_hot_streak = hot_streak & elite_tier
                momentum_amp = np.where(elite_hot_streak, 1.5, 1.0)
                momentum_amp = np.where(hot_streak & ~elite_tier, 1.2, momentum_amp)
                df['elite_momentum_amplifier'] = momentum_amp
        
        # 16-30: Features restantes con lógica simplificada pero funcional
        remaining_features = self.feature_categories['top_features'][15:]
        
        for i, feature_name in enumerate(remaining_features, 16):
            if self._register_feature(feature_name, 'top_features'):
                if 'three_points_made' in df.columns:
                    # Lógica específica para cada feature basada en su nombre
                    if feature_name == 'threept_variability':
                        df[feature_name] = self._get_historical_series(df, 'three_points_made', window=5, operation='std').fillna(0.5)
                    elif feature_name == 'curry_efficiency_factor':
                        if 'three_points_pct' in df.columns:
                            efficiency = self._get_historical_series(df, 'three_points_pct', window=7, operation='mean')
                            volume = self._get_historical_series(df, 'three_points_att', window=7, operation='mean') if 'three_points_att' in df.columns else pd.Series(3.0, index=df.index)
                            df[feature_name] = efficiency * volume
                        else:
                            df[feature_name] = 1.0
                    elif feature_name == 'time_efficiency_3pt':
                        if 'minutes' in df.columns:
                            minutes = self._get_historical_series(df, 'minutes', window=5, operation='mean')
                            # Reutilizar threep_5game_avg si existe
                            if 'threep_5game_avg' in df.columns:
                                threes = df['threep_5game_avg']
                            else:
                                threes = self._get_historical_series(df, 'three_points_made', window=5, operation='mean')
                            df[feature_name] = threes / (minutes + 1) * 48
                        else:
                            df[feature_name] = 0.1
                    elif feature_name == 'hot_streak_probability':
                        # Reutilizar threep_3game_avg si existe
                        if 'threep_3game_avg' in df.columns:
                            hot_games = (df['threep_3game_avg'] >= 2).astype(float)
                        else:
                            hot_games = (self._get_historical_series(df, 'three_points_made', window=3, operation='mean') >= 2).astype(float)
                        df[feature_name] = hot_games
                    elif feature_name == 'player_tier_classification':
                        player_avg = df.groupby('player')['three_points_made'].transform('mean')
                        df[feature_name] = np.where(player_avg >= 3, 2, np.where(player_avg >= 1.5, 1, 0))
                    elif feature_name == 'dynamic_volume_predictor':
                        if 'three_points_att' in df.columns:
                            recent_volume = self._get_historical_series(df, 'three_points_att', window=3, operation='mean')
                            season_volume = df.groupby('player')['three_points_att'].transform('mean')
                            df[feature_name] = recent_volume / (season_volume + 1)
                        else:
                            df[feature_name] = 1.0
                    elif feature_name == 'zone_entry_frequency':
                        # Frecuencia de entrar en zona caliente - REUTILIZA threep_3game_avg
                        if 'threep_3game_avg' in df.columns:
                            hot_zone = (df['threep_3game_avg'] >= 2).astype(int)
                        else:
                            hot_zone = (self._get_historical_series(df, 'three_points_made', window=3, operation='mean') >= 2).astype(int)
                        df[feature_name] = self._get_historical_series(df, 'three_points_made', window=10, operation='mean').fillna(0.2)
                    elif feature_name == 'adaptive_shooting_profile':
                        if 'three_points_pct' in df.columns and 'three_points_att' in df.columns:
                            efficiency_trend = self._get_historical_series(df, 'three_points_pct', window=5, operation='mean')
                            volume_trend = self._get_historical_series(df, 'three_points_att', window=5, operation='mean')
                            df[feature_name] = efficiency_trend * np.log1p(volume_trend)
                        else:
                            df[feature_name] = 1.0
                    elif feature_name == 'threept_minutes_impact':
                        if 'minutes' in df.columns:
                            minutes_impact = self._get_historical_series(df, 'minutes', window=5, operation='mean') / 48
                            # Reutilizar threep_5game_avg si existe
                            if 'threep_5game_avg' in df.columns:
                                threes_per_min = df['threep_5game_avg'] / (minutes_impact + 0.1)
                            else:
                                threes_per_min = self._get_historical_series(df, 'three_points_made', window=5, operation='mean') / (minutes_impact + 0.1)
                            df[feature_name] = threes_per_min
                        else:
                            df[feature_name] = 0.1
                    elif feature_name == 'volume_clutch_synergy':
                        if 'three_points_att' in df.columns:
                            volume = self._get_historical_series(df, 'three_points_att', window=5, operation='mean')
                            # Reutilizar threep_5game_avg si existe
                            if 'threep_5game_avg' in df.columns:
                                success_rate = df['threep_5game_avg'] / (volume + 1)
                            else:
                                success_rate = self._get_historical_series(df, 'three_points_made', window=5, operation='mean') / (volume + 1)
                            df[feature_name] = volume * success_rate
                        else:
                            df[feature_name] = 1.0
                    elif feature_name == 'extreme_value_momentum':
                        max_3p = self._get_historical_series(df, 'three_points_made', window=10, operation='max')
                        avg_3p = self._get_historical_series(df, 'three_points_made', window=10, operation='mean')
                        df[feature_name] = (max_3p - avg_3p).fillna(0.0)
                    elif feature_name == 'threept_attempts_last':
                        if 'three_points_att' in df.columns:
                            df[feature_name] = df.groupby('player')['three_points_att'].shift(1).fillna(df['three_points_att'].mean())
                        else:
                            df[feature_name] = 3.0
                    elif feature_name == 'volume_shooter_trait':
                        if 'three_points_att' in df.columns:
                            season_avg = df.groupby('player')['three_points_att'].transform('mean')
                            df[feature_name] = (season_avg >= 6).astype(float)
                        else:
                            df[feature_name] = 0.0
                    elif feature_name == 'is_three_point_specialist':
                        if 'three_points_pct' in df.columns and 'three_points_att' in df.columns:
                            efficiency = df.groupby('player')['three_points_pct'].transform('mean')
                            volume = df.groupby('player')['three_points_att'].transform('mean')
                            df[feature_name] = ((efficiency >= 0.36) & (volume >= 4)).astype(float)
                        else:
                            df[feature_name] = 0.0
                    elif feature_name == 'clutch_shooter_trait':
                        if 'three_points_pct' in df.columns:
                            clutch_efficiency = df.groupby('player')['three_points_pct'].transform('mean')
                            df[feature_name] = (clutch_efficiency >= 0.38).astype(float)
                        else:
                            df[feature_name] = 0.0
                    else:
                        # Default para features no específicamente definidas
                        df[feature_name] = self._get_historical_series(df, 'three_points_made', window=5, operation='mean').fillna(1.0)
                else:
                    # Si no hay columna three_points_made, usar valores por defecto
                    df[feature_name] = 1.0

    def _generate_core_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Genera las features NÚCLEO más predictivas identificadas por el modelo
        Enfoque en MP, eficiencia ofensiva, y patrones de anotación
        """
        logger.debug("Generando features NÚCLEO más predictivas...")
        
        # Promedio de puntos histórico MEJORADO 
        if 'three_points_made' in df.columns:
            # Usar ventanas espaciadas para reducir correlación
            threep_short = self._get_historical_series(df, 'three_points_made', self.windows['short'], 'mean')
            threep_long = self._get_historical_series(df, 'three_points_made', self.windows['long'], 'mean')
            threep_season = self._get_historical_series(df, 'three_points_made', self.windows['season'], 'mean')
            
            # Momentum scoring basado en aceleración de 3PT (clipping más suave)
            df['threep_momentum'] = (threep_short - threep_long) / (threep_long - threep_season + 0.1)
            df['threep_momentum'] = df['threep_momentum'].clip(-1.5, 1.5).fillna(0)  # Clipping más suave
            self._register_feature('threep_momentum', 'core_predictive')
            
            # Scoring ceiling (máximo reciente vs promedio)
            threep_max_recent = self._get_historical_series(df, 'three_points_made', self.windows['medium'], 'max')
            df['scoring_ceiling'] = threep_max_recent - threep_long
            self._register_feature('scoring_ceiling', 'core_predictive')
            
            # Weighted recent scoring
            # Weighted average que da más peso a juegos recientes
            weights_recent = np.array([0.4, 0.3, 0.2, 0.1])  # Para últimos 4 juegos
            if len(weights_recent) <= self.windows['medium']:
                # Usar _get_historical_series para evitar data leakage
                threep_weighted = self._get_historical_series(df, 'three_points_made', window=4, operation='mean')
                df['threep_weighted_recent'] = threep_weighted.fillna(threep_long)
                self._register_feature('threep_weighted_recent', 'core_predictive')
            
            # ELITE PLAYER FEATURES
            # Explosiveness Factor - Detecta capacidad de juegos excepcionales
            threep_max_7 = self._get_historical_series(df, 'three_points_made', window=7, operation='max')
            # Calcular percentil 90 usando rolling directamente
            threep_p90_7 = df.groupby('player')['three_points_made'].rolling(7, min_periods=3).quantile(0.9).shift(1).reset_index(0, drop=True)
            df['explosiveness_factor'] = ((threep_max_7 - threep_p90_7) / (threep_long + 1e-6)).clip(0, 2.0).fillna(0)
            self._register_feature('explosiveness_factor', 'elite_performance')
            
            # Elite Pressure Response - Rendimiento en situaciones críticas
            # Basado en diferencia entre máximo y mínimo recientes
            threep_min_7 = self._get_historical_series(df, 'three_points_made', window=7, operation='min')
            threep_range_7 = threep_max_7 - threep_min_7
            df['elite_pressure_response'] = (threep_long / (threep_range_7 + 1e-6)).clip(0, 3.0).fillna(1.0)
            self._register_feature('elite_pressure_response', 'elite_performance')
            
            # Dynamic Scoring Ceiling - Se adapta al rango de scoring del jugador
            scoring_tier = (threep_long // 5).clip(0, 8)  # Tiers: 0-5, 5-10, ..., 35-40
            tier_multiplier = 1.0 + scoring_tier * 0.1  # Aumenta con tier más alto
            df['dynamic_scoring_ceiling'] = (threep_max_recent - threep_long) * tier_multiplier
            self._register_feature('dynamic_scoring_ceiling', 'elite_performance')
            
            # Enhanced weighted average con ajuste por forma reciente
            recent_form_multiplier = 1 + (threep_short - threep_long) / (threep_long + 5) * 0.3
            df['threep_long_enhanced'] = threep_long * recent_form_multiplier.clip(0.7, 1.4)
            
            # CONTEXTUAL ADAPTIVE WEIGHTING - Se adapta al tier del jugador
            def adaptive_contextual_prediction(group):
                threep = group['three_points_made']
                threep_avg = self._get_historical_series(group, 'three_points_made', window=15, operation='mean')
                
                # Determinar tier del jugador dinámicamente
                tier = (threep_avg // 5).fillna(3).clip(0, 8)  # 0-8 tiers
                
                # Pesos adaptativos por tier
                # Tier bajo (0-5): Más peso a estabilidad
                # Tier medio (5-10): Balance equilibrado  
                # Tier alto (10+): Más peso a explosividad
                immediate_weight = 0.2 + tier * 0.02  # 0.2 to 0.36
                medium_weight = 0.5 - tier * 0.03     # 0.5 to 0.26
                explosive_weight = 0.3 + tier * 0.01  # 0.3 to 0.38
                
                # Calcular componentes usando _get_historical_series
                immediate_comp = self._get_historical_series(group, 'three_points_made', window=2, operation='mean') * immediate_weight
                medium_comp = self._get_historical_series(group, 'three_points_made', window=6, operation='mean') * medium_weight
                explosive_comp = self._get_historical_series(group, 'three_points_made', window=3, operation='max') * explosive_weight
                
                # Combinar con normalización
                combined = immediate_comp.fillna(0) + medium_comp.fillna(0) + explosive_comp.fillna(0)
                valid_sum = (~immediate_comp.isna()).astype(float) * immediate_weight + \
                           (~medium_comp.isna()).astype(float) * medium_weight + \
                           (~explosive_comp.isna()).astype(float) * explosive_weight
                           
                return combined / (valid_sum + 1e-6)
            
            # Aplicar contextual adaptive prediction
            try:
                contextual_results = df.groupby('player', group_keys=False).apply(adaptive_contextual_prediction)
                df['contextual_adaptive_threep'] = contextual_results.fillna(threep_long)
            except Exception:
                # Fallback seguro: usar pts_long si hay problemas con la función compleja
                df['contextual_adaptive_threep'] = threep_long
            self._register_feature('contextual_adaptive_threep', 'elite_performance')
            
            self._register_feature('threep_long_enhanced', 'core_predictive')  # 1.92% importance

            # MP Efficiency Matrix MEJORADA (MP es clave según modelo)
        if 'minutes' in df.columns and 'three_points_made' in df.columns:
            mp_avg = self._get_historical_series(df, 'minutes', self.windows['medium'], 'mean')
            mp_long = self._get_historical_series(df, 'minutes', self.windows['long'], 'mean')
            threep_avg = self._get_historical_series(df, 'three_points_made', self.windows['medium'], 'mean')
            
            # Eficiencia por minuto con boost por high minutes
            base_efficiency = self._safe_divide(threep_avg, mp_avg + 1)
            minutes_multiplier = np.where(mp_avg >= 32, 1.25,      # Elite starters
                                np.where(mp_avg >= 28, 1.15,       # Strong starters  
                                np.where(mp_avg >= 20, 1.0,        # Normal
                                np.where(mp_avg >= 15, 0.85, 0.6)))) # Bench players
            
            df['mp_efficiency_core'] = base_efficiency * minutes_multiplier

            self._register_feature('mp_efficiency_core', 'core_predictive')
        
        # Offensive Volume SUPER MEJORADO
        if 'field_goals_att' in df.columns:
            fga_avg = self._get_historical_series(df, 'field_goals_att', self.windows['medium'], 'mean')
            fga_long = self._get_historical_series(df, 'field_goals_att', self.windows['long'], 'mean')
            fga_season = self._get_historical_series(df, 'field_goals_att', self.windows['season'], 'mean')
            
            # Volumen ofensivo base
            df['offensive_volume'] = fga_avg
            self._register_feature('offensive_volume', 'core_predictive')
            
            # Volume trend con múltiples capas
            df['volume_trend'] = self._safe_divide(fga_avg, fga_season + 1, 1.0)
            self._register_feature('volume_trend', 'core_predictive')

    def _generate_elite_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Genera features ultra-predictivas basándome en las columnas ORIGINALES del dataset
        y las TOP features que ya identificó el modelo. Usa FG, FGA, 3P, TRB, AST, +/-, GmSc, etc.
        """
        logger.debug("Generando features ultra-predictivas desde columnas originales...")
        
        # Advanced Shooting Metrics
        if all(col in df.columns for col in ['field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'free_throws_made', 'free_throws_att']):
            # Shot distribution intelligence
            fg_avg = self._get_historical_series(df, 'field_goals_made', self.windows['medium'], 'mean')
            fga_avg = self._get_historical_series(df, 'field_goals_att', self.windows['medium'], 'mean')
            three_p_avg = self._get_historical_series(df, 'three_points_made', self.windows['medium'], 'mean')
            three_pa_avg = self._get_historical_series(df, 'three_points_att', self.windows['medium'], 'mean')
            ft_avg = self._get_historical_series(df, 'free_throws_made', self.windows['medium'], 'mean')
            
            # Two-point attempts and makes
            two_pa_avg = fga_avg - three_pa_avg
            two_p_avg = fg_avg - three_p_avg
            
            # Shot making ability por tipo
            three_p_acc = self._safe_divide(three_p_avg, three_pa_avg, 0.35)
            two_p_acc = self._safe_divide(two_p_avg, two_pa_avg, 0.50)
            
            # Shot difficulty profile (mezcla eficiencia con volumen)
            three_p_rate = self._safe_divide(three_pa_avg, fga_avg, 0.33)
            two_p_rate = self._safe_divide(two_pa_avg, fga_avg, 0.67)
            df['shot_difficulty_mastery'] = (three_p_acc * three_p_rate) + (two_p_acc * two_p_rate)
            
            self._register_feature('shot_difficulty_mastery', 'efficiency_metrics')  # 2.05% importance
            
        # Game Impact Metrics (basado en +/-, GmSc, BPM)
        if 'efficiency_game_score' in df.columns:
            # Game Score momentum (trending performance)
            gmsc_avg = self._get_historical_series(df, 'efficiency_game_score', self.windows['medium'], 'mean')
            gmsc_long = self._get_historical_series(df, 'efficiency_game_score', self.windows['long'], 'mean')
            
            df['game_impact_momentum'] = self._safe_divide(gmsc_avg, gmsc_long + 1, 1.0)
                        
            self._register_feature('game_impact_momentum', 'core_predictive')
        
        if 'plus_minus' in df.columns:
            # Plus/Minus trend (team performance when player is on court)
            pm_avg = self._get_historical_series(df, 'plus_minus', self.windows['medium'], 'mean')
        
        # Playmaking & Rebounds Contribution (AST, TRB, STL, BLK)
        if all(col in df.columns for col in ['assists', 'rebounds', 'steals', 'blocks']):
            ast_avg = self._get_historical_series(df, 'assists', self.windows['medium'], 'mean')
            trb_avg = self._get_historical_series(df, 'rebounds', self.windows['medium'], 'mean')
            stl_avg = self._get_historical_series(df, 'steals', self.windows['medium'], 'mean')
            blk_avg = self._get_historical_series(df, 'blocks', self.windows['medium'], 'mean')

    def _generate_advanced_context_features(self, df: pd.DataFrame) -> None:
        """
        Genera todas las features avanzadas de contexto y matchup
        Incluye rachas específicas, tendencias con regresión, matchups históricos, etc.
        """
        logger.debug("Generando features avanzadas de contexto y matchup...")
        
        # Tendencia y Forma Reciente AVANZADA
        
        # 1.1 Promedios móviles específicos (3, 5, 10 partidos)
        if 'three_points_made' in df.columns:
            df['threep_3game_avg'] = self._get_historical_series(df, 'three_points_made', 3, 'mean')
            df['threep_5game_avg'] = self._get_historical_series(df, 'three_points_made', 5, 'mean')

            self._register_feature('threep_3game_avg', 'core_predictive')
            self._register_feature('threep_5game_avg', 'core_predictive')

        # 1.2 Tendencia con regresión lineal (pendiente de últimos N partidos)  
        if 'three_points_made' in df.columns:
            # Rolling con shift(1) para evitar data leakage
            df['threep_trend_5game'] = df.groupby('player')['three_points_made'].transform(
                lambda x: x.rolling(5, min_periods=3).apply(
                    lambda y: (
                        np.polyfit(range(len(y)), y, 1)[0] / (y.mean() + 1e-6) 
                        if len(y) >= 3 and y.mean() > 0 else 0
                    )
                ).shift(1)
            ).fillna(0)
            
            df['threep_trend_10game'] = df.groupby('player')['three_points_made'].transform(
                lambda x: x.rolling(10, min_periods=3).apply(
                    lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 3 else 0
                ).shift(1)
            ).fillna(0)
            
            self._register_feature('threep_trend_5game', 'core_predictive')
            self._register_feature('threep_trend_10game', 'core_predictive')
        
        # 1.3 Consistencia avanzada (desviación estándar específica)
        if 'three_points_made' in df.columns:
            threep_5game_std = self._get_historical_series(df, 'three_points_made', 5, 'std')
            threep_10game_std = self._get_historical_series(df, 'three_points_made', 10, 'std')
            
            # Consistency score (menor std = más consistente)
            df['threep_5game_consistency'] = 1 / (1 + threep_5game_std)

            self._register_feature('threep_5game_consistency', 'core_predictive')

        # 1.4 Racha actual REAL (partidos consecutivos por encima/debajo de promedio)
        if 'three_points_made' in df.columns:
            threep_season_avg = self._get_historical_series(df, 'three_points_made', self.windows['season'], 'mean')
            
            def calculate_streak_no_leak(group):
                """Calcula racha hasta el juego ANTERIOR (sin incluir juego actual)"""
                if len(group) < 2:
                    return pd.Series([0] * len(group), index=group.index)
                
                # Usar solo datos hasta el juego anterior
                avg_expanded = threep_season_avg[group.index]
                streaks = [0]  # Primer juego no tiene streak
                
                for i in range(1, len(group)):
                    # Solo usar datos hasta i-1 (excluyendo el juego actual)
                    past_games = group.iloc[:i]  # Solo juegos pasados
                    past_avgs = avg_expanded.iloc[:i]
                    
                    if len(past_games) == 0:
                        streaks.append(0)
                        continue
                    
                    # Calcular streak hasta el juego anterior
                    above_avg = (past_games > past_avgs).astype(int) * 2 - 1  # 1 o -1
                    
                    if len(above_avg) == 0:
                        streaks.append(0)
                        continue
                    
                    # Calcular streak actual desde el final hacia atrás
                    current_direction = above_avg.iloc[-1]
                    streak = 1
                    
                    for j in range(len(above_avg) - 2, -1, -1):
                        if above_avg.iloc[j] == current_direction:
                            streak += 1
                        else:
                            break
                    
                    streaks.append(streak * current_direction)
                
                return pd.Series(streaks, index=group.index)
            
            df['current_streak'] = df.groupby('player')['three_points_made'].apply(calculate_streak_no_leak).reset_index(level=0, drop=True)
                        
            self._register_feature('current_streak', 'core_predictive')
        
        # Contextuales del Partido AVANZADOS
        
        # 2.1 Local vs Visitante diferencia histórica
        if 'is_home' in df.columns and 'three_points_made' in df.columns:
            # CRÍTICO: Calcular rendimiento histórico local/visitante sin incluir juego actual
            def calculate_historical_home_away(group):
                """Calcula rendimiento histórico local/visitante usando solo juegos anteriores"""
                home_results = []
                away_results = []
                
                for idx in range(len(group)):
                    if idx == 0:
                        home_results.append(np.nan)
                        away_results.append(np.nan)
                    else:
                        # Solo juegos anteriores
                        past_data = group.iloc[:idx]
                        
                        # Rendimiento en casa (histórico)
                        past_home = past_data[past_data['is_home'] == 1]['three_points_made']
                        if len(past_home) > 0:
                            home_results.append(past_home.mean())
                        else:
                            home_results.append(np.nan)
                        
                        # Rendimiento visitante (histórico)
                        past_away = past_data[past_data['is_home'] == 0]['three_points_made']
                        if len(past_away) > 0:
                            away_results.append(past_away.mean())
                        else:
                            away_results.append(np.nan)
                
                return pd.DataFrame({
                    'home_performance': home_results,
                    'away_performance': away_results
                }, index=group.index)
            
            # Aplicar función
            home_away_perf = df.groupby('player')[['is_home', 'three_points_made']].apply(
                calculate_historical_home_away
            ).reset_index(level=0, drop=True)
            
            df['home_performance'] = home_away_perf['home_performance']
            df['away_performance'] = home_away_perf['away_performance']
            
            # Fill NaN con promedio general del jugador
            player_avg_hist = df.groupby('player')['three_points_made'].expanding().mean().shift(1).reset_index(level=0, drop=True)
            GLOBAL_THREEP_MEAN = 1.5  # Promedio histórico NBA aproximado - CONSTANTE SIN DATA LEAKAGE
            df['home_performance'] = df['home_performance'].fillna(player_avg_hist).fillna(GLOBAL_THREEP_MEAN)
            df['away_performance'] = df['away_performance'].fillna(player_avg_hist).fillna(GLOBAL_THREEP_MEAN)
            
            # Registrar features home/away
            self._register_feature('home_performance', 'context_adjusters')
            self._register_feature('away_performance', 'context_adjusters')
            
            # Calcular home_away_diff
            df['home_away_diff'] = df['home_performance'] - df['away_performance']
            self._register_feature('home_away_diff', 'context_adjusters')
            
            # Factor contextual mejorado
            df['home_context_factor'] = np.where(
                df['is_home'] == 1,
                1.0 + np.clip(df['home_away_diff'] / 20, -0.2, 0.2),  # Boost si es mejor en casa
                1.0 - np.clip(df['home_away_diff'] / 20, -0.2, 0.2)   # Penalty si es peor visitante
            )
            
            self._register_feature('home_context_factor', 'context_adjusters')
        
        # 3.1 Rendimiento histórico vs ese equipo específico
        if 'Opp' in df.columns and 'three_points_made' in df.columns:
            # Calcular promedio histórico EXCLUYENDO el juego actual
            def calculate_historical_vs_team(group):
                """Calcula promedio vs equipo usando solo juegos ANTERIORES"""
                results = []
                for idx in range(len(group)):
                    if idx == 0:
                        # Primer juego vs este equipo, usar promedio general
                        results.append(np.nan)
                    else:
                        # Usar solo juegos anteriores vs este equipo
                        past_games = group.iloc[:idx]
                        if len(past_games) > 0:
                            results.append(past_games.mean())
                        else:
                            results.append(np.nan)
                return pd.Series(results, index=group.index)
            
            # Calcular promedio histórico
            df['vs_team_avg'] = df.groupby(['player', 'Opp'])['three_points_made'].apply(
                calculate_historical_vs_team
            ).reset_index(level=[0,1], drop=True)
            
            # Promedio general del jugador
            def calculate_historical_overall(group):
                """Calcula promedio general usando solo juegos ANTERIORES"""
                results = []
                for idx in range(len(group)):
                    if idx == 0:
                        results.append(np.nan)
                    else:
                        past_games = group.iloc[:idx]
                        if len(past_games) > 0:
                            results.append(past_games.mean())
                        else:
                            results.append(np.nan)
                return pd.Series(results, index=group.index)
            
            df['player_overall_avg'] = df.groupby('player')['three_points_made'].apply(
                calculate_historical_overall
            ).reset_index(level=0, drop=True)
            
            # Fill NaN with defaults and calculate advantage
            GLOBAL_THREEP_MEAN = 1.5  # Promedio histórico NBA aproximado
            df['vs_team_avg'] = df['vs_team_avg'].fillna(df['player_overall_avg']).fillna(GLOBAL_THREEP_MEAN)
            df['player_overall_avg'] = df['player_overall_avg'].fillna(GLOBAL_THREEP_MEAN)
            
            self._register_feature('player_overall_avg', 'core_predictive')
            df['matchup_advantage'] = df['vs_team_avg'] - df['player_overall_avg']
            
            self._register_feature('vs_team_avg', 'core_predictive')  # Reactivada para completar set
            self._register_feature('matchup_advantage', 'core_predictive')  # Reactivada para completar set
        
        # 3.2 Opponent defensive rating y pace (usar columnas ya calculadas)
        if self.teams_df is not None and 'Opp' in df.columns:
            try:
                # Usar defensive_rating y offensive_rating que ya están calculados
                opp_def_stats = self.teams_df.groupby('Team').agg({
                    'defensive_rating': 'mean',  # Ya calculado en el dataset
                    'offensive_rating': 'mean',  # Ya calculado en el dataset
                    'three_points_made': 'mean'  # Pace indicator
                }).reset_index()
                
                # Crear diccionarios para mapping
                opp_def_rating = dict(zip(opp_def_stats['Team'], opp_def_stats['defensive_rating']))
                opp_pace = dict(zip(opp_def_stats['Team'], opp_def_stats['three_points_made']))
                
                # Aplicar al dataset
                df['opp_defensive_rating'] = df['Opp'].map(opp_def_rating).fillna(110.0)
                df['opp_pace_factor'] = df['Opp'].map(opp_pace).fillna(12.0)  # Promedio NBA de triples por equipo
                
                self._register_feature('opp_defensive_rating', 'opponent_context')
                self._register_feature('opp_pace_factor', 'opponent_context')
                
            except Exception as e:
                logger.debug(f"No se pudieron crear features de opponent stats: {e}")
        
        # 4.2 Team assists context (más asistencias = más oportunidades)
        try:
            if 'assists' in df.columns and 'Team' in df.columns and 'Date' in df.columns:
                # Crear feature más simple y robusta
                # Calcular promedio de asistencias por equipo basado en datos históricos
                team_ast_rolling = df.groupby('Team')['three_points_made'].rolling(
                    window=10, min_periods=1
                ).mean().shift(1).reset_index(0, drop=True)
                
                # Asignar con manejo de índices
                if len(team_ast_rolling) == len(df):
                    df['team_assists_avg'] = team_ast_rolling.fillna(25)
                else:
                    df['team_assists_avg'] = 25.0
            else:
                # Crear feature por defecto si no hay datos
                df['team_assists_avg'] = 25.0  # Valor promedio de asistencias por equipo
        except Exception as e:
            logger.warning(f"Error creando team_assists_avg: {e}")
            df['team_assists_avg'] = 25.0
        
        self._register_feature('team_assists_avg', 'context_adjusters')
                    
    def _generate_high_range_specialist_features(self, df: pd.DataFrame) -> None:
        """
        Features especializadas en rangos altos de triples.
        """
        logger.debug("Generando features especialistas de rangos altos...")
        
        if 'three_points_made' not in df.columns or 'player' not in df.columns:
            logger.warning("No se pueden generar features de rangos altos sin PTS y Player")
            return
        
        if 'threep_trend_5game' in df.columns and 'threep_3game_avg' in df.columns:
            # Calcular baseline dinámico por jugador
            player_baseline = df.groupby('player')['three_points_made'].transform(
                lambda x: x.shift(1).expanding(min_periods=5).mean()
            ).fillna(df['threep_3game_avg'])
            
            # Momentum normalizado por el baseline del jugador
            trend_normalized = df['threep_trend_5game'] / (player_baseline + 1)
            recent_vs_baseline = (df['threep_3game_avg'] - player_baseline) / (player_baseline + 1)
            # prob_factor eliminado
            
            # Fórmula simplificada con pesos balanceados (prob_factor eliminado)
            high_range_momentum = (
                trend_normalized * 0.7 +      # Tendencia (peso aumentado)
                recent_vs_baseline * 0.3      # Nivel reciente vs baseline
            )
            
            # Sin clip arbitrario
            df['high_range_momentum'] = high_range_momentum.fillna(0.0)
            self._register_feature('high_range_momentum', 'high_range_specialist')
        
        # Ceiling predictor especializado
        # Predice el techo de triples basado en condiciones específicas
        ceiling_factors = 1.0
        
        # Factor por minutos
        if 'minutes' in df.columns:
            mp_factor = df['minutes'] / 36.0  # Factor basado en minutos starter
            ceiling_factors *= (1 + mp_factor * 0.3)
        
        # Factor por efficiency reciente
        if 'true_shooting_pct' in df.columns:
            ts_factor = df['true_shooting_pct'].fillna(0.5)
            efficient_boost = np.where(ts_factor > 0.6, 1.3, 1.0)
            ceiling_factors *= efficient_boost
        
        # Factor por momentum
        if 'threep_trend_5game' in df.columns:
            momentum_boost = 1 + np.clip(df['threep_trend_5game'] / 5.0, -0.3, 0.5)
            ceiling_factors *= momentum_boost
        
        # Ceiling base del jugador
        base_ceiling = df.get('threep_weighted_recent', 12) * 1.8  # 80% más que su promedio
        df['specialized_ceiling'] = base_ceiling * ceiling_factors
        df['specialized_ceiling'] = np.clip(df['specialized_ceiling'], 8, 50)

    def _generate_advanced_dataset_features(self, df: pd.DataFrame) -> None:
        """
        Genera features avanzadas usando métricas existentes del nuevo dataset para TRIPLES
        """
        logger.debug("Generando features avanzadas del dataset para TRIPLES...")

        # 1. TRUE SHOOTING PERCENTAGE ENHANCEMENT (del dataset)
        if 'true_shooting_pct' in df.columns:
            if self._register_feature('true_shooting_enhanced', 'top_features'):
                # Usar directamente TS% del dataset con tendencias históricas
                ts_hist = self._get_historical_series(df, 'true_shooting_pct', window=5, operation='mean')
                ts_trend = self._get_historical_series(df, 'true_shooting_pct', window=3, operation='mean') - \
                          self._get_historical_series(df, 'true_shooting_pct', window=10, operation='mean')
                df['true_shooting_enhanced'] = ts_hist + ts_trend * 0.2

        # 2. EFFECTIVE FIELD GOAL PERCENTAGE (del dataset)
        if 'effective_fg_pct' in df.columns:
            if self._register_feature('effective_fg_enhanced', 'top_features'):
                # eFG% ya considera el valor extra de los triples
                efg_hist = self._get_historical_series(df, 'effective_fg_pct', window=5, operation='mean')
                efg_consistency = self._get_historical_series(df, 'effective_fg_pct', window=7, operation='std')
                # Valorar consistencia alta (baja desviación)
                consistency_bonus = 1.0 / (efg_consistency + 0.01)
                df['effective_fg_enhanced'] = efg_hist * consistency_bonus

        # 3. THREE POINT VOLUME PERCENTAGE (del dataset)
        if all(col in df.columns for col in ['three_points_att', 'field_goals_att']):
            if self._register_feature('three_point_volume_pct', 'top_features'):
                # Porcentaje de intentos de campo que son triples (tendencia)
                threept_att_hist = self._get_historical_series(df, 'three_points_att', window=5, operation='mean')
                fg_att_hist = self._get_historical_series(df, 'field_goals_att', window=5, operation='mean')
                volume_pct = threept_att_hist / (fg_att_hist + 0.1)
                df['three_point_volume_pct'] = volume_pct.fillna(0.35)

        # 4. OFFENSIVE RATING CORRELATION (del dataset)
        if all(col in df.columns for col in ['offensive_rating', 'three_points_made']):
            if self._register_feature('offensive_rating_3pt_synergy', 'top_features'):
                # Correlación entre rating ofensivo y producción de triples
                off_rating_hist = self._get_historical_series(df, 'offensive_rating', window=5, operation='mean')
                # Reutilizar threep_5game_avg si existe
                if 'threep_5game_avg' in df.columns:
                    threept_made_hist = df['threep_5game_avg']
                else:
                    threept_made_hist = self._get_historical_series(df, 'three_points_made', window=5, operation='mean')
                # Normalizar rating ofensivo (suele estar 100-130)
                normalized_rating = (off_rating_hist - 100) / 30.0
                synergy = normalized_rating * threept_made_hist
                df['offensive_rating_3pt_synergy'] = synergy.fillna(0.5)

        # 5. FAST BREAK THREE POINTERS (Feature del dataset)
        if all(col in df.columns for col in ['fast_break_pts', 'three_points_made']):
            if self._register_feature('fast_break_three_efficiency', 'top_features'):
                # Correlación entre puntos de contraataque y triples (transiciones)
                fb_pts_hist = self._get_historical_series(df, 'fast_break_pts', window=5, operation='mean')
                # Reutilizar threep_5game_avg si existe
                if 'threep_5game_avg' in df.columns:
                    threept_made_hist = df['threep_5game_avg']
                else:
                    threept_made_hist = self._get_historical_series(df, 'three_points_made', window=5, operation='mean')
                # Los triples valen 3 puntos, relación esperada
                fb_three_ratio = threept_made_hist * 3.0 / (fb_pts_hist + 0.1)
                df['fast_break_three_efficiency'] = fb_three_ratio.fillna(0.3)

        # 6. PLUS/MINUS IMPACT ON THREES (del dataset)
        if all(col in df.columns for col in ['plus', 'minus', 'three_points_made']):
            if self._register_feature('plus_minus_three_impact', 'top_features'):
                # Impacto del +/- en la producción de triples
                plus_minus_hist = self._get_historical_series(df, 'plus', window=5, operation='mean') - \
                                 self._get_historical_series(df, 'minus', window=5, operation='mean')
                # Reutilizar threep_5game_avg si existe
                if 'threep_5game_avg' in df.columns:
                    threept_made_hist = df['threep_5game_avg']
                else:
                    threept_made_hist = self._get_historical_series(df, 'three_points_made', window=5, operation='mean')
                # Jugadores con mejor +/- tienden a tener más oportunidades de triples
                impact_score = plus_minus_hist * threept_made_hist / 10.0
                df['plus_minus_three_impact'] = impact_score.fillna(0.0)

    def _generate_quarters_features(self, df: pd.DataFrame) -> None:
        """
        Genera features de datos por cuartos específicas para TRIPLES
        """
        logger.debug("Generando features de cuartos para TRIPLES...")
        
        if self.players_quarters_df is None:
            logger.warning("Dataset de cuartos no disponible")
            return
            
        # 1. CLUTCH SHOOTING (Q4)
        if self._register_feature('clutch_three_shooting', 'top_features'):
            # Promedio de triples en situaciones clutch (Q4)
            q4_data = self.players_quarters_df[self.players_quarters_df['quarter'] == 4]
            if not q4_data.empty:
                clutch_avg = q4_data.groupby('player')['three_points_made'].apply(
                    lambda x: self._get_historical_series(x.to_frame(), 'three_points_made', window=5, operation='mean')
                ).reset_index(0, drop=True)
                
                # Mapear a DataFrame principal
                clutch_mapping = {}
                for player in q4_data['player'].unique():
                    if player in clutch_avg.index:
                        clutch_mapping[player] = clutch_avg[player]
                    else:
                        clutch_mapping[player] = 0.8
                
                df['clutch_three_shooting'] = df['player'].map(clutch_mapping).fillna(0.8)
            else:
                df['clutch_three_shooting'] = 0.8

        # 2. EARLY GAME THREE AGGRESSION (Q1-Q2)
        if self._register_feature('early_game_three_aggression', 'top_features'):
            # Agresividad en triples durante primeros dos cuartos
            early_quarters = self.players_quarters_df[self.players_quarters_df['quarter'].isin([1, 2])]
            if not early_quarters.empty:
                early_aggression = early_quarters.groupby('player')['three_points_att'].apply(
                    lambda x: self._get_historical_series(x.to_frame(), 'three_points_att', window=5, operation='mean')
                ).reset_index(0, drop=True)
                
                # Mapear a DataFrame principal
                aggression_mapping = {}
                for player in early_quarters['player'].unique():
                    if player in early_aggression.index:
                        aggression_mapping[player] = early_aggression[player]
                    else:
                        aggression_mapping[player] = 2.0
                
                df['early_game_three_aggression'] = df['player'].map(aggression_mapping).fillna(2.0)
            else:
                df['early_game_three_aggression'] = 2.0
