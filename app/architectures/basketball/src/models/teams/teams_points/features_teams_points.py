"""
Modulo de features para el modelo de prediccion de puntos 
de los equipos de la NBA en la mitad del partido
============================================================
"""

import pandas as pd
import numpy as np
import os
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class TeamPointsFeatureEngineer:
    """Feature Engineer optimizado para las 15 features más importantes del modelo entrenado."""
    
    def __init__(self, teams_total_df=None, teams_quarters_df=None, df_players=None):
        """Inicializa el feature engineer optimizado."""
        self.feature_columns = []
        self.feature_registry = {}  # Registro de features generadas
        self.df_players = df_players  # Dataset de jugadores para cálculos reales
        self.teams_total_df = teams_total_df  # Dataset de equipos total
        self.teams_quarters_df = teams_quarters_df  # Dataset de equipos por cuartos
        
    # VENTANAS ESPACIADAS para reducir correlación
        self.windows = {
            'short': 3,      # Tendencia inmediata
            'medium': 10,     # Tendencia reciente  
            'long': 20,      # Tendencia estable
            'season': 30     # Baseline temporal
        }
        self.feature_categories = {
            'shooting_efficiency': [],  # Eficiencia de tiro (genera oportunidades)
            'physical_advantage': [],   # Ventaja física (altura, peso)
            'positioning': [],          # Posicionamiento y minutos
            'team_context': [],         # Contexto del equipo
            'opponent_context': [],     # Contexto del oponente
            'rebounding_history': [],   # Historial de rebotes
            'game_situation': []        # Situación del juego
        }
        self.protected_features = ['TRB', 'Player', 'Date', 'Team', 'Opp', 'Pos']

        # Cache para cálculos repetidos
        self._rolling_cache = {}
    
    def clear_cache(self):
        """Limpia el cache de cálculos previos para nueva predicción"""
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
        Cálculo rolling optimizado con cache para evitar duplicados
        SIEMPRE usa shift(1) para prevenir data leakage (futuros juegos)
        """
        # Crear cache key que incluya el tamaño del DataFrame para evitar conflictos
        cache_key = f"{column}_{window}_{operation}_{len(df)}"
        
        if cache_key in self._rolling_cache:
            cached_result = self._rolling_cache[cache_key]
            # Verificar que el índice coincida con el DataFrame actual
            if len(cached_result) == len(df):
                return cached_result
            else:
                # Cache obsoleto, eliminarlo
                del self._rolling_cache[cache_key]
        
        if column not in df.columns:
            result = pd.Series(0, index=df.index)
        else:
            if operation == 'mean':
                    result = df.groupby('Team')[column].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
            elif operation == 'std':
                    result = df.groupby('Team')[column].rolling(window=window, min_periods=2).std().reset_index(0, drop=True).fillna(0)
            elif operation == 'max':
                    result = df.groupby('Team')[column].rolling(window=window, min_periods=1).max().reset_index(0, drop=True).fillna(0)
            elif operation == 'sum':
                    result = df.groupby('Team')[column].rolling(window=window, min_periods=1).sum().reset_index(0, drop=True).fillna(0)
            else:
                    result = df.groupby('Team')[column].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
            
            # Asegurar que el resultado tenga el mismo índice que df
            if len(result) != len(df):
                result = pd.Series(result.values, index=df.index)
            
            self._rolling_cache[cache_key] = result
            return result
    
    def _safe_divide(self, numerator, denominator, default=0.0):
        """División segura optimizada"""
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.divide(numerator, denominator)
            result = np.where(denominator == 0, default, result)
            result = np.nan_to_num(result, nan=default, posinf=default, neginf=default)
        return result
    
    def _convert_mp_to_numeric(self, df: pd.DataFrame) -> None:
        """Convierte columna MP (minutos jugados) de formato MM:SS a decimal o usa 'minutes' si existe"""
        # Priorizar 'minutes' del nuevo dataset si existe
        if 'minutes' in df.columns:
            if 'MP' not in df.columns:
                df['MP'] = df['minutes']
            else:
                # Si ambos existen, usar 'minutes' que es más confiable
                df['MP'] = df['minutes']
        elif 'MP' in df.columns and df['MP'].dtype == 'object':
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
            
            df['MP'] = df['MP'].apply(parse_time)

    def generate_all_features(self, df: pd.DataFrame) -> List[str]:
        """Pipeline completo de generacion de caracteristicas para modelo de team points."""
        
        self._convert_mp_to_numeric(df)

        df.sort_values(['Team', 'Date'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Generar features súper predictivas basadas en análisis histórico
        self._create_super_predictive_features(df)
        self._create_explosive_potential_features(df)
        self._create_consistency_features(df)
        self._create_pace_features(df)
        self._create_defensive_features(df)
        self._create_momentum_features(df)
        
        # FEATURES SÚPER PREDICTIVAS BASADAS EN PATRONES DEL MODELO
        self._create_efg_mega_features(df)  # Basado en efg_squared_effect (79% importancia)
        self._create_pace_mega_features(df)  # Basado en múltiples features de pace
        self._create_points_mega_features(df)  # Basado en features de puntos históricos
        self._create_quarters_advanced_features(df)  # Aprovechando datos por cuartos
        
        # FEATURES ESPECÍFICAS PARA HALFTIME
        self._create_halftime_specific_features(df)  # Features específicas para predicción de halftime
        self._create_ratings_mega_features(df)  # Aprovechando offensive/defensive ratings
        
        # FEATURES SÚPER AVANZADAS PARA EXTREMOS
        self._create_extreme_detection_features(df)  # Detectar y predecir extremos
        self._create_volatility_mega_features(df)  # Features de volatilidad extrema
        self._create_outlier_prediction_features(df)  # Predecir outliers y extremos
        
        # LIMPIAR VALORES INFINITOS Y NAN
        self._clean_infinite_values(df)
        
        # Compilar lista de features (excluir columnas base y auxiliares)
        base_columns = [
            # === COLUMNAS BASE DEL DATASET ORIGINAL ===
            'Team', 'Date', 'Away', 'Opp', 'Result', 'MP',
            'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
            'FT', 'FTA', 'FT%', 'PTS',
            'FG_Opp', 'FGA_Opp', 'FG%_Opp', '2P_Opp', '2PA_Opp', '2P%_Opp',
            '3P_Opp', '3PA_Opp', '3P%_Opp', 'FT_Opp', 'FTA_Opp', 'FT%_Opp', 'PTS_Opp',
            
            # === COLUMNAS CRUDAS NUEVOS DATASETS (DATA LEAKAGE - EXCLUIR) ===
            'team_id', 'market', 'name', 'game_id', 'minutes',
            'field_goals_made', 'field_goals_att', 'field_goals_pct',
            'three_points_made', 'three_points_att', 'three_points_pct',
            'two_points_made', 'two_points_att', 'two_points_pct',
            'free_throws_made', 'free_throws_att', 'free_throws_pct',
            'offensive_rebounds', 'defensive_rebounds', 'rebounds', 'assists',
            'turnovers', 'steals', 'blocks', 'personal_fouls',
            'ejections', 'foulouts', 'flagrant_fouls', 'offensive_fouls',
            'player_tech_fouls', 'team_tech_fouls', 'coach_tech_fouls', 'coach_ejections',
            'fast_break_pts', 'second_chance_pts', 'team_turnovers', 'points_off_turnovers',
            'team_rebounds', 'bench_points', 'points_in_paint', 'points_in_paint_att',
            'points_in_paint_made', 'points_in_paint_pct', 'true_shooting_att', 'true_shooting_pct',
            'blocked_att', 'assists_turnover_ratio', 'pls_min',
            'team_defensive_rebounds', 'team_offensive_rebounds', 'total_fouls', 'total_rebounds',
            'total_turnovers', 'personal_rebounds', 'player_turnovers',
            'field_goals_at_rim_made', 'field_goals_at_rim_att', 'field_goals_at_rim_pct',
            'field_goals_at_midrange_made', 'field_goals_at_midrange_att', 'field_goals_at_midrange_pct',
            'fouls_drawn', 'second_chance_att', 'second_chance_made', 'second_chance_pct',
            'fast_break_att', 'fast_break_made', 'fast_break_pct',
            'period_type', 'period_number', 'period_sequence', 'quarter',
            'most_unanswered_points', 'most_unanswered_own_score', 'most_unanswered_opp_score',
            'offensive_rating', 'defensive_rating', 'team_fouls',
            
            # === COLUMNAS CRUDAS ORIGINALES (DATA LEAKAGE - EXCLUIR) ===
            'DRB', 'ORB', 'AST', 'STL', 'BLK', 'PF', 'TOV', 'HT',
            'efficiency_game_score',
            'true_shooting_pct', 'field_goals_pct', 'three_points_pct', 'free_throws_pct',
            
            # === TARGET VARIABLES Y DATA LEAKAGE (CRÍTICO: Excluir) ===
            'points', 'team_points', 'total_points', 'points_against', 'game_total_points',
            'is_win', 'is_home', 'biggest_lead', 'time_leading',
            'has_overtime', 'overtime_periods',
            'possessions', 'opponent_possessions', 'defensive_points_per_possession', 'offensive_points_per_possession',
            
            # === COLUMNAS AUXILIARES TEMPORALES (para cálculos internos, NO son features) ===
            # Auxiliares básicas generadas durante el procesamiento
            'eFG_pct', 'shooting_efficiency_15g', 'approx_tov_rate', 'ft_rate', 'approx_oreb_rate', 
            'momentum_factor',
            
            # Auxiliares de las 15 features principales
            'eFG_pct_stabilized_temp', 'pace_estimate_temp', 'ft_rate_temp', 
            'efficiency_temp', 'pace_factor_temp', 'synergy_temp',
            
            # Auxiliares de features extremas
            'pts_per_fga_temp', 'pts_per_fga', 
            
            # Auxiliares de features de interacciones
            'efg_pts_consistency_temp', 'game_dominance_temp', 'efficiency_per_point_temp',
            'pace_efficiency_temp', 'efg_squared_temp'
        ]
        
        # REGISTRAR TODAS LAS FEATURES GENERADAS (NO SOLO LAS 15 PRINCIPALES)
        all_features = [col for col in df.columns if col not in base_columns]
        self.feature_columns = all_features
        
        # Registrar TODAS las features generadas por los métodos
        for feature in all_features:
            self._register_feature(feature, 'team_points')
        
        return all_features
    
    def _create_super_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Crea features súper predictivas basadas en las métricas avanzadas de los nuevos datasets
        usando las 40 features más importantes del modelo histórico entrenado
        """
        
        # 1. PACE PROXY TEMPORAL (Feature más importante: 13.42%)
        # Usar possessions + estimaciones avanzadas
        if 'possessions' in df.columns and 'minutes' in df.columns:
            # Aproximar pace usando possessions históricas
            possessions_hist = self._get_historical_series(df, 'possessions', 5, 'mean')
            minutes_hist = self._get_historical_series(df, 'minutes', 5, 'mean')
            df['pace_proxy'] = self._safe_divide(possessions_hist * 48, minutes_hist, 96.0)
        else:
            df['pace_proxy'] = 96.0  # Pace promedio NBA
        
        # 2. PUNTOS EMA 5 JUEGOS (9.48%)
        if 'points' in df.columns:
            pts_ema = df.groupby('Team')['points'].ewm(span=5, adjust=False).mean().shift(1).reset_index(0, drop=True)
            df['pts_ema_5g'] = pts_ema.fillna(df['points'].mean())
        else:
            df['pts_ema_5g'] = 113.0
        
        # 3. PUNTOS PROMEDIO ADAPTATIVO 5G (6.67%)
        if 'points' in df.columns:
            pts_rolling = self._get_historical_series(df, 'points', 5, 'mean')
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            df['pts_adaptive_avg_5g'] = pts_rolling + (pts_std * 0.1)  # Ajuste por volatilidad
        else:
            df['pts_adaptive_avg_5g'] = 113.0
        
        # 4. FUSIÓN MULTI-VENTANA INTELLIGENCE (5.70%)
        if 'effective_fg_pct' in df.columns and 'true_shooting_pct' in df.columns:
            efg_3g = self._get_historical_series(df, 'effective_fg_pct', 3, 'mean')
            efg_10g = self._get_historical_series(df, 'effective_fg_pct', 10, 'mean')
            ts_5g = self._get_historical_series(df, 'true_shooting_pct', 5, 'mean')
            df['fusion_multi_window_intelligence'] = (efg_3g * 0.4 + efg_10g * 0.3 + ts_5g * 0.3)
        else:
            df['fusion_multi_window_intelligence'] = 0.52
        
        # 5. PUNTOS PROMEDIO 5G (5.64%)
        df['pts_avg_5g'] = self._get_historical_series(df, 'points', 5, 'mean') if 'points' in df.columns else 113.0
        
        # 6. EFICIENCIA DE TIRO PROMEDIO 15G (3.65%)
        if 'field_goals_pct' in df.columns:
            df['shooting_efficiency_avg_15g'] = self._get_historical_series(df, 'field_goals_pct', 15, 'mean')
        else:
            df['shooting_efficiency_avg_15g'] = 0.46
        
        # 7. EFFECTIVE FG% 15G (3.20%)
        df['eFG_pct_15g'] = self._get_historical_series(df, 'effective_fg_pct', 15, 'mean') if 'effective_fg_pct' in df.columns else 0.52
        
        # 8. EFFECTIVE FG% ESTABILIZADO 15G (2.83%)
        if 'effective_fg_pct' in df.columns:
            efg_15g = self._get_historical_series(df, 'effective_fg_pct', 15, 'mean')
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 15, 'std')
            df['eFG_pct_stabilized_15g'] = efg_15g * (1 - efg_std * 0.5)  # Penalizar inconsistencia
        else:
            df['eFG_pct_stabilized_15g'] = 0.52
        
        # 9. IS_HOME (2.75%)
        df['is_home'] = df['is_home'].astype(float) if 'is_home' in df.columns else 0.5
        
        # 10. OLIVER MOMENTUM MEGA (2.45%)
        if all(col in df.columns for col in ['offensive_rating', 'defensive_rating']):
            off_rating_3g = self._get_historical_series(df, 'offensive_rating', 3, 'mean')
            def_rating_3g = self._get_historical_series(df, 'defensive_rating', 3, 'mean')
            pace_factor = df.get('pace_proxy', 96.0)
            df['oliver_momentum_mega'] = (off_rating_3g - def_rating_3g) * (pace_factor / 100.0)
        else:
            df['oliver_momentum_mega'] = 0.0
        
        # 11. MODO DESESPERACIÓN (2.14%)
        if 'points' in df.columns:
            pts_recent = self._get_historical_series(df, 'points', 3, 'mean')
            pts_season = self._get_historical_series(df, 'points', 20, 'mean')
            df['desperation_mode'] = np.maximum(0, (pts_season - pts_recent) / 10.0)
        else:
            df['desperation_mode'] = 0.0
        
        # 12. EFECTO EFG AL CUADRADO (1.95%)
        efg_current = df.get('effective_fg_pct', 0.52)
        df['efg_squared_effect'] = (efg_current ** 2) * 2.0
        
        # 14. DEAN OLIVER COMPOSITE (1.83%)
        if all(col in df.columns for col in ['offensive_rating', 'defensive_rating', 'possessions']):
            off_hist = self._get_historical_series(df, 'offensive_rating', 5, 'mean')
            def_hist = self._get_historical_series(df, 'defensive_rating', 5, 'mean')
            pace_hist = self._get_historical_series(df, 'possessions', 5, 'mean')
            df['dean_oliver_composite'] = (off_hist + (115 - def_hist)) * (pace_hist / 100.0)
        else:
            df['dean_oliver_composite'] = 110.0
        
    def _create_explosive_potential_features(self, df: pd.DataFrame) -> None:
        """Features 16-25: Potencial explosivo y eficiencia"""
        
        # 16. POTENCIAL EXPLOSIVO MEJORADO (1.78%)
        if 'points' in df.columns:
            pts_max = self._get_historical_series(df, 'points', 10, 'max')
            pts_avg = self._get_historical_series(df, 'points', 10, 'mean')
            df['explosive_potential_enhanced'] = self._safe_divide(pts_max - pts_avg, pts_avg, 0.0)
        else:
            df['explosive_potential_enhanced'] = 0.0
        
        # 17. CONSISTENCIA EFG-PUNTOS (1.77%)
        if 'effective_fg_pct' in df.columns and 'points' in df.columns:
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 5, 'std')
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            df['efg_pts_consistency'] = 1.0 - (efg_std + pts_std * 0.01)
        else:
            df['efg_pts_consistency'] = 0.8
        
        # 18. ESTIMACIÓN DE PACE (1.75%)
        if 'possessions' in df.columns:
            df['pace_estimate'] = self._get_historical_series(df, 'possessions', 3, 'mean')
        else:
            df['pace_estimate'] = 100.0
        
        # 19. MOMENTUM PUNTOS 3G (1.71%)
        if 'points' in df.columns:
            pts_3g = self._get_historical_series(df, 'points', 3, 'mean')
            pts_10g = self._get_historical_series(df, 'points', 10, 'mean')
            df['pts_momentum_3g'] = pts_3g - pts_10g
        else:
            df['pts_momentum_3g'] = 0.0
        
        # 20. EFICIENCIA POR PUNTO (1.68%)
        if 'efficiency' in df.columns and 'points' in df.columns:
            eff_hist = self._get_historical_series(df, 'efficiency', 5, 'mean')
            pts_hist = self._get_historical_series(df, 'points', 5, 'mean')
            df['efficiency_per_point'] = self._safe_divide(eff_hist, pts_hist, 0.5)
        else:
            df['efficiency_per_point'] = 0.5
        
        # 23. RATIO PACE-EFICIENCIA (1.64%)
        pace_val = df.get('pace_estimate', 100.0)
        if 'offensive_rating' in df.columns:
            off_hist = self._get_historical_series(df, 'offensive_rating', 5, 'mean')
            df['pace_efficiency_ratio'] = pace_val * (off_hist / 110.0)
        else:
            df['pace_efficiency_ratio'] = 100.0
        
        # 24. VOLATILIDAD PUNTOS 5G (1.63%)
        if 'points' in df.columns:
            df['pts_volatility_5g'] = self._get_historical_series(df, 'points', 5, 'std')
        else:
            df['pts_volatility_5g'] = 8.0
        
    def _create_consistency_features(self, df: pd.DataFrame) -> None:
        """Features 26-30: Consistencia y estabilidad"""
        
        # 26. DIFERENCIAL DE PACE (1.60%)
        if 'possessions' in df.columns:
            pace_team = self._get_historical_series(df, 'possessions', 5, 'mean')
            league_avg_pace = 100.0
            df['pace_differential'] = pace_team - league_avg_pace
        else:
            df['pace_differential'] = 0.0
        
        # 27. EFG% ESTABILIZADO (1.59%)
        if 'effective_fg_pct' in df.columns:
            efg_mean = self._get_historical_series(df, 'effective_fg_pct', 10, 'mean')
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 10, 'std')
            df['eFG_pct_stabilized'] = efg_mean * (1 - efg_std)
        else:
            df['eFG_pct_stabilized'] = 0.52
        
        # 29. FACTOR DE PACE AVANZADO (1.50%)
        if 'possessions' in df.columns and 'minutes' in df.columns:
            pace_hist = self._get_historical_series(df, 'possessions', 3, 'mean')
            minutes_hist = self._get_historical_series(df, 'minutes', 3, 'mean')
            df['advanced_pace_factor'] = self._safe_divide(pace_hist * 48, minutes_hist, 100.0)
        else:
            df['advanced_pace_factor'] = 100.0
        
        # 30. EFICIENCIA DE POSESIÓN (1.50%)
        if 'points' in df.columns and 'possessions' in df.columns:
            pts_hist = self._get_historical_series(df, 'points', 5, 'mean')
            poss_hist = self._get_historical_series(df, 'possessions', 5, 'mean')
            df['possession_efficiency'] = self._safe_divide(pts_hist, poss_hist, 1.1)
        else:
            df['possession_efficiency'] = 1.1
    
    def _create_pace_features(self, df: pd.DataFrame) -> None:
        """Features 31-35: Características de ritmo y tempo"""
        
        # 31. MOMENTUM DE PACE (1.48%)
        if 'possessions' in df.columns:
            pace_3g = self._get_historical_series(df, 'possessions', 3, 'mean')
            pace_10g = self._get_historical_series(df, 'possessions', 10, 'mean')
            df['pace_momentum'] = pace_3g - pace_10g
        else:
            df['pace_momentum'] = 0.0
        
        # 32. FACTOR DE IMPREDECIBILIDAD (1.48%)
        if 'points' in df.columns:
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            pts_mean = self._get_historical_series(df, 'points', 5, 'mean')
            df['unpredictability_factor'] = self._safe_divide(pts_std, pts_mean, 0.1)
        else:
            df['unpredictability_factor'] = 0.1
        

        # 34. EXTREMISMO DE PACE (1.45%)
        if 'possessions' in df.columns:
            pace_hist = self._get_historical_series(df, 'possessions', 5, 'mean')
            df['pace_extremeness'] = np.abs(pace_hist - 100.0) / 20.0
        else:
            df['pace_extremeness'] = 0.0
        
    def _create_defensive_features(self, df: pd.DataFrame) -> None:
        """Features 36-37: Características defensivas y de tiro"""
        
        # 36. CONSISTENCIA VS EXPLOSIVIDAD (1.39%)
        if 'points' in df.columns:
            pts_std = self._get_historical_series(df, 'points', 10, 'std')
            pts_max = self._get_historical_series(df, 'points', 10, 'max')
            pts_avg = self._get_historical_series(df, 'points', 10, 'mean')
            explosiveness = self._safe_divide(pts_max - pts_avg, pts_avg, 0.0)
            consistency = 1.0 / (1.0 + pts_std)
            df['consistency_vs_explosiveness'] = consistency * explosiveness
        else:
            df['consistency_vs_explosiveness'] = 0.0
        
        # 37. RACHA CALIENTE DE TIRO (1.29%)
        if 'field_goals_pct' in df.columns:
            fg_3g = self._get_historical_series(df, 'field_goals_pct', 3, 'mean')
            fg_season = self._get_historical_series(df, 'field_goals_pct', 20, 'mean')
            df['shooting_hot_streak'] = np.maximum(0, fg_3g - fg_season)
        else:
            df['shooting_hot_streak'] = 0.0
    
    def _create_momentum_features(self, df: pd.DataFrame) -> None:
        """Features 38-40: Momentum y características adicionales"""
        
        # 38. RACHA FRÍA DE TIRO (0.0% - Placeholder)
        if 'field_goals_pct' in df.columns:
            fg_3g = self._get_historical_series(df, 'field_goals_pct', 3, 'mean')
            fg_season = self._get_historical_series(df, 'field_goals_pct', 20, 'mean')
            df['shooting_cold_streak'] = np.maximum(0, fg_season - fg_3g)
        else:
            df['shooting_cold_streak'] = 0.0
        
        # 39. POTENCIAL EXPLOSIVO BÁSICO (0.0% - Placeholder)
        if 'points' in df.columns:
            pts_recent = self._get_historical_series(df, 'points', 3, 'mean')
            # explosive_potential - ELIMINADO (importancia 0)
            pass
        else:
            pass
        
    def _create_efg_mega_features(self, df: pd.DataFrame) -> None:
        """Features súper predictivas basadas en EFG (79% importancia en el modelo)"""
        
        # 1. EFG MEGA POWER (basado en efg_squared_effect que tiene 79% importancia)
        if 'effective_fg_pct' in df.columns:
            efg_current = df['effective_fg_pct']
            efg_3g = self._get_historical_series(df, 'effective_fg_pct', 3, 'mean')
            efg_10g = self._get_historical_series(df, 'effective_fg_pct', 10, 'mean')
            
            # Combinación súper potente de EFG
            df['efg_mega_power'] = (efg_current ** 3) * (efg_3g ** 2) * (efg_10g ** 1.5)
            
            # EFG momentum súper avanzado
            efg_momentum = efg_3g - efg_10g
            df['efg_mega_momentum'] = efg_momentum * (efg_current ** 2)
            
            # EFG explosividad (cuando está en racha)
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 5, 'std')
            df['efg_explosive_power'] = np.where(efg_momentum > 0.05, efg_current ** 4, efg_current ** 2)
            
            # EFG estabilidad premium
            df['efg_premium_stability'] = efg_current * (1 - efg_std) * 2.0
            
        else:
            df['efg_mega_power'] = 0.5
            df['efg_mega_momentum'] = 0.0
            df['efg_explosive_power'] = 0.5
            df['efg_premium_stability'] = 0.5
        
        # 2. TRUE SHOOTING MEGA (combinando con EFG)
        if 'true_shooting_pct' in df.columns and 'effective_fg_pct' in df.columns:
            ts_3g = self._get_historical_series(df, 'true_shooting_pct', 3, 'mean')
            efg_3g = self._get_historical_series(df, 'effective_fg_pct', 3, 'mean')
            
            # Súper combinación de eficiencia
            df['ts_efg_mega_fusion'] = (ts_3g ** 2) * (efg_3g ** 2) * 4.0
            
            # Diferencial de eficiencia premium
            df['efficiency_premium_differential'] = (ts_3g - efg_3g) * 10.0
            
        else:
            df['ts_efg_mega_fusion'] = 0.5
            df['efficiency_premium_differential'] = 0.0
        
    def _create_pace_mega_features(self, df: pd.DataFrame) -> None:
        """Features súper predictivas basadas en PACE (múltiples features importantes)"""
        
        # 1. PACE MEGA INTELLIGENCE (combinando todas las features de pace)
        if 'possessions' in df.columns:
            poss_3g = self._get_historical_series(df, 'possessions', 3, 'mean')
            poss_10g = self._get_historical_series(df, 'possessions', 10, 'mean')
            poss_std = self._get_historical_series(df, 'possessions', 5, 'std')
            
            # Pace mega power (combinación súper avanzada)
            df['pace_mega_power'] = (poss_3g ** 2) * (1 + poss_std / 10) * 0.1
            
            # Pace explosividad
            pace_momentum = poss_3g - poss_10g
            df['pace_explosive_factor'] = pace_momentum * (poss_3g / 100) ** 2
            
            # Pace estabilidad premium
            df['pace_premium_stability'] = poss_3g * (1 - poss_std / poss_3g) * 2.0
            
        else:
            df['pace_mega_power'] = 1.0
            df['pace_explosive_factor'] = 0.0
            df['pace_premium_stability'] = 1.0
        
        # 2. PACE VS PUNTOS MEGA CORRELATION
        if 'possessions' in df.columns and 'points' in df.columns:
            poss_5g = self._get_historical_series(df, 'possessions', 5, 'mean')
            pts_5g = self._get_historical_series(df, 'points', 5, 'mean')
            
            # Correlación pace-puntos súper avanzada
            df['pace_points_mega_correlation'] = (poss_5g * pts_5g) / 10000
            
            # Eficiencia por posesión mega
            df['points_per_possession_mega'] = pts_5g / poss_5g * 100
            
        else:
            df['pace_points_mega_correlation'] = 1.0
            df['points_per_possession_mega'] = 1.1
        
        # 3. PACE MOMENTUM MEGA
        if 'possessions' in df.columns:
            poss_1g = self._get_historical_series(df, 'possessions', 1, 'mean')
            poss_3g = self._get_historical_series(df, 'possessions', 3, 'mean')
            poss_7g = self._get_historical_series(df, 'possessions', 7, 'mean')
            
            # Momentum súper avanzado (ponderado)
            df['pace_momentum_mega'] = (poss_1g * 0.5 + poss_3g * 0.3 + poss_7g * 0.2) * 2.0
            
            # Aceleración de pace
            df['pace_acceleration'] = (poss_1g - poss_3g) * (poss_3g - poss_7g) * 0.1
            
        else:
            df['pace_momentum_mega'] = 1.0
            df['pace_acceleration'] = 0.0
            
    def _create_points_mega_features(self, df: pd.DataFrame) -> None:
        """Features súper predictivas basadas en PUNTOS HISTÓRICOS"""
        
        # 1. PUNTOS MEGA MOMENTUM (combinando todas las features de puntos)
        if 'points' in df.columns:
            pts_1g = self._get_historical_series(df, 'points', 1, 'mean')
            pts_3g = self._get_historical_series(df, 'points', 3, 'mean')
            pts_5g = self._get_historical_series(df, 'points', 5, 'mean')
            pts_10g = self._get_historical_series(df, 'points', 10, 'mean')
            
            # Mega momentum ponderado
            df['points_mega_momentum'] = (pts_1g * 0.4 + pts_3g * 0.3 + pts_5g * 0.2 + pts_10g * 0.1) * 0.1
            
            # Aceleración de puntos
            df['points_acceleration'] = (pts_1g - pts_3g) * (pts_3g - pts_5g) * 0.01
            
            # Volatilidad de puntos premium
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            df['points_volatility_premium'] = pts_std * (pts_5g / 100) ** 2
            
        else:
            df['points_mega_momentum'] = 1.13
            df['points_acceleration'] = 0.0
            df['points_volatility_premium'] = 0.8
        
        # 2. PUNTOS VS EFICIENCIA MEGA
        if 'points' in df.columns and 'effective_fg_pct' in df.columns:
            pts_5g = self._get_historical_series(df, 'points', 5, 'mean')
            efg_5g = self._get_historical_series(df, 'effective_fg_pct', 5, 'mean')
            
            # Súper correlación puntos-eficiencia
            df['points_efficiency_mega_correlation'] = (pts_5g * efg_5g) / 1000
            
            # Eficiencia por punto mega
            df['efficiency_per_point_mega'] = efg_5g / pts_5g * 1000
            
        else:
            df['points_efficiency_mega_correlation'] = 5.0
            df['efficiency_per_point_mega'] = 0.5
        
        # 3. PUNTOS EXPLOSIVIDAD MEGA
        if 'points' in df.columns:
            pts_max = self._get_historical_series(df, 'points', 10, 'max')
            pts_avg = self._get_historical_series(df, 'points', 10, 'mean')
            pts_min = self._get_historical_series(df, 'points', 10, 'min')
            
            # Rango explosivo
            df['points_explosive_range'] = (pts_max - pts_min) * 0.1
            
            # Consistencia premium
            df['points_consistency_premium'] = 1.0 / (1.0 + (pts_max - pts_min) / pts_avg)
            
        else:
            df['points_explosive_range'] = 1.0
            df['points_consistency_premium'] = 0.5
            
    def _create_quarters_advanced_features(self, df: pd.DataFrame) -> None:
        """Features súper predictivas usando datos por cuartos"""
        
        # Solo si tenemos datos de cuartos
        if self.teams_quarters_df is not None and len(self.teams_quarters_df) > 0:
            # 1. ANÁLISIS AVANZADO DE CUARTOS POR EQUIPO
            quarters_df = self.teams_quarters_df.copy()
            
            # Asegurar que las fechas estén en el mismo formato
            quarters_df['Date'] = pd.to_datetime(quarters_df['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            quarters_analysis = quarters_df.groupby(['Team', 'Date']).agg({
                'points': ['mean', 'std', 'max', 'min', 'sum'],
                'effective_fg_pct': ['mean', 'std', 'max'],
                'possessions': ['mean', 'std'],
                'field_goals_pct': ['mean', 'std'],
                'three_points_pct': ['mean', 'std'],
                'free_throws_pct': ['mean', 'std'],
                'quarter': 'count'  # Número de cuartos jugados
            }).reset_index()
            
            # Flatten column names
            quarters_analysis.columns = [
                'Team', 'Date', 'qtr_pts_mean', 'qtr_pts_std', 'qtr_pts_max', 'qtr_pts_min', 'qtr_pts_sum',
                'qtr_efg_mean', 'qtr_efg_std', 'qtr_efg_max',
                'qtr_poss_mean', 'qtr_poss_std',
                'qtr_fg_mean', 'qtr_fg_std',
                'qtr_3p_mean', 'qtr_3p_std',
                'qtr_ft_mean', 'qtr_ft_std',
                'qtr_count'
            ]
            
            # Merge con datos principales
            df = df.merge(quarters_analysis, on=['Team', 'Date'], how='left')
            
            # 2. FEATURES DE CUARTOS SÚPER AVANZADAS
            if 'qtr_pts_mean' in df.columns:
                # Consistencia por cuartos (inversa de la desviación)
                df['quarters_consistency'] = 1.0 / (1.0 + df['qtr_pts_std'].fillna(0))
                
                # Explosividad por cuartos (rango de puntos)
                df['quarters_explosiveness'] = (df['qtr_pts_max'] - df['qtr_pts_min']).fillna(0) * 0.1
                
                # Eficiencia por cuartos (EFG promedio)
                df['quarters_efficiency'] = df['qtr_efg_mean'].fillna(0.5) * 2.0
                
                # Pace por cuartos
                df['quarters_pace'] = df['qtr_poss_mean'].fillna(100) * 0.1
                
                # NUEVAS FEATURES DE CUARTOS
                # Variabilidad de eficiencia por cuartos
                df['quarters_efg_volatility'] = df['qtr_efg_std'].fillna(0) * 10.0
                
                # Máxima eficiencia en un cuarto
                df['quarters_efg_peak'] = df['qtr_efg_max'].fillna(0.5) * 2.0
                
                # Consistencia de pace por cuartos
                df['quarters_pace_consistency'] = 1.0 / (1.0 + df['qtr_poss_std'].fillna(0))
                
                # Balance de tiros por cuartos
                df['quarters_shooting_balance'] = (
                    df['qtr_fg_mean'].fillna(0.5) + 
                    df['qtr_3p_mean'].fillna(0.3) + 
                    df['qtr_ft_mean'].fillna(0.8)
                ) / 3.0
                
                # Volatilidad de tiros por cuartos
                df['quarters_shooting_volatility'] = (
                    df['qtr_fg_std'].fillna(0) + 
                    df['qtr_3p_std'].fillna(0) + 
                    df['qtr_ft_std'].fillna(0)
                ) / 3.0 * 10.0
                
                # Patrón de cuartos (si juega 4 cuartos completos)
                df['quarters_completeness'] = (df['qtr_count'].fillna(4) / 4.0) ** 2
                
            else:
                # Valores por defecto
                df['quarters_consistency'] = 0.5
                df['quarters_explosiveness'] = 0.0
                df['quarters_efficiency'] = 1.0
                df['quarters_pace'] = 1.0
                df['quarters_efg_volatility'] = 0.0
                df['quarters_efg_peak'] = 1.0
                df['quarters_pace_consistency'] = 0.5
                df['quarters_shooting_balance'] = 0.5
                df['quarters_shooting_volatility'] = 0.0
                df['quarters_completeness'] = 1.0
        else:
            # Valores por defecto si no hay datos de cuartos
            df['quarters_consistency'] = 0.5
            df['quarters_explosiveness'] = 0.0
            df['quarters_efficiency'] = 1.0
            df['quarters_pace'] = 1.0
            df['quarters_efg_volatility'] = 0.0
            df['quarters_efg_peak'] = 1.0
            df['quarters_pace_consistency'] = 0.5
            df['quarters_shooting_balance'] = 0.5
            df['quarters_shooting_volatility'] = 0.0
            df['quarters_completeness'] = 1.0
            
    def _create_ratings_mega_features(self, df: pd.DataFrame) -> None:
        """Features súper predictivas usando offensive/defensive ratings"""
        
        # 1. RATINGS MEGA COMBINATION
        if all(col in df.columns for col in ['offensive_rating', 'defensive_rating']):
            off_3g = self._get_historical_series(df, 'offensive_rating', 3, 'mean')
            def_3g = self._get_historical_series(df, 'defensive_rating', 3, 'mean')
            
            # Net rating mega
            df['net_rating_mega'] = (off_3g - def_3g) * 0.1
            
            # Offensive rating power
            df['offensive_rating_power'] = (off_3g / 110) ** 2
            
            # Defensive rating strength
            df['defensive_rating_strength'] = (115 - def_3g) / 115
            
        else:
            df['net_rating_mega'] = 0.0
            df['offensive_rating_power'] = 1.0
            df['defensive_rating_strength'] = 1.0
        
        # 2. RATINGS MOMENTUM MEGA
        if all(col in df.columns for col in ['offensive_rating', 'defensive_rating']):
            off_1g = self._get_historical_series(df, 'offensive_rating', 1, 'mean')
            off_5g = self._get_historical_series(df, 'offensive_rating', 5, 'mean')
            def_1g = self._get_historical_series(df, 'defensive_rating', 1, 'mean')
            def_5g = self._get_historical_series(df, 'defensive_rating', 5, 'mean')
            
            # Momentum ofensivo
            df['offensive_momentum_mega'] = (off_1g - off_5g) * 0.1
            
            # Momentum defensivo (invertido porque menor es mejor)
            df['defensive_momentum_mega'] = (def_5g - def_1g) * 0.1
            
            # Momentum neto
            df['net_momentum_mega'] = df['offensive_momentum_mega'] + df['defensive_momentum_mega']
            
        else:
            df['offensive_momentum_mega'] = 0.0
            df['defensive_momentum_mega'] = 0.0
            df['net_momentum_mega'] = 0.0
            
    def _clean_infinite_values(self, df: pd.DataFrame) -> None:
        """Limpia valores infinitos y NaN de todas las features numéricas"""
        
        # Obtener todas las columnas numéricas
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if col in df.columns:
                # Reemplazar infinitos con NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                
                # Si la columna tiene variación, usar percentiles para clipping
                if len(df[col].dropna().unique()) > 1:
                    q01 = df[col].quantile(0.01)
                    q99 = df[col].quantile(0.99)
                    range_val = q99 - q01
                    
                    # Clipping robusto
                    upper_limit = q99 + range_val * 0.5
                    lower_limit = q01 - range_val * 0.5
                    
                    df[col] = df[col].clip(lower=lower_limit, upper=upper_limit)
                
                # Rellenar NaN restantes con la mediana
                median_val = df[col].median()
                if not np.isnan(median_val):
                    df[col] = df[col].fillna(median_val)
                else:
                    df[col] = df[col].fillna(0.0)
        
    
    def _create_extreme_detection_features(self, df: pd.DataFrame) -> None:
        """Features súper avanzadas para detectar y predecir extremos"""
        
        # 1. DETECCIÓN DE EXTREMOS HISTÓRICOS
        if 'points' in df.columns:
            pts_10g = self._get_historical_series(df, 'points', 10, 'mean')
            pts_std = self._get_historical_series(df, 'points', 10, 'std')
            pts_max = self._get_historical_series(df, 'points', 10, 'max')
            pts_min = self._get_historical_series(df, 'points', 10, 'min')
            
            # Z-score para detectar extremos
            df['points_zscore_extreme'] = np.abs((pts_10g - 113.0) / (pts_std + 1e-6))
            
            # Rango intercuartílico para extremos
            q75 = self._get_historical_series(df, 'points', 10, lambda x: x.quantile(0.75))
            q25 = self._get_historical_series(df, 'points', 10, lambda x: x.quantile(0.25))
            iqr = q75 - q25

            # Potencial de explosión (cuando está cerca del máximo histórico)
            df['points_explosion_potential'] = pts_10g / (pts_max + 1e-6)
            
        else:
            df['points_zscore_extreme'] = 0.0
            df['points_explosion_potential'] = 0.5
    
    def _create_volatility_mega_features(self, df: pd.DataFrame) -> None:
        """Features súper avanzadas de volatilidad para capturar extremos"""
        
        # 1. VOLATILIDAD MEGA COMBINADA
        if 'points' in df.columns and 'effective_fg_pct' in df.columns:
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 5, 'std')
            
            # Volatilidad mega combinada
            df['volatility_mega_combined'] = (pts_std * 0.1 + efg_std * 10) ** 2
            
            # Ratio de volatilidad
            df['volatility_ratio_mega'] = pts_std / (efg_std + 1e-6)
            
        else:
            df['volatility_mega_combined'] = 0.0
            df['volatility_ratio_mega'] = 1.0
        
        # 2. VOLATILIDAD TEMPORAL MEGA
        if 'points' in df.columns:
            pts_3g = self._get_historical_series(df, 'points', 3, 'std')
            pts_7g = self._get_historical_series(df, 'points', 7, 'std')
            pts_15g = self._get_historical_series(df, 'points', 15, 'std')
            
            # Tendencia de volatilidad
            df['volatility_trend_mega'] = (pts_3g - pts_7g) * (pts_7g - pts_15g) * 0.1
            
            # Aceleración de volatilidad
            df['volatility_acceleration'] = pts_3g - 2 * pts_7g + pts_15g
            
            # Volatilidad estabilizada
            df['volatility_stabilized'] = 1.0 / (1.0 + pts_3g)
            
        else:
            df['volatility_trend_mega'] = 0.0
            df['volatility_acceleration'] = 0.0
            df['volatility_stabilized'] = 0.5
        
        # 3. VOLATILIDAD DE RATINGS
        if all(col in df.columns for col in ['offensive_rating', 'defensive_rating']):
            off_std = self._get_historical_series(df, 'offensive_rating', 5, 'std')
            def_std = self._get_historical_series(df, 'defensive_rating', 5, 'std')
            
            # Volatilidad de ratings combinada
            df['ratings_volatility_mega'] = (off_std + def_std) * 0.1
            
        else:
            df['ratings_volatility_mega'] = 0.0
        
    
    def _create_outlier_prediction_features(self, df: pd.DataFrame) -> None:
        """Features súper avanzadas para predecir outliers y extremos"""
        
        # 1. PREDICCIÓN DE OUTLIERS DE PUNTOS
        if 'points' in df.columns:
            pts_5g = self._get_historical_series(df, 'points', 5, 'mean')
            pts_std = self._get_historical_series(df, 'points', 5, 'std')
            pts_max = self._get_historical_series(df, 'points', 10, 'max')
            pts_min = self._get_historical_series(df, 'points', 10, 'min')
            
            # Score de outlier (combinado)
            df['outlier_score'] = (pts_5g - 113.0) ** 2 / (pts_std ** 2 + 1e-6)

        else:
            df['outlier_score'] = 0.0
        
        # 2. PREDICCIÓN DE OUTLIERS DE EFICIENCIA
        if 'effective_fg_pct' in df.columns:
            efg_5g = self._get_historical_series(df, 'effective_fg_pct', 5, 'mean')
            efg_std = self._get_historical_series(df, 'effective_fg_pct', 5, 'std')
            
            # Score de outlier de eficiencia
            df['efg_outlier_score'] = (efg_5g - 0.52) ** 2 / (efg_std ** 2 + 1e-6)
            
        else:
            df['efg_outlier_score'] = 0.0
        
        # 3. PREDICCIÓN DE OUTLIERS DE PACE
        if 'possessions' in df.columns:
            poss_5g = self._get_historical_series(df, 'possessions', 5, 'mean')
            poss_std = self._get_historical_series(df, 'possessions', 5, 'std')
            
            # Score de outlier de pace
            df['pace_outlier_score'] = (poss_5g - 100.0) ** 2 / (poss_std ** 2 + 1e-6)
            
        else:
            df['pace_outlier_score'] = 0.0
    
    def _create_halftime_specific_features(self, df: pd.DataFrame) -> None:
        """Features específicas para predicción de halftime usando datos de cuartos"""
        
        if self.teams_quarters_df is not None and len(self.teams_quarters_df) > 0:
            # Crear features específicas para halftime basadas en cuartos 1 y 2
            quarters_df = self.teams_quarters_df.copy()
            quarters_df['Date'] = pd.to_datetime(quarters_df['Date'])
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 1. ANÁLISIS ESPECÍFICO DE PRIMERA MITAD (CUARTOS 1 Y 2)
            first_half_df = quarters_df[quarters_df['quarter'].isin([1, 2])].copy()
            
            if len(first_half_df) > 0:
                # Agrupar por equipo y fecha para analizar primera mitad
                first_half_analysis = first_half_df.groupby(['Team', 'Date']).agg({
                    'points': ['sum', 'mean', 'std', 'max', 'min'],
                    'effective_fg_pct': ['mean', 'std', 'max'],
                    'possessions': ['sum', 'mean', 'std'],
                    'field_goals_pct': ['mean', 'std'],
                    'three_points_pct': ['mean', 'std'],
                    'free_throws_pct': ['mean', 'std'],
                    'assists': ['sum', 'mean'],
                    'rebounds': ['sum', 'mean'],
                    'turnovers': ['sum', 'mean'],
                    'steals': ['sum', 'mean'],
                    'blocks': ['sum', 'mean']
                }).reset_index()
                
                # Flatten column names
                first_half_analysis.columns = [
                    'Team', 'Date', 'fh_pts_sum', 'fh_pts_mean', 'fh_pts_std', 'fh_pts_max', 'fh_pts_min',
                    'fh_efg_mean', 'fh_efg_std', 'fh_efg_max',
                    'fh_poss_sum', 'fh_poss_mean', 'fh_poss_std',
                    'fh_fg_mean', 'fh_fg_std',
                    'fh_3p_mean', 'fh_3p_std',
                    'fh_ft_mean', 'fh_ft_std',
                    'fh_ast_sum', 'fh_ast_mean',
                    'fh_reb_sum', 'fh_reb_mean',
                    'fh_tov_sum', 'fh_tov_mean',
                    'fh_stl_sum', 'fh_stl_mean',
                    'fh_blk_sum', 'fh_blk_mean'
                ]
                
                # Merge con datos principales
                df = df.merge(first_half_analysis, on=['Team', 'Date'], how='left')
                
                # 2. FEATURES ESPECÍFICAS DE HALFTIME
                if 'fh_pts_sum' in df.columns:
                    # Consistencia en primera mitad
                    df['halftime_consistency'] = 1.0 / (1.0 + df['fh_pts_std'].fillna(0))
                    
                    # Explosividad en primera mitad
                    df['halftime_explosiveness'] = (df['fh_pts_max'] - df['fh_pts_min']).fillna(0) * 0.5
                    
                    # Eficiencia en primera mitad
                    df['halftime_efficiency'] = df['fh_efg_mean'].fillna(0.5) * 2.0
                    
                    # Pace en primera mitad
                    df['halftime_pace'] = df['fh_poss_mean'].fillna(100) * 0.1
                    
                    # Balance de tiros en primera mitad
                    df['halftime_shooting_balance'] = (
                        df['fh_fg_mean'].fillna(0.5) + 
                        df['fh_3p_mean'].fillna(0.3) + 
                        df['fh_ft_mean'].fillna(0.8)
                    ) / 3.0
                    
                    # Volatilidad de tiros en primera mitad
                    df['halftime_shooting_volatility'] = (
                        df['fh_fg_std'].fillna(0) + 
                        df['fh_3p_std'].fillna(0) + 
                        df['fh_ft_std'].fillna(0)
                    ) / 3.0 * 10.0
                    
                    # Asistencias por posesión en primera mitad
                    df['halftime_ast_per_poss'] = (df['fh_ast_sum'] / df['fh_poss_sum']).fillna(0) * 100
                    
                    # Rebotes por posesión en primera mitad
                    df['halftime_reb_per_poss'] = (df['fh_reb_sum'] / df['fh_poss_sum']).fillna(0) * 100
                    
                    # Ratio de turnovers en primera mitad
                    df['halftime_tov_ratio'] = (df['fh_tov_sum'] / df['fh_poss_sum']).fillna(0) * 100
                    
                    # Eficiencia defensiva en primera mitad
                    df['halftime_def_efficiency'] = (
                        df['fh_stl_sum'] + df['fh_blk_sum']
                    ) / df['fh_poss_sum'].fillna(1) * 100
                    
                    # 3. FEATURES HISTÓRICAS DE HALFTIME
                    # Promedio histórico de puntos en primera mitad
                    df['halftime_historical_avg'] = df.groupby('Team')['fh_pts_sum'].transform(
                        lambda x: x.expanding().mean().shift(1)
                    ).fillna(df['fh_pts_sum'].mean())
                    
                    # Tendencia de puntos en primera mitad (últimos 5 juegos)
                    df['halftime_trend'] = df.groupby('Team')['fh_pts_sum'].transform(
                        lambda x: x.rolling(5, min_periods=1).mean().shift(1)
                    ).fillna(df['fh_pts_sum'].mean())
                    
                    # Volatilidad histórica en primera mitad
                    df['halftime_historical_volatility'] = df.groupby('Team')['fh_pts_sum'].transform(
                        lambda x: x.expanding().std().shift(1)
                    ).fillna(df['fh_pts_sum'].std())
                    
                    # 4. FEATURES DE COMPARACIÓN CON SEGUNDA MITAD
                    # Obtener datos de segunda mitad (cuartos 3 y 4)
                    second_half_df = quarters_df[quarters_df['quarter'].isin([3, 4])].copy()
                    
                    if len(second_half_df) > 0:
                        second_half_analysis = second_half_df.groupby(['Team', 'Date']).agg({
                            'points': 'sum'
                        }).reset_index()
                        second_half_analysis.columns = ['Team', 'Date', 'sh_pts_sum']
                        
                        # Merge con datos principales
                        df = df.merge(second_half_analysis, on=['Team', 'Date'], how='left')
                        
                        # Ratio primera mitad vs segunda mitad
                        df['halftime_second_half_ratio'] = (
                            df['fh_pts_sum'] / df['sh_pts_sum'].fillna(df['fh_pts_sum'])
                        ).fillna(1.0)
                        
                        # Diferencia entre mitades
                        df['halftime_half_difference'] = (
                            df['fh_pts_sum'] - df['sh_pts_sum'].fillna(df['fh_pts_sum'])
                        ).fillna(0)
                    
                    # Registrar features
                    halftime_features = [
                        'halftime_consistency', 'halftime_explosiveness', 'halftime_efficiency',
                        'halftime_pace', 'halftime_shooting_balance', 'halftime_shooting_volatility',
                        'halftime_ast_per_poss', 'halftime_reb_per_poss', 'halftime_tov_ratio',
                        'halftime_def_efficiency', 'halftime_historical_avg', 'halftime_trend',
                        'halftime_historical_volatility', 'halftime_second_half_ratio', 'halftime_half_difference'
                    ]
                    
                    for feature in halftime_features:
                        if feature in df.columns:
                            self._register_feature(feature, 'halftime_specific')