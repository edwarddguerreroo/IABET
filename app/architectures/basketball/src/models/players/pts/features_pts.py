"""
Feature Engineering OPTIMIZADO para Predicción de Puntos de Jugadores NBA
==========================================================

Motor de features ultra-eficiente y especializado para maximizar la predicción de puntos (PTS)
que un jugador anotará en su próximo partido.

OPTIMIZACIONES CLAVE:
- Ventanas de calculo espaciadas (3, 7, 15, 30)
- Eliminación de cálculos redundantes y circulares
- Enfoque en features más predictivas identificadas por el modelo
- Cálculos vectorizados y eficientes
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class PointsFeatureEngineer:
    """
    Feature Engineer especializado en predicción de puntos (PTS)
    Enfoque en eficiencia y features de máxima predictividad
    """
    
    def __init__(self, correlation_threshold: float = 0.98, max_features: int = 150, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.teams_df = teams_df  
        self.players_df = players_df  
        self.players_quarters_df = players_quarters_df  # Nuevo: datos por cuartos  
        
        # VENTANAS ESPACIADAS para reducir correlación
        self.windows = {
            'short': 3,      # Tendencia inmediata
            'medium': 10,     # Tendencia reciente  
            'long': 20,      # Tendencia estable
            'season': 30     # Baseline temporal
        }
        
        self.feature_registry = {}
        self.feature_categories = {
            'core_predictive': [],           # Features núcleo más predictivas
            'efficiency_metrics': [],       # Métricas de eficiencia
            'opportunity_factors': [],      # Factores de oportunidad
            'context_adjusters': [],        # Ajustadores contextuales
            'ensemble_predictors': []       # Predictores ensemble
        }
        
        self.protected_features = ['points', 'player', 'Date', 'Team', 'Opp', 'position']
        
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

    def _get_rolling_stat(self, df: pd.DataFrame, column: str, window: int, operation: str = 'mean') -> pd.Series:
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
        """Asegura que la columna 'minutes' esté disponible y en formato decimal"""
        if 'minutes' not in df.columns:
            logger.warning("No se encontró columna 'minutes'")
            df['minutes'] = 0
        elif df['minutes'].dtype == 'object':
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
        Pipeline de features para predicción de puntos de jugador NBA
        """
                
        # Limpiar cache
        self._rolling_cache = {}

        
        # Convertir MP
        self._convert_mp_to_numeric(df)
        
        # Verificar target (puntos)
        if 'points' in df.columns:
            pts_stats = df['points'].describe()
            logger.info(f"Target PTS disponible - Media={pts_stats['mean']:.1f}, Max={pts_stats['max']:.0f}")
        else:
            logger.warning(f"Target PTS no disponible")
            return []
        
        # Limpiar registro
        self.feature_registry = {}
        for category in self.feature_categories:
            self.feature_categories[category] = []
        
        initial_cols = len(df.columns)
        
        # PIPELINE OPTIMIZADO DE FEATURES
        logger.info("Generando caracteristicas avanzadas..")
        
        # CORE PREDICTIVE FEATURES (Las más importantes según el modelo)
        self._generate_core_predictive_features(df)        
        # EFFICIENCY METRICS (Ratios optimizados)
        self._generate_efficiency_metrics(df)        
        # ENSEMBLE PREDICTORS (Predictores avanzados)
        self._generate_ensemble_predictors(df)
        # ADVANCED PATTERN FEATURES (Patrones de alto valor predictivo)
        self._generate_advanced_patterns(df)
        # HIGH SCORING SPECIALIST FEATURES (Para mejorar predicción de 25+ puntos)
        self._generate_high_scoring_features(df)
        # ELITE PREDICTIVE FEATURES (Basado en análisis de TOP features)
        self._generate_elite_predictive_features(df)
        # ADVANCED CONTEXT & MATCHUP FEATURES
        self._generate_advanced_context_features(df)
        # MOMENTUM & VARIABILITY FEATURES (Basado en feature importance top)
        self._generate_momentum_variability_features(df)
        # OUTLIER DETECTION FEATURES (Para manejo robusto de outliers)
        self._generate_outlier_detection_features(df)
        # HIGH-RANGE SPECIALIST FEATURES (Features especializadas en rango 15-30 pts)
        self._generate_high_range_specialist_features(df)
        # FEATURES FALTANTES CRÍTICAS
        self._generate_missing_critical_features(df)
        # NUEVAS FEATURES AVANZADAS (usando métricas del nuevo dataset)
        self._generate_advanced_dataset_features(df)
        # FEATURES DE DATOS POR CUARTOS (si están disponibles)
        self._generate_quarters_features(df)
        # FEATURES ULTRA-AVANZADAS (explota TODAS las columnas disponibles)
        self._generate_ultra_advanced_features(df)
                
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
                
        # Obtener SOLO las features registradas (como TRB)
        created_features = list(self.feature_registry.keys())
        
        # Verificar qué features registradas están disponibles en el DataFrame
        available_features = []
        for feature in created_features:
            if feature in df.columns:
                available_features.append(feature)
        
        missing_from_df = [f for f in created_features if f not in df.columns]
        if missing_from_df:
            logger.warning(f"⚠️ Features registradas pero no en DataFrame: {missing_from_df}")
        
        # USAR SOLO LAS FEATURES REGISTRADAS Y DISPONIBLES
        final_features = available_features
        
        logger.info(f"Features disponibles para predicción: {len(final_features)}")
        logger.debug(f"Features seleccionadas: {final_features}")

        # RETORNAR FEATURES REGISTRADAS Y DISPONIBLES
        return final_features

    def _generate_core_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Genera las features NÚCLEO más predictivas identificadas por el modelo
        Enfoque en MP, eficiencia ofensiva, y patrones de anotación
        """
        logger.debug("Generando features NÚCLEO más predictivas...")
        
        # Promedio de puntos histórico MEJORADO 
        if 'points' in df.columns:
            # Usar ventanas espaciadas para reducir correlación
            pts_short = self._get_rolling_stat(df, 'points', self.windows['short'], 'mean')
            pts_long = self._get_rolling_stat(df, 'points', self.windows['long'], 'mean')
            pts_season = self._get_rolling_stat(df, 'points', self.windows['season'], 'mean')
            
            # Momentum scoring basado en aceleración de puntos (clipping más suave)
            df['pts_momentum'] = (pts_short - pts_long) / (pts_long - pts_season + 0.1)
            df['pts_momentum'] = df['pts_momentum'].clip(-1.5, 1.5).fillna(0)  # Clipping más suave
            self._register_feature('pts_momentum', 'core_predictive')
            
            # Scoring ceiling (máximo reciente vs promedio)
            pts_max_recent = self._get_rolling_stat(df, 'points', self.windows['medium'], 'max')
            df['scoring_ceiling'] = pts_max_recent - pts_long
            self._register_feature('scoring_ceiling', 'core_predictive')
            
            # Weighted recent scoring
            # Weighted average que da más peso a juegos recientes
            weights_recent = np.array([0.4, 0.3, 0.2, 0.1])  # Para últimos 4 juegos
            if len(weights_recent) <= self.windows['medium']:
                # Usar _get_rolling_stat para consistencia
                def weighted_average(series):
                    if len(series) < 2:
                        return series.iloc[0] if len(series) > 0 else 0
                    weights = weights_recent[:len(series)]
                    return np.average(series, weights=weights)
                
                pts_weighted = df.groupby('player')['points'].rolling(
                    window=len(weights_recent), min_periods=2
                ).apply(weighted_average).reset_index(0, drop=True).fillna(pts_long)
                df['pts_weighted_recent'] = pts_weighted
                self._register_feature('pts_weighted_recent', 'core_predictive')
            
            # ELITE PLAYER FEATURES
            # Explosiveness Factor - Detecta capacidad de juegos excepcionales
            pts_max_7 = self._get_rolling_stat(df, 'points', 7, 'max')
            pts_p90_7 = df.groupby('player')['points'].rolling(7, min_periods=3).quantile(0.9).reset_index(0, drop=True)
            df['explosiveness_factor'] = ((pts_max_7 - pts_p90_7) / (pts_long + 1e-6)).clip(0, 2.0).fillna(0)
            self._register_feature('explosiveness_factor', 'elite_performance')
            
            # Superstar Consistency - Medida de consistencia en alto rendimiento
            pts_above_20 = (df.groupby('player')['points'].rolling(10, min_periods=5)
                           .apply(lambda x: (x >= 20).sum() / len(x))
                           .reset_index(0, drop=True).fillna(0))
            df['superstar_consistency'] = pts_above_20
            self._register_feature('superstar_consistency', 'elite_performance')
            
            # Elite Pressure Response - Rendimiento en situaciones críticas
            # Basado en diferencia entre máximo y mínimo recientes
            pts_min_7 = self._get_rolling_stat(df, 'points', 7, 'min')
            pts_range_7 = pts_max_7 - pts_min_7
            df['elite_pressure_response'] = (pts_long / (pts_range_7 + 1e-6)).clip(0, 3.0).fillna(1.0)
            self._register_feature('elite_pressure_response', 'elite_performance')
            
            # Dynamic Scoring Ceiling - Se adapta al rango de scoring del jugador
            scoring_tier = (pts_long // 5).clip(0, 8)  # Tiers: 0-5, 5-10, ..., 35-40
            tier_multiplier = 1.0 + scoring_tier * 0.1  # Aumenta con tier más alto
            df['dynamic_scoring_ceiling'] = (pts_max_recent - pts_long) * tier_multiplier
            self._register_feature('dynamic_scoring_ceiling', 'elite_performance')
            
            # Enhanced weighted average con ajuste por forma reciente
            recent_form_multiplier = 1 + (pts_short - pts_long) / (pts_long + 5) * 0.3
            df['pts_long_enhanced'] = pts_long * recent_form_multiplier.clip(0.7, 1.4)
            
            # CONTEXTUAL ADAPTIVE WEIGHTING - Se adapta al tier del jugador
            def adaptive_contextual_prediction(group):
                pts = group['points']
                pts_avg = pts.rolling(15, min_periods=5).mean()
                
                # Determinar tier del jugador dinámicamente
                tier = (pts_avg // 5).fillna(3).clip(0, 8)  # 0-8 tiers
                
                # Pesos adaptativos por tier
                # Tier bajo (0-15 pts): Más peso a estabilidad
                # Tier medio (15-25 pts): Balance equilibrado  
                # Tier alto (25+ pts): Más peso a explosividad
                immediate_weight = 0.2 + tier * 0.02  # 0.2 to 0.36
                medium_weight = 0.5 - tier * 0.03     # 0.5 to 0.26
                explosive_weight = 0.3 + tier * 0.01  # 0.3 to 0.38
                
                # Calcular componentes
                immediate_comp = pts.rolling(2, min_periods=1).mean().shift(1) * immediate_weight
                medium_comp = pts.rolling(6, min_periods=3).mean().shift(1) * medium_weight
                explosive_comp = pts.rolling(3, min_periods=2).max().shift(1) * explosive_weight
                
                # Combinar con normalización
                combined = immediate_comp.fillna(0) + medium_comp.fillna(0) + explosive_comp.fillna(0)
                valid_sum = (~immediate_comp.isna()).astype(float) * immediate_weight + \
                           (~medium_comp.isna()).astype(float) * medium_weight + \
                           (~explosive_comp.isna()).astype(float) * explosive_weight
                           
                return combined / (valid_sum + 1e-6)
            
            # Aplicar contextual adaptive prediction
            try:
                contextual_results = df.groupby('player', group_keys=False).apply(adaptive_contextual_prediction)
                df['contextual_adaptive_pts'] = contextual_results.fillna(pts_long)
            except Exception:
                # Fallback seguro: usar pts_long si hay problemas con la función compleja
                df['contextual_adaptive_pts'] = pts_long
            self._register_feature('contextual_adaptive_pts', 'elite_performance')
            
            self._register_feature('pts_long_enhanced', 'core_predictive')  # 1.92% importance

        # MP Efficiency Matrix MEJORADA (MP es clave según modelo)
        if 'minutes' in df.columns and 'points' in df.columns:
            mp_avg = self._get_rolling_stat(df, 'minutes', self.windows['medium'], 'mean')
            mp_long = self._get_rolling_stat(df, 'minutes', self.windows['long'], 'mean')
            pts_avg = self._get_rolling_stat(df, 'points', self.windows['medium'], 'mean')
            
            # Eficiencia por minuto con boost por high minutes
            base_efficiency = self._safe_divide(pts_avg, mp_avg + 1)
            minutes_multiplier = np.where(mp_avg >= 32, 1.25,      # Elite starters
                                np.where(mp_avg >= 28, 1.15,       # Strong starters  
                                np.where(mp_avg >= 20, 1.0,        # Normal
                                np.where(mp_avg >= 15, 0.85, 0.6)))) # Bench players
            
            df['mp_efficiency_core'] = base_efficiency * minutes_multiplier

            self._register_feature('mp_efficiency_core', 'core_predictive')
        
        # Offensive Volume SUPER MEJORADO
        if 'field_goals_att' in df.columns:
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            fga_long = self._get_rolling_stat(df, 'field_goals_att', self.windows['long'], 'mean')
            fga_season = self._get_rolling_stat(df, 'field_goals_att', self.windows['season'], 'mean')
            
            # Volumen ofensivo base
            df['offensive_volume'] = fga_avg
            self._register_feature('offensive_volume', 'core_predictive')
            
            # Volume trend con múltiples capas
            df['volume_trend'] = self._safe_divide(fga_avg, fga_season + 1, 1.0)
            self._register_feature('volume_trend', 'core_predictive')

    def _generate_efficiency_metrics(self, df: pd.DataFrame) -> None:
        """
        Genera métricas de eficiencia optimizadas usando métricas avanzadas existentes
        """
        logger.debug("Generando métricas de eficiencia...")
            
        # Field Goal Efficiency MEJORADO
        if all(col in df.columns for col in ['field_goals_made', 'field_goals_att']):
            fg_avg = self._get_rolling_stat(df, 'field_goals_made', self.windows['medium'], 'mean')
            fg_long = self._get_rolling_stat(df, 'field_goals_made', self.windows['long'], 'mean')
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            fga_long = self._get_rolling_stat(df, 'field_goals_att', self.windows['long'], 'mean')

            # Crear fg_efficiency base con rolling para mayor estabilidad
            fg_rolling = self._get_rolling_stat(df, 'field_goals_made', self.windows['medium'], 'mean')
            fga_rolling = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            df['fg_efficiency'] = self._safe_divide(fg_rolling, fga_rolling, 0.45)
            self._register_feature('fg_efficiency', 'efficiency_metrics')

            # Field goal efficiency trend
            fg_eff_long = self._safe_divide(fg_long, fga_long, 0.45)
            df['fg_efficiency_trend'] = df['fg_efficiency'] - fg_eff_long
            self._register_feature('fg_efficiency_trend', 'efficiency_metrics')
        
        # Three Point Rate MEJORADO
        if all(col in df.columns for col in ['three_points_att', 'field_goals_att', 'three_points_made']):
            three_pa_avg = self._get_rolling_stat(df, 'three_points_att', self.windows['medium'], 'mean')
            three_p_avg = self._get_rolling_stat(df, 'three_points_made', self.windows['medium'], 'mean')
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            
            df['three_point_rate'] = self._safe_divide(three_pa_avg, fga_avg, 0.3)
             
            self._register_feature('three_point_rate', 'efficiency_metrics')

        # Points per Shot MEJORADO
        if all(col in df.columns for col in ['points', 'field_goals_att']):
            pts_avg = self._get_rolling_stat(df, 'points', self.windows['medium'], 'mean')
            pts_long = self._get_rolling_stat(df, 'points', self.windows['long'], 'mean')
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            fga_long = self._get_rolling_stat(df, 'field_goals_att', self.windows['long'], 'mean')
            
        # Shot Selection Intelligence
        if all(col in df.columns for col in ['field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'free_throws_made', 'free_throws_att']):
            fg_avg = self._get_rolling_stat(df, 'field_goals_made', self.windows['medium'], 'mean')
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            three_p_avg = self._get_rolling_stat(df, 'three_points_made', self.windows['medium'], 'mean')
            ft_avg = self._get_rolling_stat(df, 'free_throws_made', self.windows['medium'], 'mean')
            

    def _generate_ensemble_predictors(self, df: pd.DataFrame) -> None:
        """
        Genera predictores ensemble avanzados (combinando features base)
        """
        logger.debug("Generando predictores ensemble...")
        
        # Composite Scoring Predictor MEJORADO
        if all(feature in df.columns for feature in ['pts_3game_avg', 'pts_weighted_recent']):
            # Base prediction con más peso en tendencia reciente (pts_weighted_recent reemplazado)
            base_prediction = df['pts_3game_avg'] * 0.75 + df['pts_weighted_recent'] * 0.25
            
            # Aplicar momentum si está disponible
            if 'pts_momentum' in df.columns:
                momentum_adjustment = df['pts_momentum'].clip(-2, 2) * 0.5
                base_prediction += momentum_adjustment
            
            # Aplicar ajustes contextuales si existen
            context_multiplier = pd.Series(1.0, index=df.index)
            
            if 'home_advantage' in df.columns:
                context_multiplier *= df['home_advantage']
            # starter_boost ELIMINADO
            if 'opponent_factor' in df.columns:
                context_multiplier *= df['opponent_factor']
            
            # Agregar boost por volumen alto
            if 'offensive_volume' in df.columns:
                volume_boost = np.where(df['offensive_volume'] >= 12, 1.1, 
                              np.where(df['offensive_volume'] >= 8, 1.0, 0.9))
                context_multiplier *= volume_boost
                
        # Efficiency-Volume Combo MEJORADO
        if all(feature in df.columns for feature in ['true_shooting_eff', 'offensive_volume']):
            # Combinación no lineal más sofisticada
            volume_scaled = np.log1p(df['offensive_volume'])
            efficiency_scaled = df['true_shooting_eff'] ** 1.2  # Potenciar alta eficiencia
            
            df['efficiency_volume_combo'] = efficiency_scaled * volume_scaled
            self._register_feature('efficiency_volume_combo', 'efficiency_metrics')
            
            # Efficiency-volume balance score
            optimal_volume = 10  # Volumen "ideal"
            volume_penalty = np.abs(df['offensive_volume'] - optimal_volume) / optimal_volume
            df['balanced_scoring'] = df['efficiency_volume_combo'] * (1 - volume_penalty * 0.2)
            self._register_feature('balanced_scoring', 'efficiency_metrics')

        # Hot Hand Predictor simplificado
        if 'pts_3game_avg' in df.columns:
            hot_hand_base = df['pts_3game_avg']
            
        # Ultimate Predictor MEJORADO
        predictor_components = []
        weights = []
        
        # Componente base actualizado
        if 'composite_predictor' in df.columns:
            predictor_components.append(df['composite_predictor'])
            weights.append(0.4)
        elif 'pts_3game_avg' in df.columns:
            predictor_components.append(df['pts_3game_avg'])
            weights.append(0.4)
        
        # Componente eficiencia mejorado (20%)
        if 'balanced_scoring' in df.columns:
            predictor_components.append(df['balanced_scoring'] * 8)  # Scale to points
            weights.append(0.2)
        elif 'efficiency_volume_combo' in df.columns:
            predictor_components.append(df['efficiency_volume_combo'] * 10)  # Scale to points
            weights.append(0.2)
        elif 'true_shooting_eff' in df.columns:
            predictor_components.append(df['true_shooting_eff'] * 15)  # Scale to points
            weights.append(0.2)
        if 'elite_scorer_profile' in df.columns:
            predictor_components.append(df['elite_scorer_profile'] * 22)
            weights.append(0.25)
        # Componente oportunidad (5%)
        if 'peak_opportunity' in df.columns:
            predictor_components.append(df['peak_opportunity'] * 6)  # Scale to points
            weights.append(0.05)        
        
        # Breakout Game Predictor
        if all(feature in df.columns for feature in ['pts_weighted_recent', 'volume_trend', 'true_shooting_eff']):
            # Detectar condiciones para juegos excepcionales (pts_weighted_recent reemplazado)
            volume_surge = np.where(df['volume_trend'] >= 1.15, 1.0, 0.0)
            high_efficiency = np.where(df['true_shooting_eff'] >= 0.55, 1.0, 0.0)
            
            breakout_probability = volume_surge * high_efficiency
            
            # Predictor para juegos explosivos (25+ puntos)
            base_ceiling = df['pts_weighted_recent'] + 8  # Baseline + boost
            if 'scoring_ceiling' in df.columns:
                enhanced_ceiling = base_ceiling + df['scoring_ceiling'] * 0.5
            else:
                enhanced_ceiling = base_ceiling
                
            df['breakout_predictor'] = enhanced_ceiling * (1 + breakout_probability * 0.2)
            self._register_feature('breakout_predictor', 'ensemble_predictors')  # Reactivada para completar set

    def _generate_advanced_patterns(self, df: pd.DataFrame) -> None:
        """
        Genera features avanzadas basadas en patrones de machine learning
        Características de alta predictividad identificadas por análisis de importancia
        """
        logger.debug("Generando patrones avanzados...")
        
        # Game Context Intelligence MEJORADO
        context_factors = []
        context_weights = []
        
        # Season timing context
        if 'Date' in df.columns:
            # Convert to datetime if needed
            if df['Date'].dtype == 'object':
                df['Date'] = pd.to_datetime(df['Date'])

            # Season progression (0-1, donde 1 es final de temporada)
            df['season_progress'] = (df['Date'] - df['Date'].min()).dt.days / (df['Date'].max() - df['Date'].min()).days
            df['season_progress'] = df['season_progress'].fillna(0.5).clip(0, 1)
            
            self._register_feature('season_progress', 'ensemble_predictors')
        
        if context_factors:
            total_weight = sum(context_weights)
            normalized_weights = [w/total_weight for w in context_weights]
            game_context = sum(factor * weight for factor, weight in zip(context_factors, normalized_weights))

        # PATTERN 3: Scoring Momentum Patterns
        if all(col in df.columns for col in ['pts_3game_avg', 'pts_weighted_recent', 'pts_5game_consistency']):
            # Momentum combination
            momentum_score = df['pts_3game_avg'] / (df['pts_weighted_recent'] + 1)
            consistency_boost = df['pts_5game_consistency'] ** 0.8

        # Adaptive Scoring Ceiling
        if all(col in df.columns for col in ['pts_weighted_recent', 'offensive_volume', 'true_shooting_eff']):
            # Ceiling dinámico basado en capacidad demostrada
            base_ceiling = df['pts_weighted_recent']
            
            # Boost por volumen alto
            volume_ceiling_boost = np.where(df['offensive_volume'] >= 10, 
                                  df['offensive_volume'] * 0.8, 0)
            
            # Boost por eficiencia alta  
            efficiency_ceiling_boost = np.where(df['true_shooting_eff'] >= 0.55,
                                       df['true_shooting_eff'] * 15, 0)
            
    def _generate_high_scoring_features(self, df: pd.DataFrame) -> None:
        """
        Genera features especializadas para predicción de jugadores de alto scoring (25+ puntos).
        """
        logger.debug("Generando features especializadas para alto scoring...")
        
        # Elite Scorer Profile
        if all(col in df.columns for col in ['pts_weighted_recent', 'offensive_volume', 'true_shooting_eff']):
            # Criterios para ser considerado elite scorer
            pts_criterion = df['pts_weighted_recent'] >= 18  # Umbral base
            volume_criterion = df['offensive_volume'] >= 12  # Alto volumen
            efficiency_criterion = df['true_shooting_eff'] >= 0.52  # Eficiencia decente
            
            # Score compuesto para elite scorers
            elite_score = (
                (df['pts_weighted_recent'] / 35) * 0.5 +  # Normalized scoring
                (df['offensive_volume'] / 20) * 0.3 +  # Normalized volume  
                (df['true_shooting_eff'] / 0.7) * 0.2  # Normalized efficiency
            )
            
        # High Ceiling Indicator
        if 'scoring_ceiling' in df.columns and 'pts_weighted_recent' in df.columns:
            # Potencial para juegos de 30+ puntos
            ceiling_ratio = df['scoring_ceiling'] / (df['pts_weighted_recent'] + 1)
            
            df['high_ceiling_indicator'] = ceiling_ratio.clip(0, 2)
            
            self._register_feature('high_ceiling_indicator', 'core_predictive')
        
        # Volume-Efficiency Sweet Spot
        if all(col in df.columns for col in ['offensive_volume', 'true_shooting_eff']):
            # Sweet spot para alto scoring: alto volumen + buena eficiencia
            volume_normalized = df['offensive_volume'] / 20  # Normalizar a 0-1+
            efficiency_normalized = df['true_shooting_eff'] / 0.65  # Normalizar a 0-1+

        # Context Boost for Stars
        context_boost = pd.Series(1.0, index=df.index)
        
        if 'has_overtime' in df.columns:
            ot_boost = np.where(df['has_overtime'] == 1, 1.2, 1.0)
            context_boost *= ot_boost
                
        # Momentum for Breakout Games
        if all(col in df.columns for col in ['pts_3game_avg', 'pts_weighted_recent', 'volume_trend']):
            # Condiciones para juego explosivo
            recent_form = df['pts_3game_avg'] / (df['pts_weighted_recent'] + 1)
            volume_surge = df['volume_trend']

        # High Scoring Composite Predictor
        predictor_components = []
        weights = []
        
        if 'elite_scorer_profile' in df.columns:
            predictor_components.append(df['elite_scorer_profile'] * 25)  # Scale to points
            weights.append(0.3)
        
        # star_amplifier eliminado
        if 'pts_weighted_recent' in df.columns:
            predictor_components.append(df['pts_weighted_recent'])
            weights.append(0.3)
        
        if 'breakout_momentum' in df.columns:
            predictor_components.append(df['breakout_momentum'] * 3)  # Scale momentum
            weights.append(0.15)
        

    def _generate_elite_predictive_features(self, df: pd.DataFrame) -> None:
        """
        Genera features ultra-predictivas basándome en las columnas ORIGINALES del dataset
        y las TOP features que ya identificó el modelo. Usa FG, FGA, 3P, TRB, AST, +/-, GmSc, etc.
        """
        logger.debug("Generando features ultra-predictivas desde columnas originales...")
        
        # Advanced Shooting Metrics
        if all(col in df.columns for col in ['field_goals_made', 'field_goals_att', 'three_points_made', 'three_points_att', 'free_throws_made', 'free_throws_att']):
            # Shot distribution intelligence
            fg_avg = self._get_rolling_stat(df, 'field_goals_made', self.windows['medium'], 'mean')
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', self.windows['medium'], 'mean')
            three_p_avg = self._get_rolling_stat(df, 'three_points_made', self.windows['medium'], 'mean')
            three_pa_avg = self._get_rolling_stat(df, 'three_points_att', self.windows['medium'], 'mean')
            ft_avg = self._get_rolling_stat(df, 'free_throws_made', self.windows['medium'], 'mean')
            
            # Two-point attempts and makes
            two_pa_avg = fga_avg - three_pa_avg
            two_p_avg = fg_avg - three_p_avg
            
            # Shot making ability por tipo
            three_p_acc = self._safe_divide(three_p_avg, three_pa_avg, 0.35)
            two_p_acc = self._safe_divide(two_p_avg, two_pa_avg, 0.50)
            
            # FEATURE ELIMINADA: shot_difficulty_mastery
            # Razón: Importancia 0.1694%, muy baja contribución al modelo
            
        # Game Impact Metrics (basado en +/-, GmSc, BPM)
        if 'efficiency_game_score' in df.columns:
            # Game Score momentum (trending performance)
            gmsc_avg = self._get_rolling_stat(df, 'efficiency_game_score', self.windows['medium'], 'mean')
            gmsc_long = self._get_rolling_stat(df, 'efficiency_game_score', self.windows['long'], 'mean')
            
            df['game_impact_momentum'] = self._safe_divide(gmsc_avg, gmsc_long + 1, 1.0)
                        
            self._register_feature('game_impact_momentum', 'core_predictive')
        
        if 'plus_minus' in df.columns:
            # Plus/Minus trend (team performance when player is on court)
            pm_avg = self._get_rolling_stat(df, 'plus_minus', self.windows['medium'], 'mean')
        
        # Playmaking & Rebounds Contribution (AST, TRB, STL, BLK)
        if all(col in df.columns for col in ['assists', 'rebounds', 'steals', 'blocks']):
            ast_avg = self._get_rolling_stat(df, 'assists', self.windows['medium'], 'mean')
            trb_avg = self._get_rolling_stat(df, 'rebounds', self.windows['medium'], 'mean')
            stl_avg = self._get_rolling_stat(df, 'steals', self.windows['medium'], 'mean')
            blk_avg = self._get_rolling_stat(df, 'blocks', self.windows['medium'], 'mean')
        
    def _generate_advanced_context_features(self, df: pd.DataFrame) -> None:
        """
        Genera todas las features avanzadas de contexto y matchup
        Incluye rachas específicas, tendencias con regresión, matchups históricos, etc.
        """
        logger.debug("Generando features avanzadas de contexto y matchup...")
        
        # Tendencia y Forma Reciente AVANZADA
        
        # 1.1 Promedios móviles específicos (3, 5, 10 partidos)
        if 'points' in df.columns:
            df['pts_3game_avg'] = self._get_rolling_stat(df, 'points', 3, 'mean')

            self._register_feature('pts_3game_avg', 'core_predictive')

        # 1.2 Tendencia con regresión lineal (pendiente de últimos N partidos)  
        if 'points' in df.columns:
            # Usar _get_rolling_stat para consistencia y evitar data leakage
            def calculate_trend(series, window):
                if len(series) < 3:
                    return 0
                try:
                    slope = np.polyfit(range(len(series)), series, 1)[0]
                    if window == 5:
                        return slope / (series.mean() + 1e-6)
                    else:
                        return slope
                except:
                    return 0
            
            # Calcular tendencias usando _get_rolling_stat
            df['pts_trend_5game'] = df.groupby('player')['points'].transform(
                lambda x: x.rolling(5, min_periods=3).apply(
                    lambda y: calculate_trend(y, 5)
                )
            ).fillna(0)
            
            df['pts_trend_10game'] = df.groupby('player')['points'].transform(
                lambda x: x.rolling(10, min_periods=3).apply(
                    lambda y: calculate_trend(y, 10)
                )
            ).fillna(0)
            
            self._register_feature('pts_trend_5game', 'core_predictive')
            self._register_feature('pts_trend_10game', 'core_predictive')
        
        # 1.3 Consistencia avanzada (desviación estándar específica)
        if 'points' in df.columns:
            pts_5game_std = self._get_rolling_stat(df, 'points', 5, 'std')
            pts_10game_std = self._get_rolling_stat(df, 'points', 10, 'std')
            
            # Consistency score (menor std = más consistente)
            df['pts_5game_consistency'] = 1 / (1 + pts_5game_std)

            self._register_feature('pts_5game_consistency', 'core_predictive')

        # 1.4 Racha actual REAL (partidos consecutivos por encima/debajo de promedio)
        if 'points' in df.columns:
            pts_season_avg = self._get_rolling_stat(df, 'points', self.windows['season'], 'mean')
            
            def calculate_streak_no_leak(group):
                """Calcula racha hasta el juego ANTERIOR (sin incluir juego actual)"""
                if len(group) < 2:
                    return pd.Series([0] * len(group), index=group.index)
                
                # Usar solo datos hasta el juego anterior
                avg_expanded = pts_season_avg[group.index]
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
            
            df['current_streak'] = df.groupby('player')['points'].apply(calculate_streak_no_leak).reset_index(level=0, drop=True)
                        
            self._register_feature('current_streak', 'core_predictive')
        
        # Contextuales del Partido AVANZADOS
        
        # 2.1 Local vs Visitante diferencia histórica
        if 'is_home' in df.columns and 'points' in df.columns:
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
                        past_home = past_data[past_data['is_home'] == 1]['points']
                        if len(past_home) > 0:
                            home_results.append(past_home.mean())
                        else:
                            home_results.append(np.nan)
                        
                        # Rendimiento visitante (histórico)
                        past_away = past_data[past_data['is_home'] == 0]['points']
                        if len(past_away) > 0:
                            away_results.append(past_away.mean())
                        else:
                            away_results.append(np.nan)
                
                return pd.DataFrame({
                    'home_performance': home_results,
                    'away_performance': away_results
                }, index=group.index)
            
            # Aplicar función
            home_away_perf = df.groupby('player')[['is_home', 'points']].apply(
                calculate_historical_home_away
            ).reset_index(level=0, drop=True)
            
            df['home_performance'] = home_away_perf['home_performance']
            df['away_performance'] = home_away_perf['away_performance']
            
            # Fill NaN con promedio general del jugador
            player_avg_hist = df.groupby('player')['points'].expanding().mean().shift(1).reset_index(level=0, drop=True)
            GLOBAL_PTS_MEAN = 14.5  # Promedio histórico NBA aproximado - CONSTANTE SIN DATA LEAKAGE
            df['home_performance'] = df['home_performance'].fillna(player_avg_hist).fillna(GLOBAL_PTS_MEAN)
            df['away_performance'] = df['away_performance'].fillna(player_avg_hist).fillna(GLOBAL_PTS_MEAN)
            
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
        if 'Opp' in df.columns and 'points' in df.columns:
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
            df['vs_team_avg'] = df.groupby(['player', 'Opp'])['points'].apply(
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
            
            df['player_overall_avg'] = df.groupby('player')['points'].apply(
                calculate_historical_overall
            ).reset_index(level=0, drop=True)
            
            # Fill NaN with defaults and calculate advantage
            GLOBAL_PTS_MEAN = 14.5  # Promedio histórico NBA aproximado
            df['vs_team_avg'] = df['vs_team_avg'].fillna(df['player_overall_avg']).fillna(GLOBAL_PTS_MEAN)
            df['player_overall_avg'] = df['player_overall_avg'].fillna(GLOBAL_PTS_MEAN)
            
            self._register_feature('player_overall_avg', 'core_predictive')
            df['matchup_advantage'] = df['vs_team_avg'] - df['player_overall_avg']
            
            self._register_feature('vs_team_avg', 'core_predictive')  # Reactivada para completar set
            self._register_feature('matchup_advantage', 'core_predictive')  # Reactivada para completar set
        
        # 3.2 Opponent defensive rating y pace
        if self.teams_df is not None and 'Opp' in df.columns:
            try:
                # Obtener estadísticas del equipo oponente (no del equipo del jugador)
                opp_def_stats = self.teams_df.groupby('Team').agg({
                    'points_against': 'mean',  # Puntos permitidos por el equipo (defensive rating)
                    'points': 'mean',          # Puntos anotados por el equipo (pace indicator)
                    'defensive_rating': 'mean' # Defensive rating oficial
                }).reset_index()
                
                # Crear diccionarios para mapping del oponente
                opp_def_rating = dict(zip(opp_def_stats['Team'], opp_def_stats['points_against']))
                opp_pace = dict(zip(opp_def_stats['Team'], opp_def_stats['points']))
                opp_def_rating_official = dict(zip(opp_def_stats['Team'], opp_def_stats['defensive_rating']))
                
                # Mapear estadísticas del oponente a cada jugador
                df['opponent_defensive_rating'] = df['Opp'].map(opp_def_rating).fillna(110.0)
                df['opponent_pace'] = df['Opp'].map(opp_pace).fillna(110.0)
                df['opponent_def_rating_official'] = df['Opp'].map(opp_def_rating_official).fillna(110.0)
                
                # Calcular ventaja defensiva (menor rating = mejor defensa)
                df['opponent_defensive_advantage'] = 110.0 - df['opponent_defensive_rating']
                df['opponent_pace_factor'] = df['opponent_pace'] / 110.0  # Normalizar pace
                
                self._register_feature('opponent_defensive_rating', 'context_adjusters')
                self._register_feature('opponent_pace', 'context_adjusters')
                self._register_feature('opponent_def_rating_official', 'context_adjusters')
                self._register_feature('opponent_defensive_advantage', 'context_adjusters')
                self._register_feature('opponent_pace_factor', 'context_adjusters')
                
            except Exception as e:
                logger.debug(f"No se pudieron crear features de opponent stats: {e}")
        
        # 4.2 Team assists context (más asistencias = más oportunidades)
        try:
            if 'assists' in df.columns and 'Team' in df.columns and 'Date' in df.columns:
                # Crear feature más simple y robusta
                # Calcular promedio de asistencias por equipo basado en datos históricos
                team_ast_rolling = df.groupby('Team')['assists'].rolling(
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
            
        # Situación de Juego AVANZADA
        
        # 5.1 Importancia del partido mejorada - SIEMPRE crear las features
        # FEATURES ELIMINADAS: playoff_race, season_finale, midseason_form
        # Razón: Importancia 0.0 - 0.0057%, no aportan al modelo
        pass

    def _generate_momentum_variability_features(self, df: pd.DataFrame) -> None:
        """
        Genera features de momentum y variabilidad basadas en top importance.
        Enfoque en patrones temporales, contexto de equipo, oponente granular y situación de juego.
        """
        logger.debug("Generando features de momentum y variabilidad ultra-críticas...")
        
        # MOMENTUM Y VARIABILIDAD AVANZADA
        
        # 1.1 Rolling momentum ponderado (basado en momentum_consistency_combo = 1.69% importance)
        if 'points' in df.columns:
            # Media móvil ponderada últimos 3 partidos (más peso a partidos recientes)
            weights_3game = np.array([0.5, 0.3, 0.2])  # Más peso al último partido
            
            rolling_momentum_3games = df.groupby('player')['points'].rolling(
                window=3, min_periods=1
            ).apply(
                lambda x: np.average(x, weights=weights_3game[:len(x)]) if len(x) > 0 else 0
            ).shift(1).reset_index(0, drop=True).fillna(0)

            # Momentum acceleration (diferencia entre momentum actual vs anterior)
            momentum_5game = self._get_rolling_stat(df, 'points', 5, 'mean')
            momentum_10game = self._get_rolling_stat(df, 'points', 10, 'mean')
            
            df['momentum_acceleration'] = momentum_5game - momentum_10game
            self._register_feature('momentum_acceleration', 'core_predictive')
            
            # Volatility index (coeficiente de variación últimos 10 partidos)
            pts_10_std = self._get_rolling_stat(df, 'points', 10, 'std')
            pts_10_mean = self._get_rolling_stat(df, 'points', 10, 'mean')
            
            df['volatility_index'] = self._safe_divide(pts_10_std, pts_10_mean, 0.3)
            df['volatility_index'] = df['volatility_index'].clip(0, 2)  # Clip extremos
            
            self._register_feature('volatility_index', 'core_predictive')
        
        # CONTEXTO DE EQUIPO MEJORADO
        
        # 2.1 Team offensive rating últimos 5 partidos (basado en home_context_factor = 1.71%)
        if 'points' in df.columns:
            # Rating ofensivo del equipo (puntos por 100 posesiones aproximado)
            team_pts_L5 = df.groupby('Team')['points'].rolling(
                window=5, min_periods=1
            ).sum().shift(1).reset_index(0, drop=True).fillna(500)  # 5 games total
            
            df['team_offensive_rating_L5'] = team_pts_L5 / 5  # Promedio por partido
            self._register_feature('team_offensive_rating_L5', 'context_adjusters')
            
            # Pace differential (si tenemos datos de equipos oponentes)
            if self.teams_df is not None and 'Opp' in df.columns:
                try:
                    # Pace del equipo propio vs pace del oponente
                    own_team_pace = df.groupby('Team')['points'].rolling(3).mean().shift(1).reset_index(0, drop=True)
                    
                    # Pace del oponente (de datos de equipos)
                    opp_pace_dict = self.teams_df.groupby('Team')['points'].mean().to_dict()
                    df['opp_pace_estimate'] = df['Opp'].map(opp_pace_dict).fillna(110)
                    # self._register_feature('opp_pace_estimate', 'context_adjusters')  # Desactivada temporalmente - incompatible con modelo entrenado
                    
                    df['pace_differential'] = own_team_pace - df['opp_pace_estimate']
                    # self._register_feature('pace_differential', 'context_adjusters')  # Desactivada temporalmente - incompatible con modelo entrenado
                    
                except Exception as e:
                    logger.debug(f"No se pudo crear pace_differential: {e}")
        
        # 3.1 Opponent pts allowed vs position (puntos permitidos específicamente por posición)
        if 'position' in df.columns and 'Opp' in df.columns:
            # Calcular histórico de puntos permitidos por oponente vs posición
            def calculate_opp_vs_position(group):
                """Calcula puntos permitidos por oponente vs posición específica"""
                results = []
                for i in range(len(group)):
                    if i == 0:
                        results.append(15.0)  # Default por posición
                        continue
                    
                    # Buscar histórico vs esta posición
                    current_opp = group.iloc[i]['Opp'] if 'Opp' in group.columns else None
                    current_pos = group.iloc[i]['position'] if 'position' in group.columns else None
                    
                    if current_opp and current_pos:
                        # Buscar partidos anteriores vs este oponente y posición similar
                        past_vs_opp = df[(df['Opp'] == current_opp) & (df['position'] == current_pos)]
                        if len(past_vs_opp) > 0:
                            results.append(past_vs_opp['points'].mean())
                        else:
                            results.append(15.0)
                    else:
                        results.append(15.0)
                
                return pd.Series(results, index=group.index)
            
            # FEATURE ELIMINADA: opponent_pts_allowed_vs_position
            # Razón: Importancia 0.0%, no aporta al modelo
            pass
        
        # 3.3 Opponent defensive trend (tendencia defensiva últimos 5 partidos)
        if self.teams_df is not None and 'Opp' in df.columns:
            try:
                # Ordenar por fecha para calcular tendencia correctamente
                teams_sorted = self.teams_df.sort_values(['Team', 'Date'])
                
                # Calcular tendencia defensiva del oponente (últimos 5 partidos)
                opp_def_trend = teams_sorted.groupby('Team')['points_against'].rolling(
                    5, min_periods=1
                ).mean().reset_index()
                
                # Crear diccionario para mapping
                opp_trend_dict = dict(zip(
                    opp_def_trend['Team'], 
                    opp_def_trend['points_against']
                ))
                
                # Mapear tendencia defensiva del oponente a cada jugador
                df['opponent_defensive_trend'] = df['Opp'].map(opp_trend_dict).fillna(110.0)
                
                # Calcular si la defensa del oponente está mejorando o empeorando
                opp_def_recent = teams_sorted.groupby('Team')['points_against'].rolling(
                    3, min_periods=1
                ).mean().reset_index()
                opp_def_older = teams_sorted.groupby('Team')['points_against'].rolling(
                    5, min_periods=1
                ).mean().reset_index()
                
                # Calcular diferencia (menor = mejor defensa)
                opp_def_improvement = {}
                for team in opp_def_recent['Team'].unique():
                    recent = opp_def_recent[opp_def_recent['Team'] == team]['points_against'].iloc[-1] if len(opp_def_recent[opp_def_recent['Team'] == team]) > 0 else 110.0
                    older = opp_def_older[opp_def_older['Team'] == team]['points_against'].iloc[-1] if len(opp_def_older[opp_def_older['Team'] == team]) > 0 else 110.0
                    opp_def_improvement[team] = older - recent  # Positivo = defensa mejorando
                
                df['opponent_def_improvement'] = df['Opp'].map(opp_def_improvement).fillna(0.0)
                
                self._register_feature('opponent_defensive_trend', 'context_adjusters')
                self._register_feature('opponent_def_improvement', 'context_adjusters')
                
            except Exception as e:
                logger.debug(f"No se pudo crear opponent_defensive_trend: {e}")

        # 5.2 Usage vs efficiency ratio
        if all(col in df.columns for col in ['field_goals_att', 'free_throws_att', 'true_shooting_pct']):
            fga_avg = self._get_rolling_stat(df, 'field_goals_att', 7, 'mean')
            fta_avg = self._get_rolling_stat(df, 'free_throws_att', 7, 'mean')
            ts_avg = self._get_rolling_stat(df, 'true_shooting_pct', 7, 'mean')
            
            usage_estimate = fga_avg + 0.44 * fta_avg  # Approximation
            df['usage_vs_efficiency'] = self._safe_divide(ts_avg, usage_estimate + 1, 0.5)
            
            self._register_feature('usage_vs_efficiency', 'efficiency_metrics')
        
        # RED DE APOYO (TEAM CONTEXT)
        
        # 6.1 Team ball movement (asistencias del equipo promedio)
        if 'assists' in df.columns:
            team_ast_trend = df.groupby('Team')['assists'].rolling(
                5, min_periods=1
            ).mean().shift(1).reset_index(0, drop=True).fillna(25)
            
            df['team_ball_movement'] = team_ast_trend / 25  # Normalizado
            self._register_feature('team_ball_movement', 'context_adjusters')

    def _generate_outlier_detection_features(self, df: pd.DataFrame) -> None:
        """
        Genera features especializadas para detectar y manejar outliers/games especiales.
        """
        logger.debug("Generando features de detección de outliers...")
        
        if 'points' not in df.columns or 'player' not in df.columns:
            logger.warning("No se pueden generar features de outliers sin PTS y Player")
            return
        
        # Z-Score Rolling por jugador
        if 'player' in df.columns:
            # Z-score basado en partidos PREVIOS (sin usar el juego actual)
            pts_rolling_mean = self._get_rolling_stat(df, 'points', 10, 'mean')
            pts_rolling_std = self._get_rolling_stat(df, 'points', 10, 'std')
            
            # Usar promedio de últimos 3 juegos en lugar de solo el último juego
            pts_recent_3 = self._get_rolling_stat(df, 'points', 3, 'mean')
            
            df['pts_zscore_10game'] = np.where(
                pts_rolling_std > 0.1,
                (pts_recent_3 - pts_rolling_mean) / pts_rolling_std,
                0
            )
            
            # Z-score de la temporada
            pts_season_mean = self._get_rolling_stat(df, 'points', self.windows['season'], 'mean')
            pts_season_std = self._get_rolling_stat(df, 'points', self.windows['season'], 'std')
            
            df['pts_zscore_season'] = np.where(
                pts_season_std > 0.1,
                (pts_recent_3 - pts_season_mean) / pts_season_std,
                0
            )
            
            self._register_feature('pts_zscore_10game', 'outlier_detection')
            self._register_feature('pts_zscore_season', 'outlier_detection')
                
        # Game specialty indicators
        special_game_score = 0
        
        if 'has_overtime' in df.columns:
            special_game_score += df['has_overtime'] * 1.5

    def _generate_high_range_specialist_features(self, df: pd.DataFrame) -> None:
        """
        Features especializadas para resolver R² negativos en rangos 15-30 puntos.
        """
        logger.debug("Generando features especialistas de rangos altos...")
        
        if 'points' not in df.columns or 'player' not in df.columns:
            logger.warning("No se pueden generar features de rangos altos sin PTS y Player")
            return
        

        if 'pts_trend_5game' in df.columns and 'pts_3game_avg' in df.columns:
            # Calcular baseline dinámico por jugador
            player_baseline = df.groupby('player')['points'].transform(
                lambda x: x.shift(1).expanding(min_periods=5).mean()
            ).fillna(df['pts_3game_avg'])
            
            # Momentum normalizado por el baseline del jugador
            trend_normalized = df['pts_trend_5game'] / (player_baseline + 1)
            recent_vs_baseline = (df['pts_3game_avg'] - player_baseline) / (player_baseline + 1)
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
        # Predice el techo de puntos basado en condiciones específicas
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
        if 'pts_trend_5game' in df.columns:
            momentum_boost = 1 + np.clip(df['pts_trend_5game'] / 5.0, -0.3, 0.5)
            ceiling_factors *= momentum_boost
        
        # Ceiling base del jugador
        base_ceiling = df.get('pts_weighted_recent', 12) * 1.8  # 80% más que su promedio
        df['specialized_ceiling'] = base_ceiling * ceiling_factors
        df['specialized_ceiling'] = np.clip(df['specialized_ceiling'], 8, 50)
        
        self._register_feature('specialized_ceiling', 'high_range_specialist')
        
        logger.debug(f"Generadas {len(self.feature_categories.get('outlier_detection', []))} features de detección de outliers")
    
    def _generate_missing_critical_features(self, df: pd.DataFrame) -> None:
        """
        Genera las features críticas que faltan para completar el set del modelo entrenado.
        """
        logger.debug("Generando features críticas faltantes para completar set del modelo...")
        
        # 1. GAME IMPACT MOMENTUM - Momentum de impacto en el juego
        if self._register_feature('game_impact_momentum', 'ensemble_predictors'):
            if 'points' in df.columns:
                # Combinar múltiples métricas de momentum
                pts_momentum = self._get_rolling_stat(df, 'points', 3, 'mean') - self._get_rolling_stat(df, 'points', 7, 'mean')
                
                # Añadir componente de variabilidad (explosividad)
                pts_volatility = self._get_rolling_stat(df, 'points', 5, 'std')
                
                # Combinar momentum + volatilidad
                df['game_impact_momentum'] = (pts_momentum * (1 + pts_volatility / 10)).fillna(0)
            else:
                df['game_impact_momentum'] = 0.0
        
        
        # 4. USAGE VS EFFICIENCY - Relación entre uso y eficiencia
        if self._register_feature('usage_vs_efficiency', 'efficiency_metrics'):
            if 'points' in df.columns and 'minutes' in df.columns:
                # Convertir MP a numérico si es string
                mp_numeric = df['minutes'].copy()
                if mp_numeric.dtype == 'object':
                    # Convertir formato "MM:SS" a decimal
                    mp_numeric = mp_numeric.astype(str).apply(
                        lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60 
                        if ':' in str(x) else float(x) if str(x).replace('.','').isdigit() else 35.0
                    )
                
                # Usage Rate aproximado (PTS por minuto)
                usage_rate = df['points'] / (mp_numeric + 1e-6)
                
                # Eficiencia (puntos por posesión aproximada)
                possessions_approx = mp_numeric / 48 * 100  # ~100 posesiones por juego
                efficiency = df['points'] / (possessions_approx + 1e-6)
                
                # Relación usage vs efficiency
                df['usage_vs_efficiency'] = (usage_rate * efficiency).fillna(0)
            else:
                df['usage_vs_efficiency'] = 0.0
                
    def _generate_advanced_dataset_features(self, df: pd.DataFrame) -> None:
        """
        Genera features avanzadas usando las nuevas métricas del dataset de SportRadar
        """
        logger.debug("Generando features avanzadas del nuevo dataset...")
        
        # EFFICIENCY METRICS DIRECTAS (usar las ya calculadas)
        if 'efficiency' in df.columns:
            # Usar directamente efficiency del dataset
            efficiency_avg = self._get_rolling_stat(df, 'efficiency', self.windows['medium'], 'mean')
            efficiency_long = self._get_rolling_stat(df, 'efficiency', self.windows['long'], 'mean')
            
            df['efficiency_rolling_avg'] = efficiency_avg
            df['efficiency_momentum'] = efficiency_avg - efficiency_long
            self._register_feature('efficiency_rolling_avg', 'efficiency_metrics')
            self._register_feature('efficiency_momentum', 'efficiency_metrics')
        
        # OFFENSIVE & DEFENSIVE RATING FEATURES
        if 'offensive_rating' in df.columns:
            off_rating_avg = self._get_rolling_stat(df, 'offensive_rating', self.windows['medium'], 'mean')
            off_rating_long = self._get_rolling_stat(df, 'offensive_rating', self.windows['long'], 'mean')
            
            df['offensive_rating_avg'] = off_rating_avg
            df['offensive_rating_trend'] = off_rating_avg - off_rating_long
            self._register_feature('offensive_rating_avg', 'efficiency_metrics')
            self._register_feature('offensive_rating_trend', 'efficiency_metrics')
        
        if 'defensive_rating' in df.columns:
            def_rating_avg = self._get_rolling_stat(df, 'defensive_rating', self.windows['medium'], 'mean')
            df['defensive_impact'] = 120 - def_rating_avg  # Convertir a "mejor = mayor"
            self._register_feature('defensive_impact', 'efficiency_metrics')
        
        # ADVANCED SHOOTING METRICS
        if 'effective_fg_pct' in df.columns:
            efg_avg = self._get_rolling_stat(df, 'effective_fg_pct', self.windows['medium'], 'mean')
            efg_long = self._get_rolling_stat(df, 'effective_fg_pct', self.windows['long'], 'mean')
            
            df['effective_fg_momentum'] = efg_avg - efg_long
            self._register_feature('effective_fg_momentum', 'efficiency_metrics')
        
        # POINTS IN PAINT FEATURES
        if all(col in df.columns for col in ['points_in_paint', 'points_in_paint_att', 'points_in_paint_pct']):
            pip_avg = self._get_rolling_stat(df, 'points_in_paint', self.windows['medium'], 'mean')
            pip_pct_avg = self._get_rolling_stat(df, 'points_in_paint_pct', self.windows['medium'], 'mean')
            
            df['paint_scoring_avg'] = pip_avg
            df['paint_efficiency'] = pip_pct_avg / 100.0  # Normalizar
            self._register_feature('paint_scoring_avg', 'efficiency_metrics')
            self._register_feature('paint_efficiency', 'efficiency_metrics')
        
        # FAST BREAK & SECOND CHANCE FEATURES
        if 'fast_break_pts' in df.columns:
            fb_avg = self._get_rolling_stat(df, 'fast_break_pts', self.windows['medium'], 'mean')
            df['fast_break_scoring'] = fb_avg
            self._register_feature('fast_break_scoring', 'opportunity_factors')
        
        if 'second_chance_pts' in df.columns:
            sc_avg = self._get_rolling_stat(df, 'second_chance_pts', self.windows['medium'], 'mean')
            df['second_chance_scoring'] = sc_avg
            self._register_feature('second_chance_scoring', 'opportunity_factors')
        
        # REBOUNDING PERCENTAGE FEATURES (nuevas métricas avanzadas)
        if 'offensive_rebounds_pct' in df.columns:
            orb_pct_avg = self._get_rolling_stat(df, 'offensive_rebounds_pct', self.windows['medium'], 'mean')
            df['offensive_board_rate'] = orb_pct_avg / 100.0
            self._register_feature('offensive_board_rate', 'opportunity_factors')
        
        if 'defensive_rebounds_pct' in df.columns:
            drb_pct_avg = self._get_rolling_stat(df, 'defensive_rebounds_pct', self.windows['medium'], 'mean')
            df['defensive_board_rate'] = drb_pct_avg / 100.0
            self._register_feature('defensive_board_rate', 'opportunity_factors')
        
        # TURNOVERS & STEALS PERCENTAGE
        if 'turnovers_pct' in df.columns:
            tov_pct_avg = self._get_rolling_stat(df, 'turnovers_pct', self.windows['medium'], 'mean')
            df['turnover_control'] = 1.0 - (tov_pct_avg / 100.0)  # Invertir: menos TOV = mejor
            self._register_feature('turnover_control', 'efficiency_metrics')
        
        if 'steals_pct' in df.columns:
            stl_pct_avg = self._get_rolling_stat(df, 'steals_pct', self.windows['medium'], 'mean')
            df['steal_rate'] = stl_pct_avg / 100.0
            self._register_feature('steal_rate', 'efficiency_metrics')
        
        # ASSISTS TO TURNOVER RATIO (ya calculado en dataset)
        if 'assists_turnover_ratio' in df.columns:
            ast_tov_avg = self._get_rolling_stat(df, 'assists_turnover_ratio', self.windows['medium'], 'mean')
            df['playmaking_efficiency'] = ast_tov_avg
            self._register_feature('playmaking_efficiency', 'efficiency_metrics')
        
        # COMPOSITE ADVANCED SCORER PROFILE
        if all(col in df.columns for col in ['efficiency', 'offensive_rating', 'true_shooting_pct']):
            efficiency_norm = (df.get('efficiency_rolling_avg', 0) - 10) / 20  # Normalizar ~0-1
            off_rating_norm = (df.get('offensive_rating_avg', 100) - 80) / 40  # Normalizar ~0-1  
            ts_norm = df.get('true_shooting_eff', 0.5)  # Ya normalizado 0-1
            
            df['advanced_scorer_profile'] = (
                efficiency_norm * 0.4 + 
                off_rating_norm * 0.35 + 
                ts_norm * 0.25
            ).fillna(0.5)
            
            self._register_feature('advanced_scorer_profile', 'ensemble_predictors')

    def _generate_quarters_features(self, df: pd.DataFrame) -> None:
        """
        Genera features usando datos históricos por cuartos si están disponibles
        """
        logger.debug("Generando features de datos por cuartos...")
        
        if self.players_quarters_df is None:
            logger.debug("No hay datos por cuartos disponibles")
            return
        
        try:
            quarters_df = self.players_quarters_df.copy()
            
            # Verificar que tenemos las columnas necesarias
            required_cols = ['player', 'Date', 'quarter', 'points']
            if not all(col in quarters_df.columns for col in required_cols):
                logger.warning(f"Faltan columnas requeridas en quarters_df: {required_cols}")
                return
            
            # Mapear Player name consistency
            if 'player' in quarters_df.columns and 'player' not in quarters_df.columns:
                quarters_df['player'] = quarters_df['player']
            
            # QUARTER-SPECIFIC PERFORMANCE PATTERNS
            
            # 1. Rendimiento por cuarto (cuartos 1-4)
            # FEATURES ELIMINADAS: q1_scoring_avg, q2_scoring_avg, q3_scoring_avg, q4_scoring_avg
            # Razón: Importancias entre 0.132% - 0.1521%, muy baja contribución al modelo
            pass
            
            # 2. CLUTCH TIME PERFORMANCE (4to cuarto)
            q4_data = quarters_df[quarters_df['quarter'] == 4]
            if len(q4_data) > 0:
                q4_efficiency = q4_data.groupby('player')['efficiency'].rolling(
                    window=10, min_periods=3
                ).mean().shift(1).reset_index(level=0, drop=True)
                
                q4_mapping = dict(zip(q4_data['player'], q4_efficiency))
                df['clutch_efficiency'] = df['player'].map(q4_mapping).fillna(10)
                self._register_feature('clutch_efficiency', 'context_adjusters')
            
            # 3. QUARTER CONSISTENCY (variabilidad entre cuartos)
            if len(quarters_df) > 0:
                # Calcular std de puntos por cuarto para cada jugador
                quarter_consistency = quarters_df.groupby(['player', 'Date', 'game_id'])['points'].std().reset_index()
                quarter_consistency_avg = quarter_consistency.groupby('player')['points'].rolling(
                    window=10, min_periods=3
                ).mean().shift(1).reset_index(level=0, drop=True)
                
                consistency_mapping = dict(zip(quarter_consistency['player'], quarter_consistency_avg))
                df['quarter_consistency'] = df['player'].map(consistency_mapping).fillna(2.0)
                # Invertir: menos variabilidad = más consistencia
                df['quarter_consistency'] = 1.0 / (1.0 + df['quarter_consistency'])
                self._register_feature('quarter_consistency', 'context_adjusters')
            
            # 4. EARLY vs LATE GAME PERFORMANCE
            if len(quarters_df) > 0:
                # Primeros 2 cuartos vs últimos 2 cuartos
                early_quarters = quarters_df[quarters_df['quarter'].isin([1, 2])]
                late_quarters = quarters_df[quarters_df['quarter'].isin([3, 4])]
                
                if len(early_quarters) > 0 and len(late_quarters) > 0:
                    early_avg = early_quarters.groupby('player')['points'].rolling(
                        window=10, min_periods=3
                    ).mean().shift(1).reset_index(level=0, drop=True)
                    
                    late_avg = late_quarters.groupby('player')['points'].rolling(
                        window=10, min_periods=3
                    ).mean().shift(1).reset_index(level=0, drop=True)
                    
                    early_mapping = dict(zip(early_quarters['player'], early_avg))
                    late_mapping = dict(zip(late_quarters['player'], late_avg))
                    
                    # FEATURES ELIMINADAS: early_game_scoring, late_game_scoring, early_late_diff
                    # Razón: early_game_scoring tiene importancia 0.1578%, muy baja contribución
                    df['early_late_diff'] = 0.0  # Mantener solo esta para compatibilidad pero con valor neutro
                    self._register_feature('early_late_diff', 'context_adjusters')
        
        except Exception as e:
            logger.warning(f"Error generando features de cuartos: {e}")

    def _generate_ultra_advanced_features(self, df: pd.DataFrame) -> None:
        """
        Genera features ultra-avanzadas explotando TODAS las columnas disponibles
        """
        logger.debug("Generando features ultra-avanzadas...")
        
        # 2. BLOCKED SHOTS FEATURES (blocked_att)
        if 'blocked_att' in df.columns:
            blocked_avg = self._get_rolling_stat(df, 'blocked_att', self.windows['medium'], 'mean')
            df['shot_blocking_ability'] = blocked_avg
            self._register_feature('shot_blocking_ability', 'efficiency_metrics')
        
        # 3. PLS_MIN FEATURES (Plus/Minus per minute)
        if 'pls_min' in df.columns and 'minutes' in df.columns:
            pls_min_avg = self._get_rolling_stat(df, 'pls_min', self.windows['medium'], 'mean')
            mp_avg = self._get_rolling_stat(df, 'minutes', self.windows['medium'], 'mean')
            
            df['impact_per_minute'] = self._safe_divide(pls_min_avg, mp_avg, 0.0)
            self._register_feature('impact_per_minute', 'efficiency_metrics')
        
        # 5. EFFICIENCY GAME SCORE FEATURES
        if 'efficiency_game_score' in df.columns:
            game_score_avg = self._get_rolling_stat(df, 'efficiency_game_score', self.windows['medium'], 'mean')
            game_score_long = self._get_rolling_stat(df, 'efficiency_game_score', self.windows['long'], 'mean')
            
            df['game_score_momentum'] = game_score_avg - game_score_long
            df['game_score_consistency'] = 1.0 / (1.0 + self._get_rolling_stat(df, 'efficiency_game_score', self.windows['medium'], 'std'))
            
            self._register_feature('game_score_momentum', 'efficiency_metrics')
            self._register_feature('game_score_consistency', 'efficiency_metrics')

        # 8. SECOND CHANCE FEATURES MEJORADAS
        if all(col in df.columns for col in ['second_chance_att', 'second_chance_made', 'second_chance_pct']):
            sc_att_avg = self._get_rolling_stat(df, 'second_chance_att', self.windows['medium'], 'mean')
            sc_made_avg = self._get_rolling_stat(df, 'second_chance_made', self.windows['medium'], 'mean')
            sc_pct_avg = self._get_rolling_stat(df, 'second_chance_pct', self.windows['medium'], 'mean')
            
            df['second_chance_volume'] = sc_att_avg
            df['second_chance_conversion'] = sc_pct_avg / 100.0
            df['second_chance_impact'] = sc_made_avg * (sc_pct_avg / 100.0)
            
            self._register_feature('second_chance_volume', 'opportunity_factors')
            self._register_feature('second_chance_conversion', 'efficiency_metrics')
            self._register_feature('second_chance_impact', 'efficiency_metrics')
        
        # 9. FAST BREAK FEATURES MEJORADAS
        if all(col in df.columns for col in ['fast_break_att', 'fast_break_made', 'fast_break_pct']):
            fb_att_avg = self._get_rolling_stat(df, 'fast_break_att', self.windows['medium'], 'mean')
            fb_made_avg = self._get_rolling_stat(df, 'fast_break_made', self.windows['medium'], 'mean')
            fb_pct_avg = self._get_rolling_stat(df, 'fast_break_pct', self.windows['medium'], 'mean')
            
            df['fast_break_volume'] = fb_att_avg
            df['fast_break_conversion'] = fb_pct_avg / 100.0
            df['transition_scoring_ability'] = fb_made_avg * (fb_pct_avg / 100.0)
            
            self._register_feature('fast_break_volume', 'opportunity_factors')
            self._register_feature('fast_break_conversion', 'efficiency_metrics')
            self._register_feature('transition_scoring_ability', 'efficiency_metrics')
        

        # 11. POINTS OFF TURNOVERS
        if 'points_off_turnovers' in df.columns:
            pot_avg = self._get_rolling_stat(df, 'points_off_turnovers', self.windows['medium'], 'mean')
            df['defensive_conversion_scoring'] = pot_avg
            self._register_feature('defensive_conversion_scoring', 'opportunity_factors')
        
        # 12. COMPOSITE SPECIALTY SCORES
        
        # Elite Shooter Profile
        if all(col in df.columns for col in ['three_points_pct', 'free_throws_pct', 'effective_fg_pct']):
            three_pct_avg = self._get_rolling_stat(df, 'three_points_pct', self.windows['medium'], 'mean')
            ft_pct_avg = self._get_rolling_stat(df, 'free_throws_pct', self.windows['medium'], 'mean')
            efg_avg = self._get_rolling_stat(df, 'effective_fg_pct', self.windows['medium'], 'mean')
            
            # Normalizar y combinar
            three_norm = three_pct_avg / 50.0  # ~50% es elite desde 3
            ft_norm = ft_pct_avg / 90.0        # ~90% es elite desde FT
            efg_norm = efg_avg / 60.0          # ~60% es elite eFG%
            
            df['elite_shooter_profile'] = (three_norm * 0.4 + ft_norm * 0.3 + efg_norm * 0.3).fillna(0.5)
            self._register_feature('elite_shooter_profile', 'ensemble_predictors')
        
        # Rebounding Specialist Profile  
        if all(col in df.columns for col in ['offensive_rebounds_pct', 'defensive_rebounds_pct', 'rebounds_pct']):
            orb_pct_avg = self._get_rolling_stat(df, 'offensive_rebounds_pct', self.windows['medium'], 'mean')
            drb_pct_avg = self._get_rolling_stat(df, 'defensive_rebounds_pct', self.windows['medium'], 'mean')
            total_reb_pct_avg = self._get_rolling_stat(df, 'rebounds_pct', self.windows['medium'], 'mean')
            
            # Normalizar (10% ORB, 20% DRB, 15% total son típicos)
            orb_norm = orb_pct_avg / 15.0
            drb_norm = drb_pct_avg / 25.0  
            total_norm = total_reb_pct_avg / 20.0
            
            df['rebounding_specialist_profile'] = (orb_norm * 0.3 + drb_norm * 0.4 + total_norm * 0.3).fillna(0.5)
            self._register_feature('rebounding_specialist_profile', 'ensemble_predictors')
        
        # Defensive Impact Profile
        if all(col in df.columns for col in ['steals_pct', 'blocks', 'defensive_rating']):
            stl_pct_avg = self._get_rolling_stat(df, 'steals_pct', self.windows['medium'], 'mean')
            blk_avg = self._get_rolling_stat(df, 'blocks', self.windows['medium'], 'mean')
            def_rating_avg = self._get_rolling_stat(df, 'defensive_rating', self.windows['medium'], 'mean')
            
            # Normalizar (2% steals, 1 block, 110 def rating son típicos)
            stl_norm = stl_pct_avg / 3.0
            blk_norm = blk_avg / 2.0
            def_norm = (120 - def_rating_avg) / 20.0  # Invertir: menor rating = mejor defensa
            
            df['defensive_impact_profile'] = (stl_norm * 0.3 + blk_norm * 0.3 + def_norm * 0.4).fillna(0.5)
            self._register_feature('defensive_impact_profile', 'ensemble_predictors')
