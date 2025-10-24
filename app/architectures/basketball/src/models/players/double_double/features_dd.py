"""
Módulo de Características para Predicción de Double Double
=========================================================

Este módulo contiene toda la lógica de ingeniería de características específica
para la predicción de double double de un jugador NBA por partido. Implementa características
avanzadas enfocadas en factores que determinan la probabilidad de lograr un double double.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DoubleDoubleFeatureEngineer:
    """
    Motor de features para predicción de double double
    """
    
    def __init__(self, lookback_games: int = 10, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        """Inicializa el ingeniero de características para predicción de double double."""
        self.lookback_games = lookback_games
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.teams_df = teams_df  # Datos de equipos 
        self.players_df = players_df  # Datos de jugadores 
        self.players_quarters_df = players_quarters_df  # Datos de jugadores por cuartos
        # Cache para evitar recálculos
        self._cached_calculations = {}
        # Cache para features generadas
        self._features_cache = {}
        self._last_data_hash = None
        # Cache para rolling statistics
        self._rolling_cache = {}
        # Registry para features registradas
        self.feature_registry = {}
    
    def _ensure_datetime_and_sort(self, df: pd.DataFrame) -> None:
        """Método auxiliar para asegurar que Date esté en formato datetime y ordenar datos"""
        if 'Date' in df.columns and df['Date'].dtype != 'datetime64[ns]':
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df.sort_values(['player', 'Date'], inplace=True)
            df.reset_index(drop=True, inplace=True)
            logger.debug("Datos ordenados cronológicamente por jugador")
    
    def _calculate_player_context_features(self, df: pd.DataFrame) -> None:
        """Método auxiliar para calcular features de contexto del jugador una sola vez"""
        # Features de contexto ya disponibles del data_loader
        if 'is_home' not in df.columns:
            logger.debug("is_home no encontrado del data_loader - features de ventaja local no disponibles")
        else:
            logger.debug("Usando is_home del data_loader para features de ventaja local")
            # Calcular features relacionadas con ventaja local
            df['home_advantage'] = df['is_home'] * 0.03  # 3% boost para jugadores en casa
            df['travel_penalty'] = np.where(df['is_home'] == 0, -0.01, 0.0)
        
    def _get_historical_series(self, df: pd.DataFrame, column: str, window: int, operation: str = 'mean', min_periods: int = 1) -> pd.Series:
        """
        Cálculo rolling optimizado con cache para evitar duplicados
        SIEMPRE usa shift(1) para prevenir data leakage (futuros juegos)
        MANTIENE alineación de índices con el DataFrame original
        """
        cache_key = f"{column}_{window}_{operation}_{len(df)}"
        
        if cache_key in self._rolling_cache:
            cached_result = self._rolling_cache[cache_key]
            # Verificar que el índice coincida con el DataFrame actual
            if len(cached_result) == len(df) and cached_result.index.equals(df.index):
                return cached_result
            else:
                # Cache obsoleto, eliminarlo
                del self._rolling_cache[cache_key]
        
        if column not in df.columns:
            result = pd.Series(0, index=df.index)
        else:
            if operation == 'mean':
                result = df.groupby('player')[column].rolling(window=window, min_periods=min_periods).mean().shift(1)
            elif operation == 'std':
                result = df.groupby('player')[column].rolling(window=window, min_periods=min_periods).std().shift(1)
            elif operation == 'max':
                result = df.groupby('player')[column].rolling(window=window, min_periods=min_periods).max().shift(1)
            elif operation == 'sum':
                result = df.groupby('player')[column].rolling(window=window, min_periods=min_periods).sum().shift(1)
            else:
                result = df.groupby('player')[column].rolling(window=window, min_periods=min_periods).mean().shift(1)
            
            # CRÍTICO: Reindexar para mantener alineación con el DataFrame original
            result = result.reset_index(level=0, drop=True).reindex(df.index, fill_value=0)
            
            # Almacenar en cache
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
        PIPELINE SIMPLIFICADO DE FEATURES 
        """

        logger.debug("Generando features para double double...")

        self._convert_mp_to_numeric(df)

        # VERIFICACIÓN DE double_double COMO TARGET (ya viene del dataset)
        if 'double_double' not in df.columns:
            logger.error("double_double no encontrado en el dataset - requerido para features de double double")
            return []
        
        # VERIFICAR FEATURES DEL DATA_LOADER (consolidado en un solo mensaje)
        data_loader_features = ['is_home', 'Height_Inches', 'Weight', 'BMI']
        available_features = [f for f in data_loader_features if f in df.columns]
        missing_features = [f for f in data_loader_features if f not in df.columns]
        
        if missing_features:
            logger.debug(f"Features faltantes: {missing_features}")
        
        # Trabajar directamente con el DataFrame
        if df.empty:
            return []
        
        # PASO 0: Preparación básica (SIEMPRE ejecutar)
        self._ensure_datetime_and_sort(df)
        self._calculate_player_context_features(df)
        
        # GENERAR TODAS LAS FEATURES ESPECIALIZADAS
        self._create_contextual_features_simple(df)
        self._create_performance_features_simple(df)
        self._create_double_double_features_simple(df)
        self._create_statistical_features_simple(df)
        self._create_opponent_features_simple(df)
        self._create_biometric_features_simple(df)
        self._create_game_context_features_advanced(df)
        self._create_dd_specialized_features(df)

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
        
        # Las features especializadas son las que se crearon DESPUÉS de la generación
        specialized_features = list(final_columns - initial_columns)
        
        # Filtrar solo columnas numéricas (excluir categóricas que pudieran haber quedado)
        numerical_features = []
        for feature in specialized_features:
            if df[feature].dtype in ['int64', 'float64', 'int32', 'float32']:
                numerical_features.append(feature)
        
        final_features = numerical_features
        self.feature_columns = final_features
        
        return final_features

    def _create_dd_specialized_features(self, df: pd.DataFrame) -> None:
        """Crear features especializadas para predicción de double double"""
        logger.debug("Generando features especializadas DD...")

        # 3. Proximidad a double-double (usando promedios históricos)
        for stat, threshold in [('rebounds_hist_avg_5g', 10.0), ('points_hist_avg_5g', 10.0), ('assists_hist_avg_5g', 10.0)]:
            proximity_name = stat.replace('_hist_avg_5g', '_dd_proximity')
            if stat in df.columns:
                df[proximity_name] = np.minimum(1.0, df[stat] / threshold)
            else:
                df[proximity_name] = 0.5
        
        # 4. Roles del jugador (usando promedios históricos)
        # Reboteador principal
        if 'rebounds_hist_avg_5g' in df.columns:
            df['primary_rebounder'] = (df['rebounds_hist_avg_5g'] > 8).astype(float)
        else:
            df['primary_rebounder'] = 0.3
        
        # Anotador principal
        if 'points_hist_avg_5g' in df.columns:
            df['primary_scorer'] = (df['points_hist_avg_5g'] > 15).astype(float)
        else:
            df['primary_scorer'] = 0.3
        
        # Armador principal
        if 'assists_hist_avg_5g' in df.columns:
            df['primary_playmaker'] = (df['assists_hist_avg_5g'] > 5).astype(float)
        else:
            df['primary_playmaker'] = 0.2
    
    def _create_contextual_features_simple(self, df: pd.DataFrame) -> None:
        """Features contextuales disponibles antes del juego"""
        
        # Rest advantage específico para double doubles (usando days_rest ya calculado)
        if 'days_rest' in df.columns:
            df['rest_advantage'] = np.where(
                df['days_rest'] == 0, -0.20,  # Penalización back-to-back fuerte
                np.where(df['days_rest'] == 1, -0.08,
                        np.where(df['days_rest'] >= 3, 0.12, 0.0))
            )
        
        # Season progression factor (jugadores mejoran durante temporada)
        if 'month' in df.columns:
            df['season_progression_factor'] = np.where(
                df['month'].isin([10, 11]), -0.05,  # Inicio temporada
                np.where(df['month'].isin([12, 1, 2]), 0.05,  # Mitad temporada
                        np.where(df['month'].isin([3, 4]), 0.02, 0.0))  # Final temporada
            )
        
        # Weekend boost (más energía en fines de semana)
        if 'is_weekend' in df.columns:
            df['weekend_boost'] = df['is_weekend'] * 0.02
    
    def _create_performance_features_simple(self, df: pd.DataFrame) -> None:
        """Features de rendimiento BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas pequeñas para evitar memorización
        basic_windows = [3, 5]  # Reducido de [5, 10] a [3, 5]
        
        # Estadísticas clave para double double
        key_stats = ['points', 'rebounds', 'assists', 'steals', 'blocks', 'minutes']
        
        for window in basic_windows:
            for stat in key_stats:
                if stat in df.columns:
                    # Promedio histórico básico
                    stat_hist_avg = self._get_historical_series(df, stat, window, 'mean')
                    # Limpiar índices duplicados
                    stat_hist_avg = stat_hist_avg[~stat_hist_avg.index.duplicated(keep='first')]
                    df[f'{stat.lower()}_hist_avg_{window}g'] = stat_hist_avg
                    
                    # Consistencia básica (solo para stats principales)
                    if stat in ['points', 'rebounds', 'assists', 'minutes'] and window == 5:
                        stat_std = self._get_historical_series(df, stat, window, 'std', min_periods=2)
                        # Limpiar índices duplicados
                        stat_std = stat_std[~stat_std.index.duplicated(keep='first')]
                        df[f'{stat.lower()}_consistency_{window}g'] = 1 / (stat_std.fillna(1) + 1)
        
        # 1. USAGE RATE HISTÓRICO (SIN DATA LEAKAGE)
        # Calcular usage rate histórico usando solo datos pasados
        if 'field_goals_att' in df.columns and 'free_throws_att' in df.columns and 'turnovers' in df.columns and 'minutes' in df.columns:
            # Calcular usage rate histórico: (FGA + 0.44*FTA + TOV) / MP usando shift(1)
            fga_hist = self._get_historical_series(df, 'field_goals_att', 5, 'mean')
            fta_hist = self._get_historical_series(df, 'free_throws_att', 5, 'mean')
            tov_hist = self._get_historical_series(df, 'turnovers', 5, 'mean')
            mp_hist = self._get_historical_series(df, 'minutes', 5, 'mean')
            
            df['usage_rate_approx'] = (
                fga_hist + 0.44 * fta_hist + tov_hist
            ) / (mp_hist + 0.1)  # Evitar división por 0
            
            # Consistencia de usage rate (ventana de 5 juegos)
            usage_std = self._get_historical_series(df, 'usage_rate_approx', 5, 'std', min_periods=2)
            df['usage_consistency_5g'] = 1 / (usage_std.fillna(1) + 1)
        else:
            # Fallback usando solo FGA histórico si no hay todas las stats
            if 'field_goals_att' in df.columns and 'minutes' in df.columns:
                fga_hist = self._get_historical_series(df, 'field_goals_att', 5, 'mean')
                mp_hist = self._get_historical_series(df, 'minutes', 5, 'mean')
                df['usage_rate_approx'] = fga_hist / (mp_hist + 0.1)
                usage_std = self._get_historical_series(df, 'usage_rate_approx', 5, 'std', min_periods=2)
                df['usage_consistency_5g'] = 1 / (usage_std.fillna(1) + 1)
            else:
                df['usage_consistency_5g'] = 0.5  # Valor neutral
        
        # 3. FEATURES DE RENDIMIENTO RELATIVO (SIN DATA LEAKAGE)
        # Comparar rendimiento reciente vs rendimiento más antiguo (sin usar datos actuales)
        for window in [5, 10]:
            for stat in ['points', 'rebounds', 'assists']:
                if stat in df.columns:
                    # Usar ventanas históricas diferentes para comparar
                    recent_avg = self._get_historical_series(df, stat, 3, 'mean')  # Últimos 3 juegos
                    older_avg = self._get_historical_series(df, stat, window, 'mean')  # Ventana más larga
                    
                    # Feature de si el rendimiento reciente está por encima del promedio más antiguo
                    df[f'{stat.lower()}_above_historical_{window}g'] = (
                        recent_avg > older_avg
                    ).astype(int)
        
        # 4. FEATURES DE MOMENTUM AVANZADO
        for stat in ['points', 'rebounds', 'assists']:
            if stat in df.columns:
                # Momentum: últimos 3 juegos vs anteriores 3 juegos
                recent_avg = self._get_historical_series(df, stat, 3, 'mean')
                older_avg = df.groupby('player')[stat].shift(3).rolling(window=3, min_periods=1).mean()
                df[f'{stat.lower()}_momentum_6g'] = recent_avg - older_avg.fillna(0)
        
        # 5. FEATURES DE COMBINACIÓN ESTADÍSTICA AVANZADA
        if all(col in df.columns for col in ['points', 'rebounds', 'assists']):
            # Potencial de double-double basado en dos stats más altas
            df['dd_potential_score'] = np.maximum(
                df['points_hist_avg_5g'] + df['rebounds_hist_avg_5g'],
                np.maximum(
                    df['points_hist_avg_5g'] + df['assists_hist_avg_5g'],
                    df['rebounds_hist_avg_5g'] + df['assists_hist_avg_5g']
                )
            )
        
        # 6. FEATURES DE TENDENCIA TEMPORAL
        for stat in ['points', 'rebounds', 'assists']:
            if stat in df.columns:
                # Tendencia: diferencia entre promedio reciente vs histórico
                recent_3g = self._get_historical_series(df, stat, 3, 'mean')
                historical_10g = self._get_historical_series(df, stat, 10, 'mean')
                df[f'{stat.lower()}_trend_factor'] = recent_3g - historical_10g
        
        # 7. FEATURES DE ESTABILIDAD DE RENDIMIENTO
        if 'minutes' in df.columns:
            # Estabilidad de minutos (importante para oportunidades)
            mp_cv = self._get_historical_series(df, 'minutes', 5, 'std') / (
                self._get_historical_series(df, 'minutes', 5, 'mean') + 0.1
            )
            df['minutes_stability'] = 1 / (mp_cv.fillna(1) + 1)
        
        # 8. FEATURES DE IMPACTO EN EL JUEGO
        if all(col in df.columns for col in ['points', 'rebounds', 'assists', 'steals', 'blocks']):
            # Índice de impacto total
            df['total_impact_5g'] = (
                df['points_hist_avg_5g'] + 
                df['rebounds_hist_avg_5g'] + 
                df['assists_hist_avg_5g'] + 
                self._get_historical_series(df, 'steals', 5, 'mean').fillna(0) +
                self._get_historical_series(df, 'blocks', 5, 'mean').fillna(0)
            )
    
    def _create_double_double_features_simple(self, df: pd.DataFrame) -> None:
        """Features de double double BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Double double rate histórico básico
            df[f'dd_rate_{window}g'] = (
                df.groupby('player')['double_double'].shift(1)
                .rolling(window=window, min_periods=1).mean()
            ).fillna(0.1)  # Default bajo para nuevos jugadores
            
            # Weighted double double rate básico (solo para ventana de 5)
            if window == 5:
                dd_shifted = df.groupby('player')['double_double'].shift(1).fillna(0)
                
                def simple_weighted_mean(x):
                    try:
                        x_clean = pd.to_numeric(x, errors='coerce').dropna()
                        if len(x_clean) == 0:
                            return 0.1
                        # Pesos simples: más reciente = más peso
                        weights = np.linspace(0.5, 1.0, len(x_clean))
                        weights = weights / weights.sum()
                        return float(np.average(x_clean, weights=weights))
                    except:
                        return 0.1
                
                df[f'weighted_dd_rate_{window}g'] = (
                    dd_shifted.rolling(window=window, min_periods=1)
                    .apply(simple_weighted_mean, raw=False)
                )
                
                # Double double momentum básico
                if window >= 5:
                    first_half = dd_shifted.rolling(window=3, min_periods=1).mean()
                    second_half = dd_shifted.shift(2).rolling(window=3, min_periods=1).mean()
                    df[f'dd_momentum_{window}g'] = first_half - second_half
        
        # Racha actual de double doubles - CORREGIDO
        def calculate_streak_for_group(group):
            """Calcular racha para un grupo de jugador"""
            # Usar double_double con shift(1) para evitar data leakage
            dd_series = group['double_double'].shift(1)
            streaks = []
            
            for i in range(len(group)):
                if i == 0:
                    streaks.append(0)  # Primer juego no tiene historial
                else:
                    # Obtener valores históricos hasta este punto
                    historical_values = dd_series.iloc[:i].dropna()
                    if len(historical_values) == 0:
                        streaks.append(0)
                    else:
                        # Calcular racha actual desde el final
                        streak = 0
                        for value in reversed(historical_values.tolist()):
                            if value == 1:
                                streak += 1
                            else:
                                break
                        streaks.append(streak)
            
            return pd.Series(streaks, index=group.index)
        
        try:
            # Crear feature por jugador de forma más robusta
            dd_streak_values = []
            
            for player in df['player'].unique():
                player_data = df[df['player'] == player].copy()
                # Calcular racha para este jugador específico
                if 'double_double' in player_data.columns:
                    # Usar rolling window para obtener la racha
                    dd_shifted = player_data['double_double'].shift(1).fillna(0)
                    streak = dd_shifted.rolling(window=len(dd_shifted), min_periods=1).apply(
                        lambda x: (x[::-1] == 1).cumprod().sum(), raw=False
                    )
                    dd_streak_values.extend(streak.values)
                else:
                    dd_streak_values.extend([0] * len(player_data))
            
            # Asignar directamente sin problemas de índice
            df['dd_streak'] = dd_streak_values
            
        except Exception as e:
            logger.warning(f"Error calculando dd_streak: {str(e)}")
            # Fallback: usar cálculo más simple
            df['dd_streak'] = 0
        
        # Forma reciente (últimos 3 juegos)
        df['recent_dd_form'] = (
            df.groupby('player')['double_double'].shift(1)
            .rolling(window=3, min_periods=1).mean()
        ).fillna(0.1)
    
    def _create_statistical_features_simple(self, df: pd.DataFrame) -> None:
        """Features estadísticas BÁSICAS únicamente - ANTI-OVERFITTING"""
        # Solo ventanas básicas: 5 y 10 juegos
        basic_windows = [5, 10]
        
        for window in basic_windows:
            # Usage rate aproximado (solo si tenemos FGA y FTA)
            if all(col in df.columns for col in ['field_goals_att', 'free_throws_att', 'minutes']):
                # Calcular usage histórico básico
                usage_hist = self._get_historical_series(df, 'field_goals_att', window, 'mean') + \
                           self._get_historical_series(df, 'free_throws_att', window, 'mean') * 0.44
                df[f'usage_hist_{window}g'] = usage_hist
                
                # Consistencia de usage (solo ventana 5)
                if window == 5:
                    usage_std = self._get_historical_series(df, 'field_goals_att', window, 'std', min_periods=2)
                    # Eliminar índices duplicados
                    usage_std = usage_std[~usage_std.index.duplicated(keep='first')]
                    df[f'usage_consistency_{window}g'] = 1 / (usage_std.fillna(1) + 1)
            

    def _create_opponent_features_simple(self, df: pd.DataFrame) -> None:
        """Features de oponente BÁSICAS únicamente - ANTI-OVERFITTING"""
        if 'Opp' not in df.columns:
            return
            
        # Defensive rating del oponente (aproximado usando puntos permitidos)
        if 'points' in df.columns:
            # Calcular puntos promedio permitidos por el oponente
            opp_def_rating = df.groupby('Opp')['points'].transform(
                lambda x: x.shift(1).rolling(10, min_periods=3).mean()
            )
            # Invertir: menos puntos permitidos = mejor defensa = más difícil double double
            df['opponent_def_rating'] = opp_def_rating.fillna(105.0)  # Default NBA average
        
        # Último resultado vs este oponente (para double double)
        df['last_dd_vs_opp'] = df.groupby(['player', 'Opp'])['double_double'].transform(
            lambda x: x.shift(1).tail(1).iloc[0] if len(x.shift(1).dropna()) > 0 else 0.1
        ).fillna(0.1)
        
        # Motivación extra vs rivales específicos
        df['rivalry_motivation'] = np.where(
            df['last_dd_vs_opp'] == 0, 0.05,  # No logró DD último vs este rival
            np.where(df['last_dd_vs_opp'] == 1, -0.02, 0)  # Logró DD último
        )
    
    def _create_biometric_features_simple(self, df: pd.DataFrame) -> None:
        """Features biométricas especializadas para double doubles"""
        if 'Height_Inches' not in df.columns:
            logger.debug("Height_Inches no disponible - saltando features biométricas")
            return
        
        logger.debug("Creando features biométricas especializadas para double doubles")
        
        # 1. Categorización de altura para double doubles
        # Basado en posiciones típicas NBA donde los double doubles son más comunes
        def categorize_height(height):
            if pd.isna(height):
                return 0  # Unknown
            elif height < 72:  # <6'0" - Guards pequeños
                return 1  # Small_Guard
            elif height < 75:  # 6'0"-6'3" - Guards normales
                return 2  # Guard
            elif height < 78:  # 6'3"-6'6" - Wings/Forwards pequeños
                return 3  # Wing
            elif height < 81:  # 6'6"-6'9" - Forwards
                return 4  # Forward
            else:  # >6'9" - Centers/Power Forwards
                return 5  # Big_Man
        
        df['height_category'] = df['Height_Inches'].apply(categorize_height)
        
        # 2. Factor de ventaja para rebotes basado en altura (clipping más suave)
        # Los jugadores más altos tienen ventaja natural para rebotes
        height_normalized = (df['Height_Inches'] - 72) / 12  # Normalizar desde 6'0" base
        df['height_rebounding_factor'] = np.clip(height_normalized * 0.15, -0.1, 0.3)  # Rango más amplio
        
        # 3. Factor de ventaja para bloqueos basado en altura (clipping más suave)
        # Los jugadores más altos bloquean más
        df['height_blocking_factor'] = np.clip(height_normalized * 0.1, -0.05, 0.25)  # Rango más amplio
        
        # 4. Ventaja de altura general para double doubles
        # Combina rebotes y bloqueos - jugadores altos tienen más oportunidades de DD
        df['height_advantage'] = (df['height_rebounding_factor'] + df['height_blocking_factor']) / 2
        
        # 5. Interacción altura-posición (aproximada por Height_Inches)
        # Guards altos y Centers pequeños tienen patrones únicos
        df['height_position_interaction'] = np.where(
            df['Height_Inches'] < 75,  # Guards
            np.where(df['Height_Inches'] > 73, 0.1, 0.0),  # Guards altos (+bonus)
            np.where(df['Height_Inches'] > 80, 0.05, 0.15)  # Centers vs Forwards
        )
        
        # 6. Factor de altura vs peso (si está disponible) para determinar tipo de jugador
        if 'Weight' in df.columns:
            # BMI ya está calculado en data_loader, pero podemos crear factor específico
            height_weight_ratio = df['Weight'] / df['Height_Inches']
            df['build_factor'] = np.where(
                height_weight_ratio > 2.8, 0.1,  # Jugadores "pesados" (más rebotes)
                np.where(height_weight_ratio < 2.4, -0.05, 0.0)  # Jugadores "ligeros"
            )
        
        logger.debug("Features biométricas especializadas creadas")
    
    def _create_game_context_features_advanced(self, df: pd.DataFrame) -> None:
        """Crear features avanzadas de contexto de juego para mejor precision en DD"""
        logger.debug("Creando features avanzadas de contexto de juego...")
        
        # Features de contexto de juego MEJORADAS
        if 'is_home' in df.columns:
            df['home_advantage'] = df['is_home'].astype(int)
            
            # NUEVAS FEATURES DE CONTEXTO
            # Home vs Away performance differential
            if 'double_double' in df.columns:
                home_dd_rate = df[df['is_home'] == 1].groupby('player')['double_double'].mean()
                away_dd_rate = df[df['is_home'] == 0].groupby('player')['double_double'].mean()
                df['home_away_dd_diff'] = df['player'].map(home_dd_rate - away_dd_rate).fillna(0)
        
        # Features de rivalidad y motivación EXPANDIDAS
        if 'Opp' in df.columns:
            # Crear features de rivalidad (equipos de la misma división)
            division_rivals = {
                'ATL': ['MIA', 'ORL', 'CHA', 'WAS'],
                'BOS': ['NYK', 'BRK', 'PHI', 'TOR'],
                'CLE': ['DET', 'IND', 'CHI', 'MIL'],
                'DAL': ['SAS', 'HOU', 'MEM', 'NOP'],
                'DEN': ['UTA', 'POR', 'OKC', 'MIN'],
                'GSW': ['LAC', 'LAL', 'SAC', 'PHX']
            }
            
            df['is_division_rival'] = 0
            for team, rivals in division_rivals.items():
                mask = (df['Team'] == team) & (df['Opp'].isin(rivals))
                df.loc[mask, 'is_division_rival'] = 1
            
            # NUEVAS FEATURES DE OPONENTE (SIN DATA LEAKAGE)
            # Performance vs specific opponent usando solo datos históricos
            if 'double_double' in df.columns:
                # Calcular DD rate vs oponente usando solo datos pasados (shift(1))
                df_temp = df.copy()
                df_temp['double_double_shifted'] = df_temp.groupby('player')['double_double'].shift(1)
                opp_dd_rate = df_temp.groupby(['player', 'Opp'])['double_double_shifted'].mean().to_dict()
                # Usar apply con tuplas para evitar problemas de índice
                df['vs_opp_dd_rate'] = df.apply(lambda row: opp_dd_rate.get((row['player'], row['Opp']), 0.1), axis=1)
            
            # Strong vs weak opponents
            strong_teams = ['BOS', 'MIL', 'PHI', 'CLE', 'NYK', 'DEN', 'MEM', 'SAC', 'PHX', 'LAC']
            df['vs_strong_opponent'] = df['Opp'].isin(strong_teams).astype(int)
        
        # Features de forma reciente MEJORADAS
        if 'double_double' in df.columns:
            # Forma reciente en double-doubles (últimos 3 vs anteriores 7)
            recent_dd_rate = df.groupby('player')['double_double'].shift(1).rolling(window=3, min_periods=1).mean()
            older_dd_rate = df.groupby('player')['double_double'].shift(4).rolling(window=7, min_periods=1).mean()
            df['dd_form_trend'] = recent_dd_rate - older_dd_rate.fillna(0)
            
            # DD en últimos 2 juegos
            dd_shifted = df.groupby('player')['double_double'].shift(1).fillna(0)
            df['dd_last_2_games'] = dd_shifted.rolling(window=2, min_periods=1).sum()
        
        logger.debug("Features avanzadas de contexto de juego creadas")