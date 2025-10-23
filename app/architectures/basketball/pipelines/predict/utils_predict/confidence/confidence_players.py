"""
Confidence Predict
=================

Modulo exclusivamente para el calculo de confianza en las predicciones inferidas
por los modelos tanto para jugadores como para equipos.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from difflib import SequenceMatcher
import sys
import os
import logging

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Importar modelos y data loaders
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils

logger = logging.getLogger(__name__)

class PlayersConfidence:
    """
    Clase centralizada para calcular la confianza en las predicciones de jugadores
    
    Maneja todos los cálculos de confianza para:
    - pts_predictor (puntos)
    - ast_predict (asistencias)  
    - trb_predict (rebotes)
    - 3pt_predict (triples)
    - dd_predict (double double)
    """
    def __init__(self):
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.is_loaded = False
        
        # Umbrales de confianza para jugadores
        self.base_tolerance = -1
        self.high_confidence_threshold = 75.0
        self.ultra_confidence_threshold = 85.0
        self.min_confidence_threshold = 50.0

    def load_data(self):
        """Cargar datos históricos si no están cargados"""
        if not self.is_loaded:
            try:
                data_loader = NBADataLoader(
                    players_total_path="app/architectures/basketball/data/players_total.csv",
                    players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                    teams_total_path="app/architectures/basketball/data/teams_total.csv",
                    teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                    biometrics_path="app/architectures/basketball/data/biometrics.csv"
                )
                self.historical_players, self.historical_teams, self.historical_players_quarters, self.historical_teams_quarters = data_loader.load_data()
                self.is_loaded = True
                logger.info(" PlayersConfidence: Datos históricos cargados correctamente")
            except Exception as e:
                logger.error(f" Error cargando datos históricos en PlayersConfidence: {e}")
                self.is_loaded = False

    def calculate_player_confidence(self, raw_prediction: float, stabilized_prediction: float, 
                                  tolerance: float, prediction_std: float, actual_stats_std: float,
                                  historical_games: int, player_data: Dict[str, Any], 
                                  opponent_team: str = None, game_date: str = None, 
                                  game_data: Dict = None, target_stat: str = 'points') -> float:
        """
        Calcular confianza SIMPLIFICADA para predicciones de JUGADORES
        
        SISTEMA SIMPLIFICADO - Solo 4 factores:
        1. HISTÓRICO (40% peso) - Cantidad de datos disponibles
        2. HOME (10% peso) - Jugar en casa (+10%)
        3. ESTABILIDAD (30% peso) - Consistencia histórica del jugador
        4. CONSISTENCIA (20% peso) - Variabilidad reciente
        
        Args:
            target_stat: Estadística objetivo ('points', 'rebounds', 'assists', 'triples', 'DD')
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            player_name = player_data.get('player_name', player_data.get('player', player_data.get('Player', 'Unknown')))
            
            # USAR MÉTODOS EXISTENTES: Obtener información actualizada desde SportRadar si está disponible
            if game_data and player_name != 'Unknown':
                # OPTIMIZADO: Usar métodos existentes de CommonUtils
                is_home_from_sr = self.common_utils._get_is_home_from_sportradar(game_data, player_name)
                player_status_from_sr = self.common_utils._get_player_status_from_sportradar(game_data, player_name)
                current_team_from_sr = self.common_utils._get_current_team_from_sportradar(game_data, player_name)
                
                # Actualizar información del jugador con datos más recientes
                player_data['is_home'] = is_home_from_sr
                player_data['current_status'] = player_status_from_sr
                player_data['current_team'] = current_team_from_sr
                
                logger.debug(f" Info actualizada para {player_name}: "
                           f"home={is_home_from_sr}, status={player_status_from_sr}, team={current_team_from_sr}")
            
            # ===== SISTEMA SIMPLIFICADO DE CONFIANZA =====
            # Solo 4 factores principales como solicitado
            
            # 1. FACTOR HISTÓRICO (40% peso) - Cantidad de datos disponibles
            if historical_games >= 30:
                historical_confidence = 95
            elif historical_games >= 25:
                historical_confidence = 90
            elif historical_games >= 20:
                historical_confidence = 85
            elif historical_games >= 15:
                historical_confidence = 80
            elif historical_games >= 10:
                historical_confidence = 70
            else:
                historical_confidence = max(50, historical_games * 5)
            
            # 2. FACTOR HOME (10% peso) - Jugar en casa
            is_home = player_data.get('is_home', 0)
            home_confidence = 10 if is_home == 1 else 0
            
            # 3. FACTOR ESTABILIDAD (30% peso) - Consistencia histórica del jugador
            if actual_stats_std > 0:
                stability_confidence = max(0, 100 - (actual_stats_std * 2.5))  # Menor std = mayor confianza
            else:
                stability_confidence = 90
            
            # 4. FACTOR CONSISTENCIA (20% peso) - Variabilidad reciente
            consistency_confidence = 70  # Valor base
            try:
                if self.historical_players is not None and player_name != 'Unknown':
                    # Buscar datos históricos del jugador
                    player_historical = self.common_utils._smart_player_search(self.historical_players, player_name)
                    
                    if not player_historical.empty and len(player_historical) >= 5:
                        # Analizar últimos 10 juegos usando la estadística objetivo
                        recent_games = player_historical.tail(10)
                        if len(recent_games) >= 5:
                            # Usar la columna correspondiente al target_stat
                            stat_column = self._get_stat_column(target_stat)
                            if stat_column in recent_games.columns:
                                recent_std = recent_games[stat_column].std()
                                
                                # Menor variabilidad reciente = mayor consistencia
                                if recent_std > 0:
                                    consistency_confidence = max(0, 100 - (recent_std * 3))
                                else:
                                    consistency_confidence = 95
                                
                                logger.debug(f" Consistencia reciente {player_name} en {target_stat}: "
                                           f"std={recent_std:.1f}, confianza={consistency_confidence:.1f}")
                            else:
                                logger.debug(f" Columna {stat_column} no encontrada para {target_stat}")
            except Exception as e:
                logger.debug(f" Error calculando consistencia para {player_name}: {e}")
                consistency_confidence = 70
            
            # CALCULAR CONFIANZA SIMPLIFICADA
            weighted_confidence = (
                historical_confidence * 0.40 +    # 40% - Factor histórico
                home_confidence +                 # 10% - Factor home
                stability_confidence * 0.30 +     # 30% - Factor estabilidad
                consistency_confidence * 0.20     # 20% - Factor consistencia
            )
            
            # APLICAR LÍMITES REALISTAS (60% - 95%)
            final_confidence = max(60.0, min(95.0, weighted_confidence))
            
            logger.info(f" Confianza SIMPLIFICADA {player_name}: {final_confidence:.1f}% "
                       f"(histórico:{historical_confidence:.0f}, home:{home_confidence:.0f}, "
                       f"estabilidad:{stability_confidence:.0f}, consistencia:{consistency_confidence:.0f})")
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f" Error calculando confianza player: {e}")
            return 70.0

    def _get_stat_column(self, target_stat: str) -> str:
        """
        Mapear target_stat a la columna correspondiente en el dataset
        
        Args:
            target_stat: Estadística objetivo ('points', 'rebounds', 'assists', 'triples', 'DD')
            
        Returns:
            Nombre de la columna en el dataset
        """
        stat_mapping = {
            'points': 'points',                    # Columna 37: points
            'rebounds': 'rebounds',                # Columna 27: rebounds
            'assists': 'assists',                  # Columna 28: assists
            'triples': 'three_points_made',        # Columna 15: three_points_made
            'DD': 'double_double'                  # Columna 39: double_double
        }
        
        return stat_mapping.get(target_stat, 'points')  # Default a 'points' si no se encuentra

    def calculate_player_vs_opponent_factor(self, player_name: str, opponent_team: str, 
                                          player_position: str = 'G', max_games: int = 50) -> float:
        """
        Calcular factor de rendimiento del jugador vs equipo oponente específico
        
        Args:
            player_name: Nombre del jugador
            opponent_team: Equipo oponente (abreviación)
            player_position: Posición del jugador
            max_games: Máximo de juegos históricos a considerar
            
        Returns:
            Factor multiplicativo (0.8 - 1.2)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # Buscar enfrentamientos históricos del jugador vs este oponente
            # Primero obtener todos los datos del jugador con búsqueda inteligente
            player_all_data = self.common_utils._smart_player_search(self.historical_players, player_name)
            if player_all_data.empty:
                logger.info(f"Jugador {player_name} no encontrado en datos históricos")
                return 1.0
            
            # Filtrar por oponente específico
            player_vs_opp = player_all_data[
                player_all_data['Opp'] == opponent_team
            ].copy()
            
            if len(player_vs_opp) == 0:
                logger.info(f"Sin datos H2H específicos {player_name} vs {opponent_team}")
                return 1.0
            
            # Usar todos los juegos vs el oponente para mejor precisión
            player_vs_opp = player_vs_opp.sort_values('Date')
            
            # Calcular promedio de rendimiento vs este oponente
            vs_opp_stats = {
                'pts_mean': player_vs_opp['points'].mean(),
                'ast_mean': player_vs_opp['assists'].mean(),
                'trb_mean': player_vs_opp['rebounds'].mean(),
                'threep_mean': player_vs_opp['three_points_made'].mean(),
                'games': len(player_vs_opp)
            }
            
            # Calcular promedio general del jugador (todos los juegos para mejor precisión)
            player_general = player_all_data.sort_values('Date')
            
            if len(player_general) == 0:
                return 1.0
                
            general_stats = {
                'pts_mean': player_general['points'].mean(),
                'ast_mean': player_general['assists'].mean(), 
                'trb_mean': player_general['rebounds'].mean(),
                'threep_mean': player_general['three_points_made'].mean()
            }
            
            # Calcular ratio de rendimiento vs oponente / rendimiento general
            performance_ratios = []
            
            for stat in ['points', 'assists', 'rebounds', 'three_points_made']:
                vs_opp_val = vs_opp_stats[f'{stat}_mean']
                general_val = general_stats[f'{stat}_mean']
                
                if general_val > 0:
                    ratio = vs_opp_val / general_val
                    performance_ratios.append(ratio)
            
            if performance_ratios:
                avg_ratio = np.mean(performance_ratios)
                # Limitar el factor entre 0.8 y 1.2
                factor = max(0.8, min(1.2, avg_ratio))
                
                logger.info(f" {player_name} vs {opponent_team}: factor {factor:.3f} "
                           f"({vs_opp_stats['games']} juegos H2H)")
                return factor
            
            return 1.0
            
        except Exception as e:
            logger.warning(f"Error calculando factor vs oponente: {e}")
            return 1.0

    def calculate_player_h2h_stats(self, player_name: str, opponent_team: str, 
                                 target_stat: str = 'points', max_games: int = 50) -> Dict[str, Any]:
        """
        Calcular estadísticas H2H detalladas del jugador vs equipo oponente
        
        Args:
            player_name: Nombre del jugador
            opponent_team: Equipo oponente (abreviación)
            target_stat: Estadística objetivo (points, assists, rebounds, three_points_made, etc.)
            max_games: Máximo de juegos históricos a considerar
            
        Returns:
            Dict con estadísticas H2H detalladas
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # Buscar enfrentamientos históricos del jugador vs este oponente
            # Primero obtener todos los datos del jugador con búsqueda inteligente
            player_all_data = self.common_utils._smart_player_search(self.historical_players, player_name)
            if player_all_data.empty:
                logger.info(f"Jugador {player_name} no encontrado en datos históricos")
                return {
                    'games_found': 0,
                    'h2h_mean': None,
                    'h2h_std': None,
                    'h2h_min': None,
                    'h2h_max': None,
                    'h2h_factor': 1.0,
                    'consistency_score': 0,
                    'last_5_mean': None,
                    'last_10_mean': None
                }
            
            # Filtrar por oponente específico
            player_vs_opp = player_all_data[
                player_all_data['Opp'] == opponent_team
            ].copy()
            
            if len(player_vs_opp) == 0:
                return {
                    'games_found': 0,
                    'h2h_mean': None,
                    'h2h_std': None,
                    'h2h_min': None,
                    'h2h_max': None,
                    'h2h_factor': 1.0,
                    'consistency_score': 0,
                    'last_5_mean': None,
                    'last_10_mean': None
                }
            
            # Ordenar por fecha (más reciente primero)
            if 'Date' in player_vs_opp.columns:
                player_vs_opp = player_vs_opp.sort_values('Date', ascending=False)
            
            # Limitar a máximo de juegos
            h2h_recent = player_vs_opp.sort_values('Date')
            
            # Filtrar estadística objetivo válida
            if target_stat not in h2h_recent.columns:
                return {
                    'games_found': 0,
                    'h2h_mean': None,
                    'h2h_std': None,
                    'h2h_min': None,
                    'h2h_max': None,
                    'h2h_factor': 1.0,
                    'consistency_score': 0,
                    'last_5_mean': None,
                    'last_10_mean': None
                }
            
            h2h_stats = h2h_recent[target_stat].dropna()
            
            if len(h2h_stats) == 0:
                return {
                    'games_found': 0,
                    'h2h_mean': None,
                    'h2h_std': None,
                    'h2h_min': None,
                    'h2h_max': None,
                    'h2h_factor': 1.0,
                    'consistency_score': 0,
                    'last_5_mean': None,
                    'last_10_mean': None
                }
            
            # Calcular estadísticas H2H
            h2h_mean = h2h_stats.mean()
            h2h_std = h2h_stats.std() if len(h2h_stats) > 1 else 0
            h2h_min = h2h_stats.min()
            h2h_max = h2h_stats.max()
            
            # Últimos 5 juegos
            last_5 = h2h_stats.head(5) if len(h2h_stats) >= 5 else h2h_stats
            last_5_mean = last_5.mean() if len(last_5) > 0 else None
            
            # Últimos 10 juegos
            last_10 = h2h_stats.head(10) if len(h2h_stats) >= 10 else h2h_stats
            last_10_mean = last_10.mean() if len(last_10) > 0 else None
            
            # Calcular factor H2H basado en rendimiento relativo (CONSERVADOR)
            # Usar todos los juegos para mejor precisión
            player_general = player_all_data.sort_values('Date', ascending=False)
            
            if len(player_general) > 0 and target_stat in player_general.columns:
                general_mean = player_general[target_stat].dropna().mean()
                if general_mean > 0:
                    # Factor H2H conservador para evitar sobreajuste
                    raw_factor = h2h_mean / general_mean
                    
                    # Aplicar límites conservadores más estrictos para jugadores elite
                    if general_mean >= 20:  # Jugadores elite (20+ pts promedio)
                        # Para jugadores elite, ser más conservador con H2H
                        if len(h2h_stats) < 5:
                            h2h_factor = min(raw_factor, 1.2)  # Máximo 20% de boost
                        elif len(h2h_stats) < 10:
                            h2h_factor = min(raw_factor, 1.25)  # Máximo 25% de boost
                        else:
                            h2h_factor = min(raw_factor, 1.15)  # Máximo 15% de boost
                    else:
                        # Para jugadores normales, mantener límites originales
                        if len(h2h_stats) < 5:
                            h2h_factor = min(raw_factor, 1.5)  # Máximo 50% de boost
                        elif len(h2h_stats) < 10:
                            h2h_factor = min(raw_factor, 1.8)  # Máximo 80% de boost
                        else:
                            h2h_factor = min(raw_factor, 2.0)  # Máximo 100% de boost
                    
                    # Aplicar factor de confianza basado en consistencia
                    if h2h_std > 0:
                        consistency_factor = max(0.5, 1 - (h2h_std / h2h_mean))  # Penalizar alta variabilidad
                        h2h_factor = 1 + (h2h_factor - 1) * consistency_factor
                    
                    # Asegurar que el factor esté en rango razonable
                    if general_mean >= 20:  # Jugadores elite
                        h2h_factor = max(0.8, min(h2h_factor, 1.2))  # Entre 0.8x y 1.2x para elite (menos conservador)
                    else:
                        h2h_factor = max(0.5, min(h2h_factor, 2.0))  # Entre 0.5x y 2.0x para normales
                else:
                    h2h_factor = 1.0
            else:
                h2h_factor = 1.0
            
            # Calcular consistencia (inversa de la desviación estándar)
            consistency_score = max(0, 100 - (h2h_std * 3)) if h2h_std > 0 else 100
            
            return {
                'games_found': len(h2h_stats),
                'h2h_mean': round(float(h2h_mean), 1),
                'h2h_std': round(float(h2h_std), 1),
                'h2h_min': round(float(h2h_min), 1),
                'h2h_max': round(float(h2h_max), 1),
                'h2h_factor': round(float(h2h_factor), 3),
                'consistency_score': round(float(consistency_score), 1),
                'last_5_mean': round(float(last_5_mean), 1) if last_5_mean is not None else None,
                'last_10_mean': round(float(last_10_mean), 1) if last_10_mean is not None else None
            }
            
        except Exception as e:
            logger.warning(f"Error calculando estadísticas H2H del jugador: {e}")
            return {
                'games_found': 0,
                'h2h_mean': None,
                'h2h_std': None,
                'h2h_min': None,
                'h2h_max': None,
                'h2h_factor': 1.0,
                'consistency_score': 0,
                'last_5_mean': None,
                'last_10_mean': None
            }

    def get_player_position_impact(self, position: str, opponent_team: str) -> float:
        """
        Calcular impacto de la posición del jugador vs fortalezas defensivas del oponente
        
        Args:
            position: Posición del jugador (G, F, C)
            opponent_team: Equipo oponente
            
        Returns:
            Factor multiplicativo basado en debilidades defensivas
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # Analizar rendimiento defensivo del oponente por posición
            opp_defense = self.historical_teams[
                self.historical_teams['Team'] == opponent_team
            ].copy()
            
            if len(opp_defense) == 0:
                return 1.0
            
            # Tomar datos recientes del oponente
            opp_recent = opp_defense.sort_values('Date').tail(15)
            
            # Analizar puntos permitidos (aproximación por posición)
            points_allowed = opp_recent['points_against'].mean()
            
            # Factor basado en defensa del oponente
            if points_allowed > 115:  # Defensa débil
                return 1.1
            elif points_allowed < 105:  # Defensa fuerte
                return 0.95
            else:
                return 1.0
                
        except Exception as e:
            logger.warning(f"Error calculando impacto posición: {e}")
            return 1.0

    def identify_current_active_players(self, team_name: str, lookback_games: int = 20) -> List[str]:
        """
        OPTIMIZADO: Identificar jugadores ACTIVOS del equipo actual usando CommonUtils
        
        Reutiliza métodos de CommonUtils para evitar duplicación de código
        
        Args:
            team_name: Nombre del equipo
            lookback_games: Juegos a analizar por jugador
            
        Returns:
            Lista de nombres de jugadores activos verificados
        """
        if not self.is_loaded:
            self.load_data()
        
        try:
            # USAR MÉTODO EXISTENTE: Convertir nombre del equipo a abreviación
            team_abbrev = self.common_utils._get_team_abbreviation(team_name)
            
            # Filtrar jugadores del equipo específico
            team_players = self.historical_players[
                self.historical_players['Team'] == team_abbrev
            ].copy()
            
            if len(team_players) == 0:
                logger.warning(f"No hay datos históricos para jugadores de {team_name}")
                return []
            
            # ALGORITMO MEJORADO: Solo jugadores ACTUALES del equipo específico
            if 'Date' in team_players.columns:
                team_players = team_players.sort_values('Date', ascending=False)
                
                # Obtener fecha más reciente del dataset para determinar jugadores actuales
                cutoff_date = team_players['Date'].iloc[0] if len(team_players) > 0 else None
                if cutoff_date:
                    if isinstance(cutoff_date, str):
                        cutoff_date = pd.to_datetime(cutoff_date)
                    
                    # PASO 1: Filtrar jugadores que jugaron RECIENTEMENTE (últimos 30 días)
                    current_season_threshold = cutoff_date - pd.Timedelta(days=30)
                    current_season_players = team_players[team_players['Date'] >= current_season_threshold]
                    
                    # PASO 2: Para cada jugador, verificar que su ÚLTIMO equipo sea el target
                    verified_current_players = []
                    
                    for player_name in current_season_players['player'].unique():
                        # USAR MÉTODO EXISTENTE: Búsqueda inteligente de jugador
                        player_all_data = self.common_utils._smart_player_search(self.historical_players, player_name)
                        
                        if not player_all_data.empty:
                            # Ordenar por fecha más reciente
                            player_all_data = player_all_data.sort_values('Date', ascending=False)
                            
                            # El equipo más reciente del jugador
                            latest_team = player_all_data['Team'].iloc[0]
                            latest_date = player_all_data['Date'].iloc[0]
                            
                            # CRITERIOS PARA SER CONSIDERADO JUGADOR ACTUAL:
                            # 1. Su último equipo coincide con el target
                            # 2. Su último juego fue en los últimos 45 días (jugador activo)
                            if isinstance(latest_date, str):
                                latest_date = pd.to_datetime(latest_date)
                            
                            days_since_last_game = (cutoff_date - latest_date).days
                            is_current_team = latest_team == team_abbrev
                            
                            # USAR MÉTODO EXISTENTE: Verificar si estamos en off-season
                            is_off_season = self.is_off_season_period(cutoff_date)
                            max_days_inactive = 90 if is_off_season else 45
                            is_active_player = days_since_last_game <= max_days_inactive
                            
                            if is_current_team and is_active_player:
                                # Tomar solo datos del jugador en este equipo específico
                                player_team_data = player_all_data[player_all_data['Team'] == team_abbrev].head(lookback_games)
                                
                                if len(player_team_data) >= 5:  # Mínimo 5 juegos para considerarlo
                                    verified_current_players.append(player_name)
                                    logger.debug(f" {player_name}: Jugador actual de {team_abbrev} "
                                               f"(último juego: {days_since_last_game} días atrás)")
                                else:
                                    logger.debug(f" {player_name}: Pocos datos en {team_abbrev} ({len(player_team_data)} juegos)")
                            else:
                                logger.debug(f" {player_name}: No es jugador actual "
                                           f"(equipo: {latest_team}, días: {days_since_last_game})")
                    
                    if verified_current_players:
                        logger.info(f" {team_name}: {len(verified_current_players)} jugadores actuales verificados")
                        return verified_current_players
                    else:
                        # Fallback más conservador si no encuentra jugadores actuales
                        logger.warning(f" No se encontraron jugadores actuales para {team_name}, usando fallback")
                        return team_players['player'].unique().tolist()[:10]
                else:
                    return team_players['player'].unique().tolist()[:10]
            else:
                # Sin columna Date, usar datos más recientes disponibles
                return team_players['player'].unique().tolist()[:10]
                
        except Exception as e:
            logger.error(f"Error identificando jugadores actuales: {e}")
            return []

    def calculate_biometric_matchup_factor(self, player_name: str, opponent_team: str, 
                                         game_data: Dict = None) -> float:
        """
        OPTIMIZADO: Factor biométrico usando métodos existentes de CommonUtils
        
        Args:
            player_name: Nombre del jugador
            opponent_team: Equipo oponente
            game_data: Datos del juego (opcional, para obtener posición actual)
            
        Returns:
            Factor multiplicativo basado en ventajas biométricas (0.9 - 1.1)
        """
        if not self.is_loaded:
            self.load_data()
        
        try:
            # USAR MÉTODO EXISTENTE: Búsqueda inteligente de jugador
            player_data_df = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if player_data_df.empty:
                return 1.0
            
            player_data = player_data_df.iloc[-1]  # Datos más recientes
            
            # USAR MÉTODO EXISTENTE: Obtener posición desde SportRadar si está disponible
            if game_data:
                player_position = self.common_utils._get_player_position_from_sportradar(game_data, player_name)
            else:
                player_position = player_data.get('position', 'G')
                
            player_height = player_data.get('Height_inches', 0)
            player_weight = player_data.get('Weight', 0)
            player_bmi = player_data.get('BMI', 0)
            
            if player_height == 0 or player_weight == 0:
                return 1.0
            
            # USAR MÉTODO EXISTENTE: Convertir nombre del equipo oponente
            opponent_abbrev = self.common_utils._get_team_abbreviation(opponent_team)
            opponent_players = self.historical_players[
                self.historical_players['Team'] == opponent_abbrev
            ].copy()
            
            if len(opponent_players) == 0:
                return 1.0
            
            # Filtrar por posición similar para comparación
            position_mapping = {
                'PG': ['PG', 'G'], 'SG': ['SG', 'G'], 'G': ['PG', 'SG', 'G'],
                'SF': ['SF', 'F'], 'PF': ['PF', 'F'], 'F': ['SF', 'PF', 'F'],
                'C': ['C']
            }
            
            similar_positions = position_mapping.get(player_position, [player_position])
            opponent_same_pos = opponent_players[
                opponent_players['position'].isin(similar_positions)
            ]
            
            if len(opponent_same_pos) == 0:
                return 1.0
            
            # Calcular promedios del oponente en esa posición
            avg_opp_height = opponent_same_pos['Height_inches'].mean()
            avg_opp_weight = opponent_same_pos['Weight'].mean()
            avg_opp_bmi = opponent_same_pos['BMI'].mean()
            
            # Calcular ventajas
            height_advantage = (player_height - avg_opp_height) / avg_opp_height if avg_opp_height > 0 else 0
            weight_advantage = (player_weight - avg_opp_weight) / avg_opp_weight if avg_opp_weight > 0 else 0
            
            # Factor basado en ventajas posicionales
            factor = 1.0
            
            # Para Centers y Forwards: altura y peso importan más
            if player_position in ['C', 'PF', 'F']:
                if height_advantage > 0.05:  # 5% más alto
                    factor += 0.03
                elif height_advantage < -0.05:  # 5% más bajo
                    factor -= 0.03
                    
                if weight_advantage > 0.1:  # 10% más pesado
                    factor += 0.02
                    
            # Para Guards: agilidad (menos peso) puede ser ventaja
            elif player_position in ['PG', 'SG', 'G']:
                if weight_advantage < -0.05:  # Más ligero
                    factor += 0.02
                    
            # Limitar factor entre 0.9 y 1.1
            factor = max(0.9, min(1.1, factor))
            
            logger.info(f" Factor biométrico {player_name} ({player_position}) vs {opponent_team}: {factor:.3f}")
            return factor
            
        except Exception as e:
            logger.warning(f"Error calculando factor biométrico: {e}")
            return 1.0

    def get_player_consistency_score(self, player_name: str, stat_type: str = 'points', 
                                   last_n_games: int = 15) -> float:
        """
        OPTIMIZADO: Score de consistencia usando métodos existentes de CommonUtils
        
        Args:
            player_name: Nombre del jugador
            stat_type: Tipo de estadística ('points', 'assists', 'rebounds', 'three_points_made')
            last_n_games: Número de juegos recientes a analizar
            
        Returns:
            Score de consistencia (0-100)
        """
        if not self.is_loaded:
            self.load_data()
        
        try:
            # USAR MÉTODO EXISTENTE: Búsqueda inteligente de jugador
            player_data_df = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if player_data_df.empty:
                return 50.0  # Score neutro si no se encuentra el jugador
            
            # Tomar últimos N juegos del jugador
            player_data = player_data_df.sort_values('Date').tail(last_n_games)
            
            if len(player_data) < 5:
                return 50.0  # Score neutro si no hay suficientes datos
            
            if stat_type not in player_data.columns:
                return 50.0
            
            stat_values = player_data[stat_type].dropna()
            
            if len(stat_values) < 5:
                return 50.0
            
            # Calcular métricas de consistencia
            mean_stat = stat_values.mean()
            std_stat = stat_values.std()
            
            if mean_stat == 0:
                return 50.0
            
            # Coeficiente de variación (menor = más consistente)
            cv = std_stat / mean_stat if mean_stat > 0 else 1.0
            
            # Convertir a score 0-100 (menor CV = mayor score)
            consistency_score = max(0, min(100, 100 - (cv * 50)))
            
            logger.debug(f" Consistencia {player_name} en {stat_type}: {consistency_score:.1f}")
            return consistency_score
            
        except Exception as e:
            logger.warning(f"Error calculando consistencia del jugador: {e}")
            return 50.0

    def calculate_injury_recovery_factor(self, player_name: str, game_date: str = None) -> float:
        """
        OPTIMIZADO: Factor de lesión usando métodos existentes de CommonUtils
        
        Args:
            player_name: Nombre del jugador
            game_date: Fecha del juego a predecir (formato YYYY-MM-DD)
            
        Returns:
            Factor de ajuste (0.7 - 1.0) basado en tiempo de ausencia
        """
        if not self.is_loaded:
            self.load_data()
        
        try:
            # USAR MÉTODO EXISTENTE: Búsqueda inteligente de jugador
            player_data_df = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if player_data_df.empty:
                logger.warning(f"No hay datos históricos para {player_name}")
                return 0.9  # Factor conservador por falta de datos
            
            # Ordenar por fecha más reciente
            player_data = player_data_df.sort_values('Date', ascending=False)
            
            # Último juego del jugador en el dataset
            last_game_date = player_data['Date'].iloc[0]
            if isinstance(last_game_date, str):
                last_game_date = pd.to_datetime(last_game_date)
            
            # Fecha del juego a predecir
            if game_date:
                if isinstance(game_date, str):
                    prediction_date = pd.to_datetime(game_date)
                else:
                    prediction_date = game_date
            else:
                # Si no se proporciona fecha, usar fecha actual
                prediction_date = pd.Timestamp.now()
            
            # Calcular días de ausencia
            days_absent = (prediction_date - last_game_date).days
            
            # USAR MÉTODO EXISTENTE: Verificar si estamos en off-season
            is_off_season = self.is_off_season_period(prediction_date)
            
            # LÓGICA DE FACTOR DE LESIÓN AJUSTADA POR TEMPORADA
            if is_off_season:
                # En off-season, ser más permisivo con las ausencias
                if days_absent <= 90:  # Hasta 3 meses es normal en off-season
                    injury_factor = 1.0
                    logger.debug(f"🏖️ {player_name}: Off-season normal ({days_absent} días)")
                elif days_absent <= 120:  # 4 meses
                    injury_factor = 0.95
                    logger.info(f"🏖️ {player_name}: Off-season extendido ({days_absent} días) - Factor: {injury_factor}")
                else:
                    injury_factor = 0.8  # Ausencia muy prolongada incluso en off-season
                    logger.warning(f"🚨 {player_name}: Ausencia excesiva en off-season ({days_absent} días) - Factor: {injury_factor}")
            else:
                # En temporada regular, criterios más estrictos
                if days_absent <= 3:
                    injury_factor = 1.0
                    logger.debug(f" {player_name}: Jugando regularmente ({days_absent} días)")
                elif 4 <= days_absent <= 7:
                    injury_factor = 0.95
                    logger.info(f" {player_name}: Ausencia corta ({days_absent} días) - Factor: {injury_factor}")
                elif 8 <= days_absent <= 20:
                    injury_factor = 0.85
                    logger.info(f" {player_name}: Ausencia media ({days_absent} días) - Factor: {injury_factor}")
                elif 21 <= days_absent <= 45:
                    injury_factor = 0.75
                    logger.warning(f"🚨 {player_name}: Ausencia larga ({days_absent} días) - Factor: {injury_factor}")
                else:
                    injury_factor = 0.7
                    logger.warning(f"🚨 {player_name}: Ausencia muy larga ({days_absent} días) - Factor: {injury_factor}")
            
            return injury_factor
            
        except Exception as e:
            logger.error(f"Error calculando factor de lesión para {player_name}: {e}")
            return 0.9  # Factor conservador por error

    def filter_available_players_from_roster(self, roster_data: Dict) -> List[str]:
        """
        OPTIMIZADO: Filtrar jugadores disponibles usando métodos existentes de CommonUtils
        
        Args:
            roster_data: Diccionario con datos del juego (homeTeam/awayTeam con players)
            
        Returns:
            Lista de nombres de jugadores disponibles para predicción
        """
        available_players = []
        
        try:
            # Procesar ambos equipos
            for team_key in ['homeTeam', 'awayTeam']:
                if team_key in roster_data and 'players' in roster_data[team_key]:
                    team_players = roster_data[team_key]['players']
                    
                    for player in team_players:
                        player_name = player.get('fullName', '')
                        
                        if not player_name:
                            continue
                        
                        # Obtener status del jugador directamente del objeto player
                        player_status = player.get('status', '')
                        
                        # CRITERIOS DE DISPONIBILIDAD OPTIMIZADOS
                        is_active_status = player_status in ['ACT', 'ACTIVE']  # Jugador activo
                        
                        # Verificar lesiones usando datos del roster
                        player_injuries = player.get('injuries', [])
                        has_no_serious_injuries = True
                        
                        if player_injuries:
                            serious_injuries = ['ACL', 'ACHILLES', 'FRACTURE', 'SURGERY']
                            for injury in player_injuries:
                                # Verificar que injury sea un diccionario
                                if isinstance(injury, dict):
                                    injury_type = injury.get('type', '').upper()
                                    if any(serious in injury_type for serious in serious_injuries):
                                        has_no_serious_injuries = False
                                        break
                                elif isinstance(injury, str):
                                    # Si injury es string directamente
                                    injury_type = injury.upper()
                                    if any(serious in injury_type for serious in serious_injuries):
                                        has_no_serious_injuries = False
                                        break
                        
                        # DECISIÓN FINAL
                        if is_active_status and has_no_serious_injuries:
                            available_players.append(player_name)
                            logger.debug(f" {player_name}: Disponible para predicción")
                        else:
                            reason = []
                            if not is_active_status:
                                reason.append(f"status: {player_status}")
                            if not has_no_serious_injuries:
                                reason.append("lesión grave")
                            logger.debug(f" {player_name}: No disponible ({', '.join(reason)})")
            
            logger.info(f" Jugadores disponibles para predicción: {len(available_players)}")
            return available_players
            
        except Exception as e:
            logger.error(f"Error filtrando jugadores disponibles: {e}")
            return []

    def is_off_season_period(self, game_date: str = None) -> bool:
        """
        Verificar si estamos en período de off-season (Julio-Octubre)
        
        Durante off-season, relajar criterios de "jugadores activos"
        ya que ningún jugador tendrá juegos recientes
        
        Args:
            game_date: Fecha del juego (formato YYYY-MM-DD)
            
        Returns:
            True si estamos en off-season
        """
        try:
            if game_date:
                if isinstance(game_date, str):
                    check_date = pd.to_datetime(game_date)
                else:
                    check_date = game_date
            else:
                check_date = pd.Timestamp.now()
            
            # Off-season típico: Julio a Octubre
            off_season_months = [7, 8, 9, 10]
            is_off_season = check_date.month in off_season_months
            
            if is_off_season:
                logger.info(f"🏖️ OFF-SEASON detectado (mes {check_date.month})")
            
            return is_off_season
            
        except Exception as e:
            logger.warning(f"Error verificando off-season: {e}")
            return False

    def _adaptive_prediction_strategy_players(self, raw_prediction: float, actual_stats_mean: float,
                                            confidence: float, prediction_std: float, 
                                            actual_stats_std: float) -> Tuple[float, float]:
        """
        Sistema adaptativo para jugadores - MODELO SIEMPRE 85% DEL PESO
        
        Args:
            raw_prediction: Predicción original del modelo
            actual_stats_mean: Promedio de estadísticas históricas reales
            confidence: Confianza preliminar calculada
            prediction_std: Desviación estándar de predicciones
            actual_stats_std: Desviación estándar de estadísticas reales
            
        Returns:
            Tuple[predicción_final, tolerancia_usada]
        """
        try:
            # ESTRATEGIA ÚNICA PARA JUGADORES: MODELO 85% + HISTÓRICO 15%
            # Como solicitaste: "SIEMPRE EL MODELO DEBE TENER 85% DE PESO"
            
            if confidence >= self.ultra_confidence_threshold:
                # Ultra confianza: usar predicción raw
                tolerance = 0
                final_prediction = raw_prediction
                logger.info(f" ULTRA CONFIANZA PLAYER ({confidence:.1f}%): Usando predicción RAW")
                
            else:
                # Estrategia fija: 85% modelo + 15% histórico
                tolerance = self.base_tolerance  # -1
                stabilized = (raw_prediction * 0.85) + (actual_stats_mean * 0.15)
                final_prediction = stabilized + tolerance
                
                logger.info(f" ESTRATEGIA PLAYER ({confidence:.1f}%): 85% modelo + 15% histórico, tolerancia -1")
            
            # AJUSTE ADICIONAL POR VOLATILIDAD DEL JUGADOR
            if actual_stats_std > 8:  # Jugador muy inconsistente
                volatility_adjustment = -0.5
                final_prediction += volatility_adjustment
                logger.info(f"📉 Ajuste por volatilidad del jugador: {volatility_adjustment}")
            
            return final_prediction, tolerance
            
        except Exception as e:
            logger.error(f"Error en estrategia adaptativa para jugadores: {e}")
            return raw_prediction, self.base_tolerance
