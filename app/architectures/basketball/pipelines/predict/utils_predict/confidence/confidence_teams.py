"""
Confidence Predict Teams
=================

Modulo exclusivamente para el calculo de confianza en las predicciones inferidas
por los modelos para equipos.
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

class TeamsConfidence:
    """
    Clase centralizada para calcular la confianza en las predicciones de los equipos
    
    Maneja todos los cálculos de confianza para:
    - teams_points_predict
    - total_points_predict  
    - is_win_predict
    """
    def __init__(self):
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.is_loaded = False
        
        # Umbrales de confianza para equipos
        self.base_tolerance = -1
        self.high_confidence_threshold = 75.0
        self.ultra_confidence_threshold = 85.0
        self.min_confidence_threshold = 50.0

    def load_data(self):
        """Cargar datos históricos"""
        try:
            # Cargar datos históricos
            data_loader = NBADataLoader(
                players_total_path="app/architectures/basketball/data/players_total.csv",
                players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                teams_total_path="app/architectures/basketball/data/teams_total.csv",
                teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                biometrics_path="app/architectures/basketball/data/biometrics.csv"
            )
            self.historical_players, self.historical_teams = data_loader.load_data()
            self.is_loaded = True
            logger.info("✅ TeamsConfidence: Datos históricos cargados correctamente")
            return True
        except Exception as e:
            logger.error(f"❌ Error cargando datos en TeamsConfidence: {e}")
            return False

    def calculate_teams_points_confidence(self, raw_prediction: float, stabilized_prediction: float, 
                                        tolerance: float, prediction_std: float, actual_points_std: float,
                                        historical_games: int, team_data: Dict[str, Any]) -> float:
        """
        Calcular confianza para predicciones de TEAMS POINTS (puntos de un equipo específico)
        
        Args:
            raw_prediction: Predicción original del modelo
            stabilized_prediction: Predicción estabilizada con histórico
            tolerance: Tolerancia aplicada
            prediction_std: Desviación estándar de predicciones
            actual_points_std: Desviación estándar de puntos reales
            historical_games: Número de juegos históricos
            team_data: Datos del equipo
            
        Returns:
            Porcentaje de confianza (65.0-98.0)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # FACTOR 1: Confianza base por distancia de tolerancia (40% del peso)
            tolerance_confidence = min(100, abs(tolerance) * 8)
            
            # 🤖 FACTOR PREDICCIÓN (25% peso) - pred:98
            # Consistencia del modelo ML (desviación estándar)
            # Cálculo: max(0, 100 - (prediction_std * 5))
            # Menor desviación = mayor confianza
            if prediction_std > 0:
                prediction_consistency = max(0, 100 - (prediction_std * 5))
            else:
                prediction_consistency = 95
            
            # 🏃 FACTOR ESTABILIDAD (20% peso) - stab:84
            # Estabilidad histórica del equipo
            # Cálculo: max(0, 100 - (actual_points_std * 3))
            # Menor variabilidad del equipo = mayor confianza
            if actual_points_std > 0:
                team_stability = max(0, 100 - (actual_points_std * 3))
            else:
                team_stability = 90
            
            # 📊 FACTOR DATOS (10% peso) - data:95
            # Cantidad de juegos históricos disponibles
            # 25+ juegos = 95%, 20+ = 90%, 15+ = 80%, etc.
            if historical_games >= 25:
                data_confidence = 95
            elif historical_games >= 20:
                data_confidence = 90
            elif historical_games >= 15:
                data_confidence = 80
            elif historical_games >= 10:
                data_confidence = 70
            else:
                data_confidence = max(50, historical_games * 5)
            
            # 🔗 FACTOR COHERENCIA (5% peso) - coh:95
            # Coherencia entre predicción del modelo y histórico
            # Menor diferencia = mayor confianza
            coherence_diff = abs(raw_prediction - stabilized_prediction)
            if coherence_diff <= 2:
                coherence_confidence = 95
            elif coherence_diff <= 5:
                coherence_confidence = 80
            elif coherence_diff <= 10:
                coherence_confidence = 65
            else:
                coherence_confidence = max(40, 100 - coherence_diff * 3)
            
            # BONUS LOCAL - home:+5
            # +5% si el equipo juega en casa
            # +0% si juega de visitante
            is_home = team_data.get('is_home', 0)
            home_bonus = 5 if is_home == 1 else 0
            
            # CALCULAR CONFIANZA PONDERADA
            weighted_confidence = (
                tolerance_confidence * 0.40 +      # 40% - Factor más importante
                prediction_consistency * 0.25 +    # 25% - Consistencia del modelo
                team_stability * 0.20 +           # 20% - Estabilidad del equipo
                data_confidence * 0.10 +          # 10% - Cantidad de datos
                coherence_confidence * 0.05 +     # 5% - Coherencia
                home_bonus                        # Bonus por jugar en casa
            )
            
            # APLICAR LÍMITES REALISTAS (65% - 98%)
            final_confidence = max(65.0, min(98.0, weighted_confidence))
            
            logger.info(f"🎯 Confianza teams_points: {final_confidence:.1f}% "
                       f"(tol:{tolerance_confidence:.0f}, pred:{prediction_consistency:.0f}, "
                       f"stab:{team_stability:.0f}, data:{data_confidence:.0f}, "
                       f"coh:{coherence_confidence:.0f}, home:+{home_bonus})")
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza teams_points: {e}")
            return 75.0

    def calculate_total_points_confidence(self, raw_prediction: float, stabilized_prediction: float, 
                                        tolerance: float, prediction_std: float, actual_points_std: float,
                                        historical_games: int, team_data: Dict[str, Any]) -> float:
        """
        Calcular confianza para predicciones de TOTAL POINTS (puntos totales del partido)
        
        Similar a teams_points pero SIN ventaja de local (total points es neutral)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # FACTOR 1: Confianza base por distancia de tolerancia (40% del peso)
            tolerance_confidence = min(100, abs(tolerance) * 8)
            
            # FACTOR 2: Consistencia de predicciones del modelo (25% del peso)
            if prediction_std > 0:
                prediction_consistency = max(0, 100 - (prediction_std * 5))
            else:
                prediction_consistency = 95
            
            # FACTOR 3: Estabilidad histórica del partido total (20% del peso)
            if actual_points_std > 0:
                game_stability = max(0, 100 - (actual_points_std * 3))
            else:
                game_stability = 90
            
            # FACTOR 4: Cantidad de datos históricos recientes (10% del peso)
            if historical_games >= 25:
                data_confidence = 95
            elif historical_games >= 20:
                data_confidence = 90
            elif historical_games >= 15:
                data_confidence = 80
            elif historical_games >= 10:
                data_confidence = 70
            else:
                data_confidence = max(50, historical_games * 5)
            
            # FACTOR 5: Coherencia entre modelo y histórico (5% del peso)
            coherence_diff = abs(raw_prediction - stabilized_prediction)
            if coherence_diff <= 2:
                coherence_confidence = 95
            elif coherence_diff <= 5:
                coherence_confidence = 80
            elif coherence_diff <= 10:
                coherence_confidence = 65
            else:
                coherence_confidence = max(40, 100 - coherence_diff * 3)
            
            # NO HAY BONUS DE LOCAL PARA TOTAL POINTS (neutral)
            home_bonus = 0  # Total points no depende de quien juegue en casa
            
            # CALCULAR CONFIANZA PONDERADA
            weighted_confidence = (
                tolerance_confidence * 0.40 +      # 40% - Factor más importante
                prediction_consistency * 0.25 +    # 25% - Consistencia del modelo
                game_stability * 0.20 +           # 20% - Estabilidad del partido
                data_confidence * 0.10 +          # 10% - Cantidad de datos
                coherence_confidence * 0.05 +     # 5% - Coherencia
                home_bonus                        # 0% - Sin bonus de local
            )
            
            # APLICAR LÍMITES REALISTAS (65% - 98%)
            final_confidence = max(65.0, min(98.0, weighted_confidence))
            
            logger.info(f"🎯 Confianza total_points: {final_confidence:.1f}% "
                       f"(tol:{tolerance_confidence:.0f}, pred:{prediction_consistency:.0f}, "
                       f"stab:{game_stability:.0f}, data:{data_confidence:.0f}, "
                       f"coh:{coherence_confidence:.0f}, SIN bonus local)")
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza total_points: {e}")
            return 75.0

    def calculate_is_win_confidence(self, win_probability: float, historical_data: pd.DataFrame, 
                                  home_team: str, away_team: str, game_data: Dict[str, Any] = None) -> float:
        """
        Calcular confianza para predicciones de IS_WIN (victoria/derrota)
        
        Args:
            win_probability: Probabilidad de victoria (0-1)
            historical_data: Datos históricos para contexto
            home_team: Nombre del equipo local
            away_team: Nombre del equipo visitante
            game_data: Datos del juego (opcional)
            
        Returns:
            Porcentaje de confianza (50.0-95.0)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # Base de confianza: qué tan alejada está la probabilidad de 50%
            base_confidence = abs(win_probability - 0.5) * 200
            
            # Factor por cantidad de datos históricos
            data_factor = min(len(historical_data) / 60.0, 1.0)
            
            # Factor por consistencia de los datos
            consistency_factor = 1.0
            if len(historical_data) > 10:
                recent_wins = historical_data['is_win'].tail(10) if 'is_win' in historical_data.columns else [0.5] * 10
                win_variance = np.var(recent_wins)
                consistency_factor = max(0.7, 1.0 - win_variance)
            
            # Factor de enfrentamientos directos históricos
            head_to_head_factor = self.calculate_head_to_head_factor(home_team, away_team)
            
            # Factor de jugadores estrella lesionados/ausentes
            star_player_factor = self.calculate_star_player_factor_is_win(home_team, away_team, game_data)
            
            # BONUS LOCAL - home:+5%
            # +5% si el equipo local tiene ventaja de local
            # En is_win, el equipo local siempre tiene ventaja de local
            home_bonus = 5  # Siempre +5% para el equipo local en is_win
            
            # Combinar todos los factores
            final_confidence = base_confidence * data_factor * consistency_factor * head_to_head_factor * star_player_factor + home_bonus
            
            # Asegurar que esté en rango válido
            final_confidence = max(self.min_confidence_threshold, min(95.0, final_confidence))
            
            logger.info(f"🎯 Confianza is_win: {final_confidence:.1f}% "
                       f"(base:{base_confidence:.0f}, data:{data_factor:.2f}, "
                       f"consist:{consistency_factor:.2f}, h2h:{head_to_head_factor:.2f}, star:{star_player_factor:.2f}, home:+{home_bonus})")
            
            return round(final_confidence, 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza is_win: {e}")
            return 75.0

    def calculate_star_player_factor_teams_points(self, team_name: str, opponent_name: str = None, 
                                                game_data: Dict[str, Any] = None) -> float:
        """
        Calcular factor de jugadores estrella para TEAMS POINTS (impacto en puntos del equipo específico)
        
        Returns:
            Factor multiplicador (0.85 - 1.0) basado en ausencias del equipo específico
        """
        if not game_data:
            return 1.0
        
        try:
            # Identificar jugadores estrella dinámicamente
            team_stars = self.identify_star_players_by_stats(team_name)
            team_stars_out = 0
            team_total_stars = len(team_stars)
            
            # Determinar si es equipo local o visitante
            home_team_data = game_data.get('homeTeam', {})
            away_team_data = game_data.get('awayTeam', {})
            home_team_name = home_team_data.get('name', '')
            away_team_name = away_team_data.get('name', '')
            
            # Identificar qué conjunto de jugadores analizar
            if self.common_utils._normalize_name(team_name) == self.common_utils._normalize_name(home_team_name):
                team_players = home_team_data.get('players', [])
                team_type = "HOME"
            elif self.common_utils._normalize_name(team_name) == self.common_utils._normalize_name(away_team_name):
                team_players = away_team_data.get('players', [])
                team_type = "AWAY"
            else:
                # Buscar por abreviación
                team_abbrev = self.common_utils._get_team_abbreviation(team_name)
                home_abbrev = home_team_data.get('alias', '')
                away_abbrev = away_team_data.get('alias', '')
                
                if team_abbrev == home_abbrev:
                    team_players = home_team_data.get('players', [])
                    team_type = "HOME"
                elif team_abbrev == away_abbrev:
                    team_players = away_team_data.get('players', [])
                    team_type = "AWAY"
                else:
                    logger.warning(f"No se pudo identificar jugadores para {team_name}")
                    return 1.0
            
            # Analizar ausencias de estrellas del equipo objetivo
            for star_name in team_stars:
                star_status = self.common_utils._get_player_status_from_sportradar(game_data, star_name)
                if star_status in ['OUT', 'INJURED', 'DNP', 'SUSPENSION']:
                    team_stars_out += 1
                    logger.info(f"{team_name} ({team_type}) - Estrella FUERA: {star_name} ({star_status})")
            
            if team_total_stars == 0:
                return 1.0
            
            # Porcentaje de estrellas ausentes
            stars_out_pct = team_stars_out / team_total_stars
            
            logger.info(f"ANÁLISIS JUGADORES ESTRELLA - {team_name}:")
            logger.info(f"   {team_type} {team_name}: {team_stars_out}/{team_total_stars} estrellas fuera ({stars_out_pct:.1%})")
            
            # Calcular factor ESPECÍFICO PARA TEAMS POINTS
            if stars_out_pct >= 0.67:  # ≥67% de estrellas fuera
                factor = 0.85  # -15% puntos del equipo
                logger.info(f"   Factor: 0.85 (reducción significativa - mayoría de estrellas fuera)")
            elif stars_out_pct >= 0.50:  # ≥50% de estrellas fuera
                factor = 0.90  # -10% puntos del equipo
                logger.info(f"   Factor: 0.90 (reducción considerable - mitad de estrellas fuera)")
            elif stars_out_pct >= 0.33:  # ≥33% de estrellas fuera
                factor = 0.95  # -5% puntos del equipo
                logger.info(f"   Factor: 0.95 (reducción moderada - una estrella clave fuera)")
            else:  # <33% de estrellas fuera
                factor = 1.0  # Neutro
                logger.info(f"   Factor: 1.0 (plantilla completa o ausencias menores)")
            
            return factor
            
        except Exception as e:
            logger.warning(f"Error calculando factor jugadores estrella teams_points: {e}")
            return 1.0

    def calculate_star_player_factor_total_points(self, home_team: str, away_team: str, 
                                                game_data: Dict[str, Any] = None) -> float:
        """
        Calcular factor de jugadores estrella para TOTAL POINTS (impacto conservador en total)
        
        Returns:
            Factor multiplicador (0.95 - 1.05) basado en ausencias de ambos equipos
        """
        if not game_data:
            return 1.0
        
        try:
            # Identificar jugadores estrella de ambos equipos
            home_stars = self.identify_star_players_by_stats(home_team)
            away_stars = self.identify_star_players_by_stats(away_team)
            
            home_stars_out = 0
            away_stars_out = 0
            home_total_stars = len(home_stars)
            away_total_stars = len(away_stars)
            
            # Analizar equipo local
            home_team_data = game_data.get('homeTeam', {})
            home_players = home_team_data.get('players', [])
            
            for star_name in home_stars:
                star_status = self.common_utils._get_player_status_in_game(home_players, star_name)
                if star_status in ['OUT', 'INJURED', 'DNP', 'SUSPENSION']:
                    home_stars_out += 1
                    logger.info(f"{home_team} - Estrella FUERA: {star_name} ({star_status})")
            
            # Analizar equipo visitante
            away_team_data = game_data.get('awayTeam', {})
            away_players = away_team_data.get('players', [])
            
            for star_name in away_stars:
                star_status = self.common_utils._get_player_status_in_game(away_players, star_name)
                if star_status in ['OUT', 'INJURED', 'DNP', 'SUSPENSION']:
                    away_stars_out += 1
                    logger.info(f"{away_team} - Estrella FUERA: {star_name} ({star_status})")
            
            # Calcular factor basado en diferencia de ausencias
            if home_total_stars == 0 and away_total_stars == 0:
                return 1.0
            
            # Porcentaje de estrellas ausentes por equipo
            home_stars_out_pct = home_stars_out / max(home_total_stars, 1)
            away_stars_out_pct = away_stars_out / max(away_total_stars, 1)
            
            # Factor basado en diferencia de ausencias
            difference = away_stars_out_pct - home_stars_out_pct
            
            logger.info(f"ANÁLISIS JUGADORES ESTRELLA:")
            logger.info(f"   HOME {home_team}: {home_stars_out}/{home_total_stars} estrellas fuera ({home_stars_out_pct:.1%})")
            logger.info(f"   AWAY {away_team}: {away_stars_out}/{away_total_stars} estrellas fuera ({away_stars_out_pct:.1%})")
            logger.info(f"   Diferencia: {difference:.2f}")
            
            # Calcular factor AJUSTADO PARA TOTAL POINTS (más conservador)
            if difference >= 0.33:  # Away team tiene ≥33% más ausencias
                factor = 1.05  # +5% total points
                logger.info(f"   Factor: 1.05 (ligero aumento total points - away team debilitado)")
            elif difference >= 0.15:  # Away team tiene ≥15% más ausencias
                factor = 1.02  # +2% total points
                logger.info(f"   Factor: 1.02 (muy ligero aumento total points)")
            elif difference <= -0.33:  # Home team tiene ≥33% más ausencias
                factor = 0.95  # -5% total points
                logger.info(f"   Factor: 0.95 (ligera reducción total points - home team debilitado)")
            elif difference <= -0.15:  # Home team tiene ≥15% más ausencias
                factor = 0.98  # -2% total points
                logger.info(f"   Factor: 0.98 (muy ligera reducción total points)")
            else:  # Diferencia mínima
                factor = 1.0  # Neutro
                logger.info(f"   Factor: 1.0 (ausencias equilibradas)")
            
            return factor
            
        except Exception as e:
            logger.warning(f"Error calculando factor jugadores estrella total_points: {e}")
            return 1.0

    def calculate_star_player_factor_is_win(self, home_team: str, away_team: str, 
                                          game_data: Dict[str, Any] = None) -> float:
        """
        Calcular factor de jugadores estrella para IS_WIN (impacto en probabilidad de victoria)
        
        Returns:
            Factor multiplicador (0.8 - 1.2) basado en ausencias de estrellas
        """
        if not game_data:
            return 1.0
        
        try:
            # Identificar jugadores estrella de ambos equipos
            home_stars = self.identify_star_players_by_stats(home_team)
            away_stars = self.identify_star_players_by_stats(away_team)
            
            home_stars_out = 0
            away_stars_out = 0
            home_total_stars = len(home_stars)
            away_total_stars = len(away_stars)
            
            # Analizar equipo local
            home_team_data = game_data.get('homeTeam', {})
            home_players = home_team_data.get('players', [])
            
            for star_name in home_stars:
                star_status = self.common_utils._get_player_status_in_game(home_players, star_name)
                if star_status in ['OUT', 'INJURED', 'DNP', 'SUSPENSION']:
                    home_stars_out += 1
                    logger.info(f"{home_team} - Estrella FUERA: {star_name} ({star_status})")
            
            # Analizar equipo visitante
            away_team_data = game_data.get('awayTeam', {})
            away_players = away_team_data.get('players', [])
            
            for star_name in away_stars:
                star_status = self.common_utils._get_player_status_in_game(away_players, star_name)
                if star_status in ['OUT', 'INJURED', 'DNP', 'SUSPENSION']:
                    away_stars_out += 1
                    logger.info(f"{away_team} - Estrella FUERA: {star_name} ({star_status})")
            
            # Calcular factor basado en diferencia de ausencias
            if home_total_stars == 0 and away_total_stars == 0:
                return 1.0
            
            # Porcentaje de estrellas ausentes por equipo
            home_stars_out_pct = home_stars_out / max(home_total_stars, 1)
            away_stars_out_pct = away_stars_out / max(away_total_stars, 1)
            
            # Factor basado en diferencia de ausencias
            difference = away_stars_out_pct - home_stars_out_pct
            
            logger.info(f"ANÁLISIS JUGADORES ESTRELLA:")
            logger.info(f"   HOME {home_team}: {home_stars_out}/{home_total_stars} estrellas fuera ({home_stars_out_pct:.1%})")
            logger.info(f"   AWAY {away_team}: {away_stars_out}/{away_total_stars} estrellas fuera ({away_stars_out_pct:.1%})")
            logger.info(f"   Diferencia: {difference:.2f}")
            
            # Calcular factor para IS_WIN
            if difference >= 0.33:  # Away team tiene ≥33% más ausencias
                factor = 1.2  # +20% confianza
                logger.info(f"   Factor: 1.2 (ventaja significativa para {home_team})")
            elif difference >= 0.15:  # Away team tiene ≥15% más ausencias
                factor = 1.1  # +10% confianza
                logger.info(f"   Factor: 1.1 (ligera ventaja para {home_team})")
            elif difference <= -0.33:  # Home team tiene ≥33% más ausencias
                factor = 0.8  # -20% confianza
                logger.info(f"   Factor: 0.8 (desventaja significativa para {home_team})")
            elif difference <= -0.15:  # Home team tiene ≥15% más ausencias
                factor = 0.9  # -10% confianza
                logger.info(f"   Factor: 0.9 (ligera desventaja para {home_team})")
            else:  # Diferencia mínima
                factor = 1.0  # Neutro
                logger.info(f"   Factor: 1.0 (ausencias equilibradas)")
            
            return factor
            
        except Exception as e:
            logger.warning(f"Error calculando factor jugadores estrella is_win: {e}")
            return 1.0

    def identify_star_players_by_stats(self, team_name: str, lookback_games: int = 20) -> List[str]:
        """
        Identificar jugadores estrella dinámicamente usando estadísticas del dataset
        
        Returns:
            Lista de nombres de jugadores estrella identificados
        """
        try:
            if self.historical_players is None or len(self.historical_players) == 0:
                logger.warning(f"No hay datos históricos para identificar estrellas de {team_name}")
                return []
            
            # Convertir nombre a abreviación para búsqueda
            team_abbrev = self.common_utils._get_team_abbreviation(team_name)
            
            # Filtrar jugadores del equipo específico
            team_players = self.historical_players[self.historical_players['Team'] == team_abbrev].copy()
            
            if len(team_players) == 0:
                logger.warning(f"No se encontraron datos para el equipo {team_name} ({team_abbrev})")
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
                        # Obtener TODOS los datos del jugador en TODO el dataset usando búsqueda inteligente
                        player_all_data = self.common_utils._smart_player_search(self.historical_players, player_name)
                        
                        if len(player_all_data) > 0:
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
                            is_active_player = days_since_last_game <= 45  # Máximo 45 días sin jugar
                            
                            if is_current_team and is_active_player:
                                # Tomar solo datos del jugador en este equipo específico
                                player_team_data = player_all_data[player_all_data['Team'] == team_abbrev].head(lookback_games)
                                
                                if len(player_team_data) >= 5:  # Mínimo 5 juegos para considerarlo
                                    verified_current_players.append(player_team_data)
                                    logger.debug(f"✅ {player_name}: Jugador actual de {team_abbrev} "
                                               f"(último juego: {days_since_last_game} días atrás)")
                                else:
                                    logger.debug(f"⚠️ {player_name}: Pocos datos en {team_abbrev} ({len(player_team_data)} juegos)")
                            else:
                                logger.debug(f"❌ {player_name}: No es jugador actual "
                                           f"(equipo: {latest_team}, días: {days_since_last_game})")
                    
                    if verified_current_players:
                        recent_players = pd.concat(verified_current_players, ignore_index=True)
                        logger.info(f"🎯 {team_name}: {len(verified_current_players)} jugadores actuales verificados")
                    else:
                        # Fallback más conservador si no encuentra jugadores actuales
                        logger.warning(f"⚠️ No se encontraron jugadores actuales para {team_name}, usando fallback")
                        recent_players = team_players.head(lookback_games * 10)
                else:
                    recent_players = team_players.head(lookback_games * 10)
            else:
                # Sin columna Date, usar datos más recientes disponibles
                recent_players = team_players.head(lookback_games * 10)
            
            # Verificar columnas disponibles
            available_cols = recent_players.columns.tolist()
            
            # Agrupar por jugador y calcular estadísticas
            agg_dict = {
                'points': ['mean', 'std', 'count'],
                'rebounds': ['mean', 'std'],
                'assists': ['mean', 'std'],
                'minutes': ['mean'] if 'minutes' in available_cols else ['count']
            }
            
            # Agregar columnas opcionales si existen
            if 'is_started' in available_cols:
                agg_dict['is_started'] = ['mean']
            elif 'GS' in available_cols:  # Games Started
                agg_dict['GS'] = ['mean']
            
            player_stats = recent_players.groupby('player').agg(agg_dict).round(2)
            
            # Aplanar nombres de columnas
            player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns]
            
            # Filtrar jugadores con suficientes juegos (mínimo 10)
            min_games = 10
            player_stats = player_stats[player_stats['points_count'] >= min_games].copy()
            
            if len(player_stats) == 0:
                logger.warning(f"No hay jugadores con suficientes datos para {team_name}")
                return []
            
            # Calcular métricas de impacto
            impact_score = player_stats['points_mean'] * 1.0
            
            if 'rebounds_mean' in player_stats.columns:
                impact_score += player_stats['rebounds_mean'] * 0.8
            if 'assists_mean' in player_stats.columns:
                impact_score += player_stats['assists_mean'] * 1.2
            if 'minutes_mean' in player_stats.columns:
                impact_score += player_stats['minutes_mean'] * 0.3
            
            player_stats['impact_score'] = impact_score
            
            # Calcular consistencia
            consistency_penalty = player_stats['points_std'] * 2
            
            if 'rebounds_std' in player_stats.columns:
                consistency_penalty += player_stats['rebounds_std'] * 1.5
            if 'assists_std' in player_stats.columns:
                consistency_penalty += player_stats['assists_std'] * 2
                
            player_stats['consistency_score'] = 100 - consistency_penalty
            
            # Calcular factor titular
            if 'is_started_mean' in player_stats.columns:
                player_stats['starter_bonus'] = player_stats['is_started_mean'] * 5
            elif 'GS_mean' in player_stats.columns:
                games_played = player_stats['points_count']
                starter_rate = player_stats['GS_mean'] / games_played.clip(lower=1)
                player_stats['starter_bonus'] = starter_rate * 5
            else:
                if 'minutes_mean' in player_stats.columns:
                    player_stats['starter_bonus'] = (player_stats['minutes_mean'] > 25).astype(int) * 3
                else:
                    player_stats['starter_bonus'] = 0
            
            # Score final combinado
            player_stats['star_score'] = (
                player_stats['impact_score'] * 0.6 +
                player_stats['consistency_score'] * 0.3 +
                player_stats['starter_bonus'] * 0.1
            )
            
            # Ordenar por score y seleccionar top 3-4 jugadores
            top_players = player_stats.nlargest(4, 'star_score')
            
            # Filtros adicionales para ser considerado estrella
            star_threshold_points = max(12.0, player_stats['points_mean'].median())
            star_threshold_score = player_stats['star_score'].quantile(0.75)
            
            if 'minutes_mean' in player_stats.columns:
                star_threshold_minutes = 20.0
            else:
                star_threshold_minutes = 0
            
            stars = []
            for player_name, stats in top_players.iterrows():
                pts_ok = stats['points_mean'] >= star_threshold_points
                score_ok = stats['star_score'] >= star_threshold_score
                
                if 'minutes_mean' in player_stats.columns:
                    minutes_ok = stats['minutes_mean'] >= star_threshold_minutes
                else:
                    minutes_ok = True
                
                if pts_ok and score_ok and minutes_ok:
                    stars.append(player_name)
            
            # Logging detallado
            logger.info(f"IDENTIFICACIÓN ESTRELLAS {team_name}:")
            logger.info(f"   Jugadores analizados: {len(player_stats)}")
            logger.info(f"   Umbrales: {star_threshold_points:.1f} pts, {star_threshold_minutes:.0f} min, {star_threshold_score:.1f} score")
            
            for i, (player_name, stats) in enumerate(top_players.iterrows()):
                is_star = player_name in stars
                star_prefix = "[STAR]" if is_star else "[PLAYER]"
                
                stats_str = f"{stats['points_mean']:.1f}pts"
                if 'minutes_mean' in player_stats.columns:
                    stats_str += f", {stats['minutes_mean']:.0f}min"
                stats_str += f", score:{stats['star_score']:.1f}"
                
                logger.info(f"   {star_prefix} {i+1}. {player_name}: {stats_str}")
            
            logger.info(f"   Estrellas identificadas: {len(stars)} - {stars}")
            
            return stars[:3]
            
        except Exception as e:
            logger.error(f"Error identificando estrellas para {team_name}: {e}")
            return []

    def calculate_head_to_head_stats_teams_points(self, team_name: str, opponent_name: str, 
                                                max_games: int = 50) -> Dict[str, Any]:
        """
        Calcular estadísticas H2H para TEAMS POINTS (enfoque en puntos del equipo específico)
        """
        try:
            if self.historical_teams is None or len(self.historical_teams) == 0:
                logger.warning(f"No hay datos históricos para análisis H2H")
                return {}
            
            # Convertir nombres a abreviaciones para búsqueda  
            team_abbrev = self.common_utils._get_team_abbreviation(team_name)
            opponent_abbrev = self.common_utils._get_team_abbreviation(opponent_name)
            
            logger.info(f"Buscando H2H: {team_name} ({team_abbrev}) vs {opponent_name} ({opponent_abbrev})")
            
            # Buscar enfrentamientos directos del equipo objetivo contra el oponente
            h2h_team_vs_opponent = self.historical_teams[
                (self.historical_teams['Team'] == team_abbrev) & 
                (self.historical_teams['Opp'] == opponent_abbrev)
            ].copy()
            
            if len(h2h_team_vs_opponent) == 0:
                logger.warning(f"No se encontraron enfrentamientos H2H de {team_abbrev} vs {opponent_abbrev}")
                return {}
            
            # Ordenar por fecha (más reciente primero)
            if 'Date' in h2h_team_vs_opponent.columns:
                h2h_team_vs_opponent = h2h_team_vs_opponent.sort_values('Date', ascending=False)
            
            # Usar todos los juegos H2H para mejor precisión
            h2h_recent = h2h_team_vs_opponent
            
            logger.info(f"Encontrados {len(h2h_recent)} enfrentamientos H2H recientes")
            
            if len(h2h_recent) == 0 or 'points' not in h2h_recent.columns:
                return {}
            
            # Obtener puntos del equipo específico en esos enfrentamientos
            team_points_vs_opponent = h2h_recent['points'].values
            
            # Calcular estadísticas en diferentes ventanas
            stats = {
                'games_found': len(team_points_vs_opponent),
                'team_points_mean': np.mean(team_points_vs_opponent),
                'team_points_std': np.std(team_points_vs_opponent) if len(team_points_vs_opponent) > 1 else 0,
                'team_points_median': np.median(team_points_vs_opponent),
                'team_points_min': np.min(team_points_vs_opponent),
                'team_points_max': np.max(team_points_vs_opponent)
            }
            
            # Estadísticas por ventanas temporales
            if len(team_points_vs_opponent) >= 5:
                stats['last_5_mean'] = np.mean(team_points_vs_opponent[:5])
                stats['last_5_std'] = np.std(team_points_vs_opponent[:5])
                stats['last_5_trend'] = 'increasing' if team_points_vs_opponent[0] > team_points_vs_opponent[4] else 'decreasing'
                
            if len(team_points_vs_opponent) >= 10:
                stats['last_10_mean'] = np.mean(team_points_vs_opponent[:10])
                stats['last_10_std'] = np.std(team_points_vs_opponent[:10])
                stats['last_10_trend'] = 'increasing' if np.mean(team_points_vs_opponent[:5]) > np.mean(team_points_vs_opponent[5:10]) else 'decreasing'
            
            # Tendencia general
            if len(team_points_vs_opponent) >= 3:
                recent_avg = np.mean(team_points_vs_opponent[:3])
                older_avg = np.mean(team_points_vs_opponent[3:]) if len(team_points_vs_opponent) > 3 else recent_avg
                stats['overall_trend'] = 'increasing' if recent_avg > older_avg else 'decreasing'
                stats['trend_magnitude'] = abs(recent_avg - older_avg)
            
            # Consistencia de puntos del equipo
            if stats['team_points_mean'] > 0:
                stats['consistency_score'] = 100 - (stats['team_points_std'] / stats['team_points_mean'] * 100)
            else:
                stats['consistency_score'] = 0
            
            # Factor de predicción basado en histórico vs promedio del equipo
            team_avg_all_games = 112.6  # Promedio NBA típico por equipo
            if stats['team_points_mean'] > 0:
                stats['h2h_factor'] = stats['team_points_mean'] / team_avg_all_games
                if stats['games_found'] >= 5:
                    recent_factor = stats.get('last_5_mean', stats['team_points_mean']) / team_avg_all_games
                    stats['h2h_factor'] = 0.7 * recent_factor + 0.3 * stats['h2h_factor']
            else:
                stats['h2h_factor'] = 1.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculando estadísticas H2H teams_points: {e}")
            return {}

    def calculate_head_to_head_stats_total_points(self, home_team: str, away_team: str, 
                                                max_games: int = 50) -> Dict[str, Any]:
        """
        Calcular estadísticas H2H para TOTAL POINTS (enfoque en puntos totales del partido)
        """
        try:
            if self.historical_teams is None or len(self.historical_teams) == 0:
                logger.warning(f"No hay datos históricos para análisis H2H")
                return {}
            
            # Convertir nombres a abreviaciones para búsqueda  
            home_abbrev = self.common_utils._get_team_abbreviation(home_team)
            away_abbrev = self.common_utils._get_team_abbreviation(away_team)
            
            logger.info(f"🔍 Buscando H2H: {home_team} ({home_abbrev}) vs {away_team} ({away_abbrev})")
            
            # Buscar enfrentamientos directos en ambas direcciones
            h2h_home_away = self.historical_teams[
                (self.historical_teams['Team'] == home_abbrev) & 
                (self.historical_teams['Opp'] == away_abbrev)
            ].copy()
            
            h2h_away_home = self.historical_teams[
                (self.historical_teams['Team'] == away_abbrev) & 
                (self.historical_teams['Opp'] == home_abbrev)
            ].copy()
            
            # Combinar ambos conjuntos de datos
            if len(h2h_home_away) > 0 and len(h2h_away_home) > 0:
                h2h_combined = pd.concat([h2h_home_away, h2h_away_home], ignore_index=True)
            elif len(h2h_home_away) > 0:
                h2h_combined = h2h_home_away
            elif len(h2h_away_home) > 0:
                h2h_combined = h2h_away_home
            else:
                logger.warning(f"❌ No se encontraron enfrentamientos H2H entre {home_abbrev} y {away_abbrev}")
                return {}
            
            # Ordenar por fecha (más reciente primero)
            if 'Date' in h2h_combined.columns:
                h2h_combined = h2h_combined.sort_values('Date', ascending=False)
            
            # Usar todos los juegos H2H para mejor precisión
            h2h_recent = h2h_combined
            
            logger.info(f"📊 Encontrados {len(h2h_recent)} enfrentamientos H2H recientes")
            
            if len(h2h_recent) == 0:
                return {}
            
            # Calcular total points por fecha (si hay múltiples registros por fecha)
            if 'Date' in h2h_recent.columns and 'points' in h2h_recent.columns:
                total_points_by_date = h2h_recent.groupby('Date')['points'].sum()
                total_points = total_points_by_date.values
            elif 'points' in h2h_recent.columns:
                total_points = h2h_recent['points'].values
            else:
                logger.warning("❌ No se encontró columna 'points' en datos H2H")
                return {}
            
            # Calcular estadísticas en diferentes ventanas
            stats = {
                'games_found': len(total_points),
                'total_points_mean': np.mean(total_points),
                'total_points_std': np.std(total_points) if len(total_points) > 1 else 0,
                'total_points_median': np.median(total_points),
                'total_points_min': np.min(total_points),
                'total_points_max': np.max(total_points)
            }
            
            # Estadísticas por ventanas temporales
            if len(total_points) >= 5:
                stats['last_5_mean'] = np.mean(total_points[:5])
                stats['last_5_std'] = np.std(total_points[:5])
                stats['last_5_trend'] = 'increasing' if total_points[0] > total_points[4] else 'decreasing'
                
            if len(total_points) >= 10:
                stats['last_10_mean'] = np.mean(total_points[:10])
                stats['last_10_std'] = np.std(total_points[:10])
                stats['last_10_trend'] = 'increasing' if np.mean(total_points[:5]) > np.mean(total_points[5:10]) else 'decreasing'
            
            # Tendencia general
            if len(total_points) >= 3:
                recent_avg = np.mean(total_points[:3])
                older_avg = np.mean(total_points[3:]) if len(total_points) > 3 else recent_avg
                stats['overall_trend'] = 'increasing' if recent_avg > older_avg else 'decreasing'
                stats['trend_magnitude'] = abs(recent_avg - older_avg)
            
            # Consistencia de total points
            if stats['total_points_mean'] > 0:
                stats['consistency_score'] = 100 - (stats['total_points_std'] / stats['total_points_mean'] * 100)
            else:
                stats['consistency_score'] = 0
            
            # Factor de predicción basado en histórico
            nba_average = 220  # Promedio típico NBA
            if stats['total_points_mean'] > 0:
                stats['h2h_factor'] = stats['total_points_mean'] / nba_average
                if stats['games_found'] >= 5:
                    recent_factor = stats.get('last_5_mean', stats['total_points_mean']) / nba_average
                    stats['h2h_factor'] = 0.7 * recent_factor + 0.3 * stats['h2h_factor']
            else:
                stats['h2h_factor'] = 1.0
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error calculando estadísticas H2H total_points: {e}")
            return {}

    def calculate_head_to_head_factor(self, home_team: str, away_team: str) -> float:
        """
        Calcular factor de confianza basado en enfrentamientos directos históricos para IS_WIN
        
        Returns:
            Factor multiplicador (0.8 - 1.3) basado en historial H2H
        """
        try:
            if self.historical_teams is None or len(self.historical_teams) == 0:
                return 1.0
            
            # Convertir nombres a abreviaciones para búsqueda en dataset
            home_abbrev = self.common_utils._get_team_abbreviation(home_team)
            away_abbrev = self.common_utils._get_team_abbreviation(away_team)
            
            # Obtener datos históricos de ambos equipos usando abreviaciones
            home_data = self.historical_teams[self.historical_teams['Team'] == home_abbrev].copy()
            away_data = self.historical_teams[self.historical_teams['Team'] == away_abbrev].copy()
            
            if len(home_data) == 0 or len(away_data) == 0:
                return 1.0
            
            # Buscar enfrentamientos directos
            home_vs_away = home_data[home_data['Opp'] == away_abbrev].copy()
            away_vs_home = away_data[away_data['Opp'] == home_abbrev].copy()
            
            # Combinar todos los enfrentamientos directos
            all_matchups = pd.concat([home_vs_away, away_vs_home], ignore_index=True)
            
            if len(all_matchups) == 0:
                return 1.0
            
            # Ordenar por fecha más reciente
            if 'Date' in all_matchups.columns:
                all_matchups = all_matchups.sort_values('Date', ascending=False)
            
            # Analizar últimos 5 y 10 enfrentamientos
            factor_5_games = self._analyze_recent_matchups(all_matchups.head(5), home_team, away_team)
            factor_10_games = self._analyze_recent_matchups(all_matchups.head(10), home_team, away_team)
            
            # Combinar factores: más peso a los 5 juegos más recientes
            combined_factor = (factor_5_games * 0.7) + (factor_10_games * 0.3)
            
            # Logging para debugging
            logger.info(f"H2H Factor: {home_team} vs {away_team}")
            logger.info(f"   Total enfrentamientos: {len(all_matchups)}")
            logger.info(f"   Últimos 5: factor {factor_5_games:.2f}")
            logger.info(f"   Últimos 10: factor {factor_10_games:.2f}")
            logger.info(f"   Factor final: {combined_factor:.2f}")
            
            return combined_factor
            
        except Exception as e:
            logger.warning(f"Error calculando H2H factor: {e}")
            return 1.0

    def _analyze_recent_matchups(self, matchups_df: pd.DataFrame, home_team: str, away_team: str) -> float:
        """
        Analizar enfrentamientos recientes y calcular factor de confianza para IS_WIN
        
        Returns:
            Factor de confianza basado en resultados recientes (0.8 - 1.3)
        """
        if len(matchups_df) == 0:
            return 1.0
        
        # Convertir nombres a abreviaciones para comparación
        home_abbrev = self.common_utils._get_team_abbreviation(home_team)
        away_abbrev = self.common_utils._get_team_abbreviation(away_team)
        
        # Contar victorias del equipo que estamos prediciendo como ganador
        home_wins = 0
        away_wins = 0
        
        for _, game in matchups_df.iterrows():
            game_team = game['Team']
            game_result = game.get('is_win', 0)  # 1 si ganó, 0 si perdió
            
            if game_team == home_abbrev and game_result == 1:
                home_wins += 1
            elif game_team == away_abbrev and game_result == 1:
                away_wins += 1
        
        total_games = len(matchups_df)
        
        # Calcular dominancia en el enfrentamiento
        if total_games == 0:
            return 1.0
        
        # Porcentaje de victorias del equipo con ventaja
        max_wins = max(home_wins, away_wins)
        dominance_rate = max_wins / total_games
        
        # Factor basado en dominancia
        if dominance_rate >= 0.8:  # 80%+ dominancia
            return 1.25  # +25% confianza
        elif dominance_rate >= 0.7:  # 70%+ dominancia
            return 1.15  # +15% confianza
        elif dominance_rate >= 0.6:  # 60%+ dominancia
            return 1.05  # +5% confianza
        elif dominance_rate >= 0.4:  # 40-60% equilibrado
            return 1.0   # Neutro
        else:  # Dominancia del otro equipo
            return 0.9   # -10% confianza

    def calculate_halftime_confidence(self, raw_prediction: float, stabilized_prediction: float, 
                                    tolerance: float, prediction_std: float, actual_ht_std: float,
                                    historical_games: int, team_data: Dict[str, Any]) -> float:
        """
        Calcular confianza para predicciones de HALFTIME (puntos de primera mitad)
        
        Args:
            raw_prediction: Predicción original del modelo
            stabilized_prediction: Predicción estabilizada con histórico
            tolerance: Tolerancia aplicada
            prediction_std: Desviación estándar de predicciones
            actual_ht_std: Desviación estándar de halftime reales
            historical_games: Número de juegos históricos
            team_data: Datos del equipo
            
        Returns:
            Porcentaje de confianza (65.0-98.0)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # FACTOR 1: Confianza base por distancia de tolerancia (40% del peso)
            tolerance_confidence = min(100, abs(tolerance) * 8)
            
            # 🤖 FACTOR PREDICCIÓN (25% peso) - pred:98
            # Consistencia del modelo ML (desviación estándar)
            # Cálculo: max(0, 100 - (prediction_std * 5))
            # Menor desviación = mayor confianza
            if prediction_std > 0:
                prediction_consistency = max(0, 100 - (prediction_std * 5))
            else:
                prediction_consistency = 95
            
            # 🏃 FACTOR ESTABILIDAD (20% peso) - stab:84
            # Estabilidad histórica del equipo en halftime
            # Para halftime, usar umbrales más estrictos (menor variabilidad esperada)
            # Cálculo: max(0, 100 - (actual_ht_std * 4))
            if actual_ht_std > 0:
                team_stability = max(0, 100 - (actual_ht_std * 4))  # Más estricto para halftime
            else:
                team_stability = 90
            
            # 📊 FACTOR DATOS (10% peso) - data:95
            # Cantidad de juegos históricos disponibles
            # 25+ juegos = 95%, 20+ = 90%, 15+ = 80%, etc.
            if historical_games >= 25:
                data_confidence = 95
            elif historical_games >= 20:
                data_confidence = 90
            elif historical_games >= 15:
                data_confidence = 80
            elif historical_games >= 10:
                data_confidence = 70
            else:
                data_confidence = max(50, historical_games * 5)
            
            # 🔗 FACTOR COHERENCIA (5% peso) - coh:95
            # Coherencia entre predicción del modelo y histórico
            # Menor diferencia = mayor confianza
            coherence_diff = abs(raw_prediction - stabilized_prediction)
            if coherence_diff <= 2:
                coherence_confidence = 95
            elif coherence_diff <= 5:
                coherence_confidence = 80
            elif coherence_diff <= 10:
                coherence_confidence = 65
            else:
                coherence_confidence = max(40, 100 - coherence_diff * 3)
            
            # BONUS LOCAL - home:+5
            # +5% si el equipo juega en casa
            # +0% si juega de visitante
            is_home = team_data.get('is_home', 0)
            home_bonus = 5 if is_home == 1 else 0
            
            # CALCULAR CONFIANZA PONDERADA
            weighted_confidence = (
                tolerance_confidence * 0.40 +      # 40% - Factor más importante
                prediction_consistency * 0.25 +    # 25% - Consistencia del modelo
                team_stability * 0.20 +           # 20% - Estabilidad del equipo
                data_confidence * 0.10 +          # 10% - Cantidad de datos
                coherence_confidence * 0.05 +     # 5% - Coherencia
                home_bonus                        # Bonus por jugar en casa
            )
            
            # APLICAR LÍMITES REALISTAS (65% - 98%)
            final_confidence = max(65.0, min(98.0, weighted_confidence))
            
            logger.info(f"🎯 Confianza halftime: {final_confidence:.1f}% "
                       f"(tol:{tolerance_confidence:.0f}, pred:{prediction_consistency:.0f}, "
                       f"stab:{team_stability:.0f}, data:{data_confidence:.0f}, "
                       f"coh:{coherence_confidence:.0f}, home:+{home_bonus})")
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza halftime: {e}")
            return 75.0
    
    def calculate_h2h_factor_halftime(self, team_name: str, opponent_name: str, historical_teams: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Calcular factor H2H específico para halftime
        
        Args:
            team_name: Nombre del equipo
            opponent_name: Nombre del oponente
            historical_teams: DataFrame con datos históricos
            
        Returns:
            Dict con estadísticas H2H de halftime
        """
        try:
            if historical_teams is None or len(historical_teams) == 0:
                return {'games_found': 0, 'halftime_mean': None, 'h2h_factor': 1.0, 'consistency_score': 0}
            
            # Filtrar juegos H2H
            h2h_games = historical_teams[
                (historical_teams['Team'] == team_name) & 
                (historical_teams['Opp'] == opponent_name)
            ].copy()
            
            if len(h2h_games) == 0:
                return {'games_found': 0, 'halftime_mean': None, 'h2h_factor': 1.0, 'consistency_score': 0}
            
            # Filtrar juegos con HT válido
            h2h_ht = h2h_games['HT'].dropna()
            
            if len(h2h_ht) == 0:
                return {'games_found': 0, 'halftime_mean': None, 'h2h_factor': 1.0, 'consistency_score': 0}
            
            # Calcular estadísticas H2H de halftime
            h2h_mean = h2h_ht.mean()
            h2h_std = h2h_ht.std() if len(h2h_ht) > 1 else 0
            
            # Calcular factor H2H basado en rendimiento relativo
            team_overall_ht = historical_teams[historical_teams['Team'] == team_name]['HT'].dropna().mean()
            
            if team_overall_ht > 0:
                h2h_factor = h2h_mean / team_overall_ht
            else:
                h2h_factor = 1.0
            
            # Calcular consistencia (inversa de la desviación estándar)
            # Para halftime, usar umbrales más estrictos (menor variabilidad esperada)
            if h2h_std > 0:
                consistency_score = max(0, 100 - (h2h_std * 3))  # Más estricto para halftime
            else:
                consistency_score = 100
            
            # Estadísticas adicionales
            last_5_mean = h2h_ht.tail(5).mean() if len(h2h_ht) >= 5 else h2h_mean
            last_10_mean = h2h_ht.tail(10).mean() if len(h2h_ht) >= 10 else h2h_mean
            
            return {
                'games_found': len(h2h_ht),
                'halftime_mean': h2h_mean,
                'halftime_std': h2h_std,
                'h2h_factor': h2h_factor,
                'consistency_score': consistency_score,
                'last_5_mean': last_5_mean,
                'last_10_mean': last_10_mean,
                'team_overall_ht': team_overall_ht
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando H2H halftime: {e}")
            return {'games_found': 0, 'halftime_mean': None, 'h2h_factor': 1.0, 'consistency_score': 0}
    
    def calculate_halftime_total_points_confidence(self, home_confidence: float, away_confidence: float, 
                                                 home_team: str, away_team: str, 
                                                 game_data: Dict[str, Any] = None) -> float:
        """
        Calcular confianza para predicciones de HALFTIME TOTAL POINTS (total puntos de primera mitad)
        
        Args:
            home_confidence: Confianza del equipo local en halftime
            away_confidence: Confianza del equipo visitante en halftime
            home_team: Nombre del equipo local
            away_team: Nombre del equipo visitante
            game_data: Datos del juego (opcional)
            
        Returns:
            Porcentaje de confianza (65.0-98.0)
        """
        if not self.is_loaded:
            self.load_data()
            
        try:
            # FACTOR 1: Confianza promedio base (60% del peso)
            # Promedio simple de las confianzas individuales
            avg_base_confidence = (home_confidence + away_confidence) / 2.0
            
            # FACTOR 2: Consistencia entre equipos (20% del peso)
            # Si ambos equipos tienen confianza similar, es más confiable
            confidence_diff = abs(home_confidence - away_confidence)
            if confidence_diff <= 5:
                consistency_bonus = 10  # +10% si confianzas muy similares
            elif confidence_diff <= 10:
                consistency_bonus = 5   # +5% si confianzas moderadamente similares
            elif confidence_diff <= 20:
                consistency_bonus = 0   # Neutro
            else:
                consistency_bonus = -5  # -5% si confianzas muy diferentes
            
            # FACTOR 3: H2H específico para halftime total points (10% del peso)
            # Analizar enfrentamientos históricos en halftime
            h2h_factor = self.calculate_h2h_factor_halftime(home_team, away_team, self.historical_teams)
            h2h_confidence = h2h_factor.get('consistency_score', 50)  # 0-100
            h2h_bonus = (h2h_confidence - 50) * 0.2  # Convertir a bonus/penalty
            
            # FACTOR 4: Factor de jugadores estrella (10% del peso)
            # Analizar ausencias de estrellas en ambos equipos
            home_star_factor = self.calculate_star_player_factor_teams_points(home_team, away_team, game_data)
            away_star_factor = self.calculate_star_player_factor_teams_points(away_team, home_team, game_data)
            avg_star_factor = (home_star_factor + away_star_factor) / 2.0
            star_bonus = (avg_star_factor - 1.0) * 20  # Convertir factor a bonus/penalty
            
            # CALCULAR CONFIANZA FINAL
            final_confidence = (
                avg_base_confidence * 0.60 +      # 60% - Confianza promedio base
                consistency_bonus +               # 20% - Consistencia entre equipos
                h2h_bonus +                      # 10% - H2H halftime
                star_bonus                       # 10% - Factor de estrellas
            )
            
            # APLICAR LÍMITES REALISTAS (65% - 98%)
            final_confidence = max(65.0, min(98.0, final_confidence))
            
            logger.info(f"🎯 Confianza halftime_total_points: {final_confidence:.1f}% "
                       f"(avg_base:{avg_base_confidence:.1f}, consist:{consistency_bonus:+.1f}, "
                       f"h2h:{h2h_bonus:+.1f}, star:{star_bonus:+.1f})")
            
            return round(final_confidence, 1)
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza halftime_total_points: {e}")
            return 75.0
    