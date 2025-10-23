"""
Unified Predictor
========================================

Unifica todos los predictores (teams y players) en un solo predictor.
Procesa múltiples juegos y hace predicciones para todos los targets disponibles.
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import logging
from datetime import datetime

#Ejecucion Paralela
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Importar data loaders 
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils

# Importar predictores teams
from app.architectures.basketball.pipelines.predict.teams.is_win_predict import IsWinPredictor
from app.architectures.basketball.pipelines.predict.teams.total_points_predict import TotalPointsPredictor
from app.architectures.basketball.pipelines.predict.teams.teams_points_predict import TeamsPointsPredictor

# Importar predictores halftime
from app.architectures.basketball.pipelines.predict.teams.ht_teams_points_predict import HalfTimeTeamsPointsPredictor
from app.architectures.basketball.pipelines.predict.teams.ht_total_points_predict import HalfTimeTotalPointsPredictor

# Importar predictores players
from app.architectures.basketball.pipelines.predict.players.pts_predictor import PTSPredictor
from app.architectures.basketball.pipelines.predict.players.threept_predict import ThreePointsPredictor 
from app.architectures.basketball.pipelines.predict.players.ast_predict import ASTPredictor
from app.architectures.basketball.pipelines.predict.players.dd_predict import DoubleDoublePredictor
from app.architectures.basketball.pipelines.predict.players.trb_predict import TRBPredictor

# Utilidades
from app.utils.helpers import convert_numpy_types
logger = logging.getLogger(__name__)

class UnifiedPredictor: 
    """
    Predictor unificado encargado de predecir todos los tipos de predicciones (teams y players)
    
    Procesa múltiples juegos y hace predicciones para todos los targets disponibles.
    """

    def __init__(self):
        """
        Inicializar el predictor unificado
        """
        # Predictores de equipos
        self.is_win_predictor = IsWinPredictor()
        self.total_points_predictor = TotalPointsPredictor()  
        self.teams_points_predictor = TeamsPointsPredictor()
        
        # Predictores de halftime por equipos
        self.ht_teams_points_predictor = HalfTimeTeamsPointsPredictor()
        self.ht_total_points_predictor = HalfTimeTotalPointsPredictor()
        
        # Predictores de jugadores
        self.pts_predictor = PTSPredictor()
        self.threept_predictor = ThreePointsPredictor()
        self.ast_predictor = ASTPredictor()
        self.dd_predictor = DoubleDoublePredictor()
        self.trb_predictor = TRBPredictor()
        
        # Utilidades
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.is_loaded = False

    def load_all_models(self) -> bool:
        """
        Cargar todos los modelos de predictores
        
        Returns:
            True si todos se cargaron exitosamente
        """
        try:
            logger.info(" Cargando todos los modelos...")
            
            # Cargar modelos de equipos
            logger.info(" Cargando predictores de equipos...")
            team_predictors = [
                ("IsWin", self.is_win_predictor),
                ("TotalPoints", self.total_points_predictor), 
                ("TeamsPoints", self.teams_points_predictor)
            ]
            
            for name, predictor in team_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f" Error cargando {name}")
                    return False
                logger.info(f" {name} cargado exitosamente")
            
            # Cargar modelos de halftime
            logger.info("⏰ Cargando predictores de halftime...")
            halftime_predictors = [
                ("HT_TeamsPoints", self.ht_teams_points_predictor),
                ("HT_TotalPoints", self.ht_total_points_predictor)
            ]
            
            for name, predictor in halftime_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f" Error cargando {name}")
                    return False
                logger.info(f" {name} cargado exitosamente")
            
            # Cargar modelos de jugadores
            logger.info(" Cargando predictores de jugadores...")
            player_predictors = [
                ("PTS", self.pts_predictor),
                ("3PT", self.threept_predictor),
                ("AST", self.ast_predictor),
                ("TRB", self.trb_predictor),
                ("DD", self.dd_predictor)
            ]
            
            for name, predictor in player_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f" Error cargando {name}")
                    return False
                logger.info(f" {name} cargado exitosamente")
            
            self.is_loaded = True
            logger.info(" Todos los modelos cargados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f" Error cargando modelos: {e}")
            return False

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], max_workers: int = 8) -> Dict[str, Any]:
        """
        Método unificado para predicciones con arquitectura de respuesta estandarizada.
        
        Args:
            data: Datos del juego (dict) o lista de juegos (list) en formato SportRadar
            max_workers: Número máximo de hilos paralelos para múltiples juegos (default: 8)
            
        Returns:
            Dict con estructura:
            {
                "metadata": {
                    "timestamp": "ISO timestamp",
                    "version": "1.0",
                    "total_games": int
                },
                "games": [
                    {
                        "game_id": str,
                        "scheduled": str,
                        "teams": {...},
                        "predictions": {
                            "team_level": {...},
                            "player_level": {
                                "home_players": [...],
                                "away_players": [...]
                            }
                        },
                        "summary": {...}
                    }
                ]
            }
        """
        if not self.is_loaded:
            logger.error(" Modelos no cargados. Ejecutar load_all_models() primero.")
            return {'error': 'Modelos no cargados'}
        
        try:
            # Detectar si es un solo juego o múltiples juegos
            games_list = data if isinstance(data, list) else [data]
            
            logger.info(f" Predicción de {len(games_list)} juego(s) en paralelo")
            return self.predict_games(games_list, max_workers)
                
        except Exception as e:
            logger.error(f" Error en predicción unificada: {e}")
            return {'error': str(e)}

    def predict_games(self, games_data: List[Dict[str, Any]], max_workers: int = 8) -> Dict[str, Any]:
        """
        Predecir múltiples juegos EN PARALELO con arquitectura estandarizada
        
        Args:
            games_data: Lista de datos de juegos de SportRadar
            max_workers: Número máximo de hilos paralelos (default: 8)
            
        Returns:
            Diccionario con estructura estandarizada
        """
        if not self.is_loaded:
            logger.error(" Modelos no cargados. Ejecutar load_all_models() primero.")
            return {'error': 'Modelos no cargados'}
        
        # Estructura de respuesta estandarizada
        response = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'total_games': len(games_data)
            },
            'games': []
        }
        
        logger.info(f" Iniciando predicciones PARALELAS para {len(games_data)} juegos con {max_workers} hilos")
        
        # Usar ThreadPoolExecutor para ejecución paralela
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Crear tareas para cada juego
            future_to_game = {
                executor.submit(self.predict_single_game, game_data): (idx, game_data) 
                for idx, game_data in enumerate(games_data)
            }
            
            # Procesar resultados conforme se completan
            completed_games = 0
            for future in as_completed(future_to_game):
                game_idx, game_data = future_to_game[future]
                completed_games += 1
                
                try:
                    game_result = future.result()
                    # Convertir tipos de NumPy
                    game_result_clean = convert_numpy_types(game_result)
                    response['games'].append(game_result_clean)
                    
                    # Obtener nombres de equipos para el log (sin fallbacks críticos)
                    home_name = game_data.get('homeTeam', {}).get('name', 'Unknown') if game_data else 'Unknown'
                    away_name = game_data.get('awayTeam', {}).get('name', 'Unknown') if game_data else 'Unknown'
                    logger.info(f" Juego {completed_games}/{len(games_data)} completado: {home_name} vs {away_name}")
                    
                except Exception as e:
                    logger.error(f" Error procesando juego {game_idx + 1}: {e}")
                    # Agregar juego con error (intentar extraer info si existe)
                    game_id = game_data.get('gameId', 'unknown') if game_data else 'unknown'
                    scheduled = game_data.get('scheduled', 'unknown') if game_data else 'unknown'
                    response['games'].append({
                        'game_id': game_id,
                        'scheduled': scheduled,
                        'teams': {
                            'home': {
                                'id': game_data.get('homeTeam', {}).get('teamId', '') if game_data else '',
                                'name': game_data.get('homeTeam', {}).get('name', '') if game_data else '',
                                'abbreviation': game_data.get('homeTeam', {}).get('alias', '') if game_data else ''
                            },
                            'away': {
                                'id': game_data.get('awayTeam', {}).get('teamId', '') if game_data else '',
                                'name': game_data.get('awayTeam', {}).get('name', '') if game_data else '',
                                'abbreviation': game_data.get('awayTeam', {}).get('alias', '') if game_data else ''
                            }
                        },
                        'predictions': {
                            'match_level': {'status': 'error', 'bet_type': 'match', 'errors': [{'type': 'system_error', 'message': str(e)}]},
                            'team_level': {'status': 'error', 'bet_type': 'Teams', 'errors': [{'type': 'system_error', 'message': str(e)}]},
                            'player_level': {'home_players': [], 'away_players': []}
                        },
                        'summary': {'total_predictions': 0}
                    })
        
        logger.info(f" Predicciones paralelas completadas: {len(response['games'])} juegos procesados")
        return response

    def predict_single_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predecir un solo juego con arquitectura estandarizada
        
        Args:
            game_data: Datos del juego de SportRadar
            
        Returns:
            Diccionario con estructura estandarizada para un juego individual
        """
        try:
            # Validar que game_data contiene información de equipos (sin fallbacks)
            if not game_data or 'homeTeam' not in game_data or 'awayTeam' not in game_data:
                logger.error("❌ game_data no contiene información de equipos")
                raise ValueError('game_data debe contener homeTeam y awayTeam')
            
            # Extraer información básica del juego (sin fallbacks)
            home_team_data = game_data['homeTeam']
            away_team_data = game_data['awayTeam']
            
            # Validar campos requeridos
            if 'gameId' not in game_data:
                logger.error("❌ game_data no contiene 'gameId'")
                raise ValueError('game_data debe contener gameId')
            if 'scheduled' not in game_data:
                logger.error("❌ game_data no contiene 'scheduled'")
                raise ValueError('game_data debe contener scheduled')
            
            # 1. Predicciones de equipos y partidos
            logger.info(" Haciendo predicciones de equipos y partidos...")
            match_level, team_level = self._predict_teams_and_match(game_data)
            
            # 2. Predicciones de jugadores
            logger.info(" Haciendo predicciones de jugadores...")
            player_level = self._predict_all_players(game_data)
            
            # 3. Calcular resumen (summary)
            summary = self._calculate_summary(player_level)
            
            # Construir respuesta estandarizada (sin fallbacks)
            return {
                'game_id': game_data['gameId'],
                'scheduled': game_data['scheduled'],
                'teams': {
                    'home': {
                        'id': home_team_data.get('teamId', ''),  # teamId puede ser vacío
                        'name': home_team_data.get('name', ''),  # name puede ser vacío
                        'abbreviation': home_team_data.get('alias', '')  # alias puede ser vacío
                    },
                    'away': {
                        'id': away_team_data.get('teamId', ''),  # teamId puede ser vacío
                        'name': away_team_data.get('name', ''),  # name puede ser vacío
                        'abbreviation': away_team_data.get('alias', '')  # alias puede ser vacío
                    }
                },
                'predictions': {
                    'match_level': match_level,
                    'team_level': team_level,
                    'player_level': player_level
                },
                'summary': summary
            }
            
        except Exception as e:
            logger.error(f" Error procesando juego: {e}")
            import traceback
            traceback.print_exc()
            # En caso de error, intentar extraer al menos game_id y scheduled si existen
            game_id = game_data.get('gameId', 'unknown') if game_data else 'unknown'
            scheduled = game_data.get('scheduled', 'unknown') if game_data else 'unknown'
            return {
                'game_id': game_id,
                'scheduled': scheduled,
                'teams': {
                    'home': {'id': '', 'name': '', 'abbreviation': ''},
                    'away': {'id': '', 'name': '', 'abbreviation': ''}
                },
                'predictions': {
                    'match_level': {'status': 'error', 'bet_type': 'match', 'errors': [{'type': 'system_error', 'message': str(e)}]},
                    'team_level': {'status': 'error', 'bet_type': 'Teams', 'errors': [{'type': 'system_error', 'message': str(e)}]},
                    'player_level': {'home_players': [], 'away_players': []}
                },
                'summary': {'total_predictions': 0}
            }

    def _predict_teams_and_match(self, game_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Hacer todas las predicciones de equipos y partidos EN PARALELO
        Separa las predicciones en dos niveles: match_level y team_level
        
        Returns:
            Tuple (match_level, team_level) donde:
            - match_level: Predicciones del partido completo (total_points, ht_total_points, is_win)
            - team_level: Predicciones por equipo (home_points, away_points, home_ht_points, away_ht_points)
        """
        match_predictions = {}
        team_predictions = {}
        match_errors = []
        team_errors = []
        
        try:
            # Definir tareas de predicción
            prediction_tasks = {
                'is_win': lambda: self.is_win_predictor.predict_game(game_data),
                'total_points': lambda: self.total_points_predictor.predict_game(game_data),
                'teams_points': lambda: self.teams_points_predictor.predict_game(game_data),
                'ht_teams_points': lambda: self.ht_teams_points_predictor.predict_game(game_data),
                'ht_total_points': lambda: self.ht_total_points_predictor.predict_game(game_data)
            }
            
            # Ejecutar todas las predicciones en paralelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_task = {
                    executor.submit(task_func): task_name 
                    for task_name, task_func in prediction_tasks.items()
                }
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        
                        # Separar predicciones según su tipo
                        if task_name == 'teams_points' and isinstance(result, list):
                            # Predicciones de puntos por equipo van a team_level
                            if len(result) > 0:
                                team_predictions['home_points'] = self._transform_team_prediction_format(result[0])
                            if len(result) > 1:
                                team_predictions['away_points'] = self._transform_team_prediction_format(result[1])
                        elif task_name == 'ht_teams_points' and isinstance(result, list):
                            # Predicciones de puntos de medio tiempo por equipo van a team_level
                            if len(result) > 0:
                                team_predictions['home_ht_points'] = self._transform_team_prediction_format(result[0])
                            if len(result) > 1:
                                team_predictions['away_ht_points'] = self._transform_team_prediction_format(result[1])
                        elif task_name == 'total_points':
                            # Predicción de puntos totales va a match_level
                            match_predictions['total_points'] = self._transform_team_prediction_format(result)
                        elif task_name == 'ht_total_points':
                            # Predicción de puntos totales de medio tiempo va a match_level
                            match_predictions['ht_total_points'] = self._transform_team_prediction_format(result)
                        elif task_name == 'is_win':
                            # Predicción de ganador va a match_level
                            match_predictions['is_win'] = self._transform_team_prediction_format(result)
                            
                    except Exception as e:
                        logger.error(f" Error en predicción {task_name}: {e}")
                        error_info = {
                            'type': f'{task_name}_prediction',
                            'message': str(e)
                        }
                        # Asignar error al nivel correspondiente
                        if task_name in ['total_points', 'ht_total_points', 'is_win']:
                            match_errors.append(error_info)
                        else:
                            team_errors.append(error_info)
            
            # Construir match_level
            if len(match_errors) == 0 and len(match_predictions) > 0:
                match_status = 'success'
            elif len(match_predictions) == 0:
                match_status = 'error'
            else:
                match_status = 'partial'
            
            match_level = {
                'status': match_status,
                'bet_type': 'match',
                'predictions': match_predictions
            }
            if match_errors:
                match_level['errors'] = match_errors
            
            # Construir team_level
            if len(team_errors) == 0 and len(team_predictions) > 0:
                team_status = 'success'
            elif len(team_predictions) == 0:
                team_status = 'error'
            else:
                team_status = 'partial'
            
            team_level = {
                'status': team_status,
                'bet_type': 'Teams',
                'predictions': team_predictions
            }
            if team_errors:
                team_level['errors'] = team_errors
                
        except Exception as e:
            logger.error(f" Error en predicciones de equipos y partidos: {e}")
            match_level = {
                'status': 'error',
                'bet_type': 'match',
                'errors': [{'type': 'system_error', 'message': str(e)}]
            }
            team_level = {
                'status': 'error',
                'bet_type': 'Teams',
                'errors': [{'type': 'system_error', 'message': str(e)}]
            }
            
        return match_level, team_level


    def _predict_all_players(self, game_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Hacer predicciones para todos los jugadores disponibles EN PARALELO
        
        Returns:
            Dict con formato:
            {
                "home_players": [{player_data}],
                "away_players": [{player_data}]
            }
        """
        home_player_predictions = []
        away_player_predictions = []
        
        try:
            # Validar que game_data contiene información de equipos (sin fallbacks)
            if not game_data or 'homeTeam' not in game_data or 'awayTeam' not in game_data:
                logger.error("❌ game_data no contiene información de equipos para predicciones de jugadores")
                return {'home_players': [], 'away_players': []}
            
            # Obtener todos los jugadores de ambos equipos (sin fallbacks)
            home_team_data = game_data['homeTeam']
            away_team_data = game_data['awayTeam']
            
            home_players_data = home_team_data.get('players', [])
            away_players_data = away_team_data.get('players', [])
            
            if not home_players_data and not away_players_data:
                logger.warning("⚠️ No se encontraron jugadores en game_data")
                return {'home_players': [], 'away_players': []}
            
            # Filtrar solo jugadores activos (status ACT significa disponible para jugar)
            active_home_players = [p for p in home_players_data if p.get('status', '') == 'ACT']
            active_away_players = [p for p in away_players_data if p.get('status', '') == 'ACT']
            all_active_players = active_home_players + active_away_players
            
            if not all_active_players:
                logger.warning("⚠️ No se encontraron jugadores activos (status='ACT')")
                return {'home_players': [], 'away_players': []}
            
            logger.info(f" Encontrados {len(all_active_players)} jugadores activos (Home: {len(active_home_players)}, Away: {len(active_away_players)})")
            
            # Crear mapeo de jugador a equipo (home/away)
            player_to_team = {}
            for player in active_home_players:
                full_name = player.get('fullName', '')
                if full_name:  # Solo agregar si tiene nombre
                    player_to_team[full_name] = ('home', player)
            for player in active_away_players:
                full_name = player.get('fullName', '')
                if full_name:  # Solo agregar si tiene nombre
                    player_to_team[full_name] = ('away', player)
            
            # Ejecutar predicciones de jugadores en paralelo
            with ThreadPoolExecutor(max_workers=min(16, len(all_active_players))) as executor:
                # Crear tareas para cada jugador (solo si tiene nombre)
                future_to_player = {
                    executor.submit(self._predict_single_player, game_data, player.get('fullName', '')): player.get('fullName', '')
                    for player in all_active_players
                    if player.get('fullName', '')
                }
                
                # Procesar resultados conforme se completan
                for future in as_completed(future_to_player):
                    player_name = future_to_player[future]
                    
                    try:
                        player_predictions_list = future.result()
                        
                        # Solo agregar si tiene predicciones
                        if player_predictions_list and len(player_predictions_list) > 0:
                            team_type, player_info = player_to_team.get(player_name, ('unknown', {}))
                            
                            # Si no se encuentra en player_to_team, skip
                            if team_type == 'unknown':
                                logger.warning(f"⚠️ Jugador {player_name} no encontrado en mapeo de equipos")
                                continue
                            
                            player_data = {
                                'player_id': player_info.get('playerId', ''),
                                'name': player_name,
                                'position': player_info.get('position', ''),
                                'jersey_number': player_info.get('jerseyNumber', ''),
                                'status': player_info.get('status', ''),
                                'predictions': player_predictions_list
                            }
                            
                            if team_type == 'home':
                                home_player_predictions.append(player_data)
                            elif team_type == 'away':
                                away_player_predictions.append(player_data)
                            
                    except Exception as e:
                        logger.error(f" Error prediciendo {player_name}: {e}")
                        
        except Exception as e:
            logger.error(f" Error en predicciones de jugadores: {e}")
        
        return {
            'home_players': home_player_predictions,
            'away_players': away_player_predictions
        }

    def _predict_single_player(self, game_data: Dict[str, Any], player_name: str) -> List[Dict[str, Any]]:
        """
        Hacer todas las predicciones para un jugador específico EN PARALELO
        
        Returns:
            Lista de predicciones con formato estandarizado:
            [{
                "stat_type": "points",
                "bet_line": 26.0,
                "predicted_value": 26.3,
                "confidence": 85.3,
                "recommendation": "OVER",
                "details": {...},
                "historical_context": {...}
            }]
        """
        predictions_list = []
        
        try:
            # Definir tareas de predicción de jugador
            player_tasks = {
                'points': lambda: self.pts_predictor.predict_game(game_data, player_name),
                'three_points': lambda: self.threept_predictor.predict_game(game_data, player_name),
                'assists': lambda: self.ast_predictor.predict_game(game_data, player_name),
                'rebounds': lambda: self.trb_predictor.predict_game(game_data, player_name),
                'double_double': lambda: self.dd_predictor.predict_game(game_data, player_name)
            }
            
            # Ejecutar todas las predicciones del jugador en paralelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_task = {
                    executor.submit(task_func): task_name 
                    for task_name, task_func in player_tasks.items()
                }
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        
                        # Validar y transformar resultado
                        if result is not None:
                            if task_name == 'double_double':
                                if result.get('bet_line') == 'yes':
                                    predictions_list.append(self._transform_prediction_format(result, task_name))
                            else:
                                if self._is_valid_prediction(result, task_name):
                                    predictions_list.append(self._transform_prediction_format(result, task_name))
                                
                    except Exception as e:
                        logger.error(f" Error en predicción {task_name} para {player_name}: {e}")
                
        except Exception as e:
            logger.error(f" Error prediciendo {player_name}: {e}")
            
        return predictions_list

    def _is_valid_prediction(self, prediction_result: Dict[str, Any], prediction_type: str) -> bool:
        """
        Validar si una predicción debe mostrarse
        
        Args:
            prediction_result: Resultado de la predicción
            prediction_type: Tipo de predicción ('points', 'three_points', 'assists', 'rebounds')
            
        Returns:
            True si debe mostrarse, False si no es válida
        """
        try:
            bet_line = prediction_result.get('bet_line', '0')
            
            # Convertir a número
            if isinstance(bet_line, str):
                value = float(bet_line)
            else:
                value = float(bet_line)
            
            # Para puntos, asistencias y rebotes, 0 es válido
            # Solo filtrar valores negativos o extremadamente altos
            if prediction_type in ['points', 'assists', 'rebounds']:
                is_valid = value >= 5 and value <= 45  # Límite razonable
            elif prediction_type == 'three_points':
                is_valid = value >= 1 and value <= 20   # Límite razonable para triples
            else:
                is_valid = value >= 0
            
            if not is_valid:
                logger.info(f"🚫 Predicción {prediction_type} filtrada por valor inválido {value}: {prediction_result.get('target_name', 'Unknown')}")
            
            return is_valid
            
        except (ValueError, TypeError):
            logger.error(f" Error validando predicción {prediction_type}: {bet_line}")
            return False

    def _transform_prediction_format(self, prediction: Dict[str, Any], bet_type: str) -> Dict[str, Any]:
        """
        Transformar formato de predicción actual al formato estandarizado
        
        NUEVA LÓGICA:
        - bet_line = NUESTRA PREDICCIÓN (no la de la casa de apuestas)
        - NO incluimos predicted_value (es redundante)
        - stat_type → bet_type
        - recommendation = Dirección de la apuesta (OVER/UNDER/YES/NO)
        
        Args:
            prediction: Predicción en formato actual del predictor individual
            bet_type: Tipo de apuesta ('points', 'assists', 'rebounds', 'three_points', 'double_double')
            
        Returns:
            Predicción en formato estandarizado
        """
        try:
            details_data = prediction.get('prediction_details', {})
            
            # bet_line = NUESTRA PREDICCIÓN (valor final ajustado)
            bet_line_raw = prediction.get('bet_line', '0')
            
            # Manejar casos especiales (double_double: yes/no)
            if isinstance(bet_line_raw, str) and bet_line_raw.lower() in ['yes', 'no']:
                bet_line = bet_line_raw.lower()
                recommendation = 'YES' if bet_line == 'yes' else 'NO'
            else:
                # Para estadísticas numéricas, bet_line es nuestra predicción
                bet_line = float(bet_line_raw)
                # recommendation siempre es OVER para nuestras predicciones
                # (porque estamos recomendando que el jugador HARÁ ese valor o más)
                recommendation = 'OVER'
            
            # Extraer confianza
            confidence = prediction.get('confidence_percentage', 0.0)
            
            # Construir details
            details = {
                'raw_prediction': details_data.get('raw_prediction', bet_line),
                'h2h_adjusted': details_data.get('h2h_adjusted_prediction', bet_line),
                'tolerance_applied': details_data.get('tolerance_applied', 0.0),
                'prediction_std': details_data.get('prediction_std', None)
            }
            
            # Construir historical_context
            h2h_stats = details_data.get('h2h_stats', {})
            last_5 = details_data.get('last_5_games', {})
            last_10 = details_data.get('last_10_games', {})
            trend_analysis = details_data.get('trend_analysis', {})
            
            historical_context = {
                'games_analyzed': details_data.get('historical_games_used', 0),
                'season_avg': details_data.get('actual_stats_mean', 0.0),
                'season_std': details_data.get('actual_stats_std', 0.0),
                'recent_form': {
                    'last_5': {
                        'mean': last_5.get('mean', 0.0),
                        'std': last_5.get('std', 0.0),
                        'min': last_5.get('min', 0),
                        'max': last_5.get('max', 0),
                        'count': last_5.get('count', 0)
                    },
                    'last_10': {
                        'mean': last_10.get('mean', 0.0),
                        'std': last_10.get('std', 0.0),
                        'min': last_10.get('min', 0),
                        'max': last_10.get('max', 0),
                        'count': last_10.get('count', 0)
                    }
                },
                'h2h_matchup': {
                    'games': h2h_stats.get('games_found', 0),
                    'mean': h2h_stats.get('h2h_mean', None),
                    'std': h2h_stats.get('h2h_std', None),
                    'min': h2h_stats.get('h2h_min', None),
                    'max': h2h_stats.get('h2h_max', None),
                    'count': h2h_stats.get('games_found', 0),
                    'adjustment_factor': h2h_stats.get('h2h_factor', 1.0),
                    'consistency_score': h2h_stats.get('consistency_score', 0.0)
                },
                'trend_analysis': {
                    'direction': self._determine_trend_direction(trend_analysis.get('trend_5_games', 0)),
                    'slope_5_games': trend_analysis.get('trend_5_games', 0.0),
                    'consistency_score': trend_analysis.get('consistency_score', 0.0),
                    'form_rating': trend_analysis.get('recent_form', 0.0)
                }
            }
            
            return {
                'bet_type': bet_type,
                'bet_line': bet_line,
                'confidence': round(confidence, 1),
                'recommendation': recommendation,
                'details': details,
                'historical_context': historical_context
            }
            
        except Exception as e:
            logger.error(f" Error transformando formato de predicción {bet_type}: {e}")
            # Retornar formato mínimo
            return {
                'bet_type': bet_type,
                'bet_line': prediction.get('bet_line', 0),
                'confidence': 0.0,
                'recommendation': 'SKIP',
                'details': {},
                'historical_context': {}
            }
    
    def _determine_trend_direction(self, trend_value: float) -> str:
        """Determinar dirección de tendencia basado en el valor"""
        if trend_value > 1:
            return 'up'
        elif trend_value < -1:
            return 'down'
        else:
            return 'flat'
    
    def _determine_trend_direction_from_means(self, last_5_mean: float, last_10_mean: float) -> str:
        """Determinar dirección de tendencia comparando last_5 vs last_10"""
        if last_5_mean is None or last_10_mean is None:
            return 'flat'
        
        diff = last_5_mean - last_10_mean
        if diff > 5:  # Mejora significativa
            return 'up'
        elif diff < -5:  # Declinación significativa
            return 'down'
        else:
            return 'flat'
    
    def _calculate_slope(self, last_5_mean: float, last_10_mean: float) -> float:
        """Calcular pendiente (slope) de la tendencia"""
        if last_5_mean is None or last_10_mean is None:
            return 0.0
        
        return round(last_5_mean - last_10_mean, 1)
    
    def _calculate_consistency_from_std(self, std: float, mean: float) -> float:
        """Calcular consistency score basado en desviación estándar"""
        if std is None or mean is None or mean == 0:
            return 0.0
        
        # Consistency score: menos variabilidad = más consistencia
        # Score de 0-100, donde 100 es perfecto
        cv = (std / mean) * 100  # Coeficiente de variación
        consistency = max(0, 100 - cv)
        return round(consistency, 1)
    
    def _calculate_h2h_consistency(self, std: float, mean: float) -> float:
        """Calcular consistency score de H2H basado en std y mean"""
        if std is None or mean is None or mean == 0:
            return None
        
        # Consistency score: menos variabilidad = más consistencia
        cv = (std / mean) * 100
        consistency = max(0, 100 - cv)
        return round(consistency, 1)
    
    def _calculate_win_rate_std(self, h2h_stats: Dict[str, Any]) -> float:
        """Calcular desviación estándar de win rates H2H"""
        home_rate = h2h_stats.get('home_h2h_win_rate', 0)
        away_rate = h2h_stats.get('away_h2h_win_rate', 0)
        
        # Calcular std de los dos win rates
        mean_rate = (home_rate + away_rate) / 2
        variance = ((home_rate - mean_rate) ** 2 + (away_rate - mean_rate) ** 2) / 2
        std = variance ** 0.5
        
        return round(std, 1)
    
    def _calculate_h2h_consistency_win_rate(self, h2h_stats: Dict[str, Any]) -> float:
        """Calcular consistency score de H2H para win rates"""
        home_rate = h2h_stats.get('home_h2h_win_rate', 0)
        away_rate = h2h_stats.get('away_h2h_win_rate', 0)
        
        # Consistency alta = diferencia pequeña entre win rates
        diff = abs(home_rate - away_rate)
        # Si la diferencia es 0, consistency es 100, si es 100, consistency es 0
        consistency = max(0, 100 - diff)
        
        return round(consistency, 1)
    
    def _transform_team_prediction_format(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transformar predicciones de equipos/partidos al formato estandarizado
        
        FORMATO ESTANDARIZADO:
        - bet_line = NUESTRA PREDICCIÓN
        - recommendation = OVER (para valores numéricos) o WINNER/YES/NO (para categóricos)
        - confidence = Porcentaje de confianza
        - details = Detalles técnicos de la predicción
        - historical_context = Contexto histórico completo
        
        Args:
            prediction: Predicción en formato actual del predictor de equipos
            
        Returns:
            Predicción en formato estandarizado
        """
        try:
            # Manejar errores en la predicción original
            if 'error' in prediction:
                return {
                    'bet_line': 0,
                    'confidence': 0.0,
                    'recommendation': 'SKIP',
                    'details': {'error': prediction.get('error', 'Unknown error')},
                    'historical_context': {}
                }
            
            details_data = prediction.get('prediction_details', {})
            
            # Determinar bet_line según el tipo de predicción
            if 'team_points_prediction' in prediction:
                # Predicción de puntos de equipo
                bet_line = float(prediction.get('team_points_prediction', 0))
                recommendation = 'OVER'
            elif 'total_points_prediction' in prediction:
                # Predicción de puntos totales
                bet_line = float(prediction.get('total_points_prediction', 0))
                recommendation = 'OVER'
            elif 'predicted_winner' in prediction:
                # Predicción de ganador
                bet_line = prediction.get('predicted_winner', 'unknown')
                recommendation = 'WINNER'
            else:
                # Fallback genérico - verificar tipo de predicción
                bet_type = prediction.get('bet_type', '')
                raw_bet_line = prediction.get('bet_line', prediction.get('prediction', 0))
                
                # Determinar si es predicción de ganador (categórica) o numérica (points, totals, etc.)
                # Solo 'winner' y 'is_win' son categóricos, todo lo demás es numérico
                if bet_type in ['winner', 'is_win'] or (isinstance(raw_bet_line, str) and not raw_bet_line.replace('.', '').replace('-', '').isdigit()):
                    # Es predicción categórica (ganador)
                    bet_line = raw_bet_line
                    recommendation = 'WINNER'
                else:
                    # Es predicción numérica (puntos, totales, etc.) - convertir a float
                    try:
                        bet_line = float(raw_bet_line)
                        recommendation = 'OVER'
                    except (ValueError, TypeError):
                        # Si falla la conversión, dejar como string y marcar como SKIP
                        bet_line = raw_bet_line
                        recommendation = 'SKIP'
            
            # Extraer confianza
            confidence = prediction.get('confidence_percentage', 0.0)
            
            # Construir details
            details = {
                'raw_prediction': details_data.get('raw_prediction', details_data.get('base_total', bet_line)),
                'h2h_adjusted': details_data.get('h2h_adjusted_prediction', details_data.get('final_total', bet_line)),
                'tolerance_applied': details_data.get('tolerance_applied', details_data.get('conservative_tolerance', 0.0)),
                'prediction_std': details_data.get('prediction_std', None)
            }
            
            # Construir historical_context - DISTINGUIR ENTRE TOTAL POINTS, TEAM POINTS e IS WIN
            # Total Points usa: home_team_totals, away_team_totals, h2h_totals
            # HT Total Points usa: home_team_ht_totals, away_team_ht_totals, h2h_ht_totals
            # Team Points usa: last_5_games, last_10_games, h2h_stats
            # Is Win usa: home_team_stats, away_team_stats, h2h_stats
            
            if ('home_team_totals' in details_data or 'away_team_totals' in details_data or 
                'home_team_ht_totals' in details_data or 'away_team_ht_totals' in details_data):
                # Es una predicción de TOTAL POINTS o HT TOTAL POINTS
                historical_context = self._build_total_points_historical_context(details_data)
            elif 'home_team_stats' in details_data and 'away_team_stats' in details_data:
                # Es una predicción de IS WIN (tiene home_team_stats y away_team_stats)
                historical_context = self._build_is_win_historical_context(details_data)
            else:
                # Es una predicción de TEAM POINTS
                historical_context = self._build_team_points_historical_context(details_data)
            
            return {
                'bet_line': bet_line,
                'confidence': round(confidence, 1),
                'recommendation': recommendation,
                'details': details,
                'historical_context': historical_context
            }
            
        except Exception as e:
            logger.error(f" Error transformando formato de predicción de equipo: {e}")
            import traceback
            traceback.print_exc()
            # Retornar formato mínimo
            return {
                'bet_line': prediction.get('team_points_prediction', prediction.get('total_points_prediction', 0)),
                'confidence': 0.0,
                'recommendation': 'SKIP',
                'details': {'error': str(e)},
                'historical_context': {}
            }
    
    def _build_total_points_historical_context(self, details_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construir historical_context para TOTAL POINTS (suma de ambos equipos)
        
        Maneja tanto Total Points como Halftime Total Points:
        - Total Points usa: home_team_totals, away_team_totals, h2h_totals
        - HT Total Points usa: home_team_ht_totals, away_team_ht_totals, h2h_ht_totals
        
        Args:
            details_data: prediction_details de total_points_predict o ht_total_points_predict
            
        Returns:
            historical_context formateado
        """
        # Detectar si es Halftime Total Points o Total Points
        if 'home_team_ht_totals' in details_data:
            # Es Halftime Total Points
            home_totals = details_data.get('home_team_ht_totals', {})
            away_totals = details_data.get('away_team_ht_totals', {})
            h2h_totals = details_data.get('h2h_ht_totals', {})
        else:
            # Es Total Points regular
            home_totals = details_data.get('home_team_totals', {})
            away_totals = details_data.get('away_team_totals', {})
            h2h_totals = details_data.get('h2h_totals', {})
        
        # Para Total Points, combinar estadísticas de ambos equipos
        home_last_5 = home_totals.get('last_5_games', {})
        away_last_5 = away_totals.get('last_5_games', {})
        home_last_10 = home_totals.get('last_10_games', {})
        away_last_10 = away_totals.get('last_10_games', {})
        
        # Calcular promedios y std combinados si hay datos
        # Verificar que count > 0 PRIMERO para asegurar que hay datos reales
        last_5_mean = None
        last_5_std = None
        home_count_5 = home_last_5.get('count', 0)
        away_count_5 = away_last_5.get('count', 0)
        
        if home_count_5 > 0 and away_count_5 > 0:
            home_mean_5 = home_last_5.get('mean', 0)
            away_mean_5 = away_last_5.get('mean', 0)
            last_5_mean = (home_mean_5 + away_mean_5) / 2
            
            # Combinar std usando suma cuadrática
            home_std_5 = home_last_5.get('std', 0)
            away_std_5 = away_last_5.get('std', 0)
            if home_std_5 is not None and away_std_5 is not None:
                last_5_std = ((home_std_5**2 + away_std_5**2) ** 0.5) / 2
        
        last_10_mean = None
        last_10_std = None
        home_count_10 = home_last_10.get('count', 0)
        away_count_10 = away_last_10.get('count', 0)
        
        if home_count_10 > 0 and away_count_10 > 0:
            home_mean_10 = home_last_10.get('mean', 0)
            away_mean_10 = away_last_10.get('mean', 0)
            last_10_mean = (home_mean_10 + away_mean_10) / 2
            
            # Combinar std usando suma cuadrática
            home_std_10 = home_last_10.get('std', 0)
            away_std_10 = away_last_10.get('std', 0)
            if home_std_10 is not None and away_std_10 is not None:
                last_10_std = ((home_std_10**2 + away_std_10**2) ** 0.5) / 2
        
        return {
            'games_analyzed': max(home_last_10.get('count', 0), away_last_10.get('count', 0)),
            'season_avg': last_10_mean or 0.0,
            'season_std': last_10_std or 0.0,
            'recent_form': {
                'last_5': {
                    'mean': last_5_mean or 0.0,
                    'std': last_5_std or 0.0,
                    'min': min(home_last_5.get('min', 0), away_last_5.get('min', 0)) if home_last_5.get('min') and away_last_5.get('min') else 0,
                    'max': max(home_last_5.get('max', 0), away_last_5.get('max', 0)) if home_last_5.get('max') and away_last_5.get('max') else 0,
                    'count': min(home_last_5.get('count', 0), away_last_5.get('count', 0))
                },
                'last_10': {
                    'mean': last_10_mean or 0.0,
                    'std': last_10_std or 0.0,
                    'min': min(home_last_10.get('min', 0), away_last_10.get('min', 0)) if home_last_10.get('min') and away_last_10.get('min') else 0,
                    'max': max(home_last_10.get('max', 0), away_last_10.get('max', 0)) if home_last_10.get('max') and away_last_10.get('max') else 0,
                    'count': min(home_last_10.get('count', 0), away_last_10.get('count', 0))
                }
            },
            'h2h_matchup': {
                'games': h2h_totals.get('games_found', 0),
                'mean': h2h_totals.get('mean', None),
                'std': h2h_totals.get('std', None),
                'min': h2h_totals.get('min', None),
                'max': h2h_totals.get('max', None),
                'count': h2h_totals.get('games_found', 0),  # h2h_totals usa 'games_found', no 'count'
                'adjustment_factor': None,  # No disponible para Total Points
                'consistency_score': self._calculate_h2h_consistency(h2h_totals.get('std'), h2h_totals.get('mean')) if h2h_totals.get('std') and h2h_totals.get('mean') else None
            },
            'trend_analysis': {
                'direction': self._determine_trend_direction_from_means(last_5_mean, last_10_mean),
                'slope_5_games': self._calculate_slope(last_5_mean, last_10_mean),
                'consistency_score': self._calculate_consistency_from_std(last_5_std, last_10_mean),
                'form_rating': last_5_mean or 0.0
            }
        }
    
    def _build_is_win_historical_context(self, details_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construir historical_context para IS WIN basado en win rate
        
        Args:
            details_data: prediction_details de is_win_predict
            
        Returns:
            historical_context formateado con win rates
        """
        home_stats = details_data.get('home_team_stats', {})
        away_stats = details_data.get('away_team_stats', {})
        h2h_stats = details_data.get('h2h_stats', {})
        
        # Calcular promedios de win rate combinados
        home_last_5 = home_stats.get('last_5_games', {})
        away_last_5 = away_stats.get('last_5_games', {})
        home_last_10 = home_stats.get('last_10_games', {})
        away_last_10 = away_stats.get('last_10_games', {})
        
        # Season avg y std basado en win rate (no en puntos)
        home_win_rate = home_stats.get('win_rate', 0.0)
        away_win_rate = away_stats.get('win_rate', 0.0)
        season_avg_win_rate = (home_win_rate + away_win_rate) / 2
        
        # Calcular std de win rate
        # Si ambos equipos tienen datos
        home_count_10 = home_last_10.get('count', 0)
        away_count_10 = away_last_10.get('count', 0)
        
        # Para win rate, no hay std directo, calcularlo basado en variabilidad
        # Simplificación: usar diferencia entre home y away como proxy de variabilidad
        season_std_win_rate = abs(home_win_rate - away_win_rate) / 2
        
        # Last 5 y Last 10 promedios
        last_5_win_rate = None
        if home_last_5.get('count', 0) > 0 and away_last_5.get('count', 0) > 0:
            last_5_win_rate = (home_last_5.get('win_rate', 0) + away_last_5.get('win_rate', 0)) / 2
        
        last_10_win_rate = None
        if home_count_10 > 0 and away_count_10 > 0:
            last_10_win_rate = (home_last_10.get('win_rate', 0) + away_last_10.get('win_rate', 0)) / 2
        
        return {
            'games_analyzed': max(home_stats.get('total_games', 0), away_stats.get('total_games', 0)),
            'season_avg': season_avg_win_rate,
            'season_std': season_std_win_rate,
            'recent_form': {
                'last_5': {
                    'mean': last_5_win_rate or 0.0,
                    'std': 0.0,  # No aplica para win rate
                    'min': min(home_last_5.get('win_rate', 0), away_last_5.get('win_rate', 0)),
                    'max': max(home_last_5.get('win_rate', 0), away_last_5.get('win_rate', 0)),
                    'count': max(home_last_5.get('count', 0), away_last_5.get('count', 0))
                },
                'last_10': {
                    'mean': last_10_win_rate or 0.0,
                    'std': 0.0,  # No aplica para win rate
                    'min': min(home_last_10.get('win_rate', 0), away_last_10.get('win_rate', 0)),
                    'max': max(home_last_10.get('win_rate', 0), away_last_10.get('win_rate', 0)),
                    'count': max(home_count_10, away_count_10)
                }
            },
            'h2h_matchup': {
                'games': h2h_stats.get('games_found', 0),
                'mean': (h2h_stats.get('home_h2h_win_rate', 0) + h2h_stats.get('away_h2h_win_rate', 0)) / 2 if h2h_stats.get('games_found', 0) > 0 else None,
                'std': self._calculate_win_rate_std(h2h_stats) if h2h_stats.get('games_found', 0) > 0 else None,
                'min': min(h2h_stats.get('home_h2h_win_rate', 0), h2h_stats.get('away_h2h_win_rate', 0)) if h2h_stats.get('games_found', 0) > 0 else None,
                'max': max(h2h_stats.get('home_h2h_win_rate', 0), h2h_stats.get('away_h2h_win_rate', 0)) if h2h_stats.get('games_found', 0) > 0 else None,
                'count': h2h_stats.get('games_found', 0),
                'adjustment_factor': None,  # No aplicable para Is Win
                'consistency_score': self._calculate_h2h_consistency_win_rate(h2h_stats) if h2h_stats.get('games_found', 0) > 0 else None
            },
            'trend_analysis': {
                'direction': self._determine_trend_direction_from_means(last_5_win_rate, last_10_win_rate) if last_5_win_rate and last_10_win_rate else 'flat',
                'slope_5_games': self._calculate_slope(last_5_win_rate, last_10_win_rate) if last_5_win_rate and last_10_win_rate else 0.0,
                'consistency_score': self._calculate_consistency_from_std(season_std_win_rate, season_avg_win_rate),
                'form_rating': last_5_win_rate or 0.0
            }
        }
    
    def _build_team_points_historical_context(self, details_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Construir historical_context para TEAM POINTS o HALFTIME TEAM POINTS (equipo individual)
        
        Maneja:
        - Teams Points: usa actual_points_mean, actual_points_std, team_points_mean
        - Halftime Teams Points: usa actual_ht_mean, actual_ht_std, halftime_mean
        
        Args:
            details_data: prediction_details de teams_points_predict, ht_teams_points_predict o is_win_predict
            
        Returns:
            historical_context formateado
        """
        h2h_stats = details_data.get('h2h_stats', {})
        last_5 = details_data.get('last_5_games', {})
        last_10 = details_data.get('last_10_games', {})
        trend_analysis = details_data.get('trend_analysis', {})
        
        # Detectar si es Halftime Teams Points (usa actual_ht_mean) o Teams Points regular (usa actual_points_mean)
        if 'actual_ht_mean' in details_data:
            # Es Halftime Teams Points
            season_avg = details_data.get('actual_ht_mean', 0.0)
            season_std = details_data.get('actual_ht_std', 0.0)
            h2h_mean = h2h_stats.get('halftime_mean', h2h_stats.get('team_points_mean', None))
        else:
            # Es Teams Points regular o Is Win
            season_avg = details_data.get('actual_points_mean', details_data.get('actual_stats_mean', 0.0))
            season_std = details_data.get('actual_points_std', details_data.get('actual_stats_std', 0.0))
            h2h_mean = h2h_stats.get('team_points_mean', h2h_stats.get('h2h_mean', None))
        
        return {
            'games_analyzed': details_data.get('historical_games_used', 0),
            'season_avg': season_avg,
            'season_std': season_std,
            'recent_form': {
                'last_5': {
                    'mean': last_5.get('mean', 0.0),
                    'std': last_5.get('std', 0.0),
                    'min': last_5.get('min', 0),
                    'max': last_5.get('max', 0),
                    'count': last_5.get('count', 0)
                },
                'last_10': {
                    'mean': last_10.get('mean', 0.0),
                    'std': last_10.get('std', 0.0),
                    'min': last_10.get('min', 0),
                    'max': last_10.get('max', 0),
                    'count': last_10.get('count', 0)
                }
            },
            'h2h_matchup': {
                'games': h2h_stats.get('games_found', 0),
                'mean': h2h_mean,
                'std': h2h_stats.get('h2h_std', None),
                'min': h2h_stats.get('h2h_min', None),
                'max': h2h_stats.get('h2h_max', None),
                'count': h2h_stats.get('games_found', 0),
                'adjustment_factor': h2h_stats.get('h2h_factor', 1.0),
                'consistency_score': h2h_stats.get('consistency_score', 0.0)
            },
            'trend_analysis': {
                'direction': self._determine_trend_direction(trend_analysis.get('trend_5_games', 0)),
                'slope_5_games': trend_analysis.get('trend_5_games', 0.0),
                'consistency_score': trend_analysis.get('consistency_score', 0.0),
                'form_rating': trend_analysis.get('recent_form', 0.0)
            }
        }
    
    def _calculate_summary(self, player_level: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Calcular resumen de predicciones
        
        Args:
            player_level: Dict con home_players y away_players
            
        Returns:
            Dict con resumen de predicciones
        """
        try:
            home_players = player_level.get('home_players', [])
            away_players = player_level.get('away_players', [])
            all_players = home_players + away_players
            
            # Contar predicciones
            total_predictions = 0
            by_type = {}
            confidences = []
            high_confidence_count = 0
            
            for player in all_players:
                predictions = player.get('predictions', [])
                total_predictions += len(predictions)
                
                for pred in predictions:
                    bet_type = pred.get('bet_type', 'unknown')
                    by_type[bet_type] = by_type.get(bet_type, 0) + 1
                    
                    confidence = pred.get('confidence', 0.0)
                    confidences.append(confidence)
                    
                    if confidence >= 80:
                        high_confidence_count += 1
            
            # Calcular confianza promedio
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'total_predictions': total_predictions,
                'by_type': by_type,
                'by_team': {
                    'home': {
                        'players': len(home_players),
                        'predictions': sum(len(p.get('predictions', [])) for p in home_players)
                    },
                    'away': {
                        'players': len(away_players),
                        'predictions': sum(len(p.get('predictions', [])) for p in away_players)
                    }
                },
                'avg_confidence': round(avg_confidence, 1),
                'high_confidence_bets': high_confidence_count
            }
            
        except Exception as e:
            logger.error(f" Error calculando resumen: {e}")
            return {
                'total_predictions': 0,
                'by_type': {},
                'by_team': {'home': {'players': 0, 'predictions': 0}, 'away': {'players': 0, 'predictions': 0}},
                'avg_confidence': 0.0,
                'high_confidence_bets': 0
            }


def test_unified_predictor():
    """
    Función de prueba del predictor unificado con datos reales
    """
    print("=" * 80)
    print(" PROBANDO UNIFIED PREDICTOR - KNICKS VS CAVALIERS")
    print("=" * 80)
    
    # Datos de prueba del juego Knicks vs Cavaliers
    test_games_data = [
        {
            "gameId": "sr:match:knicks_cavs_20250124",
            "status": "scheduled",
            "scheduled": "2025-01-24T19:30:00+00:00",
            "venue": {
                "id": "msg-venue-id",
                "name": "Madison Square Garden",
                "city": "New York",
                "state": "NY",
                "capacity": 19812,
            },
            "homeTeam": {
                "teamId": "583ec70e-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "New York Knicks",
                "alias": "NYK",
                "conference": "Eastern",
                "division": "Atlantic",
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "sr:player:brunson",
                        "fullName": "Jalen Brunson",
                        "jerseyNumber": 11,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:towns",
                        "fullName": "Karl-Anthony Towns",
                        "jerseyNumber": 32,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:anunoby",
                        "fullName": "OG Anunoby",
                        "jerseyNumber": 8,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:hart",
                        "fullName": "Josh Hart",
                        "jerseyNumber": 3,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:robinson",
                        "fullName": "Mitchell Robinson",
                        "jerseyNumber": 23,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "awayTeam": {
                "teamId": "583ec773-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Cleveland Cavaliers",
                "alias": "CLE",
                "conference": "Eastern",
                "division": "Central",
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "sr:player:mitchell",
                        "fullName": "Donovan Mitchell",
                        "jerseyNumber": 45,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:garland",
                        "fullName": "Darius Garland",
                        "jerseyNumber": 10,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:mobley",
                        "fullName": "Evan Mobley",
                        "jerseyNumber": 4,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:allen",
                        "fullName": "Jarrett Allen",
                        "jerseyNumber": 31,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "sr:player:strus",
                        "fullName": "Max Strus",
                        "jerseyNumber": 1,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "coverage": {
                "broadcasters": [
                    {
                        "name": "MSG",
                        "type": "tv",
                    },
                ],
            },
        }
    ]
    
    # Inicializar predictor
    predictor = UnifiedPredictor()
    
    # Cargar todos los modelos
    print("\n Cargando todos los modelos...")
    if not predictor.load_all_models():
        print(" Error cargando modelos")
        return False
    
    print("\n[OK] Todos los modelos cargados exitosamente")
    
    # Hacer predicciones usando el método unificado
    print("\n" + "=" * 80)
    print(" EJECUTANDO PREDICCIONES UNIFICADAS")
    print("=" * 80)
    all_predictions = predictor.predict(test_games_data)
    
    # Convertir tipos de NumPy antes de serializar
    all_predictions_clean = convert_numpy_types(all_predictions)
    
    # Mostrar resultados
    print("\n" + "=" * 80)
    print(" RESULTADOS COMPLETOS")
    print("=" * 80)
    
    # Resumen de predicciones
    team_preds = all_predictions_clean.get('team_predictions', [])
    total_player_preds = sum(len(game.get('player_predictions', [])) for game in team_preds)
    
    print(f"\n[OK] PREDICCIONES GENERADAS:")
    print(f"  Juegos procesados: {len(team_preds)}")
    print(f"  Jugadores con predicciones: {total_player_preds}")
    
    if team_preds:
        game = team_preds[0]
        print(f"\n  Partido: {game.get('home_team', 'N/A')} vs {game.get('away_team', 'N/A')}")
        
        # Mostrar predicciones de nivel de partido
        if 'match_level' in game:
            match = game['match_level']
            print(f"\n  PREDICCIONES NIVEL PARTIDO:")
            for pred in match.get('predictions', []):
                print(f"    - {pred.get('bet_type', 'N/A')}: {pred.get('bet_line', 'N/A')} (confidence: {pred.get('confidence', 'N/A')}%)")
        
        # Contar predicciones de jugadores por tipo
        player_stats = {}
        for player in game.get('player_predictions', []):
            for pred in player.get('predictions', []):
                bet_type = pred.get('bet_type', 'unknown')
                player_stats[bet_type] = player_stats.get(bet_type, 0) + 1
        
        print(f"\n  PREDICCIONES JUGADORES POR TIPO:")
        for bet_type, count in player_stats.items():
            print(f"    - {bet_type}: {count} predicciones")
    
    # Guardar a archivo JSON
    output_file = 'unified_predictions_output.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_predictions_clean, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Predicciones guardadas en: {output_file}")
    
    print("\n" + "=" * 80)
    print(" PRUEBA COMPLETADA EXITOSAMENTE")
    print("=" * 80)
    return True


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_unified_predictor()