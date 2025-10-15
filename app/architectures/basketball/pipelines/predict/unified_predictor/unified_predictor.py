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
            logger.info("🚀 Cargando todos los modelos...")
            
            # Cargar modelos de equipos
            logger.info("📊 Cargando predictores de equipos...")
            team_predictors = [
                ("IsWin", self.is_win_predictor),
                ("TotalPoints", self.total_points_predictor), 
                ("TeamsPoints", self.teams_points_predictor)
            ]
            
            for name, predictor in team_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f"❌ Error cargando {name}")
                    return False
                logger.info(f"✅ {name} cargado exitosamente")
            
            # Cargar modelos de halftime
            logger.info("⏰ Cargando predictores de halftime...")
            halftime_predictors = [
                ("HT_TeamsPoints", self.ht_teams_points_predictor),
                ("HT_TotalPoints", self.ht_total_points_predictor)
            ]
            
            for name, predictor in halftime_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f"❌ Error cargando {name}")
                    return False
                logger.info(f"✅ {name} cargado exitosamente")
            
            # Cargar modelos de jugadores
            logger.info("🏀 Cargando predictores de jugadores...")
            player_predictors = [
                ("PTS", self.pts_predictor),
                ("3PT", self.threept_predictor),
                ("AST", self.ast_predictor),
                ("TRB", self.trb_predictor),
                ("DD", self.dd_predictor)
            ]
            
            for name, predictor in player_predictors:
                if not predictor.load_data_and_model():
                    logger.error(f"❌ Error cargando {name}")
                    return False
                logger.info(f"✅ {name} cargado exitosamente")
            
            self.is_loaded = True
            logger.info("🎯 Todos los modelos cargados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            return False

    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]], max_workers: int = 8) -> Dict[str, Any]:
        """
        Método unificado para predicciones que maneja tanto un solo juego como múltiples juegos.
        
        Args:
            data: Datos del juego (dict) o lista de juegos (list) en formato SportRadar
            max_workers: Número máximo de hilos paralelos para múltiples juegos (default: 8)
            
        Returns:
            Dict con predicciones del juego o juegos
        """
        if not self.is_loaded:
            logger.error("❌ Modelos no cargados. Ejecutar load_all_models() primero.")
            return {'error': 'Modelos no cargados'}
        
        try:
            # Detectar si es un solo juego o múltiples juegos
            if isinstance(data, dict):
                # Un solo juego
                logger.info("🎯 Predicción de un solo juego")
                return self.predict_single_game(data)
            elif isinstance(data, list):
                # Múltiples juegos
                logger.info(f"🚀 Predicción de {len(data)} juegos en paralelo")
                return self.predict_games(data, max_workers)
            else:
                logger.error(f"❌ Tipo de datos no válido: {type(data)}")
                return {'error': f'Tipo de datos no válido: {type(data)}'}
                
        except Exception as e:
            logger.error(f"❌ Error en predicción unificada: {e}")
            return {'error': str(e)}

    def predict_games(self, games_data: List[Dict[str, Any]], max_workers: int = 8) -> Dict[str, Any]:
        """
        Predecir múltiples juegos EN PARALELO para máxima velocidad
        
        Args:
            games_data: Lista de datos de juegos de SportRadar
            max_workers: Número máximo de hilos paralelos (default: 8)
            
        Returns:
            Diccionario con todas las predicciones organizadas por juego
        """
        if not self.is_loaded:
            logger.error("❌ Modelos no cargados. Ejecutar load_all_models() primero.")
            return {'error': 'Modelos no cargados'}
        
        all_predictions = {
            'games_processed': len(games_data),
            'timestamp': datetime.now().isoformat(),
            'predictions': [],
        }
        
        logger.info(f"🚀 Iniciando predicciones PARALELAS para {len(games_data)} juegos con {max_workers} hilos")
        
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
                    game_predictions = future.result()
                    # Convertir tipos de NumPy en las predicciones del juego
                    game_predictions_clean = convert_numpy_types(game_predictions)
                    all_predictions['predictions'].append(game_predictions_clean)
                    
                    logger.info(f"✅ Juego {completed_games}/{len(games_data)} completado: {game_data.get('homeTeam', {}).get('name', 'Home')} vs {game_data.get('awayTeam', {}).get('name', 'Away')}")
                    
                except Exception as e:
                    logger.error(f"❌ Error procesando juego {game_idx + 1}: {e}")
                    all_predictions['predictions'].append({
                        'game_info': {
                            'game_id': game_data.get('gameId', 'unknown'),
                            'home_team': game_data.get('homeTeam', {}).get('name', 'Home Team'),
                            'away_team': game_data.get('awayTeam', {}).get('name', 'Away Team'),
                        },
                        'error': str(e)
                    })
        
        logger.info(f"🎯 Predicciones paralelas completadas: {len(all_predictions['predictions'])} juegos procesados")
        return all_predictions

    def predict_single_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predecir un solo juego (teams + todos los jugadores disponibles)
        
        Args:
            game_data: Datos del juego de SportRadar
            
        Returns:
            Diccionario con todas las predicciones del juego
        """
        try:
            game_info = {
                'game_id': game_data.get('gameId', 'unknown'),
                'home_team': game_data.get('homeTeam', {}).get('name', 'Home Team'),
                'away_team': game_data.get('awayTeam', {}).get('name', 'Away Team'),
                'scheduled': game_data.get('scheduled', 'unknown')
            }
            
            # 1. Predicciones de equipos
            logger.info("📊 Haciendo predicciones de equipos...")
            team_predictions = self._predict_teams(game_data)
            
            # 2. Predicciones de jugadores
            logger.info("🏀 Haciendo predicciones de jugadores...")
            player_predictions = self._predict_all_players(game_data)
            
            return {
                'game_info': game_info,
                'team_predictions': team_predictions,
                'player_predictions': player_predictions,
                'summary': {
                    'team_predictions_count': len([p for p in team_predictions.values() if p is not None]),
                    'player_predictions_count': len(player_predictions)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error procesando juego: {e}")
            return {
                'game_info': game_info if 'game_info' in locals() else {'error': 'Failed to extract game info'},
                'error': str(e)
            }

    def _predict_teams(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hacer todas las predicciones de equipos EN PARALELO
        """
        team_predictions = {}
        
        try:
            # Definir tareas de predicción de equipos
            team_tasks = {
                'is_win': lambda: self.is_win_predictor.predict_game(game_data),
                'total_points': lambda: self.total_points_predictor.predict_game(game_data),
                'teams_points': lambda: self.teams_points_predictor.predict_game(game_data),
                'ht_teams_points': lambda: self.ht_teams_points_predictor.predict_game(game_data),
                'ht_total_points': lambda: self.ht_total_points_predictor.predict_game(game_data)
            }
            
            # Ejecutar todas las predicciones de equipos en paralelo
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_task = {
                    executor.submit(task_func): task_name 
                    for task_name, task_func in team_tasks.items()
                }
                
                for future in as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        result = future.result()
                        
                        # Procesar resultados según el tipo
                        if task_name == 'teams_points' and isinstance(result, list):
                            team_predictions['home_team_points'] = result[0] if len(result) > 0 else None
                            team_predictions['away_team_points'] = result[1] if len(result) > 1 else None
                        elif task_name == 'ht_teams_points' and isinstance(result, list):
                            team_predictions['ht_home_team_points'] = result[0] if len(result) > 0 else None
                            team_predictions['ht_away_team_points'] = result[1] if len(result) > 1 else None
                        else:
                            team_predictions[task_name] = result
                            
                    except Exception as e:
                        logger.error(f"❌ Error en predicción {task_name}: {e}")
                        team_predictions[f'{task_name}_error'] = str(e)
                
        except Exception as e:
            logger.error(f"❌ Error en predicciones de equipos: {e}")
            team_predictions['error'] = str(e)
        
        return team_predictions


    def _predict_all_players(self, game_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Hacer predicciones para todos los jugadores disponibles EN PARALELO
        """
        all_player_predictions = []
        
        try:
            # Obtener todos los jugadores de ambos equipos
            home_players = game_data.get('homeTeam', {}).get('players', [])
            away_players = game_data.get('awayTeam', {}).get('players', [])
            all_players = home_players + away_players
            
            # Filtrar solo jugadores activos
            active_players = [
                player for player in all_players 
                if player.get('status', '') == 'ACT' and not player.get('injuries', [])
            ]
            
            logger.info(f"🏃 Encontrados {len(active_players)} jugadores activos para predicciones PARALELAS")
            
            # Ejecutar predicciones de jugadores en paralelo
            with ThreadPoolExecutor(max_workers=min(16, len(active_players))) as executor:
                # Crear tareas para cada jugador
                future_to_player = {
                    executor.submit(self._predict_single_player, game_data, player.get('fullName', '')): player
                    for player in active_players
                    if player.get('fullName', '')
                }
                
                # Procesar resultados conforme se completan
                for future in as_completed(future_to_player):
                    player = future_to_player[future]
                    player_name = player.get('fullName', '')
                    
                    try:
                        player_predictions = future.result()
                        
                        if player_predictions:
                            all_player_predictions.append({
                                'player_name': player_name,
                                'player_info': {
                                    'position': player.get('position', ''),
                                    'jersey_number': player.get('jerseyNumber', ''),
                                    'status': player.get('status', '')
                                },
                                'predictions': player_predictions
                            })
                            
                    except Exception as e:
                        logger.error(f"❌ Error prediciendo {player_name}: {e}")
                        
        except Exception as e:
            logger.error(f"❌ Error en predicciones de jugadores: {e}")
            
        return all_player_predictions

    def _predict_single_player(self, game_data: Dict[str, Any], player_name: str) -> Dict[str, Any]:
        """
        Hacer todas las predicciones para un jugador específico EN PARALELO
        REGLA: NO mostrar predicciones con valor 0 (puntos, asistencias, rebotes, triples)
        """
        predictions = {}
        
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
                        
                        # Validar y agregar resultado según el tipo
                        if task_name == 'double_double':
                            if result is not None and result.get('bet_line') == 'yes':
                                predictions[task_name] = result
                        else:
                            if result is not None and self._is_valid_prediction(result, task_name):
                                predictions[task_name] = result
                                
                    except Exception as e:
                        logger.error(f"❌ Error en predicción {task_name} para {player_name}: {e}")
                
        except Exception as e:
            logger.error(f"❌ Error prediciendo {player_name}: {e}")
            
        return predictions

    def _is_valid_prediction(self, prediction_result: Dict[str, Any], prediction_type: str) -> bool:
        """
        Validar si una predicción debe mostrarse (valor > 0)
        
        Args:
            prediction_result: Resultado de la predicción
            prediction_type: Tipo de predicción ('points', 'three_points', 'assists', 'rebounds')
            
        Returns:
            True si debe mostrarse, False si es 0
        """
        try:
            bet_line = prediction_result.get('bet_line', '0')
            
            # Convertir a número
            if isinstance(bet_line, str):
                value = float(bet_line)
            else:
                value = float(bet_line)
            
            # Solo mostrar si > 0
            is_valid = value > 0
            
            if not is_valid:
                logger.info(f"🚫 Predicción {prediction_type} filtrada por valor 0: {prediction_result.get('target_name', 'Unknown')}")
            
            return is_valid
            
        except (ValueError, TypeError):
            logger.error(f"❌ Error validando predicción {prediction_type}: {bet_line}")
            return False


def test_unified_predictor():
    """
    Función de prueba del predictor unificado con datos reales
    """
    print("🚀 PROBANDO UNIFIED PREDICTOR")
    print("=" * 80)
    
    # Datos de prueba del juego OKC vs DEN
    test_games_data = [
        {
            "gameId": "f71cb64f-4d52-4e2b-a3db-7436b798476d",
            "status": "scheduled",
            "scheduled": "2025-05-18T19:30:00+00:00",
            "venue": {
                "id": "a13af216-4409-5021-8dd5-255cc71bffc3",
                "name": "Paycom Center",
                "city": "Oklahoma City",
                "state": "OK",
                "capacity": 18203,
            },
            "homeTeam": {
                "teamId": "583ecfff-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Oklahoma City Thunder",
                "alias": "OKC",
                "conference": None,
                "division": None,
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "d9ea4a8f-ff51-408d-b518-980efc2a35a1",
                        "fullName": "Shai Gilgeous-Alexander",
                        "jerseyNumber": 2,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "eb91a4c8-1a8a-46bf-86e6-e16950b67ef6",
                        "fullName": "Chet Holmgren",
                        "jerseyNumber": 7,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "62c44a90-f280-438a-9c7e-252f4f376283",
                        "fullName": "Jalen Williams",
                        "jerseyNumber": 8,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "3f7e2350-e208-4791-98c2-684b53bb5a9a",
                        "fullName": "Luguentz Dort",
                        "jerseyNumber": 5,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "38745a56-7472-4844-a2dc-f61d3bcd941f",
                        "fullName": "Isaiah Hartenstein",
                        "jerseyNumber": 55,
                        "position": "C",
                        "starter": False,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "awayTeam": {
                "teamId": "583ed102-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Denver Nuggets",
                "alias": "DEN",
                "conference": None,
                "division": None,
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "f2625432-3903-4f90-9b0b-2e4f63856bb0",
                        "fullName": "Nikola Jokić",
                        "jerseyNumber": 15,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "685576ef-ea6c-4ccf-affd-18916baf4e60",
                        "fullName": "Jamal Murray",
                        "jerseyNumber": 27,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "3a7d6510-00e9-4265-81df-864a1f547269",
                        "fullName": "Michael Porter Jr.",
                        "jerseyNumber": 1,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "20f85838-0bd5-4c1f-ab85-a308bafaf5bc",
                        "fullName": "Aaron Gordon",
                        "jerseyNumber": 32,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [
                            {
                                "id": "638efe6b-bb02-4c41-a88c-b6bc09f97155",
                                "type": "Hamstring",
                                "location": "Hamstring",
                                "comment": "The Nuggets announced that Gordon is expected to be listed as Questionable for Sunday's (May. 18) Game 7 against the Thunder, per Tony Jones of The Athletic.",
                                "startDate": "2025-05-16",
                                "expectedReturn": None,
                            },
                        ],
                    },
                    {
                        "playerId": "74a45eed-f2b0-4886-ae71-d04cf7d59528",
                        "fullName": "Russell Westbrook",
                        "jerseyNumber": 4,
                        "position": "PG",
                        "starter": False,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "coverage": {
                "broadcasters": [
                    {
                        "name": "ABC",
                        "type": "tv",
                    },
                ],
            },
        }
    ]
    
    # Inicializar predictor
    predictor = UnifiedPredictor()
    
    # Cargar todos los modelos
    print("📂 Cargando todos los modelos...")
    if not predictor.load_all_models():
        print("❌ Error cargando modelos")
        return False
    
    # Hacer predicciones usando el método unificado
    print("\n🎯 Haciendo predicciones usando método unificado...")
    all_predictions = predictor.predict(test_games_data)
    
    # Mostrar resultados
    print("\n📋 RESULTADOS COMPLETOS:")
    print("=" * 80)
    
    # Convertir tipos de NumPy antes de serializar
    all_predictions_clean = convert_numpy_types(all_predictions)
    print(json.dumps(all_predictions_clean, indent=2, ensure_ascii=False))
    
    print("\n✅ Prueba completada exitosamente")
    return True


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    test_unified_predictor()