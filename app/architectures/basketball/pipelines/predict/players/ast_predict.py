#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN AST
========================================

Wrapper final unificado para predicciones de asistencias (AST) que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo AST completo con calibraciones elite
- Formato estándar para módulo de stacking
- Manejo robusto de errores y tolerancia conservadora

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Convertir con GameDataAdapter
3. Buscar datos históricos específicos del jugador
4. Generar features dinámicas
5. Aplicar modelo completo con calibraciones
6. Retornar formato estándar para stacking
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import logging

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Importar modelos y data loaders
from app.architectures.basketball.src.models.players.ast.model_ast import XGBoostASTModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence_predict import PlayersConfidence

logger = logging.getLogger(__name__)

class ASTPredictor:
    """
    Pipeline encargado de predecir asistencias para un jugador individual en un game (AST)
    
    Args:
        model: Modelo AST completo con calibraciones
        historical_players: Datos históricos de jugadores
        historical_teams: Datos históricos de equipos
        game_adapter: GameDataAdapter para convertir datos de SportRadar
        is_loaded: Flag para verificar si el modelo está cargado
        tolerance: Tolerancia para la predicción

    Returns:
        Predicción de asistencias para un jugador 
    """
    
    def __init__(self):
        """Inicializar el predictor AST unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = PlayersConfidence()
        self.is_loaded = False
        self.tolerance = -2  # Tolerancia conservadora individual
    
    def load_data_and_model(self) -> bool:
        """
        Cargar datos históricos y modelo entrenado
        
        Returns:
            True si se cargó exitosamente
        """
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
            
            # Inicializar modelo AST completo (wrapper)
            logger.info("🤖 Inicializando modelo AST completo (wrapper)...")
            self.model = XGBoostASTModel(teams_df=self.historical_teams)
            
            # Cargar modelo entrenado
            model_path = "app/architectures/basketball/.joblib/ast_model.joblib"
            logger.info(f"📦 Cargando modelo desde: {model_path}")
            self.model.load_model(model_path)
            
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos y modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def predict_game(self, game_data: Dict[str, Any], target_player: str) -> Dict[str, Any]:
        """
        Metodo principal para predecir AST desde datos de insumo (SportRadar)
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            Predicción en formato estándar de salida
        """
        if not self.is_loaded:
            return {'error': 'Modelo no cargado. Ejecutar load_data_and_model() primero.'}
        
        try:
            # Convertir datos de SportRadar con GameDataAdapter
            players_df, teams_df = self.game_adapter.convert_game_to_dataframes(game_data)
            
            # Buscar el jugador objetivo con búsqueda inteligente
            target_row = self.common_utils._smart_player_search(players_df, target_player)
            
            if target_row.empty:
                available_players = list(players_df['Player'].unique())
                # Intentar sugerir jugadores similares
                similar_players = self.common_utils._find_similar_players(target_player, available_players)
                return {
                    'error': f'Jugador "{target_player}" no encontrado',
                    'available_players': available_players[:10],
                    'similar_suggestions': similar_players[:5]
                }
            
            # Extraer datos del jugador y verificar estado
            player_data = target_row.iloc[0].to_dict()
            
            # Filtrar jugadores disponibles desde el roster
            available_players = self.confidence_calculator.filter_available_players_from_roster(game_data)
            
            if target_player not in available_players:
                logger.info(f"❌ {target_player} no disponible en el roster")
                return None
     
            # Extraer información adicional desde SportRadar
            is_home = self.common_utils._get_is_home_from_sportradar(game_data, target_player)
            is_started = self.common_utils._get_is_started_from_sportradar(game_data, target_player)
            current_team = self.common_utils._get_current_team_from_sportradar(game_data, target_player)
           
            # Agregar información extraída al player_data
            player_data['is_home'] = is_home
            player_data['is_started'] = is_started
            player_data['current_team'] = current_team

            # Corregir formato de fecha para evitar problemas de timezone
            if 'Date' in player_data and pd.notna(player_data['Date']):
                # Convertir a string sin timezone para compatibilidad
                if hasattr(player_data['Date'], 'strftime'):
                    player_data['Date'] = player_data['Date'].strftime('%Y-%m-%d')
                elif hasattr(player_data['Date'], 'date'):
                    player_data['Date'] = str(player_data['Date'].date())
            
            # Hacer predicción usando el método interno (pasando game_data también)
            prediction_result = self.predict_single_player(player_data, game_data)
            
            if prediction_result is None:
                logger.info(f"⚠️  Predicción no realizada, es menor a 4 asistencias")
                return None
            
            if 'error' in prediction_result:
                logger.error(f"❌ Error en predicción interna: {prediction_result['error']}")
                return None
            
            # Extraer predicción y confianza
            raw_prediction = prediction_result['raw_prediction']
            final_prediction = prediction_result['ast_prediction']
            confidence_percentage = prediction_result['confidence_percentage']
            prediction_details = prediction_result.get('prediction_details', {})
            
            # Obtener información de equipos desde game_data
            home_team = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            return {
                "home_team": home_team,
                "away_team": away_team,
                "target_type": "player",
                "target_name": target_player,
                "bet_line": str(int(final_prediction)),
                "bet_type": "AST",
                "confidence_percentage": round(confidence_percentage, 1),
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    def predict_single_player(self, player_data: Dict[str, Any], game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir asistencias para un jugador individual
        
        Args:
            player_data: Diccionario con datos del jugador
                - Player: Nombre del jugador
                - Team: Equipo del jugador
                - Opp: Equipo oponente
                - Date: Fecha del juego (opcional)
                
        Returns:
            Diccionario con predicción y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            
            # PASO CRÍTICO: Buscar datos históricos del jugador específico
            player_name = player_data.get('Player', 'Unknown')
            current_team = player_data.get('current_team', 'Unknown')
            
            # Filtrar datos históricos del jugador específico usando búsqueda inteligente
            player_historical = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if len(player_historical) == 0:
                logger.warning(f"⚠️ No se encontraron datos históricos para {player_name}")
                # Usar datos de jugadores similares o promedio
                player_historical = self.historical_players.head(100).copy()
            else:
                logger.info(f"✅ Encontrados {len(player_historical)} registros históricos para {player_name}")
            
            # Intentar usar datos del equipo actual primero, si no hay suficientes usar historial completo
            current_team = player_data.get('current_team', 'Unknown')
            if len(player_historical) > 0 and current_team != 'Unknown':
                current_team_data = player_historical[player_historical['Team'] == current_team]
                if len(current_team_data) >= 5:  # Mínimo 5 juegos con el equipo actual
                    player_historical = current_team_data.copy()
                    logger.info(f"🏀 Usando {len(player_historical)} registros del equipo actual ({current_team}) para {player_name}")
                else:
                    # Si no hay suficientes datos del equipo actual, usar TODOS los datos históricos
                    logger.info(f"📅 Pocos datos de {current_team} ({len(current_team_data)} juegos), usando TODOS los {len(player_historical)} registros históricos para {player_name}")
            else:
                logger.info(f"🏀 Usando TODOS los {len(player_historical)} registros históricos para {player_name} (sin filtro por equipo)")
            
            # 🎯 USAR SOLO DATOS HISTÓRICOS - Sin agregar fila de predicción
            combined_df = player_historical.copy()
            
            # Hacer predicción 
            predictions = self.model.predict(combined_df)
            
            # Usar promedio de las últimas predicciones basadas en histórico
            if len(predictions) > 0:
                # Tomar promedio de las últimas 5 predicciones para mayor estabilidad
                recent_predictions = predictions[-5:] if len(predictions) >= 5 else predictions
                raw_prediction = np.mean(recent_predictions)
            else:
                raw_prediction = 0
            
            # Calcular estadísticas para confianza
            actual_stats_mean = player_historical['assists'].mean() if len(player_historical) > 0 else 0
            actual_stats_std = player_historical['assists'].std() if len(player_historical) > 0 else 0
            prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            
            # Calcular predicción estabilizada (85% modelo + 15% histórico)
            stabilized_prediction = (raw_prediction * 0.85) + (actual_stats_mean * 0.15)
            
            # CALCULAR ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
            # Últimos 5 juegos
            last_5_games = player_historical.tail(5)['assists'] if len(player_historical) >= 5 else player_historical['assists']
            last_5_stats = {
                'mean': round(last_5_games.mean(), 1) if len(last_5_games) > 0 else 0,
                'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                'min': int(last_5_games.min()) if len(last_5_games) > 0 else 0,
                'max': int(last_5_games.max()) if len(last_5_games) > 0 else 0,
                'count': len(last_5_games)
            }
            
            # Últimos 10 juegos
            last_10_games = player_historical.tail(10)['assists'] if len(player_historical) >= 10 else player_historical['assists']
            last_10_stats = {
                'mean': round(last_10_games.mean(), 1) if len(last_10_games) > 0 else 0,
                'std': round(last_10_games.std(), 1) if len(last_10_games) > 1 else 0,
                'min': int(last_10_games.min()) if len(last_10_games) > 0 else 0,
                'max': int(last_10_games.max()) if len(last_10_games) > 0 else 0,
                'count': len(last_10_games)
            }
            
            # Análisis de tendencia
            if len(player_historical) >= 5:
                recent_5_mean = last_5_games.mean()
                recent_10_mean = last_10_games.mean() if len(player_historical) >= 10 else recent_5_mean
                trend_5_games = recent_5_mean - recent_10_mean
            else:
                trend_5_games = 0
                recent_5_mean = actual_stats_mean
            
            # Score de consistencia (inverso de la desviación estándar)
            consistency_score = max(0, 100 - (actual_stats_std * 5)) if actual_stats_std > 0 else 100
            
            # Forma reciente (promedio de últimos 3 juegos)
            recent_form = player_historical.tail(3)['assists'].mean() if len(player_historical) >= 3 else actual_stats_mean
            
            # CALCULAR ESTADÍSTICAS H2H DETALLADAS
            h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                player_name=player_name,
                opponent_team=player_data.get('Opp', 'Unknown'),
                target_stat='assists',
                max_games=10
            )
            
            # APLICAR FACTOR H2H A LA PREDICCIÓN
            h2h_factor = h2h_stats.get('h2h_factor', 1.0)
            if h2h_factor != 1.0 and h2h_stats.get('games_found', 0) >= 3:
                raw_prediction_adjusted = raw_prediction * h2h_factor
                logger.info(f"🎯 Aplicando factor H2H {h2h_factor:.3f} a predicción AST: {raw_prediction:.1f} -> {raw_prediction_adjusted:.1f}")
            else:
                raw_prediction_adjusted = raw_prediction
            
            # Calcular confianza usando PlayersConfidence
            confidence_percentage = self.confidence_calculator.calculate_player_confidence(
                raw_prediction=raw_prediction,
                stabilized_prediction=stabilized_prediction,
                tolerance=self.tolerance,  # Tolerancia individual del predictor
                prediction_std=prediction_std,
                actual_stats_std=actual_stats_std,
                historical_games=len(player_historical),
                player_data=player_data,
                opponent_team=player_data.get('Opp', 'Unknown'),
                game_date=player_data.get('Date'),
                game_data=game_data  # Datos en tiempo real
            )
            
            # Aplicar tolerancia individual del predictor
            ast_prediction = max(0, raw_prediction_adjusted + self.tolerance)  # self.tolerance es -2
            
            # REGLA DE CASAS DE APUESTAS: No inferir predicciones menores a 4 AST
            if ast_prediction < 4:
                logger.info(f"⚠️  Predicción {ast_prediction:.1f} AST < 4, no se infiere (casas de apuestas manejan líneas ≥4)")
                return None
            
            return {
                'raw_prediction': raw_prediction,
                'ast_prediction': int(ast_prediction),
                'confidence_percentage': confidence_percentage,
                'prediction_details': {
                    'player_id': self.common_utils._get_player_id(self.common_utils._normalize_name(player_name), player_data.get('Team', 'Unknown')),
                    'player': player_name,
                    'team_id': self.common_utils._get_team_id(player_data.get('Team', 'Unknown')),
                'team': player_data.get('Team', 'Unknown'),
                    'opponent_id': self.common_utils._get_team_id(player_data.get('Opp', 'Unknown')),
                'opponent': player_data.get('Opp', 'Unknown'),
                    'tolerance_applied': self.tolerance,
                    'historical_games_used': len(player_historical),
                    'raw_prediction': raw_prediction,
                    'h2h_adjusted_prediction': round(raw_prediction_adjusted, 1),
                    'final_prediction': int(ast_prediction),
                    'actual_stats_mean': round(actual_stats_mean, 1),
                    'actual_stats_std': round(actual_stats_std, 1),
                    'prediction_std': round(prediction_std, 1),
                    'last_5_games': last_5_stats,
                    'last_10_games': last_10_stats,
                    'trend_analysis': {
                        'trend_5_games': round(trend_5_games, 1),
                        'consistency_score': round(consistency_score, 1),
                        'recent_form': round(recent_form, 1)
                    },
                    'performance_metrics': {
                        'stabilized_prediction': round(stabilized_prediction, 1),
                        'confidence_factors': {
                            'tolerance': self.tolerance,
                            'historical_games': len(player_historical),
                            'data_quality': 'high' if len(player_historical) >= 20 else 'medium'
                        }
                    },
                    'h2h_stats': {
                        'games_found': h2h_stats.get('games_found', 0),
                        'h2h_mean': h2h_stats.get('h2h_mean'),
                        'h2h_std': h2h_stats.get('h2h_std'),
                        'h2h_min': h2h_stats.get('h2h_min'),
                        'h2h_max': h2h_stats.get('h2h_max'),
                        'h2h_factor': h2h_stats.get('h2h_factor', 1.0),
                        'consistency_score': h2h_stats.get('consistency_score', 0),
                        'last_5_mean': h2h_stats.get('last_5_mean'),
                        'last_10_mean': h2h_stats.get('last_10_mean')
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            import traceback
            traceback.print_exc()
            return {
                'player': player_data.get('Player', 'Unknown'),
                'error': str(e),
                'ast_prediction': None
            }


def test_ast_predictor():
    """Función de prueba rápida del predictor AST"""
    print("🧪 PROBANDO AST PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = ASTPredictor()
    
    # Cargar datos y modelo
    print("📂 Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print("❌ Error cargando modelo")
        return False
    
    # Prueba con datos simulados de SportRadar
    print("\n🎯 Prueba con datos simulados de SportRadar:")
    
    # Simular datos de SportRadar con Trae Young (elite)
    mock_sportradar_game = {
        "gameId": "sr:match:12345",
        "scheduled": "2024-01-15T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Atlanta Hawks",
            "alias": "ATL",
            "players": [
                {
                    "playerId": "sr:player:123",
                    "fullName": "Trae Young",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "11",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:999",
                    "fullName": "John Collins",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "20",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Charlotte Hornets", 
            "alias": "CHA",
            "players": [
                {
                    "playerId": "sr:player:456",
                    "fullName": "LaMelo Ball",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "1",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:789",
                    "fullName": "Terry Rozier",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "3",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:888",
                    "fullName": "P.J. Washington",
                    "position": "F",
                    "starter": False,
                    "status": "ACT",
                    "jerseyNumber": "25",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "State Farm Arena",
            "capacity": 18118
        }
    }
    
    # Probar predicción desde SportRadar
    print("   Prediciendo Trae Young desde datos SportRadar:")
    sportradar_result = predictor.predict_game(
        mock_sportradar_game, 
        "Trae Young"
    )
    
    if sportradar_result is not None and 'error' not in sportradar_result:
        print("   ✅ Resultado SportRadar (formato JSON exacto):")
        # Convertir tipos numpy a tipos nativos de Python para JSON
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            return obj
        
        sportradar_result_clean = convert_numpy_types(sportradar_result)
        print(json.dumps(sportradar_result_clean, indent=4, ensure_ascii=False))
    elif sportradar_result is None:
        print("      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"   ❌ Error: {sportradar_result['error']}")
        if 'available_players' in sportradar_result:
            print(f"   Jugadores disponibles: {sportradar_result['available_players']}")
    
    # Probar búsqueda inteligente con nombres problemáticos
    print("\n🧠 Pruebas de búsqueda inteligente:")
    
    # Caso 1: Jugador elite (Trae Young)
    print("   1. Búsqueda sin acento: 'Trae Young' -> 'Trae Young'")
    trae_result = predictor.predict_game(mock_sportradar_game, "Trae Young")
    if trae_result is not None and 'error' not in trae_result:
        print(f"      ✅ Encontrado: {trae_result['target_name']} -> bet_line: {trae_result['bet_line']}")
    elif trae_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ❌ Error: {trae_result['error']}")
    
    # Caso 2: Jugador elite (LaMelo Ball)
    print("   2. Búsqueda sin acento: 'LaMelo Ball' -> 'LaMelo Ball'")
    lamelo_result = predictor.predict_game(mock_sportradar_game, "LaMelo Ball")
    if lamelo_result is not None and 'error' not in lamelo_result:
        print(f"      ✅ Encontrado: {lamelo_result['target_name']} -> bet_line: {lamelo_result['bet_line']}")
    elif lamelo_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ❌ Error: {lamelo_result['error']}")
    
    # Caso 3: Búsqueda case-insensitive
    print("   3. Búsqueda case-insensitive: 'trae young'")
    trae_lower_result = predictor.predict_game(mock_sportradar_game, "trae young")
    if trae_lower_result is not None and 'error' not in trae_lower_result:
        print(f"      ✅ Encontrado: {trae_lower_result['target_name']} -> bet_line: {trae_lower_result['bet_line']}")
    elif trae_lower_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ❌ Error: {trae_lower_result['error']}")
    
    # Caso 4: Búsqueda parcial (solo apellido)
    print("   4. Búsqueda parcial: 'Young'")
    young_partial_result = predictor.predict_game(mock_sportradar_game, "Young")
    if young_partial_result is not None and 'error' not in young_partial_result:
        print(f"      ✅ Encontrado: {young_partial_result['target_name']} -> bet_line: {young_partial_result['bet_line']}")
    elif young_partial_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ❌ Error: {young_partial_result['error']}")
    
    # Caso 5: Jugador suplente (P.J. Washington)
    print("   5. Jugador suplente: 'P.J. Washington' (starter: False)")
    washington_result = predictor.predict_game(mock_sportradar_game, "P.J. Washington")
    if washington_result is not None and 'error' not in washington_result:
        print(f"      ✅ Predicción exitosa:")
        washington_result_clean = convert_numpy_types(washington_result)
        print(json.dumps(washington_result_clean, indent=8, ensure_ascii=False))
    elif washington_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ❌ Error: {washington_result['error']}")
    
    # Caso 6: Jugador lesionado (status: INJ)
    print("   6. Jugador lesionado: 'Russell Westbrook' (status: INJ)")
    westbrook_result = predictor.predict_game(mock_sportradar_game, "Russell Westbrook")
    if westbrook_result is None:
        print(f"      ✅ Comportamiento esperado: No se hizo predicción para jugador lesionado")
    else:
        print(f"      ⚠️ Inesperado: Se hizo predicción para jugador lesionado: {westbrook_result}")
    
    # Caso 7: Jugador no existente (para probar sugerencias)
    print("   7. Jugador inexistente: 'LeBron James'")
    lebron_result = predictor.predict_game(mock_sportradar_game, "LeBron James")
    if lebron_result is not None and 'error' in lebron_result:
        print(f"      ❌ Error esperado: {lebron_result['error']}")
        if 'similar_suggestions' in lebron_result:
            print(f"      💡 Sugerencias: {lebron_result['similar_suggestions']}")
    elif lebron_result is None:
        print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"      ⚠️ Inesperado: {lebron_result}")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_ast_predictor()
