#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN TRB
========================================

Wrapper final unificado para predicciones de rebotes totales (TRB) que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo TRB completo con calibraciones elite
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
from app.architectures.basketball.src.models.players.trb.model_trb import XGBoostTRBModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence.confidence_players import PlayersConfidence

logger = logging.getLogger(__name__)

class TRBPredictor:
    """
    Wrapper final unificado para predicciones TRB
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self):
        """Inicializar el predictor TRB unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = PlayersConfidence()
        self.is_loaded = False
        self.tolerance = -2 # Tolerancia conservadora individual

        # Cargar datos y modelo automáticamente
        self.load_data_and_model()
    
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
            
            # Inicializar modelo TRB completo (wrapper)
            logger.info("🤖 Inicializando modelo TRB completo (wrapper)...")
            self.model = XGBoostTRBModel(teams_df=self.historical_teams)
            
            # Cargar modelo entrenado
            model_path = "app/architectures/basketball/.joblib/trb_model.joblib"
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
        Metodo principal para predecir TRB desde datos de insumo (SportRadar)
        
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
            
            # Extraer oponente correcto del juego actual
            home_team_name = game_data.get('homeTeam', {}).get('name', '')
            away_team_name = game_data.get('awayTeam', {}).get('name', '')
            home_team_abbr = self.common_utils._get_team_abbreviation(home_team_name)
            away_team_abbr = self.common_utils._get_team_abbreviation(away_team_name)
            current_team_abbr = self.common_utils._get_team_abbreviation(current_team)
            
            # Determinar el oponente correcto
            if current_team_abbr == home_team_abbr:
                opponent_team = away_team_abbr
                opponent_team_id = game_data.get('awayTeam', {}).get('teamId', '')
            elif current_team_abbr == away_team_abbr:
                opponent_team = home_team_abbr
                opponent_team_id = game_data.get('homeTeam', {}).get('teamId', '')
            else:
                # Fallback a datos históricos si no coincide
                opponent_team = player_data.get('Opp', 'Unknown')
                opponent_team_id = ''
            
            # Agregar información extraída al player_data
            player_data['is_home'] = is_home
            player_data['is_started'] = is_started
            player_data['current_team'] = current_team
            player_data['opponent_team'] = opponent_team  # Oponente correcto del juego actual
            player_data['opponent_team_id'] = opponent_team_id  # ID del oponente
            
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
                logger.info(f"⚠️  Predicción no realizada, es menor a 4 rebotes")
                return None
            
            if 'error' in prediction_result:
                logger.error(f"❌ Error en predicción interna: {prediction_result['error']}")
                return None
            
            # Extraer predicción y confianza
            raw_prediction = prediction_result['raw_prediction']
            final_prediction = prediction_result['trb_prediction']
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
                "bet_type": "TRB",
                "confidence_percentage": round(confidence_percentage, 1),
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    def predict_single_player(self, player_data: Dict[str, Any], game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir rebotes totales para un jugador individual
        
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
            
            # Usar búsqueda inteligente de CommonUtils (maneja acentos y variaciones)
            player_historical_df = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if player_historical_df.empty:
                logger.warning(f"⚠️ No se encontraron datos históricos para {player_name}")
                # Usar datos de jugadores similares o promedio
                player_historical = self.historical_players.head(100).copy()
            else:
                player_historical = player_historical_df.copy()
                logger.info(f"✅ Encontrados {len(player_historical)} registros históricos para {player_name}")
            
            # Intentar usar datos del equipo actual primero, si no hay suficientes usar historial completo
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
            
            combined_df = player_historical.copy()
            
            # Hacer predicción 
            predictions = self.model.predict(combined_df)
            
            last_row = combined_df.iloc[-1]
            nan_count = last_row.isna().sum()

            # Extraer la última predicción (corresponde al último juego histórico, el más reciente)
            if len(predictions) > 0:
                raw_prediction = predictions[-1]
                recent_predictions = predictions[-5:] if len(predictions) >= 5 else predictions
            else:
                raw_prediction = 0
                recent_predictions = []
            
            # Calcular estadísticas para confianza
            actual_stats_mean = player_historical['rebounds'].mean() if len(player_historical) > 0 else 0
            actual_stats_std = player_historical['rebounds'].std() if len(player_historical) > 0 else 0
            prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            
            # CALCULAR ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
            # Últimos 5 juegos
            last_5_games = player_historical.tail(5)['rebounds'] if len(player_historical) >= 5 else player_historical['rebounds']
            last_5_stats = {
                'mean': round(last_5_games.mean(), 1) if len(last_5_games) > 0 else 0,
                'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                'min': int(last_5_games.min()) if len(last_5_games) > 0 else 0,
                'max': int(last_5_games.max()) if len(last_5_games) > 0 else 0,
                'count': len(last_5_games)
            }
            
            # Últimos 10 juegos
            last_10_games = player_historical.tail(10)['rebounds'] if len(player_historical) >= 10 else player_historical['rebounds']
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
            recent_form = player_historical.tail(3)['rebounds'].mean() if len(player_historical) >= 3 else actual_stats_mean
            
            # CALCULAR ESTADÍSTICAS H2H DETALLADAS
            # Usar opponent_team del juego actual si está disponible, sino usar histórico
            opponent_for_h2h = player_data.get('opponent_team', player_data.get('Opp', 'Unknown'))
            h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                player_name=player_name,
                opponent_team=opponent_for_h2h,
                target_stat='rebounds',
                max_games=50
            )
            
            # APLICAR FACTOR H2H A LA PREDICCIÓN
            h2h_factor = h2h_stats.get('h2h_factor', 1.0)
            if h2h_factor != 1.0 and h2h_stats.get('games_found', 0) >= 3:
                raw_prediction_adjusted = raw_prediction * h2h_factor
                logger.info(f"🎯 Aplicando factor H2H {h2h_factor:.3f} a predicción TRB: {raw_prediction:.1f} -> {raw_prediction_adjusted:.1f}")
            else:
                raw_prediction_adjusted = raw_prediction
            
            # Calcular confianza usando PlayersConfidence
            confidence_percentage = self.confidence_calculator.calculate_player_confidence(
                raw_prediction=raw_prediction,
                stabilized_prediction=raw_prediction,  # Usar solo la predicción del modelo
                tolerance=self.tolerance,  # Tolerancia individual del predictor
                prediction_std=prediction_std,
                actual_stats_std=actual_stats_std,
                historical_games=len(player_historical),
                player_data=player_data,
                opponent_team=opponent_for_h2h,
                game_date=player_data.get('Date'),
                game_data=game_data,  # Datos en tiempo real
                target_stat='rebounds'  # Estadística objetivo: rebotes
            )
            
            # Aplicar tolerancia individual del predictor
            trb_prediction = max(0, raw_prediction_adjusted + self.tolerance)  # self.tolerance es -3
            
            # REGLA DE CASAS DE APUESTAS: No inferir predicciones menores a 4 TRB
            if trb_prediction < 4:
                logger.info(f"⚠️  Predicción {trb_prediction:.1f} TRB < 4, no se infiere (casas de apuestas manejan líneas ≥4)")
                return None
            
            return {
                'raw_prediction': raw_prediction,
                'trb_prediction': int(trb_prediction),
                'confidence_percentage': confidence_percentage,
                'prediction_details': {
                    'player_id': self.common_utils._get_player_id(self.common_utils._normalize_name(player_name), player_data.get('Team', 'Unknown')),
                    'player': player_name,
                    'team_id': self.common_utils._get_team_id(player_data.get('Team', 'Unknown')),
                    'team': player_data.get('Team', 'Unknown'),
                    'opponent_id': player_data.get('opponent_team_id', self.common_utils._get_team_id(player_data.get('Opp', 'Unknown'))),
                    'opponent': opponent_for_h2h,
                    'tolerance_applied': self.tolerance,
                    'historical_games_used': len(player_historical),
                    'raw_prediction': raw_prediction,
                    'h2h_adjusted_prediction': round(raw_prediction_adjusted, 1),
                    'final_prediction': int(trb_prediction),
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
                        'stabilized_prediction': round(raw_prediction_adjusted, 1),
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
                'trb_prediction': None
            }

def test_trb_predictor():
    """Función de prueba rápida del predictor TRB"""
    def convert_numpy_types(obj):
        """Convierte tipos NumPy a tipos nativos de Python para serialización JSON"""
        if isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    print("🧪 PROBANDO TRB PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = TRBPredictor()
    
    # Cargar datos y modelo
    print("📂 Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print("❌ Error cargando modelo")
        return False
    
    # Prueba con datos simulados de SportRadar
    print("\n🎯 Prueba con datos simulados de SportRadar:")
    
    # Simular datos de SportRadar
    mock_sportradar_game = {
        "gameId": "sr:match:12345",
        "scheduled": "2024-01-15T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Denver Nuggets",
            "alias": "DEN",
            "players": [
                {
                    "playerId": "sr:player:123",
                    "fullName": "Nikola Jokić",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "15",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:124",
                    "fullName": "Aaron Gordon",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "50",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:125",
                    "fullName": "Michael Porter Jr.",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "1",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Philadelphia 76ers", 
            "alias": "PHI",
            "players": [
                {
                    "playerId": "sr:player:456",
                    "fullName": "Joel Embiid",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "21",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:457",
                    "fullName": "Tyrese Maxey",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "0",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:458",
                    "fullName": "Tobias Harris",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "12",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "Ball Arena",
            "capacity": 19520
        }
    }
    
    # Probar predicción desde SportRadar
    print("   Prediciendo Nikola Jokić desde datos SportRadar:")
    sportradar_result = predictor.predict_game(
        mock_sportradar_game, 
        "Nikola Jokić"
    )
    
    if sportradar_result is not None and 'error' not in sportradar_result:
        print("   ✅ Resultado SportRadar (formato JSON exacto):")
        from app.utils.helpers import safe_json_dumps
        print(safe_json_dumps(sportradar_result, indent=4))
    elif sportradar_result is None:
        print("      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"   ❌ Error: {sportradar_result['error']}")
        if 'available_players' in sportradar_result:
            print(f"   Jugadores disponibles: {sportradar_result['available_players']}")
    
    # Probar búsqueda inteligente con mejores reboteadores
    print("\n🧠 Pruebas con mejores reboteadores:")
    
    # Caso 1: Joel Embiid (Elite rebounder PHI)
    print("   1. Elite rebounder visitante: 'Joel Embiid'")
    embiid_result = predictor.predict_game(mock_sportradar_game, "Joel Embiid")
    if embiid_result is not None and 'error' not in embiid_result:
        print(f"      ✅ Encontrado: {embiid_result['target_name']} -> bet_line: {embiid_result['bet_line']}")
        print("      📊 Predicción detallada de Embiid:")
        from app.utils.helpers import safe_json_dumps
        print(safe_json_dumps(embiid_result, indent=4))
    elif embiid_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {embiid_result.get('error', 'Error desconocido')}")
    
    # Caso 2: Aaron Gordon (Solid rebounder DEN)
    print("   2. Solid rebounder local: 'Aaron Gordon'")
    gordon_result = predictor.predict_game(mock_sportradar_game, "Aaron Gordon")
    if gordon_result is not None and 'error' not in gordon_result:
        print(f"      ✅ Encontrado: {gordon_result['target_name']} -> bet_line: {gordon_result['bet_line']}")
        print("      📊 Predicción detallada de Gordon:")
        print(safe_json_dumps(gordon_result, indent=4))
    elif gordon_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {gordon_result.get('error', 'Error desconocido')}")
    
    # Caso 3: Búsqueda case-insensitive
    print("   3. Búsqueda case-insensitive: 'nikola jokic'")
    jokic_lower_result = predictor.predict_game(mock_sportradar_game, "nikola jokic")
    if jokic_lower_result is not None and 'error' not in jokic_lower_result:
        print(f"      ✅ Encontrado: {jokic_lower_result['target_name']} -> bet_line: {jokic_lower_result['bet_line']}")
    elif jokic_lower_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {jokic_lower_result.get('error', 'Error desconocido')}")
    
    # Caso 4: Búsqueda parcial (solo apellido)
    print("   4. Búsqueda parcial: 'Embiid'")
    embiid_partial_result = predictor.predict_game(mock_sportradar_game, "Embiid")
    if embiid_partial_result is not None and 'error' not in embiid_partial_result:
        print(f"      ✅ Encontrado: {embiid_partial_result['target_name']} -> bet_line: {embiid_partial_result['bet_line']}")
    elif embiid_partial_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {embiid_partial_result.get('error', 'Error desconocido')}")
    
    # Caso 5: Michael Porter Jr. (Forward con rebotes)
    print("   5. Forward reboteador: 'Michael Porter Jr.'")
    porter_result = predictor.predict_game(mock_sportradar_game, "Michael Porter Jr.")
    if porter_result is not None and 'error' not in porter_result:
        print(f"      ✅ Encontrado: {porter_result['target_name']} -> bet_line: {porter_result['bet_line']}")
        print("      📊 Predicción detallada de Porter Jr.:")
        print(safe_json_dumps(porter_result, indent=4))
    elif porter_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {porter_result['error']}")
    
    # Caso 6: Tyrese Maxey (Guard - pocos rebotes)
    print("   6. Guard con pocos rebotes: 'Tyrese Maxey'")
    maxey_result = predictor.predict_game(mock_sportradar_game, "Tyrese Maxey")
    if maxey_result is None:
        print(f"      ✅ Comportamiento esperado: No se hizo predicción (< 4 rebotes)")
    elif maxey_result is not None and 'error' not in maxey_result:
        print(f"      ⚠️ Predicción hecha: {maxey_result['target_name']} -> bet_line: {maxey_result['bet_line']}")
    else:
        print(f"      ❌ Error: {maxey_result.get('error', 'Error desconocido')}")
    
    # Caso 7: Jugador no existente (para probar sugerencias)
    print("   7. Jugador inexistente: 'Shaquille O'Neal'")
    shaq_result = predictor.predict_game(mock_sportradar_game, "Shaquille O'Neal")
    if shaq_result is not None and 'error' in shaq_result:
        print(f"      ❌ Error esperado: {shaq_result['error']}")
        if 'similar_suggestions' in shaq_result:
            print(f"      💡 Sugerencias: {shaq_result['similar_suggestions']}")
    elif shaq_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ⚠️ Inesperado: {shaq_result}")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_trb_predictor()
