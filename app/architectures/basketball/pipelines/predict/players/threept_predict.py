#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN 3PT
========================================

Wrapper final unificado para predicciones de triples (3PT) que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo 3PT completo con calibraciones elite
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
from app.architectures.basketball.src.models.players.triples.model_triples import XGBoost3PTModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence.confidence_players import PlayersConfidence

logger = logging.getLogger(__name__)

class ThreePointsPredictor:
    """
    Wrapper final unificado para predicciones 3PT
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self):
        """Inicializar el predictor 3PT unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = PlayersConfidence()
        self.is_loaded = False
        self.tolerance = 0 # Tolerancia conservadora individual

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
            
            # Inicializar modelo 3PT completo (wrapper)
            logger.info("🤖 Inicializando modelo 3PT completo (wrapper)...")
            self.model = XGBoost3PTModel(teams_df=self.historical_teams)
            
            # Cargar modelo entrenado
            model_path = "app/architectures/basketball/.joblib/3pt_model.joblib"
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
        Metodo principal para predecir 3PT desde datos de insumo (SportRadar)
        
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
                logger.info(f"⚠️  Predicción no realizada, es 0 triples")
                return None
            
            if 'error' in prediction_result:
                logger.error(f"❌ Error en predicción interna: {prediction_result['error']}")
                return None
            
            # Extraer predicción y confianza
            raw_prediction = prediction_result['raw_prediction']
            final_prediction = prediction_result['3pt_prediction']
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
                "bet_type": "3PT",
                "confidence_percentage": round(confidence_percentage, 1),
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    def predict_single_player(self, player_data: Dict[str, Any], game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir triples para un jugador individual
        
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
            
            combined_df = player_historical.copy()
            
            # Las features ya vienen en el orden correcto desde el feature engineer
            
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
            actual_stats_mean = player_historical['three_points_made'].mean() if len(player_historical) > 0 else 0
            actual_stats_std = player_historical['three_points_made'].std() if len(player_historical) > 0 else 0
            prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            
            # CALCULAR ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
            # Últimos 5 juegos (más recientes)
            last_5_games = player_historical.head(5)['three_points_made'] if len(player_historical) >= 5 else player_historical['three_points_made']
            last_5_stats = {
                'mean': round(last_5_games.mean(), 1) if len(last_5_games) > 0 else 0,
                'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                'min': int(last_5_games.min()) if len(last_5_games) > 0 else 0,
                'max': int(last_5_games.max()) if len(last_5_games) > 0 else 0,
                'count': len(last_5_games)
            }
            
            # Últimos 10 juegos (más recientes)
            last_10_games = player_historical.head(10)['three_points_made'] if len(player_historical) >= 10 else player_historical['three_points_made']
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
            recent_form = player_historical.tail(3)['three_points_made'].mean() if len(player_historical) >= 3 else actual_stats_mean
            
            # CALCULAR ESTADÍSTICAS H2H DETALLADAS
            # Usar opponent_team del juego actual si está disponible, sino usar histórico
            opponent_for_h2h = player_data.get('opponent_team', player_data.get('Opp', 'Unknown'))
            h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                player_name=player_name,
                opponent_team=opponent_for_h2h,
                target_stat='three_points_made',
                max_games=50
            )
            
            # APLICAR FACTOR H2H A LA PREDICCIÓN
            h2h_factor = h2h_stats.get('h2h_factor', 1.0)
            if h2h_factor != 1.0 and h2h_stats.get('games_found', 0) >= 3:
                raw_prediction_adjusted = raw_prediction * h2h_factor
                logger.info(f"🎯 Aplicando factor H2H {h2h_factor:.3f} a predicción 3PT: {raw_prediction:.1f} -> {raw_prediction_adjusted:.1f}")
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
                target_stat='triples'  # Estadística objetivo: triples
            )
            
            # Aplicar tolerancia individual del predictor
            threept_prediction = max(0, raw_prediction_adjusted + self.tolerance)  # self.tolerance es -1
            
            # REGLA DE CASAS DE APUESTAS: No inferir predicciones de 0 triples
            if threept_prediction <= 0:
                logger.info(f"⚠️  Predicción {threept_prediction:.1f} 3PT ≤ 0, no se infiere (casas de apuestas manejan líneas ≥1)")
                return None
                
            return {
                'raw_prediction': raw_prediction,
                '3pt_prediction': int(threept_prediction),
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
                    'final_prediction': int(threept_prediction),
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
                    'threept_prediction': None
                }
    
   
def test_3pt_predictor():
    """Función de prueba rápida del predictor 3PT"""
    def convert_numpy_types(obj):
        """Convierte tipos NumPy a tipos nativos de Python para serialización JSON."""
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
    
    print("🧪 PROBANDO 3PT PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = ThreePointsPredictor()
    
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
            "name": "Golden State Warriors",
            "alias": "GSW",
            "players": [
                {
                    "playerId": "sr:player:123",
                    "fullName": "Stephen Curry",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "30",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:124",
                    "fullName": "Klay Thompson",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "11",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:125",
                    "fullName": "Draymond Green",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "23",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Oklahoma City Thunder", 
            "alias": "OKC",
            "players": [
                {
                    "playerId": "sr:player:456",
                    "fullName": "Shai Gilgeous-Alexander",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "2",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:789",
                    "fullName": "Chet Holmgren",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "7",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:888",
                    "fullName": "Jalen Williams",
                    "position": "F",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "8",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "Chase Center",
            "capacity": 18064
        }
    }
    
    # Probar predicción desde SportRadar
    print("   Prediciendo Stephen Curry desde datos SportRadar:")
    sportradar_result = predictor.predict_game(
        mock_sportradar_game, 
        "Stephen Curry"
    )
    
    if sportradar_result is not None and 'error' not in sportradar_result:
        print("   ✅ Resultado SportRadar (formato JSON exacto):")
        print(json.dumps(convert_numpy_types(sportradar_result), indent=4, ensure_ascii=False))
    elif sportradar_result is None:
        print("      ⚠️ No se hizo predicción (jugador no disponible)")
    else:
        print(f"   ❌ Error: {sportradar_result['error']}")
        if 'available_players' in sportradar_result:
            print(f"   Jugadores disponibles: {sportradar_result['available_players']}")
    
    # Probar búsqueda inteligente con nombres problemáticos
    print("\n🧠 Pruebas de búsqueda inteligente:")
    
    # Caso 1: Búsqueda case-insensitive
    print("   1. Búsqueda case-insensitive: 'stephen curry'")
    curry_lower_result = predictor.predict_game(mock_sportradar_game, "stephen curry")
    if curry_lower_result is not None and 'error' not in curry_lower_result:
        print(f"      ✅ Encontrado: {curry_lower_result['target_name']} -> bet_line: {curry_lower_result['bet_line']}")
    elif curry_lower_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {curry_lower_result.get('error', 'Error desconocido')}")
    
    # Caso 2: Búsqueda parcial
    print("   2. Búsqueda parcial: 'Curry'")
    curry_partial_result = predictor.predict_game(mock_sportradar_game, "Curry")
    if curry_partial_result is not None and 'error' not in curry_partial_result:
        print(f"      ✅ Encontrado: {curry_partial_result['target_name']} -> bet_line: {curry_partial_result['bet_line']}")
    elif curry_partial_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {curry_partial_result.get('error', 'Error desconocido')}")
    
    # Caso 3: Segundo jugador del equipo - Klay Thompson
    print("   3. Segundo jugador GSW: 'Klay Thompson'")
    klay_result = predictor.predict_game(mock_sportradar_game, "Klay Thompson")
    if klay_result is not None and 'error' not in klay_result:
        print(f"      ✅ Encontrado: {klay_result['target_name']} -> bet_line: {klay_result['bet_line']}")
        print("      📊 Predicción detallada de Klay:")
        print(json.dumps(convert_numpy_types(klay_result), indent=4, ensure_ascii=False))
    elif klay_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {klay_result.get('error', 'Error desconocido')}")
    
    # Caso 4: Jugador estrella visitante - Shai Gilgeous-Alexander
    print("   4. Jugador estrella visitante: 'Shai Gilgeous-Alexander'")
    shai_result = predictor.predict_game(mock_sportradar_game, "Shai Gilgeous-Alexander")
    if shai_result is not None and 'error' not in shai_result:
        print(f"      ✅ Encontrado: {shai_result['target_name']} -> bet_line: {shai_result['bet_line']}")
        print("      📊 Predicción detallada de Shai:")
        print(json.dumps(convert_numpy_types(shai_result), indent=4, ensure_ascii=False))
    elif shai_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {shai_result.get('error', 'Error desconocido')}")
    
    # Caso 5: Segundo jugador OKC - Chet Holmgren
    print("   5. Jugador OKC: 'Chet Holmgren'")
    chet_result = predictor.predict_game(mock_sportradar_game, "Chet Holmgren")
    if chet_result is not None and 'error' not in chet_result:
        print(f"      ✅ Encontrado: {chet_result['target_name']} -> bet_line: {chet_result['bet_line']}")
    elif chet_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ❌ Error: {chet_result.get('error', 'Error desconocido')}")
    
    # Caso 6: Jugador no existente (para probar sugerencias)
    print("   6. Jugador inexistente: 'Michael Jordan'")
    jordan_result = predictor.predict_game(mock_sportradar_game, "Michael Jordan")
    if jordan_result is not None and 'error' in jordan_result:
        print(f"      ❌ Error esperado: {jordan_result['error']}")
        if 'similar_suggestions' in jordan_result:
            print(f"      💡 Sugerencias: {jordan_result['similar_suggestions']}")
    elif jordan_result is None:
        print(f"      ⚠️ Jugador no disponible para predicción")
    else:
        print(f"      ⚠️ Inesperado: {jordan_result}")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_3pt_predictor()