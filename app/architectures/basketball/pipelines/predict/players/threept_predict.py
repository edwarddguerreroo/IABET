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
            self.historical_players, self.historical_teams, self.historical_players_quarters, self.historical_teams_quarters = data_loader.load_data()
            
            # Inicializar modelo 3PT completo (wrapper)
            logger.info(" Inicializando modelo 3PT completo (wrapper)...")
            self.model = XGBoost3PTModel(teams_df=self.historical_teams)
            
            # Cargar modelo entrenado
            model_path = "app/architectures/basketball/.joblib/3pt_model.joblib"
            logger.info(f" Cargando modelo desde: {model_path}")
            self.model.load_model(model_path)
            
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f" Error cargando datos y modelo: {e}")
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
                logger.info(f" {target_player} no disponible en el roster")
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
                error_msg = f"No se pudo determinar el oponente para {target_player}: equipo actual {current_team_abbr} no coincide con home ({home_team_abbr}) ni away ({away_team_abbr})"
                logger.error(f" {error_msg}")
                return None
            
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
                logger.info(f"  Predicción no realizada, es 0 triples")
                return None
            
            if 'error' in prediction_result:
                logger.error(f" Error en predicción interna: {prediction_result['error']}")
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
            logger.error(f" Error en predicción desde SportRadar: {e}")
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
            current_team = player_data.get('current_team', 'Unknown')
            
            # Filtrar datos históricos del jugador específico usando búsqueda inteligente
            player_historical = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if len(player_historical) == 0:
                error_msg = f"No se encontraron datos históricos para {player_name}"
                logger.error(f" {error_msg}")
                return {'error': error_msg}
            
            logger.info(f" Encontrados {len(player_historical)} registros históricos para {player_name}")
            
            # Verificar mínimo de datos requerido
            if len(player_historical) < 5:
                error_msg = f"Datos insuficientes para {player_name}: solo {len(player_historical)} juegos (mínimo requerido: 5)"
                logger.error(f" {error_msg}")
                return {'error': error_msg}
            
            # Usar datos del equipo actual SIEMPRE que sea posible
            if current_team != 'Unknown':
                current_team_data = player_historical[player_historical['Team'] == current_team]
                if len(current_team_data) >= 5:  # Mínimo 5 juegos con el equipo actual
                    player_historical = current_team_data.copy()
                    logger.info(f" Usando {len(player_historical)} registros del equipo actual ({current_team}) para {player_name}")
                else:
                    error_msg = f"Datos insuficientes del equipo actual {current_team} para {player_name}: solo {len(current_team_data)} juegos (mínimo requerido: 5)"
                    logger.error(f" {error_msg}")
                    return {'error': error_msg}
            else:
                logger.info(f" Usando TODOS los {len(player_historical)} registros históricos para {player_name}")
            
            combined_df = player_historical.copy()
            
            # Las features ya vienen en el orden correcto desde el feature engineer
            
            # Hacer predicción
            predictions = self.model.predict(combined_df)
            
            if len(predictions) == 0:
                error_msg = f"Modelo no generó predicciones para {player_name}"
                logger.error(f" {error_msg}")
                return {'error': error_msg}

            last_row = combined_df.iloc[-1]
            nan_count = last_row.isna().sum()

            # Extraer la última predicción (corresponde al último juego histórico, el más reciente)
            raw_prediction = predictions[-1]
            recent_predictions = predictions[-5:] if len(predictions) >= 5 else predictions
            
            # CALCULAR MÉTRICAS DETALLADAS PARA CONFIANZA Y PREDICTION_DETAILS (SIN FALLBACKS)
            if 'three_points_made' not in player_historical.columns:
                error_msg = f"Columna 'three_points_made' no encontrada en datos históricos de {player_name}"
                logger.error(f" {error_msg}")
                return {'error': error_msg}
            
            historical_3pt = player_historical['three_points_made'].dropna()
            
            if len(historical_3pt) < 5:
                error_msg = f"Datos insuficientes de triples para {player_name}: solo {len(historical_3pt)} juegos válidos (mínimo requerido: 5)"
                logger.error(f" {error_msg}")
                return {'error': error_msg}
            
            # Calcular estadísticas para confianza
            actual_stats_mean = historical_3pt.mean()
            actual_stats_std = historical_3pt.std() if len(historical_3pt) > 1 else 1.0
            prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            
            # ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
            # Últimos 5 juegos (más recientes)
            last_5_games = historical_3pt.head(5)
            last_5_stats = {
                'mean': round(last_5_games.mean(), 1),
                'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                'min': int(last_5_games.min()),
                'max': int(last_5_games.max()),
                'count': len(last_5_games)
            }
            
            # Últimos 10 juegos (más recientes)
            if len(historical_3pt) >= 10:
                last_10_games = historical_3pt.head(10)
                last_10_stats = {
                    'mean': round(last_10_games.mean(), 1),
                    'std': round(last_10_games.std(), 1),
                    'min': int(last_10_games.min()),
                    'max': int(last_10_games.max()),
                    'count': len(last_10_games)
                }
                
                # Análisis de tendencia
                recent_5_mean = last_5_games.mean()
                recent_10_mean = last_10_games.mean()
                trend_5_games = recent_5_mean - recent_10_mean
            else:
                # Si hay menos de 10 juegos, usar todos los disponibles
                last_10_stats = {
                    'mean': round(actual_stats_mean, 1),
                    'std': round(actual_stats_std, 1),
                    'min': int(historical_3pt.min()),
                    'max': int(historical_3pt.max()),
                    'count': len(historical_3pt)
                }
                trend_5_games = 0
                recent_5_mean = actual_stats_mean
            
            # Score de consistencia (inverso de la desviación estándar)
            consistency_score = max(0, 100 - (actual_stats_std * 5)) if actual_stats_std > 0 else 100
            
            # Forma reciente (promedio de últimos 3 juegos)
            if len(historical_3pt) >= 3:
                recent_form = historical_3pt.tail(3).mean()
            else:
                recent_form = actual_stats_mean
            
            # CALCULAR ESTADÍSTICAS H2H DETALLADAS
            # Usar opponent_team del juego actual si está disponible, sino usar histórico
            opponent_for_h2h = player_data.get('opponent_team', player_data.get('Opp', 'Unknown'))
            h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                player_name=player_name,
                opponent_team=opponent_for_h2h,
                target_stat='three_points_made',
                max_games=50
            )
            
            # APLICAR FACTOR H2H A LA PREDICCIÓN (SOLO SI HAY DATOS SUFICIENTES)
            h2h_factor = h2h_stats.get('h2h_factor', None)
            h2h_games = h2h_stats.get('games_found', 0)
            
            if h2h_factor is not None and h2h_games >= 3:
                # Si hay suficientes datos H2H, aplicar el factor
                raw_prediction_adjusted = raw_prediction * h2h_factor
                logger.info(f" Aplicando factor H2H {h2h_factor:.3f} a predicción 3PT: {raw_prediction:.1f} -> {raw_prediction_adjusted:.1f} (basado en {h2h_games} juegos H2H)")
            else:
                # Si no hay suficientes datos H2H, NO AJUSTAR (sin fallback a 1.0)
                raw_prediction_adjusted = raw_prediction
                if h2h_games < 3:
                    logger.warning(f" No se aplica factor H2H: solo {h2h_games} juegos H2H encontrados (mínimo requerido: 3)")
            
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
                logger.info(f"  Predicción {threept_prediction:.1f} 3PT ≤ 0, no se infiere (casas de apuestas manejan líneas ≥1)")
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
                logger.error(f" Error en predicción: {e}")
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
    
    print("="*80)
    print(" PROBANDO 3PT PREDICTOR - KNICKS VS CAVALIERS")
    print("="*80)
    
    # Inicializar predictor
    predictor = ThreePointsPredictor()
    
    # Cargar datos y modelo
    print("\n Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print(" Error cargando modelo")
        return False
    
    print("\n[OK] Modelo cargado exitosamente")
    
    # Prueba con datos simulados de SportRadar
    print("\n" + "="*80)
    print(" PRUEBA: KNICKS VS CAVALIERS - TRIPLES")
    print("="*80)
    
    # Simular datos de SportRadar para Knicks vs Cavaliers
    mock_sportradar_game = {
        "gameId": "sr:match:knicks_cavs_20250124",
        "scheduled": "2025-01-24T19:30:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "New York Knicks",
            "alias": "NYK",
            "players": [
                {"playerId": "sr:player:brunson", "fullName": "Jalen Brunson", "position": "PG", "starter": True, "status": "ACT", "jerseyNumber": "11", "injuries": []},
                {"playerId": "sr:player:towns", "fullName": "Karl-Anthony Towns", "position": "C", "starter": True, "status": "ACT", "jerseyNumber": "32", "injuries": []},
                {"playerId": "sr:player:anunoby", "fullName": "OG Anunoby", "position": "SF", "starter": True, "status": "ACT", "jerseyNumber": "8", "injuries": []},
                {"playerId": "sr:player:hart", "fullName": "Josh Hart", "position": "SG", "starter": True, "status": "ACT", "jerseyNumber": "3", "injuries": []},
                {"playerId": "sr:player:robinson", "fullName": "Mitchell Robinson", "position": "C", "starter": True, "status": "ACT", "jerseyNumber": "23", "injuries": []}
            ]
        },
        "awayTeam": {
            "name": "Cleveland Cavaliers", 
            "alias": "CLE",
            "players": [
                {"playerId": "sr:player:mitchell", "fullName": "Donovan Mitchell", "position": "SG", "starter": True, "status": "ACT", "jerseyNumber": "45", "injuries": []},
                {"playerId": "sr:player:garland", "fullName": "Darius Garland", "position": "PG", "starter": True, "status": "ACT", "jerseyNumber": "10", "injuries": []},
                {"playerId": "sr:player:mobley", "fullName": "Evan Mobley", "position": "PF", "starter": True, "status": "ACT", "jerseyNumber": "4", "injuries": []},
                {"playerId": "sr:player:allen", "fullName": "Jarrett Allen", "position": "C", "starter": True, "status": "ACT", "jerseyNumber": "31", "injuries": []},
                {"playerId": "sr:player:strus", "fullName": "Max Strus", "position": "SF", "starter": True, "status": "ACT", "jerseyNumber": "1", "injuries": []}
            ]
        },
        "venue": {
            "name": "Madison Square Garden",
            "capacity": 19812
        }
    }
    
    # Probar predicción para todos los jugadores del juego
    print("\nPREDICCIONES PARA TODOS LOS JUGADORES:\n")
    
    # Lista de todos los jugadores
    all_players = []
    for team_key in ['homeTeam', 'awayTeam']:
        for player in mock_sportradar_game[team_key]['players']:
            all_players.append(player['fullName'])
    
    print(f"Total jugadores a predecir: {len(all_players)}\n")
    
    successful_predictions = 0
    failed_predictions = 0
    
    for idx, player_name in enumerate(all_players, 1):
        print(f"\n{idx}. {player_name}:")
        print("-" * 60)
        
        result = predictor.predict_game(mock_sportradar_game, player_name)
        
        if result is not None and 'error' not in result:
            successful_predictions += 1
            print(f"   [OK] PREDICCION EXITOSA")
            print(json.dumps(convert_numpy_types(result), indent=2, ensure_ascii=False))
        elif result is None:
            failed_predictions += 1
            print(f"   [X] No se hizo prediccion (< 1 triple o jugador no disponible)")
        else:
            failed_predictions += 1
            print(f"   [X] Error: {result.get('error', 'Error desconocido')}")
    
    # Resumen
    print("\n" + "="*80)
    print(" RESUMEN DE PREDICCIONES")
    print("="*80)
    print(f"Total jugadores: {len(all_players)}")
    print(f"[OK] Predicciones exitosas: {successful_predictions}")
    print(f"[X] Predicciones fallidas: {failed_predictions}")
    print(f"Tasa de exito: {(successful_predictions/len(all_players)*100):.1f}%")
    print("="*80)
    
    print("\nPrueba completada")
    return True


if __name__ == "__main__":
    test_3pt_predictor()