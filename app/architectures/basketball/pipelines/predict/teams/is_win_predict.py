#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN IS_WIN (CLASIFICACIÓN)
========================================

Wrapper final unificado para predicciones de victoria/derrota que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo de clasificación is_win completo
- Formato estándar para módulo de stacking
- Predicción de equipo ganador con confianza

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Convertir con GameDataAdapter
3. Buscar datos históricos específicos de ambos equipos
4. Generar features dinámicas para clasificación
5. Aplicar modelo de clasificación
6. Retornar formato estándar con equipo ganador predicho
"""

import sys
import os
import pandas as pd
import numpy as np
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
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence.confidence_teams import TeamsConfidence
from app.architectures.basketball.pipelines.predict.teams.teams_points_predict import TeamsPointsPredictor

logger = logging.getLogger(__name__)

class IsWinPredictor:
    """
    Wrapper final unificado para predicciones de victoria/derrota (is_win)
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking de clasificación.
    """
    
    def __init__(self, teams_df: pd.DataFrame = None):
        """Inicializar el predictor is_win unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = teams_df
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = TeamsConfidence()  # Calculadora de confianza centralizada
        self.teams_points_predictor = TeamsPointsPredictor()  # Predictor de puntos de equipos
        self.is_loaded = False
        self.min_confidence_threshold = 50.0  # Umbral mínimo de confianza
        self.high_confidence_threshold = 75.0  # Umbral para alta confianza
        self.ultra_confidence_threshold = 90.0  # Umbral para ultra confianza

        # Cargar datos y modelo automáticamente
        self.load_data_and_model()
    
    def load_data_and_model(self) -> bool:
        """
        Cargar datos históricos y modelo entrenado para is_win
        
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
            
            # Inicializar confidence_calculator con datos históricos
            self.confidence_calculator = TeamsConfidence()
            self.confidence_calculator.historical_teams = self.historical_teams
            self.confidence_calculator.historical_players = self.historical_players
            logger.info(" Confidence calculator inicializado con datos históricos")
            
            # Cargar predictor de team points (nueva lógica)
            logger.info(" Cargando predictor de team points...")
            if not self.teams_points_predictor.load_data_and_model():
                logger.error(" Error cargando predictor de team points")
                return False
            logger.info(" Predictor de team points cargado exitosamente")
            
            # Ya no necesitamos el modelo is_win directamente
            # pero mantenemos la compatibilidad
            self.model = "using_teams_points_model"  # Placeholder para compatibilidad
            
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f" Error cargando datos y modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método principal para predecir el equipo ganador desde datos de SportRadar
        
        Args:
            game_data: Datos del juego de SportRadar
            
        Returns:
            Predicción en formato estándar de salida
        """
        if not self.is_loaded:
            return {'error': 'Modelo no cargado. Ejecutar load_data_and_model() primero.'}
        
        try:
            # 1. Convertir datos de SportRadar con GameDataAdapter
            players_df, teams_df = self.game_adapter.convert_game_to_dataframes(game_data)
            
            # 2. Obtener información de equipos desde game_data
            home_team_name = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team_name = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            # Convertir nombres completos a abreviaciones para búsqueda en dataset
            home_team = self.common_utils._get_team_abbreviation(home_team_name)
            away_team = self.common_utils._get_team_abbreviation(away_team_name)
            
            # 3. Hacer predicción de victoria/derrota
            prediction_result = self.predict_match_winner(game_data)
            
            # Si no hay predicción (probabilidad < 68%), devolver None
            if prediction_result is None:
                return None
            
            if 'error' in prediction_result:
                return prediction_result
            
            # 4. Extraer información de la predicción (sin fallbacks)
            predicted_winner = prediction_result['bet_line']
            predicted_winner_name = prediction_result['target_name']
            confidence_value = prediction_result['confidence_percentage']
            
            # prediction_details puede ser vacío, pero debe existir como dict
            prediction_details = prediction_result.get('prediction_details')
            if prediction_details is None:
                prediction_details = {}
            
            # prediction_type puede usar un default seguro (es metadata, no dato crítico)
            prediction_type = prediction_result.get('prediction_type')
            if prediction_type is None:
                prediction_type = 'is_win'
            
            return {
                "home_team": home_team,
                "away_team": away_team,
                "target_type": "team",
                "target_name": predicted_winner_name,
                "bet_line": predicted_winner,
                "bet_type": "winner",
                "prediction_type": prediction_type,
                "confidence_percentage": confidence_value,
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f" Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    def predict_match_winner(self, game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir el equipo ganador del partido usando predicciones de team points
        
        NUEVA LÓGICA: En lugar de usar el modelo is_win directamente, 
        usamos las predicciones de puntos de ambos equipos y determinamos
        el ganador basándose en quién tiene más puntos predichos.
        
        Args:
            game_data: Datos del juego de SportRadar (opcional)
                
        Returns:
            Diccionario con predicción del equipo ganador y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            # Obtener información de equipos desde game_data
            home_team_name = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team_name = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            # Convertir nombres completos a abreviaciones para búsqueda en dataset
            home_team = self.common_utils._get_team_abbreviation(home_team_name)
            away_team = self.common_utils._get_team_abbreviation(away_team_name)
            
            logger.info(f" Prediciendo ganador usando team points: {home_team_name} ({home_team}) vs {away_team_name} ({away_team})")
            
            # NUEVA LÓGICA: Usar predictor de team points para ambos equipos
            logger.info(" Obteniendo predicciones de puntos para ambos equipos...")
            
            # Predecir puntos del equipo local
            home_prediction = self.teams_points_predictor.predict_game(game_data, home_team_name)
            if 'error' in home_prediction:
                logger.error(f"Error prediciendo puntos de {home_team_name}: {home_prediction['error']}")
                return {'error': f"Error prediciendo puntos de {home_team_name}"}
            
            # Predecir puntos del equipo visitante
            away_prediction = self.teams_points_predictor.predict_game(game_data, away_team_name)
            if 'error' in away_prediction:
                logger.error(f"Error prediciendo puntos de {away_team_name}: {away_prediction['error']}")
                return {'error': f"Error prediciendo puntos de {away_team_name}"}
            
            # Extraer puntos predichos (sin fallbacks - debe existir)
            if 'bet_line' not in home_prediction:
                logger.error(f"❌ Predicción de {home_team_name} no contiene 'bet_line'")
                return {'error': f"Predicción de {home_team_name} sin 'bet_line'"}
            if 'bet_line' not in away_prediction:
                logger.error(f"❌ Predicción de {away_team_name} no contiene 'bet_line'")
                return {'error': f"Predicción de {away_team_name} sin 'bet_line'"}
            
            home_points = float(home_prediction['bet_line'])
            away_points = float(away_prediction['bet_line'])
            
            # Extraer confianzas (sin fallbacks - debe existir)
            if 'confidence_percentage' not in home_prediction:
                logger.error(f"❌ Predicción de {home_team_name} no contiene 'confidence_percentage'")
                return {'error': f"Predicción de {home_team_name} sin 'confidence_percentage'"}
            if 'confidence_percentage' not in away_prediction:
                logger.error(f"❌ Predicción de {away_team_name} no contiene 'confidence_percentage'")
                return {'error': f"Predicción de {away_team_name} sin 'confidence_percentage'"}
            
            home_confidence = float(home_prediction['confidence_percentage'])
            away_confidence = float(away_prediction['confidence_percentage'])
            
            logger.info(f"📈 Puntos predichos - {home_team_name}: {home_points}, {away_team_name}: {away_points}")
            logger.info(f" Confianzas - {home_team_name}: {home_confidence}%, {away_team_name}: {away_confidence}%")
            
            # Determinar ganador basándose en puntos predichos
            if home_points > away_points:
                predicted_winner_name = home_team_name
                predicted_winner = self.common_utils._get_team_abbreviation(home_team_name)
                point_difference = home_points - away_points
                winner_confidence = home_confidence
                logger.info(f"🏠 GANADOR PREDICHO: {home_team_name} (+{point_difference:.1f} puntos)")
            elif away_points > home_points:
                predicted_winner_name = away_team_name
                predicted_winner = self.common_utils._get_team_abbreviation(away_team_name)
                point_difference = away_points - home_points
                winner_confidence = away_confidence
                logger.info(f"✈️ GANADOR PREDICHO: {away_team_name} (+{point_difference:.1f} puntos)")
            else:
                # Empate - usar confianza más alta
                if home_confidence >= away_confidence:
                    predicted_winner_name = home_team_name
                    predicted_winner = self.common_utils._get_team_abbreviation(home_team_name)
                    winner_confidence = home_confidence
                    logger.info(f"🏠 GANADOR PREDICHO (empate, mayor confianza): {home_team_name}")
                else:
                    predicted_winner_name = away_team_name
                    predicted_winner = self.common_utils._get_team_abbreviation(away_team_name)
                    winner_confidence = away_confidence
                    logger.info(f"✈️ GANADOR PREDICHO (empate, mayor confianza): {away_team_name}")
            
            # Calcular probabilidad de victoria basada en diferencia de puntos
            point_difference = abs(home_points - away_points)
            if point_difference > 20:
                win_probability = 0.85  # Diferencia grande = alta probabilidad
            elif point_difference > 10:
                win_probability = 0.75  # Diferencia media = probabilidad media-alta
            elif point_difference > 5:
                win_probability = 0.65  # Diferencia pequeña = probabilidad media
            else:
                win_probability = 0.60  # Diferencia muy pequeña = probabilidad mínima aceptable
            
            # Ajustar probabilidad basada en confianza promedio
            avg_confidence = (home_confidence + away_confidence) / 2
            if avg_confidence > 80:
                win_probability = min(win_probability + 0.1, 0.95)
            elif avg_confidence < 60:
                win_probability = max(win_probability - 0.05, 0.55)  # Reducir menos el ajuste
            
            # SIEMPRE emitir predicción - el equipo con más puntos predichos es el ganador
            logger.info(f" Probabilidad de victoria: {win_probability:.1%} - Emitiendo predicción")
            
            # Calcular confianza final (promedio ponderado)
            final_confidence = (winner_confidence + (win_probability * 100)) / 2
            
            logger.info(f" Predicción final: {predicted_winner_name} gana con {final_confidence:.1f}% confianza")
                
            # CALCULAR ESTADÍSTICAS DETALLADAS DE VICTORIAS PARA PREDICTION_DETAILS
            # Estadísticas de victorias del equipo local
            home_team_historical = self.common_utils._smart_team_search(self.historical_teams, home_team)
            home_wins_last_5 = home_team_historical.tail(5)['is_win'].sum() if len(home_team_historical) >= 5 else home_team_historical['is_win'].sum()
            home_wins_last_10 = home_team_historical.tail(10)['is_win'].sum() if len(home_team_historical) >= 10 else home_team_historical['is_win'].sum()
            home_total_wins = home_team_historical['is_win'].sum()
            home_total_games = len(home_team_historical)
            home_win_rate = (home_total_wins / home_total_games) * 100 if home_total_games > 0 else 0
            
            # Estadísticas estructuradas para últimos 5 y 10 juegos (HOME)
            home_last_5 = home_team_historical.tail(5) if len(home_team_historical) >= 5 else home_team_historical
            home_last_10 = home_team_historical.tail(10) if len(home_team_historical) >= 10 else home_team_historical
            
            home_last_5_win_rate = (home_last_5['is_win'].sum() / len(home_last_5)) * 100 if len(home_last_5) > 0 else 0
            home_last_10_win_rate = (home_last_10['is_win'].sum() / len(home_last_10)) * 100 if len(home_last_10) > 0 else 0
            
            # Estadísticas de victorias del equipo visitante
            away_team_historical = self.common_utils._smart_team_search(self.historical_teams, away_team)
            away_wins_last_5 = away_team_historical.tail(5)['is_win'].sum() if len(away_team_historical) >= 5 else away_team_historical['is_win'].sum()
            away_wins_last_10 = away_team_historical.tail(10)['is_win'].sum() if len(away_team_historical) >= 10 else away_team_historical['is_win'].sum()
            away_total_wins = away_team_historical['is_win'].sum()
            away_total_games = len(away_team_historical)
            away_win_rate = (away_total_wins / away_total_games) * 100 if away_total_games > 0 else 0
            
            # Estadísticas estructuradas para últimos 5 y 10 juegos (AWAY)
            away_last_5 = away_team_historical.tail(5) if len(away_team_historical) >= 5 else away_team_historical
            away_last_10 = away_team_historical.tail(10) if len(away_team_historical) >= 10 else away_team_historical
            
            away_last_5_win_rate = (away_last_5['is_win'].sum() / len(away_last_5)) * 100 if len(away_last_5) > 0 else 0
            away_last_10_win_rate = (away_last_10['is_win'].sum() / len(away_last_10)) * 100 if len(away_last_10) > 0 else 0
            
            # Estadísticas H2H (enfrentamientos directos) usando abreviaciones
            h2h_games = self.historical_teams[
                ((self.historical_teams['Team'] == home_team) & (self.historical_teams['Opp'] == away_team)) |
                ((self.historical_teams['Team'] == away_team) & (self.historical_teams['Opp'] == home_team))
            ].copy()
            
            h2h_games_count = len(h2h_games)
            home_h2h_wins = len(h2h_games[(h2h_games['Team'] == home_team) & (h2h_games['is_win'] == 1)])
            away_h2h_wins = len(h2h_games[(h2h_games['Team'] == away_team) & (h2h_games['is_win'] == 1)])
                
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "team",
                "target_name": predicted_winner_name,
                "bet_line": predicted_winner,
                "bet_type": "winner",
                "confidence_percentage": round(final_confidence, 0),
                "prediction_details": {
                    "home_team_id": game_data.get('homeTeam', {}).get('teamId', self.common_utils._get_team_id(home_team_name)) if game_data else self.common_utils._get_team_id(home_team_name),
                    "away_team_id": game_data.get('awayTeam', {}).get('teamId', self.common_utils._get_team_id(away_team_name)) if game_data else self.common_utils._get_team_id(away_team_name),
                    "home_points_predicted": home_points,
                    "away_points_predicted": away_points,
                    "point_difference": abs(home_points - away_points),
                    "win_probability": win_probability,
                    "home_team_stats": {
                        "total_games": home_total_games,
                        "total_wins": home_total_wins,
                        "win_rate": round(home_win_rate, 1),
                        "wins_last_5": home_wins_last_5,
                        "wins_last_10": home_wins_last_10,
                        "win_rate_last_5": round((home_wins_last_5 / 5) * 100, 1) if home_wins_last_5 >= 0 else 0,
                        "win_rate_last_10": round((home_wins_last_10 / 10) * 100, 1) if home_wins_last_10 >= 0 else 0,
                        "last_5_games": {
                            "count": len(home_last_5),
                            "win_rate": round(home_last_5_win_rate, 1),
                            "wins": int(home_last_5['is_win'].sum()),
                            "losses": int(len(home_last_5) - home_last_5['is_win'].sum())
                        },
                        "last_10_games": {
                            "count": len(home_last_10),
                            "win_rate": round(home_last_10_win_rate, 1),
                            "wins": int(home_last_10['is_win'].sum()),
                            "losses": int(len(home_last_10) - home_last_10['is_win'].sum())
                        }
                    },
                    "away_team_stats": {
                        "total_games": away_total_games,
                        "total_wins": away_total_wins,
                        "win_rate": round(away_win_rate, 1),
                        "wins_last_5": away_wins_last_5,
                        "wins_last_10": away_wins_last_10,
                        "win_rate_last_5": round((away_wins_last_5 / 5) * 100, 1) if away_wins_last_5 >= 0 else 0,
                        "win_rate_last_10": round((away_wins_last_10 / 10) * 100, 1) if away_wins_last_10 >= 0 else 0,
                        "last_5_games": {
                            "count": len(away_last_5),
                            "win_rate": round(away_last_5_win_rate, 1),
                            "wins": int(away_last_5['is_win'].sum()),
                            "losses": int(len(away_last_5) - away_last_5['is_win'].sum())
                        },
                        "last_10_games": {
                            "count": len(away_last_10),
                            "win_rate": round(away_last_10_win_rate, 1),
                            "wins": int(away_last_10['is_win'].sum()),
                            "losses": int(len(away_last_10) - away_last_10['is_win'].sum())
                        }
                    },
                    "h2h_stats": {
                        "games_found": h2h_games_count,
                        "home_h2h_wins": home_h2h_wins,
                        "away_h2h_wins": away_h2h_wins,
                        "home_h2h_win_rate": round((home_h2h_wins / h2h_games_count) * 100, 1) if h2h_games_count > 0 else 0,
                        "away_h2h_win_rate": round((away_h2h_wins / h2h_games_count) * 100, 1) if h2h_games_count > 0 else 0
                    }
                }
            }
                
        except Exception as e:
                logger.error(f" Error en predicción: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'error': str(e),
                    'predicted_winner': 'Unknown',
                    'predicted_winner_name': 'Unknown'
                }
    
    
def test_is_win_predictor():
    """Función de prueba rápida del predictor de is_win (nueva lógica basada en team points)"""
    print("="*80)
    print(" PROBANDO IS_WIN PREDICTOR - KNICKS VS CAVALIERS")
    print("="*80)
    
    # Inicializar predictor
    predictor = IsWinPredictor()
    
    # Cargar datos y modelo
    print("\n Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print(" Error cargando modelo")
        return False
    
    print("\n[OK] Modelo cargado exitosamente")
    
    # Prueba con datos simulados de SportRadar
    print("\n" + "="*80)
    print(" PRUEBA: KNICKS VS CAVALIERS - PREDICCION DE GANADOR")
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
    
    # Probar predicción desde SportRadar
    print("\nPrediciendo ganador del partido:")
    print("-" * 60)
    print("Metodo: Comparacion de puntos predichos por equipo")
    sportradar_result = predictor.predict_game(mock_sportradar_game)
    
    import json
    
    if sportradar_result is not None and 'error' not in sportradar_result:
        print(f"\n[OK] PREDICCION EXITOSA")
        
        # Información principal
        predicted_winner = sportradar_result['target_name']
        confidence = sportradar_result['confidence_percentage']
        
        print(f"\nGANADOR PREDICHO: {predicted_winner}")
        print(f"Confidence: {confidence}%")
        print(f"Bet Type: {sportradar_result['bet_type']}")
        
        # Detalles de la predicción
        if 'prediction_details' in sportradar_result:
            details = sportradar_result['prediction_details']
            
            print(f"\nDETALLES DE LA PREDICCION:")
            print(f"  Puntos predichos local: {details.get('home_points_predicted', 'N/A')}")
            print(f"  Puntos predichos visitante: {details.get('away_points_predicted', 'N/A')}")
            print(f"  Diferencia de puntos: {details.get('point_difference', 'N/A')}")
            print(f"  Probabilidad de victoria: {details.get('win_probability', 'N/A'):.1%}")
            
            # Estadísticas de equipos
            if 'home_team_stats' in details:
                home_stats = details['home_team_stats']
                print(f"\nESTADISTICAS EQUIPO LOCAL ({sportradar_result['home_team']}):")
                print(f"  Win Rate Total: {home_stats.get('win_rate', 'N/A')}% ({home_stats.get('total_wins', 'N/A')}/{home_stats.get('total_games', 'N/A')} juegos)")
                print(f"  Ultimos 5 juegos: {home_stats.get('wins_last_5', 'N/A')}/5 victorias ({home_stats.get('win_rate_last_5', 'N/A')}%)")
                print(f"  Ultimos 10 juegos: {home_stats.get('wins_last_10', 'N/A')}/10 victorias ({home_stats.get('win_rate_last_10', 'N/A')}%)")
            
            if 'away_team_stats' in details:
                away_stats = details['away_team_stats']
                print(f"\nESTADISTICAS EQUIPO VISITANTE ({sportradar_result['away_team']}):")
                print(f"  Win Rate Total: {away_stats.get('win_rate', 'N/A')}% ({away_stats.get('total_wins', 'N/A')}/{away_stats.get('total_games', 'N/A')} juegos)")
                print(f"  Ultimos 5 juegos: {away_stats.get('wins_last_5', 'N/A')}/5 victorias ({away_stats.get('win_rate_last_5', 'N/A')}%)")
                print(f"  Ultimos 10 juegos: {away_stats.get('wins_last_10', 'N/A')}/10 victorias ({away_stats.get('win_rate_last_10', 'N/A')}%)")
            
            # H2H
            if 'h2h_stats' in details:
                h2h = details['h2h_stats']
                print(f"\nESTADISTICAS H2H:")
                print(f"  Juegos encontrados: {h2h.get('games_found', 'N/A')}")
                print(f"  Victorias local: {h2h.get('home_h2h_wins', 'N/A')} ({h2h.get('home_h2h_win_rate', 'N/A')}%)")
                print(f"  Victorias visitante: {h2h.get('away_h2h_wins', 'N/A')} ({h2h.get('away_h2h_win_rate', 'N/A')}%)")
        
        print(f"\nJSON COMPLETO:")
        # Convertir numpy types a Python types para JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        print(json.dumps(sportradar_result, indent=2, ensure_ascii=False, default=convert_numpy))
        
    elif sportradar_result is None:
        print("\n[INFO] No se emitio prediccion (probabilidad muy baja)")
    else:
        print(f"\n[ERROR] Error en prediccion")
        if 'error' in sportradar_result:
            print(f"Error: {sportradar_result['error']}")
        else:
            print(f"Resultado inesperado: {sportradar_result}")
    
    print("\n" + "="*80)
    print(" PRUEBA COMPLETADA")
    print("="*80)
    return True


if __name__ == "__main__":
    test_is_win_predictor()