#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN TOTAL PUNTOS HALFTIME
========================================

Wrapper final unificado para predicciones de total puntos en halftime que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo halftime equipos completo con calibraciones elite
- Suma de predicciones de ambos equipos para total halftime
- Formato estándar para módulo de stacking
- Manejo robusto de errores y tolerancia conservadora

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Obtener predicciones de halftime de ambos equipos
3. Sumar predicciones para obtener total halftime
4. Aplicar tolerancia conservadora
5. Retornar formato estándar para stacking
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
from app.architectures.basketball.pipelines.predict.teams.ht_teams_points_predict import HalfTimeTeamsPointsPredictor

logger = logging.getLogger(__name__)

class HalfTimeTotalPointsPredictor:
    """
    Wrapper final unificado para predicciones total puntos halftime
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self, teams_df: pd.DataFrame = None):
        """Inicializar el predictor total puntos halftime unificado"""
        self.historical_players = None
        self.historical_teams = teams_df
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = TeamsConfidence()  # Calculadora de confianza centralizada
        self.is_loaded = False
        self.conservative_tolerance = 0  # Tolerancia conservadora para total halftime
        
        # Inicializar predictor de halftime equipos
        self.halftime_teams_predictor = HalfTimeTeamsPointsPredictor(teams_df)
        
        # Cargar datos y modelo automáticamente
        self.load_data_and_model()
    
    def load_data_and_model(self) -> bool:
        """
        Cargar datos históricos y modelo entrenado de halftime total points
        
        Returns:
            True si se cargó exitosamente
        """
        try:
            # Cargar datos históricos si no están disponibles
            if self.historical_teams is None:
                data_loader = NBADataLoader(
                    players_total_path="app/architectures/basketball/data/players_total.csv",
                    players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                    teams_total_path="app/architectures/basketball/data/teams_total.csv",
                    teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                    biometrics_path="app/architectures/basketball/data/biometrics.csv"
                )
                self.historical_players, self.historical_teams = data_loader.load_data_with_halftime_target()
                logger.info("✅ Datos históricos cargados para halftime total points")
            
            # El predictor de halftime equipos ya se inicializa automáticamente
            if self.halftime_teams_predictor.is_loaded:
                self.is_loaded = True
                logger.info("✅ Modelo halftime total points cargado exitosamente")
                return True
            else:
                logger.error("❌ Error: predictor de halftime equipos no cargado")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo halftime total points: {e}")
            return False
    
    def predict_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método principal para predecir total puntos halftime del partido desde datos de SportRadar
        NUEVA LÓGICA: Usa predicciones de halftime_teams_predictor, las suma y aplica tolerancia
        
        Args:
            game_data: Datos del juego de SportRadar
            
        Returns:
            Predicción en formato estándar de salida
        """
        if not self.is_loaded:
            return {'error': 'Modelo no cargado. Ejecutar load_data_and_model() primero.'}
        
        try:
            # Obtener información de equipos desde game_data
            home_team_name = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team_name = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            # Convertir nombres completos a abreviaciones para búsqueda en dataset
            home_team = self.common_utils._get_team_abbreviation(home_team_name)
            away_team = self.common_utils._get_team_abbreviation(away_team_name)
            
            logger.info(f"🔄 Calculando Total Points Halftime: {home_team_name} ({home_team}) vs {away_team_name} ({away_team})")
            
            # PASO 1: Obtener predicciones de ambos equipos usando HalfTimeTeamsPointsPredictor
            halftime_predictions = self.halftime_teams_predictor.predict_game(game_data)
            
            if isinstance(halftime_predictions, dict) and 'error' in halftime_predictions:
                logger.error(f"❌ Error obteniendo predicciones de halftime: {halftime_predictions['error']}")
                return halftime_predictions
            
            if not isinstance(halftime_predictions, list) or len(halftime_predictions) != 2:
                logger.error(f"❌ Formato inesperado de predicciones de halftime: {type(halftime_predictions)}")
                return {'error': 'Error: se esperaban 2 predicciones de halftime'}
            
            # PASO 2: Extraer puntos predichos de cada equipo
            home_prediction = halftime_predictions[0]  # Equipo local
            away_prediction = halftime_predictions[1]  # Equipo visitante
            
            home_ht_points = float(home_prediction.get('bet_line', 0))
            away_ht_points = float(away_prediction.get('bet_line', 0))
            
            logger.info(f"📊 Predicciones halftime individuales: {home_team_name}={home_ht_points}, {away_team_name}={away_ht_points}")
            
            # PASO 3: Sumar predicciones para obtener total halftime
            base_total_ht = home_ht_points + away_ht_points
            
            # PASO 4: Aplicar tolerancia conservadora específica para Total Points Halftime
            final_total_ht = base_total_ht + self.conservative_tolerance
            
            # PASO 5: Calcular confianza usando método específico para halftime total points
            home_confidence = home_prediction.get('confidence_percentage', 65.0)
            away_confidence = away_prediction.get('confidence_percentage', 65.0)
            
            # Usar método específico para halftime total points
            avg_confidence = self.confidence_calculator.calculate_halftime_total_points_confidence(
                home_confidence=home_confidence,
                away_confidence=away_confidence,
                home_team=home_team,
                away_team=away_team,
                game_data=game_data
            )
            
            logger.info(f"📊 Total Points Halftime calculado: {home_ht_points} + {away_ht_points} + ({self.conservative_tolerance}) = {final_total_ht}")
            
            # CALCULAR ESTADÍSTICAS DETALLADAS DE TOTALES HALFTIME HISTÓRICOS PARA PREDICTION_DETAILS
            # Estadísticas históricas de totales halftime del equipo local
            if self.historical_teams is None:
                logger.warning("⚠️ No hay datos históricos de equipos disponibles")
                home_team_historical = pd.DataFrame()
            else:
                home_team_historical = self.common_utils._smart_team_search(self.historical_teams, home_team)
            home_ht_totals_last_5 = []
            home_ht_totals_last_10 = []
            
            # Calcular totales halftime históricos (suma de HT del equipo + oponente)
            for idx, row in home_team_historical.iterrows():
                if self.historical_teams is not None:
                    opponent_ht = self.historical_teams[
                        (self.historical_teams['game_id'] == row['game_id']) & 
                        (self.historical_teams['Team'] != home_team)
                    ]['HT'].values
                else:
                    opponent_ht = []
                if len(opponent_ht) > 0 and pd.notna(row['HT']) and pd.notna(opponent_ht[0]):
                    total_ht = row['HT'] + opponent_ht[0]
                    home_ht_totals_last_10.append(total_ht)
                    if len(home_ht_totals_last_10) <= 5:
                        home_ht_totals_last_5.append(total_ht)
            
            # Estadísticas históricas de totales halftime del equipo visitante
            if self.historical_teams is None:
                away_team_historical = pd.DataFrame()
            else:
                away_team_historical = self.common_utils._smart_team_search(self.historical_teams, away_team)
            away_ht_totals_last_5 = []
            away_ht_totals_last_10 = []
            
            for idx, row in away_team_historical.iterrows():
                if self.historical_teams is not None:
                    opponent_ht = self.historical_teams[
                        (self.historical_teams['game_id'] == row['game_id']) & 
                        (self.historical_teams['Team'] != away_team)
                    ]['HT'].values
                else:
                    opponent_ht = []
                if len(opponent_ht) > 0 and pd.notna(row['HT']) and pd.notna(opponent_ht[0]):
                    total_ht = row['HT'] + opponent_ht[0]
                    away_ht_totals_last_10.append(total_ht)
                    if len(away_ht_totals_last_10) <= 5:
                        away_ht_totals_last_5.append(total_ht)
            
            # Estadísticas H2H halftime (enfrentamientos directos entre estos equipos)
            if self.historical_teams is not None:
                h2h_games = self.historical_teams[
                    ((self.historical_teams['Team'] == home_team) & (self.historical_teams['Opp'] == away_team)) |
                    ((self.historical_teams['Team'] == away_team) & (self.historical_teams['Opp'] == home_team))
                ].copy()
            else:
                h2h_games = pd.DataFrame()
            
            h2h_ht_totals = []
            for game_id in h2h_games['game_id'].unique():
                if self.historical_teams is not None:
                    game_data_h2h = self.historical_teams[self.historical_teams['game_id'] == game_id]
                else:
                    game_data_h2h = pd.DataFrame()
                if len(game_data_h2h) == 2:  # Ambos equipos presentes
                    ht_values = game_data_h2h['HT'].dropna()
                    if len(ht_values) == 2:  # Ambos tienen datos de HT
                        total_h2h_ht = ht_values.sum()
                        h2h_ht_totals.append(total_h2h_ht)
                
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "HT",
                "target_name": "total points",
                "bet_line": str(int(final_total_ht)),
                "bet_type": "points",
                "confidence_percentage": round(avg_confidence, 1),
                "prediction_details": {
                    "home_team_id": self.common_utils._get_team_id(home_team),
                    "away_team_id": self.common_utils._get_team_id(away_team),
                    "home_ht_points": home_ht_points,
                    "away_ht_points": away_ht_points,
                    "base_total_ht": home_ht_points + away_ht_points,
                    "conservative_tolerance": self.conservative_tolerance,
                    "final_total_ht": final_total_ht,
                    "method": "halftime_teams_sum_with_tolerance",
                    "home_confidence": home_confidence,
                    "away_confidence": away_confidence,
                    "home_team_ht_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(home_ht_totals_last_5), 1) if home_ht_totals_last_5 else 0,
                            "std": round(np.std(home_ht_totals_last_5), 1) if len(home_ht_totals_last_5) > 1 else 0,
                            "min": int(min(home_ht_totals_last_5)) if home_ht_totals_last_5 else 0,
                            "max": int(max(home_ht_totals_last_5)) if home_ht_totals_last_5 else 0,
                            "count": len(home_ht_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(home_ht_totals_last_10), 1) if home_ht_totals_last_10 else 0,
                            "std": round(np.std(home_ht_totals_last_10), 1) if len(home_ht_totals_last_10) > 1 else 0,
                            "min": int(min(home_ht_totals_last_10)) if home_ht_totals_last_10 else 0,
                            "max": int(max(home_ht_totals_last_10)) if home_ht_totals_last_10 else 0,
                            "count": len(home_ht_totals_last_10)
                        }
                    },
                    "away_team_ht_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(away_ht_totals_last_5), 1) if away_ht_totals_last_5 else 0,
                            "std": round(np.std(away_ht_totals_last_5), 1) if len(away_ht_totals_last_5) > 1 else 0,
                            "min": int(min(away_ht_totals_last_5)) if away_ht_totals_last_5 else 0,
                            "max": int(max(away_ht_totals_last_5)) if away_ht_totals_last_5 else 0,
                            "count": len(away_ht_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(away_ht_totals_last_10), 1) if away_ht_totals_last_10 else 0,
                            "std": round(np.std(away_ht_totals_last_10), 1) if len(away_ht_totals_last_10) > 1 else 0,
                            "min": int(min(away_ht_totals_last_10)) if away_ht_totals_last_10 else 0,
                            "max": int(max(away_ht_totals_last_10)) if away_ht_totals_last_10 else 0,
                            "count": len(away_ht_totals_last_10)
                        }
                    },
                    "h2h_ht_totals": {
                        "games_found": len(h2h_ht_totals),
                        "mean": round(np.mean(h2h_ht_totals), 1) if h2h_ht_totals else 0,
                        "std": round(np.std(h2h_ht_totals), 1) if len(h2h_ht_totals) > 1 else 0,
                        "min": int(min(h2h_ht_totals)) if h2h_ht_totals else 0,
                        "max": int(max(h2h_ht_totals)) if h2h_ht_totals else 0
                    }
                }
            }
                
        except Exception as e:
            logger.error(f"❌ Error en predicción total points halftime: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}


def test_halftime_total_points_predictor():
    """Función de prueba rápida del predictor de total points halftime"""
    print("🧪 PROBANDO HALFTIME TOTAL POINTS PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = HalfTimeTotalPointsPredictor()
    
    # Cargar datos y modelo
    print("📂 Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print("❌ Error cargando modelo")
        return False
    
    # Prueba con datos simulados de SportRadar
    print("\n🎯 Prueba con datos simulados de SportRadar:")
    
    # Simular datos de SportRadar - OKC vs HOU
    mock_sportradar_game = {
        "gameId": "sr:match:12345",
        "scheduled": "2024-01-15T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Oklahoma City Thunder",
            "alias": "OKC",
            "players": [
                {
                    "playerId": "sr:player:123",
                    "fullName": "Shai Gilgeous-Alexander",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "2",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:999",
                    "fullName": "Chet Holmgren",
                    "position": "C",
                    "starter": False,
                    "status": "ACT",
                    "jerseyNumber": "7",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Houston Rockets", 
            "alias": "HOU",
            "players": [
                {
                    "playerId": "sr:player:456",
                    "fullName": "Alperen Sengun",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "15",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:789",
                    "fullName": "Jalen Green",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "4",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:888",
                    "fullName": "Fred VanVleet",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "5",
                    "injuries": []
                }
            ]
        },
        "venue": {
            "name": "Paycom Center",
            "capacity": 18203
        }
    }
    
    print("   Prediciendo Total Points Halftime desde datos SportRadar:")
    result = predictor.predict_game(mock_sportradar_game)
    
    if 'error' not in result:
        print(f"   ✅ Resultado SportRadar (bet_line = predicción del modelo + confianza):")
        print(f"      home_team: {result['home_team']}")
        print(f"      away_team: {result['away_team']}")
        print(f"      target_type: {result['target_type']}")
        print(f"      target_name: {result['target_name']}")
        print(f"      bet_line: {result['bet_line']}")
        print(f"      bet_type: {result['bet_type']}")
        print(f"      confidence_percentage: {result['confidence_percentage']}% 🎯")
        print(f"      prediction_details: {result.get('prediction_details', 'N/A')}")
    else:
        print(f"   ❌ Error: {result['error']}")

    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_halftime_total_points_predictor()