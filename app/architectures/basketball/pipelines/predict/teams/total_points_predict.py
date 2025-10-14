#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN TOTAL PUNTOS
========================================

Wrapper final unificado para predicciones de total puntos que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo puntos equipos completo con calibraciones elite
- Formato estándar para módulo de stacking
- Manejo robusto de errores y tolerancia conservadora

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Convertir con GameDataAdapter
3. Buscar datos históricos específicos del equipo y el oponente
4. Generar features dinámicas
5. Aplicar modelo completo con calibraciones
6. Retornar formato estándar para stacking
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

class TotalPointsPredictor:
    """
    Wrapper final unificado para predicciones total puntos
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self, teams_df: pd.DataFrame = None):
        """Inicializar el predictor total puntos unificado"""
        self.model = None  # Mantenido como fallback
        self.teams_points_predictor = TeamsPointsPredictor(teams_df)  # Predictor de equipos individuales
        self.historical_players = None
        self.historical_teams = teams_df
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = TeamsConfidence()  # Calculadora de confianza centralizada
        self.is_loaded = False
        self.conservative_tolerance = 0  # Tolerancia conservadora para total points
        self.high_confidence_threshold = 75.0  # Umbral para alta confianza (más accesible)
        self.ultra_confidence_threshold = 85.0  # Umbral para ultra confianza (más accesible)

        # Cargar datos y modelo automáticamente
        self.load_data_and_model()
    
    def load_data_and_model(self) -> bool:
        """
        Cargar datos históricos y predictor de equipos (NUEVA LÓGICA)
        
        Returns:
            True si se cargó exitosamente
        """
        try:
            logger.info("🔄 Total Points ahora usa predicciones de Teams Points...")
            
            # Cargar el predictor de equipos (PRINCIPAL)
            logger.info("🏀 Cargando predictor de equipos...")
            if not self.teams_points_predictor.load_data_and_model():
                logger.error("❌ Error cargando predictor de equipos")
                return False
            
            # Cargar datos históricos para confianza
            data_loader = NBADataLoader(
                players_total_path="app/architectures/basketball/data/players_total.csv",
                players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                teams_total_path="app/architectures/basketball/data/teams_total.csv",
                teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                biometrics_path="app/architectures/basketball/data/biometrics.csv"
            )
            self.historical_players, self.historical_teams = data_loader.load_data()
            
            # Inicializar confidence_calculator con datos históricos
            self.confidence_calculator = TeamsConfidence()
            self.confidence_calculator.historical_teams = self.historical_teams
            self.confidence_calculator.historical_players = self.historical_players
            logger.info("✅ Confidence calculator inicializado con datos históricos")
            
            # Ya no necesitamos el modelo de total_points porque usamos teams_points
            self.model = None  # Será eliminado completamente
            
            self.is_loaded = True
            logger.info("✅ Total Points predictor listo (basado en Teams Points)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos y modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_game(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método principal para predecir total puntos del partido desde datos de SportRadar
        NUEVA LÓGICA: Usa predicciones de teams_points_predictor, las suma y aplica tolerancia
        
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
            
            logger.info(f"🔄 Calculando Total Points: {home_team_name} ({home_team}) vs {away_team_name} ({away_team})")
            
            # PASO 1: Obtener predicciones de ambos equipos usando TeamsPointsPredictor
            teams_predictions = self.teams_points_predictor.predict_game(game_data)
            
            if isinstance(teams_predictions, dict) and 'error' in teams_predictions:
                logger.error(f"❌ Error obteniendo predicciones de equipos: {teams_predictions['error']}")
                return teams_predictions
            
            if not isinstance(teams_predictions, list) or len(teams_predictions) != 2:
                logger.error(f"❌ Format inesperado de predicciones de equipos: {type(teams_predictions)}")
                return {'error': 'Error: se esperaban 2 predicciones de equipos'}
            
            # PASO 2: Extraer puntos predichos de cada equipo
            home_prediction = teams_predictions[0]  # Equipo local
            away_prediction = teams_predictions[1]  # Equipo visitante
            
            home_points = float(home_prediction.get('bet_line', 0))
            away_points = float(away_prediction.get('bet_line', 0))
            
            logger.info(f"📊 Predicciones individuales: {home_team_name}={home_points}, {away_team_name}={away_points}")
            
            # PASO 3: Sumar predicciones
            base_total = home_points + away_points
            
            # PASO 4: Aplicar tolerancia conservadora específica para Total Points
            final_total = base_total + self.conservative_tolerance
            
            # PASO 5: Calcular confianza promedio
            home_confidence = home_prediction.get('confidence_percentage', 65.0)
            away_confidence = away_prediction.get('confidence_percentage', 65.0)
            avg_confidence = (home_confidence + away_confidence) / 2.0
            
            logger.info(f"📊 Total Points calculado: {home_points} + {away_points} + ({self.conservative_tolerance}) = {final_total}")
            
            # CALCULAR ESTADÍSTICAS DETALLADAS DE TOTALES HISTÓRICOS PARA PREDICTION_DETAILS
            # Estadísticas históricas de totales del equipo local
            home_team_historical = self.common_utils._smart_team_search(self.historical_teams, home_team)
            home_totals_last_5 = []
            home_totals_last_10 = []
            
            # Calcular totales históricos (suma de puntos del equipo + oponente)
            for idx, row in home_team_historical.iterrows():
                opponent_points = self.historical_teams[
                    (self.historical_teams['game_id'] == row['game_id']) & 
                    (self.historical_teams['Team'] != home_team)
                ]['points'].values
                if len(opponent_points) > 0:
                    total_points = row['points'] + opponent_points[0]
                    home_totals_last_10.append(total_points)
                    if len(home_totals_last_10) <= 5:
                        home_totals_last_5.append(total_points)
            
            # Estadísticas históricas de totales del equipo visitante
            away_team_historical = self.common_utils._smart_team_search(self.historical_teams, away_team)
            away_totals_last_5 = []
            away_totals_last_10 = []
            
            for idx, row in away_team_historical.iterrows():
                opponent_points = self.historical_teams[
                    (self.historical_teams['game_id'] == row['game_id']) & 
                    (self.historical_teams['Team'] != away_team)
                ]['points'].values
                if len(opponent_points) > 0:
                    total_points = row['points'] + opponent_points[0]
                    away_totals_last_10.append(total_points)
                    if len(away_totals_last_10) <= 5:
                        away_totals_last_5.append(total_points)
            
            # Estadísticas H2H (enfrentamientos directos entre estos equipos)
            h2h_games = self.historical_teams[
                ((self.historical_teams['Team'] == home_team) & (self.historical_teams['Opp'] == away_team)) |
                ((self.historical_teams['Team'] == away_team) & (self.historical_teams['Opp'] == home_team))
            ].copy()
            
            h2h_totals = []
            for game_id in h2h_games['game_id'].unique():
                game_data_h2h = self.historical_teams[self.historical_teams['game_id'] == game_id]
                if len(game_data_h2h) == 2:  # Ambos equipos presentes
                    total_h2h = game_data_h2h['points'].sum()
                    h2h_totals.append(total_h2h)
            
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "match",
                "target_name": "total points",
                "bet_line": str(int(final_total)),
                "bet_type": "points",
                "confidence_percentage": round(avg_confidence, 1),
                "prediction_details": {
                    "home_team_id": self.common_utils._get_team_id(home_team),
                    "away_team_id": self.common_utils._get_team_id(away_team),
                    "home_points": home_points,
                    "away_points": away_points,
                    "base_total": base_total,
                    "conservative_tolerance": self.conservative_tolerance,
                    "final_total": final_total,
                    "home_confidence": home_confidence,
                    "away_confidence": away_confidence,
                    "home_team_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(home_totals_last_5), 1) if home_totals_last_5 else 0,
                            "std": round(np.std(home_totals_last_5), 1) if len(home_totals_last_5) > 1 else 0,
                            "min": int(min(home_totals_last_5)) if home_totals_last_5 else 0,
                            "max": int(max(home_totals_last_5)) if home_totals_last_5 else 0,
                            "count": len(home_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(home_totals_last_10), 1) if home_totals_last_10 else 0,
                            "std": round(np.std(home_totals_last_10), 1) if len(home_totals_last_10) > 1 else 0,
                            "min": int(min(home_totals_last_10)) if home_totals_last_10 else 0,
                            "max": int(max(home_totals_last_10)) if home_totals_last_10 else 0,
                            "count": len(home_totals_last_10)
                        }
                    },
                    "away_team_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(away_totals_last_5), 1) if away_totals_last_5 else 0,
                            "std": round(np.std(away_totals_last_5), 1) if len(away_totals_last_5) > 1 else 0,
                            "min": int(min(away_totals_last_5)) if away_totals_last_5 else 0,
                            "max": int(max(away_totals_last_5)) if away_totals_last_5 else 0,
                            "count": len(away_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(away_totals_last_10), 1) if away_totals_last_10 else 0,
                            "std": round(np.std(away_totals_last_10), 1) if len(away_totals_last_10) > 1 else 0,
                            "min": int(min(away_totals_last_10)) if away_totals_last_10 else 0,
                            "max": int(max(away_totals_last_10)) if away_totals_last_10 else 0,
                            "count": len(away_totals_last_10)
                        }
                    },
                    "h2h_totals": {
                        "games_found": len(h2h_totals),
                        "mean": round(np.mean(h2h_totals), 1) if h2h_totals else 0,
                        "std": round(np.std(h2h_totals), 1) if len(h2h_totals) > 1 else 0,
                        "min": int(min(h2h_totals)) if h2h_totals else 0,
                        "max": int(max(h2h_totals)) if h2h_totals else 0
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    
def test_total_points_predictor():
    """Función de prueba rápida del predictor de total points"""
    print("🧪 PROBANDO TOTAL POINTS PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = TotalPointsPredictor()
    
    # Cargar datos y modelo
    print("📂 Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print("❌ Error cargando modelo")
        return False
    
    # Prueba con datos simulados de SportRadar
    print("\n🎯 Prueba con datos simulados de SportRadar:")
    
    # Simular datos de SportRadar para Oklahoma vs Houston
    mock_sportradar_game = {
        "gameId": "sr:match:54321",
        "scheduled": "2024-01-20T20:00:00Z",
        "status": "scheduled",
        "homeTeam": {
            "name": "Oklahoma City Thunder",
            "alias": "OKC",
            "players": [
                {
                    "playerId": "sr:player:101",
                    "fullName": "Shai Gilgeous-Alexander",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "2",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:102",
                    "fullName": "Chet Holmgren",
                    "position": "C",
                    "starter": True,
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
                    "playerId": "sr:player:201",
                    "fullName": "Alperen Şengün",
                    "position": "C",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "28",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:202",
                    "fullName": "Jalen Green",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "4",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:203",
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
    
    # Probar predicción desde SportRadar
    print("   Prediciendo total points Oklahoma vs Houston:")
    sportradar_result = predictor.predict_game(mock_sportradar_game)
    
    if 'error' not in sportradar_result:
        print("   ✅ Resultado SportRadar (bet_line = predicción del modelo + confianza):")
        for key, value in sportradar_result.items():
            if key == 'confidence_percentage':
                print(f"      {key}: {value}% 🎯")
            else:
                print(f"      {key}: {value}")
    else:
        print(f"   ❌ Error: {sportradar_result['error']}")
        if 'available_teams' in sportradar_result:
            print(f"   Equipos disponibles: {sportradar_result['available_teams']}")
    
    # Mostrar detalles de la predicción
    print(f"\n📊 Detalles de la predicción:")
    print(f"   🏠 Equipo local: {sportradar_result.get('home_team', 'N/A')}")
    print(f"   ✈️ Equipo visitante: {sportradar_result.get('away_team', 'N/A')}")
    print(f"   🎯 Total predicho: {sportradar_result.get('bet_line', 'N/A')} puntos")
    print(f"   📈 Confianza: {sportradar_result.get('confidence_percentage', 'N/A')}%")
    
    if 'prediction_details' in sportradar_result:
        details = sportradar_result['prediction_details']
        print(f"\n📋 Detalles técnicos:")
        print(f"   🏠 Puntos equipo local: {details.get('home_points', 'N/A')}")
        print(f"   ✈️ Puntos equipo visitante: {details.get('away_points', 'N/A')}")
        print(f"   📊 Total base: {details.get('base_total', 'N/A')}")
        print(f"   📈 Tolerancia conservadora: {details.get('conservative_tolerance', 'N/A')}")
        print(f"   🎯 Total final: {details.get('final_total', 'N/A')}")
        print(f"   🏠 Confianza local: {details.get('home_confidence', 'N/A')}%")
        print(f"   ✈️ Confianza visitante: {details.get('away_confidence', 'N/A')}%")
        
        # Mostrar estadísticas de equipos
        if 'home_team_totals' in details:
            home_totals = details['home_team_totals']
            print(f"\n🏠 Estadísticas equipo local (últimos 5 juegos):")
            print(f"   📊 Promedio: {home_totals.get('last_5_games', {}).get('mean', 'N/A')}")
            print(f"   📈 Desviación: {home_totals.get('last_5_games', {}).get('std', 'N/A')}")
            print(f"   📊 Rango: {home_totals.get('last_5_games', {}).get('min', 'N/A')}-{home_totals.get('last_5_games', {}).get('max', 'N/A')}")
            print(f"   🎮 Juegos: {home_totals.get('last_5_games', {}).get('count', 'N/A')}")
            
            print(f"\n🏠 Estadísticas equipo local (últimos 10 juegos):")
            print(f"   📊 Promedio: {home_totals.get('last_10_games', {}).get('mean', 'N/A')}")
            print(f"   📈 Desviación: {home_totals.get('last_10_games', {}).get('std', 'N/A')}")
            print(f"   📊 Rango: {home_totals.get('last_10_games', {}).get('min', 'N/A')}-{home_totals.get('last_10_games', {}).get('max', 'N/A')}")
            print(f"   🎮 Juegos: {home_totals.get('last_10_games', {}).get('count', 'N/A')}")
        
        if 'away_team_totals' in details:
            away_totals = details['away_team_totals']
            print(f"\n✈️ Estadísticas equipo visitante (últimos 5 juegos):")
            print(f"   📊 Promedio: {away_totals.get('last_5_games', {}).get('mean', 'N/A')}")
            print(f"   📈 Desviación: {away_totals.get('last_5_games', {}).get('std', 'N/A')}")
            print(f"   📊 Rango: {away_totals.get('last_5_games', {}).get('min', 'N/A')}-{away_totals.get('last_5_games', {}).get('max', 'N/A')}")
            print(f"   🎮 Juegos: {away_totals.get('last_5_games', {}).get('count', 'N/A')}")
            
            print(f"\n✈️ Estadísticas equipo visitante (últimos 10 juegos):")
            print(f"   📊 Promedio: {away_totals.get('last_10_games', {}).get('mean', 'N/A')}")
            print(f"   📈 Desviación: {away_totals.get('last_10_games', {}).get('std', 'N/A')}")
            print(f"   📊 Rango: {away_totals.get('last_10_games', {}).get('min', 'N/A')}-{away_totals.get('last_10_games', {}).get('max', 'N/A')}")
            print(f"   🎮 Juegos: {away_totals.get('last_10_games', {}).get('count', 'N/A')}")
        
        # Mostrar estadísticas H2H si están disponibles
        if 'h2h_totals' in details:
            h2h = details['h2h_totals']
            print(f"\n🥊 Estadísticas Head-to-Head:")
            print(f"   🎮 Juegos encontrados: {h2h.get('games_found', 'N/A')}")
            print(f"   📊 Promedio H2H: {h2h.get('mean', 'N/A')}")
            print(f"   📈 Desviación H2H: {h2h.get('std', 'N/A')}")
            print(f"   📊 Rango H2H: {h2h.get('min', 'N/A')}-{h2h.get('max', 'N/A')}")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_total_points_predictor()
