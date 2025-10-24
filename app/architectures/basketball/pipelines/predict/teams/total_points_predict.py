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
            logger.info(" Total Points ahora usa predicciones de Teams Points...")
            
            # Cargar el predictor de equipos (PRINCIPAL)
            logger.info(" Cargando predictor de equipos...")
            if not self.teams_points_predictor.load_data_and_model():
                logger.error(" Error cargando predictor de equipos")
                return False
            
            # Cargar datos históricos para confianza
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
            
            # Ya no necesitamos el modelo de total_points porque usamos teams_points
            self.model = None  # Será eliminado completamente
            
            self.is_loaded = True
            logger.info(" Total Points predictor listo (basado en Teams Points)")
            
            return True
            
        except Exception as e:
            logger.error(f" Error cargando datos y modelo: {e}")
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
            
            logger.info(f" Calculando Total Points: {home_team_name} ({home_team}) vs {away_team_name} ({away_team})")
            
            # PASO 1: Obtener predicciones de ambos equipos usando TeamsPointsPredictor
            teams_predictions = self.teams_points_predictor.predict_game(game_data)
            
            if isinstance(teams_predictions, dict) and 'error' in teams_predictions:
                logger.error(f" Error obteniendo predicciones de equipos: {teams_predictions['error']}")
                return teams_predictions
            
            if not isinstance(teams_predictions, list) or len(teams_predictions) != 2:
                logger.error(f" Format inesperado de predicciones de equipos: {type(teams_predictions)}")
                return {'error': 'Error: se esperaban 2 predicciones de equipos'}
            
            # PASO 2: Extraer puntos predichos de cada equipo
            home_prediction = teams_predictions[0]  # Equipo local
            away_prediction = teams_predictions[1]  # Equipo visitante
            
            home_points = float(home_prediction.get('bet_line', 0))
            away_points = float(away_prediction.get('bet_line', 0))
            
            logger.info(f" Predicciones individuales: {home_team_name}={home_points}, {away_team_name}={away_points}")
            
            # PASO 3: Sumar predicciones
            base_total = home_points + away_points
            
            # PASO 4: Aplicar tolerancia conservadora específica para Total Points
            final_total = base_total + self.conservative_tolerance
            
            # PASO 5: Calcular confianza promedio (SIN FALLBACKS)
            home_confidence = home_prediction.get('confidence_percentage')
            away_confidence = away_prediction.get('confidence_percentage')
            
            if home_confidence is None or away_confidence is None:
                error_msg = "No se pudo obtener confianza de las predicciones de equipos"
                logger.error(f" {error_msg}")
                return {'error': error_msg}
            
            avg_confidence = (home_confidence + away_confidence) / 2.0
            
            logger.info(f" Total Points calculado: {home_points} + {away_points} + ({self.conservative_tolerance}) = {final_total}")
            
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
            
            # Calcular prediction_std basado en la variabilidad de los totales históricos
            prediction_std = None
            if len(home_totals_last_10) > 1 and len(away_totals_last_10) > 1:
                # Combinar std de ambos equipos usando suma cuadrática
                home_std = np.std(home_totals_last_10)
                away_std = np.std(away_totals_last_10)
                prediction_std = round(np.sqrt(home_std**2 + away_std**2), 1)
            
            # Calcular adjustment_factor basado en H2H
            adjustment_factor = None
            if len(h2h_totals) > 0:
                h2h_mean = np.mean(h2h_totals)
                # Comparar promedio H2H con promedio de ambos equipos
                if len(home_totals_last_10) > 0 and len(away_totals_last_10) > 0:
                    avg_team_totals = (np.mean(home_totals_last_10) + np.mean(away_totals_last_10)) / 2
                    if avg_team_totals > 0:
                        adjustment_factor = round(h2h_mean / avg_team_totals, 3)
            
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "match",
                "target_name": "total points",
                "bet_line": str(int(final_total)),
                "bet_type": "points",
                "confidence_percentage": round(avg_confidence, 1),
                "prediction_details": {
                    "home_team_id": game_data.get('homeTeam', {}).get('teamId', self.common_utils._get_team_id(home_team)) if game_data else self.common_utils._get_team_id(home_team),
                    "away_team_id": game_data.get('awayTeam', {}).get('teamId', self.common_utils._get_team_id(away_team)) if game_data else self.common_utils._get_team_id(away_team),
                    "home_points": home_points,
                    "away_points": away_points,
                    "base_total": base_total,
                    "conservative_tolerance": self.conservative_tolerance,
                    "final_total": final_total,
                    "home_confidence": home_confidence,
                    "away_confidence": away_confidence,
                    "prediction_std": prediction_std,
                    "adjustment_factor": adjustment_factor,
                    "home_team_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(home_totals_last_5), 1) if len(home_totals_last_5) > 0 else None,
                            "std": round(np.std(home_totals_last_5), 1) if len(home_totals_last_5) > 1 else None,
                            "min": int(min(home_totals_last_5)) if len(home_totals_last_5) > 0 else None,
                            "max": int(max(home_totals_last_5)) if len(home_totals_last_5) > 0 else None,
                            "count": len(home_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(home_totals_last_10), 1) if len(home_totals_last_10) > 0 else None,
                            "std": round(np.std(home_totals_last_10), 1) if len(home_totals_last_10) > 1 else None,
                            "min": int(min(home_totals_last_10)) if len(home_totals_last_10) > 0 else None,
                            "max": int(max(home_totals_last_10)) if len(home_totals_last_10) > 0 else None,
                            "count": len(home_totals_last_10)
                        }
                    },
                    "away_team_totals": {
                        "last_5_games": {
                            "mean": round(np.mean(away_totals_last_5), 1) if len(away_totals_last_5) > 0 else None,
                            "std": round(np.std(away_totals_last_5), 1) if len(away_totals_last_5) > 1 else None,
                            "min": int(min(away_totals_last_5)) if len(away_totals_last_5) > 0 else None,
                            "max": int(max(away_totals_last_5)) if len(away_totals_last_5) > 0 else None,
                            "count": len(away_totals_last_5)
                        },
                        "last_10_games": {
                            "mean": round(np.mean(away_totals_last_10), 1) if len(away_totals_last_10) > 0 else None,
                            "std": round(np.std(away_totals_last_10), 1) if len(away_totals_last_10) > 1 else None,
                            "min": int(min(away_totals_last_10)) if len(away_totals_last_10) > 0 else None,
                            "max": int(max(away_totals_last_10)) if len(away_totals_last_10) > 0 else None,
                            "count": len(away_totals_last_10)
                        }
                    },
                    "h2h_totals": {
                        "games_found": len(h2h_totals),
                        "mean": round(np.mean(h2h_totals), 1) if len(h2h_totals) > 0 else None,
                        "std": round(np.std(h2h_totals), 1) if len(h2h_totals) > 1 else None,
                        "min": int(min(h2h_totals)) if len(h2h_totals) > 0 else None,
                        "max": int(max(h2h_totals)) if len(h2h_totals) > 0 else None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f" Error en predicción desde SportRadar: {e}")
            import traceback
            traceback.print_exc()
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    
def test_total_points_predictor():
    """Función de prueba rápida del predictor de total points"""
    print("="*80)
    print(" PROBANDO TOTAL POINTS PREDICTOR - KNICKS VS CAVALIERS")
    print("="*80)
    
    # Inicializar predictor
    predictor = TotalPointsPredictor()
    
    # Cargar datos y modelo
    print("\n Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print(" Error cargando modelo")
        return False
    
    print("\n[OK] Modelo cargado exitosamente")
    
    # Prueba con datos simulados de SportRadar
    print("\n" + "="*80)
    print(" PRUEBA: KNICKS VS CAVALIERS - TOTAL PUNTOS DEL PARTIDO")
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
    print("\nPrediciendo total puntos del partido:")
    print("-" * 60)
    print("Metodo: Suma de predicciones individuales de equipos")
    sportradar_result = predictor.predict_game(mock_sportradar_game)
    
    import json
    
    if 'error' not in sportradar_result:
        print(f"\n[OK] PREDICCION EXITOSA")
        
        # Información principal
        total_points = sportradar_result['bet_line']
        confidence = sportradar_result['confidence_percentage']
        
        print(f"\nTOTAL PUNTOS PREDICHO: {total_points}")
        print(f"Confidence: {confidence}%")
        print(f"Bet Type: {sportradar_result['bet_type']}")
        
        # Detalles de la predicción
        if 'prediction_details' in sportradar_result:
            details = sportradar_result['prediction_details']
            
            print(f"\nDETALLES DE LA PREDICCION:")
            print(f"  Puntos equipo local: {details.get('home_points', 'N/A')}")
            print(f"  Puntos equipo visitante: {details.get('away_points', 'N/A')}")
            print(f"  Total base: {details.get('base_total', 'N/A')}")
            print(f"  Tolerancia: {details.get('conservative_tolerance', 'N/A')}")
            print(f"  Total final: {details.get('final_total', 'N/A')}")
            
            # Estadísticas históricas del equipo local
            if 'home_team_totals' in details:
                home_totals = details['home_team_totals']
                print(f"\nESTADISTICAS EQUIPO LOCAL ({sportradar_result['home_team']}):")
                print(f"  Ultimos 5 juegos: {home_totals['last_5_games']['mean']} puntos totales promedio")
                print(f"  Rango: {home_totals['last_5_games']['min']}-{home_totals['last_5_games']['max']}")
                print(f"  Ultimos 10 juegos: {home_totals['last_10_games']['mean']} puntos totales promedio")
            
            # Estadísticas históricas del equipo visitante
            if 'away_team_totals' in details:
                away_totals = details['away_team_totals']
                print(f"\nESTADISTICAS EQUIPO VISITANTE ({sportradar_result['away_team']}):")
                print(f"  Ultimos 5 juegos: {away_totals['last_5_games']['mean']} puntos totales promedio")
                print(f"  Rango: {away_totals['last_5_games']['min']}-{away_totals['last_5_games']['max']}")
                print(f"  Ultimos 10 juegos: {away_totals['last_10_games']['mean']} puntos totales promedio")
            
            # H2H
            if 'h2h_totals' in details:
                h2h = details['h2h_totals']
                print(f"\nESTADISTICAS H2H:")
                print(f"  Juegos encontrados: {h2h.get('games_found', 'N/A')}")
                print(f"  Promedio H2H: {h2h.get('mean', 'N/A')} puntos totales")
                print(f"  Rango: {h2h.get('min', 'N/A')}-{h2h.get('max', 'N/A')}")
        
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
        
    else:
        print(f"\n[ERROR] Error en prediccion")
        print(f"Error: {sportradar_result['error']}")
    
    print("\n" + "="*80)
    print(" PRUEBA COMPLETADA")
    print("="*80)
    return True


if __name__ == "__main__":
    test_total_points_predictor()
