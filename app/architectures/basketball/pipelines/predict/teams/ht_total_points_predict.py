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
                self.historical_players, self.historical_teams, self.historical_players_quarters, self.historical_teams_quarters = data_loader.load_data_with_halftime_target()
                logger.info(" Datos históricos cargados para halftime total points")
            
            # El predictor de halftime equipos ya se inicializa automáticamente
            if self.halftime_teams_predictor.is_loaded:
                self.is_loaded = True
                logger.info(" Modelo halftime total points cargado exitosamente")
                return True
            else:
                logger.error(" Error: predictor de halftime equipos no cargado")
                return False
            
        except Exception as e:
            logger.error(f" Error cargando modelo halftime total points: {e}")
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
            # Obtener información de equipos desde game_data (sin fallbacks)
            if not game_data or 'homeTeam' not in game_data or 'awayTeam' not in game_data:
                logger.error("❌ game_data no contiene información de equipos")
                return {'error': 'game_data debe contener homeTeam y awayTeam'}
            
            home_team_name = game_data['homeTeam'].get('name')
            away_team_name = game_data['awayTeam'].get('name')
            
            if not home_team_name or not away_team_name:
                logger.error("❌ No se pudo obtener nombres de equipos")
                return {'error': 'No se pudo obtener nombres de equipos desde game_data'}

            # Convertir nombres completos a abreviaciones para búsqueda en dataset
            home_team = self.common_utils._get_team_abbreviation(home_team_name)
            away_team = self.common_utils._get_team_abbreviation(away_team_name)
            
            logger.info(f" Calculando Total Points Halftime: {home_team_name} ({home_team}) vs {away_team_name} ({away_team})")
            
            # PASO 1: Obtener predicciones de ambos equipos usando HalfTimeTeamsPointsPredictor
            halftime_predictions = self.halftime_teams_predictor.predict_game(game_data)
            
            if isinstance(halftime_predictions, dict) and 'error' in halftime_predictions:
                logger.error(f" Error obteniendo predicciones de halftime: {halftime_predictions['error']}")
                return halftime_predictions
            
            if not isinstance(halftime_predictions, list) or len(halftime_predictions) != 2:
                logger.error(f" Formato inesperado de predicciones de halftime: {type(halftime_predictions)}")
                return {'error': 'Error: se esperaban 2 predicciones de halftime'}
            
            # PASO 2: Extraer puntos predichos de cada equipo (sin fallbacks)
            home_prediction = halftime_predictions[0]  # Equipo local
            away_prediction = halftime_predictions[1]  # Equipo visitante
            
            # Validar que bet_line existe en ambas predicciones
            if 'bet_line' not in home_prediction:
                logger.error(f"❌ Predicción HT de {home_team_name} no contiene 'bet_line'")
                return {'error': f"Predicción HT de {home_team_name} sin 'bet_line'"}
            if 'bet_line' not in away_prediction:
                logger.error(f"❌ Predicción HT de {away_team_name} no contiene 'bet_line'")
                return {'error': f"Predicción HT de {away_team_name} sin 'bet_line'"}
            
            home_ht_points = float(home_prediction['bet_line'])
            away_ht_points = float(away_prediction['bet_line'])
            
            logger.info(f" Predicciones halftime individuales: {home_team_name}={home_ht_points}, {away_team_name}={away_ht_points}")
            
            # PASO 3: Sumar predicciones para obtener total halftime
            base_total_ht = home_ht_points + away_ht_points
            
            # PASO 4: Aplicar tolerancia conservadora específica para Total Points Halftime
            final_total_ht = base_total_ht + self.conservative_tolerance
            
            # PASO 5: Calcular confianza usando método específico para halftime total points (sin fallbacks)
            # Validar que confidence_percentage existe en ambas predicciones
            if 'confidence_percentage' not in home_prediction:
                logger.error(f"❌ Predicción HT de {home_team_name} no contiene 'confidence_percentage'")
                return {'error': f"Predicción HT de {home_team_name} sin 'confidence_percentage'"}
            if 'confidence_percentage' not in away_prediction:
                logger.error(f"❌ Predicción HT de {away_team_name} no contiene 'confidence_percentage'")
                return {'error': f"Predicción HT de {away_team_name} sin 'confidence_percentage'"}
            
            home_confidence = home_prediction['confidence_percentage']
            away_confidence = away_prediction['confidence_percentage']
            
            # Usar método específico para halftime total points
            avg_confidence = self.confidence_calculator.calculate_halftime_total_points_confidence(
                home_confidence=home_confidence,
                away_confidence=away_confidence,
                home_team=home_team,
                away_team=away_team,
                game_data=game_data
            )
            
            logger.info(f" Total Points Halftime calculado: {home_ht_points} + {away_ht_points} + ({self.conservative_tolerance}) = {final_total_ht}")
            
            # CALCULAR ESTADÍSTICAS DETALLADAS DE TOTALES HALFTIME HISTÓRICOS PARA PREDICTION_DETAILS
            # Estadísticas históricas de totales halftime del equipo local
            if self.historical_teams is None:
                logger.warning(" No hay datos históricos de equipos disponibles")
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
            
            # Calcular prediction_std basado en la variabilidad de los totales HT históricos
            prediction_std = None
            if len(home_ht_totals_last_10) > 1 and len(away_ht_totals_last_10) > 1:
                # Combinar std de ambos equipos usando suma cuadrática
                home_std = np.std(home_ht_totals_last_10)
                away_std = np.std(away_ht_totals_last_10)
                prediction_std = round(np.sqrt(home_std**2 + away_std**2), 1)
            
            # Calcular adjustment_factor basado en H2H HT
            adjustment_factor = None
            if len(h2h_ht_totals) > 0:
                h2h_mean = np.mean(h2h_ht_totals)
                # Comparar promedio H2H con promedio de ambos equipos
                if len(home_ht_totals_last_10) > 0 and len(away_ht_totals_last_10) > 0:
                    avg_team_ht_totals = (np.mean(home_ht_totals_last_10) + np.mean(away_ht_totals_last_10)) / 2
                    if avg_team_ht_totals > 0:
                        adjustment_factor = round(h2h_mean / avg_team_ht_totals, 3)
                
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "HT",
                "target_name": "total points",
                "bet_line": str(int(final_total_ht)),
                "bet_type": "points",
                "confidence_percentage": round(avg_confidence, 1),
                "prediction_details": {
                    "home_team_id": game_data.get('homeTeam', {}).get('teamId', self.common_utils._get_team_id(home_team)) if game_data else self.common_utils._get_team_id(home_team),
                    "away_team_id": game_data.get('awayTeam', {}).get('teamId', self.common_utils._get_team_id(away_team)) if game_data else self.common_utils._get_team_id(away_team),
                    "home_ht_points": home_ht_points,
                    "away_ht_points": away_ht_points,
                    "base_total_ht": home_ht_points + away_ht_points,
                    "conservative_tolerance": self.conservative_tolerance,
                    "final_total_ht": final_total_ht,
                    "method": "halftime_teams_sum_with_tolerance",
                    "home_confidence": home_confidence,
                    "away_confidence": away_confidence,
                    "prediction_std": prediction_std,
                    "adjustment_factor": adjustment_factor,
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
            logger.error(f" Error en predicción total points halftime: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}


def test_halftime_total_points_predictor():
    """Función de prueba rápida del predictor de total points halftime"""
    print("="*80)
    print(" PROBANDO HALFTIME TOTAL POINTS PREDICTOR - KNICKS VS CAVALIERS")
    print("="*80)
    
    # Inicializar predictor
    predictor = HalfTimeTotalPointsPredictor()
    
    # Cargar datos y modelo
    print("\n Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print(" Error cargando modelo")
        return False
    
    print("\n[OK] Modelo cargado exitosamente")
    
    # Prueba con datos simulados de SportRadar
    print("\n" + "="*80)
    print(" PRUEBA: KNICKS VS CAVALIERS - PUNTOS MEDIO TIEMPO")
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
    
    print("\nPrediciendo Total Points Halftime desde datos SportRadar:")
    print("-" * 60)
    result = predictor.predict_game(mock_sportradar_game)
    
    import json
    
    if 'error' not in result:
        print(f"\n[OK] PREDICCION EXITOSA")
        
        # Información principal
        total_ht = result['bet_line']
        confidence = result['confidence_percentage']
        
        print(f"\nTOTAL PUNTOS HT PREDICHO: {total_ht}")
        print(f"Confidence: {confidence}%")
        print(f"Bet Type: {result['bet_type']}")
        
        # Detalles de la predicción
        if 'prediction_details' in result:
            details = result['prediction_details']
            
            print(f"\nDETALLES DE LA PREDICCION:")
            print(f"  Puntos HT equipo local: {details.get('home_ht_points', 'N/A')}")
            print(f"  Puntos HT equipo visitante: {details.get('away_ht_points', 'N/A')}")
            print(f"  Total HT base: {details.get('base_total_ht', 'N/A')}")
            print(f"  Tolerancia: {details.get('conservative_tolerance', 'N/A')}")
            print(f"  Total HT final: {details.get('final_total_ht', 'N/A')}")
            
            # Estadísticas históricas del equipo local
            if 'home_team_ht_totals' in details:
                home_totals = details['home_team_ht_totals']
                print(f"\nESTADISTICAS EQUIPO LOCAL ({result['home_team']}):")
                print(f"  Ultimos 5 juegos HT: {home_totals['last_5_games']['mean']} puntos totales HT promedio")
                print(f"  Rango: {home_totals['last_5_games']['min']}-{home_totals['last_5_games']['max']}")
                print(f"  Ultimos 10 juegos HT: {home_totals['last_10_games']['mean']} puntos totales HT promedio")
            
            # Estadísticas históricas del equipo visitante
            if 'away_team_ht_totals' in details:
                away_totals = details['away_team_ht_totals']
                print(f"\nESTADISTICAS EQUIPO VISITANTE ({result['away_team']}):")
                print(f"  Ultimos 5 juegos HT: {away_totals['last_5_games']['mean']} puntos totales HT promedio")
                print(f"  Rango: {away_totals['last_5_games']['min']}-{away_totals['last_5_games']['max']}")
                print(f"  Ultimos 10 juegos HT: {away_totals['last_10_games']['mean']} puntos totales HT promedio")
            
            # H2H
            if 'h2h_ht_totals' in details:
                h2h = details['h2h_ht_totals']
                print(f"\nESTADISTICAS H2H:")
                print(f"  Juegos encontrados: {h2h.get('games_found', 'N/A')}")
                print(f"  Promedio H2H HT: {h2h.get('mean', 'N/A')} puntos totales HT")
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
        
        print(json.dumps(result, indent=2, ensure_ascii=False, default=convert_numpy))
        
    else:
        print(f"\n[ERROR] Error en prediccion")
        print(f"Error: {result['error']}")

    print("\n" + "="*80)
    print(" PRUEBA COMPLETADA")
    print("="*80)
    return True


if __name__ == "__main__":
    test_halftime_total_points_predictor()