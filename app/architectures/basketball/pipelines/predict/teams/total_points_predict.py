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
from app.architectures.basketball.pipelines.predict.utils_predict.confidence_predict import TeamsConfidence
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
        self.conservative_tolerance = -7  # Tolerancia conservadora para total points
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
            home_team = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            logger.info(f"🔄 Calculando Total Points: {home_team} vs {away_team}")
            
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
            
            logger.info(f"📊 Predicciones individuales: {home_team}={home_points}, {away_team}={away_points}")
            
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
                "home_team": home_team,
                "away_team": away_team,
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
                    "method": "teams_sum_with_tolerance",
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
    
    def predict_match_total(self, teams_df: pd.DataFrame, game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir total de puntos del partido (ambos equipos)
        
        Args:
            teams_df: DataFrame con datos de ambos equipos
            game_data: Datos del juego de SportRadar (opcional)
                
        Returns:
            Diccionario con predicción de total de puntos y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            # PASO CRÍTICO: Combinar datos históricos de ambos equipos (ÚLTIMOS 30 PARTIDOS)
            if len(teams_df) < 2:
                return {'error': 'Se necesitan datos de ambos equipos para predicción de total points'}
            
            # Obtener nombres de equipos
            team_names = teams_df['Team'].unique()
            home_team_name = team_names[0] if len(team_names) > 0 else 'Unknown'
            away_team_name = team_names[1] if len(team_names) > 1 else 'Unknown'
            
            logger.info(f"🏀 Prediciendo total points: {home_team_name} vs {away_team_name}")
            
            # Combinar datos históricos de ambos equipos
            combined_historical = []
            
            for team_name in team_names:
                # Filtrar datos históricos del equipo específico usando búsqueda inteligente
                team_historical_full = self.common_utils._smart_team_search(self.historical_teams, team_name)
                
                if len(team_historical_full) > 0:
                    # LIMITAR A ÚLTIMOS 30 PARTIDOS para mejor precisión
                    team_historical = team_historical_full.tail(30).copy()
                    combined_historical.append(team_historical)
                    
                    total_available = len(team_historical_full)
                    used_games = len(team_historical)
                    logger.info(f"✅ {team_name}: {used_games} juegos recientes de {total_available} disponibles")
                else:
                    logger.warning(f"⚠️ No se encontraron datos históricos para {team_name}")
                
            if not combined_historical:
                # Usar datos de referencia si no hay históricos
                combined_df = self.historical_teams.head(60).tail(30).copy()
                logger.info(f"📊 Usando datos de referencia: {len(combined_df)} registros")
            else:
                # Combinar datos históricos de ambos equipos
                combined_df = pd.concat(combined_historical, ignore_index=True)
                logger.info(f"📊 Datos combinados: {len(combined_df)} registros de ambos equipos")
            
            # Hacer predicción con el modelo de total points
            predictions = self.model.predict(combined_df)

            # ESTABILIDAD 1: Usar promedio de las últimas predicciones basadas en histórico
            if len(predictions) > 0:
                # Tomar promedio de las últimas 5 predicciones para mayor estabilidad
                recent_predictions = predictions[-5:] if len(predictions) >= 5 else predictions
                raw_prediction = np.mean(recent_predictions)
                prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            else:
                raw_prediction = 220  # Valor por defecto para total points NBA
                prediction_std = 10  # Desviación estándar por defecto
            
            # ESTABILIDAD 2: Usar promedio de los últimos resultados reales de total points
            # Para total points, necesitamos sumar los puntos de ambos equipos por partido
            if 'points' in combined_df.columns and len(combined_df) > 0:
                # Calcular total points por fecha (suma de ambos equipos)
                if 'Date' in combined_df.columns:
                    total_points_by_date = combined_df.groupby('Date')['points'].sum().tail(10)  # Últimos 10 partidos
                    if len(total_points_by_date) > 0:
                        games_to_use = min(5, len(total_points_by_date))
                        recent_total_points = total_points_by_date.tail(games_to_use).values
                        actual_total_mean = np.mean(recent_total_points)
                        actual_total_std = np.std(recent_total_points) if len(recent_total_points) > 1 else 0
                        
                        logger.info(f"📊 Estabilización histórica: últimos {games_to_use} partidos, promedio {actual_total_mean:.1f} pts totales")
                    else:
                        actual_total_mean = raw_prediction
                        actual_total_std = prediction_std
                else:
                    # Si no hay fecha, usar promedio simple
                    actual_total_mean = raw_prediction
                    actual_total_std = prediction_std
            else:
                actual_total_mean = raw_prediction
                actual_total_std = prediction_std
            
            # FACTOR JUGADORES ESTRELLA - Ajustar predicción basada en ausencias
            star_player_factor = self.confidence_calculator.calculate_star_player_factor_total_points(
                home_team=home_team_name,
                away_team=away_team_name, 
                game_data=game_data
            )
            logger.info(f"⭐ Factor jugadores estrella: {star_player_factor:.3f}")
            
            # ANÁLISIS HEAD-TO-HEAD - Estadísticas de enfrentamientos directos
            h2h_stats = self.confidence_calculator.calculate_head_to_head_stats_total_points(home_team_name, away_team_name)
            logger.info(f"🥊 Estadísticas H2H: {h2h_stats}")
            
            # APLICAR FACTORES A LA PREDICCIÓN
            # Aplicar factor de jugadores estrella
            raw_prediction_adjusted = raw_prediction * star_player_factor
            actual_total_mean_adjusted = actual_total_mean * star_player_factor
            
            # Ajustar también basado en tendencia H2H de total points
            if h2h_stats.get('total_points_mean'):
                h2h_total_mean = h2h_stats['total_points_mean']
                # Para H2H muy consistente (>90% consistencia), dar más peso al histórico
                consistency = h2h_stats.get('consistency_score', 0)
                games_found = h2h_stats.get('games_found', 0)
                
                if consistency > 90 and games_found >= 8:
                    # H2H MUY CONSISTENTE - Dar mucho más peso al histórico
                    # Usar una combinación más agresiva: 70% H2H + 30% modelo
                    h2h_weight = 0.7
                    model_weight = 0.3
                    target_prediction = h2h_total_mean * h2h_weight + raw_prediction_adjusted * model_weight
                    h2h_factor = target_prediction / raw_prediction_adjusted
                    h2h_factor = max(0.9, min(1.4, h2h_factor))  # Limites más amplios
                    logger.info(f"🎯 H2H ULTRA CONSISTENTE ({consistency:.1f}%) - Blend 70/30: {h2h_factor:.3f}")
                elif consistency > 80 and games_found >= 5:
                    # H2H Consistente - Factor moderado
                    h2h_factor = min(1.25, max(0.9, h2h_total_mean / 220))
                    logger.info(f"🎯 H2H CONSISTENTE ({consistency:.1f}%) - Factor moderado: {h2h_factor:.3f}")
                else:
                    # Factor conservador para H2H menos consistente
                    h2h_factor = min(1.1, max(0.9, h2h_total_mean / 220))
                    logger.info(f"🎯 H2H Estándar - Factor: {h2h_factor:.3f}")
                
                raw_prediction_adjusted *= h2h_factor
                logger.info(f"🎯 Factor H2H aplicado: {h2h_factor:.3f} (basado en promedio H2H: {h2h_total_mean:.1f})")
            
            logger.info(f"📊 Predicción ajustada: {raw_prediction:.1f} → {raw_prediction_adjusted:.1f}")
            
            # CALCULAR CONFIANZA PRELIMINAR PARA DETERMINAR ESTRATEGIA
            match_data = {
                'is_home': 1,
                'star_player_factor': star_player_factor,
                'h2h_stats': h2h_stats
            }  # Datos del partido para cálculo de confianza
            preliminary_confidence = self.confidence_calculator.calculate_total_points_confidence(
                raw_prediction=raw_prediction,
                stabilized_prediction=raw_prediction,  # Usar raw para cálculo inicial
                tolerance=self.base_tolerance,
                prediction_std=prediction_std,
                actual_points_std=actual_total_std,
                historical_games=len(combined_df),
                team_data=match_data
            )
            
            # SISTEMA ADAPTATIVO BASADO EN CONFIANZA PARA 95%+ EFECTIVIDAD
            final_prediction, tolerance_used = self.confidence_calculator._adaptive_prediction_strategy(
                raw_prediction=raw_prediction_adjusted,
                actual_points_mean=actual_total_mean_adjusted,
                confidence=preliminary_confidence,
                prediction_std=prediction_std,
                actual_points_std=actual_total_std
            )
            
            total_points_prediction = max(180, final_prediction)  # Límite basado en análisis real del dataset (P1)
            
            # RECALCULAR CONFIANZA CON VALORES FINALES USANDO CLASE CENTRALIZADA
            confidence_percentage = self.confidence_calculator.calculate_total_points_confidence(
                raw_prediction=raw_prediction,
                stabilized_prediction=final_prediction,
                tolerance=tolerance_used,
                prediction_std=prediction_std,
                actual_points_std=actual_total_std,
                historical_games=len(combined_df),
                team_data=match_data
            )
                
            return {
                'total_points_prediction': int(total_points_prediction),
                'confidence_percentage': round(confidence_percentage, 1),
                'prediction_details': {
                    'home_team': home_team_name,
                    'away_team': away_team_name,
                    'tolerance_applied': tolerance_used,
                    'historical_games_used': len(combined_df),
                    'raw_prediction': round(raw_prediction, 1),
                    'adjusted_prediction': round(raw_prediction_adjusted, 1),
                    'final_prediction': round(final_prediction, 1),
                    'actual_total_mean': round(actual_total_mean, 1),
                    'prediction_std': round(prediction_std, 2),
                    'actual_total_std': round(actual_total_std, 2),
                    'preliminary_confidence': round(preliminary_confidence, 1),
                    'strategy_used': "MODERATE",  # Simplificado por ahora
                    'star_player_factor': round(star_player_factor, 3),
                    'h2h_stats': {
                        'games_found': h2h_stats.get('games_found', 0),
                        'total_points_mean': round(h2h_stats.get('total_points_mean', 0), 1),
                        'last_5_mean': round(h2h_stats.get('last_5_mean', 0), 1) if h2h_stats.get('last_5_mean') else None,
                        'last_10_mean': round(h2h_stats.get('last_10_mean', 0), 1) if h2h_stats.get('last_10_mean') else None,
                        'h2h_factor': round(h2h_stats.get('h2h_factor', 1.0), 3),
                        'consistency_score': round(h2h_stats.get('consistency_score', 0), 1)
                    }
                }
            }
                
        except Exception as e:
                logger.error(f"❌ Error en predicción: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'home_team': home_team_name if 'home_team_name' in locals() else 'Unknown',
                    'away_team': away_team_name if 'away_team_name' in locals() else 'Unknown',
                    'error': str(e),
                    'total_points_prediction': None
                }
    
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
    if 'prediction_details' in sportradar_result:
        details = sportradar_result['prediction_details']
        print(f"\n📊 Detalles de la predicción:")
        print(f"   🏠 Equipo local: {details.get('home_team', 'N/A')}")
        print(f"   ✈️ Equipo visitante: {details.get('away_team', 'N/A')}")
        print(f"   📈 Predicción RAW: {details.get('raw_prediction', 'N/A')}")
        print(f"   🎯 Predicción final: {details.get('final_prediction', 'N/A')}")
        print(f"   📊 Tolerancia aplicada: {details.get('tolerance_applied', 'N/A')}")
        print(f"   🎮 Juegos históricos: {details.get('historical_games_used', 'N/A')}")
        print(f"   🧠 Estrategia: {details.get('strategy_used', 'N/A')}")
        print(f"   📈 Confianza preliminar: {details.get('preliminary_confidence', 'N/A')}%")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_total_points_predictor()
