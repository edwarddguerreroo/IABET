#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN PTS
========================================

Wrapper final unificado para predicciones de puntos (PTS) que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo PTS completo con calibraciones elite
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
from app.architectures.basketball.src.models.players.pts.model_pts import XGBoostPTSModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence_predict import PlayersConfidence

logger = logging.getLogger(__name__)

class PTSPredictor:
    """
    Wrapper final unificado para predicciones PTS
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self):
        """Inicializar el predictor PTS unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = PlayersConfidence()
        self.is_loaded = False
        self.tolerance = 0  # Tolerancia optimista individual
    
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
            
            # Cargar modelo PTS usando joblib directo
            model_path = "app/architectures/basketball/.joblib/pts_model.joblib"
            logger.info(f"🤖 Cargando modelo PTS completo desde: {model_path}")
            
            import joblib
            self.model = joblib.load(model_path)
            logger.info("✅ Modelo PTS cargado como objeto completo")
            
            # Verificar que se cargó correctamente
            if hasattr(self.model, 'stacking_model') and self.model.stacking_model is not None:
                logger.info("✅ Modelo PTS completo cargado correctamente")
            else:
                raise ValueError("Modelo PTS no se cargó correctamente")
            
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos y modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    
    def predict_game(self, game_data: Dict[str, Any], target_player: str) -> Dict[str, Any]:
        """
        Metodo principal para predecir PTS desde datos de insumo (SportRadar)
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            Predicción en formato JSON exacto especificado o None
        """
        if not self.is_loaded:
            return None
        
        try:
            # 1. FILTRAR JUGADORES DISPONIBLES DEL ROSTER
            available_players = self.confidence_calculator.filter_available_players_from_roster(game_data)
            
            # 2. VERIFICAR SI EL JUGADOR ESTÁ EN EL ROSTER Y DISPONIBLE
            if target_player not in available_players:
                logger.info(f"❌ {target_player} no disponible en el roster")
                return None
            
            # 3. CONVERTIR DATOS DE SPORTRADAR CON GAMEADAPTER
            players_df, teams_df = self.game_adapter.convert_game_to_dataframes(game_data)
            
            # 4. BUSCAR EL JUGADOR OBJETIVO CON BÚSQUEDA INTELIGENTE
            target_row = self.common_utils._smart_player_search(players_df, target_player)
            
            if target_row.empty:
                logger.warning(f"❌ Jugador {target_player} no encontrado en datos convertidos")
                return None
            
            # 5. EXTRAER DATOS DEL JUGADOR
            player_data = target_row.iloc[0].to_dict()
            
            # 6. EXTRAER INFORMACIÓN ADICIONAL DESDE SPORTRADAR
            is_home = self.common_utils._get_is_home_from_sportradar(game_data, target_player)
            is_started = self.common_utils._get_is_started_from_sportradar(game_data, target_player)
            current_team = self.common_utils._get_current_team_from_sportradar(game_data, target_player)
            
            # Agregar información extraída al player_data
            player_data['is_home'] = is_home
            player_data['is_started'] = is_started
            player_data['current_team'] = current_team
            player_data['player_name'] = target_player
            
            # 7. CORREGIR FORMATO DE FECHA
            if 'Date' in player_data and pd.notna(player_data['Date']):
                if hasattr(player_data['Date'], 'strftime'):
                    player_data['Date'] = player_data['Date'].strftime('%Y-%m-%d')
                elif hasattr(player_data['Date'], 'date'):
                    player_data['Date'] = str(player_data['Date'].date())
            
            # 8. HACER PREDICCIÓN USANDO EL MÉTODO INTERNO
            prediction_result = self.predict_single_player(player_data, game_data)
            
            if prediction_result is None:
                logger.info(f"⚠️  Predicción no realizada, es menor a 5 points")
                return None
            
            if 'error' in prediction_result:
                logger.error(f"❌ Error en predicción interna: {prediction_result['error']}")
                return None
            
            # 9. EXTRAER RESULTADOS DE LA PREDICCIÓN
            raw_prediction = prediction_result['raw_prediction']
            final_prediction = prediction_result['pts_prediction']
            confidence_percentage = prediction_result['confidence_percentage']
            prediction_details = prediction_result.get('prediction_details', {})
            
            # 10. OBTENER INFORMACIÓN DE EQUIPOS
            home_team = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            # 11. DEVOLVER FORMATO JSON EXACTO ESPECIFICADO CON PREDICTION_DETAILS
            return {
                "home_team": home_team,
                "away_team": away_team,
                "target_type": "player",
                "target_name": target_player,
                "bet_line": str(int(final_prediction)),
                "bet_type": "points",
                "confidence_percentage": round(confidence_percentage, 1),
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_single_player(self, player_data: Dict[str, Any], game_data: Dict = None) -> Dict[str, Any]:
        """
        Predecir puntos para un jugador individual CON CONFIANZA INTEGRADA
        
        Args:
            player_data: Diccionario con datos del jugador
            game_data: Datos del juego de SportRadar (para confianza)
                
        Returns:
            Diccionario con predicción, confianza y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            # PASO CRÍTICO: Buscar datos históricos del jugador específico
            player_name = player_data.get('player_name', player_data.get('Player', 'Unknown'))
            
            # Usar búsqueda inteligente de CommonUtils (ya optimizado)
            player_historical_df = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            if player_historical_df.empty:
                logger.warning(f"⚠️ No se encontraron datos históricos para {player_name}")
                # Usar datos de jugadores similares o promedio
                player_historical = self.historical_players.head(100).copy()
            else:
                player_historical = player_historical_df.copy()
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
            
            # Hacer predicción con el modelo usando solo datos históricos
            predictions = self.model.predict(player_historical)
            
            # Extraer la última predicción (corresponde al último juego histórico)
            raw_prediction = predictions[-1] if len(predictions) > 0 else 0
            
            # CALCULAR MÉTRICAS DETALLADAS PARA CONFIANZA Y PREDICTION_DETAILS
            if len(player_historical) > 0:
                historical_pts = player_historical['points'].dropna()
                actual_stats_mean = historical_pts.mean() if len(historical_pts) > 0 else raw_prediction
                actual_stats_std = historical_pts.std() if len(historical_pts) > 1 else 1.0
                prediction_std = 1.5  # Estimación conservadora para modelo PTS
                
                # ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
                # Últimos 5 juegos
                last_5_games = historical_pts.tail(5) if len(historical_pts) >= 5 else historical_pts
                last_5_mean = last_5_games.mean() if len(last_5_games) > 0 else 0
                last_5_std = last_5_games.std() if len(last_5_games) > 1 else 0
                last_5_min = last_5_games.min() if len(last_5_games) > 0 else 0
                last_5_max = last_5_games.max() if len(last_5_games) > 0 else 0
                
                # Últimos 10 juegos
                last_10_games = historical_pts.tail(10) if len(historical_pts) >= 10 else historical_pts
                last_10_mean = last_10_games.mean() if len(last_10_games) > 0 else 0
                last_10_std = last_10_games.std() if len(last_10_games) > 1 else 0
                last_10_min = last_10_games.min() if len(last_10_games) > 0 else 0
                last_10_max = last_10_games.max() if len(last_10_games) > 0 else 0
                
                # Tendencia (diferencia entre últimos 5 vs anteriores 5)
                if len(historical_pts) >= 10:
                    recent_5 = historical_pts.tail(5).mean()
                    previous_5 = historical_pts.tail(10).head(5).mean()
                    trend_5_games = recent_5 - previous_5
                else:
                    trend_5_games = 0
                
                # Consistencia (inversa de la desviación estándar)
                consistency_score = max(0, 100 - (actual_stats_std * 2)) if actual_stats_std > 0 else 100
                
                # Forma reciente (últimos 3 juegos)
                last_3_games = historical_pts.tail(3) if len(historical_pts) >= 3 else historical_pts
                recent_form = last_3_games.mean() if len(last_3_games) > 0 else 0
                
            else:
                actual_stats_mean = raw_prediction
                actual_stats_std = 3.0
                prediction_std = 2.0
                # Valores por defecto para estadísticas detalladas
                last_5_mean = last_5_std = last_5_min = last_5_max = 0
                last_10_mean = last_10_std = last_10_min = last_10_max = 0
                trend_5_games = 0
                consistency_score = 50
                recent_form = 0
            
            # CALCULAR PREDICCIÓN ESTABILIZADA PARA CONFIANZA
            stabilized_prediction = (raw_prediction * 0.70) + (actual_stats_mean * 0.30)
            
            # CALCULAR ESTADÍSTICAS H2H DETALLADAS
            h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                player_name=player_name,
                opponent_team=player_data.get('Opp', 'Unknown'),
                target_stat='points',
                max_games=10
            )
            
            # CALCULAR CONFIANZA USANDO PLAYERSCONFIDENCE
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
            
            # APLICAR FACTOR H2H A LA PREDICCIÓN
            h2h_factor = h2h_stats.get('h2h_factor', 1.0)
            if h2h_factor != 1.0 and h2h_stats.get('games_found', 0) >= 3:
                # Si hay suficientes datos H2H, aplicar el factor
                raw_prediction_adjusted = raw_prediction * h2h_factor
                logger.info(f"🎯 Aplicando factor H2H {h2h_factor:.3f} a predicción: {raw_prediction:.1f} -> {raw_prediction_adjusted:.1f}")
            else:
                raw_prediction_adjusted = raw_prediction
            
            # APLICAR TOLERANCIA INDIVIDUAL DEL PREDICTOR
            pts_prediction = max(0, raw_prediction_adjusted + self.tolerance)  # No permitir valores negativos
            
            # No inferir predicciones menores a 5 PTS
            if pts_prediction < 5:
                logger.info(f"⚠️  Predicción {pts_prediction:.1f} PTS < 5, no se infiere (casas de apuestas manejan líneas ≥5)")
                return None
            
            return {
                'raw_prediction': raw_prediction,
                'pts_prediction': int(pts_prediction),
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
                    'raw_prediction': round(raw_prediction, 1),
                    'h2h_adjusted_prediction': round(raw_prediction_adjusted, 1),
                    'final_prediction': round(pts_prediction, 1),
                    'actual_stats_mean': round(actual_stats_mean, 1),
                    'actual_stats_std': round(actual_stats_std, 1),
                    'prediction_std': round(prediction_std, 1),
                    'last_5_games': {
                        'mean': round(last_5_mean, 1),
                        'std': round(last_5_std, 1),
                        'min': round(last_5_min, 1),
                        'max': round(last_5_max, 1),
                        'count': len(last_5_games)
                    },
                    'last_10_games': {
                        'mean': round(last_10_mean, 1),
                        'std': round(last_10_std, 1),
                        'min': round(last_10_min, 1),
                        'max': round(last_10_max, 1),
                        'count': len(last_10_games)
                    },
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
                            'data_quality': 'high' if len(player_historical) >= 10 else 'medium' if len(player_historical) >= 5 else 'low'
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
                'pts_prediction': None
            }


def test_pts_predictor():
    """Función de prueba rápida del predictor PTS"""
    print("🧪 PROBANDO PTS PREDICTOR")
    print("="*50)
    
    # Inicializar predictor
    predictor = PTSPredictor()
    
    # Cargar datos y modelo
    print("📂 Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print("❌ Error cargando modelo")
        return False
    
    # Prueba con datos simulados de SportRadar
    print("\n🎯 Prueba con datos simulados de SportRadar:")
    
    # Simular datos de SportRadar con Shai Gilgeous-Alexander (elite)
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
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "7",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:777",
                    "fullName": "Isaiah Joe",
                    "position": "G",
                    "starter": False,
                    "status": "ACT",
                    "jerseyNumber": "11",
                    "injuries": []
                }
            ]
        },
        "awayTeam": {
            "name": "Dallas Mavericks", 
            "alias": "DAL",
            "players": [
                {
                    "playerId": "sr:player:456",
                    "fullName": "Luka Dončić",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "77",
                    "injuries": []
                },
                {
                    "playerId": "sr:player:789",
                    "fullName": "Kyrie Irving",
                    "position": "G",
                    "starter": True,
                    "status": "ACT",
                    "jerseyNumber": "11",
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
            "name": "Paycom Center",
            "capacity": 18203
        }
    }
    
    # Probar predicción desde SportRadar
    print("   Prediciendo Shai Gilgeous-Alexander desde datos SportRadar:")
    sportradar_result = predictor.predict_game(
        mock_sportradar_game, 
        "Shai Gilgeous-Alexander"
    )
    
    if sportradar_result is not None:
        print("   ✅ Resultado SportRadar (formato JSON exacto):")
        import json
        
        # Convertir numpy types a Python types para JSON
        def convert_numpy_types(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        result_clean = convert_numpy_types(sportradar_result)
        print(json.dumps(result_clean, indent=4, ensure_ascii=False))
    else:
        print("   ⚠️ No se hizo predicción (jugador no disponible o error)")
    
    # Probar búsqueda inteligente con nombres problemáticos
    print("\n🧠 Pruebas de búsqueda inteligente:")
    
    # Casos para probar regla de predicciones < 5 PTS
    test_cases = [
        ("Shai Gilgeous-Alexander", "Elite OKC"),
        ("Chet Holmgren", "Promedio OKC"),
        ("Isaiah Joe", "Suplente OKC"),
        ("Luka Doncic", "Elite DAL"),
        ("P.J. Washington", "Promedio DAL"),
        ("Kyrie Irving", "Elite DAL (datos problemáticos)"),
        ("LeBron James", "Jugador inexistente")
    ]
    
    for i, (player_name, description) in enumerate(test_cases, 1):
        print(f"   {i}. {description}: '{player_name}'")
        result = predictor.predict_game(mock_sportradar_game, player_name)
        
        if result is not None:
            print(f"      ✅ Predicción exitosa:")
            import json
            
            # Convertir numpy types a Python types para JSON
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                return obj
            
            result_clean = convert_numpy_types(result)
            print(json.dumps(result_clean, indent=8, ensure_ascii=False))
    else:
            print(f"      ⚠️ No se hizo predicción (jugador no disponible)")
    
    print("\n✅ Prueba completada")
    return True


if __name__ == "__main__":
    test_pts_predictor()
