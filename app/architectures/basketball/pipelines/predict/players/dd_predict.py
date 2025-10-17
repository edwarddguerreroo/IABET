#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN DOUBLE DOUBLE
================================================

Wrapper final unificado para predicciones de double double que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo Double Double avanzado con stacking
- Formato estándar para módulo de stacking
- Validación de estado del jugador (OUT, INJURED)
- Salida binaria: yes (1) / no (0)

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Convertir con GameDataAdapter
3. Validar estado del jugador (ACT vs OUT/INJURED)
4. Buscar datos históricos específicos del jugador
5. Generar features especializadas para double double
6. Aplicar modelo de clasificación con threshold óptimo
7. Retornar formato estándar con predicción yes/no
"""

import sys
import os
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Union, Tuple, Optional
import logging
import joblib

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Importar modelos y data loaders
from app.architectures.basketball.src.models.players.double_double.dd_model import DoubleDoubleModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence.confidence_players import PlayersConfidence

logger = logging.getLogger(__name__)

class DoubleDoublePredictor:
    """
    Pipeline encargado de predecir si un jugador hará un double double en un game (DD)
    
    Args:
        model: Modelo Double Double completo con calibraciones
        historical_players: Datos históricos de jugadores
        historical_teams: Datos históricos de equipos
        game_adapter: GameDataAdapter para convertir datos de SportRadar
        is_loaded: Flag para verificar si el modelo está cargado
        confidence_threshold: Umbral de confianza para predicciones

    Returns:
        Predicción de double double para un jugador (yes/no)
    """
    
    def __init__(self):
        """Inicializar el predictor double double unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = None
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = PlayersConfidence()
        self.is_loaded = False
        self.tolerance = 0.7  # Tolerancia para DD (probabilidad) individual
        self.confidence_threshold = 0.6  # Umbral de confianza para predicciones

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
            
            # Inicializar modelo Double Double avanzado
            logger.info("🤖 Inicializando modelo Double Double avanzado...")
            self.model = DoubleDoubleModel(
                optimize_hyperparams=False  # Para carga rápida
            )
            
            # Cargar modelo entrenado
            model_path = "app/architectures/basketball/.joblib/dd_model.joblib"
            logger.info(f"📦 Cargando modelo desde: {model_path}")
            
            # Verificar si el archivo existe
            if not os.path.exists(model_path):
                logger.error(f"❌ Archivo de modelo no encontrado: {model_path}")
                logger.warning("⚠️ AVISO: El modelo necesita ser reentrenado")
                return False
            

            # Primero intentar cargar como DoubleDoubleModel completo (nuevo formato)
            self.model = DoubleDoubleModel.load_model(model_path)
            logger.info("✅ Modelo cargado en formato nuevo (objeto completo)")
                
            
            # DEBUG: Verificar estado final del modelo
            logger.info(f"🔍 DEBUG Estado final del modelo:")
            logger.info(f"   Tipo: {type(self.model)}")
            logger.info(f"   is_fitted: {getattr(self.model, 'is_fitted', 'N/A')}")
            logger.info(f"   tiene stacking_model: {hasattr(self.model, 'stacking_model') and self.model.stacking_model is not None}")
            logger.info(f"   tiene scaler: {hasattr(self.model, 'scaler')}")
            logger.info(f"   tiene feature_engineer: {hasattr(self.model, 'feature_engineer')}")
            
            self.is_loaded = True
            
            # Mostrar información del modelo
            logger.info("📊 Información del modelo Double Double:")
            if hasattr(self.model, 'training_results'):
                logger.info(f"   Features utilizadas: {len(self.model.feature_importance) if hasattr(self.model, 'feature_importance') else 'N/A'}")
                if 'optimal_threshold' in self.model.training_results:
                    logger.info(f"   Threshold óptimo: {self.model.training_results['optimal_threshold']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error cargando datos y modelo: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_game(self, game_data: Dict[str, Any], target_player: str) -> Dict[str, Any]:
        """
        Metodo principal para predecir double double desde datos de insumo (SportRadar)
        
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
            
            # 🚨 SOLO DEVOLVER PREDICCIONES PARA JUGADORES QUE HARÁN DD
            if prediction_result is None:
                logger.info(f"❌ No se predice double double para {target_player}")
                return None
            
            if 'error' in prediction_result:
                return prediction_result
            
            # Formatear salida según especificación (solo para YES)
            prediction_value = prediction_result['dd_prediction']
            confidence = prediction_result['confidence_percentage']
            prediction_details = prediction_result.get('prediction_details', {})
            
            # Obtener información de equipos desde game_data
            home_team = game_data.get('homeTeam', {}).get('name', 'Home Team')
            away_team = game_data.get('awayTeam', {}).get('name', 'Away Team')
            
            return {
                "home_team": home_team,
                "away_team": away_team,
                "target_type": "player",
                "target_name": target_player,
                "bet_line": "yes",  # Solo predecimos YES
                "bet_type": "double_double",
                "confidence_percentage": round(confidence, 1),
                "prediction_details": prediction_details
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}
    
    def predict_single_player(self, player_data: Dict[str, Any], game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir double double para un jugador individual
        
        Args:
            player_data: Diccionario con datos del jugador
            game_data: Datos del juego (opcional para contexto)
                
        Returns:
            Diccionario con predicción y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            
            # PASO CRÍTICO: Buscar datos históricos del jugador específico
            player_name = player_data.get('Player', 'Unknown')
            current_team = player_data.get('current_team', 'Unknown')
            
            # Usar búsqueda inteligente para encontrar el jugador en datos históricos
            player_historical = self.common_utils._smart_player_search(self.historical_players, player_name)
            
            # Intentar usar datos del equipo actual primero, si no hay suficientes usar historial completo
            if current_team != 'Unknown' and 'Team' in player_historical.columns:
                logger.info(f"🏀 Filtrando datos por equipo actual: {current_team}")
                
                # Buscar datos del equipo actual
                current_team_data = player_historical[
                    player_historical['Team'] == current_team
                ].copy()
                
                # Si hay suficientes datos del equipo actual (mínimo 5 juegos), usarlos
                if len(current_team_data) >= 5:
                    logger.info(f"✅ Usando {len(current_team_data)} juegos del equipo actual ({current_team})")
                    player_historical = current_team_data
                else:
                    # Si no hay suficientes datos del equipo actual, usar TODOS los datos históricos
                    logger.info(f"📅 Pocos datos de {current_team} ({len(current_team_data)} juegos), usando TODOS los {len(player_historical)} registros históricos para {player_name}")
            else:
                logger.info(f"🏀 Usando TODOS los {len(player_historical)} registros históricos para {player_name} (sin filtro por equipo)")
            
            if len(player_historical) == 0:
                logger.warning(f"⚠️ No se encontraron datos históricos para {player_name}")
                return {
                    'error': f'No hay datos históricos suficientes para {player_name}',
                    'dd_prediction': 0,
                    'confidence_percentage': 0.0
                }
            else:
                logger.info(f"✅ Encontrados {len(player_historical)} registros históricos para {player_name}")
            
            # Tomar una muestra representativa para predicción (últimos 30 juegos)
            recent_data = player_historical.tail(30).copy()
            
            # DEBUG: Analizar el estado del modelo y los datos
            logger.info(f"🔍 DEBUG - Análisis de datos para {player_name}:")
            logger.info(f"   📊 Shape de recent_data: {recent_data.shape}")
            logger.info(f"   📋 Columnas disponibles: {list(recent_data.columns)}")
            logger.info(f"   🔧 Modelo cargado: {type(self.model).__name__}")
            logger.info(f"   ⚙️ Tiene stacking_model: {hasattr(self.model, 'stacking_model')}")
            
            if hasattr(self.model, 'feature_engineer'):
                logger.info(f"   🛠️ Tiene feature_engineer: {type(self.model.feature_engineer).__name__}")
            
            # DEBUG: Intentar generar features paso a paso
            try:
                logger.info("🧪 Intentando generar features paso a paso...")
                
                # Verificar si el modelo tiene feature_engineer
                if hasattr(self.model, 'feature_engineer') and self.model.feature_engineer is not None:
                    logger.info("✅ Feature engineer encontrado, generando features...")
                    
                    # DEBUG: Verificar el estado de recent_data antes de features
                    logger.info(f"   📊 Datos antes de features - Shape: {recent_data.shape}")
                    logger.info(f"   📋 Primeras columnas: {list(recent_data.columns[:10])}")
                    logger.info(f"   💾 Memoria usada: {recent_data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                    
                    # Intentar generar features especializadas
                    try:
                        specialized_features = self.model.feature_engineer.generate_all_features(recent_data.copy())
                        
                        # DEBUG: Verificar tipo de retorno
                        if isinstance(specialized_features, list):
                            logger.info(f"✅ Features especializadas generadas: {len(specialized_features)} (lista)")
                        elif isinstance(specialized_features, dict):
                            logger.info(f"✅ Features especializadas generadas: {len(specialized_features)} (dict)")
                            logger.info(f"   📝 Features generadas: {list(specialized_features.keys())[:10]}")
                        else:
                            logger.info(f"✅ Features especializadas generadas: {type(specialized_features)}")
                        
                        # DEBUG ADICIONAL: Verificar estado del modelo antes de predicción
                        logger.info(f"🔍 Estado del modelo antes de predicción:")
                        logger.info(f"   is_fitted: {self.model.is_fitted}")
                        logger.info(f"   tiene stacking_model: {hasattr(self.model, 'stacking_model') and self.model.stacking_model is not None}")
                        logger.info(f"   tiene scaler: {hasattr(self.model, 'scaler')}")
                        
                        # Ahora intentar la predicción
                        probabilities = self.model.predict_proba(recent_data)
                        predictions = self.model.predict(recent_data)
                        logger.info("✅ Predicción exitosa!")
                        
                    except Exception as feature_error:
                        logger.error(f"❌ Error generando features: {feature_error}")
                        logger.error(f"   🔍 Tipo de error: {type(feature_error).__name__}")
                        
                        # Si es el error específico de Series comparison, intentar arreglo
                        if "Can only compare identically-labeled Series objects" in str(feature_error):
                            logger.info("🔧 Detectado error de Series comparison, intentando fix...")
                            try:
                                # Resetear índices para alinear las Series
                                recent_data_fixed = recent_data.copy().reset_index(drop=True)
                                logger.info(f"📊 DataFrame reindexado: {recent_data_fixed.shape}")
                                
                                # Intentar nuevamente con datos reindexados
                                specialized_features = self.model.feature_engineer.generate_all_features(recent_data_fixed)
                                probabilities = self.model.predict_proba(recent_data_fixed)
                                predictions = self.model.predict(recent_data_fixed)
                                logger.info("✅ Predicción exitosa con fix de índices!")
                                
                            except Exception as fix_error:
                                logger.error(f"❌ Fix de índices falló: {fix_error}")
                                # DEBUG: Información detallada del error original
                                import traceback
                                logger.error(f"   📜 Traceback del error original:")
                                for line in traceback.format_exc().split('\n'):
                                    if line.strip():
                                        logger.error(f"      {line}")
                                raise feature_error
                        else:
                            # DEBUG: Información detallada del error
                            import traceback
                            logger.error(f"   📜 Traceback completo:")
                            for line in traceback.format_exc().split('\n'):
                                if line.strip():
                                    logger.error(f"      {line}")
                            
                            # Intentar enfoque alternativo más simple
                            logger.info("🔄 Intentando enfoque alternativo...")
                            raise feature_error
                        
                else:
                    logger.warning("⚠️ No se encontró feature_engineer, usando datos directos")
                    probabilities = self.model.predict_proba(recent_data)
                    predictions = self.model.predict(recent_data)
                    
            except Exception as prediction_error:
                logger.error(f"❌ Error en predicción completa: {prediction_error}")
                logger.error(f"   🔍 Tipo de error: {type(prediction_error).__name__}")
                
                # Volver a lanzar el error para que el flujo principal lo maneje
                raise prediction_error
            
            # Usar la última predicción (corresponde al contexto más reciente)
            if len(predictions) > 0:
                # Tomar solo la última predicción del modelo
                final_prediction = int(np.round(predictions[-1]))
                
                # Para probabilidades, también tomar la última
                recent_probabilities = probabilities[-1:] if len(probabilities) > 0 else probabilities
                
                # SOLO PREDECIR PARA JUGADORES QUE HARÁN DOUBLE DOUBLE (YES)
                if final_prediction != 1:
                    logger.info(f"❌ No se predice DD para {player_name} (predicción: {final_prediction})")
                    return None  # No hacer predicción para jugadores que no harán DD
                
                # Calcular confianza usando sistema avanzado
                if len(recent_probabilities) > 0 and recent_probabilities.shape[1] >= 2:
                    # Usar directamente la probabilidad de la última predicción
                    probability_yes = recent_probabilities[0][1]  # Probabilidad de "yes" de la última predicción
                    
                    # Sistema de confianza avanzado basado en total_points
                    confidence = self._calculate_confidence(
                        player_data=recent_data,
                        probability_yes=probability_yes,
                        is_home=game_data.get('is_home', False) if game_data else False,
                        historical_games=len(recent_data)
                    )
                else:
                    confidence = 50.0  # Confianza neutral si no hay probabilidades
                
                # CALCULAR ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS
                # Últimos 5 juegos
                last_5_games = recent_data.tail(5)['DD'] if len(recent_data) >= 5 and 'DD' in recent_data.columns else recent_data['DD'] if 'DD' in recent_data.columns else pd.Series([0])
                last_5_stats = {
                    'mean': round(last_5_games.mean(), 1) if len(last_5_games) > 0 else 0,
                    'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                    'min': int(last_5_games.min()) if len(last_5_games) > 0 else 0,
                    'max': int(last_5_games.max()) if len(last_5_games) > 0 else 0,
                    'count': len(last_5_games)
                }
                
                # Últimos 10 juegos
                last_10_games = recent_data.tail(10)['DD'] if len(recent_data) >= 10 and 'DD' in recent_data.columns else recent_data['DD'] if 'DD' in recent_data.columns else pd.Series([0])
                last_10_stats = {
                    'mean': round(last_10_games.mean(), 1) if len(last_10_games) > 0 else 0,
                    'std': round(last_10_games.std(), 1) if len(last_10_games) > 1 else 0,
                    'min': int(last_10_games.min()) if len(last_10_games) > 0 else 0,
                    'max': int(last_10_games.max()) if len(last_10_games) > 0 else 0,
                    'count': len(last_10_games)
                }
                
                # Análisis de tendencia
                if len(recent_data) >= 5 and 'DD' in recent_data.columns:
                    recent_5_mean = last_5_games.mean()
                    recent_10_mean = last_10_games.mean() if len(recent_data) >= 10 else recent_5_mean
                    trend_5_games = recent_5_mean - recent_10_mean
                else:
                    trend_5_games = 0
                    recent_5_mean = 0
                
                # Score de consistencia
                dd_std = recent_data['DD'].std() if 'DD' in recent_data.columns else 0
                consistency_score = max(0, 100 - (dd_std * 100)) if dd_std > 0 else 100
                
                # Forma reciente (promedio de últimos 3 juegos)
                recent_form = recent_data.tail(3)['DD'].mean() if len(recent_data) >= 3 and 'DD' in recent_data.columns else 0
                
                # CALCULAR ESTADÍSTICAS H2H DETALLADAS
                h2h_stats = self.confidence_calculator.calculate_player_h2h_stats(
                    player_name=player_name,
                    opponent_team=player_data.get('Opp', 'Unknown'),
                    target_stat='DD',
                    max_games=10
                )
                
                # APLICAR FACTOR H2H A LA PREDICCIÓN
                h2h_factor = h2h_stats.get('h2h_factor', 1.0)
                if h2h_factor != 1.0 and h2h_stats.get('games_found', 0) >= 3:
                    raw_prediction_adjusted = final_prediction * h2h_factor
                    logger.info(f"🎯 Aplicando factor H2H {h2h_factor:.3f} a predicción DD: {final_prediction:.1f} -> {raw_prediction_adjusted:.1f}")
                else:
                    raw_prediction_adjusted = final_prediction
            else:
                final_prediction = 0
                confidence = 50.0
            
            # Asegurar que la confianza esté en rango válido
            confidence = max(50.0, min(95.0, confidence))
                
            return {
                'dd_prediction': final_prediction,
                'confidence_percentage': round(confidence, 1),
            'prediction_details': {
                    'player_id': self.common_utils._get_player_id(self.common_utils._normalize_name(player_name), player_data.get('Team', 'Unknown')),
                    'player': player_name,
                    'team_id': self.common_utils._get_team_id(player_data.get('Team', 'Unknown')),
                    'team': player_data.get('Team', 'Unknown'),
                    'opponent_id': player_data.get('opponent_team_id', self.common_utils._get_team_id(player_data.get('Opp', 'Unknown'))),
                    'opponent': player_data.get('opponent_team', player_data.get('Opp', 'Unknown')),
                    'tolerance_applied': self.tolerance,
                    'raw_prediction': final_prediction,
                    'h2h_adjusted_prediction': round(raw_prediction_adjusted, 1) if 'raw_prediction_adjusted' in locals() else final_prediction,
                    'final_prediction': final_prediction,
                    'actual_stats_mean': round(recent_data['DD'].mean(), 1) if 'DD' in recent_data.columns else 0,
                    'actual_stats_std': round(recent_data['DD'].std(), 1) if 'DD' in recent_data.columns else 0,
                    'last_5_games': last_5_stats if 'last_5_stats' in locals() else {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
                    'last_10_games': last_10_stats if 'last_10_stats' in locals() else {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0},
                    'trend_analysis': {
                        'trend_5_games': round(trend_5_games, 1) if 'trend_5_games' in locals() else 0,
                        'consistency_score': round(consistency_score, 1) if 'consistency_score' in locals() else 0,
                        'recent_form': round(recent_form, 1) if 'recent_form' in locals() else 0
                    },
                    'performance_metrics': {
                        'stabilized_prediction': round(final_prediction, 1),
                        'confidence_factors': {
                            'historical_games': len(recent_data),
                            'data_quality': 'high' if len(recent_data) >= 20 else 'medium'
                        }
                    },
                    'h2h_stats': {
                        'games_found': h2h_stats.get('games_found', 0) if 'h2h_stats' in locals() else 0,
                        'h2h_mean': h2h_stats.get('h2h_mean') if 'h2h_stats' in locals() else None,
                        'h2h_std': h2h_stats.get('h2h_std') if 'h2h_stats' in locals() else None,
                        'h2h_min': h2h_stats.get('h2h_min') if 'h2h_stats' in locals() else None,
                        'h2h_max': h2h_stats.get('h2h_max') if 'h2h_stats' in locals() else None,
                        'h2h_factor': h2h_stats.get('h2h_factor', 1.0) if 'h2h_stats' in locals() else 1.0,
                        'consistency_score': h2h_stats.get('consistency_score', 0) if 'h2h_stats' in locals() else 0,
                        'last_5_mean': h2h_stats.get('last_5_mean') if 'h2h_stats' in locals() else None,
                        'last_10_mean': h2h_stats.get('last_10_mean') if 'h2h_stats' in locals() else None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error en predict_single_player: {e}")
            return {
                'error': f'Error en predicción: {str(e)}',
                'dd_prediction': 0,
                'confidence_percentage': 0.0
            }
    
    def _calculate_confidence(self, player_data: pd.DataFrame, probability_yes: float, 
                             is_home: bool, historical_games: int) -> float:
        """
        Calcular porcentaje de confianza basado en múltiples factores (adaptado de total_points)
        
        Args:
            player_data: Datos históricos del jugador
            probability_yes: Probabilidad de hacer double double
            is_home: Si juega en casa
            historical_games: Número de juegos históricos
            
        Returns:
            Porcentaje de confianza (0-100)
        """
        try:
            # FACTOR 1: Probabilidad del modelo (40% del peso)
            model_confidence = probability_yes * 100
            
            # FACTOR 2: Consistencia histórica del jugador (25% del peso)
            if 'DD' in player_data.columns:
                dd_rate = player_data['DD'].mean()
                dd_std = player_data['DD'].std()
                
                # Mayor tasa de DD = mayor confianza
                dd_rate_confidence = dd_rate * 100
                
                # Menor variabilidad = mayor confianza
                if dd_std > 0:
                    consistency_confidence = max(0, 100 - (dd_std * 100))
                else:
                    consistency_confidence = 95
                    
                historical_consistency = (dd_rate_confidence + consistency_confidence) / 2
            else:
                historical_consistency = 70  # Default si no hay datos de DD
            
            # FACTOR 3: Tendencia reciente (20% del peso)
            if len(player_data) >= 10:
                recent_5 = player_data.tail(5)['DD'].mean() if 'DD' in player_data.columns else 0.5
                recent_10 = player_data.tail(10)['DD'].mean() if 'DD' in player_data.columns else 0.5
                
                # Si la tendencia reciente es mejor que el promedio general, aumentar confianza
                overall_rate = player_data['DD'].mean() if 'DD' in player_data.columns else 0.5
                if recent_5 > overall_rate:
                    trend_confidence = min(95, recent_5 * 120)  # Boost por tendencia positiva
                else:
                    trend_confidence = recent_5 * 100
            else:
                trend_confidence = 70  # Default para pocos datos
            
            # FACTOR 4: Cantidad de datos históricos (10% del peso)
            if historical_games >= 25:
                data_confidence = 95
            elif historical_games >= 20:
                data_confidence = 90
            elif historical_games >= 15:
                data_confidence = 80
            elif historical_games >= 10:
                data_confidence = 70
            else:
                data_confidence = max(50, historical_games * 5)
            
            # FACTOR 5: Contexto del juego (5% del peso)
            game_context = 75  # Base
            
            # Bonus por jugar en casa
            if is_home:
                game_context += 10
            
            # Bonus por minutos jugados (si disponible)
            if 'MP' in player_data.columns:
                avg_minutes = player_data['MP'].mean()
                if avg_minutes >= 30:  # Jugador titular
                    game_context += 5
                elif avg_minutes >= 25:  # Jugador importante
                    game_context += 3
            
            game_context = min(95, game_context)
            
            # CALCULAR CONFIANZA PONDERADA
            weighted_confidence = (
                model_confidence * 0.40 +           # 40% - Probabilidad del modelo
                historical_consistency * 0.25 +     # 25% - Consistencia histórica
                trend_confidence * 0.20 +           # 20% - Tendencia reciente
                data_confidence * 0.10 +            # 10% - Cantidad de datos
                game_context * 0.05                 # 5% - Contexto del juego
            )
            
            # APLICAR LÍMITES REALISTAS (50% - 95%)
            final_confidence = max(50.0, min(95.0, weighted_confidence))
            
            logger.info(f"🎯 Confianza DD calculada: {final_confidence:.1f}% "
                       f"(modelo:{model_confidence:.0f}, hist:{historical_consistency:.0f}, "
                       f"trend:{trend_confidence:.0f}, data:{data_confidence:.0f}, "
                       f"game:{game_context:.0f})")
            
            return final_confidence
            
        except Exception as e:
            logger.warning(f"⚠️ Error calculando confianza DD: {e}")
            return 75.0  # Confianza por defecto
    
   
def main():
    """Función de prueba del Double Double Predictor"""
    logger.info("🧪 PROBANDO DOUBLE DOUBLE PREDICTOR")
    logger.info("=" * 50)
    
    try:
        # Inicializar predictor
        predictor = DoubleDoublePredictor()
        
        # Cargar datos y modelo
        logger.info("📂 Cargando datos y modelo...")
        if not predictor.load_data_and_model():
            logger.error("❌ No se pudo cargar el modelo")
            return False
        
        # Crear datos reales basados en el formato de SportRadar real proporcionado
        mock_sportradar_game = {
            "gameId": "f71cb64f-4d52-4e2b-a3db-7436b798476d",
            "status": "scheduled",
            "scheduled": "2025-05-18T19:30:00+00:00",
            "venue": {
                "id": "a13af216-4409-5021-8dd5-255cc71bffc3",
                "name": "Paycom Center",
                "city": "Oklahoma City",
                "state": "OK",
                "capacity": 18203
            },
            "homeTeam": {
                "teamId": "583ecfff-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Oklahoma City Thunder",
                "alias": "OKC",
                "players": [
                    {
                        "playerId": "d9ea4a8f-ff51-408d-b518-980efc2a35a1",
                        "fullName": "Shai Gilgeous-Alexander",
                        "jerseyNumber": 2,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "eb91a4c8-1a8a-46bf-86e6-e16950b67ef6",
                        "fullName": "Chet Holmgren",
                        "jerseyNumber": 7,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "62c44a90-f280-438a-9c7e-252f4f376283",
                        "fullName": "Jalen Williams",
                        "jerseyNumber": 8,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "sabonis-001",
                        "fullName": "Domantas Sabonis",
                        "jerseyNumber": 10,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "luka-001",
                        "fullName": "Luka Doncic",
                        "jerseyNumber": 77,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "giannis-001",
                        "fullName": "Giannis Antetokounmpo",
                        "jerseyNumber": 34,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    }
                ]
            },
            "awayTeam": {
                "teamId": "583ed102-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Denver Nuggets",
                "alias": "DEN",
                "players": [
                    {
                        "playerId": "f2625432-3903-4f90-9b0b-2e4f63856bb0",
                        "fullName": "Nikola Jokić",
                        "jerseyNumber": 15,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "685576ef-ea6c-4ccf-affd-18916baf4e60",
                        "fullName": "Jamal Murray",
                        "jerseyNumber": 27,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    },
                    {
                        "playerId": "20f85838-0bd5-4c1f-ab85-a308bafaf5bc",
                        "fullName": "Aaron Gordon",
                        "jerseyNumber": 32,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": []
                    }
                ]
            },
            "coverage": {
                "broadcasters": [
                    {
                        "name": "ABC",
                        "type": "tv"
                    }
                ]
            }
        }
        
        # Pruebas con jugadores reales del formato SportRadar
        test_players = [
            ("Domantas Sabonis", "ELITE - Double double machine, debería ser YES"),
            ("Luka Doncic", "ELITE - Triple doble machine, debería ser YES"),
            ("Giannis Antetokounmpo", "ELITE - MVP, debería ser YES"),
            ("Nikola Jokić", "ELITE - Triple doble machine, debería ser YES"),
            ("Shai Gilgeous-Alexander", "Muy bueno pero más scorer, podría ser NO")
        ]
        
        logger.info("🎯 Probando predicciones de double double:")
        for player_name, description in test_players:
            logger.info(f"\n   Prediciendo: {player_name} ({description})")
            
            result = predictor.predict_game(mock_sportradar_game, player_name)
            
            if result is not None and 'error' not in result and 'target_name' in result:
                logger.info(f"   ✅ Resultado (formato JSON exacto):")
                print(json.dumps(result, indent=6, ensure_ascii=False))
            elif result is not None and 'message' in result:
                logger.info(f"   ✅ Resultado:")
                logger.info(f"      {result['message']}")
            elif result is None:
                logger.info(f"   ⚠️ Jugador no disponible para predicción")
            else:
                logger.info(f"   ❌ Error esperado: {result.get('error', 'Error desconocido')}")
                if 'player_status' in result:
                    logger.info(f"      Estado del jugador: {result['player_status']}")
        
        logger.info("\n✅ Prueba completada")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
