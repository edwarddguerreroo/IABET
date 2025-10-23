#!/usr/bin/env python3
"""
WRAPPER FINAL UNIFICADO - PREDICCIÓN HALFTIME EQUIPOS
========================================

Wrapper final unificado para predicciones de halftime equipos que integra:
- Datos de SportRadar via GameDataAdapter
- Modelo halftime equipos completo con calibraciones elite
- Formato estándar para módulo de stacking
- Manejo robusto de errores y tolerancia conservadora

FLUJO INTEGRADO:
1. Recibir datos de SportRadar (game_data)
2. Convertir con GameDataAdapter
3. Buscar datos históricos específicos del equipo
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
from app.architectures.basketball.src.models.teams.halftime_points.model_halftime import HalfTimePointsModel
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter
from app.architectures.basketball.pipelines.predict.utils_predict.common_utils import CommonUtils
from app.architectures.basketball.pipelines.predict.utils_predict.confidence.confidence_teams import TeamsConfidence

logger = logging.getLogger(__name__)

class HalfTimeTeamsPointsPredictor:
    """
    Wrapper final unificado para predicciones halftime equipos
    
    Integra completamente con SportRadar via GameDataAdapter y proporciona
    una interfaz limpia para el módulo de stacking.
    """
    
    def __init__(self, teams_df: pd.DataFrame = None):
        """Inicializar el predictor puntos equipos unificado"""
        self.model = None
        self.historical_players = None
        self.historical_teams = teams_df
        self.game_adapter = GameDataAdapter()
        self.common_utils = CommonUtils()
        self.confidence_calculator = TeamsConfidence()  # Calculadora de confianza centralizada
        self.is_loaded = False
        self.base_tolerance = 0  # Tolerancia base más agresiva
        self.high_confidence_threshold = 75.0  # Umbral para alta confianza (más accesible)
        self.ultra_confidence_threshold = 85.0  # Umbral para ultra confianza (más accesible)
        
        # Cargar datos y modelo automáticamente
        self.load_data_and_model()
    
    def load_data_and_model(self) -> bool:
        """
        Cargar datos históricos y modelo entrenado de halftime
        
        Returns:
            True si se cargó exitosamente
        """
        try:
            
            # Cargar datos históricos con target de halftime
            data_loader = NBADataLoader(
                players_total_path="app/architectures/basketball/data/players_total.csv",
                players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                teams_total_path="app/architectures/basketball/data/teams_total.csv",
                teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                biometrics_path="app/architectures/basketball/data/biometrics.csv"
            )
            self.historical_players, self.historical_teams, self.historical_players_quarters, self.historical_teams_quarters = data_loader.load_data_with_halftime_target()
            
            # Cargar modelo halftime equipos usando joblib directo
            model_path = "app/architectures/basketball/.joblib/halftime_points_model.joblib"
            logger.info(f" Cargando modelo HALFTIME_POINTS completo desde: {model_path}")
            
            import joblib
            self.model = joblib.load(model_path)
            logger.info(" Modelo HALFTIME_POINTS cargado como objeto completo")
            
            self.is_loaded = True
            
            return True
            
        except Exception as e:
            logger.error(f" Error cargando datos y modelo de halftime: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_game(self, game_data: Dict[str, Any], target_team: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Metodo principal para predecir halftime equipos desde datos de insumo (SportRadar)
        
        Args:
            game_data: Datos del juego de SportRadar
            target_team: Nombre del equipo objetivo (opcional)
            
        Returns:
            - Si target_team especificado: Dict con predicción individual
            - Si target_team None: List[Dict] con DOS predicciones individuales (local y visitante)
        """
        if not self.is_loaded:
            return {'error': 'Modelo no cargado. Ejecutar load_data_and_model() primero.'}
        
        try:
            # Convertir datos de SportRadar con GameDataAdapter
            players_df, teams_df = self.game_adapter.convert_game_to_dataframes(game_data)
            
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
            home_team_abbr = self.common_utils._get_team_abbreviation(home_team_name)
            away_team_abbr = self.common_utils._get_team_abbreviation(away_team_name)

            # Si se especifica target_team, predecir solo ese equipo
            if target_team:
                logger.info(f" Prediciendo equipo específico: {target_team}")
                prediction = self._predict_single_team_from_game(game_data, target_team, home_team_name, away_team_name)
                
                if 'error' in prediction:
                    logger.error(f"Error en predicción: {prediction['error']}")
                    return prediction
                
                # Validar que la predicción contiene los campos necesarios (sin fallbacks)
                if 'bet_line' not in prediction:
                    logger.error(f"❌ Predicción de {target_team} no contiene 'bet_line'")
                    return {'error': f"Predicción de {target_team} sin 'bet_line'"}
                if 'confidence_percentage' not in prediction:
                    logger.error(f"❌ Predicción de {target_team} no contiene 'confidence_percentage'")
                    return {'error': f"Predicción de {target_team} sin 'confidence_percentage'"}
                
                # prediction_details puede ser vacío, pero debe existir como dict
                prediction_details = prediction.get('prediction_details')
                if prediction_details is None:
                    prediction_details = {}
                
                # Retornar predicción individual como diccionario
                return {
                    "home_team": home_team_name,
                    "away_team": away_team_name,
                    "target_type": "HT",
                    "target_name": target_team,
                    "bet_line": prediction['bet_line'],
                    "bet_type": "points",
                    "confidence_percentage": prediction['confidence_percentage'],
                    "prediction_details": prediction_details
                }
            else:
                # Si no se especifica, predecir AMBOS equipos
                logger.info(f" Prediciendo ambos equipos: {home_team_name} vs {away_team_name}")
                
                # Predecir equipo local
                home_prediction = self._predict_single_team_from_game(game_data, home_team_name, home_team_name, away_team_name)
                
                # Predecir equipo visitante  
                away_prediction = self._predict_single_team_from_game(game_data, away_team_name, home_team_name, away_team_name)
                
                # Manejar errores en predicciones individuales
                if 'error' in home_prediction:
                    logger.error(f"Error en predicción equipo local: {home_prediction['error']}")
                    return home_prediction
                if 'error' in away_prediction:
                    logger.error(f"Error en predicción equipo visitante: {away_prediction['error']}")
                    return away_prediction
                
                # Validar que las predicciones contienen los campos necesarios (sin fallbacks)
                if 'bet_line' not in home_prediction:
                    logger.error(f"❌ Predicción HT de {home_team_name} no contiene 'bet_line'")
                    return {'error': f"Predicción HT de {home_team_name} sin 'bet_line'"}
                if 'confidence_percentage' not in home_prediction:
                    logger.error(f"❌ Predicción HT de {home_team_name} no contiene 'confidence_percentage'")
                    return {'error': f"Predicción HT de {home_team_name} sin 'confidence_percentage'"}
                if 'bet_line' not in away_prediction:
                    logger.error(f"❌ Predicción HT de {away_team_name} no contiene 'bet_line'")
                    return {'error': f"Predicción HT de {away_team_name} sin 'bet_line'"}
                if 'confidence_percentage' not in away_prediction:
                    logger.error(f"❌ Predicción HT de {away_team_name} no contiene 'confidence_percentage'")
                    return {'error': f"Predicción HT de {away_team_name} sin 'confidence_percentage'"}
                
                # prediction_details puede ser vacío, pero debe existir como dict
                home_details = home_prediction.get('prediction_details')
                if home_details is None:
                    home_details = {}
                away_details = away_prediction.get('prediction_details')
                if away_details is None:
                    away_details = {}
                
                # Retornar AMBAS predicciones como LISTA de predicciones individuales
                predictions_list = [
                    # Predicción del equipo local
                    {
                        "home_team": home_team_name,
                        "away_team": away_team_name,
                        "target_type": "HT",
                        "target_name": home_team_name,
                        "bet_line": home_prediction['bet_line'],
                        "bet_type": "points",
                        "confidence_percentage": home_prediction['confidence_percentage'],
                        "prediction_details": home_details
                    },
                    # Predicción del equipo visitante
                    {
                        "home_team": home_team_name,
                        "away_team": away_team_name,
                        "target_type": "HT", 
                        "target_name": away_team_name,
                        "bet_line": away_prediction['bet_line'],
                        "bet_type": "points",
                        "confidence_percentage": away_prediction['confidence_percentage'],
                        "prediction_details": away_details
                    }
                ]
                
                return predictions_list
            
        except Exception as e:
            logger.error(f" Error en predicción desde SportRadar: {e}")
            return {'error': f'Error procesando datos de SportRadar: {str(e)}'}

    def _predict_single_team_from_game(self, game_data: Dict[str, Any], target_team: str, 
                                     home_team_name: str, away_team_name: str) -> Dict[str, Any]:
        """
        Método auxiliar para predecir un equipo específico desde datos de juego
        
        Args:
            game_data: Datos completos del juego
            target_team: Nombre del equipo a predecir
            home_team_name: Nombre del equipo local
            away_team_name: Nombre del equipo visitante
            
        Returns:
            Diccionario con predicción del equipo específico
        """
        try:

            # Convertir nombre del equipo objetivo a abreviación para búsqueda
            target_team_abbr = self.common_utils._get_team_abbreviation(target_team)
            target_row = self.common_utils._smart_team_search(self.historical_teams, target_team_abbr)
            
            if target_row.empty:
                available_teams = list(self.historical_teams['Team'].unique())
                logger.warning(f" Equipo no encontrado: {target_team}")
                return {
                    'error': f'Equipo "{target_team}" no encontrado',
                    'available_teams': available_teams,
                    'message': 'Equipos disponibles en el dataset histórico'
                }
            
            # Extraer datos del equipo
            team_data = target_row.iloc[0].to_dict()
            
            # Extraer información adicional desde SportRadar
            is_home = self.common_utils._get_is_home_team_from_sportradar(game_data, target_team)
            
            # Agregar información extraída al team_data
            team_data['is_home'] = is_home
            
            # Corregir formato de fecha para evitar problemas de timezone
            if 'Date' in team_data and pd.notna(team_data['Date']):
                # Convertir a string sin timezone para compatibilidad
                if hasattr(team_data['Date'], 'strftime'):
                    team_data['Date'] = team_data['Date'].strftime('%Y-%m-%d')
                elif hasattr(team_data['Date'], 'date'):
                    team_data['Date'] = str(team_data['Date'].date())
            
            # Hacer predicción usando el método interno (pasando game_data también)
            prediction_result = self.predict_single_team(team_data, game_data)
            
            if 'error' in prediction_result:
                return prediction_result
            
            # La predicción del modelo ES el bet_line
            prediction_value = prediction_result['halftime_prediction']
            confidence_value = prediction_result['confidence_percentage']
            
            return {
                "home_team": home_team_name,
                "away_team": away_team_name,
                "target_type": "HT",
                "target_name": target_team,
                "bet_line": str(prediction_value),  
                "bet_type": "points",
                "confidence_percentage": confidence_value,
                "prediction_details": prediction_result.get('prediction_details', {})
            }
            
        except Exception as e:
            logger.error(f" Error en predicción de equipo individual: {e}")
            return {'error': f'Error procesando equipo {target_team}: {str(e)}'}
    
    def predict_single_team(self, team_data: Dict[str, Any], game_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predecir puntos para un equipo individual
        
        Args:
            team_data: Diccionario con datos del equipo
                - Team: Nombre del equipo
                - Opp: Nombre del equipo oponente
                - Date: Fecha del juego (opcional)
                
        Returns:
            Diccionario con predicción y metadata
        """
        if not self.is_loaded:
            raise ValueError("Modelo no cargado. Ejecutar load_data_and_model() primero.")
        
        try:
            
            # PASO CRÍTICO: Buscar datos históricos del equipo específico (ÚLTIMOS 30 PARTIDOS)
            team_name = team_data.get('Team', 'Unknown')
            
            # Convertir nombre del equipo a abreviación para búsqueda
            team_name_abbr = self.common_utils._get_team_abbreviation(team_name)
            team_historical_full = self.common_utils._smart_team_search(self.historical_teams, team_name_abbr)
            
            if len(team_historical_full) == 0:
                logger.warning(f" No se encontraron datos históricos para {team_name}")
                # Usar datos de equipos similares o promedio (últimos 50)
                team_historical = self.historical_teams.head(100).tail(50).copy()
                logger.info(f" Usando datos de referencia: {len(team_historical)} registros")
            else:
                # USAR TODOS LOS JUEGOS DISPONIBLES para máxima precisión
                team_historical = team_historical_full.copy()
                total_available = len(team_historical_full)
                used_games = len(team_historical)
                logger.info(f" {team_name}: {used_games} juegos históricos disponibles (TODOS)")
                
                # Si tiene menos de 10 juegos, advertir pero continuar
                if used_games < 10:
                    logger.warning(f" Pocos datos recientes para {team_name}: solo {used_games} juegos")
            
            # Usar solo datos históricos para predicción (sin fila artificial)
            combined_df = team_historical.copy()
            
            # Hacer predicción
            predictions = self.model.predict(combined_df)

            # ESTABILIDAD 1: Usar promedio de las últimas predicciones basadas en histórico
            if len(predictions) > 0:
                # Tomar promedio de las últimas 5 predicciones para mayor estabilidad
                recent_predictions = predictions[-5:] if len(predictions) >= 5 else predictions
                raw_prediction = np.mean(recent_predictions)
                prediction_std = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
            else:
                raw_prediction = 110  # Valor por defecto para equipos NBA
                prediction_std = 5  # Desviación estándar por defecto
            
            # ESTABILIDAD 2: Usar promedio de los últimos resultados reales de halftime del equipo
            if 'HT' in team_historical.columns and len(team_historical) > 0:
                # Usar últimos 5 juegos o todos si hay menos (máximo eficiencia con datos limitados)
                games_to_use = min(5, len(team_historical))
                recent_actual_ht = team_historical['HT'].tail(games_to_use).values
                actual_ht_mean = np.mean(recent_actual_ht)
                actual_ht_std = np.std(recent_actual_ht) if len(recent_actual_ht) > 1 else 0
                
                logger.info(f" Estabilización histórica: últimos {games_to_use} juegos, promedio {actual_ht_mean:.1f} HT")
            else:
                actual_ht_mean = raw_prediction
                actual_ht_std = prediction_std
            
            # FACTOR JUGADORES ESTRELLA - Ajustar predicción basada en ausencias del equipo
            # Extraer oponente REAL del juego actual desde game_data, NO de datos históricos (sin fallbacks)
            if not game_data:
                logger.error("❌ game_data es requerido para determinar oponente")
                return {'error': 'game_data es requerido para determinar oponente'}
            
            home_team_name_from_game = game_data['homeTeam'].get('name')
            away_team_name_from_game = game_data['awayTeam'].get('name')
            
            if not home_team_name_from_game or not away_team_name_from_game:
                logger.error("❌ No se pudieron obtener nombres de equipos desde game_data")
                return {'error': 'No se pudieron obtener nombres de equipos desde game_data'}
            
            home_team_abbr_from_game = self.common_utils._get_team_abbreviation(home_team_name_from_game)
            away_team_abbr_from_game = self.common_utils._get_team_abbreviation(away_team_name_from_game)
            
            # Determinar el oponente correcto (usar team_name_abbr en lugar de team_name)
            if team_name_abbr == home_team_abbr_from_game:
                opponent_name = away_team_abbr_from_game
                opponent_id = game_data['awayTeam'].get('teamId', '')
            elif team_name_abbr == away_team_abbr_from_game:
                opponent_name = home_team_abbr_from_game
                opponent_id = game_data['homeTeam'].get('teamId', '')
            else:
                logger.error(f"❌ Equipo {team_name_abbr} no coincide con equipos en game_data ({home_team_abbr_from_game} vs {away_team_abbr_from_game})")
                return {'error': f'Equipo {team_name_abbr} no encontrado en game_data'}
            
            star_player_factor = self.confidence_calculator.calculate_star_player_factor_teams_points(
                team_name=team_name,
                opponent_name=opponent_name,
                game_data=game_data
            )
            
            # ANÁLISIS HEAD-TO-HEAD - Estadísticas de enfrentamientos directos del equipo vs oponente (HALFTIME)
            h2h_stats = self.confidence_calculator.calculate_h2h_factor_halftime(team_name, opponent_name, self.historical_teams)
            
            # CALCULAR ESTADÍSTICAS H2H ADICIONALES (std, min, max) para historical_context
            # Usar todos los juegos H2H disponibles (máximo 10 más recientes)
            if h2h_stats and h2h_stats.get('games_found', 0) > 0:
                # Obtener juegos H2H del equipo específico
                h2h_games = self.historical_teams[
                    ((self.historical_teams['Team'] == team_name) & (self.historical_teams['Opp'] == opponent_name)) |
                    ((self.historical_teams['Team'] == opponent_name) & (self.historical_teams['Opp'] == team_name))
                ]
                
                if len(h2h_games) > 0:
                    # Filtrar por el equipo específico que estamos prediciendo
                    team_h2h_games = h2h_games[h2h_games['Team'] == team_name]
                    
                    if len(team_h2h_games) > 0 and 'HT' in team_h2h_games.columns:
                        # Ordenar por fecha y tomar los últimos 10 juegos más recientes
                        if 'Date' in team_h2h_games.columns:
                            team_h2h_games = team_h2h_games.sort_values('Date', ascending=False).head(10)
                        else:
                            team_h2h_games = team_h2h_games.tail(10)
                        
                        h2h_ht_values = team_h2h_games['HT'].dropna()
                        
                        if len(h2h_ht_values) > 0:
                            # Agregar std, min, max a h2h_stats (usar todos los disponibles)
                            h2h_stats['h2h_std'] = round(h2h_ht_values.std(), 1) if len(h2h_ht_values) > 1 else 0.0
                            h2h_stats['h2h_min'] = int(h2h_ht_values.min())
                            h2h_stats['h2h_max'] = int(h2h_ht_values.max())
                            logger.info(f"✅ H2H stats calculados para {team_name} ({len(h2h_ht_values)} juegos): std={h2h_stats['h2h_std']}, min={h2h_stats['h2h_min']}, max={h2h_stats['h2h_max']}")
                        else:
                            logger.warning(f"⚠️ No hay valores HT válidos para calcular std/min/max para {team_name}")
                    else:
                        logger.warning(f"⚠️ No hay juegos H2H de {team_name} o columna 'HT' no existe")
                else:
                    logger.warning(f"⚠️ No se encontraron juegos H2H entre {team_name} y {opponent_name}")
            
            # APLICAR FACTORES A LA PREDICCIÓN
            # Aplicar factor de jugadores estrella
            raw_prediction_adjusted = raw_prediction * star_player_factor
            
            # Aplicar factor H2H si hay datos suficientes (HALFTIME) - sin fallbacks
            if h2h_stats and h2h_stats.get('games_found', 0) >= 2:  # Menos juegos necesarios para halftime
                # Validar que h2h_stats contiene los campos necesarios
                if 'h2h_factor' not in h2h_stats or 'consistency_score' not in h2h_stats:
                    logger.warning(f"⚠️ H2H stats incompletos para {team_name} vs {opponent_name}, omitiendo H2H adjustment")
                    actual_ht_mean_adjusted = actual_ht_mean
                    logger.info(f" H2H incompleto, usando solo modelo y star factor")
                elif 'halftime_mean' not in h2h_stats:
                    logger.warning(f"⚠️ H2H stats sin 'halftime_mean' para {team_name} vs {opponent_name}, omitiendo H2H adjustment")
                    actual_ht_mean_adjusted = actual_ht_mean
                    logger.info(f" H2H sin halftime_mean, usando solo modelo y star factor")
                else:
                    h2h_factor = h2h_stats['h2h_factor']
                    consistency_score = h2h_stats['consistency_score']
                    
                    # Para halftime, usar factor H2H más conservador y umbrales más estrictos
                    if consistency_score >= 80 and h2h_stats.get('games_found', 0) >= 3:  # Más estricto
                        # H2H muy consistente: dar más peso al histórico
                        h2h_blend_weight = 0.3  # 30% H2H, 70% modelo (más conservador para halftime)
                        actual_ht_mean_adjusted = actual_ht_mean * h2h_factor
                        logger.info(f" H2H muy consistente ({consistency_score:.1f}%): blend 30% H2H")
                    else:
                        # H2H menos consistente: factor mínimo
                        h2h_blend_weight = 0.15  # 15% H2H, 85% modelo (más conservador)
                        actual_ht_mean_adjusted = actual_ht_mean * min(h2h_factor, 1.10)  # Limitar factor H2H más
                        logger.info(f" H2H consistencia normal ({consistency_score:.1f}%): blend 15% H2H")
                    
                    # Combinar predicción ajustada con H2H
                    raw_prediction_adjusted = (
                        raw_prediction_adjusted * (1 - h2h_blend_weight) + 
                        h2h_stats['halftime_mean'] * h2h_blend_weight
                    )
            else:
                actual_ht_mean_adjusted = actual_ht_mean
                logger.info(f" Sin datos H2H suficientes, usando solo modelo y star factor")
            
            # CALCULAR CONFIANZA PRELIMINAR PARA DETERMINAR ESTRATEGIA
            preliminary_confidence = self.confidence_calculator.calculate_halftime_confidence(
                raw_prediction=raw_prediction_adjusted,  # Usar predicción ajustada
                stabilized_prediction=raw_prediction_adjusted,  # Usar predicción ajustada para cálculo inicial
                tolerance=self.base_tolerance,
                prediction_std=prediction_std,
                actual_ht_std=actual_ht_std,
                historical_games=len(team_historical),
                team_data=team_data
            )
            
            # SISTEMA ADAPTATIVO BASADO EN CONFIANZA PARA 95%+ EFECTIVIDAD
            # Estrategia fija: 80% modelo + 20% histórico + tolerancia
            tolerance_used = self.base_tolerance  # -1
            actual_mean = actual_ht_mean_adjusted if 'actual_ht_mean_adjusted' in locals() else actual_ht_mean
            final_prediction = (raw_prediction_adjusted * 0.80) + (actual_mean * 0.20) + tolerance_used
            
            halftime_prediction = final_prediction  # Usar la predicción directa del modelo
            
            # RECALCULAR CONFIANZA CON VALORES FINALES USANDO CLASE CENTRALIZADA
            confidence_percentage = self.confidence_calculator.calculate_halftime_confidence(
                raw_prediction=raw_prediction_adjusted,  # Usar predicción ajustada
                stabilized_prediction=final_prediction,
                tolerance=tolerance_used,
                prediction_std=prediction_std,
                actual_ht_std=actual_ht_std,
                historical_games=len(team_historical),
                team_data=team_data
            )
                
            # CALCULAR ESTADÍSTICAS DETALLADAS PARA PREDICTION_DETAILS (HALFTIME)
            # Últimos 5 juegos
            last_5_games = team_historical.tail(5)['HT'] if len(team_historical) >= 5 else team_historical['HT']
            last_5_stats = {
                'mean': round(last_5_games.mean(), 1) if len(last_5_games) > 0 else 0,
                'std': round(last_5_games.std(), 1) if len(last_5_games) > 1 else 0,
                'min': int(last_5_games.min()) if len(last_5_games) > 0 else 0,
                'max': int(last_5_games.max()) if len(last_5_games) > 0 else 0,
                'count': len(last_5_games)
            }
            
            # Últimos 10 juegos
            last_10_games = team_historical.tail(10)['HT'] if len(team_historical) >= 10 else team_historical['HT']
            last_10_stats = {
                'mean': round(last_10_games.mean(), 1) if len(last_10_games) > 0 else 0,
                'std': round(last_10_games.std(), 1) if len(last_10_games) > 1 else 0,
                'min': int(last_10_games.min()) if len(last_10_games) > 0 else 0,
                'max': int(last_10_games.max()) if len(last_10_games) > 0 else 0,
                'count': len(last_10_games)
            }
            
            # Análisis de tendencia
            if len(team_historical) >= 5:
                recent_5_mean = last_5_games.mean()
                recent_10_mean = last_10_games.mean() if len(team_historical) >= 10 else recent_5_mean
                trend_5_games = recent_5_mean - recent_10_mean
            else:
                trend_5_games = 0
                recent_5_mean = actual_ht_mean
            
            # Score de consistencia (inverso de la desviación estándar)
            consistency_score = max(0, 100 - (actual_ht_std * 4)) if actual_ht_std > 0 else 100  # Más estricto para halftime
            
            # Forma reciente (promedio de últimos 3 juegos)
            recent_form = team_historical.tail(3)['HT'].mean() if len(team_historical) >= 3 else actual_ht_mean
            
            # Obtener team_id correcto (usar team_name_abbr)
            if game_data:
                if team_name_abbr == home_team_abbr_from_game:
                    team_id_final = game_data.get('homeTeam', {}).get('teamId', self.common_utils._get_team_id(team_name))
                elif team_name_abbr == away_team_abbr_from_game:
                    team_id_final = game_data.get('awayTeam', {}).get('teamId', self.common_utils._get_team_id(team_name))
                else:
                    team_id_final = self.common_utils._get_team_id(team_name)
            else:
                team_id_final = self.common_utils._get_team_id(team_name)
            
            return {
                'halftime_prediction': int(halftime_prediction),
                'confidence_percentage': round(confidence_percentage, 1),
                'prediction_details': {
                    'team': team_name,
                    'team_id': team_id_final,
                    'opponent': opponent_name,
                    'opponent_id': opponent_id if game_data else self.common_utils._get_team_id(team_data.get('Opp', 'Unknown')),
                    'tolerance_applied': tolerance_used,
                    'historical_games_used': len(team_historical),
                    'raw_prediction': round(raw_prediction_adjusted, 1),
                    'h2h_adjusted_prediction': round(raw_prediction_adjusted, 1),
                    'final_prediction': int(halftime_prediction),
                    'actual_ht_mean': round(actual_ht_mean, 1),
                    'actual_ht_std': round(actual_ht_std, 1),
                    'prediction_std': round(prediction_std, 1),
                    'last_5_games': last_5_stats,
                    'last_10_games': last_10_stats,
                    'trend_analysis': {
                        'trend_5_games': round(trend_5_games, 1),
                        'consistency_score': round(consistency_score, 1),
                        'recent_form': round(recent_form, 1)
                    },
                    'performance_metrics': {
                        'stabilized_prediction': round(final_prediction, 1),
                        'confidence_factors': {
                            'tolerance': tolerance_used,
                            'historical_games': len(team_historical),
                            'data_quality': 'high' if len(team_historical) >= 20 else 'medium'
                        }
                    },
                    'h2h_stats': {
                        'games_found': h2h_stats.get('games_found', 0) if h2h_stats else 0,
                        'halftime_mean': round(h2h_stats.get('halftime_mean', 0), 1) if h2h_stats else None,
                        'team_points_mean': round(h2h_stats.get('halftime_mean', 0), 1) if h2h_stats else None,
                        'last_5_mean': round(h2h_stats.get('last_5_mean', 0), 1) if h2h_stats and h2h_stats.get('last_5_mean') else None,
                        'last_10_mean': round(h2h_stats.get('last_10_mean', 0), 1) if h2h_stats and h2h_stats.get('last_10_mean') else None,
                        'h2h_factor': round(h2h_stats.get('h2h_factor', 1.0), 3) if h2h_stats else 1.0,
                        'consistency_score': round(h2h_stats.get('consistency_score', 0), 1) if h2h_stats else 0,
                        'h2h_std': round(h2h_stats.get('h2h_std', 0), 1) if h2h_stats and h2h_stats.get('h2h_std') is not None else None,
                        'h2h_min': int(h2h_stats.get('h2h_min', 0)) if h2h_stats and h2h_stats.get('h2h_min') is not None else None,
                        'h2h_max': int(h2h_stats.get('h2h_max', 0)) if h2h_stats and h2h_stats.get('h2h_max') is not None else None
                    }
                }
            }
                
        except Exception as e:
                logger.error(f" Error en predicción: {e}")
                import traceback
                traceback.print_exc()
                return {
                    'team': team_data.get('Team', 'Unknown'),
                    'error': str(e),
                    'halftime_prediction': None
                }
                


def test_halftime_teams_points_predictor():
    """Función de prueba rápida del predictor de halftime por equipo"""
    print("="*80)
    print(" PROBANDO HALFTIME TEAMS POINTS PREDICTOR - KNICKS VS CAVALIERS")
    print("="*80)
    
    # Inicializar predictor
    predictor = HalfTimeTeamsPointsPredictor()
    
    # Cargar datos y modelo
    print("\n Cargando datos y modelo...")
    if not predictor.load_data_and_model():
        print(" Error cargando modelo")
        return False
    
    print("\n[OK] Modelo cargado exitosamente")
    
    # Prueba con datos simulados de SportRadar
    print("\n" + "="*80)
    print(" PRUEBA: KNICKS VS CAVALIERS - PUNTOS MEDIO TIEMPO POR EQUIPO")
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
    
    # Probar predicción para ambos equipos
    print("\nPrediciendo puntos HT para ambos equipos:")
    print("-" * 60)
    sportradar_result = predictor.predict_game(mock_sportradar_game)
    
    import json
    
    if isinstance(sportradar_result, list) and len(sportradar_result) == 2:
        print(f"\n[OK] PREDICCIONES EXITOSAS - {len(sportradar_result)} equipos")
        
        # Predicción 1: Equipo local
        home_pred = sportradar_result[0]
        print(f"\n1. EQUIPO LOCAL - {home_pred['target_name']}:")
        print(f"   Bet Line: {home_pred['bet_line']} puntos HT")
        print(f"   Confidence: {home_pred['confidence_percentage']}%")
        print(f"   Bet Type: {home_pred['bet_type']}")
        
        # Predicción 2: Equipo visitante
        away_pred = sportradar_result[1]
        print(f"\n2. EQUIPO VISITANTE - {away_pred['target_name']}:")
        print(f"   Bet Line: {away_pred['bet_line']} puntos HT")
        print(f"   Confidence: {away_pred['confidence_percentage']}%")
        print(f"   Bet Type: {away_pred['bet_type']}")
        
        # Análisis comparativo
        home_line = int(home_pred['bet_line'])
        away_line = int(away_pred['bet_line'])
        total_combined = home_line + away_line
        
        print(f"\nANALISIS COMPARATIVO:")
        print(f"  Total combinado HT: {total_combined} puntos ({home_line} + {away_line})")
        print(f"  Diferencia estimada: {abs(home_line - away_line)} puntos")
        
        if home_line > away_line:
            print(f"  Favorito HT: {home_pred['target_name']} (+{home_line - away_line})")
        elif away_line > home_line:
            print(f"  Favorito HT: {away_pred['target_name']} (+{away_line - home_line})")
        else:
            print(f"  Predicciones equilibradas")
        
        print(f"\nJSON COMPLETO:")
        for idx, result in enumerate(sportradar_result):
            print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")
    else:
        print(f"\n[ERROR] No se pudieron generar predicciones")
        if isinstance(sportradar_result, dict) and 'error' in sportradar_result:
            print(f"Error: {sportradar_result['error']}")
        else:
            print(f"Resultado inesperado: {sportradar_result}")
    
    print("\n" + "="*80)
    print(" PRUEBA COMPLETADA")
    print("="*80)
    return True


if __name__ == "__main__":
    test_halftime_teams_points_predictor()