"""
SISTEMA AVANZADO DE INTEGRACIÓN MONTECARLO + MODELOS ESPECIALIZADOS
=================================================================

Este módulo integra las simulaciones de Montecarlo con TODOS los modelos
especializados del sistema para lograr precisión extrema.
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import warnings
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Imports de modelos especializados (modo lazy loading)
# Los imports se harán dinámicamente cuando sea necesario para evitar errores de inicialización

class MultiDimensionalValidator:
    """
    Validador Multi-Dimensional para Simulaciones NBA
    ===============================================
    
    Sistema avanzado que evalúa precisión en múltiples dimensiones:
    - Ganador correcto (25%)
    - Exactitud de puntuación (30%) 
    - Margen de victoria (20%)
    - Total de puntos (15%)
    - Intervalos de confianza (10%)
    """
    
    def __init__(self):
        self.validation_weights = {
            'winner_accuracy': 0.25,      # Ganador correcto
            'score_accuracy': 0.30,       # Exactitud de puntuación
            'margin_accuracy': 0.20,      # Margen de victoria
            'total_accuracy': 0.15,       # Total de puntos
            'confidence_accuracy': 0.10   # Intervalos de confianza
        }
        
        self.tolerance_thresholds = {
            'score_tolerance': 5.0,        # ±5 puntos por equipo
            'margin_tolerance': 8.0,       # ±8 puntos margen
            'total_tolerance': 12.0,       # ±12 puntos total
            'confidence_threshold': 0.80   # 80% confianza mínima
        }
        
        self.logger = logging.getLogger(__name__)
        
    def validate_simulation(self, simulation_result: dict, actual_result: dict) -> dict:
        """
        Valida una simulación individual contra resultado real
        
        Args:
            simulation_result: Resultado de simulación Monte Carlo
            actual_result: Resultado real del partido
            
        Returns:
            Dict con métricas de validación multi-dimensional
        """
        try:
            validation_metrics = {}
            
            # 1. VALIDACIÓN DE GANADOR (25%)
            winner_score = self._validate_winner(simulation_result, actual_result)
            validation_metrics['winner_accuracy'] = winner_score
            
            # 2. VALIDACIÓN DE PUNTUACIÓN (30%)
            score_accuracy = self._validate_scores(simulation_result, actual_result)
            validation_metrics['score_accuracy'] = score_accuracy
            
            # 3. VALIDACIÓN DE MARGEN (20%)
            margin_accuracy = self._validate_margin(simulation_result, actual_result)
            validation_metrics['margin_accuracy'] = margin_accuracy
            
            # 4. VALIDACIÓN DE TOTAL (15%)
            total_accuracy = self._validate_total_points(simulation_result, actual_result)
            validation_metrics['total_accuracy'] = total_accuracy
            
            # 5. VALIDACIÓN DE CONFIANZA (10%)
            confidence_accuracy = self._validate_confidence_intervals(simulation_result, actual_result)
            validation_metrics['confidence_accuracy'] = confidence_accuracy
            
            # PUNTUACIÓN FINAL PONDERADA
            overall_score = sum(
                validation_metrics[metric] * self.validation_weights[metric]
                for metric in validation_metrics
            )
            
            validation_metrics['overall_accuracy'] = overall_score
            validation_metrics['validation_breakdown'] = {
                'winner': f"{winner_score:.1%}",
                'scores': f"{score_accuracy:.1%}",
                'margin': f"{margin_accuracy:.1%}",
                'total': f"{total_accuracy:.1%}",
                'confidence': f"{confidence_accuracy:.1%}"
            }
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"Error en validación: {str(e)}")
            return {
                'overall_accuracy': 0.0,
                'winner_accuracy': 0.0,
                'score_accuracy': 0.0,
                'margin_accuracy': 0.0,
                'total_accuracy': 0.0,
                'confidence_accuracy': 0.0,
                'error': str(e)
            }
    
    def _validate_winner(self, sim_result: dict, actual_result: dict) -> float:
        """Valida la predicción del ganador"""
        try:
            # Obtener ganador simulado
            sim_home_score = sim_result['score_predictions']['home_score']['mean']
            sim_away_score = sim_result['score_predictions']['away_score']['mean']
            sim_winner = 'home' if sim_home_score > sim_away_score else 'away'
            
            # Obtener ganador real
            actual_home_score = actual_result.get('home_score', 0)
            actual_away_score = actual_result.get('away_score', 0)
            actual_winner = 'home' if actual_home_score > actual_away_score else 'away'
            
            # Evaluar con probabilidad de victoria
            win_prob = sim_result['win_probabilities'].get(f'{actual_winner}_win', 0.5)
            
            if sim_winner == actual_winner:
                # Ganador correcto - puntuación basada en confianza
                return min(1.0, 0.5 + win_prob)
            else:
                # Ganador incorrecto - penalización basada en confianza
                return max(0.0, 0.5 - win_prob)
                
        except Exception:
            return 0.0
    
    def _validate_scores(self, sim_result: dict, actual_result: dict) -> float:
        """Valida la exactitud de las puntuaciones individuales"""
        try:
            sim_home = sim_result['score_predictions']['home_score']['mean']
            sim_away = sim_result['score_predictions']['away_score']['mean']
            
            actual_home = actual_result.get('home_score', 0)
            actual_away = actual_result.get('away_score', 0)
            
            # Calcular errores absolutos
            home_error = abs(sim_home - actual_home)
            away_error = abs(sim_away - actual_away)
            
            # Calcular precisión para cada equipo
            home_accuracy = max(0.0, 1.0 - (home_error / self.tolerance_thresholds['score_tolerance']))
            away_accuracy = max(0.0, 1.0 - (away_error / self.tolerance_thresholds['score_tolerance']))
            
            # Promedio ponderado con bonificación por exactitud perfecta
            avg_accuracy = (home_accuracy + away_accuracy) / 2
            
            # Bonificación por exactitud excepcional (±2 puntos)
            if home_error <= 2 and away_error <= 2:
                avg_accuracy = min(1.0, avg_accuracy * 1.2)
            
            return avg_accuracy
            
        except Exception:
            return 0.0
    
    def _validate_margin(self, sim_result: dict, actual_result: dict) -> float:
        """Valida la exactitud del margen de victoria"""
        try:
            sim_home = sim_result['score_predictions']['home_score']['mean']
            sim_away = sim_result['score_predictions']['away_score']['mean']
            sim_margin = sim_home - sim_away
            
            actual_home = actual_result.get('home_score', 0)
            actual_away = actual_result.get('away_score', 0)
            actual_margin = actual_home - actual_away
            
            margin_error = abs(sim_margin - actual_margin)
            
            # Precisión del margen
            margin_accuracy = max(0.0, 1.0 - (margin_error / self.tolerance_thresholds['margin_tolerance']))
            
            # Bonificación por predicción de juego cerrado/blowout correcto
            sim_close = abs(sim_margin) <= 5
            actual_close = abs(actual_margin) <= 5
            sim_blowout = abs(sim_margin) >= 15
            actual_blowout = abs(actual_margin) >= 15
            
            if (sim_close and actual_close) or (sim_blowout and actual_blowout):
                margin_accuracy = min(1.0, margin_accuracy * 1.15)
            
            return margin_accuracy
            
        except Exception:
            return 0.0
    
    def _validate_total_points(self, sim_result: dict, actual_result: dict) -> float:
        """Valida la exactitud del total de puntos"""
        try:
            sim_total = (sim_result['score_predictions']['home_score']['mean'] + 
                        sim_result['score_predictions']['away_score']['mean'])
            
            actual_total = actual_result.get('home_score', 0) + actual_result.get('away_score', 0)
            
            total_error = abs(sim_total - actual_total)
            
            # Precisión del total
            total_accuracy = max(0.0, 1.0 - (total_error / self.tolerance_thresholds['total_tolerance']))
            
            # Bonificación por predicción de estilo de juego correcto
            sim_high_scoring = sim_total >= 220
            actual_high_scoring = actual_total >= 220
            sim_defensive = sim_total <= 200
            actual_defensive = actual_total <= 200
            
            if (sim_high_scoring and actual_high_scoring) or (sim_defensive and actual_defensive):
                total_accuracy = min(1.0, total_accuracy * 1.1)
            
            return total_accuracy
            
        except Exception:
            return 0.0
    
    def _validate_confidence_intervals(self, sim_result: dict, actual_result: dict) -> float:
        """Valida si los resultados reales caen dentro de intervalos de confianza"""
        try:
            confidence_score = 0.0
            validations = 0
            
            # Validar home score en intervalo de confianza
            home_pred = sim_result['score_predictions']['home_score']
            home_mean = home_pred['mean']
            home_std = home_pred['std']
            actual_home = actual_result.get('home_score', 0)
            
            # Calcular Z-score
            if home_std > 0:
                home_z = abs(actual_home - home_mean) / home_std
                if home_z <= 1.96:  # 95% confianza
                    confidence_score += 1.0
                elif home_z <= 1.28:  # 80% confianza
                    confidence_score += 0.8
                elif home_z <= 0.67:  # 50% confianza
                    confidence_score += 0.5
                validations += 1
            
            # Validar away score en intervalo de confianza
            away_pred = sim_result['score_predictions']['away_score']
            away_mean = away_pred['mean']
            away_std = away_pred['std']
            actual_away = actual_result.get('away_score', 0)
            
            if away_std > 0:
                away_z = abs(actual_away - away_mean) / away_std
                if away_z <= 1.96:  # 95% confianza
                    confidence_score += 1.0
                elif away_z <= 1.28:  # 80% confianza
                    confidence_score += 0.8
                elif away_z <= 0.67:  # 50% confianza
                    confidence_score += 0.5
                validations += 1
            
            return confidence_score / max(1, validations)
            
        except Exception:
            return 0.0
    
    def validate_batch(self, simulation_results: List[dict], actual_results: List[dict]) -> dict:
        """
        Valida un lote de simulaciones contra resultados reales
        
        Args:
            simulation_results: Lista de resultados de simulación
            actual_results: Lista de resultados reales
            
        Returns:
            Dict con métricas agregadas de validación
        """
        if len(simulation_results) != len(actual_results):
            raise ValueError("Las listas de simulaciones y resultados reales deben tener la misma longitud")
        
        batch_metrics = {
            'overall_accuracy': [],
            'winner_accuracy': [],
            'score_accuracy': [],
            'margin_accuracy': [],
            'total_accuracy': [],
            'confidence_accuracy': []
        }
        
        detailed_results = []
        
        for sim_result, actual_result in zip(simulation_results, actual_results):
            validation = self.validate_simulation(sim_result, actual_result)
            
            for metric in batch_metrics:
                if metric in validation:
                    batch_metrics[metric].append(validation[metric])
            
            detailed_results.append(validation)
        
        # Calcular estadísticas agregadas
        aggregated_metrics = {}
        for metric, values in batch_metrics.items():
            if values:
                aggregated_metrics[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Métricas de rendimiento adicionales
        overall_scores = batch_metrics['overall_accuracy']
        if overall_scores:
            aggregated_metrics['performance_summary'] = {
                'excellent_predictions': sum(1 for score in overall_scores if score >= 0.85) / len(overall_scores),
                'good_predictions': sum(1 for score in overall_scores if score >= 0.70) / len(overall_scores),
                'poor_predictions': sum(1 for score in overall_scores if score < 0.50) / len(overall_scores),
                'average_accuracy': np.mean(overall_scores),
                'improvement_needed': 0.85 - np.mean(overall_scores) if np.mean(overall_scores) < 0.85 else 0.0
            }
        
        return {
            'aggregated_metrics': aggregated_metrics,
            'detailed_results': detailed_results,
            'validation_summary': {
                'total_simulations': len(simulation_results),
                'average_overall_accuracy': aggregated_metrics.get('overall_accuracy', {}).get('mean', 0.0),
                'recommendation': self._generate_improvement_recommendation(aggregated_metrics)
            }
        }
    
    def _generate_improvement_recommendation(self, metrics: dict) -> str:
        """Genera recomendaciones específicas para mejorar el modelo"""
        overall_acc = metrics.get('overall_accuracy', {}).get('mean', 0.0)
        
        if overall_acc >= 0.85:
            return "Modelo funcionando excelentemente. Considerar ajuste fino para casos extremos."
        elif overall_acc >= 0.70:
            return "Modelo en buen estado. Focar en mejorar intervalos de confianza y precisión de puntuación."
        elif overall_acc >= 0.55:
            return "Modelo necesita mejoras significativas. Priorizar calibración de distribuciones y features."
        else:
            return "Modelo requiere reconstrucción. Revisar datos de entrada, features y arquitectura."


class AdvancedMonteCarloIntegrator:
    """
    Integrador avanzado que combina simulaciones de Montecarlo 
    con predicciones de modelos especializados para máxima precisión
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models_loaded = False
        self.specialized_models = {}
        self.confidence_weights = {
            'pts_model': 0.95,      # Muy alta confianza en puntos
            'ast_model': 0.92,      # Alta confianza en asistencias  
            'trb_model': 0.88,      # Alta confianza en rebotes
            'triples_model': 0.85,  # Buena confianza en triples
            'dd_model': 0.90,       # Muy buena confianza en double-doubles
            'win_model': 0.93,      # Muy alta confianza en victorias
            'teams_pts_model': 0.89, # Buena confianza en puntos de equipo
            'total_pts_model': 0.91  # Muy buena confianza en totales
        }
        
    def load_specialized_models(self):
        """Carga todos los modelos especializados (modo lazy loading)"""
        try:
            self.logger.info("Inicializando sistema de modelos especializados...")
            
            # MODO LAZY LOADING: Solo marcar como disponible, cargar cuando se necesite
            self.available_models = {
                'pts_model': {
                    'module': 'src.models.players.pts.model_pts',
                    'class_name': 'StackingPTSModel',
                    'loaded': False
                },
                'ast_model': {
                    'module': 'src.models.players.ast.model_ast',
                    'class_name': 'StackingASTModel',
                    'loaded': False
                },
                'trb_model': {
                    'module': 'src.models.players.trb.model_trb',
                    'class_name': 'StackingTRBModel',
                    'loaded': False
                },
                'triples_model': {
                    'module': 'src.models.players.triples.model_triples',
                    'class_name': 'Stacking3PTModel',
                    'loaded': False
                },
                'dd_model': {
                    'module': 'src.models.players.double_double.dd_model',
                    'class_name': 'DoubleDoubleAdvancedModel',
                    'loaded': False
                },
                'win_model': {
                    'module': 'src.models.teams.is_win.model_is_win',
                    'class_name': 'IsWinModel',
                    'loaded': False
                },
                'teams_pts_model': {
                    'module': 'src.models.teams.teams_points.model_teams_points',
                    'class_name': 'TeamPointsModel',
                    'loaded': False
                },
                'total_pts_model': {
                    'module': 'src.models.teams.total_points.model_total_points',
                    'class_name': 'NBATotalPointsPredictor',
                    'loaded': False
                }
            }
            
            self.models_loaded = True
            self.logger.info(" Sistema de modelos especializados LISTO (lazy loading)")
            self.logger.info(f"Modelos disponibles: {len(self.available_models)}")
            
        except Exception as e:
            self.logger.warning(f"Error inicializando sistema de modelos: {e}")
            self.models_loaded = False
            self.available_models = {}
    
    def _load_model_dynamically(self, model_name: str):
        """Carga un modelo específico dinámicamente"""
        if model_name not in self.available_models:
            return None
            
        model_info = self.available_models[model_name]
        if model_info['loaded']:
            return self.specialized_models[model_name]
            
        try:
            import importlib
            module = importlib.import_module(model_info['module'])
            model_class = getattr(module, model_info['class_name'])
            model_instance = model_class()
            
            self.specialized_models[model_name] = model_instance
            self.available_models[model_name]['loaded'] = True
            
            self.logger.info(f" Modelo {model_name} cargado dinámicamente")
            return model_instance
            
        except Exception as e:
            self.logger.warning(f"No se pudo cargar modelo {model_name}: {e}")
            return None

    def get_specialized_predictions(self, player_data: pd.DataFrame, 
                                  opponent: str, date: str) -> Dict[str, Any]:
        """
        Obtiene predicciones de TODOS los modelos especializados
        para un jugador específico
        """
        if not self.models_loaded:
            self.load_specialized_models()
            
        predictions = {}
        confidence_scores = {}
        
        try:
            # Preparar datos del jugador para los modelos
            player_features = self._prepare_player_features(player_data, opponent, date)
            
            # PREDICCIONES DE JUGADORES INDIVIDUALES
            if 'pts_model' in self.specialized_models:
                pts_pred = self.specialized_models['pts_model'].predict(player_features)
                predictions['PTS'] = {
                    'prediction': float(pts_pred),
                    'confidence': self.confidence_weights['pts_model'],
                    'source': 'specialized_model'
                }
                
            if 'ast_model' in self.specialized_models:
                ast_pred = self.specialized_models['ast_model'].predict(player_features)
                predictions['AST'] = {
                    'prediction': float(ast_pred),
                    'confidence': self.confidence_weights['ast_model'],
                    'source': 'specialized_model'
                }
                
            if 'trb_model' in self.specialized_models:
                trb_pred = self.specialized_models['trb_model'].predict(player_features)
                predictions['TRB'] = {
                    'prediction': float(trb_pred),
                    'confidence': self.confidence_weights['trb_model'],
                    'source': 'specialized_model'
                }
                
            if 'triples_model' in self.specialized_models:
                triples_pred = self.specialized_models['triples_model'].predict(player_features)
                predictions['3PM'] = {
                    'prediction': float(triples_pred),
                    'confidence': self.confidence_weights['triples_model'],
                    'source': 'specialized_model'
                }
                
            if 'dd_model' in self.specialized_models:
                dd_prob = self.specialized_models['dd_model'].predict_proba(player_features)
                predictions['DOUBLE_DOUBLE'] = {
                    'probability': float(dd_prob),
                    'confidence': self.confidence_weights['dd_model'],
                    'source': 'specialized_model'
                }
                
        except Exception as e:
            self.logger.warning(f"Error en predicciones especializadas para jugador: {e}")
            
        return predictions
    
    def get_team_predictions(self, team1: str, team2: str, 
                           team1_data: pd.DataFrame, team2_data: pd.DataFrame,
                           date: str) -> Dict[str, Any]:
        """
        Obtiene predicciones de modelos especializados de equipos
        """
        predictions = {}
        
        try:
            # Preparar datos de equipos
            team_features = self._prepare_team_features(team1, team2, team1_data, team2_data, date)
            
            # PREDICCIÓN DE VICTORIA
            if 'win_model' in self.specialized_models:
                win_prob = self.specialized_models['win_model'].predict_proba(team_features)
                predictions['WIN_PROBABILITY'] = {
                    'team1_win_prob': float(win_prob),
                    'team2_win_prob': float(1 - win_prob),
                    'confidence': self.confidence_weights['win_model'],
                    'source': 'specialized_model'
                }
            
            # PREDICCIÓN DE PUNTOS DE EQUIPOS
            if 'teams_pts_model' in self.specialized_models:
                team1_pts = self.specialized_models['teams_pts_model'].predict(team_features)
                team2_pts = self.specialized_models['teams_pts_model'].predict(
                    self._swap_team_features(team_features))
                
                predictions['TEAM_POINTS'] = {
                    'team1_points': float(team1_pts),
                    'team2_points': float(team2_pts),
                    'confidence': self.confidence_weights['teams_pts_model'],
                    'source': 'specialized_model'
                }
            
            # PREDICCIÓN DE TOTAL DE PUNTOS
            if 'total_pts_model' in self.specialized_models:
                total_pts = self.specialized_models['total_pts_model'].predict(team_features)
                predictions['TOTAL_POINTS'] = {
                    'total_points': float(total_pts),
                    'confidence': self.confidence_weights['total_pts_model'],
                    'source': 'specialized_model'
                }
                
        except Exception as e:
            self.logger.warning(f"Error en predicciones de equipos: {e}")
            
        return predictions
    
    def integrate_with_montecarlo(self, montecarlo_results: Dict, 
                                specialized_predictions: Dict) -> Dict:
        """
        Integra predicciones especializadas con resultados de Montecarlo
        usando pesos de confianza adaptativos
        """
        integrated_results = montecarlo_results.copy()
        
        try:
            # INTEGRACIÓN INTELIGENTE POR ESTADÍSTICA
            for stat, spec_pred in specialized_predictions.items():
                if stat in montecarlo_results and 'prediction' in spec_pred:
                    
                    # Obtener predicción de Montecarlo
                    mc_value = montecarlo_results[stat]['mean']
                    mc_confidence = 0.75  # Confianza base de Montecarlo
                    
                    # Obtener predicción especializada
                    spec_value = spec_pred['prediction']
                    spec_confidence = spec_pred['confidence']
                    
                    # INTEGRACIÓN PONDERADA POR CONFIANZA
                    total_confidence = mc_confidence + spec_confidence
                    integrated_value = (
                        (mc_value * mc_confidence + spec_value * spec_confidence) 
                        / total_confidence
                    )
                    
                    # Actualizar resultado integrado
                    integrated_results[stat] = {
                        'mean': integrated_value,
                        'montecarlo_prediction': mc_value,
                        'specialized_prediction': spec_value,
                        'integration_confidence': min(total_confidence / 2, 0.95),
                        'improvement': abs(integrated_value - mc_value),
                        'source': 'integrated_prediction'
                    }
                    
                    self.logger.debug(f"Integrado {stat}: MC={mc_value:.2f} + Spec={spec_value:.2f} = {integrated_value:.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error en integración: {e}")
            
        return integrated_results
    
    def validate_against_models(self, player_name: str, game_data: Dict,
                              opponent: str, date: str) -> Dict:
        """
        Valida predicciones de Montecarlo contra TODOS los modelos especializados
        """
        validation_results = {
            'player': player_name,
            'opponent': opponent,
            'date': date,
            'validations': {},
            'overall_confidence': 0.0,
            'recommendation_strength': 'LOW'
        }
        
        try:
            # Obtener predicciones especializadas
            player_data = self._get_player_data(player_name)
            specialized_preds = self.get_specialized_predictions(player_data, opponent, date)
            
            total_confidence = 0
            validated_stats = 0
            
            # VALIDAR CADA ESTADÍSTICA
            for stat in ['PTS', 'AST', 'TRB', '3PM']:
                if stat in game_data and stat in specialized_preds:
                    
                    mc_prediction = game_data[stat]
                    spec_prediction = specialized_preds[stat]['prediction']
                    spec_confidence = specialized_preds[stat]['confidence']
                    
                    # Calcular diferencia relativa
                    relative_diff = abs(mc_prediction - spec_prediction) / max(mc_prediction, 1)
                    
                    # Determinar concordancia
                    if relative_diff <= 0.15:  # Diferencia <= 15%
                        concordance = 'HIGH'
                        confidence_boost = spec_confidence
                    elif relative_diff <= 0.25:  # Diferencia <= 25% 
                        concordance = 'MEDIUM'
                        confidence_boost = spec_confidence * 0.7
                    else:
                        concordance = 'LOW'
                        confidence_boost = spec_confidence * 0.4
                    
                    validation_results['validations'][stat] = {
                        'montecarlo_pred': mc_prediction,
                        'specialized_pred': spec_prediction,
                        'relative_difference': relative_diff,
                        'concordance': concordance,
                        'confidence_boost': confidence_boost
                    }
                    
                    total_confidence += confidence_boost
                    validated_stats += 1
            
            # VALIDAR DOUBLE-DOUBLE SI APLICA
            if 'DOUBLE_DOUBLE' in specialized_preds:
                dd_prob = specialized_preds['DOUBLE_DOUBLE']['probability']
                
                # Estimar probabilidad de DD desde Montecarlo
                mc_pts = game_data.get('PTS', 0)
                mc_trb = game_data.get('TRB', 0) 
                mc_ast = game_data.get('AST', 0)
                
                mc_dd_prob = self._estimate_dd_probability(mc_pts, mc_trb, mc_ast)
                
                dd_concordance = 'HIGH' if abs(dd_prob - mc_dd_prob) <= 0.2 else 'MEDIUM'
                
                validation_results['validations']['DOUBLE_DOUBLE'] = {
                    'montecarlo_prob': mc_dd_prob,
                    'specialized_prob': dd_prob,
                    'concordance': dd_concordance,
                    'confidence_boost': specialized_preds['DOUBLE_DOUBLE']['confidence']
                }
                
                total_confidence += specialized_preds['DOUBLE_DOUBLE']['confidence']
                validated_stats += 1
            
            # CALCULAR CONFIANZA GENERAL
            if validated_stats > 0:
                validation_results['overall_confidence'] = total_confidence / validated_stats
                
                # Determinar fuerza de recomendación
                if validation_results['overall_confidence'] >= 0.85:
                    validation_results['recommendation_strength'] = 'VERY_HIGH'
                elif validation_results['overall_confidence'] >= 0.75:
                    validation_results['recommendation_strength'] = 'HIGH'
                elif validation_results['overall_confidence'] >= 0.65:
                    validation_results['recommendation_strength'] = 'MEDIUM'
                else:
                    validation_results['recommendation_strength'] = 'LOW'
                    
        except Exception as e:
            self.logger.error(f"Error en validación: {e}")
            
        return validation_results
    
    def _prepare_player_features(self, player_data: pd.DataFrame, 
                               opponent: str, date: str) -> pd.DataFrame:
        """Prepara features para modelos especializados de jugadores"""
        # Implementación simplificada - en producción sería más compleja
        return player_data.iloc[-10:].mean().to_frame().T
    
    def _prepare_team_features(self, team1: str, team2: str,
                             team1_data: pd.DataFrame, team2_data: pd.DataFrame,
                             date: str) -> pd.DataFrame:
        """Prepara features para modelos especializados de equipos"""
        # Implementación simplificada
        features = pd.concat([
            team1_data.iloc[-10:].mean(),
            team2_data.iloc[-10:].mean()
        ]).to_frame().T
        return features
    
    def _swap_team_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Intercambia features de equipos para predicción del equipo 2"""
        return features  # Simplificado
    
    def _get_player_data(self, player_name: str) -> pd.DataFrame:
        """Obtiene datos históricos del jugador"""
        # En producción, esto consultaría la base de datos
        return pd.DataFrame()
    
    def _estimate_dd_probability(self, pts: float, trb: float, ast: float) -> float:
        """Estima probabilidad de double-double desde estadísticas individuales"""
        # Lógica simplificada
        dd_combinations = [
            (pts >= 10 and trb >= 10),
            (pts >= 10 and ast >= 10),
            (trb >= 10 and ast >= 10)
        ]
        
        # Convertir a probabilidad aproximada
        base_prob = 0.3 if any(dd_combinations) else 0.1
        
        # Ajustar por proximidad a umbrales
        pts_factor = min(pts / 10, 1.5) if pts >= 8 else pts / 10
        trb_factor = min(trb / 10, 1.5) if trb >= 8 else trb / 10  
        ast_factor = min(ast / 10, 1.5) if ast >= 8 else ast / 10
        
        combined_factor = (pts_factor + trb_factor + ast_factor) / 3
        
        return min(base_prob * combined_factor, 0.95)

class EnsembleIntegrator:
    """
    Integrador de Ensemble para Monte Carlo NBA
    ==========================================
    
    Combina múltiples modelos y técnicas para maximizar precisión:
    - Monte Carlo tradicional
    - Deep Learning predictions
    - Statistical models
    - Ensemble weighting
    """
    
    def __init__(self):
        self.model_weights = {
            'montecarlo': 0.40,     # 40% Monte Carlo
            'deep_learning': 0.35,  # 35% Deep Learning 
            'statistical': 0.25     # 25% Modelos estadísticos
        }
        
        self.confidence_thresholds = {
            'high_confidence': 0.85,
            'medium_confidence': 0.65,
            'low_confidence': 0.45
        }
        
        self.logger = logging.getLogger(__name__)
    
    def integrate_predictions(self, 
                            montecarlo_result: dict,
                            dl_predictions: dict = None,
                            statistical_predictions: dict = None) -> dict:
        """
        Integra predicciones de múltiples modelos usando ensemble inteligente
        
        Args:
            montecarlo_result: Resultado de simulación Monte Carlo
            dl_predictions: Predicciones de Deep Learning (opcional)
            statistical_predictions: Predicciones estadísticas (opcional)
            
        Returns:
            Dict con predicción ensemble integrada
        """
        try:
            # Base: usar Monte Carlo como fundación
            ensemble_result = montecarlo_result.copy()
            
            predictions_available = {'montecarlo': montecarlo_result}
            
            if dl_predictions:
                predictions_available['deep_learning'] = dl_predictions
            if statistical_predictions:
                predictions_available['statistical'] = statistical_predictions
            
            # Calcular pesos dinámicos basados en confianza
            dynamic_weights = self._calculate_dynamic_weights(predictions_available)
            
            # Integrar predicciones de puntuación
            integrated_scores = self._integrate_score_predictions(predictions_available, dynamic_weights)
            ensemble_result['score_predictions'] = integrated_scores
            
            # Integrar probabilidades de victoria
            integrated_probs = self._integrate_win_probabilities(predictions_available, dynamic_weights)
            ensemble_result['win_probabilities'] = integrated_probs
            
            # Calcular métricas de confianza ensemble
            ensemble_confidence = self._calculate_ensemble_confidence(predictions_available, dynamic_weights)
            ensemble_result['ensemble_confidence'] = ensemble_confidence
            
            # Metadatos del ensemble
            ensemble_result['ensemble_metadata'] = {
                'models_used': list(predictions_available.keys()),
                'dynamic_weights': dynamic_weights,
                'integration_method': 'weighted_average_with_confidence',
                'total_models': len(predictions_available)
            }
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Error en integración ensemble: {str(e)}")
            return montecarlo_result  # Fallback al Monte Carlo base
    
    def _calculate_dynamic_weights(self, predictions: dict) -> dict:
        """Calcula pesos dinámicos basados en confianza de cada modelo"""
        weights = {}
        total_confidence = 0.0
        
        for model_name, prediction in predictions.items():
            # Extraer confianza del modelo
            confidence = self._extract_model_confidence(prediction)
            
            # Peso base del modelo
            base_weight = self.model_weights.get(model_name, 0.33)
            
            # Ajustar peso por confianza
            adjusted_weight = base_weight * (0.5 + confidence)  # Rango 0.5x - 1.5x
            
            weights[model_name] = adjusted_weight
            total_confidence += adjusted_weight
        
        # Normalizar pesos
        if total_confidence > 0:
            for model_name in weights:
                weights[model_name] /= total_confidence
        
        return weights
    
    def _extract_model_confidence(self, prediction: dict) -> float:
        """Extrae métrica de confianza de una predicción"""
        try:
            # Para Monte Carlo: usar model_reliability
            if 'confidence_metrics' in prediction:
                return prediction['confidence_metrics'].get('prediction_confidence', 0.5)
            
            # Para Deep Learning: usar accuracy o loss
            if 'confidence' in prediction:
                return prediction['confidence']
            
            # Para modelos estadísticos: usar R-squared o similar
            if 'r_squared' in prediction:
                return prediction['r_squared']
            
            # Default: confianza media
            return 0.5
            
        except Exception:
            return 0.5
    
    def _integrate_score_predictions(self, predictions: dict, weights: dict) -> dict:
        """Integra predicciones de puntuación usando pesos dinámicos"""
        integrated_home = {'mean': 0.0, 'std': 0.0}
        integrated_away = {'mean': 0.0, 'std': 0.0}
        
        total_weight = sum(weights.values())
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0) / total_weight
            
            if 'score_predictions' in prediction:
                home_pred = prediction['score_predictions']['home_score']
                away_pred = prediction['score_predictions']['away_score']
                
                integrated_home['mean'] += weight * home_pred['mean']
                integrated_away['mean'] += weight * away_pred['mean']
                
                # Integrar incertidumbre (conservadoramente)
                integrated_home['std'] += weight * home_pred.get('std', 5.0)
                integrated_away['std'] += weight * away_pred.get('std', 5.0)
        
        # Calcular percentiles basados en distribución normal
        home_mean, home_std = integrated_home['mean'], integrated_home['std']
        away_mean, away_std = integrated_away['mean'], integrated_away['std']
        
        integrated_home.update({
            'median': home_mean,
            'min': max(80, home_mean - 2 * home_std),
            'max': min(140, home_mean + 2 * home_std),
            'percentiles': {
                '25th': home_mean - 0.67 * home_std,
                '75th': home_mean + 0.67 * home_std
            }
        })
        
        integrated_away.update({
            'median': away_mean,
            'min': max(80, away_mean - 2 * away_std),
            'max': min(140, away_mean + 2 * away_std),
            'percentiles': {
                '25th': away_mean - 0.67 * away_std,
                '75th': away_mean + 0.67 * away_std
            }
        })
        
        return {
            'home_score': integrated_home,
            'away_score': integrated_away
        }
    
    def _integrate_win_probabilities(self, predictions: dict, weights: dict) -> dict:
        """Integra probabilidades de victoria usando pesos dinámicos"""
        integrated_home_prob = 0.0
        total_weight = sum(weights.values())
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0) / total_weight
            
            if 'win_probabilities' in prediction:
                home_prob = prediction['win_probabilities'].get('home_win', 0.5)
                integrated_home_prob += weight * home_prob
        
        return {
            'home_win': integrated_home_prob,
            'away_win': 1.0 - integrated_home_prob
        }
    
    def _calculate_ensemble_confidence(self, predictions: dict, weights: dict) -> dict:
        """Calcula métricas de confianza del ensemble"""
        # Confianza ponderada
        weighted_confidence = 0.0
        total_weight = sum(weights.values())
        
        for model_name, prediction in predictions.items():
            weight = weights.get(model_name, 0.0) / total_weight
            confidence = self._extract_model_confidence(prediction)
            weighted_confidence += weight * confidence
        
        # Consensus entre modelos
        home_probs = []
        for prediction in predictions.values():
            if 'win_probabilities' in prediction:
                home_probs.append(prediction['win_probabilities'].get('home_win', 0.5))
        
        consensus_score = 1.0 - np.std(home_probs) if len(home_probs) > 1 else 0.5
        
        # Confianza final
        final_confidence = (weighted_confidence + consensus_score) / 2
        
        confidence_level = 'high' if final_confidence >= self.confidence_thresholds['high_confidence'] else \
                          'medium' if final_confidence >= self.confidence_thresholds['medium_confidence'] else 'low'
        
        return {
            'weighted_confidence': weighted_confidence,
            'consensus_score': consensus_score,
            'final_confidence': final_confidence,
            'confidence_level': confidence_level,
            'models_agreement': len(home_probs),
            'prediction_variance': np.var(home_probs) if len(home_probs) > 1 else 0.0
        } 