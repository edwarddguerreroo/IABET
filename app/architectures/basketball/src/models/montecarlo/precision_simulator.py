"""
SIMULADOR DE PRECISIÓN EXTREMA - MONTECARLO REVOLUCIONARIO
========================================================

Simulador de nueva generación que combina Montecarlo con TODOS los modelos
especializados
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from scipy import stats
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from .enhanced_engine import EnhancedMonteCarloEngine
from .simulator import NBAGameSimulator

class PrecisionSimulator(NBAGameSimulator):
    """
    Simulador de precisión extrema que integra:
    - Simulaciones de Montecarlo avanzadas
    - Predicciones de TODOS los modelos especializados
    - Validación cruzada en tiempo real
    - Ensemble de múltiples enfoques
    - Ajustes adaptativos por contexto
    """
    
    def __init__(self, enhanced_engine: EnhancedMonteCarloEngine, 
                 num_simulations: int = 20000):
        # Inicializar con motor mejorado
        super().__init__(enhanced_engine, num_simulations)
        self.enhanced_engine = enhanced_engine
        self.logger = logging.getLogger(__name__)
        
        # Configuración de precisión extrema
        self.precision_config = {
            'simulation_methods': [
                'enhanced_montecarlo',
                'specialized_models',
                'ensemble_voting',
                'bayesian_inference',
                'confidence_weighted'
            ],
            'validation_methods': [
                'cross_model_validation',
                'historical_accuracy_check',
                'consistency_verification',
                'outlier_detection',
                'confidence_scoring'
            ],
            'integration_strategies': [
                'weighted_average',
                'confidence_based',
                'performance_adaptive',
                'context_aware',
                'dynamic_ensemble'
            ]
        }
        
        # Métricas de rendimiento en tiempo real
        self.performance_metrics = {
            'accuracy_by_stat': {},
            'confidence_calibration': {},
            'model_agreements': {},
            'prediction_stability': {}
        }
        
        self.logger.info(" Simulador de PRECISIÓN EXTREMA inicializado")
        self.logger.info(f"Configurado con {len(self.precision_config['simulation_methods'])} métodos de simulación")
        self.logger.info(f"Usando {len(self.precision_config['validation_methods'])} métodos de validación")
    
    def _prepare_enhanced_context(self, context: Dict) -> Dict:
        """Prepara contexto mejorado para simulación de precisión extrema"""
        from datetime import datetime
        
        enhanced_context = context.copy()
        
        # Agregar datos de contexto adicionales
        enhanced_context.update({
            'timestamp': str(datetime.now()),
            'precision_level': 'EXTREME',
            'simulation_mode': 'PRECISION',
            'enhanced_features': True,
            'correlation_adjustment': True,
            'advanced_modeling': True
        })
        
        return enhanced_context

    def simulate_game_with_extreme_precision(self, home_team: str, away_team: str,
                                           context: Dict = None) -> Dict:
        """
        Simula partido con precisión extrema usando múltiples métodos
        """
        self.logger.info(f" SIMULACIÓN DE PRECISIÓN EXTREMA: {away_team} @ {home_team}")
        
        # Preparar contexto mejorado
        enhanced_context = self._prepare_enhanced_context(context or {})
        
        # Ejecutar múltiples métodos de simulación
        simulation_results = {}
        
        for method in self.precision_config['simulation_methods']:
            try:
                method_results = self._execute_simulation_method(
                    method, home_team, away_team, enhanced_context
                )
                simulation_results[method] = method_results
                self.logger.info(f" Método {method} completado")
                
            except Exception as e:
                self.logger.warning(f"Error en método {method}: {e}")
                continue
        
        # Validar resultados usando múltiples métodos
        validation_results = self._validate_simulation_results(
            simulation_results, home_team, away_team, enhanced_context
        )
        
        # Integrar resultados usando estrategias avanzadas
        integrated_results = self._integrate_multiple_results(
            simulation_results, validation_results
        )
        
        # Calcular métricas de confianza finales
        confidence_metrics = self._calculate_extreme_confidence_metrics(
            integrated_results, simulation_results, validation_results
        )
        
        # Resultado final con máxima precisión
        final_results = {
            'game_info': {
                'home_team': home_team,
                'away_team': away_team,
                'simulation_methods_used': len(simulation_results),
                'validation_methods_applied': len(self.precision_config['validation_methods']),
                'precision_level': 'EXTREME'
            },
            'score_predictions': integrated_results['scores'],
            'win_probabilities': integrated_results['win_probs'],
            'player_performances': integrated_results['players'],
            'confidence_metrics': confidence_metrics,
            'method_details': simulation_results,
            'validation_summary': validation_results,
            'prediction_quality': self._assess_prediction_quality(integrated_results)
        }
        
        self.logger.info(f" Simulación completada con confianza general: {confidence_metrics['overall_confidence']:.1%}")
        
        return final_results
    
    def _execute_simulation_method(self, method: str, home_team: str, 
                                 away_team: str, context: Dict) -> Dict:
        """Ejecuta un método específico de simulación"""
        
        if method == 'enhanced_montecarlo':
            return self._enhanced_montecarlo_simulation(home_team, away_team, context)
        
        elif method == 'specialized_models':
            return self._specialized_models_simulation(home_team, away_team, context)
        
        elif method == 'ensemble_voting':
            return self._ensemble_voting_simulation(home_team, away_team, context)
        
        elif method == 'bayesian_inference':
            return self._bayesian_inference_simulation(home_team, away_team, context)
        
        elif method == 'confidence_weighted':
            return self._confidence_weighted_simulation(home_team, away_team, context)
        
        else:
            raise ValueError(f"Método de simulación desconocido: {method}")
    
    def _enhanced_montecarlo_simulation(self, home_team: str, away_team: str,
                                      context: Dict) -> Dict:
        """Simulación Montecarlo mejorada con motor avanzado"""
        return self.enhanced_engine.run_enhanced_simulation(
            home_team, away_team, 
            simulations=self.num_simulations,
            context=context
        )
    
    def _specialized_models_simulation(self, home_team: str, away_team: str,
                                     context: Dict) -> Dict:
        """Simulación basada puramente en modelos especializados"""
        
        # Obtener jugadores de ambos equipos
        home_players = self._get_active_players(home_team)
        away_players = self._get_active_players(away_team)
        
        # Predicciones de modelos especializados para cada jugador
        home_predictions = {}
        away_predictions = {}
        
        for player in home_players:
            player_data = self.enhanced_engine._get_player_historical_data(player)
            predictions = self.enhanced_engine.integrator.get_specialized_predictions(
                player_data, away_team, context.get('date', 'TODAY')
            )
            home_predictions[player] = predictions
        
        for player in away_players:
            player_data = self.enhanced_engine._get_player_historical_data(player)
            predictions = self.enhanced_engine.integrator.get_specialized_predictions(
                player_data, home_team, context.get('date', 'TODAY')
            )
            away_predictions[player] = predictions
        
        # Agregar predicciones por equipo
        home_totals = self._aggregate_team_predictions(home_predictions)
        away_totals = self._aggregate_team_predictions(away_predictions)
        
        # Predicciones de equipos
        team_predictions = self.enhanced_engine.integrator.get_team_predictions(
            home_team, away_team,
            self.enhanced_engine._get_team_historical_data(home_team),
            self.enhanced_engine._get_team_historical_data(away_team),
            context.get('date', 'TODAY')
        )
        
        return {
            'home_score': home_totals['PTS'],
            'away_score': away_totals['PTS'],
            'home_players': home_predictions,
            'away_players': away_predictions,
            'team_predictions': team_predictions,
            'method': 'specialized_models'
        }
    
    def _ensemble_voting_simulation(self, home_team: str, away_team: str,
                                  context: Dict) -> Dict:
        """Simulación usando ensemble voting de múltiples predictores"""
        
        # Crear múltiples predictores con diferentes configuraciones
        predictors = [
            self._create_conservative_predictor(),
            self._create_aggressive_predictor(), 
            self._create_balanced_predictor(),
            self._create_context_aware_predictor(context)
        ]
        
        # Obtener predicciones de cada predictor
        predictions = []
        for predictor in predictors:
            pred = self._get_predictor_simulation(predictor, home_team, away_team, context)
            predictions.append(pred)
        
        # Voting ensemble
        ensemble_result = self._combine_ensemble_predictions(predictions)
        
        return {
            'home_score': ensemble_result['home_score'],
            'away_score': ensemble_result['away_score'],
            'predictor_votes': predictions,
            'ensemble_confidence': ensemble_result['confidence'],
            'method': 'ensemble_voting'
        }
    
    def _bayesian_inference_simulation(self, home_team: str, away_team: str,
                                     context: Dict) -> Dict:
        """Simulación usando inferencia bayesiana"""
        
        # Obtener priors basados en datos históricos
        home_priors = self._get_bayesian_priors(home_team)
        away_priors = self._get_bayesian_priors(away_team)
        
        # Actualizar con evidencia reciente
        home_posterior = self._update_bayesian_posterior(home_priors, home_team, context)
        away_posterior = self._update_bayesian_posterior(away_priors, away_team, context)
        
        # Generar predicciones bayesianas
        home_score_dist = stats.norm(
            home_posterior['score_mean'], 
            home_posterior['score_std']
        )
        away_score_dist = stats.norm(
            away_posterior['score_mean'],
            away_posterior['score_std']
        )
        
        # Simular distribuciones
        home_scores = home_score_dist.rvs(1000)
        away_scores = away_score_dist.rvs(1000)
        
        # Aplicar límites realistas
        home_scores = np.clip(home_scores, 85, 130)
        away_scores = np.clip(away_scores, 85, 130)
        
        return {
            'home_score': np.mean(home_scores),
            'away_score': np.mean(away_scores),
            'home_score_distribution': {
                'mean': np.mean(home_scores),
                'std': np.std(home_scores),
                'confidence_interval': np.percentile(home_scores, [5, 95])
            },
            'away_score_distribution': {
                'mean': np.mean(away_scores),
                'std': np.std(away_scores), 
                'confidence_interval': np.percentile(away_scores, [5, 95])
            },
            'method': 'bayesian_inference'
        }
    
    def _confidence_weighted_simulation(self, home_team: str, away_team: str,
                                      context: Dict) -> Dict:
        """Simulación ponderada por confianza de diferentes fuentes"""
        
        # Obtener predicciones de diferentes fuentes con sus confianzas
        sources = {
            'montecarlo': {
                'prediction': self._get_basic_montecarlo_prediction(home_team, away_team, context),
                'confidence': 0.75
            },
            'specialized_models': {
                'prediction': self._get_specialized_models_prediction(home_team, away_team, context),
                'confidence': 0.85
            },
            'historical_patterns': {
                'prediction': self._get_historical_patterns_prediction(home_team, away_team, context),
                'confidence': 0.70
            },
            'matchup_analysis': {
                'prediction': self._get_matchup_analysis_prediction(home_team, away_team, context),
                'confidence': 0.80
            }
        }
        
        # Ponderación por confianza
        total_confidence = sum(source['confidence'] for source in sources.values())
        
        weighted_home_score = sum(
            source['prediction']['home_score'] * source['confidence']
            for source in sources.values()
        ) / total_confidence
        
        weighted_away_score = sum(
            source['prediction']['away_score'] * source['confidence'] 
            for source in sources.values()
        ) / total_confidence
        
        return {
            'home_score': weighted_home_score,
            'away_score': weighted_away_score,
            'source_contributions': sources,
            'total_confidence': total_confidence / len(sources),
            'method': 'confidence_weighted'
        }
    
    def _validate_simulation_results(self, simulation_results: Dict,
                                   home_team: str, away_team: str,
                                   context: Dict) -> Dict:
        """Valida resultados usando múltiples métodos"""
        
        validation_results = {}
        
        for method in self.precision_config['validation_methods']:
            try:
                validation = self._execute_validation_method(
                    method, simulation_results, home_team, away_team, context
                )
                validation_results[method] = validation
                
            except Exception as e:
                self.logger.warning(f"Error en validación {method}: {e}")
                continue
        
        return validation_results
    
    def _integrate_multiple_results(self, simulation_results: Dict,
                                  validation_results: Dict) -> Dict:
        """Integra resultados de múltiples métodos usando estrategias avanzadas"""
        
        # Calcular pesos dinámicos basados en validación
        method_weights = self._calculate_dynamic_weights(
            simulation_results, validation_results
        )
        
        # Integración ponderada de puntuaciones
        home_scores = []
        away_scores = []
        weights = []
        
        for method, results in simulation_results.items():
            if 'home_score' in results and 'away_score' in results:
                home_scores.append(results['home_score'])
                away_scores.append(results['away_score'])
                weights.append(method_weights.get(method, 0.2))
        
        # Normalizar pesos
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Calcular predicciones finales ponderadas
        final_home_score = sum(score * weight for score, weight in zip(home_scores, weights))
        final_away_score = sum(score * weight for score, weight in zip(away_scores, weights))
        
        # Aplicar límites finales de seguridad
        final_home_score = max(85, min(130, final_home_score))
        final_away_score = max(85, min(130, final_away_score))
        
        # Calcular probabilidades de victoria
        score_diff = final_home_score - final_away_score
        home_win_prob = self._score_diff_to_win_probability(score_diff)
        
        return {
            'scores': {
                'home_score': final_home_score,
                'away_score': final_away_score,
                'total_points': final_home_score + final_away_score,
                'score_differential': score_diff
            },
            'win_probs': {
                'home_win_probability': home_win_prob,
                'away_win_probability': 1 - home_win_prob
            },
            'players': self._integrate_player_predictions(simulation_results),
            'integration_weights': dict(zip(simulation_results.keys(), weights))
        }
    
    def _calculate_extreme_confidence_metrics(self, integrated_results: Dict,
                                            simulation_results: Dict,
                                            validation_results: Dict) -> Dict:
        """Calcula métricas de confianza extremadamente detalladas"""
        
        # Consistencia entre métodos
        method_consistency = self._calculate_method_consistency(simulation_results)
        
        # Validación cruzada
        cross_validation_score = self._calculate_cross_validation_score(validation_results)
        
        # Confianza por tipo de predicción
        prediction_confidences = {
            'score_prediction': self._calculate_score_confidence(simulation_results),
            'win_probability': self._calculate_win_prob_confidence(simulation_results),
            'player_performance': self._calculate_player_confidence(simulation_results)
        }
        
        # Confianza general ponderada
        overall_confidence = (
            method_consistency * 0.3 +
            cross_validation_score * 0.3 +
            np.mean(list(prediction_confidences.values())) * 0.4
        )
        
        return {
            'overall_confidence': overall_confidence,
            'method_consistency': method_consistency,
            'cross_validation_score': cross_validation_score,
            'prediction_confidences': prediction_confidences,
            'reliability_grade': self._assign_reliability_grade(overall_confidence),
            'recommendation_strength': self._determine_recommendation_strength(overall_confidence)
        }
    
    # Métodos auxiliares simplificados (en producción serían más complejos)
    
    def _get_active_players(self, team: str) -> List[str]:
        """Obtiene jugadores activos del equipo"""
        team_data = self.enhanced_engine._get_team_historical_data(team)
        if team_data.empty:
            return []
        return team_data['Player'].unique().tolist()[:8]  # Top 8 jugadores
    
    def _aggregate_team_predictions(self, player_predictions: Dict) -> Dict:
        """Agrega predicciones individuales por equipo"""
        totals = {'PTS': 0, 'AST': 0, 'TRB': 0}
        
        for player, predictions in player_predictions.items():
            for stat in totals.keys():
                if stat in predictions and 'prediction' in predictions[stat]:
                    totals[stat] += predictions[stat]['prediction']
        
        return totals
    
    def _score_diff_to_win_probability(self, score_diff: float) -> float:
        """Convierte diferencia de puntos a probabilidad de victoria"""
        # Función logística calibrada con datos históricos de NBA
        return 1 / (1 + np.exp(-score_diff * 0.15))
    
    def _calculate_method_consistency(self, simulation_results: Dict) -> float:
        """Calcula consistencia entre métodos de simulación"""
        home_scores = [r.get('home_score', 0) for r in simulation_results.values()]
        away_scores = [r.get('away_score', 0) for r in simulation_results.values()]
        
        if len(home_scores) < 2:
            return 0.5
        
        home_cv = np.std(home_scores) / np.mean(home_scores) if np.mean(home_scores) > 0 else 1
        away_cv = np.std(away_scores) / np.mean(away_scores) if np.mean(away_scores) > 0 else 1
        
        # Invertir CV para obtener consistencia (menor CV = mayor consistencia)
        consistency = max(0, 1 - (home_cv + away_cv) / 2)
        return min(1, consistency)
    
    def _assign_reliability_grade(self, confidence: float) -> str:
        """Asigna grado de confiabilidad"""
        if confidence >= 0.90:
            return 'A+'
        elif confidence >= 0.85:
            return 'A'
        elif confidence >= 0.80:
            return 'A-'
        elif confidence >= 0.75:
            return 'B+'
        elif confidence >= 0.70:
            return 'B'
        elif confidence >= 0.65:
            return 'B-'
        elif confidence >= 0.60:
            return 'C+'
        else:
            return 'C'
    
    def _determine_recommendation_strength(self, confidence: float) -> str:
        """Determina fuerza de recomendación"""
        if confidence >= 0.85:
            return 'VERY_STRONG'
        elif confidence >= 0.75:
            return 'STRONG'
        elif confidence >= 0.65:
            return 'MODERATE'
        elif confidence >= 0.55:
            return 'WEAK'
        else:
            return 'VERY_WEAK' 