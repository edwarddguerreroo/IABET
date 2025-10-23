"""
Game Predictor - ORQUESTADOR MAESTRO
=====================================

Punto final que integra:
- Predicciones (unified_predictor)
- Odds (sportradar_api)  
- Análisis de apuestas (betting_analytics)

SIEMPRE usa la fecha actual (HOY) automáticamente.

Autor: Sistema NBA
Fecha: 2025-10-21
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
import pandas as pd

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

from app.architectures.basketball.pipelines.predict.unified_predictor.unified_predictor import UnifiedPredictor
from app.architectures.basketball.utils.bookmakers.sportradar_api import SportradarAPI
from app.architectures.basketball.utils.bookmakers.betting_analytics import BettingAnalytics, FilteredPrediction
from app.architectures.basketball.utils.bookmakers.config.config import get_config

logger = logging.getLogger(__name__)


class GamePredictor:
    """
    Orquestador maestro que integra predicciones, odds y análisis de apuestas.
    
    Uso:
        predictor = GamePredictor()
        recommendations = predictor.get_betting_recommendations()
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el orquestador.
        
        Args:
            api_key: API key de Sportradar (opcional, se puede cargar de config)
        """
        logger.info("="*80)
        logger.info("INICIALIZANDO GAME PREDICTOR")
        logger.info("="*80)
        
        # 1. Inicializar Unified Predictor
        logger.info("[1/3] Inicializando Unified Predictor...")
        try:
            self.unified_predictor = UnifiedPredictor()
            # Cargar todos los modelos
            self.unified_predictor.load_all_models()
            logger.info("[OK] Unified Predictor listo con todos los modelos cargados")
        except Exception as e:
            logger.error(f"[ERROR] No se pudo inicializar Unified Predictor: {e}")
            raise
        
        # 2. Inicializar Sportradar API
        logger.info("[2/3] Inicializando Sportradar API...")
        try:
            self.sportradar = SportradarAPI(api_key=api_key)
            logger.info("[OK] Sportradar API listo")
        except Exception as e:
            logger.error(f"[ERROR] No se pudo inicializar Sportradar API: {e}")
            raise
        
        # 3. Inicializar Betting Analytics
        logger.info("[3/3] Inicializando Betting Analytics...")
        try:
            config = get_config()
            betting_config = config.config.get('betting', {})
            self.analytics = BettingAnalytics(betting_config)
            logger.info("[OK] Betting Analytics listo")
        except Exception as e:
            logger.error(f"[ERROR] No se pudo inicializar Betting Analytics: {e}")
            raise
        
        logger.info("="*80)
        logger.info("[OK] GAME PREDICTOR INICIALIZADO CORRECTAMENTE")
        logger.info("="*80)
    
    def predict(
        self, 
        games_data: List[Dict],
        min_difference: float = 0.5,
        min_confidence: float = 70.0
    ) -> Dict[str, Any]:
        """
        Método principal de predicción - ORQUESTADOR COMPLETO.
        
        Este es el punto de entrada principal que orquesta:
        1. UnifiedPredictor → Predicciones de modelos
        2. Sportradar API → Odds reales
        3. BettingAnalytics → Filtrado de viables
        
        Args:
            games_data: Lista de juegos en formato Sportradar
            min_difference: Diferencia mínima entre predicción y línea (default: 0.5)
            min_confidence: Confianza mínima del modelo en % (default: 70.0)
            
        Returns:
            Dict con recomendaciones de apuestas viables y predicciones completas
        """
        return self.predict_and_recommend(
            games_data=games_data,
            min_difference=min_difference,
            min_confidence=min_confidence
        )
    
    def predict_and_recommend(
        self, 
        games_data: List[Dict],
        min_difference: float = 0.5,
        min_confidence: float = 70.0
    ) -> Dict[str, Any]:
        """
        Flujo completo REAL de predicción y recomendación.
        
        FLUJO:
        1. UnifiedPredictor genera predicciones
        2. Sportradar API obtiene odds reales
        3. BettingAnalytics filtra solo viables
        4. Retorna apuestas recomendadas
        
        Args:
            games_data: Lista de juegos en formato Sportradar
            min_difference: Diferencia mínima entre predicción y línea
            min_confidence: Confianza mínima del modelo (%)
            
        Returns:
            Dict con predicciones filtradas y recomendaciones
        """
        logger.info("\n" + "="*80)
        logger.info("INICIANDO FLUJO COMPLETO DE PREDICCIÓN")
        logger.info("="*80)
        
        # PASO 1: Generar predicciones con UnifiedPredictor
        logger.info("\n[PASO 1/3] Generando predicciones con modelos...")
        try:
            all_predictions = self.unified_predictor.predict(games_data)
            
            # Contar predicciones
            total_preds = sum(
                len(player.get('predictions', {}))
                for game in all_predictions.get('predictions', [])
                for player in game.get('player_predictions', [])
            )
            
            logger.info(f"[OK] {total_preds} predicciones generadas")
            
        except Exception as e:
            logger.error(f"[ERROR] Fallo generando predicciones: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': [],
                'recommendations': []
            }
        
        # PASO 2: Obtener odds REALES de Sportradar
        logger.info("\n[PASO 2/3] Obteniendo odds reales de Sportradar...")
        
        today = datetime.now().strftime("%Y-%m-%d")
        odds_data = {}
        
        try:
            # Obtener player props
            player_props_result = self.sportradar.get_player_props_by_date(today)
            
            if player_props_result.get('success'):
                events = player_props_result.get('data', {}).get('events', [])
                logger.info(f"[OK] Player props obtenidas: {len(events)} eventos")
                
                # Organizar odds por jugador
                for event in events:
                    for player_data in event.get('players', []):
                        player_name = player_data.get('player_name')
                        if player_name:
                            odds_data[player_name] = player_data.get('props', {})
            else:
                logger.warning(f"[WARNING] No se pudieron obtener odds: {player_props_result.get('error')}")
                logger.info("[INFO] Continuando sin odds (solo con predicciones)")
                
        except Exception as e:
            logger.warning(f"[WARNING] Error obteniendo odds: {e}")
            logger.info("[INFO] Continuando sin odds (solo con predicciones)")
        
        # PASO 3: Filtrar predicciones viables con BettingAnalytics
        logger.info("\n[PASO 3/3] Filtrando predicciones viables...")
        logger.info(f"  Criterios: diferencia >= {min_difference}, confianza >= {min_confidence}%")
        
        try:
            viable_predictions = self.analytics.filter_predictions(
                unified_predictions=all_predictions,
                odds_data=odds_data if odds_data else None,
                min_difference=min_difference,
                min_confidence=min_confidence
            )
            
            logger.info(f"[OK] {len(viable_predictions)} predicciones viables encontradas")
            
        except Exception as e:
            logger.error(f"[ERROR] Fallo filtrando predicciones: {e}")
            return {
                'success': False,
                'error': str(e),
                'predictions': all_predictions,
                'recommendations': []
            }
        
        # Formatear resultado final
        logger.info("\n" + "="*80)
        logger.info("FLUJO COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        
        return {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'games_processed': all_predictions.get('games_processed', 0),
            'total_predictions': total_preds,
            'viable_predictions': len(viable_predictions),
            'filter_rate': len(viable_predictions) / total_preds if total_preds > 0 else 0,
            'recommendations': [
                {
                    'player_name': pred.player_name,
                    'team_name': pred.team_name,
                    'bet_type': pred.bet_type,
                    'line': pred.line,
                    'prediction': pred.prediction,
                    'recommendation': pred.recommendation,
                    'difference': pred.difference,
                    'confidence_percentage': pred.confidence_percentage,
                    'final_confidence': pred.final_confidence,
                    'over_odds': pred.over_odds,
                    'under_odds': pred.under_odds,
                    'edge': pred.edge,
                    'expected_value': pred.expected_value,
                    'kelly_fraction': pred.kelly_fraction,
                    'game_info': pred.game_info
                }
                for pred in viable_predictions
            ],
            'full_predictions': all_predictions
        }
    
    def _get_today_date(self) -> str:
        """
        Obtiene la fecha de HOY en formato YYYY-MM-DD.
        
        Returns:
            Fecha de hoy como string
        """
        return datetime.now().strftime("%Y-%m-%d")
    
    def _get_predictions_for_today(self) -> Dict[str, Any]:
        """
        Obtiene todas las predicciones para HOY.
        
        Returns:
            Dict con predicciones de jugadores y equipos
        """
        today = self._get_today_date()
        logger.info(f"\n[PREDICCIONES] Obteniendo predicciones para {today}")
        
        try:
            # Ejecutar unified predictor
            predictions = self.unified_predictor.predict_all()
            
            logger.info(f"[OK] Predicciones obtenidas exitosamente")
            return predictions
            
        except Exception as e:
            logger.error(f"[ERROR] Error obteniendo predicciones: {e}")
            return {'players': {}, 'teams': {}}
    
    def _get_odds_for_today(self) -> Dict[str, Any]:
        """
        Obtiene todas las odds para HOY.
        
        Returns:
            Dict con odds de jugadores y equipos
        """
        today = self._get_today_date()
        logger.info(f"\n[ODDS] Obteniendo odds para {today}")
        
        odds_data = {
            'player_props': {},
            'team_odds': {}
        }
        
        try:
            # 1. Obtener Player Props
            logger.info("[ODDS] Obteniendo player props...")
            player_props_result = self.sportradar.get_player_props_by_date(today)
            
            if player_props_result.get('success'):
                events = player_props_result['data']['events']
                logger.info(f"[OK] Player props: {len(events)} eventos")
                
                # Organizar por jugador
                for event in events:
                    for player_data in event['players']:
                        player_name = player_data['player_name']
                        odds_data['player_props'][player_name] = {
                            'event_id': event['sport_event_id'],
                            'home_team': event['home_team'],
                            'away_team': event['away_team'],
                            'props': player_data['props']
                        }
            else:
                logger.warning(f"[WARNING] No se pudieron obtener player props: {player_props_result.get('error')}")
            
            # 2. Obtener Team Odds
            logger.info("[ODDS] Obteniendo team odds...")
            team_odds_result = self.sportradar.get_team_odds_by_date(today)
            
            if team_odds_result.get('success'):
                events = team_odds_result['data']['events']
                logger.info(f"[OK] Team odds: {len(events)} eventos")
                
                # Organizar por evento
                for event in events:
                    event_id = event['sport_event_id']
                    odds_data['team_odds'][event_id] = {
                        'home_team': event['home_team'],
                        'away_team': event['away_team'],
                        'odds': event['odds']
                    }
            else:
                logger.warning(f"[WARNING] No se pudieron obtener team odds: {team_odds_result.get('error')}")
            
            logger.info(f"[OK] Odds obtenidas: {len(odds_data['player_props'])} jugadores, {len(odds_data['team_odds'])} partidos")
            return odds_data
            
        except Exception as e:
            logger.error(f"[ERROR] Error obteniendo odds: {e}")
            return odds_data
    
    def _match_predictions_with_odds(
        self,
        predictions: Dict[str, Any],
        odds_data: Dict[str, Any]
    ) -> List[Dict]:
        """
        Une predicciones con odds para crear oportunidades evaluables.
        
        Args:
            predictions: Predicciones del modelo
            odds_data: Odds de Sportradar
            
        Returns:
            Lista de oportunidades para evaluar
        """
        logger.info("\n[MATCHING] Uniendo predicciones con odds...")
        
        opportunities = []
        
        # TODO: Implementar lógica de matching específica
        # Por ahora retornamos ejemplo básico
        
        # Matching de Player Props
        player_predictions = predictions.get('players', {})
        player_odds = odds_data.get('player_props', {})
        
        for player_name, pred_data in player_predictions.items():
            if player_name in player_odds:
                odds = player_odds[player_name]
                
                # Para cada target predicho
                for target in ['PTS', 'AST', 'TRB', '3PT']:
                    if target in pred_data and target in odds['props']:
                        pred = pred_data[target]
                        odd = odds['props'][target]
                        
                        # Determinar bet_type (over o under)
                        if pred.get('predicted_value', 0) > odd['line']:
                            bet_type = 'over'
                            odds_decimal = odd['over']
                        else:
                            bet_type = 'under'
                            odds_decimal = odd['under']
                        
                        opportunities.append({
                            'target': target,
                            'game_id': odds['event_id'],
                            'player_name': player_name,
                            'line': odd['line'],
                            'bet_type': bet_type,
                            'odds_decimal': odds_decimal,
                            'odds_over': odd['over'],
                            'odds_under': odd['under'],
                            'bookmaker': odd.get('book', 'Unknown'),
                            'predicted_value': pred.get('predicted_value', 0),
                            'model_probability': pred.get('probability', 0.5),
                            'model_confidence': pred.get('confidence', 0.5)
                        })
        
        logger.info(f"[OK] {len(opportunities)} oportunidades creadas")
        return opportunities
    
    def filter_predictions(
        self,
        predictions: Dict[str, Any],
        game_data: Dict[str, Any],
        max_bets: int = 10,
        min_confidence: float = 70.0
    ) -> Dict[str, Any]:
        """
        FILTRO PRINCIPAL: Evalúa predicciones y decide cuáles apostar.
        
        Args:
            predictions: Predicciones de UnifiedPredictor
            game_data: Datos del partido (para extraer fecha/equipos)
            max_bets: Número máximo de apuestas a recomendar
            min_confidence: Confianza mínima requerida (0-100)
            
        Returns:
            Dict con predicciones filtradas y recomendaciones
        """
        logger.info("\n" + "="*80)
        logger.info("FILTRO DE BETTING ANALYTICS")
        logger.info("="*80)
        
        # Extraer fecha del game_data
        game_date = self._extract_date_from_game(game_data)
        
        return self._process_predictions(
            predictions=predictions,
            game_data=game_data,
            game_date=game_date,
            max_bets=max_bets,
            min_confidence=min_confidence
        )
    
    def get_betting_recommendations(
        self,
        max_bets: int = 10,
        min_confidence: float = 70.0
    ) -> Dict[str, Any]:
        """
        MÉTODO STANDALONE: Obtiene recomendaciones de apuestas para HOY.
        (Usado cuando NO hay predicciones previas)
        
        Args:
            max_bets: Número máximo de apuestas a recomendar
            min_confidence: Confianza mínima requerida (0-100)
            
        Returns:
            Dict con recomendaciones completas
        """
        today = self._get_today_date()
        
        logger.info("\n" + "="*80)
        logger.info(f"GENERANDO RECOMENDACIONES PARA: {today}")
        logger.info("="*80)
        
        # 1. Obtener predicciones
        predictions = self._get_predictions_for_today()
        
        # 2. Obtener odds
        odds_data = self._get_odds_for_today()
        
        # 3. Unir predicciones con odds
        opportunities_data = self._match_predictions_with_odds(predictions, odds_data)
        
        if not opportunities_data:
            logger.warning("[WARNING] No hay oportunidades para evaluar")
            return {
                'date': today,
                'total_opportunities': 0,
                'recommended_bets': [],
                'summary': {
                    'total_opportunities': 0,
                    'recommended_bets': 0
                }
            }
        
        # 4. Evaluar todas las oportunidades
        logger.info(f"\n[EVALUACIÓN] Evaluando {len(opportunities_data)} oportunidades...")
        evaluated_opportunities = self.analytics.evaluate_multiple_opportunities(opportunities_data)
        
        # 5. Filtrar mejores apuestas
        best_bets = self.analytics.filter_best_bets(
            evaluated_opportunities,
            max_bets=max_bets,
            min_confidence=min_confidence
        )
        
        # 6. Generar reporte resumen
        summary = self.analytics.generate_summary_report(evaluated_opportunities)
        
        logger.info(f"\n[OK] EVALUACIÓN COMPLETADA")
        logger.info(f"  Total Oportunidades: {summary['total_opportunities']}")
        logger.info(f"  Recomendadas: {summary['recommended_bets']}")
        logger.info(f"  Edge Promedio: {summary['avg_edge']*100:.2f}%")
        logger.info(f"  EV Promedio: {summary['avg_ev']*100:.2f}%")
        
        # 7. Formatear resultado
        result = {
            'date': today,
            'total_opportunities': len(evaluated_opportunities),
            'recommended_bets': [self._format_bet(bet) for bet in best_bets],
            'summary': summary,
            'all_opportunities': [self._format_bet(opp) for opp in evaluated_opportunities]
        }
        
        logger.info("="*80)
        logger.info("[OK] RECOMENDACIONES GENERADAS EXITOSAMENTE")
        logger.info("="*80)
        
        return result
    
    def _format_bet(self, opportunity: FilteredPrediction) -> Dict[str, Any]:
        """
        Formatea una BettingOpportunity a dict para API response.
        
        Args:
            opportunity: BettingOpportunity evaluada
            
        Returns:
            Dict con información formateada
        """
        return {
            'target': opportunity.target,
            'player_name': opportunity.player_name,
            'team_name': opportunity.team_name,
            'game_id': opportunity.game_id,
            'line': opportunity.line,
            'bet_type': opportunity.bet_type,
            'odds_decimal': opportunity.odds_decimal,
            'bookmaker': opportunity.bookmaker,
            'predicted_value': opportunity.predicted_value,
            'model_probability': round(opportunity.model_probability, 4),
            'model_confidence': round(opportunity.model_confidence, 4),
            'implied_probability': round(opportunity.implied_probability, 4),
            'edge': round(opportunity.edge, 4),
            'expected_value': round(opportunity.expected_value, 4),
            'kelly_fractional': round(opportunity.kelly_fractional, 4),
            'bet_size_pct': round(opportunity.bet_size_pct, 4),
            'confidence_score': round(opportunity.confidence_score, 2),
            'risk_level': opportunity.risk_level,
            'should_bet': opportunity.should_bet,
            'recommendation': opportunity.recommendation
        }
    
    def print_recommendations(self, recommendations: Dict[str, Any]):
        """
        Imprime recomendaciones de forma legible.
        
        Args:
            recommendations: Dict con recomendaciones
        """
        print("\n" + "="*80)
        print(f"RECOMENDACIONES DE APUESTAS - {recommendations['date']}")
        print("="*80)
        
        summary = recommendations['summary']
        print(f"\nRESUMEN:")
        print(f"  Total Oportunidades: {summary['total_opportunities']}")
        print(f"  Apuestas Recomendadas: {summary['recommended_bets']}")
        print(f"  Tasa de Éxito: {summary['recommendation_rate']*100:.1f}%")
        print(f"  Edge Promedio: {summary['avg_edge']*100:.2f}%")
        print(f"  EV Promedio: {summary['avg_ev']*100:.2f}%")
        print(f"  Asignación Kelly Total: {summary['total_kelly_allocation']*100:.2f}%")
        
        recommended = recommendations['recommended_bets']
        
        if not recommended:
            print("\n[INFO] No hay apuestas recomendadas para hoy")
            return
        
        print(f"\n{'='*80}")
        print(f"TOP {len(recommended)} APUESTAS RECOMENDADAS")
        print("="*80)
        
        for idx, bet in enumerate(recommended, 1):
            print(f"\n[{idx}] {bet['target']} - {bet['player_name'] or bet['team_name']}")
            print(f"    Línea: {bet['line']} | {bet['bet_type'].upper()}")
            print(f"    Odds: {bet['odds_decimal']:.2f} | {bet['bookmaker']}")
            print(f"    Predicción: {bet['predicted_value']:.2f}")
            print(f"    Edge: {bet['edge']*100:+.2f}% | EV: {bet['expected_value']*100:+.2f}%")
            print(f"    Apostar: {bet['bet_size_pct']*100:.2f}% del bankroll")
            print(f"    Confianza: {bet['confidence_score']}/100 | Riesgo: {bet['risk_level'].upper()}")
            print(f"    {bet['recommendation']}")
        
        print("\n" + "="*80)


# ============================================================================
# FUNCIONES DE CONVENIENCIA
# ============================================================================

def get_today_betting_recommendations(
    api_key: Optional[str] = None,
    max_bets: int = 10,
    min_confidence: float = 70.0
) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener recomendaciones de HOY.
    
    Args:
        api_key: API key de Sportradar
        max_bets: Número máximo de apuestas
        min_confidence: Confianza mínima (0-100)
        
    Returns:
        Dict con recomendaciones
    """
    predictor = GamePredictor(api_key=api_key)
    return predictor.get_betting_recommendations(
        max_bets=max_bets,
        min_confidence=min_confidence
    )


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("\n" + "="*80)
    print("GAME PREDICTOR - PRUEBA")
    print("="*80)
    
    # API Key (puedes pasarla o usar la de config)
    API_KEY = "EIKAwb4tGmdxpePyXcSlazAsAk4QQeDYn4jEyRq0"
    
    try:
        # Inicializar
        predictor = GamePredictor(api_key=API_KEY)
        
        # Obtener recomendaciones para HOY (automático)
        recommendations = predictor.get_betting_recommendations(
            max_bets=5,
            min_confidence=65.0
        )
        
        # Mostrar resultados
        predictor.print_recommendations(recommendations)
        
    except Exception as e:
        logger.error(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

