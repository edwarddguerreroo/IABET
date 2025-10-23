"""
Bookmakers Integration - BETTING ANALYTICS ENGINE
==================================================

Módulo especializado en cálculos avanzados de betting analytics.
Este módulo es INDEPENDIENTE y solo maneja la lógica matemática.

Técnicas Implementadas:
- Kelly Criterion (Full & Fractional)
- Expected Value (EV)
- Edge Calculation
- Vig/Juice Detection & Removal
- Confidence Scoring
- Risk Management
- Bet Sizing
- ROI Projection

Autor: Sistema NBA
Fecha: 2025-10-21
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FilteredPrediction:
    """
    Representa una predicción filtrada y viable para apostar.
    """
    # Identificación
    player_name: Optional[str]
    team_name: Optional[str]
    bet_type: str                  # 'points', 'rebounds', 'assists', etc.
    
    # Línea y Predicción
    line: float                    # Línea de la casa de apuestas
    prediction: float              # Predicción del modelo
    recommendation: str            # 'OVER' o 'UNDER'
    
    # Métricas de Confianza
    difference: float              # Diferencia absoluta (predicción - línea)
    confidence_percentage: float   # Confianza original del modelo
    final_confidence: float        # Confianza final basada en diferencia
    
    # Odds (si están disponibles)
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None
    
    # Métricas de Betting (opcionales)
    edge: Optional[float] = None
    expected_value: Optional[float] = None
    kelly_fraction: Optional[float] = None
    
    # Metadata
    game_info: Optional[Dict] = None
    original_prediction: Optional[Dict] = None


@dataclass
class BettingOpportunity:
    """
    Representa una oportunidad de apuesta evaluada.
    """
    # Identificación
    target: str                    # PTS, AST, TRB, is_win, etc.
    player_name: Optional[str]     # Nombre del jugador (si aplica)
    team_name: Optional[str]       # Nombre del equipo (si aplica)
    game_id: str                   # ID del partido
    
    # Línea y Odds
    line: float                    # Línea de la casa
    bet_type: str                  # 'over', 'under', 'home', 'away'
    odds_decimal: float            # Odds en formato decimal
    bookmaker: str                 # Casa de apuestas
    
    # Predicción del Modelo
    predicted_value: float         # Valor predicho por el modelo
    model_probability: float       # Probabilidad según modelo (0-1)
    model_confidence: float        # Confianza del modelo (0-1)
    
    # Analytics
    implied_probability: float     # Probabilidad implícita de la casa
    true_probability: float        # Probabilidad real (ajustada por vig)
    edge: float                    # Ventaja sobre la casa
    expected_value: float          # Valor esperado
    kelly_full: float              # Kelly completo
    kelly_fractional: float        # Kelly fraccional (25%)
    
    # Gestión de Riesgo
    bet_size_pct: float           # % del bankroll a apostar
    confidence_score: float        # Score de confianza (0-100)
    risk_level: str               # 'low', 'medium', 'high'
    
    # Recomendación
    should_bet: bool              # ¿Apostar?
    recommendation: str           # Texto de recomendación
    
    # Metadata
    timestamp: datetime
    

class BettingAnalytics:
    """
    Motor de análisis de apuestas con técnicas avanzadas.
    
    Este módulo NO interactúa con predicciones directamente,
    solo recibe datos y realiza cálculos matemáticos.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Inicializa el motor de analytics.
        
        Args:
            config: Configuración de parámetros
        """
        # Parámetros por defecto (pueden ser sobrescritos)
        self.min_edge = config.get('minimum_edge', 0.05) if config else 0.05
        self.min_confidence = config.get('confidence_threshold', 0.65) if config else 0.65
        self.kelly_fraction = config.get('max_kelly_fraction', 0.25) if config else 0.25
        self.min_odds = config.get('min_odds', 1.2) if config else 1.2
        self.max_odds = config.get('max_odds', 10.0) if config else 10.0
        
        logger.info(f"BettingAnalytics inicializado | Edge mínimo: {self.min_edge*100}%")
    
    # =========================================================================
    # CORE CALCULATIONS - CÁLCULOS FUNDAMENTALES
    # =========================================================================
    
    def calculate_implied_probability(self, odds_decimal: float) -> float:
        """
        Calcula la probabilidad implícita de unas odds decimales.
        
        Formula: prob = 1 / odds_decimal
        
        Args:
            odds_decimal: Odds en formato decimal (1.5, 2.0, etc.)
            
        Returns:
            Probabilidad implícita (0-1)
        """
        if odds_decimal <= 0:
            raise ValueError(f"Odds deben ser positivas, recibido: {odds_decimal}")
        
        return 1 / odds_decimal
    
    def calculate_vig(self, odds_over: float, odds_under: float) -> float:
        """
        Calcula el vig (margen de la casa) de un mercado.
        
        El vig es el margen que cobra la casa, representado por la diferencia
        entre la suma de probabilidades implícitas y 100%.
        
        Args:
            odds_over: Odds del over
            odds_under: Odds del under
            
        Returns:
            Vig como decimal (0.05 = 5% vig)
        """
        prob_over = self.calculate_implied_probability(odds_over)
        prob_under = self.calculate_implied_probability(odds_under)
        
        total_prob = prob_over + prob_under
        vig = total_prob - 1.0
        
        return max(0, vig)  # Vig no puede ser negativo
    
    def remove_vig(self, odds_decimal: float, total_vig: float) -> float:
        """
        Elimina el vig de unas odds para obtener la probabilidad "justa".
        
        Args:
            odds_decimal: Odds con vig incluido
            total_vig: Vig total del mercado
            
        Returns:
            Probabilidad sin vig (ajustada)
        """
        prob_with_vig = self.calculate_implied_probability(odds_decimal)
        prob_without_vig = prob_with_vig / (1 + total_vig)
        
        return prob_without_vig
    
    def calculate_edge(self, model_prob: float, implied_prob: float) -> float:
        """
        Calcula el edge (ventaja) sobre la casa.
        
        Edge = Prob_Modelo - Prob_Casa
        
        Args:
            model_prob: Probabilidad según nuestro modelo
            implied_prob: Probabilidad implícita de la casa
            
        Returns:
            Edge como decimal (0.15 = 15% edge)
        """
        return model_prob - implied_prob
    
    def calculate_expected_value(self, model_prob: float, odds_decimal: float) -> float:
        """
        Calcula el Expected Value (EV) de una apuesta.
        
        EV = (Prob_Ganar * Ganancia) - Prob_Perder
        EV = (Prob * Odds) - 1
        
        Args:
            model_prob: Probabilidad de ganar según modelo
            odds_decimal: Odds de la apuesta
            
        Returns:
            EV como decimal (0.20 = 20% EV positivo)
        """
        ev = (model_prob * odds_decimal) - 1
        return ev
    
    def calculate_kelly_criterion(
        self, 
        model_prob: float, 
        odds_decimal: float,
        fraction: float = 1.0
    ) -> float:
        """
        Calcula el Kelly Criterion para bet sizing óptimo.
        
        Formula: Kelly = (p * b - q) / b
        Donde:
            p = probabilidad de ganar
            q = probabilidad de perder (1 - p)
            b = odds - 1 (ganancia neta por unidad apostada)
        
        Args:
            model_prob: Probabilidad de ganar según modelo
            odds_decimal: Odds decimales
            fraction: Fracción de Kelly a usar (0.25 = Quarter Kelly)
            
        Returns:
            % del bankroll a apostar (0.1 = 10%)
        """
        if odds_decimal <= 1:
            return 0.0
        
        p = model_prob
        q = 1 - p
        b = odds_decimal - 1
        
        kelly = (p * b - q) / b
        
        # Aplicar fracción
        kelly_fractional = kelly * fraction
        
        # Límites de seguridad
        return max(0.0, min(kelly_fractional, 0.25))  # Max 25% del bankroll
    
    # =========================================================================
    # ADVANCED ANALYTICS - ANÁLISIS AVANZADO
    # =========================================================================
    
    def calculate_confidence_score(
        self,
        edge: float,
        ev: float,
        model_confidence: float,
        odds_decimal: float
    ) -> float:
        """
        Calcula un score de confianza compuesto (0-100).
        
        Considera:
        - Edge sobre la casa
        - Expected Value
        - Confianza del modelo
        - Razonabilidad de las odds
        
        Args:
            edge: Edge calculado
            ev: Expected Value
            model_confidence: Confianza del modelo (0-1)
            odds_decimal: Odds de la apuesta
            
        Returns:
            Score de confianza (0-100)
        """
        # Componente 1: Edge (30%)
        edge_score = min(abs(edge) / 0.30, 1.0) * 30  # Max a 30% edge
        
        # Componente 2: EV (30%)
        ev_score = min(abs(ev) / 0.50, 1.0) * 30  # Max a 50% EV
        
        # Componente 3: Confianza del modelo (30%)
        confidence_score = model_confidence * 30
        
        # Componente 4: Razonabilidad de odds (10%)
        # Penalizar odds muy altas (improbables) o muy bajas (poco valor)
        if 1.5 <= odds_decimal <= 3.0:
            odds_score = 10
        elif 1.2 <= odds_decimal <= 5.0:
            odds_score = 5
        else:
            odds_score = 0
        
        total_score = edge_score + ev_score + confidence_score + odds_score
        
        return min(100, max(0, total_score))
    
    def calculate_risk_level(
        self,
        kelly_pct: float,
        odds_decimal: float,
        confidence_score: float
    ) -> str:
        """
        Determina el nivel de riesgo de una apuesta.
        
        Args:
            kelly_pct: % Kelly sugerido
            odds_decimal: Odds de la apuesta
            confidence_score: Score de confianza
            
        Returns:
            'low', 'medium', o 'high'
        """
        risk_points = 0
        
        # Factor 1: Kelly size
        if kelly_pct > 0.15:
            risk_points += 2
        elif kelly_pct > 0.10:
            risk_points += 1
        
        # Factor 2: Odds
        if odds_decimal > 5.0:
            risk_points += 2
        elif odds_decimal > 3.0:
            risk_points += 1
        
        # Factor 3: Confidence
        if confidence_score < 50:
            risk_points += 2
        elif confidence_score < 70:
            risk_points += 1
        
        # Clasificación
        if risk_points >= 4:
            return 'high'
        elif risk_points >= 2:
            return 'medium'
        else:
            return 'low'
    
    def calculate_roi_projection(
        self,
        ev: float,
        num_bets: int = 100
    ) -> Dict[str, float]:
        """
        Proyecta el ROI esperado a largo plazo.
        
        Args:
            ev: Expected Value de la apuesta
            num_bets: Número de apuestas para proyección
            
        Returns:
            Dict con proyecciones de ROI
        """
        return {
            'ev_per_bet': ev,
            'roi_100_bets': ev * num_bets,
            'roi_500_bets': ev * 500,
            'roi_1000_bets': ev * 1000
        }
    
    # =========================================================================
    # MAIN EVALUATION - EVALUACIÓN PRINCIPAL
    # =========================================================================
    
    def evaluate_betting_opportunity(
        self,
        # Identificación
        target: str,
        game_id: str,
        player_name: Optional[str] = None,
        team_name: Optional[str] = None,
        
        # Odds de la casa
        line: float = None,
        bet_type: str = None,  # 'over', 'under', 'home', 'away'
        odds_decimal: float = None,
        odds_over: float = None,  # Para calcular vig
        odds_under: float = None,  # Para calcular vig
        bookmaker: str = "Unknown",
        
        # Predicción del modelo
        predicted_value: float = None,
        model_probability: float = None,
        model_confidence: float = None
        
    ) -> BettingOpportunity:
        """
        Evalúa completamente una oportunidad de apuesta.
        
        Este es el método principal que integra todos los cálculos.
        
        Args:
            target: Target de apuesta (PTS, AST, is_win, etc.)
            game_id: ID del partido
            player_name: Nombre del jugador (opcional)
            team_name: Nombre del equipo (opcional)
            line: Línea de la casa
            bet_type: Tipo de apuesta
            odds_decimal: Odds en formato decimal
            odds_over: Odds del over (para calcular vig)
            odds_under: Odds del under (para calcular vig)
            bookmaker: Casa de apuestas
            predicted_value: Valor predicho por modelo
            model_probability: Probabilidad según modelo (0-1)
            model_confidence: Confianza del modelo (0-1)
            
        Returns:
            BettingOpportunity completa con todos los análisis
        """
        # Validaciones
        if odds_decimal is None or odds_decimal <= 0:
            raise ValueError(f"Odds decimales inválidas: {odds_decimal}")
        
        if model_probability is None or not 0 <= model_probability <= 1:
            raise ValueError(f"Probabilidad del modelo inválida: {model_probability}")
        
        # 1. Calcular probabilidad implícita
        implied_prob = self.calculate_implied_probability(odds_decimal)
        
        # 2. Calcular vig y probabilidad real (si tenemos over/under)
        if odds_over and odds_under:
            vig = self.calculate_vig(odds_over, odds_under)
            true_prob = self.remove_vig(odds_decimal, vig)
        else:
            vig = 0.0
            true_prob = implied_prob
        
        # 3. Calcular edge
        edge = self.calculate_edge(model_probability, implied_prob)
        
        # 4. Calcular Expected Value
        ev = self.calculate_expected_value(model_probability, odds_decimal)
        
        # 5. Calcular Kelly Criterion
        kelly_full = self.calculate_kelly_criterion(model_probability, odds_decimal, fraction=1.0)
        kelly_frac = self.calculate_kelly_criterion(model_probability, odds_decimal, fraction=self.kelly_fraction)
        
        # 6. Calcular confidence score
        conf_score = self.calculate_confidence_score(
            edge, ev, model_confidence or 0.5, odds_decimal
        )
        
        # 7. Determinar tamaño de apuesta
        if edge >= self.min_edge and ev > 0 and conf_score >= 50:
            bet_size = kelly_frac
        else:
            bet_size = 0.0
        
        # 8. Calcular nivel de riesgo
        risk = self.calculate_risk_level(bet_size, odds_decimal, conf_score)
        
        # 9. Decisión de apuesta
        should_bet = (
            edge >= self.min_edge and
            ev > 0 and
            conf_score >= (self.min_confidence * 100) and
            self.min_odds <= odds_decimal <= self.max_odds and
            bet_size > 0
        )
        
        # 10. Generar recomendación
        if should_bet:
            recommendation = (
                f"✅ APOSTAR {bet_size*100:.1f}% del bankroll | "
                f"Edge: {edge*100:.1f}% | EV: {ev*100:.1f}% | "
                f"Confianza: {conf_score:.0f}/100"
            )
        elif edge > 0 and ev > 0:
            recommendation = (
                f"⚠️  Edge positivo pero insuficiente | "
                f"Edge: {edge*100:.1f}% (mín: {self.min_edge*100:.1f}%) | "
                f"Confianza: {conf_score:.0f}/100"
            )
        else:
            recommendation = (
                f"❌ NO APOSTAR | "
                f"Edge: {edge*100:.1f}% | EV: {ev*100:.1f}%"
            )
        
        # Crear objeto de oportunidad
        opportunity = BettingOpportunity(
            # Identificación
            target=target,
            player_name=player_name,
            team_name=team_name,
            game_id=game_id,
            
            # Línea y Odds
            line=line,
            bet_type=bet_type,
            odds_decimal=odds_decimal,
            bookmaker=bookmaker,
            
            # Predicción
            predicted_value=predicted_value,
            model_probability=model_probability,
            model_confidence=model_confidence or 0.5,
            
            # Analytics
            implied_probability=implied_prob,
            true_probability=true_prob,
            edge=edge,
            expected_value=ev,
            kelly_full=kelly_full,
            kelly_fractional=kelly_frac,
            
            # Gestión de Riesgo
            bet_size_pct=bet_size,
            confidence_score=conf_score,
            risk_level=risk,
            
            # Recomendación
            should_bet=should_bet,
            recommendation=recommendation,
            
            # Metadata
            timestamp=datetime.now()
        )
        
        return opportunity
    
    # =========================================================================
    # BATCH PROCESSING - PROCESAMIENTO EN LOTE
    # =========================================================================
    
    def evaluate_multiple_opportunities(
        self,
        opportunities: List[Dict]
    ) -> List[BettingOpportunity]:
        """
        Evalúa múltiples oportunidades en batch.
        
        Args:
            opportunities: Lista de dicts con datos de oportunidades
            
        Returns:
            Lista de BettingOpportunity evaluadas
        """
        evaluated = []
        
        for opp in opportunities:
            try:
                result = self.evaluate_betting_opportunity(**opp)
                evaluated.append(result)
            except Exception as e:
                logger.error(f"Error evaluando oportunidad {opp.get('target')}: {e}")
                continue
        
        return evaluated
    
    def rank_opportunities(
        self,
        opportunities: List[BettingOpportunity],
        sort_by: str = 'confidence_score'
    ) -> List[BettingOpportunity]:
        """
        Ordena oportunidades por criterio especificado.
        
        Args:
            opportunities: Lista de oportunidades evaluadas
            sort_by: Criterio ('edge', 'ev', 'confidence_score', 'kelly_fractional')
            
        Returns:
            Lista ordenada de mayor a menor
        """
        return sorted(
            opportunities,
            key=lambda x: getattr(x, sort_by),
            reverse=True
        )
    
    def filter_best_bets(
        self,
        opportunities: List[BettingOpportunity],
        max_bets: int = 10,
        min_confidence: float = None
    ) -> List[BettingOpportunity]:
        """
        Filtra las mejores apuestas según criterios.
        
        Args:
            opportunities: Lista de oportunidades
            max_bets: Número máximo de apuestas a retornar
            min_confidence: Confianza mínima requerida
            
        Returns:
            Lista filtrada de mejores apuestas
        """
        # Filtrar solo las que se recomienda apostar
        valid_bets = [opp for opp in opportunities if opp.should_bet]
        
        # Filtrar por confianza si se especifica
        if min_confidence:
            valid_bets = [
                opp for opp in valid_bets 
                if opp.confidence_score >= min_confidence
            ]
        
        # Ordenar por confidence score
        ranked = self.rank_opportunities(valid_bets, sort_by='confidence_score')
        
        # Retornar top N
        return ranked[:max_bets]
    
    # =========================================================================
    # REPORTING - REPORTES
    # =========================================================================
    
    def generate_summary_report(
        self,
        opportunities: List[BettingOpportunity]
    ) -> Dict[str, Any]:
        """
        Genera un reporte resumen de oportunidades.
        
        Args:
            opportunities: Lista de oportunidades evaluadas
            
        Returns:
            Dict con estadísticas resumidas
        """
        if not opportunities:
            return {
                'total_opportunities': 0,
                'recommended_bets': 0,
                'avg_edge': 0,
                'avg_ev': 0,
                'total_kelly_allocation': 0
            }
        
        recommended = [opp for opp in opportunities if opp.should_bet]
        
        return {
            'total_opportunities': len(opportunities),
            'recommended_bets': len(recommended),
            'recommendation_rate': len(recommended) / len(opportunities) if opportunities else 0,
            
            # Estadísticas de edge
            'avg_edge': np.mean([opp.edge for opp in recommended]) if recommended else 0,
            'max_edge': max([opp.edge for opp in recommended]) if recommended else 0,
            'min_edge': min([opp.edge for opp in recommended]) if recommended else 0,
            
            # Estadísticas de EV
            'avg_ev': np.mean([opp.expected_value for opp in recommended]) if recommended else 0,
            'total_ev': sum([opp.expected_value for opp in recommended]) if recommended else 0,
            
            # Estadísticas de Kelly
            'total_kelly_allocation': sum([opp.bet_size_pct for opp in recommended]) if recommended else 0,
            'avg_bet_size': np.mean([opp.bet_size_pct for opp in recommended]) if recommended else 0,
            
            # Confianza
            'avg_confidence': np.mean([opp.confidence_score for opp in recommended]) if recommended else 0,
            
            # Distribución de riesgo
            'low_risk_count': len([o for o in recommended if o.risk_level == 'low']),
            'medium_risk_count': len([o for o in recommended if o.risk_level == 'medium']),
            'high_risk_count': len([o for o in recommended if o.risk_level == 'high'])
        }
    
    def print_opportunity(self, opp: BettingOpportunity):
        """
        Imprime una oportunidad de forma legible.
        
        Args:
            opp: BettingOpportunity a imprimir
        """
        print("="*80)
        print(f"TARGET: {opp.target} | {opp.bet_type.upper()}")
        if opp.player_name:
            print(f"Jugador: {opp.player_name}")
        if opp.team_name:
            print(f"Equipo: {opp.team_name}")
        print(f"Línea: {opp.line} | Odds: {opp.odds_decimal:.2f} | Book: {opp.bookmaker}")
        print("-"*80)
        print(f"Predicción Modelo: {opp.predicted_value:.2f}")
        print(f"Prob Modelo: {opp.model_probability*100:.1f}%")
        print(f"Prob Casa: {opp.implied_probability*100:.1f}%")
        print(f"Confianza Modelo: {opp.model_confidence*100:.1f}%")
        print("-"*80)
        print(f"✅ EDGE: {opp.edge*100:+.2f}%")
        print(f"✅ EV: {opp.expected_value*100:+.2f}%")
        print(f"✅ Kelly Full: {opp.kelly_full*100:.2f}%")
        print(f"✅ Kelly Frac: {opp.kelly_fractional*100:.2f}%")
        print(f"📊 Confidence Score: {opp.confidence_score:.0f}/100")
        print(f"⚠️  Risk Level: {opp.risk_level.upper()}")
        print("-"*80)
        print(f"💰 Apuesta Sugerida: {opp.bet_size_pct*100:.2f}% del bankroll")
        print(f"🎯 {opp.recommendation}")
        print("="*80)
    
    def filter_predictions(
        self, 
        unified_predictions: Dict,
        odds_data: Optional[Dict] = None,
        min_difference: float = 2.0,
        min_confidence: float = 70.0
    ) -> List[FilteredPrediction]:
        """
        Filtra predicciones del unified_predictor y retorna solo las viables.
        
        LÓGICA SIMPLE:
        1. Calcula diferencia = predicción - línea
        2. Si diferencia >= min_difference → VIABLE
        3. Determina OVER (predicción > línea) o UNDER (predicción < línea)
        4. Confianza final = diferencia absoluta + confianza del modelo
        
        Args:
            unified_predictions: Salida del UnifiedPredictor
            odds_data: Datos de odds de Sportradar (opcional)
            min_difference: Diferencia mínima para considerar viable
            min_confidence: Confianza mínima del modelo
            
        Returns:
            Lista de FilteredPrediction viables
        """
        filtered = []
        
        # Procesar cada juego
        for game in unified_predictions.get('predictions', []):
            game_info = game.get('game_info', {})
            
            # Procesar predicciones de jugadores
            for player_pred in game.get('player_predictions', []):
                player_name = player_pred.get('player_name')
                
                # Procesar cada tipo de predicción (points, rebounds, etc.)
                for bet_type, pred_data in player_pred.get('predictions', {}).items():
                    
                    # Extraer datos
                    bet_line_raw = pred_data.get('bet_line', 0)
                    
                    # Saltar predicciones categóricas (double_double, is_win, etc.)
                    if isinstance(bet_line_raw, str) and bet_line_raw.lower() in ['yes', 'no', 'true', 'false']:
                        logger.debug(f"Saltando predicción categórica: {player_name} {bet_type}")
                        continue
                    
                    try:
                        line = float(bet_line_raw)
                    except (ValueError, TypeError):
                        logger.debug(f"No se pudo convertir bet_line a float: {bet_line_raw}")
                        continue
                    
                    prediction = pred_data.get('prediction_details', {}).get('final_prediction', 0)
                    confidence = pred_data.get('confidence_percentage', 0)
                    
                    # Validar que tenemos los datos necesarios
                    if line == 0 or prediction == 0:
                        continue
                    
                    # Calcular diferencia
                    difference = prediction - line
                    abs_difference = abs(difference)
                    
                    # Filtro 1: Confianza mínima del modelo
                    if confidence < min_confidence:
                        logger.debug(f"Filtrado {player_name} {bet_type}: confianza {confidence:.1f}% < {min_confidence}%")
                        continue
                    
                    # Filtro 2: Diferencia mínima
                    if abs_difference < min_difference:
                        logger.debug(f"Filtrado {player_name} {bet_type}: diferencia {abs_difference:.1f} < {min_difference}")
                        continue
                    
                    # Determinar recomendación (OVER o UNDER)
                    recommendation = 'OVER' if difference > 0 else 'UNDER'
                    
                    # Calcular confianza final
                    # Fórmula: confianza_modelo * (1 + diferencia_normalizada)
                    # Diferencia normalizada: diferencia / línea
                    difference_factor = abs_difference / line if line > 0 else 0
                    final_confidence = min(confidence * (1 + difference_factor), 99.9)
                    
                    # Buscar odds si están disponibles
                    over_odds = None
                    under_odds = None
                    edge = None
                    ev = None
                    kelly = None
                    
                    if odds_data:
                        # Buscar odds para este jugador/tipo
                        player_odds = self._find_odds(odds_data, player_name, bet_type, line)
                        if player_odds:
                            over_odds = player_odds.get('over')
                            under_odds = player_odds.get('under')
                            
                            # Calcular métricas de betting si tenemos odds
                            selected_odds = over_odds if recommendation == 'OVER' else under_odds
                            if selected_odds:
                                implied_prob = self.calculate_implied_probability(selected_odds)
                                model_prob = confidence / 100.0
                                edge = model_prob - implied_prob
                                ev = self.calculate_expected_value(model_prob, selected_odds, stake=100)
                                kelly = self.calculate_kelly_criterion(model_prob, selected_odds)
                    
                    # Crear predicción filtrada
                    filtered_pred = FilteredPrediction(
                        player_name=player_name,
                        team_name=None,
                        bet_type=bet_type,
                        line=line,
                        prediction=prediction,
                        recommendation=recommendation,
                        difference=difference,
                        confidence_percentage=confidence,
                        final_confidence=final_confidence,
                        over_odds=over_odds,
                        under_odds=under_odds,
                        edge=edge,
                        expected_value=ev,
                        kelly_fraction=kelly,
                        game_info=game_info,
                        original_prediction=pred_data
                    )
                    
                    filtered.append(filtered_pred)
                    
                    logger.info(
                        f"VIABLE: {player_name} {bet_type.upper()} | "
                        f"Linea: {line} | Prediccion: {prediction:.1f} | "
                        f"{recommendation} | Diferencia: {abs_difference:.1f} | "
                        f"Confianza Final: {final_confidence:.1f}%"
                    )
        
        logger.info(f"\nRESUMEN FILTRADO: {len(filtered)} predicciones viables de {self._count_total_predictions(unified_predictions)} totales")
        
        return filtered
    
    def _find_odds(self, odds_data: Dict, player_name: str, bet_type: str, line: float) -> Optional[Dict]:
        """
        Busca odds para un jugador/tipo específico.
        """
        # TODO: Implementar búsqueda en datos de Sportradar
        # Por ahora retorna None
        return None
    
    def _count_total_predictions(self, unified_predictions: Dict) -> int:
        """
        Cuenta el total de predicciones en la salida del unified_predictor.
        """
        total = 0
        for game in unified_predictions.get('predictions', []):
            for player_pred in game.get('player_predictions', []):
                total += len(player_pred.get('predictions', {}))
        return total


# Instancia global (opcional, para usar como singleton)
_analytics_instance = None

def get_analytics_engine(config: Optional[Dict] = None) -> BettingAnalytics:
    """
    Obtiene instancia singleton del motor de analytics.
    
    Args:
        config: Configuración (solo se usa en primera llamada)
        
    Returns:
        Instancia de BettingAnalytics
    """
    global _analytics_instance
    
    if _analytics_instance is None:
        _analytics_instance = BettingAnalytics(config)
    
    return _analytics_instance
