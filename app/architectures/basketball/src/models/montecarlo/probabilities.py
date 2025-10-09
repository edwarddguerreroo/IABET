"""
Calculador de Probabilidades Derivadas NBA
==========================================

Calcula probabilidades específicas para apuestas deportivas basadas
en simulaciones Monte Carlo.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ProbabilityCalculator:
    """
    Calculador de probabilidades derivadas para apuestas NBA.
    
    Calcula probabilidades para:
    - Over/Under por jugador y equipo
    - Double-doubles y triple-doubles
    - Resultados de partido
    - Combinaciones específicas
    """
    
    def __init__(self):
        """Inicializa el calculador de probabilidades."""
        # Líneas típicas de apuestas NBA
        self.common_lines = {
            'PTS': [10.5, 15.5, 20.5, 25.5],
            'TRB': [7.5, 10.5, 12.5, 15.5],
            'AST': [5.5, 7.5, 10.5, 12.5],
            'STL': [1.5, 2.5],
            'BLK': [0.5, 1.5, 2.5],
            'total_points': [210.5, 220.5, 230.5, 240.5]
        }
        
        logger.info("Calculador de probabilidades NBA inicializado")
    
    def calculate_player_probabilities(self, 
                                     simulations: pd.DataFrame,
                                     player_name: str,
                                     custom_lines: Optional[Dict[str, List[float]]] = None) -> Dict[str, any]:
        """
        Calcula probabilidades para un jugador específico.
        
        Args:
            simulations: DataFrame con simulaciones del jugador
            player_name: Nombre del jugador
            custom_lines: Líneas personalizadas de apuestas
            
        Returns:
            Diccionario con probabilidades calculadas
        """
        lines = custom_lines if custom_lines else self.common_lines
        probabilities = {'player': player_name}
        
        # Over/Under para estadísticas principales
        for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
            if stat in simulations.columns:
                stat_probs = self._calculate_over_under_probs(
                    simulations[stat], lines.get(stat, [])
                )
                probabilities[f'{stat}_over_under'] = stat_probs
        
        # Probabilidades especiales
        probabilities['special_events'] = self._calculate_special_events(simulations)
        
        # Rangos de rendimiento
        probabilities['performance_ranges'] = self._calculate_performance_ranges(simulations)
        
        # Combinaciones populares
        probabilities['combinations'] = self._calculate_combinations(simulations)
        
        return probabilities
    
    def calculate_team_probabilities(self, 
                                   team_simulations: pd.DataFrame,
                                   team_name: str,
                                   custom_lines: Optional[Dict[str, List[float]]] = None) -> Dict[str, any]:
        """
        Calcula probabilidades para un equipo.
        
        Args:
            team_simulations: DataFrame con simulaciones del equipo
            team_name: Nombre del equipo
            custom_lines: Líneas personalizadas
            
        Returns:
            Diccionario con probabilidades del equipo
        """
        lines = custom_lines if custom_lines else self.common_lines
        probabilities = {'team': team_name}
        
        # Over/Under para puntos del equipo
        if 'PTS' in team_simulations.columns:
            team_lines = [100.5, 110.5, 120.5, 130.5]
            probabilities['points_over_under'] = self._calculate_over_under_probs(
                team_simulations['PTS'], team_lines
            )
        
        # Estadísticas de equipo
        for stat in ['TRB', 'AST']:
            if stat in team_simulations.columns:
                stat_lines = lines.get(stat, [])
                if stat == 'TRB':
                    stat_lines = [40.5, 45.5, 50.5]
                elif stat == 'AST':
                    stat_lines = [20.5, 25.5, 30.5]
                
                probabilities[f'{stat}_over_under'] = self._calculate_over_under_probs(
                    team_simulations[stat], stat_lines
                )
        
        return probabilities
    
    def calculate_game_probabilities(self, 
                                   game_results: pd.DataFrame,
                                   home_team: str,
                                   away_team: str) -> Dict[str, any]:
        """
        Calcula probabilidades del resultado del partido.
        
        Args:
            game_results: DataFrame con resultados simulados
            home_team: Equipo local
            away_team: Equipo visitante
            
        Returns:
            Diccionario con probabilidades del partido
        """
        probabilities = {
            'home_team': home_team,
            'away_team': away_team
        }
        
        # Probabilidades de victoria
        if 'winner' in game_results.columns:
            # Calcular wins a partir de la columna winner
            home_wins = (game_results['winner'] == 'home').astype(int)
            away_wins = (game_results['winner'] == 'away').astype(int)
        elif 'home_wins' in game_results.columns:
            home_wins = game_results['home_wins']
            away_wins = game_results['away_wins']
        else:
            # Fallback si no tenemos datos de ganador
            home_wins = pd.Series([0.5] * len(game_results))
            away_wins = pd.Series([0.5] * len(game_results))
            
        probabilities['win_probabilities'] = {
            'home_win': float(home_wins.mean()),
            'away_win': float(away_wins.mean())
        }
        
        # Over/Under puntos totales
        total_points_lines = self.common_lines.get('total_points', [220.5])
        probabilities['total_points_over_under'] = self._calculate_over_under_probs(
            game_results['total_points'], total_points_lines
        )
        
        # Spreads (diferencial de puntos)
        spreads = [-10.5, -7.5, -5.5, -3.5, -1.5, 1.5, 3.5, 5.5, 7.5, 10.5]
        probabilities['spread_probabilities'] = self._calculate_spread_probs(
            game_results['point_differential'], spreads
        )
        
        # Tipos de juego
        probabilities['game_types'] = self._calculate_game_type_probs(game_results)
        
        return probabilities
    
    def _calculate_over_under_probs(self, 
                                  values: pd.Series, 
                                  lines: List[float]) -> Dict[str, float]:
        """Calcula probabilidades over/under para líneas específicas."""
        probs = {}
        
        for line in lines:
            over_prob = float((values > line).mean())
            under_prob = 1.0 - over_prob
            
            probs[f'over_{line}'] = over_prob
            probs[f'under_{line}'] = under_prob
        
        return probs
    
    def _calculate_spread_probs(self, 
                              point_diff: pd.Series, 
                              spreads: List[float]) -> Dict[str, float]:
        """Calcula probabilidades de spread."""
        probs = {}
        
        for spread in spreads:
            if spread < 0:
                # Favorito por X puntos
                prob = float((point_diff > abs(spread)).mean())
                probs[f'home_favored_by_{abs(spread)}'] = prob
            else:
                # Underdog por X puntos
                prob = float((point_diff < -spread).mean())
                probs[f'away_favored_by_{spread}'] = prob
        
        return probs
    
    def _calculate_special_events(self, simulations: pd.DataFrame) -> Dict[str, float]:
        """Calcula probabilidades de eventos especiales."""
        special = {}
        
        # Double-double
        if 'double_double' in simulations.columns:
            special['double_double'] = float(simulations['double_double'].mean())
        
        # Triple-double
        if 'triple_double' in simulations.columns:
            special['triple_double'] = float(simulations['triple_double'].mean())
        
        # Juegos perfectos (sin pérdidas)
        if 'TOV' in simulations.columns:
            special['zero_turnovers'] = float((simulations['TOV'] == 0).mean())
        
        # Eficiencia alta
        if all(col in simulations.columns for col in ['PTS', 'FGA']):
            # Más de 1.5 puntos por intento
            efficient_games = (simulations['PTS'] / (simulations.get('FGA', 10) + 1)) > 1.5
            special['high_efficiency'] = float(efficient_games.mean())
        
        return special
    
    def _calculate_performance_ranges(self, simulations: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calcula probabilidades por rangos de rendimiento."""
        ranges = {}
        
        # Rangos de puntos
        if 'PTS' in simulations.columns:
            pts = simulations['PTS']
            ranges['PTS'] = {
                '0-10': float((pts <= 10).mean()),
                '11-20': float(((pts > 10) & (pts <= 20)).mean()),
                '21-30': float(((pts > 20) & (pts <= 30)).mean()),
                '31-40': float(((pts > 30) & (pts <= 40)).mean()),
                '40+': float((pts > 40).mean())
            }
        
        # Rangos de rebotes
        if 'TRB' in simulations.columns:
            reb = simulations['TRB']
            ranges['TRB'] = {
                '0-5': float((reb <= 5).mean()),
                '6-10': float(((reb > 5) & (reb <= 10)).mean()),
                '11-15': float(((reb > 10) & (reb <= 15)).mean()),
                '15+': float((reb > 15).mean())
            }
        
        # Rangos de asistencias
        if 'AST' in simulations.columns:
            ast = simulations['AST']
            ranges['AST'] = {
                '0-3': float((ast <= 3).mean()),
                '4-7': float(((ast > 3) & (ast <= 7)).mean()),
                '8-12': float(((ast > 7) & (ast <= 12)).mean()),
                '12+': float((ast > 12).mean())
            }
        
        return ranges
    
    def _calculate_combinations(self, simulations: pd.DataFrame) -> Dict[str, float]:
        """Calcula probabilidades de combinaciones populares."""
        combinations = {}
        
        # Combinaciones de puntos y rebotes
        if all(col in simulations.columns for col in ['PTS', 'TRB']):
            combinations['20pts_10reb'] = float(
                ((simulations['PTS'] >= 20) & (simulations['TRB'] >= 10)).mean()
            )
            combinations['25pts_8reb'] = float(
                ((simulations['PTS'] >= 25) & (simulations['TRB'] >= 8)).mean()
            )
        
        # Combinaciones de puntos y asistencias
        if all(col in simulations.columns for col in ['PTS', 'AST']):
            combinations['20pts_5ast'] = float(
                ((simulations['PTS'] >= 20) & (simulations['AST'] >= 5)).mean()
            )
            combinations['15pts_8ast'] = float(
                ((simulations['PTS'] >= 15) & (simulations['AST'] >= 8)).mean()
            )
        
        # Triple combinación
        if all(col in simulations.columns for col in ['PTS', 'TRB', 'AST']):
            combinations['15pts_8reb_5ast'] = float(
                ((simulations['PTS'] >= 15) & 
                 (simulations['TRB'] >= 8) & 
                 (simulations['AST'] >= 5)).mean()
            )
        
        return combinations
    
    def _calculate_game_type_probs(self, game_results: pd.DataFrame) -> Dict[str, float]:
        """Calcula probabilidades por tipo de juego."""
        game_types = {}
        
        if 'total_points' in game_results.columns:
            total_pts = game_results['total_points']
            
            game_types['low_scoring'] = float((total_pts < 210).mean())
            game_types['average_scoring'] = float(((total_pts >= 210) & (total_pts < 230)).mean())
            game_types['high_scoring'] = float((total_pts >= 230).mean())
        
        if 'point_differential' in game_results.columns:
            diff = game_results['point_differential'].abs()
            
            game_types['blowout'] = float((diff > 15).mean())
            game_types['close_game'] = float((diff <= 5).mean())
            game_types['competitive'] = float(((diff > 5) & (diff <= 15)).mean())
        
        return game_types
    
    def generate_betting_report(self, 
                              player_probs: Dict[str, Dict[str, any]],
                              team_probs: Dict[str, Dict[str, any]],
                              game_probs: Dict[str, any]) -> Dict[str, any]:
        """
        Genera reporte completo para apuestas deportivas.
        
        Args:
            player_probs: Probabilidades por jugador
            team_probs: Probabilidades por equipo
            game_probs: Probabilidades del partido
            
        Returns:
            Reporte completo de apuestas
        """
        report = {
            'game_overview': game_probs,
            'team_analysis': team_probs,
            'player_analysis': player_probs,
            'recommendations': self._generate_recommendations(
                player_probs, team_probs, game_probs
            ),
            'summary': self._generate_summary(player_probs, team_probs, game_probs)
        }
        
        return report
    
    def _generate_recommendations(self, 
                                player_probs: Dict[str, Dict[str, any]],
                                team_probs: Dict[str, Dict[str, any]],
                                game_probs: Dict[str, any]) -> List[Dict[str, any]]:
        """Genera recomendaciones de apuestas basadas en probabilidades."""
        recommendations = []
        
        # Recomendaciones de victoria
        home_win_prob = game_probs.get('win_probabilities', {}).get('home_win', 0.5)
        if home_win_prob > 0.65:
            recommendations.append({
                'type': 'game_result',
                'bet': f"Victoria {game_probs.get('home_team', 'Local')}",
                'probability': home_win_prob,
                'confidence': 'Alta' if home_win_prob > 0.75 else 'Media'
            })
        elif home_win_prob < 0.35:
            recommendations.append({
                'type': 'game_result',
                'bet': f"Victoria {game_probs.get('away_team', 'Visitante')}",
                'probability': 1 - home_win_prob,
                'confidence': 'Alta' if home_win_prob < 0.25 else 'Media'
            })
        
        # Recomendaciones de jugadores
        for player_name, probs in player_probs.items():
            if isinstance(probs, dict) and 'special_events' in probs:
                dd_prob = probs['special_events'].get('double_double', 0)
                if dd_prob > 0.6:
                    recommendations.append({
                        'type': 'player_special',
                        'player': player_name,
                        'bet': 'Double-Double',
                        'probability': dd_prob,
                        'confidence': 'Alta' if dd_prob > 0.75 else 'Media'
                    })
        
        return recommendations
    
    def _generate_summary(self, 
                        player_probs: Dict[str, Dict[str, any]],
                        team_probs: Dict[str, Dict[str, any]],
                        game_probs: Dict[str, any]) -> Dict[str, any]:
        """Genera resumen ejecutivo del análisis."""
        summary = {
            'total_players_analyzed': len(player_probs),
            'total_teams_analyzed': len(team_probs),
            'game_competitiveness': 'Equilibrado',
            'highest_probability_events': [],
            'key_insights': []
        }
        
        # Determinar competitividad del juego
        home_win_prob = game_probs.get('win_probabilities', {}).get('home_win', 0.5)
        if abs(home_win_prob - 0.5) > 0.2:
            summary['game_competitiveness'] = 'Desequilibrado'
        
        # Eventos de alta probabilidad
        all_events = []
        
        # Agregar eventos de jugadores
        for player_name, probs in player_probs.items():
            if isinstance(probs, dict) and 'special_events' in probs:
                for event, prob in probs['special_events'].items():
                    if prob > 0.7:
                        all_events.append({
                            'event': f"{player_name} - {event}",
                            'probability': prob
                        })
        
        # Ordenar por probabilidad
        all_events.sort(key=lambda x: x['probability'], reverse=True)
        summary['highest_probability_events'] = all_events[:5]
        
        return summary 