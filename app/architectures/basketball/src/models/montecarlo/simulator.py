"""
Simulador de Partidos NBA Monte Carlo
====================================

Simulador completo que orquesta simulaciones de jugadores individuales
y equipos para generar escenarios coherentes de partidos NBA.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json

from .engine import MonteCarloEngine
from .correlations import CorrelationMatrix
from .config import MonteCarloConfig
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader

logger = logging.getLogger(__name__)


class NBAGameSimulator:
    """
    Simulador de partidos NBA MEJORADO con precisión elite
    Incorpora análisis contextual profundo y correlaciones dinámicas
    """
    
    def __init__(self, monte_carlo_engine, num_simulations=MonteCarloConfig.DEFAULT_N_SIMULATIONS):
        self.monte_carlo_engine = monte_carlo_engine
        self.num_simulations = num_simulations
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Simulador de partidos NBA inicializado con {num_simulations} simulaciones")

    def simulate_game(self, home_team: str, away_team: str, date: str = None) -> dict:
        """
        Simula un partido NBA con análisis contextual completo
        
        Args:
            home_team: Equipo local
            away_team: Equipo visitante
            date: Fecha del partido (para contexto temporal)
        """
        self.logger.info(f"Simulando partido: {away_team} @ {home_team}")
        
        # OBTENER CONTEXTO DEL PARTIDO
        game_context = self._analyze_game_context(home_team, away_team, date)
        
        # OBTENER ROSTERS CON ANÁLISIS AVANZADO
        home_roster = self._get_enhanced_roster(home_team, is_home=True, opponent=away_team, context=game_context)
        away_roster = self._get_enhanced_roster(away_team, is_home=False, opponent=home_team, context=game_context)
        
        # VALIDAR ROSTERS
        if not home_roster or not away_roster:
            self.logger.warning(f"Rosters incompletos para {home_team} vs {away_team}")
            return {'error': 'Rosters incompletos'}
        
        # EJECUTAR SIMULACIONES MASIVAS
        simulation_results = []
        successful_sims = 0
        failed_sims = 0
        
        for sim_idx in range(self.num_simulations):
            try:
                # CONTEXTO ESPECÍFICO DE SIMULACIÓN
                sim_context = self._generate_simulation_context(game_context, sim_idx)
                
                # SIMULAR RENDIMIENTOS DE JUGADORES
                home_performances = self._simulate_team_performance(
                    home_roster, sim_context, is_home=True
                )
                away_performances = self._simulate_team_performance(
                    away_roster, sim_context, is_home=False
                )
                
                # CALCULAR TOTALES DE EQUIPO
                home_totals = self._calculate_team_totals(home_performances)
                away_totals = self._calculate_team_totals(away_performances)
                
                # APLICAR FACTORES DE EQUIPO
                home_totals = self._apply_team_factors(home_totals, home_team, True, sim_context)
                away_totals = self._apply_team_factors(away_totals, away_team, False, sim_context)
                
                # DETERMINAR GANADOR CON ANÁLISIS AVANZADO
                winner = self._determine_winner_advanced(home_totals, away_totals, sim_context)
                
                # ALMACENAR RESULTADO
                simulation_results.append({
                    'home_score': home_totals['PTS'],
                    'away_score': away_totals['PTS'],
                    'winner': winner,
                    'home_stats': home_totals,
                    'away_stats': away_totals,
                    'home_players': home_performances,
                    'away_players': away_performances,
                    'context_factors': sim_context.get('applied_factors', {})
                })
                
                successful_sims += 1
                
            except Exception as e:
                failed_sims += 1
                if failed_sims < 10:  # Solo loguear las primeras fallas
                    self.logger.warning(f"Simulación {sim_idx} falló: {str(e)}")
                continue
        
        if successful_sims == 0:
            return {'error': 'Todas las simulaciones fallaron'}
        
        self.logger.info(f"Simulación completada: {successful_sims} escenarios generados")
        
        # PROCESAR RESULTADOS
        processed_results = self._process_simulation_results(
            simulation_results, home_team, away_team, game_context
        )
        
        return processed_results

    def _analyze_game_context(self, home_team: str, away_team: str, date: str = None) -> dict:
        """Analiza el contexto completo del partido"""
        context = {
            'home_team': home_team,
            'away_team': away_team,
            'date': date,
            'is_playoffs': False,  # Se podría detectar por fecha
            'is_nationally_televised': False,  # Se podría determinar por equipos
            'rivalry_game': False,  # Se podría detectar por historial
            'rest_advantage': None,
            'injury_impacts': {},
            'momentum_factors': {},
            'historical_matchup': None
        }
        
        # ANÁLISIS DE EQUIPOS
        if hasattr(self.monte_carlo_engine, 'team_profiles'):
            if home_team in self.monte_carlo_engine.team_profiles:
                context['home_team_profile'] = self.monte_carlo_engine.team_profiles[home_team]
            if away_team in self.monte_carlo_engine.team_profiles:
                context['away_team_profile'] = self.monte_carlo_engine.team_profiles[away_team]
        
        # DETECTAR PARTIDOS ESPECIALES
        high_profile_teams = ['LAL', 'GSW', 'BOS', 'MIA', 'NYK', 'CHI']
        if home_team in high_profile_teams or away_team in high_profile_teams:
            context['is_nationally_televised'] = True
            context['pressure_multiplier'] = 1.05
        
        # RIVALIDADES HISTÓRICAS
        rivalries = [
            ('LAL', 'BOS'), ('LAL', 'GSW'), ('MIA', 'BOS'),
            ('NYK', 'BOS'), ('CHI', 'DET'), ('LAC', 'GSW')
        ]
        for team1, team2 in rivalries:
            if (home_team == team1 and away_team == team2) or (home_team == team2 and away_team == team1):
                context['rivalry_game'] = True
                context['intensity_multiplier'] = 1.08
                break
        
        return context

    def _get_enhanced_roster(self, team: str, is_home: bool, opponent: str, context: dict) -> list:
        """Obtiene roster mejorado con análisis avanzado de jugadores"""
        team_players = self.monte_carlo_engine.players_df[
            self.monte_carlo_engine.players_df['Team'] == team
        ]['Player'].unique()
        
        if len(team_players) == 0:
            self.logger.warning(f"No se encontraron jugadores para {team}")
            return []
        
        enhanced_roster = []
        
        for player in team_players:
            if player in self.monte_carlo_engine.player_profiles:
                profile = self.monte_carlo_engine.player_profiles[player]
                
                # CALCULAR PROBABILIDAD DE JUGAR
                play_probability = self._calculate_play_probability(player, profile, context)
                
                # CALCULAR MINUTOS ESPERADOS
                expected_minutes = self._calculate_expected_minutes(player, profile, context, is_home)
                
                # DETERMINAR SI ES TITULAR
                is_starter = profile['situational_stats']['starter_profile']['is_regular_starter']
                
                # ANÁLISIS DE MATCHUP
                matchup_analysis = self._analyze_player_matchup(player, opponent, profile)
                
                enhanced_roster.append({
                    'name': player,
                    'profile': profile,
                    'play_probability': play_probability,
                    'expected_minutes': expected_minutes,
                    'is_starter': is_starter,
                    'matchup_analysis': matchup_analysis,
                    'position': profile['position_info']['primary_position'],
                    'archetype': profile['position_info']['archetype'],
                    'role': profile['position_info']['role']
                })
            else:
                # Jugador sin perfil - usar datos básicos
                enhanced_roster.append({
                    'name': player,
                    'profile': None,
                    'play_probability': 0.85,  # Probabilidad estándar
                    'expected_minutes': 20.0,  # Minutos promedio
                    'is_starter': False,
                    'matchup_analysis': {},
                    'position': 'G',
                    'archetype': 'role_player',
                    'role': 'bench_player'
                })
        
        # ORDENAR POR IMPORTANCIA (titulares primero, luego por minutos esperados)
        enhanced_roster.sort(key=lambda x: (-int(x['is_starter']), -x['expected_minutes']))
        
        return enhanced_roster[:15]  # Máximo 15 jugadores activos

    def _calculate_play_probability(self, player: str, profile: dict, context: dict) -> float:
        """Calcula probabilidad de que el jugador juegue"""
        base_probability = 0.85  # Base 85%
        
        # FACTORES DE LESIÓN
        if hasattr(self.monte_carlo_engine, 'injury_patterns'):
            if player in self.monte_carlo_engine.injury_patterns:
                injury_risk = self.monte_carlo_engine.injury_patterns[player]['injury_risk']
                base_probability *= (1.0 - injury_risk * 0.3)  # Reducir hasta 30%
        
        # CARGA DE TRABAJO
        if profile:
            games_played = profile.get('games_played', 50)
            season_workload = games_played / 82.0  # Normalizado a temporada completa
            if season_workload < 0.6:  # Menos del 60% de juegos
                base_probability *= 0.9  # Reducir probabilidad
        
        # CONTEXTO DEL PARTIDO
        if context.get('is_playoffs'):
            base_probability = min(0.98, base_probability * 1.1)  # Aumentar en playoffs
        
        return max(0.1, min(1.0, base_probability))

    def _calculate_expected_minutes(self, player: str, profile: dict, context: dict, is_home: bool) -> float:
        """Calcula minutos esperados del jugador"""
        if not profile:
            return 20.0  # Default para jugadores sin perfil
        
        base_minutes = profile['advanced_stats']['MP']['mean']
        
        # AJUSTE POR TITULARIDAD
        if profile['situational_stats']['starter_profile']['is_regular_starter']:
            base_minutes = max(25, base_minutes)  # Mínimo 25 min para titulares
        else:
            base_minutes = min(30, base_minutes)  # Máximo 30 min para suplentes
        
        # AJUSTE POR VENTAJA LOCAL
        if is_home and profile['situational_stats']['home_advantage']['pts_boost'] > 0:
            base_minutes *= 1.03  # 3% más en casa
        
        # AJUSTE POR CONTEXTO DEL PARTIDO
        if context.get('rivalry_game'):
            base_minutes *= 1.05  # 5% más en rivalidades
        
        if context.get('is_playoffs'):
            base_minutes *= 1.08  # 8% más en playoffs
        
        # AJUSTE POR CARGA DE TRABAJO RECIENTE
        if hasattr(self.monte_carlo_engine, 'injury_patterns'):
            if player in self.monte_carlo_engine.injury_patterns:
                workload = self.monte_carlo_engine.injury_patterns[player]['workload_heavy']
                if workload > 0.7:  # Carga alta
                    base_minutes *= 0.95  # Reducir 5%
        
        return max(0, min(48, base_minutes))

    def _analyze_player_matchup(self, player: str, opponent: str, profile: dict) -> dict:
        """Analiza el matchup específico del jugador vs oponente"""
        matchup_analysis = {
            'advantage': 0.0,  # -1 a 1, donde 1 es máxima ventaja
            'historical_performance': {},
            'style_matchup': 'neutral',
            'expected_impact': 1.0
        }
        
        if not profile:
            return matchup_analysis
        
        # ANÁLISIS HISTÓRICO VS OPONENTE
        if hasattr(self.monte_carlo_engine, 'players_df'):
            player_vs_opp = self.monte_carlo_engine.players_df[
                (self.monte_carlo_engine.players_df['Player'] == player) &
                (self.monte_carlo_engine.players_df['Opp'] == opponent)
            ]
            
            if len(player_vs_opp) > 0:
                vs_opp_avg = {
                    'pts': player_vs_opp['PTS'].mean(),
                    'trb': player_vs_opp['TRB'].mean(),
                    'ast': player_vs_opp['AST'].mean(),
                    'fg_pct': player_vs_opp['FG%'].mean()
                }
                
                overall_avg = {
                    'pts': profile['basic_stats']['PTS']['mean'],
                    'trb': profile['basic_stats']['TRB']['mean'],
                    'ast': profile['basic_stats']['AST']['mean'],
                    'fg_pct': profile['advanced_stats']['FG%']['mean']
                }
                
                # Calcular ventaja basada en rendimiento histórico
                pts_advantage = (vs_opp_avg['pts'] - overall_avg['pts']) / (overall_avg['pts'] + 1)
                fg_advantage = vs_opp_avg['fg_pct'] - overall_avg['fg_pct']
                
                matchup_analysis['advantage'] = (pts_advantage + fg_advantage) / 2
                matchup_analysis['historical_performance'] = vs_opp_avg
                matchup_analysis['games_played'] = len(player_vs_opp)
        
        # ANÁLISIS DE ESTILO DE JUEGO
        player_archetype = profile['position_info']['archetype']
        
        # Obtener perfil del equipo oponente
        if hasattr(self.monte_carlo_engine, 'team_profiles'):
            if opponent in self.monte_carlo_engine.team_profiles:
                opp_profile = self.monte_carlo_engine.team_profiles[opponent]
                
                # Ventajas por arquetipo vs defensa del oponente
                def_efficiency = opp_profile['defensive_stats']['def_efficiency']
                league_avg = 1.1  # Aproximación
                
                if player_archetype == 'shooter' and def_efficiency > league_avg:
                    matchup_analysis['style_matchup'] = 'favorable'
                    matchup_analysis['expected_impact'] = 1.1
                elif player_archetype == 'inside_scorer' and def_efficiency < league_avg:
                    matchup_analysis['style_matchup'] = 'unfavorable'
                    matchup_analysis['expected_impact'] = 0.9
        
        return matchup_analysis

    def _generate_simulation_context(self, base_context: dict, sim_index: int) -> dict:
        """Genera contexto específico para cada simulación"""
        sim_context = base_context.copy()
        
        # VARIACIONES ALEATORIAS
        sim_context.update({
            'game_flow': np.random.choice(['slow', 'normal', 'fast'], p=[0.2, 0.6, 0.2]),
            'officiating': np.random.normal(1.0, 0.05),  # Factor de arbitraje
            'clutch_situation': np.random.random() < 0.3,  # 30% partidos cerrados
            'injury_during_game': np.random.random() < 0.05,  # 5% lesión en juego
            'technical_fouls': np.random.poisson(0.3),  # Promedio 0.3 técnicas por juego
            'momentum_swings': np.random.randint(2, 6),  # 2-5 cambios de momentum
            'simulation_index': sim_index
        })
        
        # FACTORES DINÁMICOS
        if sim_context['game_flow'] == 'fast':
            sim_context['pace_multiplier'] = 1.12
        elif sim_context['game_flow'] == 'slow':
            sim_context['pace_multiplier'] = 0.88
        else:
            sim_context['pace_multiplier'] = 1.0
        
        # FACTOR DE PRESIÓN
        pressure_factor = 1.0
        if sim_context.get('is_nationally_televised'):
            pressure_factor *= 1.02
        if sim_context.get('rivalry_game'):
            pressure_factor *= 1.03
        if sim_context.get('clutch_situation'):
            pressure_factor *= 1.05
        
        sim_context['pressure_factor'] = pressure_factor
        
        return sim_context

    def _simulate_team_performance(self, roster: list, context: dict, is_home: bool) -> dict:
        """Simula el rendimiento de todo el equipo"""
        team_performance = {}
        
        for player_info in roster:
            player_name = player_info['name']
            
            # DECIDIR SI JUEGA
            if np.random.random() > player_info['play_probability']:
                continue  # Jugador no juega
            
            # GENERAR CONTEXTO DEL JUGADOR
            player_context = {
                'is_home': is_home,
                'is_starter': player_info['is_starter'],
                'opponent': context['away_team'] if is_home else context['home_team'],
                'expected_minutes': player_info['expected_minutes'],
                'matchup_analysis': player_info['matchup_analysis'],
                'game_context': context,
                'days_rest': 1,  # Default - se podría calcular de fechas
                'is_playoffs': context.get('is_playoffs', False),
                'is_nationally_televised': context.get('is_nationally_televised', False)
            }
            
            # GENERAR RENDIMIENTO
            performance = self.monte_carlo_engine.generate_player_performance(
                player_name, player_context
            )
            
            # APLICAR AJUSTES CONTEXTUALES ADICIONALES
            performance = self._apply_game_context_adjustments(
                performance, player_info, context
            )
            
            team_performance[player_name] = performance
        
        return team_performance

    def _apply_game_context_adjustments(self, performance: dict, player_info: dict, context: dict) -> dict:
        """Aplica ajustes finales basados en el contexto del juego"""
        adjusted_performance = performance.copy()
        
        # AJUSTE POR RITMO DEL JUEGO
        pace_mult = context.get('pace_multiplier', 1.0)
        for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK']:
            if stat in adjusted_performance:
                adjusted_performance[stat] *= pace_mult
        
        # AJUSTE POR PRESIÓN
        pressure_factor = context.get('pressure_factor', 1.0)
        player_clutch = 1.0
        if player_info.get('profile'):
            player_clutch = player_info['profile']['clutch_analysis']['clutch_factor']
        
        # Jugadores clutch mejoran bajo presión, otros empeoran
        if pressure_factor > 1.0:
            clutch_adjustment = 0.95 + (player_clutch - 1.0) * 0.1
            for stat in ['PTS', 'FG%', '3P%']:
                if stat in adjusted_performance:
                    adjusted_performance[stat] *= clutch_adjustment
        
        # AJUSTE POR VENTAJA DEL MATCHUP
        matchup_advantage = player_info['matchup_analysis'].get('advantage', 0.0)
        if abs(matchup_advantage) > 0.1:  # Solo aplicar si hay ventaja significativa
            matchup_mult = 1.0 + matchup_advantage * 0.15  # Hasta ±15%
            for stat in ['PTS', 'TRB', 'AST']:
                if stat in adjusted_performance:
                    adjusted_performance[stat] *= max(0.8, min(1.25, matchup_mult))
        
        return adjusted_performance

    def _calculate_team_totals(self, team_performances: dict) -> dict:
        """Calcula totales del equipo con lógica mejorada"""
        totals = {
            'PTS': 0, 'TRB': 0, 'AST': 0, 'STL': 0, 'BLK': 0, 'TOV': 0,
            'FG%': 0, '3P%': 0, 'FT%': 0, '3P': 0,
            'double_doubles': 0, 'triple_doubles': 0,
            'active_players': len(team_performances),
            'starter_contribution': 0,
            'bench_contribution': 0
        }
        
        total_minutes = 0
        shooting_stats = {'makes': 0, 'attempts': 0, '3makes': 0, '3attempts': 0}
        
        for player, performance in team_performances.items():
            # SUMAR ESTADÍSTICAS BÁSICAS
            for stat in ['PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV', '3P']:
                if stat in performance:
                    totals[stat] += performance[stat]
            
            # CONTAR LOGROS ESPECIALES
            totals['double_doubles'] += performance.get('double_double', 0)
            totals['triple_doubles'] += performance.get('triple_double', 0)
            
            # ACUMULAR PARA PORCENTAJES
            total_minutes += performance.get('MP', 0)
            
            # Estimar intentos de tiro para porcentajes
            if 'FG%' in performance and performance['FG%'] > 0:
                estimated_attempts = performance.get('PTS', 0) / (performance['FG%'] * 2.2)
                estimated_makes = estimated_attempts * performance['FG%']
                shooting_stats['makes'] += estimated_makes
                shooting_stats['attempts'] += estimated_attempts
            
            if '3P%' in performance and performance['3P%'] > 0 and '3P' in performance:
                three_attempts = performance['3P'] / (performance['3P%'] + 0.001)
                shooting_stats['3makes'] += performance['3P']
                shooting_stats['3attempts'] += three_attempts
        
        # CALCULAR PORCENTAJES DE EQUIPO
        if shooting_stats['attempts'] > 0:
            totals['FG%'] = shooting_stats['makes'] / shooting_stats['attempts']
        else:
            totals['FG%'] = 0.45  # Promedio NBA
            
        if shooting_stats['3attempts'] > 0:
            totals['3P%'] = shooting_stats['3makes'] / shooting_stats['3attempts']
        else:
            totals['3P%'] = 0.35  # Promedio NBA
        
        totals['FT%'] = 0.77  # Aproximación promedio NBA
        
        # MÉTRICAS ADICIONALES
        totals['avg_minutes_per_player'] = total_minutes / max(1, len(team_performances))
        totals['pace'] = totals['PTS'] + totals['AST'] + totals['TOV']  # Aproximación
        
        return totals

    def _apply_team_factors(self, team_totals: dict, team: str, is_home: bool, context: dict) -> dict:
        """Aplica factores de equipo al rendimiento total"""
        adjusted_totals = team_totals.copy()
        
        # VENTAJA LOCAL
        if is_home:
            home_boost = 1.06  # 6% boost promedio NBA en casa
            adjusted_totals['PTS'] *= home_boost
            adjusted_totals['FG%'] *= 1.02  # 2% mejor tiro en casa
            adjusted_totals['TRB'] *= 1.03  # 3% más rebotes en casa
        
        # PERFIL DEL EQUIPO
        if hasattr(self.monte_carlo_engine, 'team_profiles'):
            if team in self.monte_carlo_engine.team_profiles:
                team_profile = self.monte_carlo_engine.team_profiles[team]
                
                # Ajustar por estilo de juego del equipo (limitado y conservador)
                pace_factor = team_profile['offensive_stats']['pace'] / 95.0  # Normalizado
                efficiency_factor = min(1.5, max(0.7, team_profile['offensive_stats']['efficiency']))  # Límites 0.7-1.5
                
                # Ajustes muy conservadores para evitar puntuaciones irrealistas
                efficiency_mult = 0.95 + 0.1 * (efficiency_factor - 1.0)  # Rango 0.90-1.05
                pace_mult = 0.95 + 0.1 * (pace_factor - 1.0)  # Rango 0.90-1.05
                
                adjusted_totals['PTS'] *= max(0.90, min(1.10, efficiency_mult))
                adjusted_totals['AST'] *= max(0.90, min(1.10, pace_mult))
        
        # FACTORES CONTEXTUALES
        if context.get('rivalry_game'):
            # Rivalidades tienden a ser más defensivas
            adjusted_totals['PTS'] *= 0.96
            adjusted_totals['FG%'] *= 0.97
        
        if context.get('is_nationally_televised'):
            # Juegos televisados tienden a ser mejores
            adjusted_totals['PTS'] *= 1.02
            adjusted_totals['FG%'] *= 1.01
        
        # FACTOR DE ARBITRAJE
        officiating_factor = context.get('officiating', 1.0)
        adjusted_totals['PTS'] *= officiating_factor
        
        # APLICAR LÍMITES REALISTAS FINALES NBA (más conservadores)
        adjusted_totals['PTS'] = max(85, min(125, adjusted_totals['PTS']))  # Límites más estrictos NBA
        adjusted_totals['TRB'] = max(35, min(55, adjusted_totals.get('TRB', 45)))  # Límites de rebotes
        adjusted_totals['AST'] = max(15, min(35, adjusted_totals.get('AST', 25)))  # Límites de asistencias
        adjusted_totals['STL'] = max(4, min(12, adjusted_totals.get('STL', 7)))   # Límites de robos
        adjusted_totals['BLK'] = max(2, min(8, adjusted_totals.get('BLK', 5)))    # Límites de bloqueos
        adjusted_totals['TOV'] = max(8, min(25, adjusted_totals.get('TOV', 14)))  # Límites de pérdidas
        
        return adjusted_totals

    def _determine_winner_advanced(self, home_totals: dict, away_totals: dict, context: dict) -> str:
        """Determina ganador con análisis avanzado"""
        home_score = home_totals['PTS']
        away_score = away_totals['PTS']
        
        # FACTOR DE CLUTCH EN JUEGOS CERRADOS
        if abs(home_score - away_score) < 5:  # Juego cerrado
            # Aplicar factores clutch adicionales
            clutch_home = 1.0 + np.random.normal(0, 0.02)  # Variación ±2%
            clutch_away = 1.0 + np.random.normal(0, 0.02)
            
            home_score *= clutch_home
            away_score *= clutch_away
        
        # OVERTIME SIMULATION
        if abs(home_score - away_score) < 1:  # Muy cerrado
            if np.random.random() < 0.15:  # 15% probabilidad de overtime
                # Simular overtime (5 minutos adicionales)
                ot_home = np.random.normal(6, 2)  # ~6 puntos promedio en OT
                ot_away = np.random.normal(6, 2)
                home_score += max(0, ot_home)
                away_score += max(0, ot_away)
        
        return 'home' if home_score > away_score else 'away'

    def _process_simulation_results(self, results: list, home_team: str, away_team: str, context: dict) -> dict:
        """Procesa y analiza todos los resultados de simulación"""
        if not results:
            return {'error': 'No hay resultados para procesar'}
        
        # ESTADÍSTICAS BÁSICAS
        home_wins = sum(1 for r in results if r['winner'] == 'home')
        away_wins = len(results) - home_wins
        
        home_scores = [r['home_score'] for r in results]
        away_scores = [r['away_score'] for r in results]
        
        # COMPILAR SIMULACIONES DE JUGADORES
        player_simulations = {'home': {}, 'away': {}}
        team_simulations = {'home': [], 'away': []}
        game_results = []
        
        # Procesar cada resultado de simulación
        for result in results:
            # Agregar resultado del juego
            game_results.append({
                'home_score': result['home_score'],
                'away_score': result['away_score'],
                'winner': result['winner'],
                'total_points': result['home_score'] + result['away_score'],
                'point_differential': result['home_score'] - result['away_score']
            })
            
            # Agregar estadísticas de equipo
            team_simulations['home'].append(result['home_stats'])
            team_simulations['away'].append(result['away_stats'])
            
            # Compilar estadísticas de jugadores
            for player_name, player_stats in result['home_players'].items():
                if player_name not in player_simulations['home']:
                    player_simulations['home'][player_name] = []
                player_simulations['home'][player_name].append(player_stats)
                
            for player_name, player_stats in result['away_players'].items():
                if player_name not in player_simulations['away']:
                    player_simulations['away'][player_name] = []
                player_simulations['away'][player_name].append(player_stats)
        
        # ANÁLISIS DETALLADO
        analysis = {
            'game_info': {
                'home_team': home_team,
                'away_team': away_team,
                'simulations_run': len(results),
                'context': context
            },
            'win_probabilities': {
                'home_win': home_wins / len(results),
                'away_win': away_wins / len(results),
                'home_wins': home_wins,
                'away_wins': away_wins
            },
            'score_predictions': {
                'home_score': {
                    'mean': np.mean(home_scores),
                    'median': np.median(home_scores),
                    'std': np.std(home_scores),
                    'min': np.min(home_scores),
                    'max': np.max(home_scores),
                    'percentiles': {
                        '25th': np.percentile(home_scores, 25),
                        '75th': np.percentile(home_scores, 75)
                    }
                },
                'away_score': {
                    'mean': np.mean(away_scores),
                    'median': np.median(away_scores),
                    'std': np.std(away_scores),
                    'min': np.min(away_scores),
                    'max': np.max(away_scores),
                    'percentiles': {
                        '25th': np.percentile(away_scores, 25),
                        '75th': np.percentile(away_scores, 75)
                    }
                }
            },
            'game_characteristics': {
                'total_points_avg': np.mean([r['home_score'] + r['away_score'] for r in results]),
                'margin_avg': np.mean([abs(r['home_score'] - r['away_score']) for r in results]),
                'close_games_pct': sum(1 for r in results if abs(r['home_score'] - r['away_score']) <= 5) / len(results),
                'blowouts_pct': sum(1 for r in results if abs(r['home_score'] - r['away_score']) >= 20) / len(results)
            },
            'confidence_metrics': {
                'prediction_confidence': max(home_wins, away_wins) / len(results),
                'outcome_certainty': abs(home_wins - away_wins) / len(results),
                'model_reliability': len(results) / self.num_simulations  # % simulaciones exitosas
            },
            # AGREGAR DATOS REQUERIDOS POR EL PIPELINE
            'player_simulations': player_simulations,
            'team_simulations': team_simulations,
            'game_results': game_results
        }
        
        # ANÁLISIS DE TENDENCIAS
        analysis['trends'] = {
            'home_advantage_realized': (home_wins / len(results)) > 0.55,  # Ventaja local típica
            'high_scoring_game': analysis['game_characteristics']['total_points_avg'] > 220,
            'defensive_battle': analysis['game_characteristics']['total_points_avg'] < 200,
            'competitive_game': analysis['game_characteristics']['close_games_pct'] > 0.3
        }
        
        return analysis 