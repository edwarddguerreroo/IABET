"""
Bookmakers Data Fetcher - Integración Completa con Sportradar
============================================================

Fetcher avanzado que obtiene, procesa y gestiona datos de cuotas desde Sportradar API.

Funciones principales:
1. Integración completa con Sportradar API para cuotas NBA
2. Obtener player props (PTS, AST, TRB, 3P) en tiempo real
3. Cache inteligente para optimizar rendimiento
4. Análisis comparativo entre casas de apuestas
"""

import pandas as pd
import numpy as np
import requests
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import os
from pathlib import Path

from .sportradar_api import SportradarAPI
from .config.config import get_config
from .config.exceptions import (
    BookmakersAPIError,
    SportradarAPIError,
    InsufficientDataError,
    DataValidationError,
    CacheError
)

logger = logging.getLogger(__name__)

class BookmakersDataFetcher:
    """
    Fetcher avanzado para datos de cuotas con integración completa de Sportradar.
    
    Funcionalidades principales:
    - Sportradar API como fuente principal
    - APIs secundarias como respaldo
    - Cache inteligente con validación temporal
    - Simulación avanzada con varianza realista
    - Normalización de datos entre proveedores
    """
    
    def __init__(
        self,
        api_keys: Dict[str, str] = None,
        odds_data_dir: str = "data/bookmakers",
        cache_expiry: int = None,  # Horas - None para auto-optimización
        config_override: Optional[Dict] = None,
        auto_optimize_cache: bool = True
    ):
        """
        Inicializa el fetcher con Sportradar API únicamente.
        
        Args:
            api_keys: Diccionario con API keys (solo se usa 'sportradar')
            odds_data_dir: Directorio para datos y cache
            cache_expiry: Horas antes de que expire el cache (None para auto-optimización)
            config_override: Configuración personalizada
            auto_optimize_cache: Si optimizar automáticamente el cache según la temporada
        """
        # Configuración
        self.config = get_config()
        if config_override:
            for section, values in config_override.items():
                for key, value in values.items():
                    self.config.set(section, key, value=value)
        
        # Configuración básica
        self.api_keys = api_keys or {}
        self.odds_data_dir = Path(odds_data_dir)
        self.auto_optimize_cache = auto_optimize_cache
        
        # Configurar cache con optimización automática
        if cache_expiry is None and auto_optimize_cache:
            # Optimización automática según temporada
            seasonal_info = self._get_seasonal_info()
            self.cache_expiry = seasonal_info['recommendations']['cache_expiry_hours']
            logger.info(f"Cache auto-optimizado para temporada {seasonal_info['current_phase']}: {self.cache_expiry}h")
        else:
            self.cache_expiry = cache_expiry or 12  # Default 12 horas
        
        # Crear directorios necesarios
        self.odds_data_dir.mkdir(parents=True, exist_ok=True)
        (self.odds_data_dir / 'cache').mkdir(exist_ok=True)
        
        # Inicializar Sportradar API (única fuente de datos)
        sportradar_key = (
            self.api_keys.get('sportradar') or 
            self.config.get('sportradar', 'api_key') or
            os.getenv('SPORTRADAR_API')  
        )
        
        if sportradar_key:
            try:
                self.sportradar_api = SportradarAPI(
                    api_key=sportradar_key,
                    config_override=config_override.get('sportradar') if config_override else None
                )
                logger.info("Sportradar API inicializada correctamente")
                logger.info(f"API Key configurada: {'*' * (len(sportradar_key) - 4)}{sportradar_key[-4:]}")
                
                # Validar conexión (modo no crítico para APIs de trial)
                self._validate_sportradar_connection(critical=False)
                
            except Exception as e:
                logger.error(f"Error inicializando Sportradar API: {e}")
                raise SportradarAPIError(f"No se pudo inicializar Sportradar API: {e}")
        else:
            error_msg = "Sportradar API key no encontrada. Variable de entorno SPORTRADAR_API requerida."
            logger.error(error_msg)
            raise SportradarAPIError(error_msg)
        
        # Métricas de uso
        self.api_calls_made = 0
        self.cache_hits = 0
        self.cache_misses = 0

    def _validate_sportradar_connection(self, critical: bool = True):
        """
        Valida que la conexión con Sportradar funcione correctamente.
        
        Args:
            critical: Si es True, falla si no hay conexión. Si es False, solo advierte.
        """
        try:
            logger.info("Validando conexión con Sportradar API...")
            test_result = self.sportradar_api.test_connection()
            
            if test_result.get('success', False):
                logger.info("Conexión con Sportradar API validada correctamente")
            else:
                error_msg = f"Problema en conexión con Sportradar: {test_result.get('error', 'Unknown error')}"
                if critical:
                    logger.error(error_msg)
                    raise SportradarAPIError(error_msg)
                else:
                    logger.warning(error_msg)
                    logger.warning("Continuando con API configurada - puede funcionar para peticiones reales")
                
        except Exception as e:
            error_msg = f"Problema en validación de Sportradar API: {e}"
            if critical:
                logger.error(error_msg)
                raise SportradarAPIError(error_msg)
            else:
                logger.warning(error_msg)
                logger.warning("Continuando con API configurada - puede funcionar para peticiones reales")

    # === MÉTODOS SPORTRADAR (PROVEEDOR PRINCIPAL) ===
    
    def get_nba_odds_from_sportradar(
        self,
        date: Optional[str] = None,
        team_filter: Optional[List[str]] = None,
        include_props: bool = True
    ) -> Dict[str, Any]:
        """
        Obtiene cuotas NBA desde Sportradar API.
        
        Args:
            date: Fecha específica (YYYY-MM-DD). Si es None, obtiene próximos partidos
            team_filter: Lista de equipos para filtrar
            include_props: Si incluir player props
            
        Returns:
            Diccionario con cuotas organizadas por partido
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        try:
            # Usar método optimizado para obtener todos los targets a la vez
            if date:
                # Obtener odds para fecha específica
                odds_data = self.sportradar_api.get_nba_odds_for_targets(date=date)
            else:
                # Obtener odds para próximos partidos (método automatizado)
                odds_data = self.sportradar_api.get_live_and_upcoming_odds()
                
            # Procesar resultados
            if not odds_data.get('games', {}):
                today = datetime.now().strftime("%Y-%m-%d")
                return {
                    'success': True,
                    'date': date or today,
                    'games': [],
                    'total_games': 0
                }
            
            # Convertir datos al formato esperado
            games_with_odds = []
            
            # Obtener juegos de la estructura correcta según el método usado
            if 'all_games' in odds_data:
                games_dict = odds_data['all_games']
            else:
                games_dict = odds_data.get('games', {})
                
            for game_id, game_data in games_dict.items():
                # Filtrar por equipo si es necesario
                if team_filter:
                    home_team = game_data.get('teams', {}).get('home', '').lower()
                    away_team = game_data.get('teams', {}).get('away', '').lower()
                    
                    if not any(team.lower() in home_team or team.lower() in away_team 
                              for team in team_filter):
                        continue
                
                # Extraer datos del juego
                game_info = {
                    'game_id': game_id,
                    'home_team': game_data.get('teams', {}).get('home'),
                    'away_team': game_data.get('teams', {}).get('away'),
                    'scheduled': game_data.get('scheduled'),
                    'odds': game_data.get('odds', {}).get('game_markets', {}),
                    'props': game_data.get('odds', {}).get('player_props', {}) if include_props else None
                }
                
                games_with_odds.append(game_info)
                self.api_calls_made += 1
            
            return {
                'success': True,
                'source': 'sportradar',
                'date': date or odds_data.get('date', datetime.now().strftime("%Y-%m-%d")),
                'games': games_with_odds,
                'total_games': len(games_dict),
                'games_with_odds': len(games_with_odds),
                'api_calls_made': self.api_calls_made,
                'cache_stats': self.sportradar_api.get_cache_stats() if hasattr(self.sportradar_api, 'get_cache_stats') else {}
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de Sportradar: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': 'sportradar'
            }
    
    def get_player_props_sportradar(
        self,
        player_name: str,
        target: str = 'PTS',
        days_ahead: int = 7
    ) -> List[Dict[str, Any]]:
        """
        Obtiene player props específicos desde Sportradar.
        
        Args:
            player_name: Nombre del jugador
            target: Estadística objetivo (PTS, AST, TRB, 3P)
            days_ahead: Días hacia adelante para buscar
            
        Returns:
            Lista de props encontrados
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        try:
            # Obtener partidos próximos
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            
            games = self.sportradar_api.get_games_by_date_range(start_date, end_date)
            
            player_props = []
            
            for game in games:
                game_id = game.get('id')
                if game_id:
                    try:
                        props_data = self.sportradar_api.get_player_props(game_id)
                        
                        # Buscar props del jugador específico
                        if 'markets' in props_data:
                            for market in props_data['markets']:
                                if target.lower() in market.get('description', '').lower():
                                    for outcome in market.get('outcomes', []):
                                        player_in_outcome = outcome.get('player', {}).get('name', '')
                                        if player_name.lower() in player_in_outcome.lower():
                                            player_props.append({
                                                'game_id': game_id,
                                                'player': player_in_outcome,
                                                'target': target,
                                                'line': outcome.get('point'),
                                                'over_odds': outcome.get('over_odds'),
                                                'under_odds': outcome.get('under_odds'),
                                                'market': market.get('description'),
                                                'bookmaker': outcome.get('bookmaker'),
                                                'game_info': {
                                                    'home': game.get('home', {}).get('name'),
                                                    'away': game.get('away', {}).get('name'),
                                                    'date': game.get('scheduled')
                                                }
                                            })
                    except SportradarAPIError as e:
                        logger.warning(f"Error obteniendo props para {game_id}: {e}")
            
            return player_props
            
        except Exception as e:
            logger.error(f"Error buscando props para {player_name}: {e}")
            return []
    
    def get_market_overview_sportradar(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene resumen completo del mercado NBA desde Sportradar.
        
        Args:
            date: Fecha específica (YYYY-MM-DD)
            
        Returns:
            Resumen del mercado con estadísticas
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        return self.sportradar_api.get_market_overview(date)
    
    # === MÉTODOS ESPECÍFICOS PARA NUESTROS TARGETS DE PREDICCIÓN ===
    
    def get_optimized_props_for_targets(
        self,
        date: Optional[str] = None,
        players: Optional[List[str]] = None,
        targets: Optional[List[str]] = None,
        auto_adjust_frequency: bool = True
    ) -> Dict[str, Any]:
        """
        Obtiene props con optimización automática según la temporada.
        
        Args:
            date: Fecha específica (YYYY-MM-DD)
            players: Lista de jugadores específicos
            targets: Lista de targets específicos
            auto_adjust_frequency: Si ajustar automáticamente la frecuencia de llamadas
            
        Returns:
            Props optimizadas según la temporada actual
        """
        # Optimizar cache automáticamente
        if self.auto_optimize_cache:
            cache_optimization = self.optimize_cache_for_season()
            if cache_optimization['optimized']:
                logger.info(f"Cache optimizado automáticamente: {cache_optimization['reason']}")
        
        # Obtener información estacional
        seasonal_info = self.get_seasonal_props_availability()
        current_phase = seasonal_info['current_phase']
        
        # Ajustar targets según la temporada si no se especifican
        if targets is None:
            # Obtener targets prioritarios según la temporada
            priority_player_targets = seasonal_info['recommendations']['priority_targets']
            
            # Siempre incluir targets de equipos (disponibles todo el año)
            team_targets = ['is_win', 'total_points', 'teams_points']
            
            # Combinar targets de jugadores y equipos
            targets = priority_player_targets + team_targets
            logger.info(f"Targets ajustados para {current_phase}: Player={priority_player_targets}, Team={team_targets}")
        
        # Obtener props usando el método base
        props_data = self.get_player_props_for_targets(
            date=date,
            players=players,
            targets=targets
        )
        
        # Añadir información de optimización
        if props_data.get('success', False):
            props_data['optimization_info'] = {
                'season_phase': current_phase,
                'cache_hours': self.cache_expiry,
                'priority_targets': seasonal_info['recommendations']['priority_targets'],
                'api_frequency': seasonal_info['recommendations']['api_call_frequency'],
                'optimized': True
            }
        
        return props_data
    
    def get_player_props_for_targets(
        self,
        date: Optional[str] = None,
        players: Optional[List[str]] = None,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Obtiene props específicas para TODOS nuestros targets de predicción.
        
        PLAYER TARGETS:
        - PTS: Puntos del jugador
        - AST: Asistencias del jugador  
        - TRB: Rebotes del jugador
        - 3P: Triples del jugador
        - DD: Double-double del jugador
        
        TEAM/GAME TARGETS:
        - is_win: Victoria del equipo (1x2/moneyline)
        - total_points: Puntos totales del partido
        - teams_points: Puntos por equipo (home/away)
        
        Args:
            date: Fecha específica (YYYY-MM-DD). Si es None, obtiene próximos partidos
            players: Lista de jugadores específicos (opcional)
            targets: Lista de targets específicos (opcional, por defecto todos)
            
        Returns:
            Diccionario con props organizadas por target y jugador/equipo
        """
        if not self.sportradar_api:
            raise SportradarAPIError("Sportradar API no inicializada")
        
        # TODOS los targets disponibles en el sistema
        if targets is None:
            targets = [
                # Player targets
                'PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double',
                # Team/Game targets  
                'is_win', 'total_points', 'teams_points'
            ]
        
        # Mapeo de targets a mercados de Sportradar
        target_markets = {
            # Player props
            'PTS': ['total points', 'total points (incl. overtime)', 'player points'],
            'AST': ['total assists', 'total assists (incl. overtime)', 'player assists'],
            'TRB': ['total rebounds', 'total rebounds (incl. overtime)', 'player rebounds'],
            '3P': ['total threes', 'total three pointers', 'player threes', 'total 3-pointers'],
            'DD': ['double_double', 'double double', 'player double double'],
            'triple_double': ['triple double', 'triple double (incl. extra overtime)', 'player triple double'],
            
            # Team/Game props
            'is_win': ['1x2', 'moneyline', 'match_winner', 'winner'],
            'total_points': ['total_incl_overtime', 'total points', 'game total'],
            'teams_points': ['home_total_incl_overtime', 'away_total_incl_overtime', 'team total']
        }
        
        try:
            # Obtener próximos partidos
            if date:
                games_data = self.sportradar_api.get_nba_odds_for_targets(date=date)
            else:
                games_data = self.sportradar_api.get_live_and_upcoming_odds()
            
            if not games_data.get('success', False):
                logger.warning(f"No se pudieron obtener datos de partidos: {games_data.get('error', 'Unknown error')}")
                return {'success': False, 'error': 'No games data available'}
            
            # Procesar cada partido
            all_props = {
                'success': True,
                'date': date or datetime.now().strftime('%Y-%m-%d'),
                'targets': targets,
                'games': {},
                'summary': {
                    'total_games': 0,
                    'games_with_props': 0,
                    'total_props': 0,
                    'props_by_target': {target: 0 for target in targets}
                }
            }
            
            games_dict = games_data.get('games', {})
            if 'all_games' in games_data:
                games_dict = games_data['all_games']
            
            for game_id, game_data in games_dict.items():
                try:
                    # Obtener props del partido (player props y game markets)
                    game_props = self._get_comprehensive_game_props(
                        game_id, game_data, target_markets, players, targets
                    )
                    
                    if game_props['props']:
                        all_props['games'][game_id] = {
                            'game_info': {
                                'home_team': game_data.get('teams', {}).get('home', 'Unknown'),
                                'away_team': game_data.get('teams', {}).get('away', 'Unknown'),
                                'scheduled': game_data.get('scheduled', ''),
                                'status': game_data.get('status', 'scheduled')
                            },
                            'props': game_props['props'],
                            'stats': game_props['stats']
                        }
                        
                        # Actualizar estadísticas
                        all_props['summary']['games_with_props'] += 1
                        all_props['summary']['total_props'] += game_props['stats']['total_props']
                        
                        for target in targets:
                            all_props['summary']['props_by_target'][target] += game_props['stats'].get(f'{target}_props', 0)
                
                    self.api_calls_made += 1
                    
                except Exception as e:
                    logger.warning(f"Error procesando props para partido {game_id}: {e}")
                    continue
            
            all_props['summary']['total_games'] = len(games_dict)
            
            logger.info(f"Props obtenidas: {all_props['summary']['games_with_props']}/{all_props['summary']['total_games']} partidos con props")
            logger.info(f"Total props por target: {all_props['summary']['props_by_target']}")
            
            return all_props
            
        except Exception as e:
            logger.error(f"Error obteniendo props para targets: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': 'sportradar'
            }
    
    def _get_comprehensive_game_props(
        self,
        game_id: str,
        game_data: Dict[str, Any],
        target_markets: Dict[str, List[str]],
        players: Optional[List[str]] = None,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Obtiene props completas del partido incluyendo player props y game markets.
        
        Args:
            game_id: ID del partido
            game_data: Datos del partido
            target_markets: Mapeo de targets a mercados
            players: Lista de jugadores específicos
            targets: Lista de targets específicos
            
        Returns:
            Props completas del partido
        """
        comprehensive_props = {
            'props': {},
            'stats': {
                'total_props': 0,
                'PTS_props': 0,
                'AST_props': 0,
                'TRB_props': 0,
                '3P_props': 0,
                'DD_props': 0,
                'triple_double_props': 0,
                'is_win_props': 0,
                'total_points_props': 0,
                'teams_points_props': 0
            }
        }
        
        # Inicializar estructura por target
        for target in targets or ['PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double', 'is_win', 'total_points', 'teams_points']:
            comprehensive_props['props'][target] = []
        
        # 1. Obtener PLAYER PROPS desde Sportradar Player Props API
        # INCLUYE: PTS, AST, TRB, 3P, DD, triple_double (todos son player props)
        player_targets = [t for t in targets if t in ['PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double']]
        if player_targets:
            try:
                player_props_data = self.sportradar_api.get_player_props(game_id)
                if player_props_data.get('success', False):
                    player_props = self._filter_target_props(
                        player_props_data, 
                        {k: v for k, v in target_markets.items() if k in player_targets}, 
                        players, 
                        player_targets
                    )
                    
                    # Combinar player props
                    for target in player_targets:
                        comprehensive_props['props'][target].extend(player_props['props'].get(target, []))
                        comprehensive_props['stats'][f'{target}_props'] += player_props['stats'].get(f'{target}_props', 0)
                        comprehensive_props['stats']['total_props'] += player_props['stats'].get(f'{target}_props', 0)
                        
            except Exception as e:
                logger.warning(f"Error obteniendo player props para {game_id}: {e}")
        
        # 2. Obtener GAME MARKETS (solo team/game props) desde PREMATCH API
        # SOLO INCLUYE: is_win, total_points, teams_points (markets de equipos/juego)
        team_targets = [t for t in targets if t in ['is_win', 'total_points', 'teams_points']]
        if team_targets:
            try:
                # Usar Prematch API (Odds Comparison v2) para obtener game markets
                prematch_data = self.sportradar_api.get_prematch_odds(game_id)
                
                if prematch_data.get('success', False):
                    markets_data = prematch_data.get('markets', [])
                    
                    for market in markets_data:
                        market_id = market.get('id')
                        market_name = market.get('name', '').lower()
                        
                        # Mapear market_id a nuestros targets usando configuración
                        target_found = None
                        
                        # is_win: Market ID 1 o sr:market:1 (1x2/moneyline)
                        if (market_id == 1 or market_id == 'sr:market:1') and 'is_win' in team_targets:
                            target_found = 'is_win'
                        
                        # total_points: Market ID 225 o sr:market:225 (total_incl_overtime)
                        elif (market_id == 225 or market_id == 'sr:market:225') and 'total_points' in team_targets:
                            target_found = 'total_points'
                        
                        # teams_points: Market ID 227/228 o sr:market:227/228 (home/away totals)
                        elif (market_id in [227, 228] or market_id in ['sr:market:227', 'sr:market:228']) and 'teams_points' in team_targets:
                            target_found = 'teams_points'
                        
                        if target_found:
                            # Procesar outcomes del market
                            for outcome in market.get('outcomes', []):
                                prop = {
                                    'target': target_found,
                                    'market': market.get('name', ''),
                                    'market_id': market_id,
                                    'outcome': outcome.get('name', ''),
                                    'line': outcome.get('point') or outcome.get('total'),
                                    'odds': {
                                        'decimal': outcome.get('odds_decimal'),
                                        'american': outcome.get('odds_american'),
                                        'fractional': outcome.get('odds_fraction')
                                    },
                                    'bookmaker': outcome.get('bookmaker', 'sportradar'),
                                    'last_update': datetime.now().isoformat()
                                }
                                
                                # Agregar información específica del target
                                if target_found == 'is_win':
                                    prop['team'] = outcome.get('competitor', outcome.get('team', ''))
                                elif target_found == 'total_points':
                                    prop['team'] = 'both'  # Total del partido
                                elif target_found == 'teams_points':
                                    # Distinguir entre home (227/sr:market:227) y away (228/sr:market:228)
                                    prop['team'] = 'home' if market_id in [227, 'sr:market:227'] else 'away'
                                
                                comprehensive_props['props'][target_found].append(prop)
                                comprehensive_props['stats'][f'{target_found}_props'] += 1
                                comprehensive_props['stats']['total_props'] += 1
                else:
                    logger.warning(f"No se pudieron obtener datos de Prematch API para {game_id}")
                                
            except Exception as e:
                logger.warning(f"Error obteniendo game markets desde Prematch API para {game_id}: {e}")
        
        return comprehensive_props
    
    def _filter_target_props(
        self,
        props_data: Dict[str, Any],
        target_markets: Dict[str, List[str]],
        players: Optional[List[str]] = None,
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Filtra props por nuestros targets específicos.
        
        Args:
            props_data: Datos de props del partido
            target_markets: Mapeo de targets a mercados
            players: Lista de jugadores específicos
            targets: Lista de targets específicos
            
        Returns:
            Props filtradas y organizadas
        """
        filtered_props = {
            'props': {},
            'stats': {
                'total_props': 0,
                'PTS_props': 0,
                'AST_props': 0,
                'TRB_props': 0,
                '3P_props': 0,
                'DD_props': 0,
                'triple_double_props': 0
            }
        }
        
        # Inicializar estructura por target
        for target in targets or ['PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double']:
            filtered_props['props'][target] = []
        
        # Procesar markets del partido
        markets = props_data.get('markets', [])
        if not markets and 'player_props' in props_data:
            # Formato alternativo
            markets = []
            for player_name, player_data in props_data['player_props'].items():
                for prop_type, prop_details in player_data.items():
                    markets.append({
                        'description': prop_type,
                        'player': player_name,
                        'outcomes': prop_details.get('lines', [])
                    })
        
        for market in markets:
            market_name = market.get('description', '').lower()
            
            # Identificar target
            target_found = None
            for target, market_keywords in target_markets.items():
                if any(keyword.lower() in market_name for keyword in market_keywords):
                    target_found = target
                    break
            
            if not target_found or target_found not in (targets or ['PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double']):
                continue
            
            # Procesar outcomes del market
            for outcome in market.get('outcomes', []):
                player_name = outcome.get('player_name') or market.get('player', '')
                
                # Filtrar por jugador si se especifica
                if players and not any(player.lower() in player_name.lower() for player in players):
                    continue
                
                # Crear prop estructurada
                prop = {
                    'player': player_name,
                    'target': target_found,
                    'market': market_name,
                    'line': outcome.get('total') or outcome.get('point'),
                    'over_odds': {
                        'decimal': outcome.get('odds_decimal'),
                        'american': outcome.get('odds_american'),
                        'fractional': outcome.get('odds_fraction')
                    },
                    'under_odds': {
                        'decimal': outcome.get('under_odds_decimal'),
                        'american': outcome.get('under_odds_american'),
                        'fractional': outcome.get('under_odds_fraction')
                    },
                    'bookmaker': outcome.get('bookmaker', 'sportradar'),
                    'last_update': datetime.now().isoformat()
                }
                
                # Solo agregar si tiene línea válida
                if prop['line'] is not None:
                    filtered_props['props'][target_found].append(prop)
                    filtered_props['stats']['total_props'] += 1
                    filtered_props['stats'][f'{target_found}_props'] += 1
        
        return filtered_props
    
    def get_seasonal_props_availability(self) -> Dict[str, Any]:
        """
        Analiza la disponibilidad de props según la temporada NBA.
        
        Returns:
            Información sobre disponibilidad y recomendaciones
        """
        current_date = datetime.now()
        current_month = current_date.month
        
        # Definir fases de la temporada NBA
        season_phases = {
            'offseason': {
                'months': [7, 8],  # Julio-Agosto
                'description': 'Temporada baja',
                'props_availability': 'Mínima',
                'activity_level': 'Bajo',
                'recommended_actions': [
                    'Mantener cache de datos históricos',
                    'Preparar sistema para nueva temporada',
                    'Actualizar datos de jugadores y equipos'
                ]
            },
            'preseason': {
                'months': [9, 10],  # Septiembre-Octubre
                'description': 'Pretemporada y inicio',
                'props_availability': 'Moderada',
                'activity_level': 'Medio',
                'recommended_actions': [
                    'Comenzar monitoreo regular',
                    'Validar modelos con datos frescos',
                    'Ajustar parámetros de cache'
                ]
            },
            'regular_season': {
                'months': [11, 12, 1, 2, 3, 4],  # Noviembre-Abril
                'description': 'Temporada regular',
                'props_availability': 'Máxima',
                'activity_level': 'Alto',
                'recommended_actions': [
                    'Monitoreo intensivo',
                    'Actualizaciones frecuentes',
                    'Máxima utilización de APIs'
                ]
            },
            'playoffs': {
                'months': [4, 5, 6],  # Abril-Junio
                'description': 'Playoffs',
                'props_availability': 'Alta',
                'activity_level': 'Alto',
                'recommended_actions': [
                    'Enfoque en equipos clasificados',
                    'Análisis de tendencias playoff',
                    'Monitoreo de lesiones críticas'
                ]
            }
        }
        
        # Determinar fase actual
        current_phase = None
        for phase, info in season_phases.items():
            if current_month in info['months']:
                current_phase = phase
                break
        
        if not current_phase:
            current_phase = 'offseason'  # Default
        
        # Calcular días hasta próxima temporada regular
        if current_month in [7, 8, 9, 10]:
            # Calcular días hasta noviembre
            next_season_start = datetime(current_date.year, 11, 1)
            if current_month >= 11:
                next_season_start = datetime(current_date.year + 1, 11, 1)
        else:
            next_season_start = None
        
        days_to_season = (next_season_start - current_date).days if next_season_start else 0
        
        return {
            'current_date': current_date.strftime('%Y-%m-%d'),
            'current_month': current_month,
            'current_phase': current_phase,
            'phase_info': season_phases[current_phase],
            'days_to_regular_season': max(0, days_to_season),
            'recommendations': {
                'cache_expiry_hours': self._get_recommended_cache_expiry(current_phase),
                'api_call_frequency': self._get_recommended_api_frequency(current_phase),
                'priority_targets': self._get_priority_targets(current_phase)
            },
            'all_phases': season_phases
        }
    
    def _get_recommended_cache_expiry(self, phase: str) -> int:
        """Obtiene tiempo de expiración de cache recomendado según la fase."""
        cache_settings = {
            'offseason': 24,      # 24 horas - datos cambian poco
            'preseason': 12,      # 12 horas - actividad moderada
            'regular_season': 2,  # 2 horas - máxima actividad
            'playoffs': 1         # 1 hora - actividad crítica
        }
        return cache_settings.get(phase, 12)
    
    def _get_recommended_api_frequency(self, phase: str) -> str:
        """Obtiene frecuencia recomendada de llamadas API según la fase."""
        frequency_settings = {
            'offseason': 'daily',
            'preseason': 'every_6_hours',
            'regular_season': 'every_2_hours',
            'playoffs': 'hourly'
        }
        return frequency_settings.get(phase, 'daily')
    
    def _get_priority_targets(self, phase: str) -> List[str]:
        """Obtiene targets prioritarios según la fase."""
        if phase in ['regular_season', 'playoffs']:
            return ['PTS', 'AST', 'TRB', '3P', 'DD', 'triple_double']  # Todos los targets incluyendo DD y triple double
        elif phase == 'preseason':
            return ['PTS', 'TRB', 'DD']  # Targets más estables incluyendo DD
        else:
            return ['PTS']  # Solo target principal
    
    def _get_seasonal_info(self) -> Dict[str, Any]:
        """
        Obtiene información básica de temporada para inicialización.
        Versión simplificada de get_seasonal_props_availability() para evitar recursión.
        """
        current_date = datetime.now()
        current_month = current_date.month
        
        # Determinar fase actual
        if current_month in [7, 8]:
            phase = 'offseason'
        elif current_month in [9, 10]:
            phase = 'preseason'
        elif current_month in [11, 12, 1, 2, 3, 4]:
            phase = 'regular_season'
        else:  # 4, 5, 6 (overlapping with regular season)
            phase = 'playoffs'
        
        return {
            'current_phase': phase,
            'recommendations': {
                'cache_expiry_hours': self._get_recommended_cache_expiry(phase),
                'api_call_frequency': self._get_recommended_api_frequency(phase),
                'priority_targets': self._get_priority_targets(phase)
            }
        }
    
    def optimize_cache_for_season(self) -> Dict[str, Any]:
        """
        Optimiza automáticamente el cache según la temporada actual.
        
        Returns:
            Información sobre la optimización realizada
        """
        if not self.auto_optimize_cache:
            return {
                'optimized': False,
                'reason': 'Auto-optimización deshabilitada'
            }
        
        # Obtener información estacional
        seasonal_info = self.get_seasonal_props_availability()
        current_phase = seasonal_info['current_phase']
        recommended_cache = seasonal_info['recommendations']['cache_expiry_hours']
        old_cache = self.cache_expiry
        
        # Aplicar optimización
        if old_cache != recommended_cache:
            self.cache_expiry = recommended_cache
            logger.info(f"Cache optimizado: {old_cache}h → {recommended_cache}h para fase {current_phase}")
            
            return {
                'optimized': True,
                'old_cache_hours': old_cache,
                'new_cache_hours': recommended_cache,
                'phase': current_phase,
                'reason': f'Optimización automática para {current_phase}'
            }
        else:
            return {
                'optimized': False,
                'cache_hours': self.cache_expiry,
                'phase': current_phase,
                'reason': 'Cache ya está optimizado'
            }

    def load_odds_from_api(
        self,
        sport: str = 'basketball_nba',
        markets: List[str] = None,
        bookmakers: List[str] = None,
        api_provider: str = 'sportradar',  # Cambiado a Sportradar como default
        force_refresh: bool = False,
        date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Carga odds desde un proveedor de API (The Odds API o similar)
        
        Este método:
        1. Consulta APIs externas para obtener datos actualizados de odds
        2. Gestiona caché para minimizar llamadas a la API y cumplir con límites
        3. Soporta diferentes proveedores de API y tipos de mercados
        
        Args:
            sport: Deporte a consultar
            markets: Lista de mercados (h2h, spreads, totals, player_props)
            bookmakers: Lista de casas de apuestas a incluir
            api_provider: Proveedor de API ('odds_api', 'sportsdata_io', etc.)
            force_refresh: Si forzar la actualización ignorando caché
            
        Returns:
            Datos de odds en formato JSON
        """
        # Mercados por defecto si no se especifican
        if markets is None:
            markets = ['h2h', 'spreads', 'totals', 'player_props']
            
        # Casas de apuestas por defecto
        if bookmakers is None:
            bookmakers = list(self.supported_bookmakers.keys())
        
        # Archivo de caché para esta consulta
        cache_key = f"{sport}_{'-'.join(markets)}_{'-'.join(sorted(bookmakers))}"
        cache_file = self.odds_data_dir / f"{cache_key}.json"
        
        # Verificar si tenemos datos en caché y son recientes
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                
                # Verificar si los datos son recientes
                last_update = datetime.fromisoformat(cached_data.get('last_update', '2000-01-01'))
                if datetime.now() - last_update < timedelta(hours=self.cache_expiry):
                    logger.info(f"Usando datos de odds en caché para {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"Error al leer caché: {e}")
        
        # Si llegamos aquí, necesitamos obtener nuevos datos
        # Diferentes implementaciones según el proveedor
        if api_provider == 'sportradar':
            # Usar Sportradar como proveedor principal
            if not self.sportradar_api:
                logger.error("Sportradar API no está configurada")
                return {'success': False, 'error': "Sportradar API no configurada"}
            
            try:
                # Obtener datos usando los métodos específicos de Sportradar
                if sport == 'basketball_nba':
                    odds_data = self.get_nba_odds_from_sportradar(
                        date=date,
                        include_props=True
                    )
                    
                    # Asegurar que el formato sea consistente
                    if odds_data.get('success', False):
                        # Convertir al formato esperado por el sistema
                        odds_data = self._convert_sportradar_to_standard_format(odds_data)
                else:
                    logger.error(f"Deporte no soportado en Sportradar: {sport}")
                    return {'success': False, 'error': f"Deporte no soportado: {sport}"}
            except Exception as e:
                logger.error(f"Error obteniendo datos de Sportradar: {e}")
                return {'success': False, 'error': str(e)}
                

            
        else:
            logger.error(f"Proveedor de API no soportado: {api_provider}. Solo se soporta 'sportradar'")
            return {'success': False, 'error': f"Proveedor no soportado: {api_provider}. Solo se soporta 'sportradar'"}
        
        # Verificar si fue exitoso
        if odds_data.get('success', False):
            # Añadir timestamp
            odds_data['last_update'] = datetime.now().isoformat()
            
            # Guardar en caché
            try:
                with open(cache_file, 'w') as f:
                    json.dump(odds_data, f)
                logger.info(f"Datos de odds guardados en caché para {cache_key}")
            except Exception as e:
                logger.warning(f"Error al guardar caché: {e}")
        
        return odds_data
    
    def merge_odds_with_player_data(
        self,
        player_data: pd.DataFrame,
        odds_data: Dict[str, Any],
        target: str = 'PTS',
        line_values: List[float] = None
    ) -> pd.DataFrame:
        """
        Combina datos de jugadores con odds de casas de apuestas
        
        Este método es crucial para:
        1. Integrar datos de diferentes fuentes (predicciones + mercado)
        2. Alinear correctamente los datos por jugador, fecha y línea de apuesta 
        3. Crear columnas específicas para cada combinación de línea y casa
        4. Habilitar análisis comparativo entre nuestras predicciones y el mercado
        
        Args:
            player_data: DataFrame con datos de jugadores
            odds_data: Datos de odds obtenidos mediante load_odds_from_api o load_odds_from_file
            target: Estadística objetivo (PTS, TRB, AST, 3P)
            line_values: Lista de valores de línea a analizar
            
        Returns:
            DataFrame combinado con datos de jugadores y odds de casas
        """
        if not odds_data.get('success', False) or not odds_data.get('data'):
            logger.error("No hay datos de odds válidos para combinar")
            return player_data
        
        # Crear copia para no modificar el original
        df = player_data.copy()
        
        # Líneas por defecto según el target si no se especifican
        if line_values is None:
            if target == 'PTS':
                line_values = [10, 15, 20, 25, 30, 35]
            elif target == 'TRB':
                line_values = [4, 6, 8, 10, 12]
            elif target == 'AST':
                line_values = [4, 6, 8, 10, 12]
            elif target == '3P':
                line_values = [1, 2, 3, 4, 5]
            else:
                line_values = []
        
        # Procesar datos de odds - solo formato Sportradar API
        try:
            # Solo procesamos datos de Sportradar API
            if not odds_data.get('success', False):
                logger.error("No hay datos válidos de Sportradar para combinar")
                return player_data
            
            # Aquí se implementaría la lógica específica de combinación con Sportradar
            # Por ahora retornamos los datos originales
            logger.info("Combinación de odds con datos de jugadores - implementación pendiente")
            return df
            
        except Exception as e:
            logger.error(f"Error al combinar datos de odds: {str(e)}")
            return player_data

 
    
    # === MÉTODOS DE UTILIDAD Y GESTIÓN ===
    
    def get_api_status(self) -> Dict[str, Any]:
        """
        Obtiene el estado de la API de Sportradar.
        
        Returns:
            Estado de Sportradar API
        """
        status = {
            'sportradar': {'configured': False, 'accessible': False, 'error': None}
        }
        
        # Verificar Sportradar
        if self.sportradar_api:
            status['sportradar']['configured'] = True
            try:
                test_result = self.sportradar_api.test_connection()
                status['sportradar']['accessible'] = test_result['success']
                if not test_result['success']:
                    status['sportradar']['error'] = test_result.get('error', 'Unknown error')
            except Exception as e:
                status['sportradar']['error'] = str(e)
        
        return status
    
    def get_cache_status(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Estadísticas del cache
        """
        cache_stats = {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses),
            'api_calls_made': self.api_calls_made,
            'cache_directory': str(self.odds_data_dir),
            'cache_expiry_hours': self.cache_expiry
        }
        
        # Añadir estadísticas de Sportradar si está disponible
        if self.sportradar_api:
            try:
                sportradar_cache = self.sportradar_api.get_cache_stats()
                cache_stats['sportradar_cache'] = sportradar_cache
            except:
                pass
        
        return cache_stats
    
    def clear_all_cache(self):
        """Limpia todo el cache."""
        try:
            # Limpiar cache de archivos
            cache_dir = self.odds_data_dir / 'cache'
            for cache_file in cache_dir.glob('*.json'):
                cache_file.unlink()
            
            # Limpiar cache de Sportradar
            if self.sportradar_api:
                self.sportradar_api.clear_cache()
            
            # Resetear contadores
            self.cache_hits = 0
            self.cache_misses = 0
            
            logger.info("Cache limpiado completamente")
            
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
            raise CacheError(f"Error limpiando cache: {e}")
    
    def get_supported_features(self) -> Dict[str, List[str]]:
        """
        Obtiene características soportadas por Sportradar.
        
        Returns:
            Características de Sportradar
        """
        features = {
            'sportradar': [
                'nba_odds',
                'player_props',
                'game_schedule',
                'team_info',
                'market_overview',
                'real_time_updates',
                'cache_management',
                'seasonal_optimization'
            ]
        }
        
        return features
    
    def export_data_summary(self, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Exporta resumen de datos obtenidos.
        
        Args:
            output_file: Archivo opcional para guardar el resumen
            
        Returns:
            Resumen de datos
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'api_status': self.get_api_status(),
            'cache_status': self.get_cache_status(),
            'supported_features': self.get_supported_features(),
            'configuration': {
                'cache_expiry_hours': self.cache_expiry,
                'data_directory': str(self.odds_data_dir),
                'sportradar_enabled': self.sportradar_api is not None
            }
        }
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                logger.info(f"Resumen exportado a {output_file}")
            except Exception as e:
                logger.error(f"Error exportando resumen: {e}")
        
        return summary
    
    def __str__(self) -> str:
        """Representación string del fetcher."""
        sportradar_status = "OK" if self.sportradar_api else "NO CONFIGURADO"
        return (f"BookmakersDataFetcher("
                f"sportradar={sportradar_status}, "
                f"cache_hits={self.cache_hits}, "
                f"api_calls={self.api_calls_made})")
    
    def __repr__(self) -> str:
        """Representación detallada del fetcher."""
        return self.__str__()
    
    def _convert_sportradar_to_standard_format(self, sportradar_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convierte datos de Sportradar al formato estándar esperado por el sistema.
        
        Args:
            sportradar_data: Datos en formato Sportradar
            
        Returns:
            Datos en formato estándar
        """
        try:
            # Estructura estándar para el sistema
            standard_data = {
                'success': True,
                'source': 'sportradar',
                'data': [],
                'last_update': datetime.now().isoformat(),
                'games_processed': 0
            }
            
            # Procesar cada juego
            for game_id, game_data in sportradar_data.get('games', {}).items():
                # Extraer información básica del juego
                game_info = {
                    'id': game_id,
                    'commence_time': game_data.get('scheduled'),
                    'home_team': game_data.get('home_team'),
                    'away_team': game_data.get('away_team'),
                    'sport_key': 'basketball_nba',
                    'sport_title': 'NBA',
                    'bookmakers': []
                }
                
                # Procesar odds del juego
                odds_data = game_data.get('odds', {})
                
                # Crear estructura de bookmaker genérica
                bookmaker_data = {
                    'key': 'sportradar',
                    'title': 'Sportradar',
                    'last_update': datetime.now().isoformat(),
                    'markets': []
                }
                
                # Procesar game markets (moneyline, totals, etc.)
                game_markets = odds_data.get('game_markets', {})
                for market_type, market_data in game_markets.items():
                    market_info = {
                        'key': market_type,
                        'last_update': datetime.now().isoformat(),
                        'outcomes': []
                    }
                    
                    # Procesar outcomes del market
                    for outcome in market_data.get('outcomes', []):
                        outcome_info = {
                            'name': outcome.get('name'),
                            'price': outcome.get('price'),
                            'point': outcome.get('point')
                        }
                        market_info['outcomes'].append(outcome_info)
                    
                    bookmaker_data['markets'].append(market_info)
                
                # Procesar player props
                player_props = odds_data.get('player_props', {})
                for prop_type, props_data in player_props.items():
                    for player_name, player_data in props_data.items():
                        market_info = {
                            'key': f'player_{prop_type}',
                            'last_update': datetime.now().isoformat(),
                            'outcomes': []
                        }
                        
                        # Procesar líneas del jugador
                        for line_data in player_data.get('lines', []):
                            # Over outcome
                            over_outcome = {
                                'name': f"{player_name} Over",
                                'price': line_data.get('over_odds'),
                                'point': line_data.get('line'),
                                'player': player_name
                            }
                            market_info['outcomes'].append(over_outcome)
                            
                            # Under outcome
                            under_outcome = {
                                'name': f"{player_name} Under",
                                'price': line_data.get('under_odds'),
                                'point': line_data.get('line'),
                                'player': player_name
                            }
                            market_info['outcomes'].append(under_outcome)
                        
                        bookmaker_data['markets'].append(market_info)
                
                game_info['bookmakers'].append(bookmaker_data)
                standard_data['data'].append(game_info)
                standard_data['games_processed'] += 1
            
            logger.info(f"Convertidos {standard_data['games_processed']} juegos de Sportradar a formato estándar")
            return standard_data
            
        except Exception as e:
            logger.error(f"Error convirtiendo datos de Sportradar: {e}")
            return {
                'success': False,
                'error': f"Error en conversión: {e}",
                'source': 'sportradar'
            } 