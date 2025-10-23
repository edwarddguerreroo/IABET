"""
Sportradar API Client - VERSIÓN OPTIMIZADA
===========================================

Cliente limpio y eficiente para obtener cuotas de Sportradar:
- Team Odds: Moneyline (is_win), Total Points, Team Points, Halftime
- Player Props: PTS, AST, TRB, 3PT, Double-Double

"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config.config import get_config
from .config.exceptions import (
    SportradarAPIError,
    RateLimitError,
    AuthenticationError,
    NetworkError,
    create_http_error
)

logger = logging.getLogger(__name__)


class SportradarAPI:
    """
    Cliente optimizado para Sportradar API.
    
    Obtiene cuotas de:
    - Teams: Moneyline, Total Points, Team Points, Halftime
    - Players: PTS, AST, TRB, 3PT, Double-Double
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el cliente de Sportradar.
        
        Args:
            api_key: API key de Sportradar
        """
        # Configuración
        self.config = get_config()
        
        # API Key
        self.api_key = api_key or self.config.config['player_props_api']['api_key']
        if not self.api_key:
            raise AuthenticationError("API key de Sportradar requerida")
        
        # URLs base
        self.player_props_url = self.config.get_api_url('player_props')
        self.prematch_url = self.config.get_api_url('prematch')
        
        # Configuración
        self.timeout = 30
        self.sport_id = 2  # NBA
        
        # Rate limiting
        self.rate_limit_calls = 5
        self.rate_limit_period = 1
        self.last_calls = []
        
        # Sesión HTTP
        self.session = self._setup_session()
        
        # Cache
        from .config.optimized_cache import OptimizedCache
        self._cache = OptimizedCache(self.config.get_data_config())
        
        # IDs de mercados (según documentación verificada)
        self.MARKETS = {
            # Player Props
            'PTS': 'sr:market:921',  # total points (incl. overtime)
            'AST': 'sr:market:922',  # total assists (incl. overtime)
            'TRB': 'sr:market:923',  # total rebounds (incl. overtime)
            '3PT': 'sr:market:924',  # total 3-point field goals (incl. overtime)
            'DD': 'sr:market:8008',  # double double (incl. extra overtime)
            
            # Team Odds
            'is_win': 'sr:market:219',        # winner (incl. overtime)
            'total_points': 'sr:market:225',  # total (incl. overtime)
            'home_points': 'sr:market:227',   # home total (incl. overtime)
            'away_points': 'sr:market:228',   # away total (incl. overtime)
            'ht_total': 'sr:market:68',       # 1st half - total
            'ht_handicap': 'sr:market:66',    # 1st half - handicap
        }
        
        logger.info(f"SportradarAPI inicializada | Player Props: {self.player_props_url}")
    
    def _setup_session(self) -> requests.Session:
        """Configura sesión HTTP con retry."""
        session = requests.Session()
        retry = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session
    
    def _check_rate_limit(self):
        """Verifica y aplica rate limiting."""
        now = time.time()
        self.last_calls = [t for t in self.last_calls if now - t < self.rate_limit_period]
        
        if len(self.last_calls) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self.last_calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit alcanzado. Esperando {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        self.last_calls.append(now)
    
    def _make_request(self, url: str, params: Dict = None) -> Dict[str, Any]:
        """
        Realiza petición HTTP con manejo de errores.
        
        Args:
            url: URL completa del endpoint
            params: Parámetros de query
            
        Returns:
            Respuesta JSON parseada
        """
        self._check_rate_limit()
        
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                raise AuthenticationError(f"API key inválida: {response.status_code}")
            elif response.status_code == 429:
                raise RateLimitError("Rate limit excedido")
            elif response.status_code == 404:
                logger.warning(f"Endpoint no encontrado: {url}")
                return {}
            else:
                raise create_http_error(response, url)
                
        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Error de red: {e}")
    
    # =========================================================================
    # MÉTODOS PRINCIPALES - PLAYER PROPS
    # =========================================================================
    
    def get_player_props_by_date(self, date: str) -> Dict[str, Any]:
        """
        Obtiene player props para todos los partidos en una fecha.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            
        Returns:
            {
                'success': bool,
                'data': {
                    'events': [
                        {
                            'sport_event_id': str,
                            'home_team': str,
                            'away_team': str,
                            'start_time': str,
                            'players': [
                                {
                                    'player_id': str,
                                    'player_name': str,
                                    'team_id': str,
                                    'props': {
                                        'PTS': {'over': float, 'under': float, 'line': float},
                                        'AST': {...},
                                        'TRB': {...},
                                        '3PT': {...}
                                    }
                                }
                            ]
                        }
                    ]
                }
            }
        """
        endpoint = f"en/sports/sr:sport:{self.sport_id}/schedules/{date}/players_props.json"
        url = f"{self.player_props_url}/{endpoint}"
        params = {'api_key': self.api_key}
        
        try:
            logger.info(f"Obteniendo player props para {date}")
            data = self._make_request(url, params)
            
            if not data:
                return {'success': False, 'error': 'No data returned'}
            
            # Procesar respuesta
            events_data = data.get('sport_schedule_sport_events_players_props', [])
            processed_events = []
            
            for event in events_data:
                sport_event = event.get('sport_event', {})
                players_props = event.get('players_props', [])
                
                # Info del partido
                competitors = sport_event.get('competitors', [])
                home_team = next((c['name'] for c in competitors if c.get('qualifier') == 'home'), 'Unknown')
                away_team = next((c['name'] for c in competitors if c.get('qualifier') == 'away'), 'Unknown')
                
                # Procesar jugadores
                processed_players = []
                for player_prop in players_props:
                    player_info = player_prop.get('player', {})
                    markets = player_prop.get('markets', [])
                    
                    props = {}
                    for market in markets:
                        market_name = market.get('name', '')
                        
                        # Mapear a nuestros targets
                        target = None
                        if 'total points' in market_name and 'plus' not in market_name:
                            target = 'PTS'
                        elif 'total assists' in market_name and 'plus' not in market_name:
                            target = 'AST'
                        elif 'total rebounds' in market_name and 'plus' not in market_name:
                            target = 'TRB'
                        elif 'total 3-point' in market_name:
                            target = '3PT'
                        elif 'double double' in market_name:
                            target = 'DD'
                        
                        if target:
                            # Extraer mejor línea
                            best_line = self._extract_best_line(market)
                            if best_line:
                                props[target] = best_line
                    
                    if props:  # Solo agregar si tiene props
                        processed_players.append({
                            'player_id': player_info.get('id'),
                            'player_name': player_info.get('name'),
                            'team_id': player_info.get('competitor_id'),
                            'props': props
                        })
                
                if processed_players:  # Solo agregar evento si tiene jugadores con props
                    processed_events.append({
                        'sport_event_id': sport_event.get('id'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'start_time': sport_event.get('start_time'),
                        'players': processed_players
                    })
            
            logger.info(f"Player props procesados: {len(processed_events)} eventos, {sum(len(e['players']) for e in processed_events)} jugadores")
            
            return {
                'success': True,
                'data': {
                    'date': date,
                    'total_events': len(processed_events),
                    'events': processed_events
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo player props: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_player_props_by_event(self, sport_event_id: str) -> Dict[str, Any]:
        """
        Obtiene player props para un partido específico.
        
        Args:
            sport_event_id: ID del evento (ej: sr:sport_event:12345)
            
        Returns:
            Similar a get_player_props_by_date pero para un solo evento
        """
        event_id_encoded = sport_event_id.replace(':', '%3A')
        endpoint = f"en/sport_events/{event_id_encoded}/players_props.json"
        url = f"{self.player_props_url}/{endpoint}"
        params = {'api_key': self.api_key}
        
        try:
            logger.info(f"Obteniendo player props para evento {sport_event_id}")
            data = self._make_request(url, params)
            
            if not data:
                return {'success': False, 'error': 'No data returned'}
            
            # Procesar respuesta (estructura similar a by_date pero con clave diferente)
            event_props = data.get('sport_event_players_props', {})
            sport_event = event_props.get('sport_event', {})
            players_props = event_props.get('players_props', [])
            
            # Info del partido
            competitors = sport_event.get('competitors', [])
            home_team = next((c['name'] for c in competitors if c.get('qualifier') == 'home'), 'Unknown')
            away_team = next((c['name'] for c in competitors if c.get('qualifier') == 'away'), 'Unknown')
            
            # Procesar jugadores (mismo proceso que by_date)
            processed_players = []
            for player_prop in players_props:
                player_info = player_prop.get('player', {})
                markets = player_prop.get('markets', [])
                
                props = {}
                for market in markets:
                    market_name = market.get('name', '')
                    
                    target = None
                    if 'total points' in market_name and 'plus' not in market_name:
                        target = 'PTS'
                    elif 'total assists' in market_name and 'plus' not in market_name:
                        target = 'AST'
                    elif 'total rebounds' in market_name and 'plus' not in market_name:
                        target = 'TRB'
                    elif 'total 3-point' in market_name:
                        target = '3PT'
                    elif 'double double' in market_name:
                        target = 'DD'
                    
                    if target:
                        best_line = self._extract_best_line(market)
                        if best_line:
                            props[target] = best_line
                
                if props:
                    processed_players.append({
                        'player_id': player_info.get('id'),
                        'player_name': player_info.get('name'),
                        'team_id': player_info.get('competitor_id'),
                        'props': props
                    })
            
            logger.info(f"Player props procesados: {len(processed_players)} jugadores")
            
            return {
                'success': True,
                'data': {
                    'sport_event_id': sport_event.get('id'),
                    'home_team': home_team,
                    'away_team': away_team,
                    'start_time': sport_event.get('start_time'),
                    'players': processed_players
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo player props para evento: {e}")
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # MÉTODOS PRINCIPALES - TEAM ODDS
    # =========================================================================
    
    def get_team_odds_by_date(self, date: str) -> Dict[str, Any]:
        """
        Obtiene odds de equipos para todos los partidos en una fecha.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            
        Returns:
            {
                'success': bool,
                'data': {
                    'events': [
                        {
                            'sport_event_id': str,
                            'home_team': str,
                            'away_team': str,
                            'odds': {
                                'is_win': {'home': float, 'away': float},
                                'total_points': {'over': float, 'under': float, 'line': float},
                                'home_points': {'over': float, 'under': float, 'line': float},
                                'away_points': {'over': float, 'under': float, 'line': float},
                                'ht_total': {'over': float, 'under': float, 'line': float},
                                'ht_handicap': {'home': float, 'away': float, 'line': float}
                            }
                        }
                    ]
                }
            }
        """
        endpoint = f"en/sports/sr:sport:{self.sport_id}/schedules/{date}/sport_event_markets.json"
        url = f"{self.prematch_url}/{endpoint}"
        params = {'api_key': self.api_key}
        
        try:
            logger.info(f"Obteniendo team odds para {date}")
            data = self._make_request(url, params)
            
            if not data:
                return {'success': False, 'error': 'No data returned'}
            
            # Procesar respuesta
            events_data = data.get('sport_schedule_sport_event_markets', [])
            processed_events = []
            
            for event in events_data:
                sport_event = event.get('sport_event', {})
                markets = event.get('markets', [])
                
                # Info del partido
                competitors = sport_event.get('competitors', [])
                home_team = next((c['name'] for c in competitors if c.get('qualifier') == 'home'), 'Unknown')
                away_team = next((c['name'] for c in competitors if c.get('qualifier') == 'away'), 'Unknown')
                
                # Extraer odds
                odds = {}
                for market in markets:
                    market_id = market.get('id')
                    market_name = market.get('name', '')
                    
                    # Moneyline (is_win)
                    if market_id == self.MARKETS['is_win']:
                        home_odds = self._get_outcome_odds(market, 'home')
                        away_odds = self._get_outcome_odds(market, 'away')
                        if home_odds and away_odds:
                            odds['is_win'] = {'home': home_odds, 'away': away_odds}
                    
                    # Total Points
                    elif market_id == self.MARKETS['total_points']:
                        line_data = self._extract_over_under(market)
                        if line_data:
                            odds['total_points'] = line_data
                    
                    # Home Points
                    elif market_id == self.MARKETS['home_points']:
                        line_data = self._extract_over_under(market)
                        if line_data:
                            odds['home_points'] = line_data
                    
                    # Away Points
                    elif market_id == self.MARKETS['away_points']:
                        line_data = self._extract_over_under(market)
                        if line_data:
                            odds['away_points'] = line_data
                    
                    # Halftime Total
                    elif market_id == self.MARKETS['ht_total']:
                        line_data = self._extract_over_under(market)
                        if line_data:
                            odds['ht_total'] = line_data
                    
                    # Halftime Handicap
                    elif market_id == self.MARKETS['ht_handicap']:
                        home_odds = self._get_outcome_odds(market, 'home')
                        away_odds = self._get_outcome_odds(market, 'away')
                        line = self._get_handicap_line(market)
                        if home_odds and away_odds:
                            odds['ht_handicap'] = {'home': home_odds, 'away': away_odds, 'line': line}
                
                if odds:  # Solo agregar si tiene odds
                    processed_events.append({
                        'sport_event_id': sport_event.get('id'),
                        'home_team': home_team,
                        'away_team': away_team,
                        'start_time': sport_event.get('start_time'),
                        'odds': odds
                    })
            
            logger.info(f"Team odds procesados: {len(processed_events)} eventos")
            
            return {
                'success': True,
                'data': {
                    'date': date,
                    'total_events': len(processed_events),
                    'events': processed_events
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo team odds: {e}")
            return {'success': False, 'error': str(e)}
    
    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================
    
    def _extract_best_line(self, market: Dict) -> Optional[Dict]:
        """
        Extrae la mejor línea (over/under) de un mercado de player props.
        
        Returns:
            {'over': str, 'under': str, 'line': float, 'book': str} o None
        """
        books = market.get('books', [])
        if not books:
            return None
        
        # Tomar el primer book disponible (generalmente es el mejor)
        first_book = books[0]
        outcomes = first_book.get('outcomes', [])
        
        if len(outcomes) < 2:
            return None
        
        over = next((o for o in outcomes if o.get('type') == 'over'), None)
        under = next((o for o in outcomes if o.get('type') == 'under'), None)
        
        if not over or not under:
            return None
        
        try:
            # USAR ODDS DECIMAL - Mejor para cálculos matemáticos y Kelly Criterion
            over_odds = float(over.get('odds_decimal', 0))
            under_odds = float(under.get('odds_decimal', 0))
            
            if over_odds == 0 or under_odds == 0:
                return None
            
            return {
                'over': over_odds,  # Float para cálculos directos
                'under': under_odds,  # Float para cálculos directos
                'line': float(over.get('total', 0) or under.get('total', 0)),
                'book': first_book.get('name', 'Unknown'),
                # Agregar probabilidades implícitas para comparación directa
                'over_prob': round(1 / over_odds, 4),
                'under_prob': round(1 / under_odds, 4)
            }
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def _extract_over_under(self, market: Dict) -> Optional[Dict]:
        """
        Extrae línea over/under de mercados de team odds.
        
        Returns:
            {'over': str, 'under': str, 'line': float, 'book': str} o None
        """
        books = market.get('books', [])
        if not books:
            return None
        
        first_book = books[0]
        outcomes = first_book.get('outcomes', [])
        
        over = next((o for o in outcomes if 'over' in o.get('type', '').lower()), None)
        under = next((o for o in outcomes if 'under' in o.get('type', '').lower()), None)
        
        if not over or not under:
            return None
        
        try:
            # USAR ODDS DECIMAL - Mejor para cálculos matemáticos
            over_odds = float(over.get('odds_decimal', 0))
            under_odds = float(under.get('odds_decimal', 0))
            
            if over_odds == 0 or under_odds == 0:
                return None
            
            return {
                'over': over_odds,
                'under': under_odds,
                'line': float(over.get('total', 0) or under.get('total', 0)),
                'book': first_book.get('name', 'Unknown'),
                'over_prob': round(1 / over_odds, 4),
                'under_prob': round(1 / under_odds, 4)
            }
        except (ValueError, TypeError, ZeroDivisionError):
            return None
    
    def _get_outcome_odds(self, market: Dict, qualifier: str) -> Optional[Dict]:
        """
        Obtiene odds para un outcome específico (home/away).
        
        Args:
            market: Datos del mercado
            qualifier: 'home' o 'away'
            
        Returns:
            Dict con odds decimal y probabilidad implícita, o None
        """
        books = market.get('books', [])
        if not books:
            return None
        
        first_book = books[0]
        outcomes = first_book.get('outcomes', [])
        
        outcome = next((o for o in outcomes if o.get('type') == qualifier), None)
        
        if outcome:
            try:
                # USAR ODDS DECIMAL
                odds = float(outcome.get('odds_decimal', 0))
                if odds == 0:
                    return None
                
                return {
                    'odds': odds,
                    'prob': round(1 / odds, 4)
                }
            except (ValueError, TypeError, ZeroDivisionError):
                return None
        return None
    
    def _get_handicap_line(self, market: Dict) -> Optional[float]:
        """Obtiene la línea de handicap."""
        books = market.get('books', [])
        if not books:
            return None
        
        first_book = books[0]
        outcomes = first_book.get('outcomes', [])
        
        if outcomes:
            try:
                return float(outcomes[0].get('spread', 0))
            except (ValueError, TypeError):
                return None
        return None
    
    # =========================================================================
    # MÉTODOS ADICIONALES
    # =========================================================================
    
    def get_schedule(self, date: str) -> Dict[str, Any]:
        """
        Obtiene el calendario de partidos para una fecha.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            
        Returns:
            Lista de partidos programados
        """
        endpoint = f"en/sports/sr:sport:{self.sport_id}/schedules/{date}/schedules.json"
        url = f"{self.player_props_url}/{endpoint}"
        params = {'api_key': self.api_key}
        
        try:
            logger.info(f"Obteniendo schedule para {date}")
            data = self._make_request(url, params)
            
            schedules = data.get('schedules', [])
            sport_events = []
            
            for schedule in schedules:
                events = schedule.get('sport_events', [])
                sport_events.extend(events)
            
            logger.info(f"Schedule obtenido: {len(sport_events)} eventos")
            
            return {
                'success': True,
                'data': {
                    'date': date,
                    'total_events': len(sport_events),
                    'sport_events': sport_events
                }
            }
            
        except Exception as e:
            logger.error(f"Error obteniendo schedule: {e}")
            return {'success': False, 'error': str(e)}
    
    def test_connection(self) -> Dict[str, Any]:
        """Prueba la conexión con la API."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            result = self.get_schedule(today)
            
            if result.get('success'):
                return {
                    'success': True,
                    'status': 'connected',
                    'message': 'Conexión exitosa con Sportradar API'
                }
            else:
                return {
                    'success': False,
                    'status': 'error',
                    'message': result.get('error', 'Unknown error')
                }
        except Exception as e:
            return {
                'success': False,
                'status': 'error',
                'error': str(e)
            }
    
    def clear_cache(self):
        """Limpia el caché."""
        try:
            self._cache.clear()
            logger.info("Cache limpiado")
        except Exception as e:
            logger.error(f"Error limpiando cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché."""
        try:
            return self._cache.get_stats()
        except Exception as e:
            logger.error(f"Error obteniendo stats de cache: {e}")
            return {}
