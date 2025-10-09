"""
Sportradar API Client para NBA
=============================

Cliente completo para la API de Sportradar que obtiene:
- Cuotas de casas de apuestas para partidos NBA
- Líneas de player props (PTS, AST, TRB, 3P)
- Líneas de team props (Winner, Total Points, Teams Points)
- Información de partidos y equipos

Documentación API: https://developer.sportradar.com/docs/read/basketball/NBA_v8
"""

import time
import json
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
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
    DataValidationError,
    create_http_error
)

logger = logging.getLogger(__name__)


class SportradarAPI:
    """
    Cliente avanzado para la API de Sportradar NBA.
    
    Proporciona acceso completo a:
    - Schedules y partidos
    - Cuotas de casas de apuestas
    - Player props y líneas
    - Datos de equipos y jugadores
    """
    
    def __init__(self, api_key: Optional[str] = None, config_override: Optional[Dict] = None):
        """
        Inicializa el cliente de Sportradar.
        
        Args:
            api_key: API key de Sportradar. Si no se proporciona, se obtiene de config
            config_override: Configuración personalizada para sobrescribir defaults
        """
        # Configuración
        self.config = get_config()
        
        # API Key - usar la misma para ambas APIs
        self.api_key = api_key or self.config.config['player_props_api']['api_key']
        if not self.api_key:
            raise AuthenticationError("API key de Sportradar requerida")
        
        # URLs base según documentación oficial
        self.player_props_url = self.config.get_api_url('player_props')
        self.odds_base_url = self.config.get_api_url('prematch')
        self.base_url = self.odds_base_url  # Para compatibilidad
        
        # Configuración de red - usar de player_props_api como base
        player_props_config = self.config.config['player_props_api']
        self.timeout = player_props_config.get('timeout', 30)
        self.retry_attempts = player_props_config.get('retry_attempts', 3)
        self.retry_delay = player_props_config.get('retry_delay', 1)
        
        # Sport ID fijo para NBA
        self.sport_id = 2  # sr:sport:2 para NBA
        
        # Rate limiting
        self.rate_limit_calls = player_props_config.get('rate_limit_calls', 5)
        self.rate_limit_period = player_props_config.get('rate_limit_period', 1)
        self.last_calls = []
        
        # Configurar sesión HTTP con retry strategy
        self.session = self._setup_session()
        
        # Cache optimizado
        from .config.optimized_cache import OptimizedCache
        self._cache = OptimizedCache(self.config.get_data_config())
        
        # Inicializar contadores de cache
        self.cache_hits = 0
        self.cache_misses = 0
        
        logger.info(f"SportradarAPI inicializada | Base URL: {self.base_url}")
    
    def _setup_session(self) -> requests.Session:
        """Configura sesión HTTP con estrategia de retry."""
        session = requests.Session()
        
        # Configurar retry strategy
        retry_strategy = Retry(
            total=self.retry_attempts,
            backoff_factor=self.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        # Configurar adapter
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Headers por defecto
        session.headers.update({
            'User-Agent': 'NBA-Prediction-System/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        return session
    
    def _check_rate_limit(self):
        """Verifica y aplica rate limiting."""
        now = time.time()
        
        # Limpiar llamadas antiguas
        self.last_calls = [call_time for call_time in self.last_calls 
                          if now - call_time < self.rate_limit_period]
        
        # Verificar límite
        if len(self.last_calls) >= self.rate_limit_calls:
            sleep_time = self.rate_limit_period - (now - self.last_calls[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit alcanzado. Esperando {sleep_time:.1f}s")
                time.sleep(sleep_time)
        
        # Registrar llamada actual
        self.last_calls.append(now)
    
    def _get_cached_data(self, endpoint: str, params: Dict, ttl: int = None) -> Optional[Dict[str, Any]]:
        """Obtiene datos del cache optimizado."""
        return self._cache.get(endpoint, params, ttl)
    
    def _set_cached_data(self, endpoint: str, params: Dict, data: Dict[str, Any], ttl: int = None) -> bool:
        """Almacena datos en el cache optimizado."""
        return self._cache.set(endpoint, params, data, ttl)
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None,
                     use_cache: bool = True, use_odds_api: bool = False, expect_xml: bool = False, 
                     cache_ttl: int = None) -> Dict[str, Any]:
        """
        Realiza petición HTTP a la API de Sportradar.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de query
            use_cache: Si usar cache
            use_odds_api: Si usar la API de Odds en lugar de Sports
            cache_ttl: TTL personalizado para cache
            
        Returns:
            Respuesta JSON de la API
            
        Raises:
            SportradarAPIError: Error en la API
        """
        params = params or {}
        params['api_key'] = self.api_key
        
        # Verificar cache optimizado
        if use_cache:
            cached_data = self._get_cached_data(endpoint, params, cache_ttl)
            if cached_data is not None:
                logger.debug(f"Cache hit para {endpoint}")
                return cached_data
        
        # Rate limiting
        self._check_rate_limit()
        
        # Construir URL - usar base URL apropiada
        base_url = self.odds_base_url if use_odds_api else self.base_url
        # Asegurar que base_url termine con / y endpoint no empiece con /
        if not base_url.endswith('/'):
            base_url += '/'
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]
        url = base_url + endpoint
        
        try:
            logger.debug(f"GET {url} | Params: {len(params)} items")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            # Manejo de errores HTTP
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            # Parsear respuesta
            try:
                if expect_xml:
                    # Parsear XML y convertir a diccionario
                    data = self._parse_xml_response(response.text)
                else:
                    data = response.json()
            except (json.JSONDecodeError, ET.ParseError) as e:
                raise SportradarAPIError(
                    f"Respuesta inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Guardar en cache optimizado
            if use_cache:
                self._set_cached_data(endpoint, params, data, cache_ttl)
            
            logger.debug(f"Respuesta exitosa de {endpoint}")
            return data
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def _handle_http_error(self, response: requests.Response, endpoint: str):
        """Maneja errores HTTP específicos."""
        status_code = response.status_code
        
        # Intentar obtener mensaje de error de la respuesta
        try:
            error_data = response.json()
            message = error_data.get('message', f"Error HTTP {status_code}")
        except:
            message = f"Error HTTP {status_code}"
        
        # Rate limiting específico
        if status_code == 429:
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    retry_after = int(retry_after)
                except ValueError:
                    retry_after = None
            raise RateLimitError(retry_after)
        
        # Crear excepción apropiada
        raise create_http_error(
            status_code, 
            message, 
            response_data=error_data if 'error_data' in locals() else None
        )
    
    # === MÉTODOS PRINCIPALES DE LA API ===
    
    def get_daily_odds_schedule(self, date: str, sport_id: int = None) -> Dict[str, Any]:
        """
        Obtiene el schedule con odds disponibles para una fecha específica usando Player Props v2 API.
        
        CORRECCIÓN: Usar la URL correcta de Player Props v2 que SÍ funciona.
        
        Args:
            date: Fecha en formato YYYY-MM-DD
            sport_id: ID del deporte (default: basketball from config)
            
        Returns:
            Schedule con sport events que tienen odds
        """
        if sport_id is None:
            sport_id = self.sport_id
            
        # Usar Player Props v2 API que SÍ funciona
        sport_id_formatted = f"sr:sport:{sport_id}"
        endpoint = f"en/sports/{sport_id_formatted}/schedules/{date}/schedules"
        
        # Construir URL usando Player Props v2 API
        player_props_url = self.player_props_url
        if not player_props_url.endswith('/'):
            player_props_url += '/'
        url = player_props_url + endpoint
        params = {'api_key': self.api_key}
        
        # Rate limiting
        self._check_rate_limit()
        
        try:
            logger.debug(f"GET {url} | Daily Schedule Player Props v2 (URL corregida)")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Extraer sport_events de la estructura de schedules
            sport_events = []
            if 'schedules' in data:
                for schedule in data['schedules']:
                    if 'sport_event' in schedule:
                        sport_events.append(schedule['sport_event'])
            
            logger.info(f"✅ Schedule obtenido para {date}: {len(sport_events)} eventos")
            return {
                'sport_events': sport_events,
                'date': date,
                'method': 'player_props_v2_schedules',
                'success': True,
                'raw_data': data
            }
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def get_schedule(self, date: Optional[str] = None, season: str = "2024") -> Dict[str, Any]:
        """
        Obtiene el schedule de partidos NBA.
                
        Args:
            date: Fecha específica (YYYY-MM-DD). Si es None, obtiene schedule completo
            season: Temporada (ej: "2024" para 2024-25)
            
        Returns:
            Datos del schedule
        """
        if date:
            # Para fecha específica, usar get_daily_odds_schedule que ya está corregido
            logger.debug(f"Schedule para fecha específica: {date}")
            return self.get_daily_odds_schedule(date)
        else:
            # Para schedule completo, usar Competition Schedules con Player Props v2 API
            logger.debug(f"Schedule completo para temporada: {season}")
            
            try:
                # Usar NBA competition con Player Props v2 API
                nba_competition_id = 'sr:competition:132'
                competition_schedule = self.get_competition_schedules(nba_competition_id)
                
                if competition_schedule and 'schedules' in competition_schedule:
                    logger.info(f"✅ Schedule completo obtenido: {len(competition_schedule['schedules'])} eventos")
                    return {
                        'schedules': competition_schedule['schedules'],
                        'season': season,
                        'method': 'player_props_v2_competition',
                        'success': True
                    }
                
            except Exception as e:
                logger.warning(f"Error obteniendo schedule completo: {e}")
                
            # Fallback: devolver estructura vacía pero válida
            logger.warning(f"No se pudo obtener schedule para temporada {season}")
            return {
                'schedules': [],
                'season': season,
                'method': 'fallback_empty',
                'success': False,
                'error': 'No se pudo obtener schedule'
            }
    
    def get_odds(self, sport_event_id: str, format: str = "american") -> Dict[str, Any]:
        """
        Obtiene cuotas para un partido específico usando Odds Comparison API.
        
        SOLUCIÓN PARA ERROR 401: Usar endpoints alternativos que SÍ funcionan.
        En lugar de obtener odds por sport_event_id específico (que requiere permisos especiales),
        usamos endpoints por fecha/competición que están disponibles.
        
        Args:
            sport_event_id: ID del sport event de Sportradar (ej: sr:sport_event:12345)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Datos de cuotas del partido
        """
        logger.warning(f"get_odds para sport_event_id específico no disponible con cuenta trial")
        logger.info("Usando método alternativo: obtener odds por fecha/competición")
        
        try:
            # Estrategia alternativa: obtener odds por fecha
            today = datetime.now().strftime("%Y-%m-%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Intentar obtener odds por fecha usando Schedule Markets
            for date in [today, tomorrow]:
                try:
                    odds_data = self.get_schedule_markets(sport_id=2, date=date)
                    
                    if odds_data and 'sport_events' in odds_data:
                        # Buscar el sport_event_id específico
                        for event in odds_data['sport_events']:
                            if event.get('id') == sport_event_id:
                                logger.info(f"✅ Odds encontradas para {sport_event_id} en fecha {date}")
                                return {
                                    'success': True,
                                    'sport_event_id': sport_event_id,
                                    'odds': event.get('markets', {}),
                                    'method': 'by_date_filtered',
                                    'date': date
                                }
                except Exception as e:
                    logger.debug(f"Error obteniendo odds por fecha {date}: {e}")
                    continue
            
            # Si no encontramos el evento específico, devolver estructura vacía pero válida
            logger.warning(f"No se encontraron odds para sport_event_id {sport_event_id}")
            return {
                'success': False,
                'sport_event_id': sport_event_id,
                'odds': {},
                'method': 'not_found',
                'error': 'Sport event not found in available dates'
            }
            
        except Exception as e:
            logger.error(f"Error en método alternativo de odds: {e}")
            return {
                'success': False,
                'sport_event_id': sport_event_id,
                'odds': {},
                'error': str(e)
            }
    
    def get_player_props(self, sport_event_id: str, format: str = "us") -> Dict[str, Any]:
        """
        Obtiene player props para un sport_event_id específico usando Player Props v2 API.
        
        Este método usa el endpoint exacto que SÍ funciona:
        /en/sport_events/{sport_event_id}/players_props
        
        Args:
            sport_event_id: ID del sport event de Sportradar
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Datos de player props estructurados para comparación con modelo
        """
        cache_key = f"player_props_direct_{sport_event_id}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key, {})
        if cached_data:
            self.cache_hits += 1
            logger.debug(f"Cache hit para props de {sport_event_id}")
            return cached_data
        
        self.cache_misses += 1
        
        try:
            # Formatear sport_event_id para URL
            sport_event_id_formatted = sport_event_id.replace(':', '%3A')
            
            # Usar endpoint exacto que funciona
            endpoint = f"en/sport_events/{sport_event_id_formatted}/players_props"
            
            # Construir URL usando Player Props v2 API
            player_props_url = self.player_props_url
            url = urljoin(player_props_url, endpoint)
            params = {'api_key': self.api_key}
            
            # Rate limiting
            self._check_rate_limit()
            
            logger.debug(f"GET {url} | Player Props Direct")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Procesar datos para estructura optimizada para comparación con modelo
            processed_data = self._process_player_props_for_model_comparison(data, sport_event_id)
            
            # Guardar en cache
            self._set_cached_data(cache_key, {}, processed_data)
            
            logger.info(f"✅ Player props obtenidas directamente para {sport_event_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error obteniendo props directas para {sport_event_id}: {e}")
            return {
                'success': False,
                'sport_event_id': sport_event_id,
                'props': {},
                'error': str(e)
            }
    
    def get_player_props_by_date(self, date: str, sport_id: int = None) -> Dict[str, Any]:
        """
        Obtiene player props para una fecha específica usando Player Props v2 API.
                
        Args:
            date: Fecha en formato YYYY-MM-DD
            sport_id: ID del deporte (default: basketball = 2)
            
        Returns:
            Datos de player props para la fecha
        """
        if sport_id is None:
            sport_id = self.sport_id
            
        # Usar endpoint de Player Props v2 API que SÍ funciona
        sport_id_formatted = f"sr:sport:{sport_id}"
        endpoint = f"en/sports/{sport_id_formatted}/schedules/{date}/players_props"
        
        # Construir URL usando Player Props v2 API
        player_props_url = self.player_props_url
        if not player_props_url.endswith('/'):
            player_props_url += '/'
        url = player_props_url + endpoint
        params = {'api_key': self.api_key}
        
        # Rate limiting
        self._check_rate_limit()
        
        try:
            logger.debug(f"GET {url} | Player Props by Date")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Procesar datos para extraer eventos con props
            events_with_props = []
            
            # La estructura puede variar, intentar diferentes formatos
            if 'sport_events' in data:
                for event in data['sport_events']:
                    if 'players_props' in event:
                        events_with_props.append({
                            'sport_event_id': event.get('id'),
                            'props': event.get('players_props', {}),
                            'event_info': {
                                'start_time': event.get('start_time'),
                                'competitors': event.get('competitors', [])
                            }
                        })
            
            logger.info(f"✅ Player props obtenidas para {date}: {len(events_with_props)} eventos")
            return {
                'success': True,
                'date': date,
                'events': events_with_props,
                'method': 'player_props_v2_by_date',
                'raw_data': data
            }
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def get_player_props_by_competition(self, competition_id: str = 'sr:competition:486') -> Dict[str, Any]:
        """
        Obtiene player props para una competición específica usando Player Props v2 API.
        
        Este método SÍ funciona con nuestra cuenta trial y es ideal para WNBA activa.
        
        Args:
            competition_id: ID de la competición (default: WNBA que está activa)
            
        Returns:
            Datos de player props para la competición
        """
        # Usar endpoint de Player Props v2 API que SÍ funciona
        endpoint = f"en/competitions/{competition_id}/players_props"
        
        # Construir URL usando Player Props v2 API
        player_props_url = self.player_props_url
        if not player_props_url.endswith('/'):
            player_props_url += '/'
        url = player_props_url + endpoint
        params = {'api_key': self.api_key}
        
        # Rate limiting
        self._check_rate_limit()
        
        try:
            logger.debug(f"GET {url} | Player Props by Competition")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Procesar datos para extraer eventos con props
            events_with_props = []
            
            # La estructura puede variar, intentar diferentes formatos
            if 'sport_events' in data:
                for event in data['sport_events']:
                    if 'players_props' in event or 'markets' in event:
                        events_with_props.append({
                            'sport_event_id': event.get('id'),
                            'props': event.get('players_props', event.get('markets', {})),
                            'event_info': {
                                'start_time': event.get('start_time'),
                                'competitors': event.get('competitors', []),
                                'status': event.get('status', 'unknown')
                            }
                        })
            
            logger.info(f"✅ Player props obtenidas para competición {competition_id}: {len(events_with_props)} eventos")
            return {
                'success': True,
                'competition_id': competition_id,
                'events': events_with_props,
                'method': 'player_props_v2_by_competition',
                'raw_data': data
            }
            
        except requests.exceptions.Timeout:
            raise NetworkError(f"Timeout en petición a {endpoint}", url, self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(f"Error de conexión: {e}", url)
        except requests.exceptions.RequestException as e:
            raise SportradarAPIError(f"Error en petición: {e}", endpoint=endpoint)
    
    def get_prematch_odds(self, sport_event_id: str, format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds de prematch (1x2, totals, spreads) usando Odds Comparison v2 - Prematch API.
        
        Este método obtiene los game markets (no player props) desde la Prematch API:
        - Market ID 1: 1x2/moneyline (is_win)
        - Market ID 225: total_incl_overtime (total_points)
        - Market ID 227: home_total_incl_overtime (teams_points home)
        - Market ID 228: away_total_incl_overtime (teams_points away)
        
        Args:
            sport_event_id: ID del sport event (ej: sr:sport_event:12345)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Game markets del partido (is_win, total_points, teams_points)
        """
        cache_key = f"prematch_odds_{sport_event_id}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key, {})
        if cached_data:
            self.cache_hits += 1
            logger.debug(f"Cache hit para prematch odds de {sport_event_id}")
            return cached_data
        
        self.cache_misses += 1
        
        try:
            # Formatear sport_event_id para URL
            sport_event_id_formatted = sport_event_id.replace(':', '%3A')
            
            # Endpoint para sport event markets (Prematch API)
            endpoint = f"en/sport_events/{sport_event_id_formatted}/sport_event_markets"
            
            # Construir URL usando Prematch API (Odds Comparison v2)
            prematch_url = self.odds_base_url
            url = urljoin(prematch_url, endpoint)
            params = {'api_key': self.api_key}
            
            # Rate limiting
            self._check_rate_limit()
            
            logger.debug(f"GET {url} | Prematch Odds")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # Procesar datos para estructura optimizada para comparación con modelo
            processed_data = self._process_prematch_odds_for_model_comparison(data, sport_event_id)
            
            # Guardar en cache
            self._set_cached_data(cache_key, {}, processed_data)
            
            logger.info(f"✅ Prematch odds obtenidas para {sport_event_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error obteniendo prematch odds para {sport_event_id}: {e}")
            return {
                'success': False,
                'sport_event_id': sport_event_id,
                'markets': [],
                'error': str(e)
            }
    
    def get_prematch_odds_by_competition(self, competition_id: str = 'sr:competition:132', 
                                       offset: int = 0, limit: int = 50, start: int = 0) -> Dict[str, Any]:
        """
        Obtiene odds de prematch para una competición específica usando Odds Comparison API v2.
        
        Args:
            competition_id: ID de la competición (default: NBA)
            offset: Desplazamiento para paginación
            limit: Límite de resultados
            start: Inicio para paginación
            
        Returns:
            Diccionario con odds de prematch de la competición
        """
        cache_key = f"prematch_competition_{competition_id}_{offset}_{limit}_{start}"
        
        # Verificar cache
        cached_data = self._get_cached_data(cache_key, {})
        if cached_data:
            self.cache_hits += 1
            logger.debug(f"Cache hit para prematch odds de competición {competition_id}")
            return cached_data
        
        self.cache_misses += 1
        
        try:
            # Formatear competition_id para URL
            competition_id_formatted = competition_id.replace(':', '%3A')
            
            # Endpoint para competition sport event markets
            endpoint = f"en/competitions/{competition_id_formatted}/sport_event_markets"
            
            # Construir URL usando Prematch API (Odds Comparison v2)
            prematch_url = self.odds_base_url
            url = urljoin(prematch_url, endpoint)
            params = {
                'api_key': self.api_key,
                'offset': offset,
                'limit': limit,
                'start': start
            }
            
            # Rate limiting
            self._check_rate_limit()
            
            logger.debug(f"GET {url} | Prematch Competition Odds")
            
            response = self.session.get(
                url, 
                params=params, 
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                self._handle_http_error(response, endpoint)
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise SportradarAPIError(
                    f"Respuesta JSON inválida: {e}",
                    response.status_code,
                    endpoint=endpoint
                )
            
            # CORRECCIÓN: Procesar datos con la estructura real
            # La respuesta tiene 'competition_sport_event_markets', no 'sport_event_markets'
            sport_event_markets = data.get('competition_sport_event_markets', [])
            
            processed_data = {
                'success': True,
                'sport_event_markets': sport_event_markets,  # Estructura esperada por el sistema
                'competition_id': competition_id,
                'offset': offset,
                'limit': limit,
                'total_events': len(sport_event_markets),
                'api_source': 'prematch_odds_v2',
                'raw_data': data  # Mantener datos originales para debug
            }
            
            # Guardar en cache
            self._set_cached_data(cache_key, {}, processed_data)
            
            logger.info(f"✅ Prematch odds obtenidas para competición {competition_id}")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error obteniendo prematch odds para competición {competition_id}: {e}")
            return {
                'success': False,
                'competition_id': competition_id,
                'error': str(e)
            }
    
    def get_teams(self) -> Dict[str, Any]:
        """
        Obtiene información de todos los equipos NBA.
        
        Returns:
            Datos de equipos
        """
        endpoint = "league/hierarchy.json"
        return self._make_request(endpoint)
    
    def get_game_info(self, sport_event_id: str) -> Dict[str, Any]:
        """
        Obtiene información detallada de un partido usando Sports API.
        
        Args:
            sport_event_id: ID del sport event
            
        Returns:
            Información del partido
        """
        # Usar la API de Sports para información del partido
        endpoint = f"sport_events/{sport_event_id}/summary.json"
        return self._make_request(endpoint, use_odds_api=False)
    
    def get_bookmakers(self) -> Dict[str, Any]:
        """
        Obtiene lista de casas de apuestas disponibles.
        
        Returns:
            Lista de bookmakers
        """
        endpoint = "en/books"
        # Bookmakers cambian poco, cache por 12 horas
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False, cache_ttl=43200)
    
    def get_sports(self) -> Dict[str, Any]:
        """
        Obtiene lista de deportes disponibles.
        
        Returns:
            Lista de deportes
        """
        endpoint = "en/sports"
        # Deportes cambian raramente, cache por 24 horas
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False, cache_ttl=86400)
    
    def get_sport_competitions(self, sport_id: int = None) -> Dict[str, Any]:
        """
        Obtiene lista de competiciones para un deporte específico.
        
        Args:
            sport_id: ID del deporte (default: basketball = 2)
            
        Returns:
            Lista de competiciones
        """
        if sport_id is None:
            sport_id = 2  # Basketball
        
        # Usar formato correcto sr:sport:X
        sport_id_formatted = f"sr:sport:{sport_id}"
        endpoint = self.config.get_endpoint('sport_competitions', sport_id=sport_id_formatted)
        
        return self._make_request(endpoint, use_odds_api=True, expect_xml=False)
    
    def get_competition_schedules(self, competition_id: str, offset: int = 0, 
                                 limit: int = 50, start: int = 0) -> Dict[str, Any]:
        """
        Obtiene schedules para una competición específica.
        
        Args:
            competition_id: ID de la competición (ej: sr:competition:132 para NBA)
            offset: Offset para paginación
            limit: Límite de resultados
            start: Inicio para paginación
            
        Returns:
            Schedules de la competición
        """
        endpoint = self.config.get_endpoint('competition_schedules', competition_id=competition_id)
        params = {
            'offset': offset,
            'limit': limit,
            'start': start
        }
        return self._make_request(endpoint, params=params, use_odds_api=True, expect_xml=False)
    
    def get_schedule_markets(self, sport_id: int = None, date: str = None, 
                            limit: int = 5) -> Dict[str, Any]:
        """
        Obtiene markets para una fecha específica usando el formato correcto.
        
        Args:
            sport_id: ID del deporte (default: basketball = 2)
            date: Fecha en formato YYYY-MM-DD
            limit: Límite de resultados
            
        Returns:
            Markets para la fecha especificada
        """
        if sport_id is None:
            sport_id = 2  # Basketball
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Usar formato correcto sr:sport:X y endpoint de prematch API
        sport_id_formatted = f"sr:sport:{sport_id}"
        endpoint = f"en/sports/{sport_id_formatted}/schedules/{date}/sport_event_markets"
        
        params = {'limit': limit}
        
        return self._make_request(endpoint, params=params, use_odds_api=True, expect_xml=False)
    
    # === MÉTODOS DE ANÁLISIS AVANZADO ===
    
    def get_games_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Obtiene partidos en un rango de fechas.
        
        Args:
            start_date: Fecha inicio (YYYY-MM-DD)
            end_date: Fecha fin (YYYY-MM-DD)
            
        Returns:
            Lista de partidos
        """
        games = []
        
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise DataValidationError(f"Formato de fecha inválido: {e}")
        
        current = start
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            try:
                schedule = self.get_schedule(date_str)
                if 'games' in schedule:
                    games.extend(schedule['games'])
            except SportradarAPIError as e:
                logger.warning(f"Error obteniendo schedule para {date_str}: {e}")
            
            current += timedelta(days=1)
        
        return games
    
    def get_odds_for_multiple_games(self, sport_event_ids: List[str], 
                                   include_props: bool = True, 
                                   format: str = "us") -> Dict[str, Dict[str, Any]]:
        """
        Obtiene cuotas para múltiples partidos.
        
        Args:
            sport_event_ids: Lista de IDs de sport events
            include_props: Si incluir player props
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario con cuotas por sport_event_id
        """
        results = {}
        
        for sport_event_id in sport_event_ids:
            try:
                # Obtener cuotas principales
                odds = self.get_odds(sport_event_id, format)
                results[sport_event_id] = {'odds': odds}
                
                # Obtener player props si se solicita
                if include_props:
                    try:
                        props = self.get_player_props(sport_event_id, format)
                        results[sport_event_id]['props'] = props
                    except SportradarAPIError as e:
                        logger.warning(f"Error obteniendo props para {sport_event_id}: {e}")
                        results[sport_event_id]['props'] = None
                
            except SportradarAPIError as e:
                logger.error(f"Error obteniendo datos para {sport_event_id}: {e}")
                results[sport_event_id] = {'error': str(e)}
        
        return results
    
    def search_games_by_teams(self, home_team: str, away_team: str,
                             date_range_days: int = 7) -> List[Dict[str, Any]]:
        """
        Busca partidos entre equipos específicos.
        
        Args:
            home_team: Nombre o código del equipo local
            away_team: Nombre o código del equipo visitante
            date_range_days: Días hacia adelante para buscar
            
        Returns:
            Lista de partidos encontrados
        """
        end_date = datetime.now() + timedelta(days=date_range_days)
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        games = self.get_games_by_date_range(start_date, end_date_str)
        
        # Filtrar por equipos
        matching_games = []
        for game in games:
            home = game.get('home', {}).get('name', '').lower()
            away = game.get('away', {}).get('name', '').lower()
            
            if (home_team.lower() in home and away_team.lower() in away):
                matching_games.append(game)
        
        return matching_games
    
    def get_market_overview(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene resumen del mercado para una fecha.
        
        Args:
            date: Fecha (YYYY-MM-DD). Si es None, usa fecha actual
            
        Returns:
            Resumen del mercado
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Obtener schedule del día
        schedule = self.get_schedule(date)
        
        if 'games' not in schedule:
            return {'date': date, 'games': [], 'total_games': 0}
        
        # Obtener cuotas para todos los partidos
        games_with_odds = []
        for game in schedule['games']:
            game_id = game.get('id')
            if game_id:
                try:
                    odds = self.get_odds(game_id)
                    game_data = {
                        'game': game,
                        'odds': odds,
                        'id': game_id
                    }
                    games_with_odds.append(game_data)
                except SportradarAPIError as e:
                    logger.warning(f"No se pudieron obtener odds para {game_id}: {e}")
        
        return {
            'date': date,
            'games': games_with_odds,
            'total_games': len(schedule['games']),
            'games_with_odds': len(games_with_odds)
        }
    
    def get_nba_odds_today(self, format: str = "us", include_props: bool = True) -> Dict[str, Any]:
        """
        Obtiene odds para todos los partidos NBA de hoy.
        
        Args:
            format: Formato de odds ("us", "eu", "uk")
            include_props: Si incluir player props
            
        Returns:
            Diccionario con odds de partidos NBA del día
        """
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            # Obtener schedule del día para NBA (sport_id = 2)
            schedule = self.get_daily_odds_schedule(today, sport_id=2)
            
            if 'sport_events' not in schedule:
                return {
                    'date': today,
                    'sport_events': [],
                    'odds_data': {},
                    'message': 'No hay partidos NBA programados para hoy'
                }
            
            # Extraer sport_event_ids
            sport_event_ids = [event['id'] for event in schedule['sport_events']]
            
            # Obtener odds para todos los partidos
            odds_data = self.get_odds_for_multiple_games(
                sport_event_ids, 
                include_props=include_props, 
                format=format
            )
            
            return {
                'date': today,
                'sport_events': schedule['sport_events'],
                'odds_data': odds_data,
                'total_games': len(sport_event_ids),
                'message': f'Odds obtenidas para {len(sport_event_ids)} partidos NBA'
            }
            
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo odds NBA para {today}: {e}")
            return {
                'date': today,
                'error': str(e),
                'sport_events': [],
                'odds_data': {}
            }
    
    # === MÉTODOS ESPECÍFICOS PARA TARGETS DEL MODELO ===
    
    def get_specific_nba_odds(self, sport_event_id: str, targets: List[str] = None, 
                             format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds específicas para los targets del modelo de predicción NBA.
        
        Args:
            sport_event_id: ID del sport event
            targets: Lista de targets específicos a obtener:
                   - 'moneyline': Victoria/derrota (is_win)
                   - 'totals': Total de puntos del partido (total_points)
                   - 'spreads': Spread de puntos
                   - 'team_totals': Puntos por equipo (teams_points)
                   - 'player_points': Puntos de jugadores (PTS)
                   - 'player_assists': Asistencias de jugadores (AST)
                   - 'player_rebounds': Rebotes de jugadores (TRB)
                   - 'player_threes': Triples de jugadores (3P)
                   - 'player_combos': Combinaciones (double_double, etc.)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario estructurado con odds por target
        """
        if targets is None:
            targets = ['moneyline', 'totals', 'spreads', 'team_totals', 
                      'player_points', 'player_assists', 'player_rebounds', 
                      'player_threes', 'player_combos']
        
        result = {
            'sport_event_id': sport_event_id,
            'targets': {},
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'requested_targets': targets
            }
        }
        
        try:
            # Obtener odds principales del juego
            if any(target in ['moneyline', 'totals', 'spreads', 'team_totals'] for target in targets):
                game_odds = self.get_odds(sport_event_id, format)
                result['targets']['game_markets'] = self._extract_game_markets(game_odds, targets)
            
            # Obtener player props si se solicitan
            if any(target.startswith('player_') for target in targets):
                player_props = self.get_player_props(sport_event_id, format)
                result['targets']['player_props'] = self._extract_player_markets(player_props, targets)
                
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo odds específicas para {sport_event_id}: {e}")
            result['error'] = str(e)
        
        return result
    
    def _extract_game_markets(self, odds_data: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
        """
        Extrae mercados de juego específicos según los targets solicitados.
        
        Args:
            odds_data: Datos completos de odds del juego
            targets: Targets específicos solicitados
            
        Returns:
            Diccionario con mercados extraídos
        """
        markets = {}
        
        if 'markets' not in odds_data:
            return markets
        
        for market in odds_data['markets']:
            market_name = market.get('name', '').lower()
            
            # Moneyline (is_win target)
            if 'moneyline' in targets and any(keyword in market_name for keyword in ['moneyline', 'match winner', 'winner']):
                markets['moneyline'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'is_win'
                }
            
            # Totals (total_points target)
            elif 'totals' in targets and any(keyword in market_name for keyword in ['total points', 'over/under', 'totals']):
                markets['totals'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'total_points'
                }
            
            # Spreads
            elif 'spreads' in targets and any(keyword in market_name for keyword in ['spread', 'handicap', 'point spread']):
                markets['spreads'] = {
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'point_spread'
                }
            
            # Team Totals (teams_points target)
            elif 'team_totals' in targets and any(keyword in market_name for keyword in ['team total', 'team points']):
                if 'team_totals' not in markets:
                    markets['team_totals'] = []
                markets['team_totals'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'teams_points'
                })
        
        return markets
    
    def _extract_player_markets(self, props_data: Dict[str, Any], targets: List[str]) -> Dict[str, Any]:
        """
        Extrae mercados de jugadores específicos según los targets solicitados.
        
        Args:
            props_data: Datos completos de player props
            targets: Targets específicos solicitados
            
        Returns:
            Diccionario con mercados de jugadores extraídos
        """
        player_markets = {}
        
        if 'markets' not in props_data:
            return player_markets
        
        for market in props_data['markets']:
            market_name = market.get('name', '').lower()
            
            # Player Points (PTS target)
            if 'player_points' in targets and any(keyword in market_name for keyword in ['player points', 'points over/under', 'total points']):
                if 'player_points' not in player_markets:
                    player_markets['player_points'] = []
                player_markets['player_points'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'PTS'
                })
            
            # Player Assists (AST target)
            elif 'player_assists' in targets and any(keyword in market_name for keyword in ['assists', 'player assists']):
                if 'player_assists' not in player_markets:
                    player_markets['player_assists'] = []
                player_markets['player_assists'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'AST'
                })
            
            # Player Rebounds (TRB target)
            elif 'player_rebounds' in targets and any(keyword in market_name for keyword in ['rebounds', 'total rebounds', 'player rebounds']):
                if 'player_rebounds' not in player_markets:
                    player_markets['player_rebounds'] = []
                player_markets['player_rebounds'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'TRB'
                })
            
            # Player Threes (3P target)
            elif 'player_threes' in targets and any(keyword in market_name for keyword in ['3-pointers', 'three pointers', 'threes made']):
                if 'player_threes' not in player_markets:
                    player_markets['player_threes'] = []
                player_markets['player_threes'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': '3P'
                })
            
            # Player Combos (double_double, etc.)
            elif 'player_combos' in targets and any(keyword in market_name for keyword in ['double-double', 'triple-double', 'double double']):
                if 'player_combos' not in player_markets:
                    player_markets['player_combos'] = []
                player_markets['player_combos'].append({
                    'market_name': market.get('name'),
                    'outcomes': market.get('outcomes', []),
                    'target_mapping': 'double_double'
                })
        
        return player_markets
    
    def _process_player_props_for_model_comparison(self, data: Dict[str, Any], sport_event_id: str) -> Dict[str, Any]:
        """
        Procesa datos de player props para comparación optimizada con predicciones del modelo.
        
        Estructura los datos para facilitar la comparación entre:
        - Predicciones del modelo (valor esperado)
        - Líneas de casas de apuestas (valor del mercado)
        - Cuotas disponibles (rentabilidad potencial)
        
        Args:
            data: Datos crudos de Sportradar API
            sport_event_id: ID del evento deportivo
            
        Returns:
            Datos estructurados para comparación con modelo
        """
        try:
            # Extraer información del evento
            sport_event_info = data.get('sport_event_players_props', {}).get('sport_event', {})
            players_props = data.get('sport_event_players_props', {}).get('players_props', [])
            
            # Mapeo de mercados Sportradar a nuestros targets
            target_mapping = {
                'total points (incl. overtime)': 'PTS',
                'total assists (incl. overtime)': 'AST', 
                'total rebounds (incl. overtime)': 'TRB',
                'total threes (incl. overtime)': '3P',
                'total three pointers (incl. overtime)': '3P',
                'total 3-pointers (incl. overtime)': '3P'
            }
            
            # Estructura optimizada para comparación
            processed_data = {
                'success': True,
                'sport_event_id': sport_event_id,
                'event_info': {
                    'start_time': sport_event_info.get('start_time'),
                    'status': sport_event_info.get('status'),
                    'competitors': sport_event_info.get('competitors', [])
                },
                'players': {},
                'market_summary': {
                    'total_players': 0,
                    'total_markets': 0,
                    'targets_available': set(),
                    'bookmakers': set()
                }
            }
            
            # Procesar cada jugador
            for player_data in players_props:
                player_info = player_data.get('player', {})
                player_id = player_info.get('id')
                player_name = player_info.get('name')
                
                if not player_name:
                    continue
                
                # Inicializar estructura del jugador
                processed_data['players'][player_name] = {
                    'player_id': player_id,
                    'competitor_id': player_info.get('competitor_id'),
                    'targets': {}
                }
                
                # Procesar mercados del jugador
                for market in player_data.get('markets', []):
                    market_name = market.get('name', '').lower()
                    
                    # Identificar target
                    target = None
                    for market_key, target_key in target_mapping.items():
                        if market_key in market_name:
                            target = target_key
                            break
                    
                    if not target:
                        continue
                    
                    # Inicializar target si no existe
                    if target not in processed_data['players'][player_name]['targets']:
                        processed_data['players'][player_name]['targets'][target] = {
                            'lines': [],
                            'best_odds': {'over': None, 'under': None},
                            'market_consensus': None
                        }
                    
                    # Procesar casas de apuestas
                    for book in market.get('books', []):
                        book_name = book.get('name')
                        
                        if book.get('removed', False):
                            continue
                        
                        # Procesar outcomes (over/under)
                        for outcome in book.get('outcomes', []):
                            line_value = outcome.get('total')
                            outcome_type = outcome.get('type')  # 'over' o 'under'
                            
                            if line_value is None or outcome_type is None:
                                continue
                            
                            # Convertir cuotas a diferentes formatos
                            odds_data = {
                                'decimal': float(outcome.get('odds_decimal', 0)),
                                'american': outcome.get('odds_american', ''),
                                'fractional': outcome.get('odds_fraction', ''),
                                'probability': self._odds_to_probability(outcome.get('odds_decimal', '2.0'))
                            }
                            
                            # Buscar si ya existe esta línea
                            existing_line = None
                            for line in processed_data['players'][player_name]['targets'][target]['lines']:
                                if line['value'] == line_value:
                                    existing_line = line
                                    break
                            
                            if not existing_line:
                                # Crear nueva línea
                                existing_line = {
                                    'value': line_value,
                                    'over': {'odds': [], 'best_odds': None},
                                    'under': {'odds': [], 'best_odds': None}
                                }
                                processed_data['players'][player_name]['targets'][target]['lines'].append(existing_line)
                            
                            # Agregar odds a la línea
                            existing_line[outcome_type]['odds'].append({
                                'bookmaker': book_name,
                                'odds': odds_data,
                                'external_id': outcome.get('external_outcome_id')
                            })
                            
                            # Actualizar mejores odds
                            if (not existing_line[outcome_type]['best_odds'] or 
                                odds_data['decimal'] > existing_line[outcome_type]['best_odds']['decimal']):
                                existing_line[outcome_type]['best_odds'] = odds_data.copy()
                                existing_line[outcome_type]['best_odds']['bookmaker'] = book_name
                            
                            # Actualizar estadísticas
                            processed_data['market_summary']['bookmakers'].add(book_name)
                    
                    processed_data['market_summary']['targets_available'].add(target)
                    processed_data['market_summary']['total_markets'] += 1
                
                processed_data['market_summary']['total_players'] += 1
            
            # Convertir sets a listas para serialización
            processed_data['market_summary']['targets_available'] = list(processed_data['market_summary']['targets_available'])
            processed_data['market_summary']['bookmakers'] = list(processed_data['market_summary']['bookmakers'])
            
            logger.info(f"Props procesadas para comparación: {processed_data['market_summary']['total_players']} jugadores, "
                       f"{processed_data['market_summary']['total_markets']} mercados, "
                       f"targets: {processed_data['market_summary']['targets_available']}")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error procesando props para comparación: {e}")
            return {
                'success': False,
                'sport_event_id': sport_event_id,
                'error': str(e),
                'players': {}
            }
    
    def _odds_to_probability(self, odds_decimal: str) -> float:
        """
        Convierte odds decimales a probabilidad.
        
        Args:
            odds_decimal: Odds en formato decimal
            
        Returns:
            Probabilidad (0.0 a 1.0)
        """
        try:
            decimal_value = float(odds_decimal)
            if decimal_value <= 1.0:
                return 0.5  # Fallback para odds inválidas
            return 1.0 / decimal_value
        except (ValueError, TypeError):
            return 0.5  # Fallback para odds inválidas
    
    def _process_prematch_odds_for_model_comparison(self, data: Dict[str, Any], sport_event_id: str) -> Dict[str, Any]:
        """
        Procesa datos de prematch odds para comparación optimizada con modelo.
        
        CORRECCIÓN: Procesa la estructura real de la respuesta de Sportradar Prematch API.
        La respuesta tiene 'markets' directamente, no 'sport_event_markets'.
        
        Args:
            data: Datos raw de Prematch API
            sport_event_id: ID del sport event
            
        Returns:
            Datos estructurados para comparación con modelo
        """
        # Mapeo de market IDs a nuestros targets (formato real de Sportradar)
        market_id_mapping = {
            # Formato numérico
            1: 'is_win',                    # 1x2/moneyline
            225: 'total_points',            # total_incl_overtime
            227: 'teams_points_home',       # home_total_incl_overtime
            228: 'teams_points_away',       # away_total_incl_overtime
            
            # Formato sr:market:X (formato real de la API)
            'sr:market:1': 'is_win',        # 1x2/moneyline
            'sr:market:225': 'total_points', # total_incl_overtime
            'sr:market:227': 'teams_points_home', # home_total_incl_overtime
            'sr:market:228': 'teams_points_away', # away_total_incl_overtime
            
            # Player Props Markets adicionales encontrados
            'sr:market:8008': 'double_double',  # double double (incl. extra overtime) 
            'sr:market:8009': 'triple_double'   # triple double (incl. extra overtime)
        }
        
        processed_markets = []
        event_info = {}
        market_types = set()
        
        try:
            # Extraer información del evento
            if 'sport_event' in data:
                event = data['sport_event']
                event_info = {
                    'id': event.get('id', sport_event_id),
                    'start_time': event.get('start_time', ''),
                    'competitors': []
                }
                
                # Extraer equipos
                competitors = event.get('competitors', [])
                for comp in competitors:
                    event_info['competitors'].append({
                        'id': comp.get('id', ''),
                        'name': comp.get('name', ''),
                        'qualifier': comp.get('qualifier', ''),  # 'home' o 'away'
                        'abbreviation': comp.get('abbreviation', '')
                    })
            
            # CORRECCIÓN: Procesar markets de la estructura real
            markets = data.get('markets', [])
            
            logger.debug(f"Procesando {len(markets)} markets de Prematch API para {sport_event_id}")
            
            for market in markets:
                market_id = market.get('id')
                market_name = market.get('name', '')
                
                logger.debug(f"Analizando market: ID={market_id}, Name='{market_name}'")
                
                # Solo procesar markets que nos interesan
                if market_id in market_id_mapping:
                    target_type = market_id_mapping[market_id]
                    market_types.add(target_type)
                    
                    logger.debug(f"Market {market_id} mapeado a target '{target_type}'")
                    
                    # Procesar outcomes
                    outcomes = []
                    for outcome in market.get('outcomes', []):
                        # Estructura correcta de odds según la respuesta real
                        odds_info = outcome.get('odds', {})
                        
                        outcome_data = {
                            'id': outcome.get('id', ''),
                            'name': outcome.get('name', ''),
                            'odds_decimal': odds_info.get('decimal'),
                            'odds_american': odds_info.get('american'),
                            'odds_fractional': odds_info.get('fractional'),
                            'point': outcome.get('point'),  # Para totals
                            'total': outcome.get('total'),  # Alternativo para totals
                            'competitor': outcome.get('competitor', {}),  # Para 1x2
                            'bookmaker': outcome.get('bookmaker', {}).get('name', 'sportradar')
                        }
                        
                        # Calcular probabilidad implícita
                        if outcome_data['odds_decimal']:
                            try:
                                prob = 1 / float(outcome_data['odds_decimal'])
                                outcome_data['probability'] = min(0.99, max(0.01, prob))
                            except (ValueError, ZeroDivisionError):
                                outcome_data['probability'] = 0.5
                        
                        outcomes.append(outcome_data)
                    
                    processed_market = {
                        'id': market_id,
                        'name': market_name,
                        'target_type': target_type,
                        'outcomes': outcomes,
                        'outcome_count': len(outcomes)
                    }
                    
                    processed_markets.append(processed_market)
                    logger.debug(f"Procesado market {market_id} con {len(outcomes)} outcomes")
                else:
                    logger.debug(f"Market {market_id} no está en nuestros targets - ignorado")
            
            logger.info(f"Procesados {len(processed_markets)} markets prematch para {sport_event_id}")
            logger.info(f"Market types encontrados: {market_types}")
            
        except Exception as e:
            logger.error(f"Error procesando prematch odds para {sport_event_id}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'success': True,
            'sport_event_id': sport_event_id,
            'markets': processed_markets,
            'event_info': event_info,
            'market_types': list(market_types),
            'total_markets_in_response': len(data.get('markets', [])),
            'markets_processed': len(processed_markets)
        }

    def _parse_xml_response(self, xml_text: str) -> Dict[str, Any]:
        """
        Parsea respuesta XML de Sportradar y convierte a diccionario.
        
        Args:
            xml_text: Texto XML de la respuesta
            
        Returns:
            Diccionario con datos parseados
        """
        try:
            root = ET.fromstring(xml_text)
            
            # Extraer namespace si existe
            namespace = ''
            if root.tag.startswith('{'):
                namespace = root.tag.split('}')[0] + '}'
            
            # Procesar según el tipo de respuesta
            if root.tag.endswith('books'):
                # Lista de bookmakers
                books = []
                for book in root.findall(f'{namespace}book'):
                    books.append({
                        'id': book.get('id'),
                        'name': book.get('name')
                    })
                return {'books': books}
            
            elif root.tag.endswith('markets'):
                # Mercados de odds
                markets = []
                for market in root.findall(f'{namespace}market'):
                    market_data = {
                        'id': market.get('id'),
                        'name': market.get('name'),
                        'outcomes': []
                    }
                    
                    for outcome in market.findall(f'{namespace}outcome'):
                        outcome_data = {
                            'id': outcome.get('id'),
                            'name': outcome.get('name'),
                            'odds': outcome.get('odds')
                        }
                        market_data['outcomes'].append(outcome_data)
                    
                    markets.append(market_data)
                
                return {'markets': markets}
            
            else:
                # Respuesta genérica - convertir a dict básico
                result = {}
                for child in root:
                    tag_name = child.tag.replace(namespace, '')
                    if child.text:
                        result[tag_name] = child.text
                    else:
                        result[tag_name] = {attr: child.get(attr) for attr in child.attrib}
                
                return result
                
        except ET.ParseError as e:
            logger.error(f"Error parseando XML: {e}")
            raise
    
    def get_nba_odds_for_targets(self, date: str = None, targets: List[str] = None, 
                                format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds específicas para targets del modelo NBA en una fecha determinada.
        
        Args:
            date: Fecha en formato YYYY-MM-DD (None para hoy)
            targets: Lista de targets específicos (None para todos)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Diccionario con odds estructuradas por target para todos los partidos del día
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        if targets is None:
            targets = ['moneyline', 'totals', 'team_totals', 'player_points', 
                      'player_assists', 'player_rebounds', 'player_threes']
        
        logger.info(f"Obteniendo odds NBA para targets específicos: {date}")
        
        result = {
            'date': date,
            'targets_requested': targets,
            'games': {},
            'summary': {
                'total_games': 0,
                'games_with_odds': 0,
                'targets_found': {}
            },
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        try:
            # Obtener schedule del día
            schedule = self.get_daily_odds_schedule(date, sport_id=2)
            
            if 'sport_events' not in schedule:
                result['message'] = 'No hay partidos NBA programados para esta fecha'
                return result
            
            sport_events = schedule['sport_events']
            result['summary']['total_games'] = len(sport_events)
            
            # Procesar cada partido
            for event in sport_events:
                sport_event_id = event['id']
                
                try:
                    # Obtener odds específicas para este partido
                    game_odds = self.get_specific_nba_odds(sport_event_id, targets, format)
                    
                    # Estructurar datos del partido
                    game_info = {
                        'sport_event_id': sport_event_id,
                        'teams': {
                            'home': event.get('competitors', [{}])[0].get('name', 'Unknown'),
                            'away': event.get('competitors', [{}])[1].get('name', 'Unknown') if len(event.get('competitors', [])) > 1 else 'Unknown'
                        },
                        'scheduled': event.get('scheduled'),
                        'odds': game_odds.get('targets', {}),
                        'has_odds': not game_odds.get('error')
                    }
                    
                    result['games'][sport_event_id] = game_info
                    
                    if game_info['has_odds']:
                        result['summary']['games_with_odds'] += 1
                        
                        # Contar targets encontrados
                        for target_category in game_odds.get('targets', {}):
                            if target_category not in result['summary']['targets_found']:
                                result['summary']['targets_found'][target_category] = 0
                            result['summary']['targets_found'][target_category] += 1
                
                except SportradarAPIError as e:
                    logger.warning(f"Error obteniendo odds para {sport_event_id}: {e}")
                    result['games'][sport_event_id] = {
                        'sport_event_id': sport_event_id,
                        'error': str(e),
                        'has_odds': False
                    }
            
            result['message'] = f'Procesados {len(sport_events)} partidos, {result["summary"]["games_with_odds"]} con odds'
            
        except SportradarAPIError as e:
            logger.error(f"Error obteniendo schedule para {date}: {e}")
            result['error'] = str(e)
            result['message'] = f'Error obteniendo datos: {e}'
        
        return result
    
    def extract_odds_for_predictions(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extrae y estructura odds específicamente para el sistema de predicciones.
        
        Args:
            odds_data: Datos de odds obtenidos de get_nba_odds_for_targets
            
        Returns:
            Diccionario estructurado para facilitar la integración con el modelo
        """
        predictions_data = {
            'date': odds_data.get('date'),
            'games_count': odds_data.get('summary', {}).get('total_games', 0),
            'games': {}
        }
        
        for game_id, game_data in odds_data.get('games', {}).items():
            if not game_data.get('has_odds'):
                continue
                
            game_predictions = {
                'teams': game_data.get('teams', {}),
                'scheduled': game_data.get('scheduled'),
                'targets': {}
            }
            
            odds = game_data.get('odds', {})
            
            # Extraer odds para targets específicos del modelo
            
            # 1. Game-level targets
            if 'game_markets' in odds:
                game_markets = odds['game_markets']
                
                # is_win (moneyline)
                if 'moneyline' in game_markets:
                    game_predictions['targets']['is_win'] = {
                        'market': game_markets['moneyline']['market_name'],
                        'odds': game_markets['moneyline']['outcomes']
                    }
                
                # total_points (totals)
                if 'totals' in game_markets:
                    game_predictions['targets']['total_points'] = {
                        'market': game_markets['totals']['market_name'],
                        'odds': game_markets['totals']['outcomes']
                    }
                
                # teams_points (team totals)
                if 'team_totals' in game_markets:
                    game_predictions['targets']['teams_points'] = game_markets['team_totals']
            
            # 2. Player-level targets
            if 'player_props' in odds:
                player_props = odds['player_props']
                
                # Mapear cada tipo de prop a su target correspondiente
                target_mapping = {
                    'player_points': 'PTS',
                    'player_assists': 'AST', 
                    'player_rebounds': 'TRB',
                    'player_threes': '3P',
                    'player_combos': 'double_double'
                }
                
                for prop_type, target_name in target_mapping.items():
                    if prop_type in player_props:
                        game_predictions['targets'][target_name] = player_props[prop_type]
            
            predictions_data['games'][game_id] = game_predictions
        
        return predictions_data
    
    # === UTILIDADES ===
    
    def clear_cache(self, pattern: Optional[str] = None):
        """
        Limpia el cache optimizado.
        
        Args:
            pattern: Patrón opcional para limpiar solo ciertas entradas
        """
        self._cache.clear(pattern)
        logger.info(f"Cache limpiado{f' (patrón: {pattern})' if pattern else ''}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas completas del cache optimizado."""
        stats = self._cache.get_stats()
        size_info = self._cache.get_size_info()
        
        return {
            **stats,
            **size_info,
            'cache_type': getattr(self._cache, 'cache_type', 'unknown'),
            'cache_enabled': getattr(self._cache, 'cache_enabled', False)
        }
    
    def cleanup_cache(self):
        """Limpia entradas expiradas del cache."""
        self._cache.cleanup_expired()
        logger.info("Cleanup de cache completado")
    
    def warm_up_cache(self, targets: List[str] = None):
        """
        Pre-carga datos frecuentemente usados en cache.
        
        Args:
            targets: Lista de datos a pre-cargar (deportes, bookmakers, etc.)
        """
        if targets is None:
            targets = ['sports', 'bookmakers']
        
        logger.info(f"Iniciando warm-up de cache para: {targets}")
        
        try:
            if 'sports' in targets:
                self.get_sports()
                logger.debug("Sports pre-cargados en cache")
            
            if 'bookmakers' in targets:
                self.get_bookmakers()
                logger.debug("Bookmakers pre-cargados en cache")
            
            logger.info("Cache warm-up completado")
            
        except Exception as e:
            logger.warning(f"Error durante cache warm-up: {e}")
    
    def prefetch_odds_data(self, days_ahead: int = 2, max_games: int = 10):
        """
        Pre-carga odds para próximos partidos de manera asíncrona.
        
        Args:
            days_ahead: Días hacia adelante
            max_games: Máximo de partidos a pre-cargar
        """
        logger.info(f"Pre-cargando odds para próximos {max_games} partidos")
        
        try:
            # Obtener próximos partidos sin usar cache para tener datos frescos
            next_games = self.get_next_nba_games(days_ahead, max_games)
            
            if not next_games.get('games'):
                logger.info("No hay partidos próximos para pre-cargar")
                return
            
            # Pre-cargar odds para cada partido
            prefetch_count = 0
            for game in next_games['games'][:max_games]:
                try:
                    sport_event_id = game['sport_event_id']
                    
                    # Pre-cargar odds principales (TTL corto para mantener actualizado)
                    self.get_odds(sport_event_id)
                    
                    # Pre-cargar player props si disponible
                    try:
                        self.get_player_props(sport_event_id)
                    except Exception:
                        pass  # Player props puede no estar disponible
                    
                    prefetch_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error pre-cargando {game.get('sport_event_id', 'unknown')}: {e}")
            
            logger.info(f"Pre-cargadas odds para {prefetch_count} partidos")
            
        except Exception as e:
            logger.warning(f"Error durante prefetch de odds: {e}")
    
    def get_cache_performance_report(self) -> Dict[str, Any]:
        """
        Genera reporte detallado de rendimiento del cache.
        
        Returns:
            Reporte completo de rendimiento
        """
        stats = self.get_cache_stats()
        
        # Calcular métricas adicionales
        total_requests = stats.get('hits', 0) + stats.get('misses', 0)
        hit_ratio = stats.get('hit_ratio', 0)
        
        # Estimar ahorro de tiempo/calls
        estimated_time_saved = stats.get('hits', 0) * 0.5  # Asumimos 0.5s por call evitada
        
        performance_report = {
            'cache_performance': {
                'hit_ratio_percentage': hit_ratio,
                'total_requests': total_requests,
                'cache_hits': stats.get('hits', 0),
                'cache_misses': stats.get('misses', 0),
                'estimated_time_saved_seconds': estimated_time_saved,
                'memory_efficiency': stats.get('memory_hit_ratio', 0)
            },
            'storage_info': {
                'memory_entries': stats.get('memory_entries', 0),
                'disk_files': stats.get('disk_files', 0),
                'memory_size_mb': stats.get('estimated_memory_mb', 0),
                'disk_size_mb': stats.get('disk_size_mb', 0)
            },
            'optimization_suggestions': []
        }
        
        # Generar sugerencias de optimización
        if hit_ratio < 50:
            performance_report['optimization_suggestions'].append(
                "Hit ratio bajo - considerar aumentar TTL para datos estables"
            )
        
        if stats.get('memory_entries', 0) > 500:
            performance_report['optimization_suggestions'].append(
                "Muchas entradas en memoria - considerar cleanup más frecuente"
            )
        
        if stats.get('disk_size_mb', 0) > 50:
            performance_report['optimization_suggestions'].append(
                "Cache en disco grande - considerar limpieza de archivos antiguos"
            )
        
        if stats.get('errors', 0) > 0:
            performance_report['optimization_suggestions'].append(
                f"Se detectaron {stats.get('errors', 0)} errores de cache - revisar logs"
            )
        
        return performance_report
    
    def test_connection(self) -> Dict[str, Any]:
        """
        Prueba la conexión con la API usando endpoints que funcionan.
        
        Returns:
            Resultado del test
        """
        try:
            start_time = time.time()
            # Usar endpoint que sabemos que funciona: sports
            sports = self.get_sports()
            response_time = time.time() - start_time
            
            return {
                'success': True,
                'response_time': response_time,
                'api_accessible': True,
                'sports_count': len(sports.get('sports', [])),
                'message': "Conexión exitosa con Sportradar Odds Comparison API"
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'api_accessible': False,
                'message': f"Error de conexión: {e}"
            }
    
    def __str__(self) -> str:
        """Representación string del cliente."""
        return f"SportradarAPI(base_url={self.base_url}, cache_entries={len(self._cache)})" 

    
    def get_next_nba_games(self, days_ahead: int = 5, max_games: int = 20) -> Dict[str, Any]:
        """
        Obtiene automáticamente los próximos partidos NBA en los próximos días.
        
        Args:
            days_ahead: Número de días hacia adelante a buscar (default: 5)
            max_games: Máximo número de partidos a retornar (default: 20)
            
        Returns:
            Diccionario con los próximos partidos NBA y sus IDs
        """
        logger.info(f"Obteniendo próximos {max_games} partidos NBA en {days_ahead} días")
        
        result = {
            'search_period': {
                'start_date': datetime.now().strftime("%Y-%m-%d"),
                'end_date': (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
                'days_ahead': days_ahead
            },
            'games': [],
            'total_found': 0,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'max_games': max_games
            }
        }
        
        try:
            # Buscar partidos día por día
            current_date = datetime.now()
            
            for day_offset in range(days_ahead + 1):
                search_date = (current_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
                
                try:
                    # Obtener schedule del día
                    daily_schedule = self.get_daily_odds_schedule(search_date, sport_id=2)
                    
                    if 'sport_events' in daily_schedule:
                        for event in daily_schedule['sport_events']:
                            if len(result['games']) >= max_games:
                                break
                                
                            # Extraer información del partido
                            game_info = {
                                'sport_event_id': event['id'],
                                'date': search_date,
                                'scheduled': event.get('scheduled'),
                                'teams': {
                                    'home': event.get('competitors', [{}])[0].get('name', 'Unknown'),
                                    'away': event.get('competitors', [{}])[1].get('name', 'Unknown') if len(event.get('competitors', [])) > 1 else 'Unknown'
                                },
                                'venue': event.get('venue', {}).get('name'),
                                'status': event.get('status')
                            }
                            
                            result['games'].append(game_info)
                    
                    if len(result['games']) >= max_games:
                        break
                        
                except SportradarAPIError as e:
                    logger.debug(f"No hay partidos disponibles para {search_date}: {e}")
                    continue
            
            result['total_found'] = len(result['games'])
            result['message'] = f'Encontrados {len(result["games"])} partidos NBA en los próximos {days_ahead} días'
            
        except Exception as e:
            logger.error(f"Error obteniendo próximos partidos NBA: {e}")
            result['error'] = str(e)
            result['message'] = f'Error en búsqueda: {e}'
        
        return result
    
    def get_odds_for_next_games(self, days_ahead: int = 3, max_games: int = 10, 
                               targets: List[str] = None, format: str = "us") -> Dict[str, Any]:
        """
        Obtiene automáticamente odds para los próximos partidos NBA.
        
        Args:
            days_ahead: Días hacia adelante a buscar (default: 3)
            max_games: Máximo de partidos a procesar (default: 10)
            targets: Targets específicos (None para todos)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Odds completas para los próximos partidos
        """
        logger.info(f"Obteniendo odds automáticamente para próximos {max_games} partidos")
        
        # Obtener próximos partidos
        next_games = self.get_next_nba_games(days_ahead, max_games)
        
        if not next_games['games']:
            return {
                'message': f'No se encontraron partidos NBA en los próximos {days_ahead} días',
                'games': {},
                'summary': {'total_games': 0, 'games_with_odds': 0},
                'search_period': next_games['search_period']
            }
        
        # Configurar targets por defecto
        if targets is None:
            targets = ['moneyline', 'totals', 'team_totals', 'player_points', 
                      'player_assists', 'player_rebounds', 'player_threes']
        
        result = {
            'search_period': next_games['search_period'],
            'targets_requested': targets,
            'games': {},
            'summary': {
                'total_games': len(next_games['games']),
                'games_with_odds': 0,
                'games_processed': 0,
                'targets_found': {}
            },
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'automated': True
            }
        }
        
        # Obtener odds para cada partido
        for game in next_games['games']:
            sport_event_id = game['sport_event_id']
            result['summary']['games_processed'] += 1
            
            try:
                # Obtener odds específicas
                game_odds = self.get_specific_nba_odds(sport_event_id, targets, format)
                
                # Estructurar datos del partido
                game_data = {
                    'sport_event_id': sport_event_id,
                    'date': game['date'],
                    'scheduled': game['scheduled'],
                    'teams': game['teams'],
                    'venue': game.get('venue'),
                    'odds': game_odds.get('targets', {}),
                    'has_odds': not game_odds.get('error'),
                    'error': game_odds.get('error')
                }
                
                result['games'][sport_event_id] = game_data
                
                if game_data['has_odds']:
                    result['summary']['games_with_odds'] += 1
                    
                    # Contar targets encontrados
                    for target_category in game_odds.get('targets', {}):
                        if target_category not in result['summary']['targets_found']:
                            result['summary']['targets_found'][target_category] = 0
                        result['summary']['targets_found'][target_category] += 1
                
            except Exception as e:
                logger.warning(f"Error obteniendo odds para {sport_event_id}: {e}")
                result['games'][sport_event_id] = {
                    'sport_event_id': sport_event_id,
                    'date': game['date'],
                    'teams': game['teams'],
                    'error': str(e),
                    'has_odds': False
                }
        
        result['message'] = f'Procesados {result["summary"]["games_processed"]} partidos, {result["summary"]["games_with_odds"]} con odds disponibles'
        
        return result
    
    def get_live_and_upcoming_odds(self, targets: List[str] = None, format: str = "us") -> Dict[str, Any]:
        """
        Obtiene odds para partidos de hoy + próximos 2 días automáticamente.
        Ideal para usar en tiempo real sin especificar fechas.
        
        Args:
            targets: Targets específicos (None para todos)
            format: Formato de odds ("us", "eu", "uk")
            
        Returns:
            Odds para partidos actuales y próximos
        """
        logger.info("Obteniendo odds para partidos en vivo y próximos automáticamente")
        
        if targets is None:
            targets = ['moneyline', 'totals', 'team_totals', 'player_points', 
                      'player_assists', 'player_rebounds', 'player_threes']
        
        # Obtener fechas: hoy + próximos 2 días
        today = datetime.now()
        dates_to_check = [
            today.strftime("%Y-%m-%d"),  # Hoy
            (today + timedelta(days=1)).strftime("%Y-%m-%d"),  # Mañana
            (today + timedelta(days=2)).strftime("%Y-%m-%d")   # Pasado mañana
        ]
        
        result = {
            'dates_checked': dates_to_check,
            'targets_requested': targets,
            'by_date': {},
            'all_games': {},
            'summary': {
                'total_dates': len(dates_to_check),
                'total_games': 0,
                'games_with_odds': 0,
                'dates_with_games': 0
            },
            'metadata': {
                'format': format,
                'timestamp': datetime.now().isoformat(),
                'automated': True,
                'type': 'live_and_upcoming'
            }
        }
        
        # Procesar cada fecha
        for date in dates_to_check:
            try:
                # Obtener odds para la fecha específica
                date_odds = self.get_nba_odds_for_targets(date, targets, format)
                
                result['by_date'][date] = date_odds
                
                # Agregar partidos al resultado consolidado
                for game_id, game_data in date_odds.get('games', {}).items():
                    result['all_games'][game_id] = game_data
                    result['summary']['total_games'] += 1
                    
                    if game_data.get('has_odds'):
                        result['summary']['games_with_odds'] += 1
                
                # Contar fechas con partidos
                if date_odds.get('summary', {}).get('total_games', 0) > 0:
                    result['summary']['dates_with_games'] += 1
                
            except Exception as e:
                logger.warning(f"Error obteniendo odds para {date}: {e}")
                result['by_date'][date] = {
                    'date': date,
                    'error': str(e),
                    'games': {}
                }
        
        result['message'] = f'Encontrados {result["summary"]["total_games"]} partidos en {result["summary"]["dates_with_games"]} fechas con partidos'
        
        return result
    
    def get_automated_predictions_data(self, max_games: int = 15, days_ahead: int = 5) -> Dict[str, Any]:
        """
        Método principal automatizado para obtener datos de odds para el sistema de predicciones.
        No requiere especificar event_id ni date - todo es automático.
        
        Args:
            max_games: Máximo número de partidos a procesar (default: 15)
            days_ahead: Días hacia adelante para buscar partidos (default: 5)
            
        Returns:
            Datos estructurados listos para el sistema de predicciones
        """
        logger.info(f"Iniciando proceso automatizado para sistema de predicciones")
        
        # Obtener odds para próximos partidos automáticamente
        odds_data = self.get_odds_for_next_games(
            days_ahead=days_ahead,
            max_games=max_games,
            targets=None,  # Todos los targets
            format="us"
        )
        
        # Extraer datos para predicciones
        predictions_data = self.extract_odds_for_predictions(odds_data)
        
        # Agregar información adicional
        predictions_data['automation_info'] = {
            'method': 'automated_predictions_data',
            'search_period': odds_data.get('search_period', {}),
            'games_found': odds_data.get('summary', {}).get('total_games', 0),
            'games_with_odds': odds_data.get('summary', {}).get('games_with_odds', 0),
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'max_games': max_games,
                'days_ahead': days_ahead
            }
        }
        
        return predictions_data
    
    def quick_odds_check(self, game_limit: int = 5) -> Dict[str, Any]:
        """
        Verificación rápida de odds disponibles sin especificar parámetros.
        Útil para tests rápidos y verificaciones.
        
        Args:
            game_limit: Límite de partidos a verificar (default: 5)
            
        Returns:
            Resumen rápido de odds disponibles
        """
        logger.info(f"Verificación rápida de odds para {game_limit} partidos")
        
        try:
            # Obtener partidos de hoy y mañana
            today_tomorrow_odds = self.get_live_and_upcoming_odds()
            
            # Tomar solo los primeros N partidos
            limited_games = {}
            count = 0
            for game_id, game_data in today_tomorrow_odds.get('all_games', {}).items():
                if count >= game_limit:
                    break
                limited_games[game_id] = game_data
                count += 1
            
            return {
                'quick_check': True,
                'games_checked': len(limited_games),
                'games_limit': game_limit,
                'games': limited_games,
                'summary': {
                    'has_live_odds': any(game.get('has_odds', False) for game in limited_games.values()),
                    'total_with_odds': sum(1 for game in limited_games.values() if game.get('has_odds', False))
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error en verificación rápida: {e}")
            return {
                'quick_check': True,
                'error': str(e),
                'games': {},
                'timestamp': datetime.now().isoformat()
            } 