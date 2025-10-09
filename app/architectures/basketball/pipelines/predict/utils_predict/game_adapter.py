"""
Game Data Adapter
=================

Adaptador que convierte la estructura de datos del juego recibida desde SportRadar
al formato requerido por los modelos de predicción NBA.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class GameDataAdapter:
    """
    Adaptador para convertir datos de juego al formato requerido por los modelos.
    
    Convierte la estructura del objeto Game a DataFrames compatibles
    con los feature engineers de los modelos.
    """
    
    def __init__(self):
        """Inicializa el adaptador de datos"""
        self.team_alias_mapping = {}  # Mapeo de alias a nombres completos si es necesario
    
    def convert_game_to_dataframes(self, game_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convierte los datos del juego a DataFrames para jugadores y equipos.
        
        Args:
            game_data: Datos del juego en formato original
            
        Returns:
            Tuple con (players_df, teams_df)
        """
        try:
            # Verificar que game_data sea un diccionario
            if not isinstance(game_data, dict):
                raise ValueError(f"Expected dict, got {type(game_data).__name__}: {str(game_data)[:100]}")
            
            logger.info(f"Convirtiendo datos de juego: {game_data.get('homeTeam', {}).get('alias', 'HOME')} vs {game_data.get('awayTeam', {}).get('alias', 'AWAY')}")
            
            # Extraer información básica del juego
            game_info = self._extract_game_info(game_data)
            
            # Convertir datos de jugadores
            players_df = self._convert_players_data(game_data, game_info)
            
            # Convertir datos de equipos
            teams_df = self._convert_teams_data(game_data, game_info)
            
            logger.info(f"✅ Conversión completada: {len(players_df)} jugadores, {len(teams_df)} equipos")
            return players_df, teams_df
            
        except Exception as e:
            logger.error(f"❌ Error convirtiendo datos del juego: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _extract_game_info(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrae información básica del juego"""
        return {
            'game_id': game_data.get('gameId'),
            'scheduled': game_data.get('scheduled'),
            'status': game_data.get('status', 'scheduled'),
            'home_team_alias': game_data.get('homeTeam', {}).get('alias', 'HOME'),
            'away_team_alias': game_data.get('awayTeam', {}).get('alias', 'AWAY'),
            'home_team_name': game_data.get('homeTeam', {}).get('name', 'Home Team'),
            'away_team_name': game_data.get('awayTeam', {}).get('name', 'Away Team'),
            'venue': game_data.get('venue', {})
        }
    
    def _convert_players_data(self, game_data: Dict[str, Any], game_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Convierte datos de jugadores al formato requerido por los modelos.
        
        Args:
            game_data: Datos originales del juego
            game_info: Información básica extraída del juego
            
        Returns:
            DataFrame con datos de jugadores formateados
        """
        players_data = []
        
        # Procesar jugadores del equipo local
        home_players = game_data.get('homeTeam', {}).get('players', [])
        for player in home_players:
            if self._should_include_player(player):
                player_row = self._create_player_row(
                    player, 
                    game_info['home_team_alias'], 
                    game_info['away_team_alias'], 
                    is_home=True,
                    game_info=game_info
                )
                players_data.append(player_row)
        
        # Procesar jugadores del equipo visitante
        away_players = game_data.get('awayTeam', {}).get('players', [])
        for player in away_players:
            if self._should_include_player(player):
                player_row = self._create_player_row(
                    player, 
                    game_info['away_team_alias'], 
                    game_info['home_team_alias'], 
                    is_home=False,
                    game_info=game_info
                )
                players_data.append(player_row)
        
        if not players_data:
            logger.warning("No se encontraron jugadores válidos para predicción")
            # Crear DataFrame vacío con columnas requeridas
            return pd.DataFrame(columns=['Player', 'Team', 'Opp', 'Date', 'is_home', 'is_started', 'Pos'])
        
        df = pd.DataFrame(players_data)
        logger.info(f"Datos de jugadores convertidos: {len(df)} jugadores")
        return df
    
    def _convert_teams_data(self, game_data: Dict[str, Any], game_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Convierte datos de equipos al formato requerido por los modelos.
        
        Args:
            game_data: Datos originales del juego
            game_info: Información básica extraída del juego
            
        Returns:
            DataFrame con datos de equipos formateados
        """
        teams_data = []
        
        # Datos del equipo local
        home_team_row = self._create_team_row(
            game_info['home_team_alias'], 
            game_info['away_team_alias'], 
            is_home=True,
            game_info=game_info
        )
        teams_data.append(home_team_row)
        
        # Datos del equipo visitante
        away_team_row = self._create_team_row(
            game_info['away_team_alias'], 
            game_info['home_team_alias'], 
            is_home=False,
            game_info=game_info
        )
        teams_data.append(away_team_row)
        
        df = pd.DataFrame(teams_data)
        logger.info(f"Datos de equipos convertidos: {len(df)} equipos")
        return df
    
    def _should_include_player(self, player: Dict[str, Any]) -> bool:
        """
        Determina si un jugador debe incluirse en las predicciones.
        
        Args:
            player: Datos del jugador
            
        Returns:
            True si el jugador debe incluirse
        """
        # Verificar estado del jugador
        status = player.get('status', '')
        if status in ['INACTIVE', 'OUT', 'SUSPENDED']:
            return False
        
        # Verificar lesiones graves
        injuries = player.get('injuries', [])
        for injury in injuries:
            # injury puede ser un string (ej: "Knee") o un dict (ej: {"type": "Knee", "status": "day-to-day"})
            if isinstance(injury, dict):
                injury_type = injury.get('type', '').upper()
            elif isinstance(injury, str):
                injury_type = injury.upper()
            else:
                injury_type = str(injury).upper()
            
            if injury_type in ['ACL', 'ACHILLES', 'FRACTURE']:
                # Lesiones graves - no incluir
                return False
        
        # Jugadores TWO-WAY pueden no jugar en playoffs/juegos importantes
        if status == 'TWO-WAY':
            # Por ahora los incluimos pero con menor prioridad
            pass
        
        return True
    
    def _create_player_row(self, player: Dict[str, Any], team_alias: str, 
                          opponent_alias: str, is_home: bool, game_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una fila de datos para un jugador.
        
        Args:
            player: Datos del jugador
            team_alias: Alias del equipo del jugador
            opponent_alias: Alias del equipo oponente
            is_home: Si el jugador juega en casa
            game_info: Información del juego
            
        Returns:
            Diccionario con datos del jugador formateados
        """
        # Convertir fecha programada a formato datetime
        scheduled_date = self._parse_game_date(game_info.get('scheduled'))
        
        # Determinar si es titular
        is_starter = player.get('starter', False)
        
        # Estimar minutos basado en role y status
        estimated_minutes = self._estimate_player_minutes(player, is_starter)
        
        # Obtener nombre del jugador y limpiar caracteres especiales si es necesario
        player_name = player.get('fullName', 'Unknown Player')
        
        # Log para debugging del mapeo de nombres
        logger.debug(f"Creando fila para jugador: '{player_name}' ({team_alias} vs {opponent_alias})")
        
        return {
            'Player': player_name,
            'Team': team_alias,
            'Opp': opponent_alias,
            'Date': scheduled_date,
            'is_home': 1 if is_home else 0,
            'Away': '' if is_home else '@',
            'is_started': 1 if is_starter else 0,
            'Pos': player.get('position', 'G'),  # Default a Guard si no se especifica
            'MP': estimated_minutes,
            'playerId': player.get('playerId', 'unknown'),
            'jerseyNumber': player.get('jerseyNumber', '0'),
            'status': player.get('status', 'ACT'),
            'injuries_count': len(player.get('injuries', []))
        }
    
    def _create_team_row(self, team_alias: str, opponent_alias: str, 
                        is_home: bool, game_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una fila de datos para un equipo.
        
        Args:
            team_alias: Alias del equipo
            opponent_alias: Alias del equipo oponente
            is_home: Si el equipo juega en casa
            game_info: Información del juego
            
        Returns:
            Diccionario con datos del equipo formateados
        """
        # Convertir fecha programada
        scheduled_date = self._parse_game_date(game_info.get('scheduled'))
        
        return {
            'Team': team_alias,
            'Opp': opponent_alias,
            'Date': scheduled_date,
            'is_home': 1 if is_home else 0,
            'Away': '' if is_home else '@',
            'game_id': game_info.get('game_id'),
            'venue_capacity': self._safe_get_venue_capacity(game_info.get('venue', {}))
        }
    
    def _parse_game_date(self, scheduled_str: str) -> pd.Timestamp:
        """
        Convierte string de fecha programada a Timestamp.
        
        Args:
            scheduled_str: Fecha en formato ISO string
            
        Returns:
            Pandas Timestamp
        """
        try:
            if scheduled_str:
                # Convertir de ISO format a datetime
                return pd.to_datetime(scheduled_str)
            else:
                # Fecha por defecto
                return pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
        except Exception as e:
            logger.warning(f"Error parseando fecha '{scheduled_str}': {e}")
            return pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    
    def _estimate_player_minutes(self, player: Dict[str, Any], is_starter: bool) -> float:
        """
        Estima los minutos que jugará un jugador basado en su rol y estado.
        
        Args:
            player: Datos del jugador
            is_starter: Si es titular
            
        Returns:
            Minutos estimados
        """
        # Verificar estado
        status = player.get('status', 'ACT')
        
        # Verificar lesiones
        injuries = player.get('injuries', [])
        has_injury = len(injuries) > 0
        
        # Estimar minutos basado en role
        if status == 'INACTIVE' or status == 'OUT':
            return 0.0
        elif has_injury:
            # Reducir minutos si tiene lesión
            base_minutes = 28.0 if is_starter else 15.0
            return base_minutes * 0.8  # 20% menos por lesión
        elif is_starter:
            # Titulares típicamente juegan 28-36 minutos
            return 32.0
        elif status == 'TWO-WAY':
            # Jugadores two-way juegan menos
            return 8.0
        else:
            # Suplentes regulares
            return 18.0
    
    def create_prediction_summary_structure(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea la estructura para el resumen de predicciones.
        
        Args:
            game_data: Datos originales del juego
            
        Returns:
            Estructura base para las predicciones
        """
        game_info = self._extract_game_info(game_data)
        
        return {
            'game_info': {
                'gameId': game_info['game_id'],
                'scheduled': game_info['scheduled'],
                'status': game_info['status'],
                'venue': game_info['venue']
            },
            'matchup': {
                'home_team': {
                    'alias': game_info['home_team_alias'],
                    'name': game_info['home_team_name']
                },
                'away_team': {
                    'alias': game_info['away_team_alias'],
                    'name': game_info['away_team_name']
                }
            },
            'predictions': {
                'players': {},
                'teams': {},
                'game_totals': {}
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'prediction_version': '1.0'
            }
        }
    
    def _safe_get_venue_capacity(self, venue_data: Any) -> int:
        """
        Obtener capacidad del venue de forma segura
        
        Args:
            venue_data: Datos del venue (puede ser dict, string o None)
            
        Returns:
            Capacidad del venue o valor por defecto
        """
        default_capacity = 20000
        
        if isinstance(venue_data, dict):
            return venue_data.get('capacity', default_capacity)
        elif isinstance(venue_data, str):
            # Si venue es un string, retornar capacidad por defecto
            logger.debug(f"Venue es string: '{venue_data}', usando capacidad por defecto")
            return default_capacity
        else:
            # Si venue es None u otro tipo
            return default_capacity