"""
Common Utils
=================

Funciones comunes utilizadas en los pipelines de predicción.
"""

import logging
import pandas as pd
from typing import Dict, List, Any
from difflib import SequenceMatcher
import sys
import os

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Importar modelos y data loaders
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.pipelines.predict.utils_predict.game_adapter import GameDataAdapter

logger = logging.getLogger(__name__)

class CommonUtils:
    """
    Funciones comunes utilizadas en los pipelines de predicción.
    """
    def __init__(self):

        data_loader = NBADataLoader(
                players_total_path="app/architectures/basketball/data/players_total.csv",
                players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
                teams_total_path="app/architectures/basketball/data/teams_total.csv",
                teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
                biometrics_path="app/architectures/basketball/data/biometrics.csv"
            )
        self.historical_players, self.historical_teams = data_loader.load_data()
        self.game_adapter = GameDataAdapter()

    def _get_team_id(self, team_abbrev: str) -> str:
        """Obtiene el team_id basado en la abreviación del equipo usando búsqueda inteligente"""
        if self.historical_teams is None or self.historical_teams.empty:
            return 'unknown'
        
        try:
            # Usar búsqueda inteligente para encontrar el equipo
            found_team = self._smart_team_search(self.historical_teams, team_abbrev)
            
            if not found_team.empty:
                team_id = found_team.iloc[0]['team_id']
                logger.info(f"✅ Team ID encontrado para '{team_abbrev}': {team_id}")
                return team_id
            else:
                logger.warning(f"❌ No se encontró team_id para '{team_abbrev}'")
                return 'unknown'
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo team_id para '{team_abbrev}': {e}")
            return 'unknown'
    
    def _get_player_id(self, player_name: str, team_abbrev: str = None) -> str:
        """Obtiene el player_id basado en el nombre del jugador usando búsqueda inteligente"""
        if self.historical_players is None or self.historical_players.empty:
            return 'unknown'
        
        try:
            # Usar búsqueda inteligente para encontrar el jugador
            if team_abbrev and team_abbrev != 'Unknown':
                # Filtrar por equipo específico primero
                team_players = self.historical_players[self.historical_players['Team'] == team_abbrev]
                if not team_players.empty:
                    found_player = self._smart_player_search(team_players, player_name)
                    if found_player.empty:
                        # Si no se encuentra en el equipo específico, buscar en todos
                        found_player = self._smart_player_search(self.historical_players, player_name)
                else:
                    found_player = self._smart_player_search(self.historical_players, player_name)
            else:
                # Buscar en todos los jugadores
                found_player = self._smart_player_search(self.historical_players, player_name)
            
            if not found_player.empty:
                player_id = found_player.iloc[0]['player_id']
                logger.debug(f"✅ Player ID encontrado para '{player_name}': {player_id}")
                return player_id
            else:
                logger.debug(f"❌ No se encontró player_id para '{player_name}'")
                return 'unknown'
                
        except Exception as e:
            logger.error(f"❌ Error obteniendo player_id para '{player_name}': {e}")
            return 'unknown'
    
    def _get_team_full_name(self, team_abbr: str) -> str:
        """Convertir abreviación de equipo a nombre completo"""
        team_names = {
            'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
            'CHA': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
            'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
            'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
            'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
            'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
            'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
            'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns',
            'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
            'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
        }
        return team_names.get(team_abbr, team_abbr)

    def _get_team_abbreviation(self, team_name: str) -> str:
        """Convertir nombre completo de equipo a abreviación"""
        # Mapeo inverso: nombre completo -> abreviación
        team_abbreviations = {
            'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
            'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
            'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
            'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
            'LA Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM',
            'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN',
            'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC',
            'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
            'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS',
            'Toronto Raptors': 'TOR', 'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
        }
        
        # Búsqueda exacta
        exact_match = team_abbreviations.get(team_name)
        if exact_match:
            return exact_match
        
        # Búsqueda case-insensitive
        for full_name, abbr in team_abbreviations.items():
            if team_name.lower() == full_name.lower():
                return abbr
        
        # Búsqueda parcial (por palabras clave)
        team_name_lower = team_name.lower()
        for full_name, abbr in team_abbreviations.items():
            full_name_lower = full_name.lower()
            # Si el nombre del equipo contiene palabras clave significativas
            if ('knicks' in team_name_lower and 'knicks' in full_name_lower) or \
               ('nuggets' in team_name_lower and 'nuggets' in full_name_lower) or \
               ('lakers' in team_name_lower and 'lakers' in full_name_lower) or \
               ('warriors' in team_name_lower and 'warriors' in full_name_lower) or \
               ('celtics' in team_name_lower and 'celtics' in full_name_lower) or \
               ('heat' in team_name_lower and 'heat' in full_name_lower):
                return abbr
        
        # Si no se encuentra, devolver el nombre original
        return team_name
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalizar nombre de jugador para búsqueda inteligente
        
        Args:
            name: Nombre original del jugador
            
        Returns:
            Nombre normalizado
        """
        import unicodedata
        import re
        
        if not name:
            return ""
        
        # Convertir a minúsculas
        normalized = name.lower()
        
        # PASO 1: Remover acentos usando unicodedata (más robusto)
        normalized = unicodedata.normalize('NFD', normalized)
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # PASO 2: Reemplazar caracteres problemáticos específicos comunes en NBA
        replacements = {
            # Acentos y caracteres especiales
            "ć": "c", "č": "c", "ç": "c",          # Jokić -> jokic, Dončić -> doncic
            "š": "s", "ś": "s",                    # Šarić -> saric
            "ž": "z", "ź": "z",                    # Žižić -> zizic
            "ñ": "n", "ń": "n",                    # Peña -> pena
            "ü": "u", "ű": "u", "ù": "u", "û": "u", # Schröder -> schroder
            "ö": "o", "ő": "o", "ò": "o", "ô": "o", # Pöltl -> poltl
            "é": "e", "è": "e", "ê": "e", "ë": "e", # José -> jose
            "í": "i", "ì": "i", "î": "i", "ï": "i", # Martín -> martin
            "ó": "o", "ò": "o", "ô": "o", "õ": "o", # López -> lopez
            "ú": "u", "ù": "u", "û": "u", "ũ": "u", # Hernangómez -> hernangomez
            "á": "a", "à": "a", "â": "a", "ã": "a", # Calderón -> calderon
            "ý": "y", "ÿ": "y",                    # Nombres con y acentuada
            
            # Caracteres especiales y puntuación
            "'": "", "'": "", "`": "",             # Apostrophes y quotes
            "-": " ", "_": " ",                    # Guiones -> espacios
            ".": "",                               # Puntos (Jr., Sr.)
            ",": "",                               # Comas
        }
        
        # Aplicar reemplazos
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # PASO 3: Limpiar caracteres no deseados
        # Mantener solo letras, números y espacios
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # PASO 4: Normalizar espacios
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _smart_player_search(self, players_df: pd.DataFrame, target_player: str) -> pd.DataFrame:
        """
        Búsqueda inteligente de jugador con múltiples estrategias mejoradas
        Maneja ambos formatos: 'player' (minúscula) y 'Player' (mayúscula)
        
        Args:
            players_df: DataFrame con jugadores disponibles
            target_player: Nombre del jugador a buscar
            
        Returns:
            DataFrame con el jugador encontrado (vacío si no se encuentra)
        """
        if players_df.empty or not target_player:
            return pd.DataFrame()
        
        # Determinar qué columna usar para el nombre del jugador
        player_column = 'player' if 'player' in players_df.columns else 'Player'
        
        # Estrategia 1: Búsqueda exacta
        exact_match = players_df[players_df[player_column] == target_player]
        if not exact_match.empty:
            logger.info(f"✅ Jugador encontrado (exacto): {target_player}")
            return exact_match
        
        # Estrategia 2: Búsqueda case-insensitive
        case_match = players_df[players_df[player_column].str.lower() == target_player.lower()]
        if not case_match.empty:
            logger.info(f"✅ Jugador encontrado (case-insensitive): {target_player}")
            return case_match
        
        # Estrategia 3: Búsqueda normalizada (MEJORADA para acentos)
        target_normalized = self._normalize_name(target_player)
        if target_normalized:  # Solo si la normalización produjo algo válido
            # Crear una columna temporal con nombres normalizados para búsqueda eficiente
            players_df_temp = players_df.copy()
            players_df_temp['normalized_name'] = players_df_temp[player_column].apply(self._normalize_name)
            normalized_matches = players_df_temp[players_df_temp['normalized_name'] == target_normalized]
            
            if not normalized_matches.empty:
                found_player = normalized_matches.iloc[0][player_column]
                logger.info(f"✅ Jugador encontrado (normalizado): '{target_player}' -> '{found_player}' ({len(normalized_matches)} registros)")
                return normalized_matches.drop('normalized_name', axis=1)
        
        # Estrategia 4: Búsqueda por contención normalizada (NUEVA)
        # Buscar si el target normalizado está contenido en algún nombre normalizado
        if target_normalized:
            # Usar la columna temporal ya creada
            if 'normalized_name' not in players_df_temp.columns:
                players_df_temp = players_df.copy()
                players_df_temp['normalized_name'] = players_df_temp[player_column].apply(self._normalize_name)
            
            containment_matches = players_df_temp[
                players_df_temp['normalized_name'].str.contains(target_normalized, case=False, na=False) |
                players_df_temp['normalized_name'].apply(lambda x: target_normalized in x if pd.notna(x) else False)
            ]
            
            if not containment_matches.empty:
                found_player = containment_matches.iloc[0][player_column]
                logger.info(f"✅ Jugador encontrado (contención normalizada): '{target_player}' -> '{found_player}' ({len(containment_matches)} registros)")
                return containment_matches.drop('normalized_name', axis=1)
        
        # Estrategia 5: Búsqueda por contención simple (apellido)
        target_words = target_player.lower().split()
        if target_words:
            # Buscar por apellido (última palabra)
            last_word = target_words[-1]
            if len(last_word) > 2:  # Solo apellidos de más de 2 caracteres
                matches = players_df[players_df[player_column].str.contains(last_word, case=False, na=False)]
                if not matches.empty:
                    found_player = matches.iloc[0][player_column]
                    logger.info(f"✅ Jugador encontrado (apellido): '{target_player}' -> '{found_player}'")
                    return matches.head(1)
        
        # Estrategia 6: Búsqueda por palabras clave múltiples
        target_words = target_player.lower().split()
        if len(target_words) >= 2:
            best_match = None
            max_matches = 0
            
            for idx, row in players_df.iterrows():
                player_name = row[player_column].lower()
                word_matches = sum(1 for word in target_words if word in player_name)
                
                # Si encontramos todas las palabras o la mayoría
                if word_matches >= len(target_words) or word_matches >= max_matches:
                    if word_matches > max_matches:
                        max_matches = word_matches
                        best_match = idx
            
            if best_match is not None and max_matches >= max(1, len(target_words) - 1):
                found_player = players_df.loc[best_match, player_column]
                logger.info(f"✅ Jugador encontrado (palabras múltiples): '{target_player}' -> '{found_player}' ({max_matches}/{len(target_words)} palabras)")
                return players_df[players_df.index == best_match]
        
        # Estrategia 7: Búsqueda por similitud usando difflib (NUEVA)
        from difflib import SequenceMatcher
        best_similarity = 0
        best_match_idx = None
        
        target_for_similarity = self._normalize_name(target_player)
        
        for idx, row in players_df.iterrows():
            player_for_similarity = self._normalize_name(row[player_column])
            similarity = SequenceMatcher(None, target_for_similarity, player_for_similarity).ratio()
            
            if similarity > best_similarity and similarity > 0.7:  # Umbral de 70%
                best_similarity = similarity
                best_match_idx = idx
        
        if best_match_idx is not None:
            found_player = players_df.loc[best_match_idx, player_column]
            logger.info(f"✅ Jugador encontrado (similitud {best_similarity:.2f}): '{target_player}' -> '{found_player}'")
            return players_df[players_df.index == best_match_idx]
        
        logger.warning(f"❌ Jugador no encontrado con ninguna estrategia: {target_player}")
        return pd.DataFrame()
    
    def _find_similar_players(self, target_player: str, available_players: List[str]) -> List[str]:
        """
        Encontrar jugadores similares usando distancia de edición en caso de error
        
        Args:
            target_player: Nombre del jugador buscado
            available_players: Lista de jugadores disponibles
            
        Returns:
            Lista de jugadores similares ordenados por similitud
        """
        from difflib import SequenceMatcher
        
        target_normalized = self._normalize_name(target_player)
        similarities = []
        
        for player in available_players:
            player_normalized = self._normalize_name(player)
            
            # Calcular similitud usando SequenceMatcher
            similarity = SequenceMatcher(None, target_normalized, player_normalized).ratio()
            
            # Bonus si contiene palabras clave
            target_words = target_normalized.split()
            if len(target_words) >= 2:
                for word in target_words:
                    if len(word) > 2 and word in player_normalized:
                        similarity += 0.2  # Bonus por palabra encontrada
            
            similarities.append((player, similarity))
        
        # Ordenar por similitud descendente y filtrar por umbral mínimo
        similarities.sort(key=lambda x: x[1], reverse=True)
        similar_players = [player for player, sim in similarities if sim > 0.4]  # Umbral 40%
        
        return similar_players
    
    def _get_player_status_from_sportradar(self, game_data: Dict[str, Any], target_player: str) -> str:
        """
        Extraer el estado del jugador desde datos de SportRadar (método mejorado y unificado)
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            Estado del jugador ('ACT', 'INJ', 'OUT', etc.)
        """
        # Buscar en homeTeam
        home_players = game_data.get('homeTeam', {}).get('players', [])
        status = self._get_player_status_in_game(home_players, target_player)
        if status != 'NOT_FOUND':
            return status
        
        # Buscar en awayTeam
        away_players = game_data.get('awayTeam', {}).get('players', [])
        status = self._get_player_status_in_game(away_players, target_player)
        if status != 'NOT_FOUND':
            return status
        
        return 'UNKNOWN'
    
    def _get_is_home_from_sportradar(self, game_data: Dict[str, Any], target_player: str) -> int:
        """
        Determinar si el jugador juega en casa desde datos de SportRadar
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            1 si juega en casa, 0 si juega de visitante
        """
        # Buscar en homeTeam
        home_players = game_data.get('homeTeam', {}).get('players', [])
        for player in home_players:
            if player.get('fullName') == target_player:
                return 1  # Juega en casa
        
        # Si no está en homeTeam, está en awayTeam
        return 0  # Juega de visitante
    
    def _get_is_started_from_sportradar(self, game_data: Dict[str, Any], target_player: str) -> int:
        """
        Determinar si el jugador es titular desde datos de SportRadar
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            1 si es titular, 0 si es suplente
        """
        # Buscar en homeTeam
        home_players = game_data.get('homeTeam', {}).get('players', [])
        for player in home_players:
            if player.get('fullName') == target_player:
                return 1 if player.get('starter', False) else 0
        
        # Buscar en awayTeam
        away_players = game_data.get('awayTeam', {}).get('players', [])
        for player in away_players:
            if player.get('fullName') == target_player:
                return 1 if player.get('starter', False) else 0
        
        return 0  # Por defecto, no es titular
    
    def _get_player_position_from_sportradar(self, game_data: Dict[str, Any], target_player: str) -> str:
        """
        Obtener la posición del jugador desde datos de SportRadar
        
        Args:
            game_data: Datos del juego de SportRadar
            target_player: Nombre del jugador objetivo
            
        Returns:
            Posición del jugador (G, F, C, etc.)
        """
        # Buscar en homeTeam
        home_players = game_data.get('homeTeam', {}).get('players', [])
        for player in home_players:
            if player.get('fullName') == target_player:
                return player.get('position', 'G')
        
        # Buscar en awayTeam
        away_players = game_data.get('awayTeam', {}).get('players', [])
        for player in away_players:
            if player.get('fullName') == target_player:
                return player.get('position', 'G')
        
        return 'G'  # Por defecto, Guard

    def _get_current_team_from_sportradar(self, game_data: Dict[str, Any], target_player: str) -> str:
        """
        Obtener el equipo actual del jugador desde los datos de SportRadar
        """
        try:
            # Buscar en ambos equipos
            for team_key in ['homeTeam', 'awayTeam']:
                team = game_data.get(team_key, {})
                players = team.get('players', [])
                
                for player in players:
                    if player.get('fullName', '').strip().lower() == target_player.strip().lower():
                        # Devolver la abreviación del equipo
                        team_alias = team.get('alias', team.get('name', 'Unknown'))
                        logger.info(f"🏀 Equipo actual de {target_player}: {team_alias}")
                        return team_alias
            
            logger.warning(f"Jugador {target_player} no encontrado en ningún equipo del juego")
            return 'Unknown'
            
        except Exception as e:
            logger.warning(f"Error obteniendo equipo actual para {target_player}: {e}")
            return 'Unknown'

    def _smart_team_search(self, teams_df: pd.DataFrame, target_team: str) -> pd.DataFrame:
        """
        Búsqueda inteligente de equipos con múltiples estrategias
        
        Args:
            teams_df: DataFrame con datos de equipos
            target_team: Nombre del equipo a buscar
            
        Returns:
            DataFrame con el equipo encontrado (puede estar vacío)
        """
        if teams_df.empty or not target_team:
            return pd.DataFrame()
        
        # 1. Búsqueda exacta
        exact_match = teams_df[teams_df['Team'] == target_team]
        if not exact_match.empty:
            logger.debug(f"✅ Equipo encontrado (exacto): '{target_team}'")
            return exact_match
        
        # 2. Búsqueda case-insensitive
        target_normalized = self._normalize_name(target_team)
        for idx, row in teams_df.iterrows():
            if self._normalize_name(row['Team']) == target_normalized:
                found_team = teams_df[teams_df.index == idx]
                logger.debug(f"✅ Equipo encontrado (case-insensitive): '{target_team}' -> '{row['Team']}'")
                return found_team
        
        # 3. Búsqueda parcial (contiene)
        for idx, row in teams_df.iterrows():
            team_normalized = self._normalize_name(row['Team'])
            if target_normalized in team_normalized or team_normalized in target_normalized:
                found_team = teams_df[teams_df.index == idx]
                logger.debug(f"✅ Equipo encontrado (parcial): '{target_team}' -> '{row['Team']}'")
                return found_team
        
        # 4. Búsqueda usando mapeo de equipos
        # Intentar obtener nombre completo desde abreviación
        full_name = self._get_team_full_name(target_team)
        if full_name != target_team:  # Se encontró una conversión
            match = teams_df[teams_df['Team'] == full_name]
            if not match.empty:
                logger.debug(f"✅ Equipo encontrado (mapeo completo): '{target_team}' -> '{full_name}'")
                return match
        
        # 5. Búsqueda usando abreviación desde nombre completo
        abbreviation = self._get_team_abbreviation(target_team)
        if abbreviation != target_team.upper()[:3]:  # Se encontró una conversión válida
            match = teams_df[teams_df['Team'] == abbreviation]
            if not match.empty:
                logger.debug(f"✅ Equipo encontrado (abreviación): '{target_team}' -> '{abbreviation}'")
                return match
        
        # 6. Búsqueda por similitud usando difflib (NUEVA)
        from difflib import SequenceMatcher
        best_similarity = 0
        best_match_idx = None
        
        target_for_similarity = self._normalize_name(target_team)
        
        for idx, row in teams_df.iterrows():
            team_for_similarity = self._normalize_name(row['Team'])
            similarity = SequenceMatcher(None, target_for_similarity, team_for_similarity).ratio()
            
            if similarity > best_similarity and similarity > 0.6:  # Umbral de 60% para equipos
                best_similarity = similarity
                best_match_idx = idx
        
        if best_match_idx is not None:
            found_team = teams_df[teams_df.index == best_match_idx]
            logger.debug(f"✅ Equipo encontrado (similitud {best_similarity:.2f}): '{target_team}' -> '{teams_df.loc[best_match_idx, 'Team']}'")
            return found_team
        
        logger.warning(f"❌ Equipo no encontrado con ninguna estrategia: {target_team}")
        return pd.DataFrame()

    def _get_player_status_in_game(self, players_list: List[Dict], target_name: str) -> str:
        """
        Obtener el estado de un jugador específico en el juego
        
        Args:
            players_list: Lista de jugadores del equipo
            target_name: Nombre del jugador a buscar
            
        Returns:
            Estado del jugador ('ACT', 'OUT', 'INJURED', etc.)
        """
        if not players_list:
            return 'UNKNOWN'
        
        # Normalizar nombre objetivo
        target_normalized = self._normalize_name(target_name)
        
        for player in players_list:
            if not isinstance(player, dict):
                continue
                
            player_name = player.get('fullName', '')
            if not player_name:
                continue
            
            # Comparación normalizada
            if self._normalize_name(player_name) == target_normalized:
                # Revisar lesiones primero
                injuries = player.get('injuries', [])
                if injuries:
                    return 'INJURED'
                
                # Revisar estado general
                status = player.get('status', 'ACT')
                return status
        
        return 'NOT_FOUND'
        
    def _get_is_home_team_from_sportradar(self, game_data: Dict[str, Any], target_team: str) -> int:
        """
        Determinar si el equipo juega en casa desde datos de SportRadar
        
        Args:
            game_data: Datos del juego de SportRadar
            target_team: Nombre del equipo objetivo
            
        Returns:
            1 si juega en casa, 0 si juega de visitante
        """
        # Verificar si es el equipo local
        home_team_name = game_data.get('homeTeam', {}).get('name', '')
        if target_team.lower() in home_team_name.lower() or home_team_name.lower() in target_team.lower():
            return 1
        
        # Si no es el equipo local, es visitante
        return 0
    
