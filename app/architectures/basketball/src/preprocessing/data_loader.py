"""
NBA Data Loader - Nueva versión para datos de SportRadar

Modulo encargado de cargar, procesar y limpiar los datos de los partidos (Jugadores y equipos) 

Argumentos:
- players_total.csv: Estadísticas totales por juego de jugadores
- players_quarters.csv: Estadísticas por cuarto de jugadores  
- teams_total.csv: Estadísticas totales por juego de equipos
- teams_quarters.csv: Estadísticas por cuarto de equipos
- biometrics.csv: Datos biométricos de los jugadores
"""

import logging
import os
import warnings
from datetime import datetime
import sys

import numpy as np
import pandas as pd

# Import circular evitado - usaremos método de normalización local
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ============================================================================
# SINGLETON PATTERN - Caché global de datos
# ============================================================================
_CACHED_DATA = {
    'players_data': None,
    'teams_data': None,
    'players_quarters_data': None,
    'teams_quarters_data': None,
    'biometrics': None,
    'timestamp': None,
    'loaded': False
}

class NBADataLoader:
    """
    Cargador de datos para estadísticas NBA desde SportRadar API
    """
    def __init__(self, 
                 players_total_path: str = "app/architectures/basketball/data/players_total.csv",
                 players_quarters_path: str = "app/architectures/basketball/data/players_quarters.csv", 
                 teams_total_path: str = "app/architectures/basketball/data/teams_total.csv",
                 teams_quarters_path: str = "app/architectures/basketball/data/teams_quarters.csv",
                 biometrics_path: str = "app/architectures/basketball/data/biometrics.csv"):
        """
        Inicializa el cargador de datos
        
        Args:
            players_total_path: Ruta al archivo CSV con estadísticas totales de jugadores
            players_quarters_path: Ruta al archivo CSV con estadísticas por cuarto de jugadores
            teams_total_path: Ruta al archivo CSV con estadísticas totales de equipos  
            teams_quarters_path: Ruta al archivo CSV con estadísticas por cuarto de equipos
            biometrics_path: Ruta al archivo CSV con datos biométricos
        """
        self.players_total_path = players_total_path
        self.players_quarters_path = players_quarters_path
        self.teams_total_path = teams_total_path
        self.teams_quarters_path = teams_quarters_path
        self.biometrics_path = biometrics_path
        
    def load_data(self, use_quarters: bool = False, force_reload: bool = False):
        """
        Carga, valida y combina los datos de partidos y biométricos.
        
        USA CACHÉ GLOBAL para evitar cargar los datos múltiples veces.
        
        Args:
            use_quarters: Si usar datos de cuartos
            force_reload: Forzar recarga de datos (ignorar caché)
        """
        global _CACHED_DATA
        
        # Si ya están cargados y no forzamos recarga, usar caché
        if _CACHED_DATA['loaded'] and not force_reload:
            logger.info(" Usando datos cacheados (evita recarga)")
            
            if use_quarters:
                return (
                    _CACHED_DATA['players_quarters_data'],
                    _CACHED_DATA['teams_quarters_data'],
                    _CACHED_DATA['players_quarters_data'],
                    _CACHED_DATA['teams_quarters_data']
                )
            else:
                return (
                    _CACHED_DATA['players_data'],
                    _CACHED_DATA['teams_data'],
                    _CACHED_DATA['players_quarters_data'],
                    _CACHED_DATA['teams_quarters_data']
                )
        
        # Si no están cargados, cargar por primera vez
        logger.info("Cargando datos de NBA...")
        
        if use_quarters:
            players_path = self.players_quarters_path  # Datos por cuarto
            teams_path = self.teams_quarters_path
        else:
            players_path = self.players_total_path     # Datos totales por juego
            teams_path = self.teams_total_path
        
        # Cargar datasets principales
        players_data = pd.read_csv(players_path)
        teams_data = pd.read_csv(teams_path)
        biometrics = pd.read_csv(self.biometrics_path)
        players_quarters_data = pd.read_csv(self.players_quarters_path)
        teams_quarters_data = pd.read_csv(self.teams_quarters_path)
        
        # Validar datos
        self._validate_players_data(players_data)
        self._validate_teams_data(teams_data)
        self._validate_biometrics(biometrics)
        
        # Validar datos de cuartos con validación específica
        self._validate_quarters_data(players_quarters_data, 'jugadores')
        self._validate_quarters_data(teams_quarters_data, 'equipos')
        
        # Procesar datos
        players_data = self._preprocess_players_data(players_data)
        teams_data = self._preprocess_teams_data(teams_data)
        biometrics = self._preprocess_biometrics(biometrics)
        
        # Procesar datos de cuartos para mantener consistencia de índices
        players_quarters_data = self._preprocess_players_data(players_quarters_data)
        teams_quarters_data = self._preprocess_teams_data(teams_quarters_data)
        
        # Agregar columna HT (halftime) a teams_data si no existe
        if 'HT' not in teams_data.columns:
            ht_target_df = self._generate_halftime_target(teams_quarters_data)
            teams_data = self._merge_halftime_target(teams_data, ht_target_df)
        
        # Merge de jugadores con biometrics
        players_data = self._merge_players_with_biometrics(players_data, biometrics)
        
        # Guardar en caché global
        _CACHED_DATA['players_data'] = players_data
        _CACHED_DATA['teams_data'] = teams_data
        _CACHED_DATA['players_quarters_data'] = players_quarters_data
        _CACHED_DATA['teams_quarters_data'] = teams_quarters_data
        _CACHED_DATA['biometrics'] = biometrics
        _CACHED_DATA['timestamp'] = pd.Timestamp.now()
        _CACHED_DATA['loaded'] = True
        
        logger.info(f" Datos cargados y cacheados exitosamente ({len(players_data)} jugadores)")

        return players_data, teams_data, players_quarters_data, teams_quarters_data
    
    def load_data_with_halftime_target(self, force_reload: bool = False):
        """
        Carga datos de equipos con target de halftime (HT) generado desde datos de cuartos.
        
        USA CACHÉ GLOBAL para evitar cargar los datos múltiples veces.
        
        Args:
            force_reload: Forzar recarga de datos (ignorar caché)
            
        Returns:
            tuple: (players_data, teams_data_with_ht, players_quarters_data, teams_quarters_data) - DataFrames procesados con target HT
        """
        global _CACHED_DATA
        
        # Si ya están cargados, usar caché
        if _CACHED_DATA['loaded'] and not force_reload:
            logger.info(" Usando datos cacheados con HT (evita recarga)")
            return (
                _CACHED_DATA['players_data'],
                _CACHED_DATA['teams_data'],  # Ya tiene HT incluido
                _CACHED_DATA['players_quarters_data'],
                _CACHED_DATA['teams_quarters_data']
            )
        
        # Cargar datos totales de equipos
        teams_total = pd.read_csv(self.teams_total_path)
        teams_quarters = pd.read_csv(self.teams_quarters_path)
        players_data = pd.read_csv(self.players_total_path)
        players_quarters_data = pd.read_csv(self.players_quarters_path)
        biometrics = pd.read_csv(self.biometrics_path)
        
        # Validar datos
        self._validate_teams_data(teams_total)
        self._validate_quarters_data(teams_quarters, 'equipos')
        self._validate_players_data(players_data)
        self._validate_quarters_data(players_quarters_data, 'jugadores')
        self._validate_biometrics(biometrics)
        
        # Generar target HT desde datos de cuartos
        ht_target = self._generate_halftime_target(teams_quarters)
        
        # Merge HT target con datos totales
        teams_with_ht = self._merge_halftime_target(teams_total, ht_target)
        
        # Procesar datos
        players_data = self._preprocess_players_data(players_data)
        players_quarters_data = self._preprocess_players_data(players_quarters_data)
        teams_with_ht = self._preprocess_teams_data(teams_with_ht)
        teams_quarters = self._preprocess_teams_data(teams_quarters)
        biometrics = self._preprocess_biometrics(biometrics)
        
        # Merge de jugadores con biometrics
        players_data = self._merge_players_with_biometrics(players_data, biometrics)
        
        # Guardar en caché global
        _CACHED_DATA['players_data'] = players_data
        _CACHED_DATA['teams_data'] = teams_with_ht
        _CACHED_DATA['players_quarters_data'] = players_quarters_data
        _CACHED_DATA['teams_quarters_data'] = teams_quarters
        _CACHED_DATA['biometrics'] = biometrics
        _CACHED_DATA['timestamp'] = pd.Timestamp.now()
        _CACHED_DATA['loaded'] = True
        
        logger.info(f" Datos con HT cargados y cacheados exitosamente")

        return players_data, teams_with_ht, players_quarters_data, teams_quarters
    
    @staticmethod
    def clear_cache():
        """Limpia el caché de datos para forzar recarga en próxima llamada"""
        global _CACHED_DATA
        _CACHED_DATA['players_data'] = None
        _CACHED_DATA['teams_data'] = None
        _CACHED_DATA['players_quarters_data'] = None
        _CACHED_DATA['teams_quarters_data'] = None
        _CACHED_DATA['biometrics'] = None
        _CACHED_DATA['timestamp'] = None
        _CACHED_DATA['loaded'] = False
        logger.info(" Cache limpiado - próxima carga será desde archivos")
    
    @staticmethod
    def get_cache_info():
        """Retorna información sobre el estado del caché"""
        global _CACHED_DATA
        if _CACHED_DATA['loaded']:
            return {
                'loaded': True,
                'timestamp': _CACHED_DATA['timestamp'],
                'num_players': len(_CACHED_DATA['players_data']) if _CACHED_DATA['players_data'] is not None else 0,
                'num_teams': len(_CACHED_DATA['teams_data']) if _CACHED_DATA['teams_data'] is not None else 0
            }
        return {'loaded': False}
    
    def _validate_players_data(self, df):
        """Valida el DataFrame de datos de jugadores con validación robusta"""
        required_columns = [
            'player_id', 'player', 'Team', 'Opp', 'Date', 'game_id',
            'points', 'rebounds', 'assists', 'is_home'
        ]
        
        # 1. Validar columnas requeridas
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de jugadores: {missing_cols}")
        
        # 2. Validar tipos de datos críticos
        numeric_cols = ['points', 'rebounds', 'assists', 'three_points_made']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f" Columna {col} no es numérica, convirtiendo...")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Validar rangos de valores
        for col in numeric_cols:
            if col in df.columns:
                invalid_values = (df[col] < 0) | (df[col] > 100)
                if invalid_values.any():
                    invalid_count = invalid_values.sum()
                    logger.warning(f" {invalid_count} valores inválidos en {col} (fuera de rango 0-100)")
        
        # 4. Validar fechas
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            null_dates = df['Date'].isnull().sum()
            if null_dates > 0:
                logger.warning(f" {null_dates} fechas inválidas encontradas")
        except Exception as e:
            logger.error(f" Error validando fechas: {e}")
            raise
        
        # 5. Validar duplicados críticos
        duplicate_mask = df.duplicated(subset=['player', 'Date', 'game_id'], keep=False)
        if duplicate_mask.any():
            duplicate_count = duplicate_mask.sum()
            logger.warning(f" {duplicate_count} registros duplicados encontrados (player, Date, game_id)")
        
        # 6. Validar integridad de índices
        if df.index.duplicated().any():
            logger.warning(" Índices duplicados encontrados, reseteando...")
            df = df.reset_index(drop=True)
        
        # 7. Estadísticas de calidad
        logger.info(f" Calidad de datos jugadores:")
        logger.info(f"    Total registros: {len(df):,}")
        logger.info(f"    Jugadores únicos: {df['player'].nunique():,}")
        logger.info(f"    Fechas únicas: {df['Date'].nunique():,}")
        logger.info(f"    Valores nulos: {df.isnull().sum().sum():,}")
        
        return df
            
    def _validate_teams_data(self, df):
        """Valida el DataFrame de datos de equipos con validación robusta"""
        required_columns = [
            'team_id', 'Team', 'Opp', 'Date', 'game_id', 
            'points', 'is_home', 'is_win'
        ]
        
        # 1. Validar columnas requeridas
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de equipos: {missing_cols}")
        
        # 2. Validar tipos de datos críticos
        numeric_cols = ['points']
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f" Columna {col} no es numérica, convirtiendo...")
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 3. Validar rangos de valores
        if 'points' in df.columns:
            invalid_points = (df['points'] < 0) | (df['points'] > 200)
            if invalid_points.any():
                invalid_count = invalid_points.sum()
                logger.warning(f" {invalid_count} valores de puntos inválidos (fuera de rango 0-200)")
        
        # 4. Validar fechas
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            null_dates = df['Date'].isnull().sum()
            if null_dates > 0:
                logger.warning(f" {null_dates} fechas inválidas encontradas")
        except Exception as e:
            logger.error(f" Error validando fechas: {e}")
            raise
        
        # 5. Validar duplicados críticos
        duplicate_mask = df.duplicated(subset=['Team', 'Date', 'game_id'], keep=False)
        if duplicate_mask.any():
            duplicate_count = duplicate_mask.sum()
            logger.warning(f" {duplicate_count} registros duplicados encontrados (Team, Date, game_id)")
        
        # 6. Validar integridad de índices
        if df.index.duplicated().any():
            logger.warning(" Índices duplicados encontrados, reseteando...")
            df = df.reset_index(drop=True)
        
        # 7. Verificar points_against para game_total_points
        if 'points_against' in df.columns:
            points_against_null = df['points_against'].isnull().sum()
            if points_against_null > 0:
                logger.warning(f" {points_against_null} valores nulos en columna points_against")
        
        # 8. Estadísticas de calidad
        logger.info(f" Calidad de datos equipos:")
        logger.info(f"    Total registros: {len(df):,}")
        logger.info(f"    Equipos únicos: {df['Team'].nunique():,}")
        logger.info(f"    Fechas únicas: {df['Date'].nunique():,}")
        logger.info(f"    Valores nulos: {df.isnull().sum().sum():,}")
        
        return df
    
    def _validate_quarters_data(self, df, data_type):
        """
        Valida datos de cuartos (jugadores o equipos) con validación específica
        
        Args:
            df: DataFrame con datos de cuartos
            data_type: 'jugadores' o 'equipos'
        """        
        # Columnas requeridas para cuartos
        if data_type == 'jugadores':
            required_columns = ['player_id', 'player', 'Team', 'Opp', 'Date', 'game_id', 'quarter', 'points']
        else:  # equipos
            required_columns = ['team_id', 'Team', 'Opp', 'Date', 'game_id', 'quarter', 'points']
        
        # 1. Validar columnas requeridas
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de cuartos de {data_type}: {missing_cols}")
        
        # 2. Validar columna quarter
        if 'quarter' in df.columns:
            valid_quarters = df['quarter'].isin([1, 2, 3, 4])
            if not valid_quarters.all():
                invalid_quarters = df[~valid_quarters]['quarter'].unique()
                logger.warning(f"Cuartos inválidos encontrados: {invalid_quarters}")
        
        # 3. Validar puntos por cuarto
        if 'points' in df.columns:
            invalid_points = (df['points'] < 0) | (df['points'] > 50)  # Máximo 50 puntos por cuarto
            if invalid_points.any():
                invalid_count = invalid_points.sum()
        
        # 4. Validar fechas
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            null_dates = df['Date'].isnull().sum()
            if null_dates > 0:
                logger.warning(f"{null_dates} fechas inválidas en datos de cuartos")
        except Exception as e:
            logger.error(f"Error validando fechas en cuartos: {e}")
            raise
        
        return df
    
    def _validate_biometrics(self, df):
        """Valida el DataFrame de datos biométricos"""
        required_columns = ['Player', 'Height', 'Weight']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos biométricos: {missing_cols}")
    
    def _preprocess_players_data(self, df):
        """
        Preprocesa los datos de jugadores con validación robusta
        
        Los datos YA VIENEN LIMPIOS desde SportRadar, pero se valida y mejora:
        - Convierte fechas con validación
        - Valida tipos de datos
        - Crea columnas de compatibilidad
        - Preserva integridad de índices
        """
        df = df.copy()
        
        # 1. Convertir fechas con validación robusta
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            null_dates = df['Date'].isnull().sum()
            if null_dates > 0:
                logger.warning(f" {null_dates} fechas inválidas, eliminando registros...")
                df = df.dropna(subset=['Date'])
        except Exception as e:
            logger.error(f" Error procesando fechas: {e}")
            raise
        
        # 2. Validar orden cronológico por jugador (optimizado para memoria)
        try:
            df = df.sort_values(['player', 'Date'], inplace=False)
            df.reset_index(drop=True, inplace=True)
        except MemoryError:
            logger.warning(" Memoria insuficiente, usando método alternativo...")
            # Método más eficiente en memoria
            df.sort_values(['player', 'Date'], inplace=True)
            df.index = range(len(df))
        
        # 4. Limpiar valores no válidos de forma segura
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            # Reemplazar inf con NaN
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            
            # Validar rangos lógicos
            if col in ['points', 'rebounds', 'assists']:
                invalid_mask = (df[col] < 0) | (df[col] > 100)
                if invalid_mask.any():
                    invalid_count = invalid_mask.sum()
                    logger.warning(f" {invalid_count} valores inválidos en {col}, corrigiendo...")
                    df.loc[invalid_mask, col] = np.nan
        
        # 5. Validar integridad de índices
        if df.index.duplicated().any():
            logger.warning(" Índices duplicados encontrados, reseteando...")
            df = df.reset_index(drop=True)
        
        return df
    
    def _preprocess_teams_data(self, df):
        """
        Preprocesa los datos de equipos
        
        Los datos YA VIENEN LIMPIOS desde SportRadar, así que solo:
        - Convierte fechas  
        - Valida tipos de datos
        - Crea columnas de compatibilidad
        """
        df = df.copy()
        
        # Convertir fechas
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        # Limpiar valores no válidos
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['Team', 'Date']).reset_index(drop=True)
        
        # Validar integridad de índices
        if df.index.duplicated().any():
            logger.warning(" Índices duplicados encontrados en equipos, reseteando...")
            df = df.reset_index(drop=True)
        
        return df
    
    def _preprocess_biometrics(self, df):
        """
        Preprocesa los datos biométricos
        
        - Convierte altura a pulgadas
        - Limpia y valida valores
        - Calcula BMI
        """
        df = df.copy()
        
        # Convertir altura a pulgadas
        def height_to_inches(height_str):
            try:
                feet, inches = height_str.replace('"', '').split("'")
                return int(feet) * 12 + int(inches)
            except:
                return np.nan
        
        df['Height_Inches'] = df['Height'].apply(height_to_inches)
        
        # Validar peso
        df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
        
        # CRÍTICO: Calcular BMI
        # BMI = peso (kg) / altura (m)²
        # Convertir pulgadas a metros: pulgadas * 0.0254
        # Convertir libras a kg: libras * 0.453592
        df['BMI'] = (df['Weight'] * 0.453592) / ((df['Height_Inches'] * 0.0254) ** 2)
        
        # Limpiar valores infinitos o nulos en BMI
        df['BMI'] = df['BMI'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalizar nombre de jugador para búsqueda inteligente
        (Copiado de common_utils.py para evitar import circular)
        """
        import unicodedata
        import re
        
        if not name:
            return ""
        
        # Convertir a minúsculas
        normalized = name.lower()
        
        # PASO 1: Remover acentos usando unicodedata
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
        normalized = re.sub(r'[^a-z0-9\s]', '', normalized)
        
        # PASO 4: Normalizar espacios
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _merge_players_with_biometrics(self, players_data, biometrics):
        """
        Combina los datos de jugadores con los datos biométricos con validación robusta
        """
        # Verificar si los datos biométricos ya están integrados
        biometric_columns = ['Height_Inches', 'Weight', 'BMI']
        has_biometrics = all(col in players_data.columns for col in biometric_columns)
        
        if has_biometrics:
            logger.info("Datos biométricos ya están integrados")
            return players_data
        
        # Validar integridad antes del merge
        original_players_count = len(players_data)
        original_players_index = players_data.index.copy()
        
        logger.info("Integrando datos biométricos con datos de jugadores...")
        
        # Renombrar columna en biometrics para que coincida
        biometrics_renamed = biometrics.rename(columns={'Player': 'player'})
        
        # Validar que no hay duplicados en biometrics
        if biometrics_renamed['player'].duplicated().any():
            logger.warning(" Duplicados encontrados en biometrics, eliminando...")
            biometrics_renamed = biometrics_renamed.drop_duplicates(subset=['player'], keep='first')
        
        # Crear copia para trabajar
        merged = players_data.copy()
        merged['Height_Inches'] = np.nan
        merged['Weight'] = np.nan
        
        # Contadores para estadísticas
        matches_found = 0
        matches_exact = 0
        matches_normalized = 0
        
        # Procesar cada jugador único
        unique_players = merged['player'].unique()
        
        for player in unique_players:
            # Buscar en biometrics con normalización usando método local
            target_normalized = self._normalize_name(player)
            
            # Estrategia 1: Búsqueda exacta
            exact_match = biometrics[biometrics['Player'] == player]
            if not exact_match.empty:
                merged.loc[merged['player'] == player, 'Height_Inches'] = exact_match.iloc[0]['Height_Inches']
                merged.loc[merged['player'] == player, 'Weight'] = exact_match.iloc[0]['Weight']
                matches_exact += 1
                matches_found += 1
                continue
            
            # Estrategia 2: Búsqueda normalizada
            if target_normalized:
                biometrics_temp = biometrics.copy()
                biometrics_temp['normalized_name'] = biometrics_temp['Player'].apply(self._normalize_name)
                normalized_match = biometrics_temp[biometrics_temp['normalized_name'] == target_normalized]
                
                if not normalized_match.empty:
                    merged.loc[merged['player'] == player, 'Height_Inches'] = normalized_match.iloc[0]['Height_Inches']
                    merged.loc[merged['player'] == player, 'Weight'] = normalized_match.iloc[0]['Weight']
                    matches_normalized += 1
                    matches_found += 1
                    logger.debug(f"Match normalizado: '{player}' -> '{normalized_match.iloc[0]['Player']}'")
                    continue
            
            # Estrategia 3: Búsqueda por contención normalizada
            if target_normalized:
                biometrics_temp = biometrics.copy()
                if 'normalized_name' not in biometrics_temp.columns:
                    biometrics_temp['normalized_name'] = biometrics_temp['Player'].apply(self._normalize_name)
                
                containment_match = biometrics_temp[
                    biometrics_temp['normalized_name'].str.contains(target_normalized, case=False, na=False) |
                    biometrics_temp['normalized_name'].apply(lambda x: target_normalized in x if pd.notna(x) else False)
                ]
                
                if not containment_match.empty:
                    merged.loc[merged['player'] == player, 'Height_Inches'] = containment_match.iloc[0]['Height_Inches']
                    merged.loc[merged['player'] == player, 'Weight'] = containment_match.iloc[0]['Weight']
                    matches_found += 1
                    logger.debug(f"Match por contención: '{player}' -> '{containment_match.iloc[0]['Player']}'")
        
        # Validar integridad del merge
        if len(merged) != original_players_count:
            logger.error(f" CRÍTICO: Se perdieron {original_players_count - len(merged)} registros durante el merge")
            raise ValueError("Pérdida de datos durante merge biométrico")
        
        # Validar que los índices se mantuvieron
        if not merged.index.equals(original_players_index):
            logger.warning(" Índices cambiaron durante merge, corrigiendo...")
            merged = merged.set_index(original_players_index)
        
        # Calcular BMI después del merge
        if 'Weight' in merged.columns and 'Height_Inches' in merged.columns:
            # Crear una máscara para valores válidos
            valid_mask = (
                (merged['Weight'].notna()) & 
                (merged['Height_Inches'].notna()) & 
                (merged['Height_Inches'] > 0) & 
                (merged['Weight'] > 0) 
            )
            merged['BMI'] = np.nan
            
            # Calcular BMI de forma segura
            if valid_mask.any():
                # BMI = peso (kg) / altura (m)²
                weight_kg = merged.loc[valid_mask, 'Weight'] * 0.453592
                height_m = merged.loc[valid_mask, 'Height_Inches'] * 0.0254
                height_squared = height_m ** 2
                merged.loc[valid_mask, 'BMI'] = weight_kg / height_squared
            
            bmi_valid = merged['BMI'].notna().sum()
        
        # Estadísticas del merge
        players_with_bio = merged['Height_Inches'].notna().sum()
        total_records = len(merged)
        bio_coverage = (players_with_bio / total_records) * 100
        
        logger.info(f"Merge biométrico completado:")
            
        return merged
    
    def _generate_halftime_target(self, teams_quarters_df):
        """
        Genera el target de halftime (HT) sumando puntos de cuartos 1 y 2
        
        Args:
            teams_quarters_df: DataFrame con datos de cuartos por equipo
            
        Returns:
            DataFrame con target HT por equipo y partido
        """
        
        # Filtrar solo cuartos 1 y 2
        quarters_1_2 = teams_quarters_df[teams_quarters_df['quarter'].isin([1, 2])].copy()
        
        # Agrupar por equipo, partido y fecha para sumar puntos de cuartos 1 y 2
        ht_target = quarters_1_2.groupby(['Team', 'game_id', 'Date', 'Opp', 'is_home']).agg({
            'points': 'sum',  # Suma de puntos de cuartos 1 y 2
            'team_id': 'first',  # Mantener team_id
            'market': 'first',   # Mantener market
            'name': 'first'      # Mantener name
        }).reset_index()
        
        # Renombrar columna de puntos a HT
        ht_target = ht_target.rename(columns={'points': 'HT'})
        
        # Validar que tenemos datos
        if ht_target.empty:
            raise ValueError("No se pudieron generar targets de halftime - verificar datos de cuartos")
        
        # Verificar que no hay valores nulos en HT
        null_ht = ht_target['HT'].isnull().sum()
        if null_ht > 0:
            logger.warning(f"Encontrados {null_ht} valores nulos en target HT")

        return ht_target
    
    def _merge_halftime_target(self, teams_total_df, ht_target_df):
        """
        Hace merge del target HT con el dataset total de equipos
        
        Args:
            teams_total_df: DataFrame con datos totales de equipos
            ht_target_df: DataFrame con target HT
            
        Returns:
            DataFrame con datos totales + target HT
        """
        
        # CRÍTICO: Preservar índices originales
        original_index = teams_total_df.index.copy()
        original_count = len(teams_total_df)
        
        # Convertir Date a datetime en ambos DataFrames para el merge
        teams_total_df['Date'] = pd.to_datetime(teams_total_df['Date'], format='mixed')
        ht_target_df['Date'] = pd.to_datetime(ht_target_df['Date'], format='mixed')
        
        # CRÍTICO: Eliminar duplicados antes del merge para evitar multiplicación de registros
        ht_target_clean = ht_target_df.drop_duplicates(subset=['Team', 'Date'], keep='first')

        # Hacer merge por Team y Date (sin game_id ya que pueden ser diferentes)
        merged_df = pd.merge(
            teams_total_df,
            ht_target_clean[['Team', 'Date', 'HT']],
            on=['Team', 'Date'],
            how='left'
        )
        
        # CRÍTICO: Validar integridad del merge (debe mantener el mismo número de registros)
        if len(merged_df) != original_count:
            logger.error(f" CRÍTICO: Se perdieron {original_count - len(merged_df)} registros durante merge HT")
            raise ValueError("Pérdida de datos durante merge HT")
        
        # CRÍTICO: Preservar índices originales
        if not merged_df.index.equals(original_index):
            logger.warning(" Índices cambiaron durante merge HT, corrigiendo...")
            merged_df = merged_df.set_index(original_index)
        
        # Verificar merge
        original_count = len(teams_total_df)
        merged_count = len(merged_df)
        ht_available = merged_df['HT'].notna().sum()

        return merged_df