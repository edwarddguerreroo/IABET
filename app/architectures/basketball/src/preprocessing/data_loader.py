"""
NBA Data Loader - Nueva versión para datos de SportRadar

Modulo encargado de cargar, procesar y limpiar los datos de los partidos (Jugadores y equipos) 
desde los nuevos datasets de SportRadar que incluyen:
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

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# CommonUtils se importa dentro de la función para evitar imports circulares


logger = logging.getLogger(__name__)

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
        
    def load_data(self, use_quarters: bool = False):
        """
        Carga, valida y combina los datos de partidos y biométricos
        
        Args:
            use_quarters: Si True, carga datos por cuartos. Si False, datos totales por juego.
        
        Returns:
            tuple: (players_data, teams_data) - DataFrames procesados
        """
        logger.info("Cargando datos de NBA...")
        
        # Seleccionar datasets según si queremos cuartos o totales
        if use_quarters:
            players_path = self.players_quarters_path
            teams_path = self.teams_quarters_path
            logger.info("Cargando datos por cuartos")
        else:
            players_path = self.players_total_path
            teams_path = self.teams_total_path
            logger.info("Cargando datos totales por juego")
        
        # Cargar datasets
        players_data = pd.read_csv(players_path)
        teams_data = pd.read_csv(teams_path)
        biometrics = pd.read_csv(self.biometrics_path)
        
        # Validar datos
        self._validate_players_data(players_data)
        self._validate_teams_data(teams_data)
        self._validate_biometrics(biometrics)
        
        # Procesar datos
        players_data = self._preprocess_players_data(players_data)
        teams_data = self._preprocess_teams_data(teams_data)
        biometrics = self._preprocess_biometrics(biometrics)
        
        # Agregar columna HT (halftime) a teams_data si no existe
        if 'HT' not in teams_data.columns:
            logger.info("Agregando columna HT (halftime) a teams_total...")
            teams_quarters_data = pd.read_csv(self.teams_quarters_path)
            ht_target_df = self._generate_halftime_target(teams_quarters_data)
            teams_data = self._merge_halftime_target(teams_data, ht_target_df)
        
        # Merge de jugadores con biometrics
        players_data = self._merge_players_with_biometrics(players_data, biometrics)
        
        logger.info(f"Datos cargados exitosamente:")
        logger.info(f"    Jugadores: {players_data.shape[0]:,} filas, {players_data.shape[1]} columnas")
        logger.info(f"    Equipos: {teams_data.shape[0]:,} filas, {teams_data.shape[1]} columnas")
        
        return players_data, teams_data
    
    def load_data_with_halftime_target(self):
        """
        Carga datos de equipos con target de halftime (HT) generado desde datos de cuartos
        
        Returns:
            tuple: (players_data, teams_data_with_ht) - DataFrames procesados con target HT
        """
        logger.info("Cargando datos de NBA con target de halftime...")
        
        # Cargar datos totales de equipos
        teams_total = pd.read_csv(self.teams_total_path)
        teams_quarters = pd.read_csv(self.teams_quarters_path)
        players_data = pd.read_csv(self.players_total_path)
        biometrics = pd.read_csv(self.biometrics_path)
        
        # Validar datos
        self._validate_teams_data(teams_total)
        self._validate_teams_data(teams_quarters)
        self._validate_players_data(players_data)
        self._validate_biometrics(biometrics)
        
        # Generar target HT desde datos de cuartos
        ht_target = self._generate_halftime_target(teams_quarters)
        
        # Merge HT target con datos totales
        teams_with_ht = self._merge_halftime_target(teams_total, ht_target)
        
        # Procesar datos
        players_data = self._preprocess_players_data(players_data)
        teams_with_ht = self._preprocess_teams_data(teams_with_ht)
        biometrics = self._preprocess_biometrics(biometrics)
        
        # Merge de jugadores con biometrics
        players_data = self._merge_players_with_biometrics(players_data, biometrics)
        
        logger.info(f"Datos cargados exitosamente con target HT:")
        logger.info(f"    Jugadores: {players_data.shape[0]:,} filas, {players_data.shape[1]} columnas")
        logger.info(f"    Equipos con HT: {teams_with_ht.shape[0]:,} filas, {teams_with_ht.shape[1]} columnas")
        logger.info(f"    Target HT creado: {teams_with_ht['HT'].notna().sum():,} valores válidos")
        
        return players_data, teams_with_ht
    
    def _validate_players_data(self, df):
        """Valida el DataFrame de datos de jugadores"""
        required_columns = [
            'player_id', 'player', 'Team', 'Opp', 'Date', 'game_id',
            'points', 'rebounds', 'assists', 'is_home'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de jugadores: {missing_cols}")
            
    def _validate_teams_data(self, df):
        """Valida el DataFrame de datos de equipos"""
        required_columns = [
            'team_id', 'Team', 'Opp', 'Date', 'game_id', 
            'points', 'is_home', 'is_win'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos de equipos: {missing_cols}")
        
        # Verificar que points_against existe para calcular game_total_points
        if 'points_against' in df.columns:
            points_against_null = df['points_against'].isnull().sum()
            if points_against_null > 0:
                logger.warning(f"{points_against_null} valores nulos en columna points_against")
    
    def _validate_biometrics(self, df):
        """Valida el DataFrame de datos biométricos"""
        required_columns = ['Player', 'Height', 'Weight']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columnas faltantes en datos biométricos: {missing_cols}")
    
    def _preprocess_players_data(self, df):
        """
        Preprocesa los datos de jugadores
        
        Los datos YA VIENEN LIMPIOS desde SportRadar, así que solo:
        - Convierte fechas
        - Valida tipos de datos
        - Crea columnas de compatibilidad
        """
        df = df.copy()
        
        # Convertir fechas
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        # Los datos ya vienen con los nombres de columnas correctos desde SportRadar
        # No se necesita mapeo de columnas
        
        # Crear alias específicos para modelos
        if 'double_double' in df.columns:
            df['has_double_double'] = df['double_double']
            df['DD'] = df['double_double']
        
        # Limpiar valores no válidos
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)

        # Ordenar por jugador y fecha
        df = df.sort_values(['player', 'Date'])
        
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
        
        # Los datos ya vienen con los nombres de columnas correctos desde SportRadar
        # No se necesita mapeo de columnas
        
        # Crear game_total_points (suma de puntos de ambos equipos del juego)
        if 'game_total_points' not in df.columns:
            if 'points' in df.columns and 'points_against' in df.columns:
                df['game_total_points'] = df['points'] + df['points_against']
            else:
                logger.warning("⚠️ No se pudo crear 'game_total_points': faltan columnas points o points_against")
        
        # Crear PTS_Opp si no existe
        if 'PTS_Opp' not in df.columns and 'points_against' in df.columns:
            df['PTS_Opp'] = df['points_against']
        
        # Limpiar valores no válidos
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['Team', 'Date'])
        
        return df
    
    def _preprocess_biometrics(self, df):
        """
        Preprocesa los datos biométricos (igual que antes)
        
        - Convierte altura a pulgadas
        - Limpia y valida valores
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
        
        return df
    
    def _merge_players_with_biometrics(self, players_data, biometrics):
        """
        Combina los datos de jugadores con los datos biométricos usando búsqueda inteligente
        """
        # Verificar si los datos biométricos ya están integrados
        biometric_columns = ['Height_Inches', 'Weight', 'BMI']
        has_biometrics = all(col in players_data.columns for col in biometric_columns)
        
        if has_biometrics:
            logger.info("Datos biométricos ya están integrados")
            return players_data
        
        # Si no están integrados, hacer el merge directo (rápido)
        logger.info("Integrando datos biométricos con datos de jugadores...")
        
        # Renombrar columna en biometrics para que coincida
        biometrics_renamed = biometrics.rename(columns={'Player': 'player'})
        
        # Merge directo por player
        merged = pd.merge(
            players_data,
            biometrics_renamed[['player', 'Height_Inches', 'Weight']],
            on='player',
            how='left'
        )
        
        # Estadísticas del merge
        players_with_bio = merged['Height_Inches'].notna().sum()
        total_records = len(merged)
        
        logger.info(f"Merge biométrico completado:")
        logger.info(f"    Registros con biometrics: {players_with_bio:,}/{total_records:,}")
        
        # Verificar que no perdimos datos
        if len(merged) != len(players_data):
            logger.warning("⚠️ Se perdieron registros durante el merge biométrico")
            
        # Calcular BMI si no existe
        if 'BMI' not in merged.columns:
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
                    height_squared = merged.loc[valid_mask, 'Height_Inches'] ** 2
                    safe_height_squared = np.where(height_squared > 0, height_squared, np.nan)
                    merged.loc[valid_mask, 'BMI'] = np.divide(
                        merged.loc[valid_mask, 'Weight'] * 703, 
                        safe_height_squared, 
                        out=np.full_like(merged.loc[valid_mask, 'Weight'], np.nan),
                        where=safe_height_squared > 0
                    )
                
                bmi_valid = merged['BMI'].notna().sum()
            else:
                logger.warning("No se pueden calcular métricas BMI: faltan columnas Weight o Height_Inches")
        
        return merged
    
    def _generate_halftime_target(self, teams_quarters_df):
        """
        Genera el target de halftime (HT) sumando puntos de cuartos 1 y 2
        
        Args:
            teams_quarters_df: DataFrame con datos de cuartos por equipo
            
        Returns:
            DataFrame con target HT por equipo y partido
        """
        logger.info("Generando target de halftime desde datos de cuartos...")
        
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
        
        logger.info(f"Target HT generado: {len(ht_target):,} registros")
        logger.info(f"Rango de HT: {ht_target['HT'].min():.1f} - {ht_target['HT'].max():.1f}")
        
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
        logger.info("Haciendo merge del target HT con datos totales...")
        
        # Convertir Date a datetime en ambos DataFrames para el merge
        teams_total_df['Date'] = pd.to_datetime(teams_total_df['Date'], format='mixed')
        ht_target_df['Date'] = pd.to_datetime(ht_target_df['Date'], format='mixed')
        
        # Hacer merge por Team y Date (sin game_id ya que pueden ser diferentes)
        merged_df = pd.merge(
            teams_total_df,
            ht_target_df[['Team', 'Date', 'HT']],
            on=['Team', 'Date'],
            how='left'
        )
        
        # Verificar merge
        original_count = len(teams_total_df)
        merged_count = len(merged_df)
        ht_available = merged_df['HT'].notna().sum()
        
        return merged_df

    # MÉTODOS DE COMPATIBILIDAD PARA NO ROMPER CÓDIGO EXISTENTE
    def load_data_legacy(self):
        """
        Método de compatibilidad para mantener la interfaz anterior
        Retorna datos en el formato esperado por el código existente
        """
        players_data, teams_data = self.load_data(use_quarters=False)
        return players_data, teams_data

# Función de compatibilidad
def load_nba_data(players_total_path=None, biometrics_path=None, teams_path=None, use_quarters=False):
    """
    Función de compatibilidad para cargar datos NBA
    """
    if players_total_path is None:
        players_total_path = "app/architectures/basketball/data/players_total.csv"
    if biometrics_path is None:
        biometrics_path = "app/architectures/basketball/data/biometrics.csv"
    if teams_path is None:
        teams_path = "app/architectures/basketball/data/teams_total.csv"
    
    loader = NBADataLoader(
        players_total_path=players_total_path,
        teams_total_path=teams_path,
        biometrics_path=biometrics_path
    )
    
    return loader.load_data(use_quarters=use_quarters)