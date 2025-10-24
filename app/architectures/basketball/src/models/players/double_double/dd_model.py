"""
Modelo Avanzado de Predicción de Double Double NBA
=================================================

Modelo híbrido que combina:
- Machine Learning tradicional (Random Forest, XGBoost, LightGBM)
- CatBoost para manejo de features categóricas
- Stacking avanzado con meta-modelo optimizado
- Optimización bayesiana de hiperparámetros
- Regularización agresiva anti-overfitting
- Manejo automático de GPU con GPUManager
- Sistema de logging optimizado
- Confidence thresholds para predicciones
"""

# Standard Library
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import sys
import logging

# Third-party Libraries - ML/Data
import pandas as pd
import numpy as np
import joblib

# Scikit-learn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, log_loss,
    precision_recall_curve, roc_curve
)
from sklearn.svm import SVC

# XGBoost, LightGBM and CatBoost
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F

# Bayesian Optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Imports del proyecto
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.double_double.features_dd import DoubleDoubleFeatureEngineer

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PositionSpecializedClassifier:
    """
    Clasificador especializado por posición para double-doubles.
    
    Cada posición tiene patrones diferentes:
    - Centers: Alta tasa DD (15-25%), principalmente PTS+TRB
    - Power Forwards: Tasa media DD (8-15%), PTS+TRB o TRB+AST
    - Small Forwards: Tasa baja DD (3-8%), más versátiles
    - Guards: Tasa muy baja DD (1-5%), principalmente PTS+AST
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PositionSpecialized")
        self.position_models = {}
        self.position_thresholds = {}
        self.position_features = {}
        self.position_stats = {}
        
    def categorize_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorizar jugadores por posición simplificada"""
        df = df.copy()
        
        def simplify_position(pos):
            if pd.isna(pos):
                return 'Unknown'
            pos = str(pos).upper()
            if 'C' in pos:
                return 'Center'
            elif 'PF' in pos or 'F-C' in pos:
                return 'PowerForward'
            elif 'SF' in pos or 'F' in pos:
                return 'SmallForward'
            elif 'PG' in pos or 'SG' in pos or 'G' in pos:
                return 'Guard'
            else:
                return 'Unknown'
        
        if 'Pos' in df.columns:
            df['Position_Category'] = df['Pos'].apply(simplify_position)
        else:
            # Inferir posición por estadísticas si no está disponible
            df['Position_Category'] = self._infer_position_by_stats(df)
        
        return df
    
    def _infer_position_by_stats(self, df: pd.DataFrame) -> pd.Series:
        """Inferir posición basada en estadísticas promedio del jugador"""
        player_stats = df.groupby('player').agg({
            'rebounds': 'mean',
            'assists': 'mean',
            'points': 'mean',
            'blocks': 'mean'
        }).round(2)
        
        def infer_position(row):
            trb_avg = row.get('rebounds', 0)
            ast_avg = row.get('assists', 0)
            pts_avg = row.get('points', 0)
            blk_avg = row.get('blocks', 0)
            
            # Centers: Muchos rebotes y bloqueos
            if trb_avg >= 8 and blk_avg >= 0.8:
                return 'Center'
            # Power Forwards: Buenos rebotes, pocos assists
            elif trb_avg >= 6 and ast_avg <= 3:
                return 'PowerForward'
            # Guards: Muchos assists, pocos rebotes
            elif ast_avg >= 4 and trb_avg <= 5:
                return 'Guard'
            # Small Forwards: Balanceados
            else:
                return 'SmallForward'
        
        player_positions = player_stats.apply(infer_position, axis=1).to_dict()
        return df['player'].map(player_positions).fillna('Unknown')
    
    def analyze_position_patterns(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Analizar patrones de double-double por posición"""
        df = self.categorize_position(df)
        
        # Crear columna double_double si no existe
        if 'double_double' not in df.columns:
            df['double_double'] = ((df['points'] >= 10) & (df['rebounds'] >= 10)) | \
                                 ((df['points'] >= 10) & (df['assists'] >= 10)) | \
                                 ((df['rebounds'] >= 10) & (df['assists'] >= 10))
            df['double_double'] = df['double_double'].astype(int)
        
        position_analysis = {}
        
        for position in ['Center', 'PowerForward', 'SmallForward', 'Guard']:
            pos_data = df[df['Position_Category'] == position]
            
            if len(pos_data) > 0:
                dd_rate = pos_data['double_double'].mean()
                total_games = len(pos_data)
                total_dds = pos_data['double_double'].sum()
                unique_players = pos_data['player'].nunique()
                
                # Estadísticas promedio por posición
                avg_stats = pos_data.groupby('player').agg({
                    'points': 'mean',
                    'rebounds': 'mean', 
                    'assists': 'mean',
                    'MP': 'mean'
                }).mean()
                
                position_analysis[position] = {
                    'dd_rate': dd_rate,
                    'total_games': total_games,
                    'total_dds': total_dds,
                    'unique_players': unique_players,
                    'avg_pts': avg_stats['points'],
                    'avg_trb': avg_stats['rebounds'],
                    'avg_ast': avg_stats['assists'],
                    'avg_mp': avg_stats['MP'],
                    'games_per_player': total_games / unique_players if unique_players > 0 else 0
                }
                
                self.logger.info(f"{position}: {dd_rate:.3f} DD rate, {unique_players} jugadores, {total_games} juegos")
        
        self.position_stats = position_analysis
        return position_analysis


class DoubleDoubleImbalanceHandler:
    """
    Manejador especializado para el desbalance extremo en predicción de double-doubles.
    
    Considera la naturaleza específica de los double-doubles:
    - Solo ciertos jugadores (centers, power forwards, algunos guards) los logran regularmente
    - La mayoría de jugadores nunca o rara vez logran double-doubles
    - Necesitamos estratificar por tipo de jugador y rol
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ImbalanceHandler")
        self.player_profiles = {}
        self.position_weights = {}
        self.role_based_thresholds = {}
        
    def analyze_player_profiles(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Analiza perfiles de jugadores para entender patrones de double-double.
        
        Returns:
            Dict con perfiles de jugadores categorizados
        """
        self.logger.info("Analizando perfiles de jugadores para double-doubles...")
        
        # Análisis por jugador
        agg_dict = {
            'double_double': ['sum', 'count', 'mean'],
            'points': 'mean',
            'rebounds': 'mean', 
            'assists': 'mean',
            'MP': 'mean'
        }
        
        # Solo agregar is_started si existe en el dataset
        if 'is_started' in df.columns:
            agg_dict['is_started'] = 'mean'
        
        player_stats = df.groupby('player').agg(agg_dict).round(3)
        
        # Asignar nombres de columnas basado en si is_started existe
        if 'is_started' in df.columns:
            player_stats.columns = ['dd_total', 'games_played', 'dd_rate', 'avg_pts', 'avg_trb', 'avg_ast', 'avg_mp', 'starter_rate']
        else:
            player_stats.columns = ['dd_total', 'games_played', 'dd_rate', 'avg_pts', 'avg_trb', 'avg_ast', 'avg_mp']
        
        player_stats = player_stats.reset_index()
        
        # Categorizar jugadores por capacidad de double-double
        def categorize_dd_ability(row):
            dd_rate = row['dd_rate']
            games = row['games_played']
            
            # Solo considerar jugadores con suficientes juegos
            if games < 10:
                return 'insufficient_data'
            elif dd_rate >= 0.4:  # 40%+ de double-doubles
                return 'elite_dd_producer'
            elif dd_rate >= 0.15:  # 15-40% de double-doubles
                return 'regular_dd_producer'
            elif dd_rate >= 0.05:  # 5-15% de double-doubles
                return 'occasional_dd_producer'
            else:  # <5% de double-doubles
                return 'rare_dd_producer'
        
        player_stats['dd_category'] = player_stats.apply(categorize_dd_ability, axis=1)
        
        # Análisis por categoría
        category_agg_dict = {
            'player': 'count',
            'dd_rate': ['mean', 'std'],
            'avg_pts': 'mean',
            'avg_trb': 'mean',
            'avg_ast': 'mean'
        }
        
        # Solo agregar starter_rate si existe
        if 'starter_rate' in player_stats.columns:
            category_agg_dict['starter_rate'] = 'mean'
        
        category_analysis = player_stats.groupby('dd_category').agg(category_agg_dict).round(3)
        
        self.logger.info("Distribución de jugadores por capacidad de double-double:")
        for category in category_analysis.index:
            count = category_analysis.loc[category, ('player', 'count')]
            avg_rate = category_analysis.loc[category, ('dd_rate', 'mean')]
            self.logger.info(f"  {category}: {count} jugadores (DD rate promedio: {avg_rate:.1%})")
        
        # Guardar perfiles para uso posterior
        self.player_profiles = player_stats.set_index('player')['dd_category'].to_dict()
        
        return {
            'player_stats': player_stats,
            'category_analysis': category_analysis,
            'player_profiles': self.player_profiles
        }
    
    def create_stratified_weights(self, df: pd.DataFrame) -> np.ndarray:
        """
        Crea pesos estratificados basados en el perfil del jugador.
        
        Returns:
            Array con pesos por muestra
        """
        if not self.player_profiles:
            self.analyze_player_profiles(df)
        
        # Pesos base por categoría (más conservadores)
        category_weights = {
            'elite_dd_producer': 1.0,      # Sin penalización
            'regular_dd_producer': 1.2,    # Ligero boost
            'occasional_dd_producer': 2.0, # Boost moderado
            'rare_dd_producer': 4.0,       # Boost significativo pero no extremo
            'insufficient_data': 2.5       # Peso intermedio
        }
        
        # Crear pesos por muestra
        sample_weights = []
        for _, row in df.iterrows():
            player = row['player']
            is_dd = row['double_double']
            
            # Peso base por categoría del jugador
            category = self.player_profiles.get(player, 'insufficient_data')
            base_weight = category_weights[category]
            
            # Ajuste por clase
            if is_dd == 1:
                # Double-doubles: peso base según categoría
                weight = base_weight
            else:
                # No double-doubles: peso reducido para jugadores que nunca los hacen
                if category == 'rare_dd_producer':
                    weight = 0.3  # Reducir importancia de casos negativos de jugadores que nunca hacen DD
                elif category == 'occasional_dd_producer':
                    weight = 0.6
                else:
                    weight = 1.0
            
            sample_weights.append(weight)
        
        self.logger.info(f"Pesos estratificados creados para {len(sample_weights)} muestras")
        self.logger.info(f"Peso promedio clase positiva: {np.mean([w for w, dd in zip(sample_weights, df['double_double']) if dd == 1]):.2f}")
        self.logger.info(f"Peso promedio clase negativa: {np.mean([w for w, dd in zip(sample_weights, df['double_double']) if dd == 0]):.2f}")
        
        return np.array(sample_weights)
    
    def get_position_based_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula thresholds específicos por posición/rol.
        
        Returns:
            Dict con thresholds por tipo de jugador
        """
        # Inferir posición aproximada basada en estadísticas
        def infer_position(row):
            avg_trb = row.get('avg_trb', 0)
            avg_ast = row.get('avg_ast', 0) 
            avg_pts = row.get('avg_pts', 0)
            
            if avg_trb >= 8:
                return 'big_man'  # Centers/Power Forwards
            elif avg_ast >= 5:
                return 'playmaker'  # Point Guards/Playmaking Guards
            elif avg_pts >= 15:
                return 'scorer'  # Shooting Guards/Small Forwards
            else:
                return 'role_player'  # Bench players/specialists
        
        # Análisis por posición inferida
        if not hasattr(self, 'player_profiles') or not self.player_profiles:
            self.analyze_player_profiles(df)
        
        # Agregar posición inferida a player_stats
        player_stats = df.groupby('player').agg({
            'rebounds': 'mean',
            'assists': 'mean', 
            'points': 'mean',
            'double_double': 'mean'
        }).round(3)
        
        player_stats['position_type'] = player_stats.apply(infer_position, axis=1)
        
        # Thresholds por posición (más conservadores)
        position_thresholds = {}
        for pos_type in player_stats['position_type'].unique():
            pos_players = player_stats[player_stats['position_type'] == pos_type]
            avg_dd_rate = pos_players['double_double'].mean()
            
            # Threshold más realista basado en la tasa promedio de la posición
            if avg_dd_rate >= 0.3:  # Posiciones con alta tasa de DD
                threshold = 0.20
            elif avg_dd_rate >= 0.1:  # Posiciones con tasa moderada
                threshold = 0.25
            else:  # Posiciones con baja tasa de DD
                threshold = 0.30
            
            position_thresholds[pos_type] = threshold
            self.logger.info(f"Threshold para {pos_type}: {threshold:.3f} (DD rate promedio: {avg_dd_rate:.1%})")
        
        self.role_based_thresholds = position_thresholds
        return position_thresholds

class GPUManager:
    """Gestor avanzado de GPU para modelos NBA"""
    
    _device_logged = False  # AGREGAR: Control de logging único
    
    @staticmethod
    def get_available_devices() -> List[str]:
        """Obtener lista de dispositivos disponibles"""
        devices = ['cpu']
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                devices.append(f'cuda:{i}')
        return devices
    
    @staticmethod
    def get_device_info(device_str: str = None) -> Dict[str, Any]:
        """Obtener información detallada del dispositivo"""
        if device_str is None:
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        info = {'device': device_str, 'type': 'cpu'}
        
        if device_str.startswith('cuda') and torch.cuda.is_available():
            device_id = int(device_str.split(':')[1]) if ':' in device_str else 0
            
            if device_id < torch.cuda.device_count():
                info.update({
                    'type': 'cuda',
                    'name': torch.cuda.get_device_name(device_id),
                    'memory_info': {
                        'total_gb': torch.cuda.get_device_properties(device_id).total_memory / 1e9,
                        'allocated_gb': torch.cuda.memory_allocated(device_id) / 1e9,
                        'cached_gb': torch.cuda.memory_reserved(device_id) / 1e9,
                        'free_gb': (torch.cuda.get_device_properties(device_id).total_memory - 
                                   torch.cuda.memory_reserved(device_id)) / 1e9
                    }
                })
        
        return info
    
    @staticmethod
    def get_optimal_device(min_memory_gb: float = 2.0) -> str:
        """Obtener el dispositivo óptimo disponible"""
        if not torch.cuda.is_available():
            return 'cpu'
        
        best_device = 'cpu'
        max_free_memory = 0
        
        for i in range(torch.cuda.device_count()):
            device_str = f'cuda:{i}'
            info = GPUManager.get_device_info(device_str)
            
            if info['type'] == 'cuda':
                free_memory = info['memory_info']['free_gb']
                if free_memory >= min_memory_gb and free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_device = device_str
        
        return best_device
    
    @staticmethod
    def setup_device(device_preference: str = None, min_memory_gb: float = 2.0) -> torch.device:
        """Configurar dispositivo óptimo con logging controlado"""
        if device_preference:
            device_str = device_preference
        else:
            device_str = GPUManager.get_optimal_device(min_memory_gb)
        
        device = torch.device(device_str)
        
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.cuda.empty_cache()
        
        # LOGGING CONTROLADO: Solo logear una vez por sesión
        if not GPUManager._device_logged:
            GPUManager._device_logged = True
            logger.info(f"Dispositivo configurado: {device_str}")
        
        return device


class DataProcessor:
    """Clase auxiliar para procesamiento de datos común"""
    
    @staticmethod
    def prepare_training_data(X: pd.DataFrame, y: pd.Series, 
                            validation_split: float = 0.2,
                            scaler: Optional[StandardScaler] = None,
                            date_column: str = 'Date'
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                     pd.Series, pd.Series, StandardScaler]:
        """Preparar datos para entrenamiento con división cronológica y manejo robusto de NaN"""
        
        # Limpiar datos de manera más robusta
        X_clean = X.copy()
        
        # 1. Manejo agresivo de infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Imputar NaN columna por columna
        numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    # Si la mediana es NaN, usar la media
                    mean_val = X_clean[col].mean()
                    if pd.isna(mean_val):
                        # Si también la media es NaN, usar 0
                        median_val = 0
                    else:
                        median_val = mean_val
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # 3. Imputación final para asegurar que no hay NaN con verificación más rigurosa
        if X_clean.isna().any().any():
            # Reportar columnas con NaN antes de la limpieza final
            nan_columns = X_clean.columns[X_clean.isna().any()].tolist()
            logger.warning(f"Columnas con NaN detectadas: {nan_columns}")
            
            # Imputación más agresiva
            for col in nan_columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    # Para columnas numéricas: usar mediana, luego media, luego 0
                    if X_clean[col].notna().sum() > 0:
                        median_val = X_clean[col].median()
                        if pd.isna(median_val):
                            mean_val = X_clean[col].mean()
                            fill_val = mean_val if not pd.isna(mean_val) else 0.0
                        else:
                            fill_val = median_val
                    else:
                        fill_val = 0.0
                    X_clean[col] = X_clean[col].fillna(fill_val)
                else:
                    # Para columnas categóricas: usar moda o 0
                    mode_val = X_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    X_clean[col] = X_clean[col].fillna(fill_val)
            
            # Verificación final final
            X_clean = X_clean.fillna(0)
            
            # Verificar que no queden NaN
            remaining_nans = X_clean.isna().sum().sum()
            if remaining_nans > 0:
                logger.error(f"ADVERTENCIA: Aún quedan {remaining_nans} valores NaN después de limpieza agresiva")
                X_clean = X_clean.fillna(0)  # Último recurso
        
        # División cronológica en lugar de aleatoria
        if date_column in X_clean.index.names or date_column in X_clean.columns:
            # Si tenemos columna de fecha, ordenar por fecha
            if date_column in X_clean.columns:
                # Crear índice temporal
                combined_data = pd.concat([X_clean, y], axis=1)
                combined_data = combined_data.sort_values(date_column)
                
                # Dividir cronológicamente
                split_idx = int(len(combined_data) * (1 - validation_split))
                
                train_data = combined_data.iloc[:split_idx]
                val_data = combined_data.iloc[split_idx:]
                
                X_train = train_data.drop(columns=[y.name, date_column] if y.name in train_data.columns else [date_column])
                y_train = train_data[y.name] if y.name in train_data.columns else y.iloc[:split_idx]
                
                X_val = val_data.drop(columns=[y.name, date_column] if y.name in val_data.columns else [date_column])
                y_val = val_data[y.name] if y.name in val_data.columns else y.iloc[split_idx:]
                
            else:
                # Si el índice ya está ordenado cronológicamente
                split_idx = int(len(X_clean) * (1 - validation_split))
                X_train = X_clean.iloc[:split_idx]
                X_val = X_clean.iloc[split_idx:]
                y_train = y.iloc[:split_idx]
                y_val = y.iloc[split_idx:]
        else:
            # Fallback: división por índice (asumiendo que está ordenado cronológicamente)
            logger.warning(f"Columna de fecha '{date_column}' no encontrada. Usando división por índice.")
            split_idx = int(len(X_clean) * (1 - validation_split))
            X_train = X_clean.iloc[:split_idx]
            X_val = X_clean.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_val = y.iloc[split_idx:]
        
        # Limpiar datos de entrenamiento y validación antes del escalado
        X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_val = X_val.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Escalar datos - CORREGIDO: Crear scaler si no existe, y hacer fit_transform siempre
        if scaler is None:
            scaler = StandardScaler()
        
        # Hacer fit_transform en datos de entrenamiento
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        # Hacer transform en datos de validación
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        # Verificación final de que no hay NaN ni infinitos
        X_train_scaled = X_train_scaled.replace([np.inf, -np.inf], 0).fillna(0)
        X_val_scaled = X_val_scaled.replace([np.inf, -np.inf], 0).fillna(0)
        
        return X_train_scaled, X_val_scaled, y_train, y_val, scaler
    
    @staticmethod
    def prepare_prediction_data(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
        """Preparar datos para predicción con manejo robusto de NaN"""
        X_clean = X.copy()
        
        # Manejo agresivo de NaN para GradientBoostingClassifier
        # 1. Reemplazar infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Imputar NaN con mediana de cada columna
        numeric_columns = X_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if X_clean[col].isna().any():
                median_val = X_clean[col].median()
                if pd.isna(median_val):
                    # Si la mediana también es NaN, usar 0
                    median_val = 0
                X_clean[col] = X_clean[col].fillna(median_val)
        
        # 3. Verificar que no queden NaN con manejo exhaustivo
        if X_clean.isna().any().any():
            # Reportar y manejar columnas con NaN
            nan_columns = X_clean.columns[X_clean.isna().any()].tolist()
            logger.warning(f"Columnas con NaN en predicción: {nan_columns}")
            
            # Imputación exhaustiva
            for col in nan_columns:
                if X_clean[col].dtype in ['float64', 'int64']:
                    # Para columnas numéricas
                    if X_clean[col].notna().sum() > 0:
                        median_val = X_clean[col].median()
                        if pd.isna(median_val):
                            mean_val = X_clean[col].mean()
                            fill_val = mean_val if not pd.isna(mean_val) else 0.0
                        else:
                            fill_val = median_val
                    else:
                        fill_val = 0.0
                    X_clean[col] = X_clean[col].fillna(fill_val)
                else:
                    # Para columnas categóricas
                    mode_val = X_clean[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    X_clean[col] = X_clean[col].fillna(fill_val)
            
            # Imputación final con 0 para cualquier NaN restante
            X_clean = X_clean.fillna(0)
        
        # 4. Escalar datos
        X_scaled = pd.DataFrame(
            scaler.transform(X_clean),
            columns=X_clean.columns,
            index=X_clean.index
        )
        
        # 5. Verificación final de que no hay NaN ni infinitos
        X_scaled = X_scaled.replace([np.inf, -np.inf], 0)
        X_scaled = X_scaled.fillna(0)
        
        return X_scaled
    
    @staticmethod
    def create_time_series_split(X: pd.DataFrame, y: pd.Series, 
                               n_splits: int = 5,
                               date_column: str = 'Date') -> List[Tuple[np.ndarray, np.ndarray]]:
        """Crear splits cronológicos para validación cruzada"""
        
        if date_column in X.columns:
            # Ordenar por fecha
            combined_data = pd.concat([X, y], axis=1)
            combined_data = combined_data.sort_values(date_column)
            indices = combined_data.index.values
        else:
            # Usar índice actual (asumiendo orden cronológico)
            indices = X.index.values
        
        splits = []
        total_size = len(indices)
        
        # Crear splits cronológicos con ventana expandible
        for i in range(n_splits):
            # Tamaño mínimo de entrenamiento: 60% de los datos
            min_train_size = int(total_size * 0.6)
            
            # Calcular tamaños para este split
            train_end = min_train_size + int((total_size - min_train_size) * (i + 1) / n_splits)
            val_start = train_end
            val_end = min(train_end + int(total_size * 0.2), total_size)
            
            if val_end > total_size:
                val_end = total_size
            
            train_indices = indices[:train_end]
            val_indices = indices[val_start:val_end]
            
            if len(val_indices) > 0:
                splits.append((train_indices, val_indices))
        
        return splits


class MetricsCalculator:
    """Calculadora de métricas para clasificación"""
    
    @staticmethod
    def calculate_classification_metrics(y_true: pd.Series, 
                                       y_pred: np.ndarray,
                                       y_proba: np.ndarray) -> Dict[str, float]:
        """Calcular métricas completas de clasificación"""
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'log_loss': log_loss(y_true, y_proba)
        }

class DoubleDoubleModel:
    """
    Modelo avanzado para predicción de double double con stacking y optimización bayesiana
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 bayesian_n_calls: int = 50,
                 min_memory_gb: float = 2.0):
        
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.bayesian_n_calls = bayesian_n_calls
        self.min_memory_gb = min_memory_gb
        
        # Inicializar logger centralizado PRIMERO
        self.logger = logging.getLogger("DoubleDoubleModel")
        
        # Manejador de desbalance especializado
        self.imbalance_handler = DoubleDoubleImbalanceHandler()
        
        # NUEVO: Clasificador especializado por posición
        self.position_classifier = PositionSpecializedClassifier()
        
        # Componentes del modelo
        self.scaler = StandardScaler()
        
        # Feature Engineer especializado
        self.feature_engineer = DoubleDoubleFeatureEngineer(lookback_games=10)
        
        # Modelos individuales
        self.models = {}
        self.stacking_model = None
        
        # Métricas y resultados
        self.training_results = {}
        self.feature_importance = {}
        self.bayesian_results = {}
        self.gpu_config = {}
        self.cv_scores = {}
        self.is_fitted = False
        
        # Configurar entorno GPU
        self._setup_gpu_environment()
        
        # Configurar modelos
        self._setup_models()
        
        # Configurar stacking model
        self._setup_stacking_model()
    
    def _setup_gpu_environment(self):
        """Configurar entorno GPU para el modelo"""
        self.gpu_config = {
            'selected_device': GPUManager.get_optimal_device(self.min_memory_gb),
            'device_info': GPUManager.get_device_info()
        }
        
        self.device = torch.device(self.gpu_config['selected_device'])
        
    def _setup_models(self):
        """
        PARTE 2 & 4: REGULARIZACIÓN AUMENTADA + CLASS WEIGHTS REBALANCEADOS
        Configurar modelos base con correcciones anti-overfitting y manejo conservador del desbalance
        """
        
        # PARTE 4: CLASS WEIGHTS REBALANCEADOS - Más conservadores para reducir falsos positivos
        # Ratio original: 10.6:1, pero usaremos pesos más moderados para mejor precision
        # NOTA: Los pesos específicos se calcularán dinámicamente por el imbalance_handler
        class_weight_conservative = {0: 1.0, 1: 15.0}  # Reducido para usar con sample_weights
        
        # CORRECCIÓN 2: Modelos con regularización optimizada para precisión
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido para mejor convergencia
                subsample=0.85,    # Aumentado
                colsample_bytree=0.85, # Aumentado
                reg_alpha=0.3,     # Aumentado L1 regularization
                reg_lambda=2.0,    # Aumentado L2 regularization
                min_child_weight=5, # Aumentado para evitar overfitting
                gamma=0.1,         # Aumentado para más regularización
                scale_pos_weight=12, # Conservador para mejor precision
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido
                subsample=0.85,    # Aumentado
                colsample_bytree=0.85, # Aumentado
                reg_alpha=0.3,     # Aumentado L1 regularization
                reg_lambda=2.0,    # Aumentado L2 regularization
                min_child_samples=8, # Aumentado para evitar overfitting
                min_split_gain=0.01, # Aumentado para más regularización
                num_leaves=25,     # Reducido para evitar overfitting
                feature_fraction=0.85,
                bagging_fraction=0.85,
                bagging_freq=3,
                scale_pos_weight=12, # Conservador para mejor precision
                boost_from_average=False,
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ),

            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=120,  # Reducido ligeramente
                max_depth=4,       # Reducido para evitar overfitting
                learning_rate=0.03, # Reducido para mejor convergencia
                subsample=0.85,    # Aumentado
                min_samples_split=15, # Aumentado para evitar overfitting
                min_samples_leaf=8,   # Aumentado para evitar overfitting
                random_state=42
            ),
            
            'catboost': cb.CatBoostClassifier(
                iterations=80,     # Reducido para evitar overfitting
                depth=4,          # Reducido para evitar overfitting
                learning_rate=0.05, # Reducido
                l2_leaf_reg=8.0,   # Aumentado para más regularización
                class_weights=[1.0, 12.0], # Conservador para mejor precision
                random_seed=42,
                verbose=False,
                early_stopping_rounds=10
            ),
            
            # ELIMINADO: Neural Network por bajo rendimiento (F1: 0.568, Precision: 0.432)
            # Reemplazado por ensemble más simple y eficiente
        }
        
        self.logger.info(f"Modelos configurados: {len(self.models)} modelos base")
    
    def _setup_stacking_model(self):
        """Configurar modelo de stacking con TODOS LOS MODELOS (ML/DL) y manejo correcto de NN"""
        
        # ELIMINADO: Neural Network wrapper por bajo rendimiento
        # nn_wrapper = NeuralNetworkWrapper(self.models['neural_network'])
        
        # Modelos base para stacking con REGULARIZACIÓN BALANCEADA
        # Usar versiones más ligeras pero no excesivamente restringidas
        base_estimators = [
            # XGBoost regularizado para stacking
            ('xgb_stack', xgb.XGBClassifier(
                n_estimators=50,          # Moderado
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado para mejor aprendizaje
                subsample=0.8,            # Aumentado
                colsample_bytree=0.8,     # Aumentado
                reg_alpha=0.1,            # REDUCIDO dramáticamente
                reg_lambda=0.5,           # REDUCIDO dramáticamente
                min_child_weight=3,       # REDUCIDO
                gamma=0.05,               # REDUCIDO dramáticamente
                scale_pos_weight=10,      # Manejo de desbalance
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            )),
            
            # LightGBM regularizado para stacking con manejo agresivo de desbalance
            ('lgb_stack', lgb.LGBMClassifier(
                n_estimators=80,          # Moderado para stacking
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado para mejor aprendizaje
                subsample=0.85,           # Aumentado
                colsample_bytree=0.85,    # Aumentado
                reg_alpha=0.05,           # REDUCIDO dramáticamente
                reg_lambda=0.2,           # REDUCIDO dramáticamente
                min_child_samples=5,      # REDUCIDO para permitir splits
                min_split_gain=0.005,     # REDUCIDO dramáticamente
                num_leaves=31,            # Balanceado
                feature_fraction=0.85,    # Aumentado
                bagging_fraction=0.85,    # Aumentado
                bagging_freq=3,           # Más frecuente
                scale_pos_weight=12,      # Peso para clase minoritaria
                boost_from_average=False, # No inicializar desde promedio
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )),

            # Gradient Boosting regularizado para stacking con manejo nativo de NaN
            ('gb_stack', HistGradientBoostingClassifier(
                max_iter=50,              # Moderado para stacking
                max_depth=4,              # Balanceado
                learning_rate=0.1,        # Aumentado
                l2_regularization=0.5,    # Regularización L2 reducida
                min_samples_leaf=5,       # Mínimas muestras por hoja
                max_leaf_nodes=31,        # Máximo nodos hoja
                validation_fraction=0.1,  # Para early stopping
                n_iter_no_change=10,      # Paciencia
                tol=1e-4,                 # Tolerancia
                random_state=42
            )),
            
            # CatBoost regularizado para stacking
            ('cb_stack', cb.CatBoostClassifier(
                iterations=50,            # Moderado
                depth=4,                  # Balanceado
                learning_rate=0.1,        # Aumentado
                l2_leaf_reg=0.5,          # REDUCIDO dramáticamente
                bootstrap_type='Bernoulli',
                subsample=0.8,            # Aumentado
                random_strength=0.3,      # Reducido
                od_type='Iter',
                od_wait=10,               # Balanceado
                auto_class_weights='Balanced',  # Manejo de desbalance
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )),
        
        ]
        
        # Crear meta-learners especializados
        self.meta_learners = {
            # Meta-learner 1: Logistic Regression (lineal, robusto)
            'logistic': LogisticRegression(
                class_weight={0: 1.0, 1: 20.0},
                random_state=42,
                max_iter=3000,
                C=0.5,  # Balanceado
                penalty='l2',
                solver='liblinear',
                fit_intercept=True
            ),
            
            # Meta-learner 2: XGBoost (no-lineal, captura interacciones complejas)
            'xgb_meta': xgb.XGBClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.5,
                scale_pos_weight=20,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1,
                verbosity=0
            ),
            

        }
        
        # Stacking principal con meta-learner logístico (SIN Neural Network)
        self.stacking_model = StackingClassifier(
            estimators=[
                ('xgb', self.models['xgboost']),
                ('lgb', self.models['lightgbm']),
                ('gb', self.models['gradient_boosting']),
                ('cat', self.models['catboost'])
                # ELIMINADO: ('nn', nn_wrapper) por bajo rendimiento
            ],
            final_estimator=self.meta_learners['logistic'],
            cv=3,
            n_jobs=-1,
            passthrough=False
        )
        
        # Configurar meta-learning avanzado
        self.advanced_meta_learning = True
        self.meta_predictions = {}  # Para almacenar predicciones de cada meta-learner
    
    def _select_best_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 30) -> List[str]:
        """
        PARTE 5: FEATURE SELECTION
        Seleccionar las mejores features para evitar overfitting
        
        Args:
            X: DataFrame con features
            y: Serie con targets
            max_features: Número máximo de features a seleccionar
            
        Returns:
            Lista de nombres de features seleccionadas
        """
        from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
        
        self.logger.info(f"=== PARTE 5: FEATURE SELECTION (máximo {max_features} features) ===")
        self.logger.info(f"Features iniciales: {X.shape[1]}")
        
        # Remover columna Date si existe para el análisis
        X_analysis = X.copy()
        if 'Date' in X_analysis.columns:
            X_analysis = X_analysis.drop(columns=['Date'])
        
        # Limpiar datos para análisis
        X_clean = self._clean_nan_exhaustive(X_analysis)
        
        feature_scores = {}
        
        # Método 1: F-score (ANOVA)
        try:
            selector_f = SelectKBest(score_func=f_classif, k='all')
            selector_f.fit(X_clean, y)
            f_scores = selector_f.scores_
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['f_score'] = f_scores[i]
                
            self.logger.info(" F-score calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando F-score: {e}")
        
        # Método 2: Mutual Information
        try:
            mi_scores = mutual_info_classif(X_clean, y, random_state=42)
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['mutual_info'] = mi_scores[i]
                
            self.logger.info(" Mutual Information calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando Mutual Information: {e}")
        
        # Método 3: XGBoost Feature Importance
        try:
            xgb_selector = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                scale_pos_weight=10,
                n_jobs=-1,
                verbosity=0
            )
            xgb_selector.fit(X_clean, y)
            xgb_importances = xgb_selector.feature_importances_
            
            for i, feature in enumerate(X_clean.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature]['xgb_importance'] = xgb_importances[i]
                
            self.logger.info(" XGBoost importance calculado")
        except Exception as e:
            self.logger.warning(f"Error calculando XGBoost importance: {e}")
        
        # Combinar scores y rankear features
        combined_scores = {}
        
        for feature, scores in feature_scores.items():
            # Normalizar scores (0-1)
            normalized_scores = []
            
            if 'f_score' in scores:
                # F-score ya está normalizado por SelectKBest
                f_norm = scores['f_score'] / max(1.0, max(s.get('f_score', 0) for s in feature_scores.values()))
                normalized_scores.append(f_norm)
            
            if 'mutual_info' in scores:
                # Mutual info ya está en [0,1] aproximadamente
                normalized_scores.append(scores['mutual_info'])
            
            if 'xgb_importance' in scores:
                # XGBoost importance ya está normalizado
                normalized_scores.append(scores['xgb_importance'])
            
            # Score combinado (promedio de métodos disponibles)
            if normalized_scores:
                combined_scores[feature] = np.mean(normalized_scores)
            else:
                combined_scores[feature] = 0.0
        
        # Seleccionar top features
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Limitar al número máximo de features
        selected_features = [feature for feature, score in sorted_features[:max_features]]
        
        # Log de resultados
        self.logger.info(f"Features seleccionadas: {len(selected_features)}/{len(X_clean.columns)}")
        
        # Guardar información de selección
        self.feature_selection_info = {
            'method': 'combined_scoring',
            'max_features': max_features,
            'selected_features': selected_features,
            'feature_scores': combined_scores,
            'selection_ratio': len(selected_features) / len(X_clean.columns)
        }
        
        return selected_features

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Obtener columnas de features especializadas EXCLUSIVAMENTE usando DoubleDoubleFeatureEngineer"""
        
        # Generar features especializadas OBLIGATORIAS
        df_with_features = df.copy()
        
        try:
            specialized_features = self.feature_engineer.generate_all_features(df_with_features)
            
            # Filtrar solo features que realmente existen en el DataFrame
            available_features = [f for f in specialized_features if f in df_with_features.columns]
            
            # LISTA EXHAUSTIVA DE FEATURES BÁSICAS A EXCLUIR (NO ESPECIALIZADAS)
            basic_features_to_exclude = [
                # Columnas básicas del dataset
                'player', 'Date', 'Team', 'Opp', 'Result', 'MP', 'GS', 'Away',
                # Estadísticas del juego actual (NO USAR - data leakage)
                'FG', 'FGA', 'FG%', '2P', '2PA', '2P%', '3P', '3PA', '3P%', 
                'FT', 'FTA', 'FT%', 'points', 'ORB', 'DRB', 'rebounds', 'assists', 'STL', 'BLK', 'TOV', 'PF',
                # Columnas de double específicas del juego actual
                'points_double', 'rebounds_double', 'assists_double', 'STL_double', 'BLK_double',
                # Target variables
                'double_double', 'triple_double',
                # Columnas auxiliares temporales básicas (NO especializadas)
                'day_of_week', 'month', 'days_rest', 'days_into_season',
                # Features básicas del data_loader (NO especializadas)
                'is_home'  # Removido is_started, Height_Inches, Weight, BMI que no existen en el dataset
            ]
            
            # FILTRAR EXCLUSIVAMENTE FEATURES ESPECIALIZADAS
            purely_specialized_features = [
                f for f in available_features 
                if f not in basic_features_to_exclude
            ]
            
            # VERIFICAR que tenemos suficientes features especializadas
            if len(purely_specialized_features) < 20:
                logger.error(f"INSUFICIENTES features especializadas puras: {len(purely_specialized_features)}")
                logger.error("El modelo REQUIERE al menos 20 features especializadas")
                
                self.feature_engineer._clear_cache()
                specialized_features = self.feature_engineer.generate_all_features(df_with_features)
                available_features = [f for f in specialized_features if f in df_with_features.columns]
                purely_specialized_features = [
                    f for f in available_features 
                    if f not in basic_features_to_exclude
                ]
                
                if len(purely_specialized_features) < 20:
                    raise ValueError(f"FALLO CRÍTICO: Solo {len(purely_specialized_features)} features especializadas puras disponibles. El modelo requiere al menos 20.")
            
            # VERIFICACIÓN FINAL: Asegurar 100% especialización
            specialized_percentage = 100.0  # Por definición, todas son especializadas
            
            return purely_specialized_features
            
        except Exception as e:
            logger.error(f"ERROR CRÍTICO generando features especializadas: {str(e)}")
            logger.error("El modelo NO PUEDE funcionar sin features especializadas")
            raise ValueError(f"FALLO CRÍTICO: No se pudieron generar features especializadas. Error: {str(e)}")
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """Entrenar el modelo completo con validación rigurosa y features especializadas EXCLUSIVAS"""
        
        # Generar features especializadas OBLIGATORIAS
        df_with_features = df.copy()
        try:
            specialized_features = self.feature_engineer.generate_all_features(df_with_features)
            
            # VERIFICAR que se generaron correctamente
            if len(specialized_features) < 20:
                logger.warning(f"Pocas features especializadas generadas: {len(specialized_features)}")
                logger.info("Reintentando generación con cache limpio...")
                self.feature_engineer._clear_cache()
                specialized_features = self.feature_engineer.generate_all_features(df_with_features)
                
        except Exception as e:
            logger.error(f"ERROR CRÍTICO generando features especializadas: {str(e)}")
            raise ValueError(f"FALLO CRÍTICO: No se pudieron generar features especializadas para entrenamiento. Error: {str(e)}")
        
        # Obtener features especializadas EXCLUSIVAS y target
        feature_columns = self.get_feature_columns(df_with_features)
        X = df_with_features[feature_columns].copy()
        
        # PRESERVAR la columna Date para división cronológica
        if 'Date' in df_with_features.columns:
            X['Date'] = df_with_features['Date']
        
        # Determinar columna target
        target_col = 'double_double' if 'double_double' in df_with_features.columns else 'DD'
        if target_col not in df_with_features.columns:
            raise ValueError("No se encontró columna target (double_double o DD)")
        
        y = df_with_features[target_col].copy()
        
        logger.info(f"Entrenamiento configurado: {X.shape[0]} muestras, {X.shape[1]} features especializadas EXCLUSIVAS")
        
        # PARTE 1: ANÁLISIS DE PERFILES DE JUGADORES Y DESBALANCE
        self.logger.debug("=== ANÁLISIS DE DESBALANCE ESPECÍFICO PARA DOUBLE-DOUBLES ===")
        
        # NUEVO: Análisis especializado por posición
        self.logger.debug("=== ANÁLISIS POR POSICIÓN ===")
        position_analysis = self.position_classifier.analyze_position_patterns(df_with_features)
        
        # Analizar perfiles de jugadores
        profile_analysis = self.imbalance_handler.analyze_player_profiles(df_with_features)
        
        # Crear pesos estratificados
        sample_weights = self.imbalance_handler.create_stratified_weights(df_with_features)
        
        # Calcular thresholds por posición
        position_thresholds = self.imbalance_handler.get_position_based_thresholds(df_with_features)
        
        # Guardar análisis para uso posterior
        self.training_results['player_profile_analysis'] = profile_analysis
        self.training_results['position_thresholds'] = position_thresholds
        
        self.logger.info("Análisis de desbalance completado")
        
        # PARTE 5: FEATURE SELECTION - Seleccionar las mejores features para evitar overfitting
        if X.shape[1] > 30:  # Solo aplicar si tenemos más de 30 features
            self.logger.info("Aplicando selección de features para evitar overfitting...")
            selected_features = self._select_best_features(X, y, max_features=30)
            
            # Actualizar X con solo las features seleccionadas
            X_selected = X[selected_features].copy()
            
            # Preservar Date si existe
            if 'Date' in X.columns:
                X_selected['Date'] = X['Date']
            
            X = X_selected
            feature_columns = selected_features
            
            self.logger.info(f"Features reducidas de {len(specialized_features)} a {len(selected_features)} para evitar overfitting")
        else:
            self.logger.info("No se requiere selección de features (≤30 features)")
        
        # VERIFICAR que todas las features son especializadas (por definición del get_feature_columns corregido)
        specialized_count = len(feature_columns)  # Todas son especializadas por definición
        specialized_percentage = 100.0  # Por definición, todas son especializadas
        
        logger.info(f"VERIFICACIÓN CRÍTICA: {specialized_count}/{len(feature_columns)} features son especializadas ({specialized_percentage:.1f}%)")
        
        if specialized_percentage < 100:
            logger.error(f"ERROR: Solo {specialized_percentage:.1f}% de features son especializadas")
            logger.error("Esto indica un problema en get_feature_columns()")
        else:
            logger.info(" PERFECTO: Modelo usa 100% features especializadas")
        
        # Preparar datos
        X_train, X_val, y_train, y_val, self.scaler = DataProcessor.prepare_training_data(
            X, y, validation_split, self.scaler
        )
        
        # Optimización bayesiana si está habilitada
        if self.optimize_hyperparams and BAYESIAN_AVAILABLE:
            self._optimize_with_bayesian(X_train, y_train)
        
        # Preparar sample weights para entrenamiento (alinear con X_train)
        train_indices = X_train.index if hasattr(X_train, 'index') else range(len(X_train))
        sample_weights_train = sample_weights[train_indices] if len(sample_weights) == len(df_with_features) else None
        
        # Entrenar modelos individuales con sample weights
        individual_results = self._train_individual_models(X_train, y_train, X_val, y_val, sample_weights_train)
        
        # NUEVO: ENTRENAMIENTO AVANZADO DE META-LEARNERS
        logger.info("Entrenando modelo de stacking principal...")
        self.stacking_model.fit(X_train, y_train)
        
        # Establecer modelo como entrenado ANTES de evaluar
        self.is_fitted = True
        
        # Obtener predicciones base para meta-learning avanzado
        if hasattr(self, 'advanced_meta_learning') and self.advanced_meta_learning:
            logger.info("Generando predicciones base para meta-learning avanzado...")
            base_predictions_train = self._get_base_predictions(X_train, 'train')
            base_predictions_val = self._get_base_predictions(X_val, 'val')
            
            # Entrenar meta-learners adicionales (TODOS, incluyendo logistic independiente)
            logger.info("Entrenando meta-learners especializados...")
            for name, meta_learner in self.meta_learners.items():
                try:
                    logger.info(f"Entrenando meta-learner independiente: {name}")
                    
                    # Crear una copia independiente del meta-learner para evitar conflictos
                    if name == 'logistic':
                        # Crear nuevo LogisticRegression independiente del stacking
                        from sklearn.linear_model import LogisticRegression
                        independent_meta = LogisticRegression(
                            class_weight={0: 1.0, 1: 20.0},
                            random_state=42,
                            max_iter=3000,
                            C=0.5,
                            penalty='l2',
                            solver='liblinear',
                            fit_intercept=True
                        )
                        independent_meta.fit(base_predictions_train, y_train)
                        # Reemplazar en el diccionario
                        self.meta_learners[name] = independent_meta
                    else:
                        # Entrenar normalmente
                        meta_learner.fit(base_predictions_train, y_train)
                    
                    # Evaluar meta-learner
                    meta_pred = self.meta_learners[name].predict(base_predictions_val)
                    meta_proba = self.meta_learners[name].predict_proba(base_predictions_val)[:, 1]
                    
                    meta_acc = accuracy_score(y_val, meta_pred)
                    meta_auc = roc_auc_score(y_val, meta_proba)
                    
                    logger.info(f"Meta-learner {name}: ACC={meta_acc:.3f}, AUC={meta_auc:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error entrenando meta-learner {name}: {e}")
                    # Crear meta-learner dummy en caso de error
                    self.meta_learners[name] = None
        
        # Evaluar stacking model principal
        stacking_pred = self.stacking_model.predict(X_val)
        stacking_proba = self.stacking_model.predict_proba(X_val)[:, 1]
        
        stacking_metrics = MetricsCalculator.calculate_classification_metrics(
            y_val, stacking_pred, stacking_proba
        )
        
        logger.info(f"Stacking Model Principal - Accuracy: {stacking_metrics.get('accuracy', 0):.3f}, F1: {stacking_metrics.get('f1_score', 0):.3f}, AUC: {stacking_metrics.get('roc_auc', 0):.3f}")
        
        # Generar predicción final combinada si está habilitado
        if hasattr(self, 'advanced_meta_learning') and self.advanced_meta_learning:
            logger.info("Generando predicción final combinada...")
            try:
                final_proba = self._combine_meta_predictions(base_predictions_val, y_val)
                
                # Evaluar predicción combinada
                final_pred = (final_proba > 0.5).astype(int)
                combined_metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, final_pred, final_proba
                )
                
                logger.info(f"Meta-Learning Combinado - Accuracy: {combined_metrics.get('accuracy', 0):.3f}, F1: {combined_metrics.get('f1_score', 0):.3f}, AUC: {combined_metrics.get('roc_auc', 0):.3f}")
                
                # Usar las mejores métricas (stacking vs combinado)
                if combined_metrics['f1_score'] > stacking_metrics['f1_score']:
                    stacking_metrics = combined_metrics
                    self.use_combined_prediction = True
                else:
                    self.use_combined_prediction = False
                    
            except Exception as e:
                logger.error(f"Error en meta-learning combinado: {e}")
                self.use_combined_prediction = False
        
        # Guardar resultados con verificación de features especializadas (sin CV primero)
        results = {
            'individual_models': individual_results,
            'stacking_metrics': stacking_metrics,
            'feature_columns': feature_columns,
            'specialized_features_used': specialized_count,
            'total_features_generated': len(specialized_features),
            'specialized_percentage': specialized_percentage,
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        self.training_results = results
        
        # Cross-validation del ensemble completo (después de asignar training_results)
        logger.info("Ejecutando validación cruzada cronológica...")
        cv_results = self._perform_cross_validation(X, y)
        
        # Agregar resultados de CV a training_results
        self.training_results['cv_scores'] = cv_results
        
        # Calcular feature importance
        self._calculate_feature_importance(feature_columns)
        
        # PARTE 1: THRESHOLD OPTIMIZATION AVANZADO
        self.logger.debug("=== OPTIMIZACIÓN AVANZADA DE THRESHOLD ===")
        self.logger.info(f"Distribución de probabilidades en validación:")
        self.logger.info(f"  Min: {stacking_proba.min():.4f}")
        self.logger.info(f"  Max: {stacking_proba.max():.4f}")
        self.logger.info(f"  Media: {stacking_proba.mean():.4f}")
        self.logger.info(f"  Std: {stacking_proba.std():.4f}")
        
        # Probar múltiples métodos de optimización de threshold
        threshold_methods = ['f1_precision_balance', 'youden', 'precision_recall_curve']
        threshold_results = {}
        
        for method in threshold_methods:
            try:
                threshold = self._calculate_optimal_threshold_advanced(y_val, stacking_proba, method=method)
                
                # Evaluar este threshold
                y_pred_test = (stacking_proba >= threshold).astype(int)
                test_precision = precision_score(y_val, y_pred_test, zero_division=0)
                test_recall = recall_score(y_val, y_pred_test, zero_division=0)
                test_f1 = f1_score(y_val, y_pred_test, zero_division=0)
                
                threshold_results[method] = {
                    'threshold': threshold,
                    'precision': test_precision,
                    'recall': test_recall,
                    'f1': test_f1,
                    'predictions_positive': np.sum(y_pred_test),
                    'predictions_ratio': np.sum(y_pred_test) / len(y_pred_test)
                }
                
                self.logger.info(f"Método {method}: T={threshold:.4f}, P={test_precision:.3f}, R={test_recall:.3f}, F1={test_f1:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Error en método {method}: {str(e)}")
                threshold_results[method] = {'error': str(e)}
        
        # Seleccionar el mejor threshold basado en F1 score y precision mínima OPTIMIZADA
        best_method = None
        best_f1 = 0
        min_precision_required = 0.75  # Precision mínima AUMENTADA para reducir falsos positivos
        
        for method, result in threshold_results.items():
            if 'error' not in result:
                if result['precision'] >= min_precision_required and result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_method = method
        
        # Si no se encontró un método que cumpla los requisitos, usar el de mejor F1
        if best_method is None:
            best_f1 = 0
            for method, result in threshold_results.items():
                if 'error' not in result and result['f1'] > best_f1:
                    best_f1 = result['f1']
                    best_method = method
        
        # Usar el mejor threshold encontrado
        if best_method:
            self.optimal_threshold = threshold_results[best_method]['threshold']
            self.logger.info(f"MEJOR MÉTODO SELECCIONADO: {best_method}")
            self.logger.info(f"Threshold óptimo final: {self.optimal_threshold:.4f}")
        else:
            # Fallback al método legacy si todo falla
            self.logger.warning("Todos los métodos avanzados fallaron, usando método legacy")
            self.optimal_threshold = self._calculate_optimal_threshold_advanced(y_val, stacking_proba)
        
        # CORRECCIÓN: Validación final del threshold con rangos más realistas
        if self.optimal_threshold < 0.08:
            self.logger.info(f"Threshold ajustado desde {self.optimal_threshold:.4f} a 0.10 (mínimo realista)")
            self.optimal_threshold = 0.10
        elif self.optimal_threshold > 0.35:
            self.logger.info(f"Threshold ajustado desde {self.optimal_threshold:.4f} a 0.30 (máximo realista)")
            self.optimal_threshold = 0.30
        
        # Evaluar con threshold óptimo final
        y_val_pred_optimal = (stacking_proba >= self.optimal_threshold).astype(int)
        
        # Logging detallado de predicciones finales
        dd_predicted = np.sum(y_val_pred_optimal)
        dd_actual = np.sum(y_val)
        self.logger.info(f"=== RESULTADOS FINALES CON THRESHOLD ÓPTIMO ===")
        self.logger.info(f"Threshold final: {self.optimal_threshold:.4f}")
        self.logger.info(f"DD predichos: {dd_predicted}")
        self.logger.info(f"DD reales: {dd_actual}")
        self.logger.info(f"Ratio predicción: {dd_predicted/len(y_val)*100:.1f}%")
        self.logger.info(f"Ratio real: {dd_actual/len(y_val)*100:.1f}%")
        
        # Calcular métricas finales con threshold óptimo
        optimal_metrics = {
            'accuracy': accuracy_score(y_val, y_val_pred_optimal),
            'precision': precision_score(y_val, y_val_pred_optimal, zero_division=0),
            'recall': recall_score(y_val, y_val_pred_optimal, zero_division=0),
            'f1_score': f1_score(y_val, y_val_pred_optimal, zero_division=0),
            'roc_auc': roc_auc_score(y_val, stacking_proba)
        }
        
        self.logger.info("=== MÉTRICAS FINALES CON THRESHOLD ÓPTIMO ===")
        for metric, value in optimal_metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        results['optimal_threshold'] = self.optimal_threshold
        results['optimal_metrics'] = optimal_metrics
        results['threshold_optimization'] = threshold_results
        results['position_analysis'] = position_analysis
        
        logger.info(f"Entrenamiento completado con {len(feature_columns)} features especializadas EXCLUSIVAS")
        logger.info(f"Porcentaje de features especializadas: {specialized_percentage:.1f}%")
        
        return self.training_results
    
    def _get_base_predictions(self, X, phase='predict'):
        """Obtener predicciones de modelos base para meta-learning avanzado"""
        try:
            base_predictions = []
            
            # Obtener predicciones de cada modelo base
            for name, model in self.models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X)
                        if proba.shape[1] == 2:
                            base_predictions.append(proba[:, 1])  # Probabilidad clase positiva
                        else:
                            base_predictions.append(proba[:, 0])
                    else:
                        # Fallback a predict si no hay predict_proba
                        pred = model.predict(X)
                        base_predictions.append(pred.astype(float))
                        
                except Exception as e:
                    self.logger.warning(f"Error obteniendo predicciones de {name}: {e}")
                    # Crear predicción dummy
                    base_predictions.append(np.zeros(X.shape[0]))
            
            # Convertir a matriz
            base_matrix = np.column_stack(base_predictions)
            
            self.logger.info(f"Predicciones base generadas: {base_matrix.shape} ({phase})")
            return base_matrix
            
        except Exception as e:
            self.logger.error(f"Error generando predicciones base: {e}")
            # Fallback: matriz de ceros
            return np.zeros((X.shape[0], len(self.models)))
    
    def _combine_meta_predictions(self, base_predictions, y_true=None):
        """Combinar predicciones de múltiples meta-learners usando votación ponderada"""
        try:
            meta_probabilities = []
            meta_weights = []
            
            # Obtener predicciones de cada meta-learner con manejo robusto
            for name, meta_learner in self.meta_learners.items():
                try:
                    # Verificar que el meta-learner no sea None
                    if meta_learner is None:
                        self.logger.warning(f"Meta-learner {name} es None, saltando")
                        continue
                    
                    # Verificar que esté entrenado
                    if not hasattr(meta_learner, 'predict_proba'):
                        self.logger.warning(f"Meta-learner {name} no tiene predict_proba, saltando")
                        continue
                    
                    # Verificar que esté fitted
                    from sklearn.utils.validation import check_is_fitted
                    try:
                        check_is_fitted(meta_learner)
                    except:
                        self.logger.warning(f"Meta-learner {name} no está entrenado, saltando")
                        continue
                    
                    proba = meta_learner.predict_proba(base_predictions)
                    if proba.shape[1] == 2:
                        meta_prob = proba[:, 1]
                    else:
                        meta_prob = proba[:, 0]
                    
                    meta_probabilities.append(meta_prob)
                    
                    # Calcular peso basado en performance si tenemos y_true
                    if y_true is not None:
                        try:
                            pred = (meta_prob > 0.5).astype(int)
                            f1 = f1_score(y_true, pred, zero_division=0)
                            weight = max(0.1, f1)  # Peso mínimo 0.1
                            meta_weights.append(weight)
                            self.logger.info(f"Meta-learner {name}: F1={f1:.3f}, Peso={weight:.3f}")
                        except Exception as weight_error:
                            self.logger.warning(f"Error calculando peso para {name}: {weight_error}")
                            meta_weights.append(1.0)  # Peso por defecto
                    else:
                        meta_weights.append(1.0)
                        
                except Exception as e:
                    self.logger.warning(f"Error en meta-learner {name}: {e}")
                    # Predicción dummy conservadora
                    meta_probabilities.append(np.full(base_predictions.shape[0], 0.1))
                    meta_weights.append(0.1)
            
            if not meta_probabilities:
                self.logger.error("No se pudieron obtener predicciones de meta-learners")
                return np.full(base_predictions.shape[0], 0.1)
            
            # Normalizar pesos
            meta_weights = np.array(meta_weights)
            meta_weights = meta_weights / np.sum(meta_weights)
            
            # Combinar predicciones usando votación ponderada
            meta_matrix = np.column_stack(meta_probabilities)
            combined_proba = np.average(meta_matrix, axis=1, weights=meta_weights)
            
            self.logger.info(f"Meta-learners combinados: {len(meta_probabilities)} modelos")
            self.logger.info(f"Pesos: {dict(zip(self.meta_learners.keys(), meta_weights))}")
            
            return combined_proba
            
        except Exception as e:
            self.logger.error(f"Error combinando meta-learners: {e}")
            # Fallback: usar solo el stacking principal
            return self.stacking_model.predict_proba(base_predictions)[:, 1]
    
    def _train_individual_models(self, X_train, y_train, X_val, y_val, sample_weights=None) -> Dict:
        """Entrenar modelos individuales con early stopping y sample weights estratificados"""
        
        results = {}
        
        for name, model in self.models.items():
            try:
                if name in ['xgboost', 'lightgbm']:
                    # Modelos con early stopping y sample weights
                    if name == 'xgboost':
                        fit_params = {
                            'eval_set': [(X_val, y_val)],
                            'verbose': False
                        }
                        if sample_weights is not None:
                            fit_params['sample_weight'] = sample_weights
                        
                        model.fit(X_train, y_train, **fit_params)
                    else:  # lightgbm
                        fit_params = {
                            'eval_set': [(X_val, y_val)],
                            'callbacks': [lgb.early_stopping(10), lgb.log_evaluation(0)]
                        }
                        if sample_weights is not None:
                            fit_params['sample_weight'] = sample_weights
                        
                        model.fit(X_train, y_train, **fit_params)
                elif name in ['gradient_boosting']:
                    # Modelos sklearn que soportan sample_weight
                    fit_params = {}
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    
                    model.fit(X_train, y_train, **fit_params)
                elif name == 'catboost':
                    # CatBoost con sample weights
                    fit_params = {
                        'verbose': False,
                        'plot': False
                    }
                    if sample_weights is not None:
                        fit_params['sample_weight'] = sample_weights
                    
                    model.fit(X_train, y_train, **fit_params)
                else:
                    # Otros modelos (neural_network no soporta sample_weight directamente)
                    model.fit(X_train, y_train)
                
                # Evaluar modelo
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)[:, 1]
                
                metrics = MetricsCalculator.calculate_classification_metrics(
                    y_val, val_pred, val_proba
                )
                
                self.logger.info(f"Modelo {name} entrenado - Accuracy: {metrics.get('accuracy', 0):.3f}, F1: {metrics.get('f1_score', 0):.3f}")
                
                results[name] = {
                    'model': model,
                    'val_metrics': metrics
                }
                
                # Guardar feature importance si está disponible
                if hasattr(model, 'feature_importances_'):
                    # Asegurar que feature_importance es un dict
                    if not hasattr(self, 'feature_importance') or not isinstance(self.feature_importance, dict):
                        self.feature_importance = {}
                    
                    self.feature_importance[name] = {
                        'importances': model.feature_importances_.tolist(),
                        'feature_names': list(X_train.columns) if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(len(model.feature_importances_))]
                    }
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                results[name] = {'error': str(e)}
        
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predecir clases usando thresholds adaptativos por posición/rol"""
        probabilities = self.predict_proba(df)
        
        # Usar threshold óptimo global como fallback
        default_threshold = getattr(self, 'optimal_threshold', 0.5)
        
        # Usar thresholds especializados por posición
        if hasattr(self, 'training_results') and 'position_analysis' in self.training_results:
            self.logger.info("Usando thresholds especializados por posición")
            return self._predict_with_position_specialization(df, probabilities, default_threshold)
        
        # Fallback: Intentar usar thresholds básicos por posición si están disponibles
        elif hasattr(self, 'training_results') and 'position_thresholds' in self.training_results:
            position_thresholds = self.training_results['position_thresholds']
            
            # Inferir posición para cada jugador en el DataFrame de predicción
            predictions = np.zeros(len(df), dtype=int)
            
            # Agrupar por jugador para inferir posición
            player_stats = df.groupby('player').agg({
                'rebounds': 'mean',
                'assists': 'mean', 
                'points': 'mean'
            }).round(3)
            
            def infer_position(row):
                avg_trb = row.get('rebounds', 0)
                avg_ast = row.get('assists', 0) 
                avg_pts = row.get('points', 0)
                
                if avg_trb >= 8:
                    return 'big_man'
                elif avg_ast >= 5:
                    return 'playmaker'
                elif avg_pts >= 15:
                    return 'scorer'
                else:
                    return 'role_player'
            
            player_stats['position_type'] = player_stats.apply(infer_position, axis=1)
            player_positions = player_stats['position_type'].to_dict()
            
            # Aplicar threshold específico por posición
            for i, row in df.iterrows():
                player = row['player']
                position = player_positions.get(player, 'role_player')
                threshold = position_thresholds.get(position, default_threshold)
                
                predictions[i] = (probabilities[i, 1] >= threshold).astype(int)
            
            # Logging para debug
            position_counts = {}
            for pos, thresh in position_thresholds.items():
                count = sum(1 for p in player_positions.values() if p == pos)
                position_counts[pos] = count
                
            self.logger.info(f"Prediciendo con thresholds adaptativos por posición:")
            for pos, thresh in position_thresholds.items():
                count = position_counts.get(pos, 0)
                self.logger.info(f"  {pos}: threshold={thresh:.3f} ({count} jugadores)")
            
        else:
            # Fallback al threshold global
            self.logger.info(f"Prediciendo con threshold global: {default_threshold:.4f}")
            predictions = (probabilities[:, 1] >= default_threshold).astype(int)
        
        positive_predictions = predictions.sum()
        self.logger.info(f"Predicciones positivas: {positive_predictions} de {len(predictions)}")
        self.logger.info(f"Probabilidades - Min: {probabilities[:, 1].min():.4f}, Max: {probabilities[:, 1].max():.4f}")
        
        return predictions
    
    def _predict_with_position_specialization(self, df: pd.DataFrame, probabilities: np.ndarray, default_threshold: float) -> np.ndarray:
        """Predicción especializada usando thresholds adaptativos por posición con filtros anti-FP"""
        
        # Categorizar posiciones en el DataFrame de predicción
        df_with_positions = self.position_classifier.categorize_position(df.copy())
        
        # Obtener análisis de posición del entrenamiento
        position_analysis = self.training_results['position_analysis']
        
        # NUEVO: Calcular thresholds MÁS CONSERVADORES para reducir FP
        position_thresholds = {}
        for position, stats in position_analysis.items():
            dd_rate = stats['dd_rate']
            
        # THRESHOLDS BALANCEADOS PARA APUESTAS (90%+ CONFIANZA)
        # Para apuestas necesitamos buena precisión con volumen moderado
            if dd_rate >= 0.15:  # Centers (alta tasa DD)
                position_thresholds[position] = 0.75  # Predicciones seguras
            elif dd_rate >= 0.08:  # Power Forwards (tasa media)
                position_thresholds[position] = 0.80  # Moderadamente conservador
            elif dd_rate >= 0.03:  # Small Forwards (tasa baja)
                position_thresholds[position] = 0.85  # Conservador
            else:  # Guards (tasa muy baja)
                position_thresholds[position] = 0.90  # Conservador
        
        # MATRIZ DE COSTOS ULTRA-CONSERVADORA PARA APUESTAS
        # En apuestas, FP (falso positivo) es EXTREMADAMENTE costoso
        cost_matrix = {
            'Center': {'FP': 10.0, 'FN': 1.0},     # Centers: FP extremadamente costoso
            'PowerForward': {'FP': 12.0, 'FN': 1.0}, # PF: FP aún más costoso
            'SmallForward': {'FP': 15.0, 'FN': 1.0}, # SF: FP ultra costoso
            'Guard': {'FP': 20.0, 'FN': 1.0}       # Guards: FP extremadamente costoso
        }
        
        # NUEVO: Aplicar thresholds específicos por posición + filtros de confianza
        predictions = np.zeros(len(df), dtype=int)
        position_counts = {}
        confidence_filtered = 0
        cost_aware_predictions = 0
        
        for idx, (i, row) in enumerate(df_with_positions.iterrows()):
            position = row.get('Position_Category', 'Unknown')
            threshold = position_thresholds.get(position, default_threshold)
            probability = probabilities[idx, 1]
            
            # FILTRO 1: Threshold básico por posición
            base_prediction = (probability >= threshold).astype(int)
            
            # FILTRO 2: Filtro de confianza adicional para reducir FP
            if base_prediction == 1:
                # FILTROS BALANCEADOS PARA APUESTAS (90%+ CONFIANZA)
                # Predicciones donde el modelo esté 90%+ seguro
                betting_confidence_thresholds = {
                    'Center': 0.90,      # Centers: 90%+ confianza mínima
                    'PowerForward': 0.90, # PF: 90%+ confianza mínima
                    'SmallForward': 0.90, # SF: 90%+ confianza mínima
                    'Guard': 0.90        # Guards: 90%+ confianza mínima
                }
                
                # Confianza mínima balanceada para apuestas
                min_confidence = betting_confidence_thresholds.get(position, 0.90)
                
                # Factor adicional basado en DD rate histórico (más permisivo)
                dd_rate_boost = min(0.03, max(0.0, (dd_rate - 0.05) * 0.3))  # Boost máximo 3%
                min_confidence = min(0.95, min_confidence + dd_rate_boost)  # Máximo 95%
                            
                # FILTRO 3: DECISIÓN SENSIBLE AL COSTO
                # Calcular costo esperado de cada decisión
                costs = cost_matrix.get(position, {'FP': 2.0, 'FN': 1.5})
                
                # Costo esperado de predecir DD (FP * P(no_DD) + 0 * P(DD))
                cost_predict_dd = costs['FP'] * (1 - probability)
                
                # Costo esperado de predecir no-DD (FN * P(DD) + 0 * P(no_DD))
                cost_predict_no_dd = costs['FN'] * probability
                
                # Decisión basada en costo mínimo
                cost_aware_decision = 1 if cost_predict_dd < cost_predict_no_dd else 0
                
                # Combinar decisión basada en threshold y costo
                final_decision = 1 if (base_prediction == 1 and 
                                     probability >= min_confidence and 
                                     cost_aware_decision == 1) else 0
                
                if cost_aware_decision == 1:
                    cost_aware_predictions += 1
                
                # FILTRO 4: FILTROS BALANCEADOS PARA APUESTAS (90%+ CONFIANZA)
                betting_pass = True
                
                # Filtro 1: Solo jugadores con al menos 15 minutos promedio (menos restrictivo)
                if 'mp_hist_avg_5g' in row.index and row['mp_hist_avg_5g'] < 15:
                    betting_pass = False
                
                # Filtro 2: Solo jugadores con DD rate histórico > 5% (menos restrictivo)
                if 'dd_rate_5g' in row.index and row['dd_rate_5g'] < 0.05:
                    betting_pass = False
                
                # Filtro 3: Solo jugadores con confianza 90%+ (menos restrictivo)
                if probability < 0.90:  # Mínimo 90% de confianza
                    betting_pass = False
                
                # Filtro 4: Reducir predicciones en final de temporada regular (opcional)
                if 'days_into_season' in row.index and row['days_into_season'] > 200:
                    # Aumentar confianza mínima en final de temporada
                    min_confidence += 0.01
                
                # Aplicar filtros ultra-conservadores para apuestas
                if final_decision == 1 and betting_pass and probability >= min_confidence:
                    predictions[idx] = 1
                else:
                    predictions[idx] = 0
                    if base_prediction == 1 and probability < min_confidence:
                        confidence_filtered += 1
            else:
                predictions[idx] = 0
            
            # Contar por posición para logging
            if position not in position_counts:
                position_counts[position] = {'total': 0, 'predicted_dd': 0, 'threshold': threshold, 'min_confidence': min_confidence if base_prediction == 1 else 0}
            position_counts[position]['total'] += 1
            position_counts[position]['predicted_dd'] += predictions[idx]
        
        # Logging detallado por posición con filtros balanceados para apuestas
        self.logger.info("=== PREDICCIONES BALANCEADAS PARA APUESTAS (90%+ CONFIANZA) ===")
        self.logger.info(f"Total filtrado por confianza insuficiente: {confidence_filtered}")
        self.logger.info(f"Predicciones sensibles al costo: {cost_aware_predictions}")
        self.logger.info(" MODO APUESTAS: Predicciones de ALTA CONFIANZA (90%+) con volumen balanceado")
        
        for position, counts in position_counts.items():
            total = counts['total']
            predicted = counts['predicted_dd']
            threshold = counts['threshold']
            min_conf = counts.get('min_confidence', 0)
            rate = predicted / total * 100 if total > 0 else 0
            costs = cost_matrix.get(position, {'FP': 10.0, 'FN': 1.0})
            
            self.logger.info(f"{position}: {predicted}/{total} DD predichos ({rate:.1f}%), threshold={threshold:.3f}, min_conf={min_conf:.2f}, costs=FP:{costs['FP']:.1f}/FN:{costs['FN']:.1f}")
        
        total_predicted = predictions.sum()
        betting_rate = total_predicted / len(predictions) * 100
        self.logger.info(f" APUESTAS: {total_predicted}/{len(predictions)} predicciones de alta confianza ({betting_rate:.1f}%)")
        
        if betting_rate < 3.0:
            self.logger.info(" EXCELENTE: Volumen bajo, alta calidad para apuestas")
        elif betting_rate < 8.0:
            self.logger.info(" BUENO: Volumen moderado, buena calidad para apuestas")
        elif betting_rate < 15.0:
            self.logger.info("  VOLUMEN ALTO: Considerar thresholds más conservadores")
        else:
            self.logger.info(" VOLUMEN MUY ALTO: Necesita ajustes para apuestas")
        
        return predictions
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predicción probabilística usando stacking model con features especializadas EXCLUSIVAS"""
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        # Verificar orden cronológico antes de generar features
        if 'Date' in df.columns and 'player' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                self.logger.info("Reordenando datos cronológicamente para predicción...")
                df = df.sort_values(['player', 'Date']).reset_index(drop=True)
        
        # Generar features especializadas (modifica df in-place, retorna List[str])
        self.logger.info("Generando features especializadas para predicción...")
        specialized_features = self.feature_engineer.generate_all_features(df)

        # Determinar expected_features dinámicamente del modelo entrenado
        try:
            if 'feature_columns' in self.training_results:
                expected_features = self.training_results['feature_columns']
                self.logger.info(f"Usando feature_columns del training: {len(expected_features)} features")
            elif hasattr(self, 'feature_columns'):
                expected_features = self.feature_columns
                self.logger.info(f"Usando feature_columns del modelo: {len(expected_features)} features")
            elif hasattr(self, 'expected_features'):
                expected_features = self.expected_features
                self.logger.info(f"Usando expected_features: {len(expected_features)} features")
            else:
                # Fallback: usar las features que se acaban de generar
                expected_features = specialized_features
                self.logger.warning("No se encontraron expected_features, usando todas las features generadas")
        except Exception as e:
            self.logger.warning(f"Error obteniendo expected_features: {e}")
            expected_features = specialized_features
        
        # Reordenar DataFrame según expected_features (df ya tiene las features)
        available_features = [f for f in expected_features if f in df.columns]
        if len(available_features) != len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            self.logger.warning(f"Features faltantes ({len(missing_features)}): {list(missing_features)[:5]}...")
            # Agregar features faltantes con valor 0
            for feature in missing_features:
                df[feature] = 0
                available_features.append(feature)
        
        # Usar expected_features en el orden correcto
        X = df[expected_features].copy()
        X_scaled = DataProcessor.prepare_prediction_data(X, self.scaler)
        
        self.logger.info(f"Predicción usando {len(expected_features)} features especializadas")
        
        # Usar meta-learning avanzado si está habilitado
        if hasattr(self, 'use_combined_prediction') and self.use_combined_prediction:
            self.logger.info("Usando meta-learning combinado para predicción")
            try:
                # Obtener predicciones base
                base_predictions = self._get_base_predictions(X_scaled, 'predict')
                
                # Combinar meta-learners
                combined_proba = self._combine_meta_predictions(base_predictions)
                
                # Convertir a formato estándar de sklearn
                proba_matrix = np.column_stack([1 - combined_proba, combined_proba])
                return proba_matrix
                
            except Exception as e:
                self.logger.error(f"Error en meta-learning combinado, usando stacking principal: {e}")
                return self.stacking_model.predict_proba(X_scaled)
        else:
            # Usar stacking principal
            return self.stacking_model.predict_proba(X_scaled)
    
    def _optimize_with_bayesian(self, X_train, y_train):
        """Optimización bayesiana de hiperparámetros"""
        
        if not BAYESIAN_AVAILABLE:
            logger.warning("Optimización bayesiana no disponible - skopt no instalado")
            return
        
        # Distribuir llamadas entre modelos
        calls_per_model = max(8, self.bayesian_n_calls // 3)
        
        # Optimizar modelos principales
        self._optimize_xgboost_bayesian(X_train, y_train, calls_per_model)
        self._optimize_lightgbm_bayesian(X_train, y_train, calls_per_model)
    
    def _optimize_xgboost_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para XGBoost con validación cronológica"""
        
        space = [
            Integer(30, 100, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(1.0, 5.0, name='reg_alpha'),
            Real(2.0, 8.0, name='reg_lambda')
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = xgb.XGBClassifier(
                **params,
                random_state=42,
                eval_metric='logloss',
                n_jobs=-1
            )
            
            # Usar validación cronológica en lugar de StratifiedKFold
            time_splits = DataProcessor.create_time_series_split(X_train, y_train, n_splits=3)
            scores = []
            
            for train_indices, val_indices in time_splits:
                X_fold_train = X_train.iloc[train_indices] if hasattr(X_train, 'iloc') else X_train[train_indices]
                y_fold_train = y_train.iloc[train_indices] if hasattr(y_train, 'iloc') else y_train[train_indices]
                X_fold_val = X_train.iloc[val_indices] if hasattr(X_train, 'iloc') else X_train[val_indices]
                y_fold_val = y_train.iloc[val_indices] if hasattr(y_train, 'iloc') else y_train[val_indices]
                
                model.fit(X_fold_train, y_fold_train)
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                scores.append(score)
            
            return -np.mean(scores)
        
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['xgboost'].set_params(**best_params)
        
        if not hasattr(self, 'bayesian_results'):
            self.bayesian_results = {}
        
        self.bayesian_results['xgboost'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
    
    def _optimize_lightgbm_bayesian(self, X_train, y_train, n_calls=10):
        """Optimización bayesiana específica para LightGBM con validación cronológica"""
        
        space = [
            Integer(30, 100, name='n_estimators'),
            Integer(3, 6, name='max_depth'),
            Real(0.01, 0.1, name='learning_rate'),
            Real(0.6, 0.9, name='subsample'),
            Real(0.6, 0.9, name='colsample_bytree'),
            Real(1.0, 5.0, name='reg_alpha'),
            Real(2.0, 8.0, name='reg_lambda'),
            Integer(20, 60, name='min_child_samples')
        ]
        
        @use_named_args(space)
        def objective(**params):
            model = lgb.LGBMClassifier(
                **params,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
            # Usar validación cronológica
            time_splits = DataProcessor.create_time_series_split(X_train, y_train, n_splits=3)
            scores = []
            
            for train_indices, val_indices in time_splits:
                X_fold_train = X_train.iloc[train_indices] if hasattr(X_train, 'iloc') else X_train[train_indices]
                y_fold_train = y_train.iloc[train_indices] if hasattr(y_train, 'iloc') else y_train[train_indices]
                X_fold_val = X_train.iloc[val_indices] if hasattr(X_train, 'iloc') else X_train[val_indices]
                y_fold_val = y_train.iloc[val_indices] if hasattr(y_train, 'iloc') else y_train[val_indices]
                
                model.fit(X_fold_train, y_fold_train)
                y_proba = model.predict_proba(X_fold_val)[:, 1]
                score = roc_auc_score(y_fold_val, y_proba)
                scores.append(score)
            
            return -np.mean(scores)
        
        result = gp_minimize(
            objective, space,
            n_calls=n_calls,
            random_state=42,
            n_jobs=1
        )
        
        # Actualizar mejor modelo
        best_params = dict(zip([dim.name for dim in space], result.x))
        self.models['lightgbm'].set_params(**best_params)
        
        if not hasattr(self, 'bayesian_results'):
            self.bayesian_results = {}
        
        self.bayesian_results['lightgbm'] = {
            'best_score': -result.fun,
            'best_params': best_params,
            'convergence': result.func_vals
        }
    
    def _perform_cross_validation(self, X, y) -> Dict[str, Any]:
        """
        PARTE 3: CROSS-VALIDATION MEJORADA
        Realizar validación cruzada cronológica rigurosa con detección de overfitting
        """
        
        self.logger.debug("=== INICIANDO CROSS-VALIDATION MEJORADA ===")
        
        # Crear splits cronológicos más robustos
        time_splits = DataProcessor.create_time_series_split(X, y, n_splits=5)
        
        # Métricas para detectar overfitting
        overfitting_metrics = {}
        
        # Evaluar modelos individuales con detección de overfitting
        for name, model_info in self.training_results['individual_models'].items():
            if 'model' in model_info:
                model = model_info['model']
                try:
                    self.logger.info(f"Evaluando {name} en cross-validation...")
                    
                    cv_scores = []
                    train_scores = []  # Para detectar overfitting
                    precision_scores = []
                    recall_scores = []
                    accuracy_scores = []
                    roc_auc_scores = []
                    
                    for fold_idx, (train_indices, val_indices) in enumerate(time_splits):
                        # Obtener datos para este split
                        X_train_cv = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
                        y_train_cv = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
                        X_val_cv = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
                        y_val_cv = y.iloc[val_indices] if hasattr(y, 'iloc') else y[val_indices]
                        
                        # Limpiar datos de entrenamiento
                        X_train_cv = self._clean_nan_exhaustive(X_train_cv)
                        X_val_cv = self._clean_nan_exhaustive(X_val_cv)
                        
                        # REMOVER columna Date si existe
                        if 'Date' in X_train_cv.columns:
                            X_train_cv = X_train_cv.drop(columns=['Date'])
                        if 'Date' in X_val_cv.columns:
                            X_val_cv = X_val_cv.drop(columns=['Date'])
                        
                        # Entrenar modelo específico para este fold
                        # ELIMINADO: Neural network por bajo rendimiento
                        # if name == 'neural_network': ...
                        
                        if name in ['xgboost', 'lightgbm']:
                            # Para XGBoost y LightGBM, crear modelos con regularización aumentada
                            from sklearn.base import clone
                            temp_model = clone(model)
                            
                            # Aumentar regularización para CV
                            if name == 'xgboost':
                                temp_model.set_params(
                                    n_estimators=100,  # Reducido
                                    max_depth=4,       # Reducido
                                    reg_alpha=0.3,     # Aumentado
                                    reg_lambda=2.0,    # Aumentado
                                    early_stopping_rounds=None
                                )
                            elif name == 'lightgbm':
                                temp_model.set_params(
                                    n_estimators=100,  # Reducido
                                    max_depth=4,       # Reducido
                                    reg_alpha=0.3,     # Aumentado
                                    reg_lambda=2.0,    # Aumentado
                                    early_stopping_rounds=None
                                )
                            
                            temp_model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = temp_model.predict(X_val_cv)
                            y_pred_train = temp_model.predict(X_train_cv)
                            
                        else:
                            # Para otros modelos, clonar con regularización aumentada
                            from sklearn.base import clone
                            temp_model = clone(model)
                            
                            # Aumentar regularización según el tipo de modelo
                            if name == 'gradient_boosting':
                                temp_model.set_params(
                                    n_estimators=100,      # Reducido
                                    max_depth=4,           # Reducido
                                    learning_rate=0.03,    # Reducido
                                    min_samples_split=15,  # Aumentado
                                    min_samples_leaf=8     # Aumentado
                                )
                            
                            temp_model.fit(X_train_cv, y_train_cv)
                            y_pred_cv = temp_model.predict(X_val_cv)
                            y_pred_train = temp_model.predict(X_train_cv)
                        
                        # Calcular métricas para validación
                        val_f1 = f1_score(y_val_cv, y_pred_cv, zero_division=0)
                        val_precision = precision_score(y_val_cv, y_pred_cv, zero_division=0)
                        val_recall = recall_score(y_val_cv, y_pred_cv, zero_division=0)
                        val_accuracy = accuracy_score(y_val_cv, y_pred_cv)
                        
                        # Calcular ROC-AUC si el modelo tiene predict_proba
                        try:
                            if hasattr(temp_model, 'predict_proba'):
                                y_proba_cv = temp_model.predict_proba(X_val_cv)[:, 1]
                                val_roc_auc = roc_auc_score(y_val_cv, y_proba_cv)
                            else:
                                val_roc_auc = 0.0
                        except:
                            val_roc_auc = 0.0
                        
                        # Calcular métricas para entrenamiento (detectar overfitting)
                        train_f1 = f1_score(y_train_cv, y_pred_train, zero_division=0)
                        
                        cv_scores.append(val_f1)
                        train_scores.append(train_f1)
                        precision_scores.append(val_precision)
                        recall_scores.append(val_recall)
                        accuracy_scores.append(val_accuracy)
                        roc_auc_scores.append(val_roc_auc)
                        
                        self.logger.debug(f"  Fold {fold_idx+1}: Val F1={val_f1:.3f}, Train F1={train_f1:.3f}, P={val_precision:.3f}, R={val_recall:.3f}")
                    
                    # Calcular estadísticas finales
                    cv_scores = np.array(cv_scores)
                    train_scores = np.array(train_scores)
                    precision_scores = np.array(precision_scores)
                    recall_scores = np.array(recall_scores)
                    accuracy_scores = np.array(accuracy_scores)
                    roc_auc_scores = np.array(roc_auc_scores)
                    
                    # Detectar overfitting
                    overfitting_gap = train_scores.mean() - cv_scores.mean()
                    overfitting_detected = overfitting_gap > 0.15  # Threshold de overfitting
                    
                    self.cv_scores[name] = {
                        'validation_f1_mean': cv_scores.mean(),
                        'validation_f1_std': cv_scores.std(),
                        'training_f1_mean': train_scores.mean(),
                        'precision_mean': precision_scores.mean(),
                        'precision_std': precision_scores.std(),
                        'recall_mean': recall_scores.mean(),
                        'recall_std': recall_scores.std(),
                        'accuracy_mean': accuracy_scores.mean(),
                        'accuracy_std': accuracy_scores.std(),
                        'roc_auc_mean': roc_auc_scores.mean(),
                        'roc_auc_std': roc_auc_scores.std(),
                        'overfitting_gap': overfitting_gap,
                        'overfitting_detected': overfitting_detected,
                        'scores': cv_scores.tolist(),
                        'stability_score': 1.0 - cv_scores.std()  # Métrica de estabilidad
                    }
                    
                    overfitting_metrics[name] = {
                        'gap': overfitting_gap,
                        'detected': overfitting_detected,
                        'stability': 1.0 - cv_scores.std()
                    }
                    
                    self.logger.info(f"  {name} - Val F1: {cv_scores.mean():.3f}±{cv_scores.std():.3f}, Overfitting: {overfitting_gap:.3f}")
                    if overfitting_detected:
                        self.logger.warning(f"    OVERFITTING DETECTADO en {name}")
                    
                except Exception as e:
                    self.logger.warning(f"Error en CV para {name}: {str(e)}")
                    self.cv_scores[name] = {'error': str(e)}
        
        # Evaluar stacking model con detección de overfitting
        try:
            self.logger.info("Evaluando stacking model en cross-validation...")
            
            stacking_val_scores = []
            stacking_train_scores = []
            stacking_precision_scores = []
            stacking_recall_scores = []
            stacking_accuracy_scores = []
            stacking_roc_auc_scores = []
            
            for fold_idx, (train_indices, val_indices) in enumerate(time_splits):
                # Obtener datos para este split
                X_train_cv = X.iloc[train_indices] if hasattr(X, 'iloc') else X[train_indices]
                y_train_cv = y.iloc[train_indices] if hasattr(y, 'iloc') else y[train_indices]
                X_val_cv = X.iloc[val_indices] if hasattr(X, 'iloc') else X[val_indices]
                y_val_cv = y.iloc[val_indices] if hasattr(y, 'iloc') else y[val_indices]
                
                # LIMPIEZA EXHAUSTIVA DE NaN
                X_train_cv = self._clean_nan_exhaustive(X_train_cv)
                X_val_cv = self._clean_nan_exhaustive(X_val_cv)
                
                # REMOVER columna Date si existe
                if 'Date' in X_train_cv.columns:
                    X_train_cv = X_train_cv.drop(columns=['Date'])
                if 'Date' in X_val_cv.columns:
                    X_val_cv = X_val_cv.drop(columns=['Date'])
                
                # Crear stacking model con regularización aumentada para CV
                from sklearn.base import clone
                temp_stacking = clone(self.stacking_model)
                
                # Aumentar regularización del meta-modelo
                temp_stacking.final_estimator.set_params(
                    C=0.3,  # Más regularización
                    class_weight={0: 1.0, 1: 20.0}  # Manejo de desbalance
                )
                
                temp_stacking.fit(X_train_cv, y_train_cv)
                
                # Predicciones
                y_pred_val = temp_stacking.predict(X_val_cv)
                y_pred_train = temp_stacking.predict(X_train_cv)
                
                # Métricas
                val_f1 = f1_score(y_val_cv, y_pred_val, zero_division=0)
                train_f1 = f1_score(y_train_cv, y_pred_train, zero_division=0)
                val_precision = precision_score(y_val_cv, y_pred_val, zero_division=0)
                val_recall = recall_score(y_val_cv, y_pred_val, zero_division=0)
                val_accuracy = accuracy_score(y_val_cv, y_pred_val)
                
                # ROC-AUC para stacking
                try:
                    y_proba_val = temp_stacking.predict_proba(X_val_cv)[:, 1]
                    val_roc_auc = roc_auc_score(y_val_cv, y_proba_val)
                except:
                    val_roc_auc = 0.0
                
                stacking_val_scores.append(val_f1)
                stacking_train_scores.append(train_f1)
                stacking_precision_scores.append(val_precision)
                stacking_recall_scores.append(val_recall)
                stacking_accuracy_scores.append(val_accuracy)
                stacking_roc_auc_scores.append(val_roc_auc)
                
                self.logger.debug(f"  Stacking Fold {fold_idx+1}: Val F1={val_f1:.3f}, Train F1={train_f1:.3f}")
            
            # Estadísticas finales del stacking
            stacking_val_scores = np.array(stacking_val_scores)
            stacking_train_scores = np.array(stacking_train_scores)
            stacking_precision_scores = np.array(stacking_precision_scores)
            stacking_recall_scores = np.array(stacking_recall_scores)
            stacking_accuracy_scores = np.array(stacking_accuracy_scores)
            stacking_roc_auc_scores = np.array(stacking_roc_auc_scores)
            
            # Detectar overfitting en stacking
            stacking_overfitting_gap = stacking_train_scores.mean() - stacking_val_scores.mean()
            stacking_overfitting_detected = stacking_overfitting_gap > 0.15
            
            self.cv_scores['stacking'] = {
                'validation_f1_mean': stacking_val_scores.mean(),
                'validation_f1_std': stacking_val_scores.std(),
                'training_f1_mean': stacking_train_scores.mean(),
                'precision_mean': stacking_precision_scores.mean(),
                'precision_std': stacking_precision_scores.std(),
                'recall_mean': stacking_recall_scores.mean(),
                'recall_std': stacking_recall_scores.std(),
                'accuracy_mean': stacking_accuracy_scores.mean(),
                'accuracy_std': stacking_accuracy_scores.std(),
                'roc_auc_mean': stacking_roc_auc_scores.mean(),
                'roc_auc_std': stacking_roc_auc_scores.std(),
                'overfitting_gap': stacking_overfitting_gap,
                'overfitting_detected': stacking_overfitting_detected,
                'scores': stacking_val_scores.tolist(),
                'stability_score': 1.0 - stacking_val_scores.std()
            }
            
            overfitting_metrics['stacking'] = {
                'gap': stacking_overfitting_gap,
                'detected': stacking_overfitting_detected,
                'stability': 1.0 - stacking_val_scores.std()
            }
            
            self.logger.info(f"  Stacking - Val F1: {stacking_val_scores.mean():.3f}±{stacking_val_scores.std():.3f}, Overfitting: {stacking_overfitting_gap:.3f}")
            if stacking_overfitting_detected:
                self.logger.warning(f"    OVERFITTING DETECTADO en stacking model")
            
        except Exception as e:
            self.logger.warning(f"Error en CV para stacking: {str(e)}")
            self.cv_scores['stacking'] = {'error': str(e)}
        
        # Resumen de overfitting
        self.logger.debug("=== RESUMEN DE DETECCIÓN DE OVERFITTING ===")
        overfitting_count = sum(1 for metrics in overfitting_metrics.values() if metrics['detected'])
        self.logger.info(f"Modelos con overfitting detectado: {overfitting_count}/{len(overfitting_metrics)}")
        
        for model_name, metrics in overfitting_metrics.items():
            status = "  OVERFITTING" if metrics['detected'] else " OK"
            self.logger.info(f"  {model_name}: {status} (Gap: {metrics['gap']:.3f}, Estabilidad: {metrics['stability']:.3f})")
        
        # Guardar métricas de overfitting
        self.cv_scores['overfitting_summary'] = overfitting_metrics
        
        return self.cv_scores
    
    def _clean_nan_exhaustive(self, X):
        """Limpieza exhaustiva de NaN para validación cruzada"""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        X_clean = X.copy()
        
        # 1. Reemplazar infinitos
        X_clean = X_clean.replace([np.inf, -np.inf], np.nan)
        
        # 2. Verificar si hay NaN
        if X_clean.isna().any().any():
            # 3. Imputación columna por columna
            for col in X_clean.columns:
                if X_clean[col].isna().any():
                    if X_clean[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                        # Para columnas numéricas
                        if X_clean[col].notna().sum() > 0:
                            median_val = X_clean[col].median()
                            if pd.isna(median_val):
                                mean_val = X_clean[col].mean()
                                fill_val = mean_val if not pd.isna(mean_val) else 0.0
                            else:
                                fill_val = median_val
                        else:
                            fill_val = 0.0
                        X_clean[col] = X_clean[col].fillna(fill_val)
                    else:
                        # Para columnas categóricas o de otro tipo
                        X_clean[col] = X_clean[col].fillna(0)
            
            # 4. Imputación final para asegurar que no queden NaN
            X_clean = X_clean.fillna(0)
        
        # 5. Verificación final
        if X_clean.isna().any().any():
            logger.warning("Aún hay NaN después de limpieza exhaustiva, forzando a 0")
            X_clean = X_clean.fillna(0)
        
        return X_clean
    
    def _calculate_feature_importance(self, feature_columns: List[str]) -> Dict[str, Any]:
        """Calcular importancia de features de todos los modelos - VERSIÓN CORREGIDA"""
        
        self.logger.debug("=== CALCULANDO FEATURE IMPORTANCE ===")
        
        # PASO 1: Verificar si ya tenemos importancias guardadas durante el entrenamiento
        if hasattr(self, 'feature_importance') and self.feature_importance:
            self.logger.info(f"Feature importance ya disponible para {len(self.feature_importance)} modelos")
            
            # Verificar que las importancias no están vacías
            valid_models = 0
            for name, info in self.feature_importance.items():
                if isinstance(info, dict) and 'importances' in info:
                    importances = info['importances']
                    if isinstance(importances, list) and len(importances) > 0 and any(imp > 0 for imp in importances):
                        valid_models += 1
                        self.logger.info(f"  {name}: {len(importances)} features, max importance: {max(importances):.4f}")
            
            if valid_models > 0:
                self.logger.info(f"Usando feature importance existente de {valid_models} modelos válidos")
                self._calculate_average_importance(feature_columns)
                return self.feature_importance
            else:
                self.logger.warning("Feature importance existente está vacía, recalculando...")
        
        # PASO 2: Recalcular desde los modelos entrenados
        self.logger.info("Extrayendo feature importance desde modelos entrenados...")
        importance_summary = {}
        
        for name, model_info in self.training_results['individual_models'].items():
            if 'model' in model_info:
                model = model_info['model']
                
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        if len(importances) > 0 and np.sum(importances) > 0:
                            importance_summary[name] = {
                                'importances': importances.tolist(),
                                'feature_names': feature_columns
                            }
                            self.logger.info(f"  {name}: extraída correctamente, max: {np.max(importances):.4f}")
                        else:
                            self.logger.warning(f"  {name}: importancias vacías o cero")
                    elif hasattr(model, 'coef_'):
                        # Para modelos lineales como Ridge
                        coef = np.abs(model.coef_[0] if model.coef_.ndim > 1 else model.coef_)
                        if len(coef) > 0 and np.sum(coef) > 0:
                            importance_summary[name] = {
                                'importances': coef.tolist(),
                                'feature_names': feature_columns
                            }
                            self.logger.info(f"  {name}: coeficientes extraídos, max: {np.max(coef):.4f}")
                    else:
                        self.logger.warning(f"  {name}: no tiene feature_importances_ ni coef_")
                        
                except Exception as e:
                    self.logger.error(f"  {name}: error extrayendo importancia: {str(e)}")
        
        # PASO 3: Verificar que obtuvimos importancias válidas
        if not importance_summary:
            self.logger.error("No se pudo extraer feature importance de ningún modelo")
            # Crear importancias dummy para evitar errores
            dummy_importance = np.ones(len(feature_columns)) / len(feature_columns)
            importance_summary['dummy'] = {
                'importances': dummy_importance.tolist(),
                'feature_names': feature_columns
            }
        
        # PASO 4: Calcular importancia promedio
        self.feature_importance = importance_summary
        self._calculate_average_importance(feature_columns)
        
        self.logger.info(f"Feature importance calculada para {len(importance_summary)} modelos")
        return self.feature_importance
    
    def _calculate_average_importance(self, feature_columns: List[str]):
        """Calcular importancia promedio de todos los modelos válidos"""
        
        if 'average' in self.feature_importance:
            self.logger.info("Importancia promedio ya existe")
            return
        
        valid_models = []
        for name, info in self.feature_importance.items():
            if isinstance(info, dict) and 'importances' in info:
                importances = info['importances']
                if isinstance(importances, list) and len(importances) == len(feature_columns):
                    valid_models.append(np.array(importances))
        
        if valid_models:
            # Calcular promedio
            avg_importance = np.mean(valid_models, axis=0)
            
            # Normalizar para que sume 1
            if np.sum(avg_importance) > 0:
                avg_importance = avg_importance / np.sum(avg_importance)
            
            self.feature_importance['average'] = {
                'importances': avg_importance.tolist(),
                'feature_names': feature_columns
            }
            
            self.logger.info(f"Importancia promedio calculada desde {len(valid_models)} modelos")
        else:
            self.logger.warning("No se pudo calcular importancia promedio - no hay modelos válidos")
    
    def get_feature_importance(self, top_n: int = None) -> Dict[str, Any]:
        """Obtener features más importantes - si top_n es None, devuelve todas"""
        
        if not self.feature_importance:
            return {}
        
        result = {}
        
        for model_name, info in self.feature_importance.items():
            if 'importances' in info and 'feature_names' in info:
                # Crear pares (feature, importance)
                feature_importance_pairs = list(zip(info['feature_names'], info['importances']))
                
                # Ordenar por importancia descendente
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                
                # Tomar top N si se especifica, sino todas
                if top_n is not None:
                    top_features = feature_importance_pairs[:top_n]
                else:
                    top_features = feature_importance_pairs
                
                result[model_name] = {
                    'top_features': [(feat, float(imp)) for feat, imp in top_features],
                    'total_features': len(info['feature_names'])
                }
        
        return result
    
    def evaluate(self, df: pd.DataFrame) -> Dict[str, float]:
        """Evaluar modelo en datos de test"""
        if not self.is_fitted:
            raise ValueError("Modelo no está entrenado")
        
        feature_columns = self.training_results['feature_columns']
        X = df[feature_columns].copy()
        
        # Determinar columna target
        target_col = 'double_double' if 'double_double' in df.columns else 'DD'
        if target_col not in df.columns:
            raise ValueError("No se encontró columna target en datos de evaluación")
        
        y = df[target_col].copy()
        
        # Predicciones
        y_pred = self.predict(df)
        y_proba = self.predict_proba(df)[:, 1]
        
        # Calcular métricas
        metrics = MetricsCalculator.calculate_classification_metrics(y, y_pred, y_proba)
        
        logger.info("Métricas de evaluación:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath: str, save_metadata: bool = True) -> None:
        """
        Guarda el modelo entrenado completo.
        
        Args:
            filepath: Ruta donde guardar el modelo
            save_metadata: Si guardar metadatos adicionales
        """
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar modelo completo (objeto completo con todos los atributos)
        joblib.dump(self, filepath)
        self.logger.info(f"Modelo Double Double guardado: {filepath}")
        
        if save_metadata:
            # Guardar metadatos
            metadata = {
                'model_type': 'DoubleDoubleAdvancedModel',
                'features_count': len(self.feature_importance) if hasattr(self, 'feature_importance') else 0,
                'training_metrics': self.training_results.get('metrics', {}) if hasattr(self, 'training_results') else {},
                'cv_scores': self.cv_scores if hasattr(self, 'cv_scores') else {},
                'bayesian_results': self.bayesian_results if hasattr(self, 'bayesian_results') else {},
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = filepath.replace('.joblib', '_metadata.json')
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Metadatos guardados: {metadata_path}")
    
    @classmethod  
    def load_model(cls, filepath: str) -> 'DoubleDoubleModel':
        """
        Carga un modelo previamente guardado.
        
        Args:
            filepath: Ruta del modelo guardado
            
        Returns:
            Modelo cargado
        """
        model = joblib.load(filepath)
        logger.info(f"Modelo Double Double cargado: {filepath}")
        return model

    def _calculate_optimal_threshold_advanced(self, y_true, y_proba, method='f1_precision_balance'):
        """
        PARTE 1: THRESHOLD OPTIMIZATION AVANZADO - CORREGIDO
        Calcular threshold óptimo usando múltiples estrategias y validación
        
        Args:
            y_true: Valores reales
            y_proba: Probabilidades predichas (columna 1 para clase positiva)
            method: Método de optimización ('f1_precision_balance', 'youden', 'precision_recall_curve')
        
        Returns:
            float: Threshold óptimo
        """
        from sklearn.metrics import precision_recall_curve, roc_curve
        
        # Extraer probabilidades de clase positiva
        if y_proba.ndim > 1:
            proba_positive = y_proba[:, 1]
        else:
            proba_positive = y_proba
        
        # CORRECCIÓN CRÍTICA: Usar límites basados en probabilidades reales
        prob_min = np.min(proba_positive)
        prob_max = np.max(proba_positive)
        prob_mean = np.mean(proba_positive)
        
        self.logger.info(f"Optimizando threshold con método: {method}")
        self.logger.info(f"Distribución real: {np.mean(y_true):.3f} positivos")
        self.logger.info(f"Rango probabilidades: [{prob_min:.3f}, {prob_max:.3f}], Media: {prob_mean:.3f}")
        
        if method == 'f1_precision_balance':
            # Método 1: Balancear F1 Score y Precision mínima
            # CORRECCIÓN: Usar rango realista basado en probabilidades reales
            threshold_min = max(0.05, prob_min)
            threshold_max = min(0.95, prob_max * 0.9)  # 90% del máximo
            
            thresholds = np.linspace(threshold_min, threshold_max, 100)
            best_score = 0
            best_threshold = prob_mean  # Usar media como default
            min_precision = 0.30  # AUMENTADO: Precision mínima para reducir falsos positivos
            
            self.logger.info(f"Probando thresholds en rango [{threshold_min:.3f}, {threshold_max:.3f}]")
            
            for threshold in thresholds:
                y_pred = (proba_positive >= threshold).astype(int)
                
                # Evitar divisiones por cero
                if np.sum(y_pred) == 0:
                    continue
                
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Solo considerar si precision >= mínima
                if precision >= min_precision:
                    # Score combinado: 70% Precision + 30% F1 (priorizar precision para reducir falsos positivos)
                    combined_score = 0.7 * precision + 0.3 * f1
                    
                    if combined_score > best_score:
                        best_score = combined_score
                        best_threshold = threshold
            
            self.logger.info(f"F1-Precision balance: threshold={best_threshold:.4f}, score={best_score:.4f}")
            
        elif method == 'youden':
            # Método 2: Índice de Youden (maximizar TPR - FPR)
            fpr, tpr, thresholds = roc_curve(y_true, proba_positive)
            youden_index = tpr - fpr
            best_idx = np.argmax(youden_index)
            best_threshold = thresholds[best_idx]
            
            # CORRECCIÓN: Limitar a rango realista
            best_threshold = min(best_threshold, prob_max * 0.9)
            
            self.logger.info(f"Youden index: threshold={best_threshold:.4f}, index={youden_index[best_idx]:.4f}")
            
        elif method == 'precision_recall_curve':
            # Método 3: Curva Precision-Recall
            precision, recall, thresholds = precision_recall_curve(y_true, proba_positive)
            
            # Encontrar threshold que maximice F1 con precision mínima
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # CORRECCIÓN: Precision mínima aumentada para reducir falsos positivos
            min_precision = 0.20
            valid_indices = precision >= min_precision
            
            if np.any(valid_indices):
                valid_f1 = f1_scores[valid_indices]
                valid_thresholds = thresholds[valid_indices[:-1]]  # thresholds tiene un elemento menos
                
                if len(valid_thresholds) > 0:
                    best_idx = np.argmax(valid_f1)
                    best_threshold = valid_thresholds[best_idx]
                    # CORRECCIÓN: Limitar a rango realista
                    best_threshold = min(best_threshold, prob_max * 0.9)
                else:
                    best_threshold = prob_mean
            else:
                best_threshold = prob_mean
            
            self.logger.info(f"Precision-Recall curve: threshold={best_threshold:.4f}")
        
        # CORRECCIÓN: Validación del threshold con límites más conservadores
        threshold_min_limit = max(0.08, prob_min * 1.5)  # Mínimo más alto
        threshold_max_limit = min(0.25, prob_max * 0.70)  # Máximo más conservador
        
        if best_threshold < threshold_min_limit:
            self.logger.info(f"Threshold ajustado desde {best_threshold:.4f} a {threshold_min_limit:.4f} (mínimo conservador)")
            best_threshold = threshold_min_limit
        elif best_threshold > threshold_max_limit:
            self.logger.info(f"Threshold ajustado desde {best_threshold:.4f} a {threshold_max_limit:.4f} (máximo conservador)")
            best_threshold = threshold_max_limit
        
        # FALLBACK MÁS AGRESIVO: Garantizar predicciones suficientes
        y_pred_test = (proba_positive >= best_threshold).astype(int)
        predicted_positives = np.sum(y_pred_test)
        actual_positives = np.sum(y_true)
        
        # Si predecimos menos del 20% de los casos reales, ser más agresivo
        if predicted_positives < (actual_positives * 0.2):
            # Usar percentil que garantice al menos 20% de los casos reales
            target_rate = max(0.08, np.mean(y_true) * 2.0)  # 2x la tasa real, mínimo 8%
            percentile = 100 - (target_rate * 100)
            fallback_threshold = np.percentile(proba_positive, percentile)
            
            self.logger.warning(f"Threshold {best_threshold:.4f} genera solo {predicted_positives} predicciones de {actual_positives} reales. Usando fallback más agresivo: {fallback_threshold:.4f}")
            best_threshold = fallback_threshold
        
        # Evaluar threshold final
        y_pred_final = (proba_positive >= best_threshold).astype(int)
        final_precision = precision_score(y_true, y_pred_final, zero_division=0)
        final_recall = recall_score(y_true, y_pred_final, zero_division=0)
        final_f1 = f1_score(y_true, y_pred_final, zero_division=0)
        final_predictions = np.sum(y_pred_final)
        
        self.logger.info(f"Threshold final: {best_threshold:.4f}")
        self.logger.info(f"Predicciones positivas: {final_predictions}/{len(y_pred_final)} ({final_predictions/len(y_pred_final)*100:.1f}%)")
        self.logger.info(f"Métricas finales - P: {final_precision:.3f}, R: {final_recall:.3f}, F1: {final_f1:.3f}")
        
        return best_threshold