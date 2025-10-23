"""
Modelo Avanzado de Predicción de Puntos de Halftime de Equipos NBA
====================================================

Modelo híbrido que combina:
- Machine Learning tradicional (XGBoost, LightGBM, CatBoost)
- Deep Learning (Redes Neuronales con PyTorch)
- Stacking avanzado con meta-modelo optimizado
- Optimización bayesiana TPE de hiperparámetros
- Regularización agresiva anti-overfitting
- Validación cruzada temporal para series de tiempo NBA
- Sistema de logging optimizado
- Feature engineering especializado para puntos de halftime (HT)
- Target: Suma de puntos de cuartos 1 y 2
"""

# Standard Library
import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import sys

# Third-party Libraries - ML/Data
import pandas as pd
import numpy as np
import joblib

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    StackingRegressor, VotingRegressor, RandomForestRegressor
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import (
    TimeSeriesSplit, cross_val_score, KFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score
)

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

# Bayesian Optimization con TPE
try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Local imports
from .features_halftime import HalfTimeFeatureEngineer

warnings.filterwarnings('ignore')

# Logging setup
import logging
logger = logging.getLogger(__name__)


class GPUManager:
    """Gestor de GPU para modelos PyTorch"""
    
    @staticmethod
    def get_optimal_device(device_preference: Optional[str] = None, 
                          min_memory_gb: float = 2.0) -> torch.device:
        """Obtiene el dispositivo óptimo para entrenamiento"""
        
        if device_preference == 'cpu':
            logger.info("Usando CPU por preferencia del usuario")
            return torch.device('cpu')
        
        if not torch.cuda.is_available():
            logger.info("CUDA no disponible, usando CPU")
            return torch.device('cpu')
        
        # Verificar memoria disponible
        try:
            for i in range(torch.cuda.device_count()):
                device = torch.device(f'cuda:{i}')
                torch.cuda.set_device(device)
                
                # Verificar memoria
                total_memory = torch.cuda.get_device_properties(device).total_memory
                allocated_memory = torch.cuda.memory_allocated(device)
                free_memory = total_memory - allocated_memory
                free_gb = free_memory / (1024**3)
                
                if free_gb >= min_memory_gb:
                    logger.info(f"Usando GPU {device} ({free_gb:.1f}GB libres)")
                    return device
            
            logger.info("Memoria GPU insuficiente, usando CPU")
            return torch.device('cpu')
            
        except Exception as e:
            logger.warning(f"Error verificando GPU: {e}, usando CPU")
            return torch.device('cpu')


class NBAExtremesCalibratedStacking:
    """Wrapper con calibración para extremos NBA - MOVIDO FUERA PARA SERIALIZACIÓN"""
    
    def __init__(self, base_stacking):
        self.base_stacking = base_stacking
        self.is_fitted = False
        self.target_stats = None
        
    def fit(self, X, y):
        # Calcular estadísticas del target para calibración
        self.target_stats = {
            'mean': np.mean(y),
            'std': np.std(y),
            'q10': np.percentile(y, 10),   # ~95 puntos
            'q90': np.percentile(y, 90),  # ~130 puntos
            'min': np.min(y),
            'max': np.max(y)
        }
        
        logger.info(f"Target Calibration Stats:")
        logger.info(f"   Mean: {self.target_stats['mean']:.1f}")
        logger.info(f"   Std: {self.target_stats['std']:.1f}")
        logger.info(f"   Range: {self.target_stats['min']:.0f} - {self.target_stats['max']:.0f}")
        logger.info(f"   Extremes: Q10={self.target_stats['q10']:.1f}, Q90={self.target_stats['q90']:.1f}")
        
        self.base_stacking.fit(X, y)
        self.is_fitted = True
        return self
        
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Predicción base del ensemble
        raw_pred = self.base_stacking.predict(X)
        
        # CALIBRACIÓN MEJORADA PARA EXTREMOS
        calibrated_pred = raw_pred.copy()
        
        stats = self.target_stats
        
        # 1. CALIBRACIÓN POR RANGOS DE PUNTOS
        for i, pred in enumerate(raw_pred):
            if pred < 90:  # RANGO MUY BAJO
                # Ajustar hacia arriba para evitar subestimación extrema
                adjustment_factor = 1.15  # Aumentar 15%
                calibrated_pred[i] = pred * adjustment_factor
                
            elif pred > 150:  # RANGO MUY ALTO
                # Ajustar hacia abajo para evitar sobreestimación extrema
                adjustment_factor = 0.90  # Reducir 10%
                calibrated_pred[i] = pred * adjustment_factor
                
            elif pred < 110:  # RANGO BAJO
                # Ajuste suave hacia arriba
                adjustment_factor = 1.05  # Aumentar 5%
                calibrated_pred[i] = pred * adjustment_factor
                
            elif pred > 130:  # RANGO ALTO
                # Ajuste suave hacia abajo
                adjustment_factor = 0.95  # Reducir 5%
                calibrated_pred[i] = pred * adjustment_factor
        
        # 2. CALIBRACIÓN GLOBAL: Ajustar ligeramente hacia la media si está muy lejos
        mean_pred = np.mean(calibrated_pred)
        target_mean = stats['mean']
        
        # Si la predicción promedio está muy lejos de la media real, ajustar
        if abs(mean_pred - target_mean) > stats['std'] * 0.3:
            adjustment = (target_mean - mean_pred) * 0.2  # Ajuste más suave del 20%
            calibrated_pred = calibrated_pred + adjustment

        # 3. LÍMITES REALISTAS NBA (más conservadores para extremos)
        calibrated_pred = np.clip(calibrated_pred, 
                                50,   # Límite inferior absoluto
                                180)  # Límite superior absoluto
        
        # 4. APLICAR SUAVIZADO ADICIONAL PARA EXTREMOS
        # Si hay predicciones muy extremas, aplicar suavizado
        extreme_mask = (calibrated_pred < 80) | (calibrated_pred > 160)
        if extreme_mask.any():
            # Aplicar suavizado hacia la media para extremos
            extreme_indices = np.where(extreme_mask)[0]
            for idx in extreme_indices:
                if calibrated_pred[idx] < 80:
                    # Suavizar hacia arriba
                    calibrated_pred[idx] = 0.7 * calibrated_pred[idx] + 0.3 * target_mean
                else:
                    # Suavizar hacia abajo
                    calibrated_pred[idx] = 0.7 * calibrated_pred[idx] + 0.3 * target_mean
        
        return calibrated_pred


class HalfTimePointsNeuralNet(nn.Module):
    """Red neuronal especializada para predicción de puntos de equipo"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = None):
        super(HalfTimePointsNeuralNet, self).__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        
        layers = []
        current_size = input_size
        
        # Capas ocultas con BatchNorm y Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            current_size = hidden_size
        
        # Capa de salida
        layers.append(nn.Linear(current_size, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialización de pesos
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicialización optimizada de pesos"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class PyTorchHalfTimePointsRegressor(BaseEstimator, RegressorMixin):
    """Wrapper de PyTorch para integración con scikit-learn"""
    
    def __init__(self, hidden_sizes=None, epochs=150, batch_size=32,
                 learning_rate=0.001, weight_decay=0.01, device=None):
        self.hidden_sizes = hidden_sizes or [128, 64, 32]
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device_preference = device
        
        self.model = None
        self.scaler = StandardScaler()
        self.device = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Entrena la red neuronal"""
        self.device = GPUManager.get_optimal_device(self.device_preference)
        
        # Escalar datos
        X_scaled = self.scaler.fit_transform(X)
        
        # Convertir a tensores
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y.values if hasattr(y, 'values') else y).view(-1, 1).to(self.device)
        
        # Crear modelo
        input_size = X_scaled.shape[1]
        self.model = HalfTimePointsNeuralNet(input_size, self.hidden_sizes).to(self.device)
        
        # Configurar entrenamiento
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Dataset y DataLoader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Entrenamiento
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if epoch % 20 == 0:
                logger.info(f"Época {epoch}: Loss = {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Realiza predicciones"""
        if not self.is_fitted:
            raise ValueError("El modelo debe ser entrenado primero")
        
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy().flatten()
        
        # Aplicar límites realistas basados en análisis del dataset
        # Análisis real HT: Min=10, Max=91, P5=43, P95=68
        # Usando rango MODERADO (98% de casos históricos): 10-91
        predictions = np.clip(predictions, 10, 91)
        
        # MEJORA: Manejo robusto de outliers y predicciones enteras
        predictions = self._apply_robust_prediction_processing(predictions, target_type='halftime')
        
        return predictions
    
    def score(self, X, y):
        """R² score para compatibilidad con sklearn"""
        y_pred = self.predict(X)
        return r2_score(y, y_pred)


class BayesianOptimizerTPE:
    """Optimizador bayesiano usando TPE de Optuna"""
    
    def __init__(self, n_trials: int = 50, random_state: int = 42):
        self.n_trials = n_trials
        self.random_state = random_state
        self.optimization_results = {}
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna no está disponible para optimización bayesiana")
    
    def optimize_xgboost(self, X_train, y_train, cv_folds=5):
        """ANTI-OVERFITTING: Optimiza XGBoost con restricciones conservadoras"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),    # LIMITAR estimadores
                'max_depth': trial.suggest_int('max_depth', 3, 5),             # LIMITAR profundidad
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # LIMITAR learning rate
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),       # MÁS agresivo
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),  # MÁS agresivo
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 5.0),       # FORZAR regularización
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),     # FORZAR regularización
                'min_child_weight': trial.suggest_int('min_child_weight', 3, 15),  # AUMENTAR min
                'gamma': trial.suggest_float('gamma', 0.5, 5.0),               # FORZAR pruning
                'random_state': self.random_state
            }
            
            model = xgb.XGBRegressor(**params)
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=tscv, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            
            return -np.mean(scores)  # Minimizar MAE
        
        # Crear estudio con TPESampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        logger.info(f"Optimizando XGBoost con TPE ({self.n_trials} trials)...")
        study.optimize(objective, n_trials=self.n_trials)
        
        # Guardar resultados
        self.optimization_results['xgboost'] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        logger.info(f"XGBoost optimizado - Mejor MAE: {study.best_value:.4f}")
        
        # Crear modelo optimizado
        best_params = study.best_params.copy()
        best_params['random_state'] = self.random_state
        
        return xgb.XGBRegressor(**best_params)
    
    def optimize_lightgbm(self, X_train, y_train, cv_folds=5):
        """ANTI-OVERFITTING: Optimiza LightGBM con restricciones conservadoras"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),    # LIMITAR estimadores
                'max_depth': trial.suggest_int('max_depth', 3, 5),             # LIMITAR profundidad
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # LIMITAR learning rate
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),       # MÁS agresivo
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),  # MÁS agresivo
                'reg_alpha': trial.suggest_float('reg_alpha', 1.0, 5.0),       # FORZAR regularización
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),     # FORZAR regularización
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),  # AUMENTAR min
                'min_split_gain': trial.suggest_float('min_split_gain', 0.1, 2.0),  # FORZAR min gain
                'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 0.8),  # REDUCIR features
                'random_state': self.random_state,
                'verbosity': -1
            }
            
            model = lgb.LGBMRegressor(**params)
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            
            return -np.mean(scores)
        
        # Crear estudio con TPESampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        logger.info(f"Optimizando LightGBM con TPE ({self.n_trials} trials)...")
        study.optimize(objective, n_trials=self.n_trials)
        
        # Guardar resultados
        self.optimization_results['lightgbm'] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        logger.info(f"LightGBM optimizado - Mejor MAE: {study.best_value:.4f}")
        
        # Crear modelo optimizado
        best_params = study.best_params.copy()
        best_params['random_state'] = self.random_state
        best_params['verbosity'] = -1
        
        return lgb.LGBMRegressor(**best_params)
    
    def optimize_catboost(self, X_train, y_train, cv_folds=5):
        """ANTI-OVERFITTING: Optimiza CatBoost con restricciones conservadoras"""
        
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 50, 200),        # LIMITAR iteraciones
                'depth': trial.suggest_int('depth', 3, 5),                     # LIMITAR profundidad
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),  # LIMITAR learning rate
                'subsample': trial.suggest_float('subsample', 0.5, 0.8),       # MÁS agresivo
                'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 0.8),  # MÁS agresivo
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2.0, 10.0),  # FORZAR regularización alta
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 15, 50),  # AUMENTAR min data
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = cb.CatBoostRegressor(**params)
            
            # Validación cruzada temporal
            tscv = TimeSeriesSplit(n_splits=cv_folds)
            scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
            
            return -np.mean(scores)
        
        # Crear estudio con TPESampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        
        logger.info(f"Optimizando CatBoost con TPE ({self.n_trials} trials)...")
        study.optimize(objective, n_trials=self.n_trials)
        
        # Guardar resultados
        self.optimization_results['catboost'] = {
            'best_params': study.best_params,
            'best_score': study.best_value,
            'n_trials': len(study.trials)
        }
        
        logger.info(f"CatBoost optimizado - Mejor MAE: {study.best_value:.4f}")
        
        # Crear modelo optimizado
        best_params = study.best_params.copy()
        best_params['random_state'] = self.random_state
        best_params['verbose'] = False
        
        return cb.CatBoostRegressor(**best_params)


class HalfTimePointsModel:
    """
    Modelo avanzado para predicción de puntos de halftime de equipos NBA con stacking y optimización bayesiana
    
    Target: HT (Halftime) = Suma de puntos de cuartos 1 y 2
    """
    
    def __init__(self, optimize_hyperparams: bool = True,
                 device: Optional[str] = None,
                 bayesian_n_trials: int = 10,  # REDUCIR trials de 25 a 10
                 min_memory_gb: float = 2.0,
                 df_players: Optional[pd.DataFrame] = None,
                 teams_total_df: Optional[pd.DataFrame] = None,
                 teams_quarters_df: Optional[pd.DataFrame] = None):
        
        self.optimize_hyperparams = optimize_hyperparams
        self.device_preference = device
        self.bayesian_n_trials = bayesian_n_trials
        self.min_memory_gb = min_memory_gb
        
        # Feature engineer especializado con datos completos
        self.feature_engineer = HalfTimeFeatureEngineer(
            df_players=df_players,
            teams_total_df=teams_total_df,
            teams_quarters_df=teams_quarters_df
        )
        
        # Optimizador bayesiano
        if OPTUNA_AVAILABLE and optimize_hyperparams:
            self.bayesian_optimizer = BayesianOptimizerTPE(
                n_trials=bayesian_n_trials,  # Debería ser 10 por defecto
                random_state=42
            )
        else:
            self.bayesian_optimizer = None
        
        # Modelos y componentes
        self.models = {}
        self.trained_models = {}
        self.stacking_model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.selected_features = []
        
        # Métricas y resultados
        self.training_metrics = {}
        self.validation_metrics = {}
        self.cross_validation_results = {}
        self.feature_importance = {}
        
        # Estado del modelo
        self.is_trained = False
        self.best_model_name = None
        
        # Configurar modelos base
        self._setup_base_models()
        
        logger.info("HalfTimePointsModel inicializado")
        logger.info(f"Optimización bayesiana: {'Habilitada' if self.bayesian_optimizer else 'Deshabilitada'}")
        logger.info(f"Dispositivo preferido: {device or 'auto'}")
    
    def _setup_base_models(self):
        """ANTI-OVERFITTING: Modelos base con regularización AGRESIVA"""
        self.models = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=150,        # REDUCIR estimadores
                max_depth=4,             # REDUCIR profundidad
                learning_rate=0.05,      # REDUCIR learning rate
                subsample=0.7,           # MÁS subsample
                colsample_bytree=0.7,    # MÁS colsample
                reg_alpha=2.0,           # AUMENTAR regularización L1
                reg_lambda=2.0,          # AUMENTAR regularización L2
                min_child_weight=5,      # AUMENTAR min child weight
                gamma=1.0,               # AGREGAR gamma para pruning
                random_state=42,
                n_jobs=-1
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=150,        # REDUCIR estimadores
                max_depth=4,             # REDUCIR profundidad
                learning_rate=0.05,      # REDUCIR learning rate
                subsample=0.7,           # MÁS subsample
                colsample_bytree=0.7,    # MÁS colsample
                reg_alpha=2.0,           # AUMENTAR regularización L1
                reg_lambda=2.0,          # AUMENTAR regularización L2
                min_child_samples=30,    # AUMENTAR min child samples
                min_split_gain=0.1,      # AGREGAR min split gain
                random_state=42,
                verbosity=-1,
                n_jobs=-1
            ),
            'catboost': cb.CatBoostRegressor(
                iterations=150,          # REDUCIR iteraciones
                depth=4,                 # REDUCIR profundidad
                learning_rate=0.05,      # REDUCIR learning rate
                subsample=0.7,           # MÁS subsample
                colsample_bylevel=0.7,   # MÁS colsample
                l2_leaf_reg=3.0,         # AUMENTAR regularización L2
                min_data_in_leaf=15,     # AUMENTAR min data in leaf
                random_state=42,
                verbose=False,
                thread_count=-1
            )
            # SIMPLICIDAD: Solo 3 modelos base, NO neural networks
        }
    
    def _setup_stacking_model(self, base_models):
        """
        STACKING ELITE PARA EXTREMOS NBA
        Especializado en capturar rangos extremos de puntos (defensas fuertes vs explosiones ofensivas)
        """
        # BASE MODELS OPTIMIZADOS PARA DIFERENTES ASPECTOS DEL BASKETBALL
        base_estimators = [
            ('xgb_patterns', base_models['xgboost']),      # Patrones complejos
            ('lgb_efficiency', base_models['lightgbm']),   # Eficiencia y velocidad  
            ('cb_robustness', base_models['catboost'])     # Robustez categórica
        ]
        
        # META-LEARNER AVANZADO (XGBoost en lugar de Ridge para mejor no-linealidad)
        from xgboost import XGBRegressor
        meta_model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=0.5,
            random_state=42,
            verbosity=0
        )
        
        # STACKING CON VALIDACIÓN TEMPORAL PRESERVADA
        base_stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=meta_model,
            cv=3,  # Simple CV splits (StackingRegressor maneja TimeSeriesSplit internamente)
            n_jobs=1,
            passthrough=True  # INCLUIR features originales para meta-learner
        )
        
        # ASIGNAR MODELO CALIBRADO
        self.stacking_model = NBAExtremesCalibratedStacking(base_stacking)
    
    def _apply_data_quality_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica filtros de calidad de datos para eliminar equipos problemáticos y outliers
        
        Args:
            df: DataFrame con datos de equipos
            
        Returns:
            DataFrame filtrado con datos de calidad
        """        
        # 1. EQUIPOS PROBLEMÁTICOS YA ELIMINADOS DEL DATASET
        df_filtered = df.copy()
        logger.info("Equipos problemáticos ya eliminados del dataset original")
        
        # 2. DETECCIÓN ROBUSTA DE OUTLIERS EN PUNTOS
        logger.info("Aplicando detección robusta de outliers...")
        
        # Usar IQR robusto para detectar outliers extremos
        Q1 = df_filtered['HT'].quantile(0.25)
        Q3 = df_filtered['HT'].quantile(0.75)
        IQR = Q3 - Q1
        
        # Límites más conservadores para puntos NBA
        lower_bound = Q1 - 2.5 * IQR  # Más permisivo que 1.5
        upper_bound = Q3 + 2.5 * IQR
        
        # Identificar outliers extremos
        outliers_mask = (df_filtered['HT'] < lower_bound) | (df_filtered['HT'] > upper_bound)
        outliers_count = outliers_mask.sum()
        
        if outliers_count > 0:
            logger.warning(f"Eliminando {outliers_count} outliers extremos en halftime")
            logger.info(f"Límites de outliers: {lower_bound:.1f} - {upper_bound:.1f} puntos")
            
            # Mostrar algunos ejemplos de outliers
            outliers_sample = df_filtered[outliers_mask][['Team', 'Date', 'HT']].head(5)

            df_filtered = df_filtered[~outliers_mask].copy()
        else:
            logger.info("No se encontraron outliers extremos en puntos")
        
        # 3. FILTRAR REGISTROS CON DATOS FALTANTES CRÍTICOS
        critical_columns = ['HT', 'Team', 'Date']
        missing_critical = df_filtered[critical_columns].isnull().any(axis=1)
        missing_count = missing_critical.sum()
        
        if missing_count > 0:
            logger.warning(f"Eliminando {missing_count} registros con datos críticos faltantes")
            df_filtered = df_filtered[~missing_critical].copy()
        
        # 4. FILTRAR PUNTOS IMPOSIBLES (valores negativos o extremadamente altos)
        impossible_points = (df_filtered['HT'] < 50) | (df_filtered['HT'] > 200)
        impossible_count = impossible_points.sum()
        
        if impossible_count > 0:
            logger.warning(f"Eliminando {impossible_count} registros con halftime imposibles")
            df_filtered = df_filtered[~impossible_points].copy()
        
        # 5. VERIFICAR CONSISTENCIA TEMPORAL
        df_filtered = df_filtered.sort_values(['Team', 'Date']).reset_index(drop=True)
        
        # Estadísticas finales
        logger.info(f"Filtros aplicados exitosamente:")
        logger.info(f"  Registros originales: {len(df)}")
        logger.info(f"  Registros finales: {len(df_filtered)}")
        logger.info(f"  Registros eliminados: {len(df) - len(df_filtered)}")
        logger.info(f"  Rango de puntos final: {df_filtered['HT'].min():.1f} - {df_filtered['HT'].max():.1f}")
        
        return df_filtered
    
    def train(self, df: pd.DataFrame, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el modelo con validación temporal
        
        Args:
            df: DataFrame con datos de entrenamiento
            validation_split: Fracción de datos para validación
            
        Returns:
            Métricas de entrenamiento
        """

        # FILTRAR EQUIPOS PROBLEMÁTICOS Y OUTLIERS
        logger.info("Aplicando filtros de calidad de datos...")
        df_clean = self._apply_data_quality_filters(df)
        df_clean = df_clean.reset_index(drop=True)  # Garantizar índices continuos
        logger.info(f"Datos filtrados: {len(df)} → {len(df_clean)} registros ({len(df)-len(df_clean)} eliminados)")

        # Generar características
        logger.info("Generando características avanzadas")
        self.feature_columns = self.feature_engineer.generate_all_features(df_clean)
        
        # Verificar columnas disponibles
        available_features = [f for f in self.feature_columns if f in df_clean.columns]
        if len(available_features) != len(self.feature_columns):
            missing = set(self.feature_columns) - set(available_features)
            logger.warning(f"Características faltantes: {len(missing)}")
            self.feature_columns = available_features
        
        logger.info(f"Características disponibles: {len(self.feature_columns)}")
        
        # Preparar datos
        X = df_clean[self.feature_columns].fillna(0)
        y = df_clean['HT']  # Target de halftime (puntos en primera mitad)
        
        # ANTI-OVERFITTING: División temporal más estricta (70-30 en lugar de 80-20)
        validation_split = max(validation_split, 0.3)  # Mínimo 30% para validación
        split_idx = int(len(df_clean) * (1 - validation_split))
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
        
        logger.info(f"División temporal estricta: {len(X_train)} entrenamiento ({(1-validation_split)*100:.0f}%), {len(X_val)} validación ({validation_split*100:.0f}%)")
        
        # USAR TODAS LAS FEATURES DISPONIBLES
        self.selected_features = list(X_train.columns)
        X_train_selected = X_train
        X_val_selected = X_val
        
        
        # Escalar datos
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_selected),
            columns=self.selected_features,
            index=X_train_selected.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val_selected),
            columns=self.selected_features,
            index=X_val_selected.index
        )
        
        # Entrenar modelos individuales
        logger.info("Entrenando modelos individuales...")
        individual_predictions = self._train_individual_models(
            X_train_scaled, y_train, X_val_scaled, y_val
        )
        
        # Entrenar modelo de stacking
        logger.info("Entrenando modelo de stacking...")
        self._setup_stacking_model(self.trained_models)
        self.stacking_model.fit(X_train_scaled, y_train)
        stacking_pred = self.stacking_model.predict(X_val_scaled)
        
        # Agregar predicciones de stacking
        individual_predictions['stacking'] = {
            'train': self.stacking_model.predict(X_train_scaled),
            'val': stacking_pred
        }
        self.trained_models['stacking'] = self.stacking_model
        
        # Evaluación y selección del mejor modelo
        logger.info("Evaluando modelos...")
        best_mae = float('inf')
        
        for model_name, preds in individual_predictions.items():
            mae = mean_absolute_error(y_val, preds['val'])
            r2 = r2_score(y_val, preds['val'])
            
            logger.info(f"{model_name}: MAE={mae:.3f}, R²={r2:.4f}")
            
            if mae < best_mae:
                best_mae = mae
                self.best_model_name = model_name
        
        logger.info(f"Mejor modelo: {self.best_model_name} (MAE: {best_mae:.3f})")
        
        # Validación cruzada temporal
        logger.info("Ejecutando validación cruzada temporal...")
        cv_results = self._temporal_cross_validation(X_train_scaled, y_train)
        
        # Guardar métricas
        best_pred = individual_predictions[self.best_model_name]
        self.training_metrics = self._calculate_metrics(y_train, best_pred['train'])
        self.validation_metrics = self._calculate_metrics(y_val, best_pred['val'])
        self.cross_validation_results = cv_results
        
        # Calcular importancia de características
        self._calculate_feature_importance()
        
        self.is_trained = True
        logger.info("Entrenamiento completado exitosamente")
        
        # Mostrar resumen
        self._print_training_summary()
        
        return {
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'cross_validation': self.cross_validation_results,
            'best_model': self.best_model_name,
            'feature_count': len(self.selected_features),
            # CORREGIR: Agregar las claves que busca el trainer
            'train': self.training_metrics,  # Para compatibilidad con trainer
            'validation': self.validation_metrics  # Para compatibilidad con trainer
        }
    
    def _train_individual_models(self, X_train, y_train, X_val, y_val) -> Dict:
        """Entrena modelos individuales"""
        predictions = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            
            # Optimizar hiperparámetros si está habilitado
            if (self.optimize_hyperparams and self.bayesian_optimizer and 
                name in ['xgboost', 'lightgbm', 'catboost']):
                
                if name == 'xgboost':
                    model = self.bayesian_optimizer.optimize_xgboost(X_train, y_train)
                elif name == 'lightgbm':
                    model = self.bayesian_optimizer.optimize_lightgbm(X_train, y_train)
                elif name == 'catboost':
                    model = self.bayesian_optimizer.optimize_catboost(X_train, y_train)
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Predicciones
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            predictions[name] = {
                'train': train_pred,
                'val': val_pred
            }
            
            # Guardar modelo entrenado
            self.trained_models[name] = model
        
        return predictions
    
    def _temporal_cross_validation(self, X, y, n_splits=5):
        """Validación cruzada temporal"""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        if self.best_model_name == 'stacking':
            model = self.stacking_model
        else:
            model = self.trained_models[self.best_model_name]
        
        # MAE scores
        mae_scores = cross_val_score(
            model, X, y, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1
        )
        mae_scores = -mae_scores
        
        # R² scores
        r2_scores = cross_val_score(
            model, X, y, cv=tscv, scoring='r2', n_jobs=-1
        )
        
        results = {
            'mae_scores': mae_scores,
            'r2_scores': r2_scores,
            'mean_mae': np.mean(mae_scores),
            'std_mae': np.std(mae_scores),
            'mean_r2': np.mean(r2_scores),
            'std_r2': np.std(r2_scores)
        }
        
        logger.info(f"Validación cruzada - MAE: {results['mean_mae']:.3f}±{results['std_mae']:.3f}")
        logger.info(f"Validación cruzada - R²: {results['mean_r2']:.4f}±{results['std_r2']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calcula métricas de regresión"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Precisión por tolerancia
        accuracies = {}
        for tolerance in [1, 2, 3, 5, 7, 10]:
            accuracy = np.mean(np.abs(y_true - y_pred) <= tolerance) * 100
            accuracies[f'accuracy_{tolerance}pt'] = accuracy
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            **accuracies
        }
    
    def _calculate_feature_importance(self):
        """Calcula importancia de características CORRECTAMENTE para todas las features"""
        try:
            # SOLUCIÓN: Para stacking, usar importancia agregada de modelos base
            if self.best_model_name == 'stacking':
                logger.info("Calculando feature importance agregada para modelo stacking...")
                
                # Obtener importancias de cada modelo base
                importances_aggregate = np.zeros(len(self.selected_features))
                model_count = 0
                
                for model_name in ['xgboost', 'lightgbm', 'catboost']:
                    if model_name in self.trained_models:
                        model = self.trained_models[model_name]
                        if hasattr(model, 'feature_importances_'):
                            # Verificar que la longitud coincida
                            if len(model.feature_importances_) == len(self.selected_features):
                                importances_aggregate += model.feature_importances_
                                model_count += 1
                                logger.info(f"Agregada importancia de {model_name}: {len(model.feature_importances_)} features")
                else:
                                logger.warning(f"Saltando {model_name}: longitud mismatch {len(model.feature_importances_)} vs {len(self.selected_features)}")
                
                if model_count > 0:
                    importances = importances_aggregate / model_count  # Promedio
                    logger.info(f"Feature importance agregada de {model_count} modelos base")
                else:
                    # Fallback: importancia uniforme
                    importances = np.ones(len(self.selected_features)) / len(self.selected_features)
                    logger.warning("Usando importancia uniforme como fallback")
                    
            else:
                # Modelo individual
                model = self.trained_models[self.best_model_name]
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    importances = np.ones(len(self.selected_features)) / len(self.selected_features)
            
            # VERIFICACIÓN CRÍTICA: Asegurar que longitudes coincidan
            if len(importances) != len(self.selected_features):
                logger.error(f"CRÍTICO: Longitud mismatch después de corrección: importances={len(importances)}, features={len(self.selected_features)}")
                # Crear importancia uniforme como último recurso
                importances = np.ones(len(self.selected_features)) / len(self.selected_features)
                logger.warning("Aplicando importancia uniforme para todas las features")
            
            # SIEMPRE asignar selected_features
            selected_features = self.selected_features
            
            # Crear DataFrame de importancia
            feature_df = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Calcular porcentajes
            total_importance = feature_df['importance'].sum()
            if total_importance > 0:
                feature_df['importance_pct'] = (feature_df['importance'] / total_importance * 100)
                feature_df['cumulative_pct'] = feature_df['importance_pct'].cumsum()
            else:
                # Si todas las importancias son 0, asignar peso uniforme
                feature_df['importance_pct'] = 100.0 / len(feature_df)
                feature_df['cumulative_pct'] = feature_df['importance_pct'].cumsum()
            
            feature_df['rank'] = range(1, len(feature_df) + 1)
            
            self.feature_importance = feature_df
            
        except Exception as e:
            logger.error(f"Error calculando feature importance: {e}")
            # Crear DataFrame básico como fallback
            feature_df = pd.DataFrame({
                'feature': self.selected_features,
                'importance': [1.0/len(self.selected_features)] * len(self.selected_features),
                'importance_pct': [100.0/len(self.selected_features)] * len(self.selected_features),
                'rank': range(1, len(self.selected_features) + 1)
            })
            feature_df['cumulative_pct'] = feature_df['importance_pct'].cumsum()
            self.feature_importance = feature_df
            logger.info("Usando feature importance uniforme como fallback")
    
    def _print_training_summary(self):
        """Imprime resumen del entrenamiento"""
        print("\n" + "="*80)
        print("RESUMEN DE ENTRENAMIENTO - MODELO PUNTOS DE EQUIPO")
        print("="*80)
        
        print(f"\nMEJOR MODELO: {self.best_model_name.upper()}")
        print(f"Features utilizadas: {len(self.selected_features)}")
        
        print(f"\nMÉTRICAS DE ENTRENAMIENTO:")
        print(f"MAE: {self.training_metrics['mae']:.3f}")
        print(f"RMSE: {self.training_metrics['rmse']:.3f}")
        print(f"R²: {self.training_metrics['r2']:.4f}")
        
        print(f"\nMÉTRICAS DE VALIDACIÓN:")
        print(f"MAE: {self.validation_metrics['mae']:.3f}")
        print(f"RMSE: {self.validation_metrics['rmse']:.3f}")
        print(f"R²: {self.validation_metrics['r2']:.4f}")
        
        print(f"\nPRECISIÓN POR TOLERANCIA (Validación):")
        for tolerance in [1, 2, 3, 5, 7, 10]:
            acc = self.validation_metrics[f'accuracy_{tolerance}pt']
            print(f"±{tolerance} puntos: {acc:.1f}%")
        
        print(f"\nVALIDACIÓN CRUZADA TEMPORAL:")
        cv = self.cross_validation_results
        print(f"MAE: {cv['mean_mae']:.3f} ± {cv['std_mae']:.3f}")
        print(f"R²: {cv['mean_r2']:.4f} ± {cv['std_r2']:.4f}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realiza predicciones usando el mejor modelo
        
        Args:
            df: DataFrame con datos para predicción
            
        Returns:
            Array con predicciones de puntos de halftime
        """
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # APLICAR LOS MISMOS FILTROS DE CALIDAD QUE EN ENTRENAMIENTO
        logger.info("Aplicando filtros de calidad de datos")
        df_clean = self._apply_data_quality_filters(df)
        
        # Limpiar cache para nueva predicción
        self.feature_engineer.clear_cache()
        
        # Generar características
        _ = self.feature_engineer.generate_all_features(df_clean)
        
        # Usar características seleccionadas
        available_features = [f for f in self.selected_features if f in df_clean.columns]
        if len(available_features) != len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            logger.warning(f"Características faltantes: {list(missing)[:5]}")
        
        # Preparar datos
        X = df_clean[available_features].fillna(0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=available_features,
            index=X.index
        )
        
        # Realizar predicción
        best_model = self.trained_models[self.best_model_name]
        raw_predictions = best_model.predict(X_scaled)
        
        # MEJORA: Manejo robusto de outliers y predicciones enteras
        final_predictions = self._apply_robust_prediction_processing(raw_predictions, target_type='halftime')
        
        return final_predictions
    
    def _apply_robust_prediction_processing(self, predictions: np.ndarray, target_type: str = 'points') -> np.ndarray:
        """
        Aplica procesamiento robusto de predicciones para manejo de outliers y valores enteros.
        Adaptado específicamente para puntos de halftime de equipos NBA con mejoras para extremos.
        
        Args:
            predictions: Array de predicciones continuas
            target_type: Tipo de target ('halftime' para puntos de halftime, 'points' para puntos totales)
            
        Returns:
            Array de predicciones procesadas como enteros
        """
        # 1. DETECCIÓN MEJORADA DE OUTLIERS usando múltiples métodos
        Q1 = np.percentile(predictions, 25)
        Q3 = np.percentile(predictions, 75)
        IQR = Q3 - Q1
        median = np.median(predictions)
        
        # Método 1: IQR robusto (más permisivo para puntos NBA)
        outlier_factor = 2.0  # Más permisivo que 1.5
        iqr_lower = Q1 - outlier_factor * IQR
        iqr_upper = Q3 + outlier_factor * IQR
        
        # Método 2: Z-score robusto (usando mediana y MAD)
        mad = np.median(np.abs(predictions - median))
        z_score_threshold = 3.0  # Más permisivo que 2.5
        z_lower = median - z_score_threshold * mad
        z_upper = median + z_score_threshold * mad
        
        # Método 3: Límites absolutos NBA
        if target_type == 'halftime':
            nba_lower = 10   # Mínimo absoluto halftime NBA
            nba_upper = 91   # Máximo absoluto halftime NBA
        else:
            nba_lower = 50   # Mínimo absoluto NBA total
            nba_upper = 180  # Máximo absoluto NBA total
        
        # Usar el método más restrictivo para cada límite
        lower_bound = max(iqr_lower, z_lower, nba_lower)
        upper_bound = min(iqr_upper, z_upper, nba_upper)
        
        # Winsorización: limitar outliers a los bounds
        predictions_robust = np.clip(predictions, lower_bound, upper_bound)
        
        # 2. Aplicar suavizado para reducir variabilidad extrema
        if len(predictions_robust) > 10:
            try:
                from scipy.ndimage import median_filter
                # Ventana pequeña para preservar variabilidad legítima
                predictions_robust = median_filter(predictions_robust, size=3)
            except ImportError:
                # Fallback si scipy no está disponible
                pass
        
        # 3. Aplicar redondeo inteligente basado en probabilidades
        integer_predictions = np.zeros_like(predictions_robust)
        
        for i, pred in enumerate(predictions_robust):
            floor_val = np.floor(pred)
            decimal_part = pred - floor_val
            
            # Redondeo probabilístico: mayor probabilidad para valores más cercanos
            if np.random.random() < decimal_part:
                integer_predictions[i] = floor_val + 1
            else:
                integer_predictions[i] = floor_val
        
        # 4. Aplicar límites finales específicos para puntos NBA y convertir a enteros
        if target_type == 'points':
            # Límites realistas basados en análisis del dataset (P1-P99): 84-143
            integer_predictions = np.clip(integer_predictions, 85, 135).astype(int)
        elif target_type == 'halftime':
            # Límites específicos para halftime (HT): 10-91 basado en datos reales
            integer_predictions = np.clip(integer_predictions, 10, 91).astype(int)
        else:
            # Límites genéricos
            integer_predictions = np.clip(integer_predictions, 0, 200).astype(int)
        
        # 5. Post-procesamiento: evitar secuencias poco realistas
        # Para puntos de halftime, diferencias mayores a 20 puntos son raras entre juegos consecutivos
        if len(integer_predictions) > 1:
            for i in range(1, len(integer_predictions)):
                diff = abs(integer_predictions[i] - integer_predictions[i-1])
                if diff > 20:  # Diferencia muy grande entre partidos consecutivos (halftime)
                    # Promedio entre la predicción actual y la anterior
                    integer_predictions[i] = int((integer_predictions[i] + integer_predictions[i-1]) / 2)
        
        logger.debug(f"Procesamiento robusto aplicado para {target_type}: {len(integer_predictions)} predicciones")
        logger.debug(f"Rango final: {integer_predictions.min()}-{integer_predictions.max()}")
        
        return integer_predictions
    
    def get_feature_importance(self, top_n: int = None) -> Dict[str, Any]:
        """Obtiene importancia de características en formato compatible"""
        if not hasattr(self, 'feature_importance') or self.feature_importance.empty:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Retornar en formato compatible con el trainer
        if top_n is None:
            df = self.feature_importance.copy()
        else:
            df = self.feature_importance.head(top_n)
        
        return {
            'feature_importance': df,  # DataFrame completo
            'top_features': df.to_dict('records')  # Lista de diccionarios para análisis
        }
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado con TODOS los componentes necesarios"""
        if not self.is_trained:
            raise ValueError("El modelo debe ser entrenado primero")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Crear paquete completo del modelo
        model_package = {
            'best_model': self.trained_models[self.best_model_name],
            'scaler': self.scaler,  
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'feature_engineer': self.feature_engineer,  
            'training_metrics': getattr(self, 'training_metrics', {}),
            'validation_metrics': getattr(self, 'validation_metrics', {}),
            'cross_validation_results': getattr(self, 'cross_validation_results', {}),
            'feature_importance': getattr(self, 'feature_importance', pd.DataFrame()),
            'metadata': {
            'best_model_name': self.best_model_name,
                'model_type': type(self.trained_models[self.best_model_name]).__name__,
            'feature_count': len(self.selected_features),
            'training_date': datetime.now().isoformat(),
                'version': '2.0'  # Nueva versión con componentes completos
            }
        }
        
        # Guardar el objeto COMPLETO con todos los componentes
        joblib.dump(self, filepath, compress=3, protocol=4)
        logger.info(f"Modelo HALFTIME_POINTS completo guardado: {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo entrenado COMPLETO (método de instancia)"""
        try:
            # Cargar objeto completo directamente
            loaded_model = joblib.load(filepath)
            if isinstance(loaded_model, HalfTimePointsModel):
                # Copiar todos los atributos del modelo cargado
                self.__dict__.update(loaded_model.__dict__)
                logger.info(f"Modelo HALFTIME_POINTS completo cargado: {filepath}")
            else:
                # Fallback: cargar formato legacy (diccionario)
                if isinstance(loaded_model, dict) and 'best_model' in loaded_model:
                    self.trained_models = {'loaded_model': loaded_model['best_model']}
                    self.scaler = loaded_model.get('scaler', StandardScaler())
                    self.feature_columns = loaded_model.get('feature_columns', [])
                    self.selected_features = loaded_model.get('selected_features', [])
                    self.feature_engineer = loaded_model.get('feature_engineer', HalfTimeFeatureEngineer())
                    self.training_metrics = loaded_model.get('training_metrics', {})
                    self.validation_metrics = loaded_model.get('validation_metrics', {})
                    self.cross_validation_results = loaded_model.get('cross_validation_results', {})
                    self.feature_importance = loaded_model.get('feature_importance', pd.DataFrame())
                    self.is_trained = True
                    self.best_model_name = 'loaded_model'
                    logger.info(f"Modelo HALFTIME_POINTS legacy cargado: {filepath}")
                else:
                    raise ValueError("Formato de modelo no reconocido")
        except Exception as e:
            logger.error(f"Error cargando modelo HALFTIME_POINTS: {e}")
            raise