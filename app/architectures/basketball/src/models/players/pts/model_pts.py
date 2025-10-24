"""
Modelo Ensemble Optimizado para Predicción de Puntos de un jugador NBA
=======================================================

Modelo optimizado con solo LightGBM + CatBoost (97.4% del peso ensemble)
para predecir puntos que anotará un jugador en su próximo partido.

Arquitectura OPTIMIZADA:
- Solo LightGBM + CatBoost (eliminados XGBoost y Ridge por redundancia y sobrefitting)
- Meta-learner sofisticado LightGBM para combinar predicciones
- Validación cruzada temporal
- Optimización bayesiana con Optuna (TPESampler)
- Early stopping para prevenir overfitting
- Métricas de evaluación robustas
- Ensemble máximamente sofisticado
"""

# Standard Library
import json
import logging
import os
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import time
import joblib

# Data Science (Manejo de datos)
import numpy as np
import pandas as pd

#Machine Learning  
# IMPORTAR XGBOOST PRIMERO (como en DD model) para evitar conflictos de dependencias
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Bayesian Optimization
import optuna
from optuna.samplers import TPESampler

# Scikit-learn (Metricas, Modelos, Preprocesamiento)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import (
    Lasso, 
    ElasticNet, 
    LinearRegression,
    BayesianRidge,
    Ridge
)

# Feature Engineer (Características especializadas para predicción de puntos)
from .features_pts import PointsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

xgb.set_config(verbosity=0)

# Configurar LightGBM para no mostrar logs
import os
os.environ['LIGHTGBM_VERBOSE'] = '0'

# Configurar CatBoost para no mostrar logs
os.environ['CATBOOST_VERBOSE'] = '0'


class StackingPTSModel:
    """
    Modelo de Stacking Ensemble para Predicción de Puntos de jugador NBA
    
    Combina múltiples algoritmos de ML/DL con:
    - Regularización avanzada en todos los modelos
    - Early stopping inteligente
    - Validación cruzada temporal estricta
    - Optimización bayesiana por modelo
    - Meta-learner adaptativo
    """
    
    def __init__(self,
                 n_trials: int = 25,  # Optimización bayesiana por modelo
                 cv_folds: int = 3,
                 early_stopping_rounds: int = 30,
                 random_state: int = 42,
                 enable_gpu: bool = False,
                 teams_df: pd.DataFrame = None,
                 players_df: pd.DataFrame = None,
                 players_quarters_df: pd.DataFrame = None):
        """
        Inicializa el modelo de stacking ensemble.
        
        Args:
            n_trials: Número de trials para optimización bayesiana por modelo (TPESampler)
            cv_folds: Número de folds para validación cruzada
            early_stopping_rounds: Rounds para early stopping
            random_state: Semilla para reproducibilidad
            enable_gpu: Si usar GPU para modelos compatibles
            teams_df: DataFrame con datos de equipos
            players_df: DataFrame con datos de jugadores (total)
            players_quarters_df: DataFrame con datos de jugadores por cuartos
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds # Reducido a 3 folds para mejor rendimiento
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.enable_gpu = enable_gpu
        
        # Componentes principales con datasets ultra-avanzados
        self.feature_engineer = PointsFeatureEngineer(
            teams_df=teams_df, 
            players_df=players_df,
            players_quarters_df=players_quarters_df
        )
        self.scaler = StandardScaler()
        
        # Modelos base
        self.base_models = {}
        self.trained_base_models = {}
        self.best_params_per_model = {}
        
        # Modelo de stacking
        self.stacking_model = None
        self.meta_learner = None
        
        # Métricas y resultados
        self.training_metrics = {}
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        
        # Features utilizadas
        self.selected_features = []
        self.feature_names = []
        
        # Estado del modelo
        self.is_trained = False
        
        # Configurar modelos base
        self._setup_base_models()
        
        logger.info("Modelo Stacking PTS inicializado")
        logger.info(f"Modelos habilitados: GPU={self.enable_gpu}")
    
    def _setup_base_models(self):
        """
        Configuración de stacking ensemble con 4 modelos base + Ridge meta-learner.
        Refactorizado siguiendo la arquitectura optimizada de TRB/AST/3PT.
        Mantiene parámetros optimizados existentes y agrega OOF.
        """
        logger.info("============================================================")
        logger.info("STACKING ARCHITECTURE OPTIMIZED")
        logger.info("============================================================")
        
        # 1. XGBoost - Tree-based con regularización anti-overfitting
        self.base_models['xgboost'] = {
            'model_class': xgb.XGBRegressor,
            'param_space': {
                'n_estimators': (100, 300),
                'learning_rate': (0.05, 0.15),
                'max_depth': (3, 6),
                'min_child_weight': (1, 5),
                'subsample': (0.8, 0.95),
                'colsample_bytree': (0.8, 0.95),
                'reg_alpha': (0, 1),
                'reg_lambda': (1, 3)
            },
            'fixed_params': {
                'random_state': self.random_state,
                'n_jobs': -1,
                'verbosity': 0,
                'early_stopping_rounds': self.early_stopping_rounds
            },
            'early_stopping': True
        }
        
        # 2. LightGBM - Parámetros normalizados
        self.base_models['lightgbm'] = {
            'model_class': lgb.LGBMRegressor,
            'param_space': {
                'n_estimators': (100, 1000),
                'learning_rate': (0.01, 0.3),
                'max_depth': (3, 10),
                'num_leaves': (10, 50),
                'min_child_samples': (1, 20),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (0, 10),
                'min_split_gain': (0, 1)
            },
            'fixed_params': {
                'objective': 'regression',
                'metric': 'mae',
                'random_state': self.random_state,
                'verbosity': -1,
                'n_jobs': -1
            },
            'early_stopping': True
        }
        
        # 3. CatBoost - Parámetros normalizados
        self.base_models['catboost'] = {
            'model_class': cb.CatBoostRegressor,
            'param_space': {
                'iterations': (100, 1000),
                'learning_rate': (0.01, 0.3),
                'depth': (3, 10),
                'l2_leaf_reg': (1, 20),
                'border_count': (32, 255),
                'subsample': (0.6, 1.0),
                'rsm': (0.6, 1.0)
            },
            'fixed_params': {
                'loss_function': 'MAE',
                'random_seed': self.random_state,
                'verbose': False,
                'allow_writing_files': False,
                'thread_count': -1
            },
            'early_stopping': True
        }
        
        
        # 5. Meta-learner Ridge con parámetros fijos (como TRB)
        self.meta_learner_config = {
            'model_class': Ridge,
            'fixed_params': {
                'alpha': 20.0,                    # Regularización ALTA en meta-nivel
                'fit_intercept': True,
                'max_iter': 3000,
                'solver': 'auto',
                'random_state': self.random_state
            }
        }
        
        # Inicializar atributos para OOF
        self.trained_base_models = {}
        self.meta_learner = None
        self.oof_predictions = None
        self.base_models_performance = {}
        self.meta_learner_performance = {}
        self.cv_scores = []
        self.feature_importance = {}
        self.is_trained = False
        
        logger.info(f"Base models: {len(self.base_models)}")
        logger.info(f"  - Tree-based: XGBoost, LightGBM, CatBoost")
        logger.info(f"  - Meta-learner: Ridge (alpha={self.meta_learner_config['fixed_params']['alpha']})")
        logger.info(f"  - CV folds: {self.cv_folds} (optimized)")
        logger.info("============================================================")

    def _optimize_hyperparameters(self, model_name: str, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """
        Optimización bayesiana para un modelo individual.
        
        Args:
            model_name: Nombre del modelo a optimizar
            X_train: Datos de entrenamiento
            y_train: Target de entrenamiento
            X_val: Datos de validación
            y_val: Target de validación
            
        Returns:
            Dict con mejores parámetros y score
        """
        logger.info(f"Optimizando hiperparámetros para {model_name}...")
        
        model_config = self.base_models[model_name]
        
        def objective(trial):
            # Construir parámetros del trial
            params = model_config['fixed_params'].copy()
            
            for param_name, param_range in model_config['param_space'].items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, *param_range)
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, *param_range)
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # Crear y entrenar modelo
            model = model_config['model_class'](**params)
            
            # Calcular pesos adaptativos - REACTIVADO CON AJUSTES SUTILES
            sample_weights = self._calculate_adaptive_weights(y_train.values)
            
            try:
                # Función de entrenamiento
                def train_model():
                    # Entrenar con early stopping si es compatible
                    if model_config['early_stopping'] and hasattr(model, 'fit'):
                        if 'lightgbm' in model_name.lower():
                            try:
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    callbacks=[
                                        lgb.early_stopping(self.early_stopping_rounds),
                                        lgb.log_evaluation(0)
                                    ]
                                )
                            except Exception as lgb_error:
                                # Fallback sin early stopping
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                        elif 'catboost' in model_name.lower():
                            try:
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=(X_val, y_val),
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except Exception as cb_error:
                                # Fallback sin early stopping
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                        elif 'xgboost' in model_name.lower():
                            try:
                                model.fit(
                                    X_train, y_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_val, y_val)],
                                    verbose=False
                                )
                            except Exception as xgb_error:
                                # Fallback sin early stopping
                                model.fit(X_train, y_train, sample_weight=sample_weights)
                        else:
                            # Para modelos con early stopping nativo (sklearn)
                            model.fit(X_train, y_train, sample_weight=sample_weights)
                    else:
                        # Entrenamiento estándar
                        model.fit(X_train, y_train, sample_weight=sample_weights)
                        return X_train, y_train
                
                # Ejecutar entrenamiento
                train_result = train_model()
                if train_result:
                    X_train_used, y_train_used = train_result
                else:
                    X_train_used, y_train_used = X_train, y_train
                
                # Predecir y evaluar
                y_pred = model.predict(X_val)
                
                # Calcular score con pesos adaptativos - REACTIVADO CON PESOS SUTILES
                val_weights = self._calculate_adaptive_weights(y_val.values)
                weighted_errors = np.abs(y_pred - y_val.values) * val_weights
                weighted_mae = np.mean(weighted_errors)
                
                # Penalizaciones SUTILES basadas en el análisis del modelo base
                # Sesgos identificados: +2.08 (bajos), -1.94 (altos) - Penalizaciones sutiles
                
                # Penalización por subestimación en rangos altos (sutil)
                high_scoring_mask = y_val.values >= 25
                if high_scoring_mask.sum() > 0:
                    high_scoring_errors = y_pred[high_scoring_mask] - y_val.values[high_scoring_mask]
                    underestimation_penalty = np.mean(np.maximum(-high_scoring_errors, 0)) * 0.1  # Penalización sutil
                else:
                    underestimation_penalty = 0
                
                # Penalización por sobrestimación en rangos bajos (sutil)
                low_scoring_mask = y_val.values < 10
                if low_scoring_mask.sum() > 0:
                    low_scoring_errors = y_val.values[low_scoring_mask] - y_pred[low_scoring_mask]
                    overestimation_penalty = np.mean(np.maximum(low_scoring_errors, 0)) * 0.05  # Penalización mínima
                else:
                    overestimation_penalty = 0
                
                # Penalización por overfitting SUTIL
                train_pred = model.predict(X_train_used)
                train_mae = mean_absolute_error(y_train_used, train_pred)
                overfitting_penalty = max(0, train_mae - weighted_mae) * 0.05  # Penalización mínima
                
                # Penalización por complejidad SUTIL
                complexity_penalty = 0
                if 'max_depth' in params:
                    complexity_penalty += (params['max_depth'] / 20.0) * 0.01  # Penalización mínima
                if 'n_estimators' in params:
                    complexity_penalty += (params['n_estimators'] / 2000.0) * 0.01  # Penalización mínima
                elif 'max_iter' in params:
                    complexity_penalty += (params['max_iter'] / 1000.0) * 0.01  # Penalización mínima
                
                final_score = weighted_mae + underestimation_penalty + overestimation_penalty + overfitting_penalty + complexity_penalty
                
                return final_score
                
            except Exception as e:
                logger.warning(f"Error en optimización de {model_name}: {e}")
                return float('inf')
        
        # Crear estudio Optuna
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # SILENCIAR COMPLETAMENTE OPTUNA
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        
        n_trials_model = self.n_trials
        
        # Optimizar SIN VERBOSIDAD
        try:
            study.optimize(
                objective, 
                n_trials=n_trials_model, 
                show_progress_bar=False,
                catch=(Exception,)
            )
        except Exception as e:
            logger.error(f"Error en optimización de {model_name}: {e}")
            # Usar parámetros por defecto si falla la optimización
            best_params = model_config['fixed_params']
            best_score = float('inf')
        else:
            best_params = {**model_config['fixed_params'], **study.best_params}
            best_score = study.best_value
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'study': study if 'study' in locals() else None
        }
    
    def _train_base_models_with_oof(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena modelos base con out-of-fold predictions para prevenir data leakage.
        Utiliza optimización bayesiana en cada fold y mantiene sample weights.
        """
        
        # Configurar TimeSeriesSplit para out-of-fold
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Almacenar predicciones out-of-fold y y_true por fold
        oof_predictions = np.zeros((len(X), len(self.base_models)))
        self.trained_base_models = {}
        self.best_params_per_model = {}
        self.oof_y_true = []  # Guardar y_true por fold para métricas CV
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_fold_train = X.iloc[train_idx].copy()
            y_fold_train = y.iloc[train_idx].copy()
            X_fold_val = X.iloc[val_idx].copy()
            y_fold_val = y.iloc[val_idx].copy()
            
            # Guardar y_true para este fold
            self.oof_y_true.append(y_fold_val.values)
            
            logger.info(f"Fold {fold_idx + 1}/{self.cv_folds}")
            
            for model_idx, (model_name, model_config) in enumerate(self.base_models.items()):
                logger.info(f"  Optimizando y entrenando {model_name}...")
                
                try:
                    # Optimizar hiperparámetros para este fold
                    opt_result = self._optimize_hyperparameters(
                        model_name, X_fold_train, y_fold_train, X_fold_val, y_fold_val
                    )
                    
                    # Crear modelo con mejores parámetros
                    best_params = opt_result['best_params']
                    model_class = model_config['model_class']
                    model = model_class(**best_params)
                    
                    # Calcular sample weights para este fold - REACTIVADO CON PESOS SUTILES
                    sample_weights = self._calculate_adaptive_weights(y_fold_train.values)
                    
                    # Entrenar modelo con sample weights y early stopping
                    if model_config.get('early_stopping', False) and hasattr(model, 'fit'):
                        if 'lightgbm' in model_name.lower():
                            try:
                                model.fit(
                                    X_fold_train, y_fold_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_fold_val, y_fold_val)],
                                    callbacks=[
                                        lgb.early_stopping(self.early_stopping_rounds),
                                        lgb.log_evaluation(0)
                                    ]
                                )
                            except Exception as lgb_error:
                                logger.warning(f"LightGBM early stopping falló: {lgb_error}")
                                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                        elif 'catboost' in model_name.lower():
                            try:
                                model.fit(
                                    X_fold_train, y_fold_train,
                                    sample_weight=sample_weights,
                                    eval_set=(X_fold_val, y_fold_val),
                                    early_stopping_rounds=self.early_stopping_rounds,
                                    verbose=False
                                )
                            except Exception as cb_error:
                                logger.warning(f"CatBoost early stopping falló: {cb_error}")
                                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                        elif 'xgboost' in model_name.lower():
                            try:
                                # XGBoost con early stopping usando eval_set
                                model.fit(
                                    X_fold_train, y_fold_train,
                                    sample_weight=sample_weights,
                                    eval_set=[(X_fold_val, y_fold_val)],
                                    verbose=False
                                )
                            except Exception as xgb_error:
                                logger.warning(f"XGBoost early stopping falló: {xgb_error}")
                                model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                        else:
                            model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                    else:
                        model.fit(X_fold_train, y_fold_train, sample_weight=sample_weights)
                    
                    # Predicciones out-of-fold
                    fold_predictions = model.predict(X_fold_val)
                    oof_predictions[val_idx, model_idx] = fold_predictions
                    
                    # Guardar modelo entrenado (solo el último fold)
                    if fold_idx == self.cv_folds - 1:
                        self.trained_base_models[model_name] = model
                        self.best_params_per_model[model_name] = best_params
                    
                    logger.info(f"  {model_name} fold {fold_idx + 1} - MAE: {opt_result['best_score']:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error entrenando {model_name} en fold {fold_idx + 1}: {e}")
                    # Crear modelo con parámetros por defecto como fallback
                    model_class = model_config['model_class']
                    fixed_params = model_config['fixed_params']
                    model = model_class(**fixed_params)
                    model.fit(X_fold_train, y_fold_train)
                    fold_predictions = model.predict(X_fold_val)
                    oof_predictions[val_idx, model_idx] = fold_predictions
                    
                    if fold_idx == self.cv_folds - 1:
                        self.trained_base_models[model_name] = model
                        self.best_params_per_model[model_name] = fixed_params
        
        # Almacenar predicciones out-of-fold
        self.oof_predictions = oof_predictions
        
        # Calcular métricas de modelos base
        for model_idx, (model_name, _) in enumerate(self.base_models.items()):
            model_oof = oof_predictions[:, model_idx]
            mae = mean_absolute_error(y, model_oof)
            r2 = r2_score(y, model_oof)
            
            self.base_models_performance[model_name] = {
                'mae': mae,
                'r2': r2
            }
            
            logger.info(f"  {model_name} OOF - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    def _train_meta_learner_with_oof(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Entrena meta-learner Ridge con predicciones out-of-fold.
        """
        logger.info("Entrenando meta-learner con predicciones out-of-fold...")
        
        # Crear meta-learner Ridge con parámetros fijos (como TRB)
        self.meta_learner = Ridge(**self.meta_learner_config['fixed_params'])
        
        # Entrenar con predicciones out-of-fold
        self.meta_learner.fit(self.oof_predictions, y)
        
        # Calcular métricas del meta-learner
        meta_predictions = self.meta_learner.predict(self.oof_predictions)
        mae = mean_absolute_error(y, meta_predictions)
        r2 = r2_score(y, meta_predictions)
        
        self.meta_learner_performance = {
            'mae': mae,
            'r2': r2
        }
        
        logger.info(f"Meta-learner entrenado - MAE: {mae:.4f}, R²: {r2:.4f}")
    
    def _calculate_cv_metrics_from_oof(self) -> List[Dict]:
        """
        Calcula métricas de validación cruzada desde las predicciones OOF ya generadas.
        Evita entrenamiento duplicado.
        """
        if self.oof_predictions is None or not hasattr(self, 'oof_y_true'):
            logger.warning("No hay predicciones OOF o y_true disponibles")
            return []
        
        cv_scores = []
        
        # Calcular métricas para cada fold usando las predicciones OOF y y_true reales
        for fold_idx in range(len(self.oof_y_true)):
            y_fold_true = self.oof_y_true[fold_idx]
            
            # Obtener predicciones OOF para este fold desde la matriz completa
            # Necesitamos encontrar los índices de validación para este fold
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            X_dummy = np.zeros((len(self.oof_predictions), 1))  # Dummy para obtener índices
            
            fold_val_indices = None
            for i, (_, val_idx) in enumerate(tscv.split(X_dummy)):
                if i == fold_idx:
                    fold_val_indices = val_idx
                    break
            
            if fold_val_indices is not None and len(y_fold_true) > 0:
                # Obtener predicciones OOF para este fold
                fold_oof_preds = self.oof_predictions[fold_val_indices]
                
                # Calcular métricas del meta-learner para este fold
                if len(fold_oof_preds) > 0:
                    meta_preds = self.meta_learner.predict(fold_oof_preds)
                    
                    mae = mean_absolute_error(y_fold_true, meta_preds)
                    r2 = r2_score(y_fold_true, meta_preds)
                    
                    cv_scores.append({
                        'fold': fold_idx + 1,
                        'mae': mae,
                        'r2': r2
                    })
        
        logger.info(f"Calculadas métricas CV para {len(cv_scores)} folds")
        return cv_scores
    
    def _evaluate_final_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evalúa el modelo final en datos de prueba con métricas avanzadas.
        """
        # Predicciones de modelos base
        base_predictions = np.zeros((len(X_test), len(self.trained_base_models)))
        for model_idx, (model_name, model) in enumerate(self.trained_base_models.items()):
            base_predictions[:, model_idx] = model.predict(X_test)
        
        # Predicción final del meta-learner
        final_predictions = self.meta_learner.predict(base_predictions)
        
        # Métricas básicas
        mae = mean_absolute_error(y_test, final_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
        r2 = r2_score(y_test, final_predictions)
        
        # Métricas avanzadas por rangos
        range_metrics = self._calculate_range_metrics(y_test.values, final_predictions)
        
        # Métricas específicas para alto scoring (20+ puntos)
        high_scoring_mask = y_test >= 20
        if high_scoring_mask.sum() > 0:
            high_scoring_metrics = self._calculate_high_scoring_aggregate_metrics(
                y_test[high_scoring_mask].values, 
                final_predictions[high_scoring_mask]
            )
        else:
            high_scoring_metrics = {}
        
        # Compilar métricas finales
        final_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'range_metrics': range_metrics,
            'high_scoring_metrics': high_scoring_metrics
        }
        
        # Log de métricas por rangos
        logger.info("Métricas por rangos de puntuación:")
        for range_name, metrics in range_metrics.items():
            logger.info(f"  {range_name}: MAE={metrics['mae']:.3f}, R²={metrics['r2']:.3f}")
        
        # Log de métricas de alto scoring
        if high_scoring_metrics:
            logger.info("Métricas de alto scoring (20+ pts):")
            logger.info(f"  MAE: {high_scoring_metrics['mae']:.3f}")
            logger.info(f"  R²: {high_scoring_metrics['r2']:.3f}")
            logger.info(f"  Recall 25+: {high_scoring_metrics.get('recall_25plus', 0):.3f}")
            logger.info(f"  Recall 30+: {high_scoring_metrics.get('recall_30plus', 0):.3f}")
        
        return final_metrics
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        División temporal de datos para evitar data leakage.
        CRÍTICO: Asegura división cronológica correcta para NBA.
        
        Args:
            df: DataFrame con datos
            test_size: Proporción de datos para prueba
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: DataFrames de train y test
        """
        if 'Date' not in df.columns:
            raise ValueError("Columna 'Date' requerida para división temporal")
        
        # Convertir fechas si es necesario
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # DIAGNÓSTICO CRÍTICO: Verificar distribución de fechas
        unique_dates = df['Date'].unique()
        logger.info(f"Fechas únicas en dataset: {len(unique_dates)}")
        logger.info(f"Rango de fechas: {df['Date'].min()} -> {df['Date'].max()}")
        
        if len(unique_dates) == 1:
            logger.warning("PROBLEMA: Todas las fechas son iguales - división temporal imposible")
            logger.warning(f"Fecha única: {unique_dates[0]}")
            # Fallback: división aleatoria pero con advertencia
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            logger.warning("USANDO DIVISIÓN ALEATORIA (NO TEMPORAL) - REVISAR DATOS")
            return train_df, test_df
        
        if len(unique_dates) < 10:
            logger.warning(f"Pocas fechas únicas ({len(unique_dates)}) - verificar calidad de datos")
        
        # Ordenar por fecha y Player para asegurar orden correcto
        df_sorted = df.sort_values(['Date', 'player']).reset_index(drop=True)
        
        # Encontrar punto de corte basado en fechas únicas
        sorted_dates = sorted(unique_dates)
        split_date_idx = int(len(sorted_dates) * (1 - test_size))
        split_date = sorted_dates[split_date_idx]
        
        logger.info(f"Fecha de corte seleccionada: {split_date}")
        
        # División temporal basada en fecha de corte
        train_df = df_sorted[df_sorted['Date'] < split_date].copy().reset_index(drop=True)
        test_df = df_sorted[df_sorted['Date'] >= split_date].copy().reset_index(drop=True)
        
        # Verificar que la división fue exitosa
        if len(train_df) == 0:
            logger.error("DIVISIÓN FALLÓ: No hay datos de entrenamiento")
            raise ValueError("División temporal falló - revisar datos de fecha")
        
        if len(test_df) == 0:
            logger.error("DIVISIÓN FALLÓ: No hay datos de prueba")
            raise ValueError("División temporal falló - revisar datos de fecha")
        
        # Logs informativos mejorados
        logger.info(f"División temporal: {len(train_df)} entrenamiento, {len(test_df)} prueba")
        logger.info(f"Fecha corte: {train_df['Date'].max()} -> {test_df['Date'].min()}")
        logger.info(f"Proporción real: {len(test_df)/(len(train_df)+len(test_df)):.1%} test")
        
        # Verificación crítica: asegurar orden cronológico
        if train_df['Date'].max() >= test_df['Date'].min():
            logger.warning("  ADVERTENCIA: Posible solapamiento temporal detectado")
            # Mostrar fechas problemáticas
            overlap_train = train_df[train_df['Date'] >= test_df['Date'].min()]
            if len(overlap_train) > 0:
                logger.warning(f"  {len(overlap_train)} registros de entrenamiento en fechas de test")
        
        return train_df, test_df
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento completo del modelo de stacking ensemble.
        
        Args:
            df: DataFrame con datos NBA
            
        Returns:
            Dict[str, float]: Métricas de validación
        """
        
        # Verificar target
        if 'points' not in df.columns:
            raise ValueError("Columna 'PTS' (target) no encontrada en el dataset")
        
        # Verificar y ordenar datos cronológicamente
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['player', 'Date']).reset_index(drop=True)
            logger.info("Datos en orden cronológico confirmado")
        else:
            logger.warning("Columna 'Date' no encontrada - no se puede verificar orden cronológico")
        

        # Generar características en datos filtrados
        logger.info("Generando características especializadas...")
        df = self.feature_engineer.generate_all_features(df)  # Retorna DataFrame transformado
        
        # Obtener lista de features generadas
        feature_names = [col for col in df.columns if col not in ['player', 'Date', 'Team', 'Opp', 'points']]
        self.selected_features = feature_names
        self.feature_names = feature_names
        
        # División temporal en datos originales
        train_df, test_df = self._temporal_split(df)
        
        # Preparar datos de entrenamiento
        X_train = train_df[feature_names]
        y_train = train_df['points']
        X_test = test_df[feature_names]
        y_test = test_df['points']
        
        # Verificar datos válidos
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Filtrar columnas problemáticas
        # Solo mantener columnas numéricas (int, float, bool)
        numeric_cols = []
        for col in X_train.columns:
            dtype = X_train[col].dtype
            if dtype in ['int64', 'int32', 'float64', 'float32', 'bool', 'int8', 'int16', 'float16']:
                numeric_cols.append(col)
            elif str(dtype).startswith('int') or str(dtype).startswith('float') or str(dtype).startswith('bool'):
                numeric_cols.append(col)
            else:
                logger.debug(f"Excluyendo columna de entrenamiento {col} con dtype {dtype}")
        
        # Usar solo columnas numéricas
        X_train = X_train[numeric_cols]
        X_test = X_test[numeric_cols]
        
        # Actualizar selected_features y feature_names para consistencia
        self.selected_features = numeric_cols
        self.feature_names = numeric_cols
        
        if len(numeric_cols) == 0:
            raise ValueError("No hay columnas numéricas disponibles para entrenamiento")
        
        # Entrenar el scaler con los datos de entrenamiento
        logger.info("Entrenando StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir de vuelta a DataFrame para mantener compatibilidad
        X_train = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)
        
        logger.info(f"Datos de entrenamiento: {X_train.shape}")
        logger.info(f"Datos de prueba: {X_test.shape}")
        logger.info("Scaler entrenado y datos escalados")
        
        # División para optimización de hiperparámetros
        split_idx = int(len(X_train) * 0.8)
        X_opt_train = X_train.iloc[:split_idx]
        y_opt_train = y_train.iloc[:split_idx]
        X_opt_val = X_train.iloc[split_idx:]
        y_opt_val = y_train.iloc[split_idx:]
        
        # Entrenar modelos base con out-of-fold
        logger.info("Entrenando modelos base con out-of-fold predictions...")
        self._train_base_models_with_oof(X_train, y_train)
        
        # Entrenar meta-learner con out-of-fold
        logger.info("Entrenando meta-learner con out-of-fold predictions...")
        self._train_meta_learner_with_oof(X_train, y_train)
        
        # Calcular métricas de validación cruzada desde OOF training
        logger.info("Calculando métricas de validación cruzada...")
        self.cv_scores = self._calculate_cv_metrics_from_oof()
        
        # Evaluación final
        logger.info("Evaluación final...")
        final_metrics = self._evaluate_final_model(X_test, y_test)
        
        # Marcar como entrenado
        self.is_trained = True
        
        logger.info("=================================================")
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=================================================")
        logger.info(f"MAE: {final_metrics['mae']:.4f}")
        logger.info(f"RMSE: {final_metrics['rmse']:.4f}")
        logger.info(f"R²: {final_metrics['r2']:.4f}")
        logger.info("=================================================")
        
        return final_metrics

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones usando el modelo entrenado con stacking OOF
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Array con predicciones de puntos
        """
        if not hasattr(self, 'trained_base_models') or not hasattr(self, 'meta_learner'):
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Generar features (modifica df in-place, retorna List[str])
        features = self.feature_engineer.generate_all_features(df)
        
        # Determinar expected_features dinámicamente del modelo entrenado
        try:
            if hasattr(self, 'expected_features'):
                expected_features = self.expected_features
            elif hasattr(self, 'selected_features') and self.selected_features:
                expected_features = self.selected_features
            else:
                # Fallback: usar todas las features generadas
                expected_features = features
        except Exception as e:
            logger.warning(f"No se pudieron obtener expected_features: {e}")
            expected_features = features if features else []
        
        # Reordenar DataFrame según expected_features (df ya tiene las features)
        available_features = [f for f in expected_features if f in df.columns]
        if len(available_features) != len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            logger.warning(f"Features faltantes: {missing_features}")
            # Agregar features faltantes con valor 0
            for feature in missing_features:
                df[feature] = 0
                available_features.append(feature)
        
        # Usar expected_features en el orden correcto
        X = df[expected_features].fillna(0)
        
        # Aplicar el scaler antes de predecir (igual que en entrenamiento)
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=expected_features, index=X.index)
        
        # Realizar predicciones usando stacking con out-of-fold predictions
        predictions = self._predict_with_stacking(X)
    
        # Aplicar procesamiento robusto para puntos - DESACTIVADO TEMPORALMENTE
        predictions = self._apply_robust_prediction_processing(predictions)
        
        return predictions
    
    def _predict_with_stacking(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicción usando stacking con out-of-fold predictions
        """
        # Obtener predicciones de todos los modelos base
        base_predictions = np.zeros((len(X), len(self.trained_base_models)))
        
        for i, (name, model) in enumerate(self.trained_base_models.items()):
            base_predictions[:, i] = model.predict(X)
        
        # Usar meta-learner para combinar predicciones
        meta_predictions = self.meta_learner.predict(base_predictions)
        
        return meta_predictions
    
    def _apply_robust_prediction_processing(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplicar procesamiento robusto para obtener predicciones enteras de PUNTOS
        
        Args:
            predictions: Predicciones en float
            
        Returns:
            np.ndarray: Predicciones procesadas como enteros
        """
        try:
            predictions = np.array(predictions, dtype=float)
            predictions_int = np.zeros_like(predictions, dtype=int)
            
            # Manejo de outliers usando IQR - CORREGIDO para no cortar predicciones bajas
            Q1 = np.percentile(predictions, 25)
            Q3 = np.percentile(predictions, 75)
            IQR = Q3 - Q1
            
            # Factor para puntos (más permisivo para casos altos) (Elite range)
            outlier_factor = 3.0
            
            # CORRECCIÓN: No aplicar límite inferior agresivo para jugadores de rol
            # Solo limitar el límite superior para casos extremos
            lower_bound = 0  # Permitir predicciones bajas (jugadores de rol)
            upper_bound = min(60, Q3 + outlier_factor * IQR)  # Máximo realista: 60 puntos
            
            # CORRECCIONES SUTILES POR RANGO basadas en el análisis del modelo base
            # Sesgos identificados: +2.08 (bajos), -1.94 (altos) - Corrección sutil
            
            # Rangos bajos: Reducir sobrestimación (factores sutiles)
            mask_suplentes = predictions < 5
            predictions[mask_suplentes] *= 0.9  # Reducción sutil
            
            mask_rotacion = (predictions >= 5) & (predictions < 10)
            predictions[mask_rotacion] *= 0.95  # Reducción mínima
            
            # Rangos medios: Sin corrección (ya están equilibrados)
            mask_importantes = (predictions >= 10) & (predictions < 15)
            # Sin corrección para rangos medios
            
            # Rangos altos: Aumentar para reducir subestimación (factores sutiles)
            mask_estrellas = (predictions >= 20) & (predictions < 25)
            predictions[mask_estrellas] *= 1.02  # Aumento mínimo
            
            mask_superestrellas = (predictions >= 25) & (predictions < 30)
            predictions[mask_superestrellas] *= 1.03  # Aumento sutil
            
            mask_elite = (predictions >= 30) & (predictions < 40)
            predictions[mask_elite] *= 1.05  # Aumento moderado
            
            mask_historico = predictions >= 40
            predictions[mask_historico] *= 1.08  # Aumento moderado
            
            # Solo aplicar winsorización al límite superior, no al inferior
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            # Asegurar que sean no negativas
            predictions = np.maximum(predictions, 0)
            
            # Redondeo probabilístico para mantener la distribución
            for i, pred in enumerate(predictions):
                floor_val = int(np.floor(pred))
                prob = pred - floor_val
                predictions_int[i] = floor_val + np.random.binomial(1, prob)
            
            # Aplicar límites finales para puntos
            predictions_int = np.clip(predictions_int, 0, 60)
            
            return predictions_int.astype(int)
            
        except Exception as e:
            logger.warning(f"Error en procesamiento robusto de puntos: {e}")
            # Fallback: solo redondear y limitar
            predictions = np.maximum(predictions, 0)
            return np.round(np.clip(predictions, 0, 60)).astype(int)

    def save_model(self, filepath: str):
        """
        Guardar modelo entrenado COMPLETO con todos los componentes OOF.
        
        Args:
            filepath: Ruta donde guardar el modelo. Si no se especifica, usa la ruta por defecto.
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado.")
        
        if not hasattr(self, 'trained_base_models') or not self.trained_base_models:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar el objeto COMPLETO con todos los componentes OOF
        joblib.dump(self, filepath, compress=3, protocol=4)
        logger.info(f"Modelo PTS OOF completo guardado: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Cargar modelo entrenado COMPLETO (método de instancia).
        
        Args:
            filepath: Ruta del modelo a cargar
        """
        try:
            # Cargar objeto completo directamente
            loaded_model = joblib.load(filepath)
            if isinstance(loaded_model, StackingPTSModel):
                # Copiar todos los atributos del modelo cargado
                self.__dict__.update(loaded_model.__dict__)
                logger.info(f"Modelo PTS OOF completo cargado: {filepath}")
            else:
                raise ValueError("Objeto cargado no es un StackingPTSModel válido")
            
        except Exception as e:
            # Fallback: intentar cargar formato legacy
            logger.warning(f"Error cargando modelo OOF, intentando formato legacy: {e}")
            try:
                model_data = joblib.load(filepath)
                if isinstance(model_data, dict):
                    # Crear nueva instancia y cargar datos legacy
                    new_model = self.__class__()
                    
                    # Cargar componentes OOF principales
                    new_model.trained_base_models = model_data.get('trained_base_models', {})
                    new_model.meta_learner = model_data.get('meta_learner', None)
                    new_model.best_params_per_model = model_data.get('best_params_per_model', {})
                    new_model.base_models_performance = model_data.get('base_models_performance', {})
                    new_model.meta_learner_performance = model_data.get('meta_learner_performance', {})
                    new_model.oof_predictions = model_data.get('oof_predictions', None)
                    
                    # Cargar componentes de features y escalado
                    new_model.feature_engineer = model_data.get('feature_engineer', PointsFeatureEngineer())
                    new_model.scaler = model_data.get('scaler', StandardScaler())
                    new_model.selected_features = model_data.get('selected_features', [])
                    new_model.feature_names = model_data.get('feature_names', [])
                    new_model.expected_features = model_data.get('expected_features', [])
                    
                    # Cargar métricas y estado
                    new_model.is_trained = model_data.get('is_trained', False)
                    new_model.training_metrics = model_data.get('training_metrics', {})
                    new_model.validation_metrics = model_data.get('validation_metrics', {})
                    new_model.cv_scores = model_data.get('cv_scores', [])
                    new_model.feature_importance = model_data.get('feature_importance', {})
                    new_model.cutoff_date = model_data.get('cutoff_date', None)
                    
                    # Cargar configuración de modelos base
                    new_model.base_models = model_data.get('base_models', {})
                    new_model.meta_learner_config = model_data.get('meta_learner_config', {})
                    
                    # Copiar atributos del modelo legacy
                    self.__dict__.update(new_model.__dict__)
                    logger.info(f"Modelo PTS legacy convertido a OOF: {filepath}")
                else:
                    raise ValueError("Formato de archivo legacy no reconocido")
            except Exception as e2:
                raise ValueError(f"No se pudo cargar el modelo. Error formato OOF: {e}, Error formato legacy: {e2}")
    
    def _calculate_adaptive_weights(self, y_true: np.ndarray) -> np.ndarray:
        """
        Calcula pesos adaptativos SUTILES basados en el análisis del modelo base.
        Sesgos identificados: +2.08 (bajos), -1.94 (altos) - Corrección sutil.
        
        Args:
            y_true: Valores reales de puntos
            
        Returns:
            np.ndarray: Pesos adaptativos sutiles para cada muestra
        """
        weights = np.ones_like(y_true, dtype=float)
        
        # PESOS SUTILES basados en el análisis del modelo base
        # Sesgo identificado: sobrestimación en bajos (+2.08), subestimación en altos (-1.94)
        
        # Rangos bajos: Más peso para reducir sobrestimación (sutil)
        weights = np.where((y_true >= 0) & (y_true < 5), 1.3, weights)   # Suplentes: peso sutil
        weights = np.where((y_true >= 5) & (y_true < 10), 1.2, weights)   # Rotación: peso sutil
        
        # Rangos medios: Peso normal (ya están equilibrados)
        weights = np.where((y_true >= 10) & (y_true < 15), 1.1, weights)   # Importantes: peso mínimo
        weights = np.where((y_true >= 15) & (y_true < 20), 1.0, weights)  # Clave: peso normal
        
        # Rangos altos: Más peso para reducir subestimación (sutil)
        weights = np.where((y_true >= 20) & (y_true < 25), 1.2, weights)    # Estrellas: peso sutil
        weights = np.where((y_true >= 25) & (y_true < 30), 1.4, weights)    # Superestrellas: peso moderado
        weights = np.where((y_true >= 30) & (y_true < 40), 1.6, weights)    # Élite: peso moderado
        weights = np.where(y_true >= 40, 1.8, weights)                      # Histórico: peso moderado
        
        return weights

    def _calculate_range_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calcula métricas de evaluación por rangos de puntuación MEJORADAS.
        Rangos más específicos y métricas especializadas para alto scoring. (Elite range)
        
        Args:
            y_true: Valores reales de puntos
            y_pred: Predicciones de puntos
            
        Returns:
            Dict[str, Dict[str, float]]: Métricas de evaluación por rangos de puntuación
        """
        range_metrics = {}
        
        # RANGOS MEJORADOS Y MÁS ESPECÍFICOS
        ranges = [
            (0, 5, "Suplentes"),
            (5, 10, "Rotación"),
            (10, 15, "Importantes"),
            (15, 20, "Clave"),
            (20, 25, "Estrellas"),
            (25, 30, "Superstrellas"),
            (30, 40, "Elite"),
            (40, float('inf'), "Histórico")
        ]
        
        for start, end, label in ranges:
            range_mask = (y_true >= start) & (y_true < end)
            range_samples = np.sum(range_mask)
            
            if range_samples > 0:
                range_y_true = y_true[range_mask]
                range_y_pred = y_pred[range_mask]
                
                # Métricas básicas
                range_mae = mean_absolute_error(range_y_true, range_y_pred)
                range_rmse = np.sqrt(mean_squared_error(range_y_true, range_y_pred))
                range_r2 = r2_score(range_y_true, range_y_pred)
                
                # NUEVAS MÉTRICAS ESPECÍFICAS PARA ALTO SCORING
                
                # 1. Sesgo de predicción (bias)
                bias = np.mean(range_y_pred - range_y_true)
                
                # 2. Porcentaje de subestimación
                underestimation_pct = np.mean(range_y_pred < range_y_true) * 100
                
                # 3. Error promedio en subestimación
                underestimated_mask = range_y_pred < range_y_true
                if underestimated_mask.sum() > 0:
                    avg_underestimation = np.mean(range_y_true[underestimated_mask] - range_y_pred[underestimated_mask])
                else:
                    avg_underestimation = 0
                
                # 4. Accuracy por tolerancia específica para el rango
                if start >= 20:  # Para alto scoring, tolerancia mayor
                    tolerance = 4
                elif start >= 10:  # Para scoring medio, tolerancia media
                    tolerance = 3
                else:  # Para bajo scoring, tolerancia menor
                    tolerance = 2
                
                accuracy_range = np.mean(np.abs(range_y_pred - range_y_true) <= tolerance) * 100
                
                # 5. Coeficiente de variación del error
                error_cv = np.std(np.abs(range_y_pred - range_y_true)) / (range_mae + 1e-6)
                
                # 6. Métricas específicas para rangos altos
                if start >= 20:
                    # Capacidad de predecir juegos explosivos (30+ puntos)
                    explosive_games_true = np.sum(range_y_true >= 30)
                    explosive_games_pred = np.sum(range_y_pred >= 30)
                    
                    # CORREGIDO: Cálculo correcto de explosive_recall
                    if explosive_games_true > 0:
                        # Recall: cuántos juegos explosivos reales predecimos correctamente
                        correctly_pred_explosive = np.sum((range_y_true >= 30) & (range_y_pred >= 27))  # Tolerancia de 3 puntos
                        explosive_recall = correctly_pred_explosive / explosive_games_true
                    else:
                        explosive_recall = 0.0
                    
                    # Error en predicciones de alto impacto
                    high_impact_mask = range_y_true >= 25
                    if high_impact_mask.sum() > 0:
                        high_impact_mae = mean_absolute_error(
                            range_y_true[high_impact_mask], 
                            range_y_pred[high_impact_mask]
                        )
                    else:
                        high_impact_mae = 0
                else:
                    explosive_recall = 0.0
                    high_impact_mae = 0.0
                
                range_metrics[f"{start}-{end}_{label}"] = {
                    'mae': range_mae,
                    'rmse': range_rmse,
                    'r2': range_r2,
                    'bias': bias,
                    'underestimation_pct': underestimation_pct,
                    'avg_underestimation': avg_underestimation,
                    'accuracy_tolerance': accuracy_range,
                    'error_cv': error_cv,
                    'explosive_recall': explosive_recall,
                    'high_impact_mae': high_impact_mae,
                    'samples': range_samples
                }
        
        # MÉTRICAS AGREGADAS PARA ALTO SCORING
        high_scoring_mask = y_true >= 20
        if high_scoring_mask.sum() > 0:
            high_scoring_metrics = self._calculate_high_scoring_aggregate_metrics(
                y_true[high_scoring_mask], 
                y_pred[high_scoring_mask]
            )
            range_metrics['HIGH_SCORING_AGGREGATE'] = high_scoring_metrics
        
        return range_metrics
    
    def _calculate_high_scoring_aggregate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calcula métricas agregadas específicas para jugadores de alto scoring (20+ puntos). (Elite range)
        
        Args:
            y_true: Valores reales de puntos (solo 20+)
            y_pred: Predicciones de puntos (solo 20+)
            
        Returns:
            Dict[str, float]: Métricas agregadas para alto scoring
        """
        metrics = {}
        
        # Métricas básicas
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Sesgo sistemático
        metrics['systematic_bias'] = np.mean(y_pred - y_true)
        
        # Distribución de errores
        errors = y_pred - y_true
        metrics['error_std'] = np.std(errors)
        metrics['error_skewness'] = self._calculate_skewness(errors)
        
        # Capacidad de predecir diferentes niveles
        for threshold in [25, 30, 35, 40]:
            true_above = np.sum(y_true >= threshold)
            pred_above = np.sum(y_pred >= threshold)
            
            if true_above > 0:
                # Recall: qué porcentaje de juegos de threshold+ puntos predecimos
                correctly_pred_above = np.sum((y_true >= threshold) & (y_pred >= threshold - 3))
                recall = correctly_pred_above / true_above
                
                # Precision: de los que predecimos threshold+, qué porcentaje son correctos
                if pred_above > 0:
                    precision = correctly_pred_above / pred_above
                else:
                    precision = 0
                
                metrics[f'recall_{threshold}plus'] = recall
                metrics[f'precision_{threshold}plus'] = precision
            else:
                metrics[f'recall_{threshold}plus'] = 0
                metrics[f'precision_{threshold}plus'] = 0
        
        # Métricas de calibración
        metrics['mean_prediction'] = np.mean(y_pred)
        metrics['mean_actual'] = np.mean(y_true)
        metrics['prediction_calibration'] = metrics['mean_prediction'] / metrics['mean_actual']
        
        # Consistencia de predicción
        metrics['prediction_std'] = np.std(y_pred)
        metrics['actual_std'] = np.std(y_true)
        metrics['variance_ratio'] = metrics['prediction_std'] / metrics['actual_std']
        
        metrics['samples'] = len(y_true)
        
        return metrics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calcula la asimetría (skewness) de los datos.
        
        Args:
            data: Array de datos
            
        Returns:
            float: Valor de skewness
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
        except:
            return 0

class XGBoostPTSModel(StackingPTSModel):
    """
    Clase de compatibilidad con el modelo original (Wrapper)
    """
    
    def __init__(self, teams_df: pd.DataFrame = None, players_df: pd.DataFrame = None, 
                 players_quarters_df: pd.DataFrame = None, **kwargs):
        """
        Inicializa el modelo manteniendo compatibilidad con la interfaz original.
        """
        # Mapear parámetros antiguos a nuevos
        if 'n_trials' not in kwargs:
            kwargs['n_trials'] = 25  # Optimización bayesiana por modelo
        
        # Pasar datasets al constructor padre
        super().__init__(teams_df=teams_df, players_df=players_df, 
                        players_quarters_df=players_quarters_df, **kwargs)
        
        # Mantener atributos para compatibilidad
        self.model = None  # Se asignará a trained_base_models después del entrenamiento
        self.best_params = None  # Se asignará a los parámetros del mejor modelo base
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento manteniendo compatibilidad con la interfaz original.
        """
        # Llamar al método de entrenamiento del stacking OOF
        result = super().train(df)
        
        # Asignar para compatibilidad con la nueva arquitectura OOF
        self.model = self.trained_base_models  # Usar modelos base entrenados
        if self.best_params_per_model:
            # Usar parámetros del modelo con mejor rendimiento OOF
            best_model_name = min(
                self.best_params_per_model.keys(),
                key=lambda x: self.base_models_performance.get(x, {}).get('mae', float('inf'))
            )
            self.best_params = self.best_params_per_model[best_model_name]
        
        # Crear stacking_model para compatibilidad con predictor
        if self.trained_base_models and self.meta_learner:
            # Crear lista de estimadores para StackingRegressor
            base_estimators = [(name, model) for name, model in self.trained_base_models.items()]
            
            # Crear StackingRegressor
            self.stacking_model = StackingRegressor(
                estimators=base_estimators,
                final_estimator=self.meta_learner,
                cv=3,
                n_jobs=1,
                passthrough=True
            )
        
        # Calcular y guardar cutoff_date para compatibilidad con trainer
        if 'Date' in df.columns:
            df_sorted = df.sort_values('Date').reset_index(drop=True)
            split_idx = int(len(df_sorted) * 0.8)  # Mismo split que en _temporal_split
            self.cutoff_date = df_sorted.iloc[split_idx]['Date']
            logger.info(f"Cutoff date establecido: {self.cutoff_date}")
        else:
            # Si no hay columna Date, usar fecha actual como fallback
            from datetime import datetime
            self.cutoff_date = datetime.now()
            logger.warning("No se encontró columna 'Date', usando fecha actual como cutoff_date")
        
        return result
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Obtener importancia de características para compatibilidad.
        
        Args:
            top_n: Número de características más importantes a retornar
            
        Returns:
            Dict[str, float]: Diccionario con importancia de características
        """
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Intentar obtener feature importance del stacking ensemble
        if hasattr(self, 'feature_importance') and self.feature_importance:
            # Si tenemos importancia del meta-learner (importancia de modelos base)
            stacking_importance = self.feature_importance.copy()
        else:
            stacking_importance = {}
        
        # Intentar obtener feature importance de modelos base individuales
        feature_importance_combined = {}
        
        # Combinar importancia de todos los modelos base
        for model_name, model in self.trained_base_models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    # Para modelos tree-based (XGBoost, LightGBM, RandomForest, etc.)
                    importances = model.feature_importances_
                    feature_names = self.selected_features
                    
                    for feature, importance in zip(feature_names, importances):
                        if feature not in feature_importance_combined:
                            feature_importance_combined[feature] = 0
                        feature_importance_combined[feature] += importance
                        
                elif hasattr(model, 'coef_'):
                    # Para modelos lineales (Ridge, Lasso, etc.)
                    importances = np.abs(model.coef_)
                    feature_names = self.selected_features
                    
                    for feature, importance in zip(feature_names, importances):
                        if feature not in feature_importance_combined:
                            feature_importance_combined[feature] = 0
                        feature_importance_combined[feature] += importance
                        
            except Exception as e:
                logger.warning(f"No se pudo obtener feature importance de {model_name}: {e}")
                continue
        
        # Normalizar importancias combinadas
        if feature_importance_combined:
            total_importance = sum(feature_importance_combined.values())
            if total_importance > 0:
                feature_importance_combined = {
                    feature: importance / total_importance 
                    for feature, importance in feature_importance_combined.items()
                }
        
        # Si no hay importancia de features, usar importancia uniforme
        if not feature_importance_combined:
            feature_importance_combined = {
                feature: 1.0 / len(self.selected_features) 
                for feature in self.selected_features
            }
        
        # Ordenar por importancia y tomar top_n
        sorted_features = sorted(
            feature_importance_combined.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_features = dict(sorted_features[:top_n])
        
        logger.info(f"Feature importance calculada para top {len(top_features)} características")
        
        return top_features

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Realizar predicciones con procesamiento robusto"""
        return super().predict(df)
    
    def save_model(self, filepath: str):
        """Guardar modelo completo"""
        super().save_model(filepath)
    
    def load_model(self, filepath: str):
        """Cargar modelo completo"""
        # Usar el método de instancia del padre
        super().load_model(filepath)
        # Asignar para compatibilidad con la nueva arquitectura OOF
        self.model = self.trained_base_models
        logger.info(f"Modelo PTS wrapper cargado correctamente desde: {filepath}")