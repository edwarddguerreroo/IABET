"""
Modelo de Predicción de Rebotes Totales (TRB)
============================================

Sistema avanzado de stacking ensemble para predicción de rebotes totales
que capturará un jugador NBA en su próximo partido.

CARACTERÍSTICAS PRINCIPALES:
- Stacking ensemble con XGBoost, LightGBM, CatBoost, Ridge
- Optimización bayesiana de hiperparámetros
- Validación cruzada temporal (respeta orden cronológico)
- Regularización avanzada (L1/L2, Dropout, Early Stopping)
- División temporal para evitar data leakage
- Meta-learner adaptativo para stacking
- Features especializadas para rebotes

ARQUITECTURA:
1. Modelos Base: XGBoost, LightGBM, CatBoost, Ridge
2. Meta-learner: LightGBM + CatBoost (multi-nivel)
3. Validación: TimeSeriesSplit cronológico
4. Optimización: Optuna (Bayesian Optimization)
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings
from pathlib import Path
import joblib
import json
import os
from scipy import stats

# CONFIGURACIÓN CRÍTICA PARA WINDOWS - EVITAR ERRORES DE PARALELIZACIÓN
os.environ['LOKY_MAX_CPU_COUNT'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['JOBLIB_TEMP_FOLDER'] = './tmp'

# Configurar joblib para uso single-thread
from joblib import parallel_backend
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, ElasticNet
import optuna
from optuna.samplers import TPESampler

# Configurar Optuna para ser silencioso
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from .features_trb import ReboundsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StackingTRBModel:
    """
    Modelo de Stacking Ensemble para Predicción de Rebotes Totales
    ULTRA-OPTIMIZADO con regularización balanceada y validación temporal
    """
    
    def __init__(self, enable_neural_network: bool = False, enable_gpu: bool = False, 
                 random_state: int = 42, teams_df: pd.DataFrame = None, 
                 players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        """
        Inicializar el modelo de stacking para TRB
        
        Args:
            enable_gpu: Usar GPU para XGBoost/LightGBM si está disponible
            random_state: Semilla para reproducibilidad
            teams_df: DataFrame con datos de equipos para features avanzadas
            players_df: DataFrame con datos de jugadores (total)
            players_quarters_df: DataFrame con datos de jugadores por cuartos
        """
        self.random_state = random_state
        self.enable_neural_network = enable_neural_network
        self.enable_gpu = enable_gpu
        self.teams_df = teams_df
        self.players_df = players_df
        self.players_quarters_df = players_quarters_df
        
        # Configuración de modelos con datasets avanzados
        self.base_models = {}
        self.stacking_model = None
        self.feature_engineer = ReboundsFeatureEngineer(
            teams_df=teams_df, 
            players_df=players_df,
            players_quarters_df=players_quarters_df
        )
        self.scaler = StandardScaler()
        
        # Métricas y resultados
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params_per_model = {}
        
        # Configuración de optimización
        self.n_trials = 25  # MÁS trials para mejor exploración de hiperparámetros
        self.cv_folds = 3   # Folds para validación cruzada temporal
        
        # Mostrar ensemble final
        model_names = list(self.base_models.keys())
        logger.info(f"Configuración: NN={enable_neural_network}, GPU={enable_gpu}")

    def _setup_base_models(self):
        """Configurar modelos base con hiperparámetros optimizados para rebotes"""
        
        # XGBoost - Excelente para features categóricas y no lineales
        self.base_models['xgboost'] = {
            'model': xgb.XGBRegressor(
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist',
                gpu_id=0 if self.enable_gpu else None,
                n_jobs=1
            ),
            'param_space': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (1, 10),
                'min_child_weight': (1, 10)
            }
        }
        
        # LightGBM - Rápido y eficiente para datasets grandes
        self.base_models['lightgbm'] = {
            'model': lgb.LGBMRegressor(
                random_state=self.random_state,
                device_type='cpu',  # Forzar CPU para evitar problemas
                n_jobs=1,
                verbose=-1,
                num_threads=1,
                force_col_wise=True,
                deterministic=True
            ),
            'param_space': {
                'n_estimators': (100, 500),
                'max_depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.7, 1.0),
                'colsample_bytree': (0.7, 1.0),
                'reg_alpha': (0, 10),
                'reg_lambda': (1, 10),
                'min_child_samples': (5, 50),
                'num_leaves': (10, 100)
            }
        }
        
        # CatBoost - Excelente para features categóricas sin preprocessing
        self.base_models['catboost'] = {
            'model': cb.CatBoostRegressor(
                random_state=self.random_state,
                task_type='GPU' if self.enable_gpu else 'CPU',
                devices='0' if self.enable_gpu else None,
                verbose=False,
                allow_writing_files=False
            ),
            'param_space': {
                'iterations': (100, 500),
                'depth': (3, 8),
                'learning_rate': (0.01, 0.3),
                'l2_leaf_reg': (1, 10),
                'subsample': (0.7, 1.0),
                'colsample_bylevel': (0.7, 1.0),
                'min_data_in_leaf': (1, 50)
            }
        }
        
        # Ridge Regression - Modelo lineal para diversidad
        self.base_models['ridge'] = {
            'model': Ridge(
                alpha=15.0,
                random_state=self.random_state
            ),
            'param_space': {}  # Hiperparámetros fijos, no requiere optimización
        }
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        División temporal de datos respetando orden cronológico
        CRÍTICO: Evita data leakage usando fechas
        """
        if 'Date' not in df.columns:
            logger.warning("Columna 'Date' no encontrada, usando división secuencial")
            split_idx = int(len(df) * (1 - test_size))
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
        
        # Ordenar por fecha
        df_sorted = df.sort_values('Date').reset_index(drop=True)
        
        # Encontrar punto de corte temporal
        split_idx = int(len(df_sorted) * (1 - test_size))
        cutoff_date = df_sorted.iloc[split_idx]['Date']
        
        train_data = df_sorted[df_sorted['Date'] < cutoff_date].copy()
        test_data = df_sorted[df_sorted['Date'] >= cutoff_date].copy()
        
        logger.info(f"División temporal: {len(train_data)} entrenamiento, {len(test_data)} prueba")
        logger.info(f"Fecha corte: {cutoff_date}")
        
        return train_data, test_data
    
    def _train_base_models_with_oof(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """
        Entrena modelos base con out-of-fold predictions para evitar data leakage
        
        Returns:
            np.ndarray: Predicciones out-of-fold de todos los modelos base
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Configurar TimeSeriesSplit para datos temporales
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Inicializar array para predicciones OOF
        n_samples = len(X_train)
        n_models = len(self.base_models)
        oof_predictions = np.zeros((n_samples, n_models))
        
        # Entrenar cada modelo base con OOF predictions
        for i, (name, model_config) in enumerate(self.base_models.items()):
            logger.info(f"Entrenando {name} con out-of-fold predictions...")
            
            # Aplicar optimización de hiperparámetros si está habilitada
            if name in ['xgboost', 'lightgbm', 'catboost']:
                logger.info(f"  Optimizando hiperparámetros para {name}...")
                best_params = self._optimize_hyperparameters(
                    model_config['model'], 
                    model_config['param_space'], 
                    X_train, 
                    y_train,
                    name
                )
                # Crear modelo con mejores parámetros
                model = model_config['model']
                model.set_params(**best_params)
            else:
                model = model_config['model']
            
            fold_predictions = np.zeros(n_samples)
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                
                # Entrenar modelo en fold de entrenamiento
                model.fit(X_fold_train, y_fold_train)
                
                # Predecir en fold de validación
                fold_pred = model.predict(X_fold_val)
                fold_predictions[val_idx] = fold_pred
                
                logger.debug(f"  Fold {fold + 1}: MAE = {mean_absolute_error(y_train.iloc[val_idx], fold_pred):.4f}")
            
            # Guardar predicciones OOF para este modelo
            oof_predictions[:, i] = fold_predictions
            
            # Entrenar modelo final con todos los datos para predicciones futuras
            model.fit(X_train, y_train)
            
            # Calcular métricas OOF para este modelo
            oof_mae = mean_absolute_error(y_train, fold_predictions)
            oof_r2 = r2_score(y_train, fold_predictions)
            logger.info(f"  {name} OOF - MAE: {oof_mae:.4f}, R²: {oof_r2:.4f}")
        
        return oof_predictions
    
    def _train_meta_learner_with_oof(self, oof_predictions: np.ndarray, y_train: pd.Series):
        """
        Entrena meta-learner usando predicciones out-of-fold de modelos base
        """
        # Crear DataFrame con predicciones OOF
        oof_df = pd.DataFrame(oof_predictions, columns=list(self.base_models.keys()))
        
        # Entrenar meta-learner con predicciones OOF
        self.meta_learner = Ridge(alpha=20.0, random_state=self.random_state)
        self.meta_learner.fit(oof_df, y_train)
        
        # Calcular métricas del meta-learner
        meta_pred = self.meta_learner.predict(oof_df)
        meta_mae = mean_absolute_error(y_train, meta_pred)
        meta_r2 = r2_score(y_train, meta_pred)
        
        logger.info(f"Meta-learner entrenado - MAE: {meta_mae:.4f}, R²: {meta_r2:.4f}")
        
        # Guardar modelos base entrenados para predicciones futuras
        self.trained_base_models = {}
        for name, model_config in self.base_models.items():
            self.trained_base_models[name] = model_config['model']
    
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
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento completo del modelo Stacking TRB
        
        Args:
            df: DataFrame con datos de jugadores y estadísticas
            
        Returns:
            Dict con métricas de validación
        """
        logger.info("Iniciando entrenamiento del modelo TRB...")
        
        # FORZAR CONFIGURACIÓN SINGLE-THREAD PARA EVITAR ERRORES
        with parallel_backend('threading', n_jobs=1):
            # Crear directorio temporal si no existe
            os.makedirs('./tmp', exist_ok=True)
        
        # Verificar orden cronológico
        if 'Date' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['player', 'Date']).reset_index(drop=True)
        
        # Generar features especializadas para rebotes
        logger.info("Generando características especializadas...")
        features = self.feature_engineer.generate_all_features(df)  # Modificar DataFrame directamente
        
        if not features:
            raise ValueError("No se pudieron generar features para TRB")
        
        logger.info(f"Features seleccionadas: {len(features)}")

        # Preparar datos (ahora df tiene las features)
        X = df[features].fillna(0)
        y = df['rebounds']
        
        # Usar todas las features disponibles directamente
        selected_features = features
        self.selected_features = selected_features  # Guardar para usar en predict()

        # División temporal
        train_data, test_data = self._temporal_split(df)
        
        X_train = train_data[selected_features].fillna(0)
        y_train = train_data['rebounds']
        X_test = test_data[selected_features].fillna(0)
        y_test = test_data['rebounds']
        
        # Entrenar el scaler con los datos de entrenamiento
        logger.info("Entrenando StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir de vuelta a DataFrame para mantener compatibilidad
        X_train = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
        
        logger.info("Scaler entrenado y datos escalados")
        
        # PASO 1: Configurar modelos base
        logger.info("Configurando modelos base...")
        self._setup_base_models()
        
        # PASO 2: Entrenar modelos base con out-of-fold predictions (STACKING CORRECTO)
        logger.info("Entrenando modelos base con out-of-fold predictions...")
        oof_predictions = self._train_base_models_with_oof(X_train, y_train)
        
        # PASO 3: Entrenar meta-learner con predicciones OOF
        logger.info("Entrenando meta-learner con predicciones out-of-fold...")
        self._train_meta_learner_with_oof(oof_predictions, y_train)
        
        # PASO 4: Validación cruzada temporal del stack completo
        logger.info("Validación cruzada del stack completo...")
        self._perform_temporal_cross_validation(X_train, y_train)
        
        # PASO 5: Evaluación final
        logger.info("Evaluación final...")
        y_pred = self._predict_with_stacking(X_test)
        
        # Calcular métricas
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Métricas específicas para rebotes
        accuracy_1reb = np.mean(np.abs(y_test - y_pred) <= 1) * 100
        accuracy_2reb = np.mean(np.abs(y_test - y_pred) <= 2) * 100
        accuracy_3reb = np.mean(np.abs(y_test - y_pred) <= 3) * 100
        
        self.validation_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_1reb': accuracy_1reb,
            'accuracy_2reb': accuracy_2reb,
            'accuracy_3reb': accuracy_3reb
        }
        
        # Calcular importancia de features seleccionadas
        self._calculate_feature_importance(selected_features)
        
        # Mostrar resultados FINALES
        logger.info("=" * 50)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.validation_metrics['mae']:.4f}")
        logger.info(f"RMSE: {self.validation_metrics['rmse']:.4f}")
        logger.info(f"R²: {self.validation_metrics['r2']:.4f}")
        logger.info(f"Accuracy ±1reb: {accuracy_1reb:.1f}%")
        logger.info(f"Accuracy ±2reb: {accuracy_2reb:.1f}%")
        logger.info(f"Accuracy ±3reb: {accuracy_3reb:.1f}%")
        logger.info("=" * 50)
        
        return self.validation_metrics

    def _optimize_hyperparameters(self, model, param_space: Dict, X_train: pd.DataFrame, 
                                 y_train: pd.Series, model_name: str) -> Dict:
        """Optimización bayesiana de hiperparámetros con CV robusto y Early stopping adaptativo"""
        
        import optuna
        from optuna.samplers import TPESampler
        
        def objective(trial):
            # Generar parámetros para el trial
            params = {}
            for param_name, param_range in param_space.items():
                if isinstance(param_range, tuple) and len(param_range) == 2:
                    if isinstance(param_range[0], int):
                        params[param_name] = trial.suggest_int(param_name, param_range[0], param_range[1])
                    elif isinstance(param_range[0], float):
                        params[param_name] = trial.suggest_float(param_name, param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    params[param_name] = trial.suggest_categorical(param_name, param_range)
            
            # EARLY STOPPING ADAPTATIVO - añadir parámetros de early stopping
            if 'XGB' in model.__class__.__name__ or 'LGBM' in model.__class__.__name__:
                # Añadir early stopping rounds adaptativo basado en el tamaño del dataset
                early_stopping_rounds = max(20, min(100, len(X_train) // 100))
                if 'early_stopping_rounds' not in params:
                    params['early_stopping_rounds'] = early_stopping_rounds
            
            # Configurar modelo con parámetros del trial
            trial_model = model.__class__(**{**model.get_params(), **params})
            
            # CROSS-VALIDATION MÁS ROBUSTO
            return self._robust_cross_validation(trial_model, X_train, y_train, model_name)
        
        # Crear estudio de optimización con early stopping para trials
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Early stopping para estudios Optuna
        early_stopping_callback = optuna.study.MaxTrialsCallback(self.n_trials, states=None)
        
        # Optimizar con early stopping
        import optuna.logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False, 
                      callbacks=[early_stopping_callback])
        
        return study.best_params
    
    def _robust_cross_validation(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                               model_name: str) -> float:
        """
        Cross-validation adaptativo para optimización de hiperparámetros durante Optuna trials.
        Retorna score de MAE para que Optuna encuentre los mejores hiperparámetros.
        """
        # CONFIGURACIÓN ADAPTATIVA DE CV
        data_size = len(X_train)
        
        if data_size < 500:
            # Dataset pequeño: usar holdout robusto
            n_splits = 1
            test_size = 0.25
        elif data_size < 2000:
            # Dataset mediano: 3 folds temporales
            n_splits = 3
            test_size = 0.2
        else:
            # Dataset grande: 5 folds temporales
            n_splits = 5
            test_size = 0.15
        
        scores = []
        
        if n_splits == 1:
            # Holdout simple pero robusto
            split_idx = int(len(X_train) * (1 - test_size))
            X_fold_train, X_fold_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
            y_fold_train, y_fold_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
            
            score = self._train_and_evaluate_fold(model, X_fold_train, y_fold_train, 
                                                 X_fold_val, y_fold_val, model_name)
            scores.append(score)
        else:
            # Time Series Cross-Validation robusto
            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=int(data_size * test_size))
            
            for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                score = self._train_and_evaluate_fold(model, X_fold_train, y_fold_train, 
                                                     X_fold_val, y_fold_val, model_name, fold_idx)
                scores.append(score)
        
        # Retornar score robusto (media con penalización por varianza)
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0
        
        # Penalizar alta varianza entre folds (indica overfitting)
        robust_score = mean_score + (std_score * 0.1)
        
        return robust_score
    
    def _train_and_evaluate_fold(self, model, X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                               model_name: str, fold_idx: int = 0) -> float:
        """
        Entrenar y evaluar un fold con early stopping adaptativo
        """
        try:
            # EARLY STOPPING ADAPTATIVO SEGÚN EL TIPO DE MODELO
            if hasattr(model, 'fit'):
                if 'XGB' in model.__class__.__name__:
                    # XGBoost con early stopping
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        verbose=False
                    )
                elif 'LGBM' in model.__class__.__name__:
                    # LightGBM con early stopping
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=[(X_val_fold, y_val_fold)],
                        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
                    )
                elif 'CatBoost' in model.__class__.__name__:
                    # CatBoost con early stopping
                    model.fit(
                        X_train_fold, y_train_fold,
                        eval_set=(X_val_fold, y_val_fold),
                        early_stopping_rounds=20,
                        verbose=False
                    )
                else:
                    # Modelos estándar sin early stopping
                    model.fit(X_train_fold, y_train_fold)
            
            # Evaluar predicción
            y_pred = model.predict(X_val_fold)
            mae = mean_absolute_error(y_val_fold, y_pred)
            
            return mae
            
        except Exception as e:
            logger.warning(f"Error en fold {fold_idx} para {model_name}: {str(e)}")
            # Retornar score alto para indicar error
            return 999.0
    
    def _perform_temporal_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Validación cruzada temporal del modelo completo ya entrenado para evaluar estabilidad.
        Usa modelos base y meta-learner entrenados para calcular métricas de CV finales.
        """
        
        # Usar menos folds para evitar problemas con datasets pequeños
        n_splits = min(3, len(X_train) // 100)  # Máximo 3 folds, mínimo 100 muestras por fold
        if n_splits < 2:
            logger.warning("Dataset muy pequeño para CV temporal, saltando validación cruzada")
            self.cv_scores = {'mae_mean': 0, 'mae_std': 0, 'r2_mean': 0, 'cv_scores': []}
            return
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
            try:
                X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Usar modelos base ya entrenados para predicciones en este fold
                fold_base_predictions = np.zeros((len(X_fold_val), len(self.trained_base_models)))
                
                for i, (name, model) in enumerate(self.trained_base_models.items()):
                    fold_base_predictions[:, i] = model.predict(X_fold_val)
                
                # Usar meta-learner ya entrenado
                fold_predictions = self.meta_learner.predict(fold_base_predictions)
                
                # Métricas del fold
                mae = mean_absolute_error(y_fold_val, fold_predictions)
                r2 = r2_score(y_fold_val, fold_predictions)
                
                cv_scores.append({'mae': mae, 'r2': r2})
                
                # Solo mostrar el último fold
                if fold == n_splits:
                    logger.info(f"CV completado: MAE={mae:.3f}, R²={r2:.3f}")
                
            except Exception as e:
                logger.warning(f"Error en CV fold {fold}: {str(e)}")
                continue
        
        if not cv_scores:
            logger.warning("No se pudo completar ningún fold de CV")
            self.cv_scores = {'mae_mean': 0, 'mae_std': 0, 'r2_mean': 0, 'cv_scores': []}
            return
        
        # Promediar scores
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        std_mae = np.std([score['mae'] for score in cv_scores])
        
        self.cv_scores = {
            'mae_mean': avg_mae,
            'mae_std': std_mae,
            'r2_mean': avg_r2,
            'cv_scores': cv_scores
        }
    
    def _calculate_feature_importance(self, features: List[str]):
        """Calcular importancia de features desde modelos base"""
        
        feature_importance = {}
        
        for model_name, model_config in self.base_models.items():
            model = model_config['model']
            
            try:
                if hasattr(model, 'feature_importances_'):
                    # XGBoost, LightGBM, CatBoost
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    # Ridge, Linear models
                    importances = np.abs(model.coef_)
                else:
                    continue
                
                # Normalizar importancias
                importances = importances / np.sum(importances)
                
                for i, feature in enumerate(features):
                    if feature not in feature_importance:
                        feature_importance[feature] = 0
                    feature_importance[feature] += importances[i]
                    
            except Exception as e:
                logger.warning(f"No se pudo calcular importancia para {model_name}: {e}")
                continue
        
        # Normalizar importancias finales
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            for feature in feature_importance:
                feature_importance[feature] /= total_importance
        
        # Ordenar por importancia
        self.feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones usando el modelo entrenado con stacking OOF
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Array con predicciones de rebounds
        """
        if not hasattr(self, 'trained_base_models') or not hasattr(self, 'meta_learner'):
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Generar features (modificar DataFrame directamente)
        features = self.feature_engineer.generate_all_features(df)
        
        # Determinar expected_features dinámicamente del modelo entrenado
        try:
            if hasattr(self, 'expected_features'):
                expected_features = self.expected_features
            elif hasattr(self, 'selected_features') and self.selected_features:
                expected_features = self.selected_features
            else:
                # Fallback: usar todas las features numéricas disponibles
                expected_features = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
        except Exception as e:
            logger.warning(f"No se pudieron obtener expected_features: {e}")
            expected_features = features if features else []
        
        # Reordenar DataFrame según expected_features
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
        
        # 🔧 CRÍTICO: Aplicar el scaler antes de predecir (igual que en entrenamiento)
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=expected_features, index=X.index)
        
        # Realizar predicciones usando stacking con out-of-fold predictions
        predictions = self._predict_with_stacking(X)
    
        # Aplicar procesamiento robusto para rebotes
        predictions = self._apply_robust_prediction_processing(predictions)
        
        return predictions
    
    def _apply_robust_prediction_processing(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplica procesamiento robusto para predicciones de rebotes (TRB)
        - Manejo de outliers con IQR
        - Predicciones enteras
        - Límites realistas para rebotes (0-25)
        """
        try:
            # 1. Manejo de outliers usando IQR
            Q1 = np.percentile(predictions, 25)
            Q3 = np.percentile(predictions, 75)
            IQR = Q3 - Q1
            
            # Factor permisivo para rebotes (valores más variables que triples)
            outlier_factor = 3.0  # Más permisivo
            lower_bound = max(0, Q1 - outlier_factor * IQR)  # No permitir negativos
            upper_bound = Q3 + outlier_factor * IQR  # SIN límite artificial de 25
            
            # Aplicar winsorización
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            # 2. Asegurar que sean no negativas
            predictions = np.maximum(predictions, 0)
            
            # 3. CALIBRACIÓN POR RANGOS basada en análisis detallado
            predictions_calibrated = self._apply_range_specific_calibration(predictions)
            
            # 4. Convertir a enteros usando redondeo probabilístico
            predictions_int = np.zeros_like(predictions_calibrated)
            for i, pred in enumerate(predictions_calibrated):
                floor_val = int(np.floor(pred))
                prob = pred - floor_val
                predictions_int[i] = floor_val + np.random.binomial(1, prob)
            
            # 5. Aplicar límites finales para rebotes (sin límite superior artificial)
            predictions_int = np.maximum(predictions_int, 0)  # Solo evitar negativos
            
            return predictions_int.astype(int)
            
        except Exception as e:
            logger.warning(f"Error en procesamiento robusto: {e}")
            # Fallback: solo redondear y limitar
            predictions = np.maximum(predictions, 0)
            return np.round(np.maximum(predictions, 0)).astype(int)  # Sin límite superior
    
    def _apply_range_specific_calibration(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplica calibración específica por rangos de predicción basada en análisis detallado
        """
        calibrated = predictions.copy()
        
        # Rangos y factores de calibración basados en el análisis
        calibration_ranges = [
            (0, 3, 0.7),      # Muy Bajo: reducir sobreestimación (+71.1% bias)
            (4, 6, 1.05),     # Bajo: ligero ajuste para mejorar R²
            (7, 9, 1.08),     # Medio-Bajo: ajuste para mejorar R²
            (10, 12, 1.12),   # Medio: ajuste para mejorar R²
            (13, 15, 1.15),   # Medio-Alto: reducir subestimación (-11.2% bias)
            (16, 20, 1.25),   # Alto: reducir subestimación (-26.4% bias)
            (21, 50, 1.35)    # Muy Alto: reducir subestimación extrema
        ]
        
        for min_val, max_val, factor in calibration_ranges:
            if max_val == 50:  # Para el rango extremo
                mask = predictions >= min_val
            else:
                mask = (predictions >= min_val) & (predictions < max_val)
            
            if mask.sum() > 0:
                calibrated[mask] = predictions[mask] * factor
                logger.debug(f"Calibración rango {min_val}-{max_val}: {mask.sum()} predicciones, factor {factor}")
        
        return calibrated
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado completo con scaler y metadata"""
        if not hasattr(self, 'trained_base_models') or self.trained_base_models is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar objeto completo con todos los componentes necesarios
        joblib.dump(self, filepath, compress=3, protocol=4)
        logger.info(f"Modelo TRB completo guardado (incluye scaler): {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo entrenado (compatible con ambos formatos)"""
        try:
            # Intentar cargar objeto completo (nuevo formato)
            loaded_obj = joblib.load(filepath)
            if isinstance(loaded_obj, StackingTRBModel):
                # Es un objeto completo, copiar todos los atributos
                self.__dict__.update(loaded_obj.__dict__)
                logger.info(f"Modelo TRB completo cargado (incluye scaler): {filepath}")
                return
            elif hasattr(loaded_obj, 'predict'):
                # Es solo el stacking_model
                self.stacking_model = loaded_obj
                logger.info(f"Modelo TRB (stacking_model) cargado desde: {filepath}")
                return
            else:
                # No es un modelo directo, tratar como formato antiguo
                raise ValueError("No es modelo directo")
        except (ValueError, AttributeError):
            # Formato antiguo (diccionario)
            try:
                model_data = joblib.load(filepath)
                if isinstance(model_data, dict) and 'stacking_model' in model_data:
                    self.stacking_model = model_data['stacking_model']
                    self.base_models = model_data.get('base_models', {})
                    self.feature_engineer = model_data.get('feature_engineer')
                    self.validation_metrics = model_data.get('validation_metrics', {})
                    self.cv_scores = model_data.get('cv_scores', {})
                    self.feature_importance = model_data.get('feature_importance', {})
                    self.best_params_per_model = model_data.get('best_params_per_model', {})
                    logger.info(f"Modelo TRB (formato legacy) cargado desde: {filepath}")
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except Exception as e:
                raise ValueError(f"No se pudo cargar el modelo TRB: {e}")
        
        # Crear stacking_model para compatibilidad con predictor
        if hasattr(self, 'trained_base_models') and hasattr(self, 'meta_learner') and self.trained_base_models and self.meta_learner:
            from sklearn.ensemble import StackingRegressor
            
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
            logger.info("✅ StackingRegressor creado para compatibilidad con predictor TRB")

class XGBoostTRBModel (StackingTRBModel):
    """
    Modelo XGBoost simplificado para TRB con compatibilidad con el sistema existente
    Mantiene la interfaz del modelo de PTS pero optimizado para rebotes
    """
    
    def __init__(self, enable_neural_network: bool = False, enable_gpu: bool = True, 
                 random_state: int = 42, teams_df: pd.DataFrame = None,
                 players_df: pd.DataFrame = None, players_quarters_df: pd.DataFrame = None):
        """Inicializar modelo XGBoost TRB con stacking ensemble"""
        self.stacking_model = StackingTRBModel(
            enable_neural_network=enable_neural_network,
            enable_gpu=enable_gpu,
            random_state=random_state,
            teams_df=teams_df,
            players_df=players_df,
            players_quarters_df=players_quarters_df
        )
        
        # Atributos para compatibilidad
        self.model = None
        self.validation_metrics = {}
        self.best_params = {}
        self.cutoff_date = None
        
        logger.info("Modelo XGBoost TRB inicializado con stacking completo")
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento manteniendo compatibilidad con la interfaz original
        """
        # Llamar al método de entrenamiento del stacking
        result = self.stacking_model.train(df)
        
        # Asignar para compatibilidad
        self.model = self.stacking_model.stacking_model
        self.validation_metrics = result
        
        if self.stacking_model.best_params_per_model:
            # Usar parámetros del modelo con mejor rendimiento
            best_model_name = min(
                self.stacking_model.best_params_per_model.keys(),
                key=lambda x: self.stacking_model.cv_scores.get('mae_mean', float('inf'))
            )
            self.best_params = self.stacking_model.best_params_per_model[best_model_name]
        
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
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Realizar predicciones con procesamiento robusto"""
        # Usar el método predict del StackingTRBModel que maneja las características correctamente
        return self.stacking_model.predict(df)
    
    def save_model(self, filepath: str):
        """Guardar modelo"""
        self.stacking_model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Cargar modelo"""
        self.stacking_model.load_model(filepath)
        self.model = self.stacking_model.stacking_model
        self.validation_metrics = self.stacking_model.validation_metrics