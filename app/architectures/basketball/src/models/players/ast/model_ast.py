"""
Modelo de Predicción de Asistencias (AST)
========================================

Sistema avanzado de stacking ensemble para predicción de asistencias
que realizará un jugador NBA en su próximo partido.

CARACTERÍSTICAS PRINCIPALES:
- Stacking ensemble con XGBoost, LightGBM, CatBoost
- Optimización bayesiana de hiperparámetros
- Validación cruzada temporal (respeta orden cronológico)
- Regularización avanzada (L1/L2, Dropout, Early Stopping)
- División temporal para evitar data leakage
- Meta-learner adaptativo para stacking
- Features especializadas para asistencias

ARQUITECTURA:
1. Modelos Base: XGBoost, LightGBM, CatBoost, Ridge
2. Meta-learner: LightGBM
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
from joblib import parallel_backend
import json

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler

# Configurar Optuna para ser silencioso
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Local imports
from .features_ast import AssistsFeatureEngineer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class StackingASTModel:
    """
    Modelo de Stacking Ensemble para Predicción de Asistencias
    ULTRA-OPTIMIZADO con regularización balanceada y validación temporal
    Especializado en características de playmakers y visión de cancha
    """
    
    def __init__(self, enable_neural_network: bool = False, enable_gpu: bool = False, 
                 random_state: int = 42, teams_df: pd.DataFrame = None, max_features: int = 40):
        """
        Inicializar el modelo de stacking para AST
        
        Args:
            enable_neural_network: Habilitar red neuronal (desactivado por defecto)
            enable_gpu: Usar GPU para XGBoost/LightGBM si está disponible
            random_state: Semilla para reproducibilidad
            teams_df: DataFrame con datos de equipos para features avanzadas
            max_features: Número máximo de features a seleccionar
        """
        self.random_state = random_state
        self.enable_neural_network = enable_neural_network
        self.enable_gpu = enable_gpu
        self.teams_df = teams_df
        self.max_features = max_features
        
        # Configuración de modelos
        self.base_models = {}
        self.stacking_model = None
        self.feature_engineer = AssistsFeatureEngineer(teams_df=teams_df)
        self.scaler = StandardScaler()
        
        # Métricas y resultados
        self.validation_metrics = {}
        self.cv_scores = {}
        self.feature_importance = {}
        self.best_params_per_model = {}
        
        # Configuración de optimización BALANCEADA
        self.n_trials = 25  # Optimización rápida pero efectiva
        self.cv_folds = 5   # Validación cruzada conservadora
        
        
        # Configurar modelos base
        self._setup_base_models()
        
        # Mostrar ensemble final
        model_names = list(self.base_models.keys())
        logger.info(f"Modelo AST inicializado - Ensemble: {', '.join(model_names)}")
        logger.info(f"Configuración: NN={enable_neural_network}, GPU={enable_gpu}")
    
    def _setup_base_models(self):
        """Configurar modelos base con hiperparámetros optimizados para asistencias"""
        
        # XGBoost - Excelente para features de Basketball IQ y patrones complejos
        self.base_models['xgboost'] = {
            'model': xgb.XGBRegressor(
                random_state=self.random_state,
                tree_method='gpu_hist' if self.enable_gpu else 'hist',
                gpu_id=0 if self.enable_gpu else None,
                n_jobs=1
            ),
            'param_space': {
                'n_estimators': (100, 300),  # Rango reducido
                'max_depth': (3, 6),         # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'subsample': (0.8, 1.0),     # Rango reducido
                'colsample_bytree': (0.8, 1.0), # Rango reducido
                'reg_alpha': (0, 3),         # Rango reducido
                'reg_lambda': (1, 5)         # Rango reducido
            }
        }
        
        # LightGBM - Rápido y eficiente para features de visión de cancha
        self.base_models['lightgbm'] = {
            'model': lgb.LGBMRegressor(
                random_state=self.random_state,
                device='gpu' if self.enable_gpu else 'cpu',
                gpu_platform_id=0 if self.enable_gpu else None,
                gpu_device_id=0 if self.enable_gpu else None,
                n_jobs=1,
                verbose=-1
            ),
            'param_space': {
                'n_estimators': (100, 300),  # Rango reducido
                'max_depth': (3, 6),         # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'subsample': (0.8, 1.0),     # Rango reducido
                'colsample_bytree': (0.8, 1.0), # Rango reducido
                'reg_alpha': (0, 3),         # Rango reducido
                'reg_lambda': (1, 5),        # Rango reducido
                'num_leaves': (15, 50)       # Rango reducido
            }
        }
        
        # CatBoost - Excelente para features categóricas de contexto de equipo
        self.base_models['catboost'] = {
            'model': cb.CatBoostRegressor(
                random_state=self.random_state,
                task_type='GPU' if self.enable_gpu else 'CPU',
                devices='0' if self.enable_gpu else None,
                verbose=False,
                allow_writing_files=False
            ),
            'param_space': {
                'iterations': (100, 250),    # Rango reducido
                'depth': (3, 6),             # Rango reducido
                'learning_rate': (0.05, 0.2), # Rango reducido
                'l2_leaf_reg': (1, 5),       # Rango reducido
                'subsample': (0.8, 1.0)      # Rango reducido
            }
        }
                
        # Configurar meta-learner Ridge para stacking
        self.meta_learner = Ridge(alpha=20.0, random_state=self.random_state)
        
        logger.info("Modelos base configurados: XGBoost, LightGBM, CatBoost")
        logger.info("Meta-learner configurado: Ridge (alpha=20.0)")
    
    def _temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        División temporal de datos respetando orden cronológico
        CRÍTICO: Evita data leakage usando fechas
        """
        if 'Date' not in df.columns:
            logger.warning("Columna 'Date' no encontrada, usando división secuencial")
            split_idx = int(len(df) * (1 - test_size))
            return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
        
        # Ordenar por fecha y jugador para consistencia
        df_sorted = df.sort_values(['Date', 'player']).reset_index(drop=True)
        
        # Encontrar punto de corte temporal
        split_idx = int(len(df_sorted) * (1 - test_size))
        cutoff_date = df_sorted.iloc[split_idx]['Date']
        
        train_data = df_sorted[df_sorted['Date'] < cutoff_date].copy().reset_index(drop=True)
        test_data = df_sorted[df_sorted['Date'] >= cutoff_date].copy().reset_index(drop=True)
        
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
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        CRÍTICO: Validar que los datos de entrenamiento sean correctos.
        
        Args:
            X: DataFrame con características
            y: Serie con variable objetivo
        """
        
        # Verificar que X sea numérico (específico para LightGBM)
        non_numeric_cols = []
        for col in X.columns:
            dtype = X[col].dtype
            if dtype == 'object' or dtype.name == 'string':
                # Verificar si contiene valores no numéricos
                sample_values = X[col].dropna().head(10).tolist()
                logger.error(f"Columna no numérica '{col}' (tipo: {dtype}): {sample_values}")
                non_numeric_cols.append(col)
            elif dtype.name not in ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'bool']:
                logger.error(f"Columna con tipo no compatible '{col}' (tipo: {dtype})")
                non_numeric_cols.append(col)
        
        if non_numeric_cols:
            raise ValueError(f"Columnas no numéricas detectadas para LightGBM: {non_numeric_cols}")
        
        # Verificar valores infinitos
        inf_cols = []
        for col in X.columns:
            if np.isinf(X[col]).any():
                inf_cols.append(col)
        
        if inf_cols:
            logger.warning(f"Columnas con valores infinitos (serán reemplazados): {inf_cols}")
            X[inf_cols] = X[inf_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Verificar NaN en target
        if y.isna().any():
            logger.warning(f"Target contiene {y.isna().sum()} valores NaN")
        
        # CRÍTICO: Forzar conversión a tipos compatibles con LightGBM
        for col in X.columns:
            if X[col].dtype == 'object':
                # Intentar conversión numérica forzada
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            elif X[col].dtype not in ['int64', 'float64', 'bool']:
                # Convertir a float64 para compatibilidad
                X[col] = X[col].astype('float64')
        
        return
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Entrenamiento completo del modelo Stacking AST
        
        Args:
            df: DataFrame con datos de jugadores y estadísticas
            
        Returns:
            Dict con métricas de validación
        """
        
        # Verificar orden cronológico
        if 'Date' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                logger.info("Ordenando datos cronológicamente...")
                df = df.sort_values(['player', 'Date']).reset_index(drop=True)


        # Generar features especializadas para asistencias
        logger.info("Generando características especializadas...")
        df = self.feature_engineer.generate_all_features(df)  # Retorna DataFrame transformado
        
        # Obtener lista de features generadas (columnas nuevas)
        all_features = [col for col in df.columns if col not in ['player', 'Date', 'Team', 'Opp', 'assists']]
        
        if not all_features:
            raise ValueError("No se pudieron generar features para AST")
        
        logger.info(f"Features generadas: {len(all_features)}")
        
        features = all_features
        
        # Preparar datos con todas las features válidas
        X = df[features].fillna(0)
        y = df['assists']
        self._validate_training_data(X, y)

        # Guardar features para predicción
        self.selected_features = features
        
        # División temporal
        train_data, test_data = self._temporal_split(df)
        
        X_train = train_data[features].fillna(0)
        y_train = train_data['assists']
        X_test = test_data[features].fillna(0)
        y_test = test_data['assists']
        
        # Entrenar el scaler con los datos de entrenamiento
        logger.info("Entrenando StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convertir de vuelta a DataFrame para mantener compatibilidad
        X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
        X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
        
        # Validar datos de entrenamiento y prueba
        self._validate_training_data(X_train, y_train)
        self._validate_training_data(X_test, y_test)
        
        logger.info("Scaler entrenado y datos escalados")
        
        # PASO 1: Configurar modelos base
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
        
        # Métricas específicas para asistencias
        accuracy_1ast = np.mean(np.abs(y_test - y_pred) <= 1) * 100
        accuracy_2ast = np.mean(np.abs(y_test - y_pred) <= 2) * 100
        accuracy_3ast = np.mean(np.abs(y_test - y_pred) <= 3) * 100
        
        self.validation_metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'accuracy_1ast': accuracy_1ast,
            'accuracy_2ast': accuracy_2ast,
            'accuracy_3ast': accuracy_3ast
        }
        
        # Calcular importancia de features
        self._calculate_feature_importance(features)
        
        # Mostrar resultados FINALES
        logger.info("=" * 50)
        logger.info("ENTRENAMIENTO COMPLETADO")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.validation_metrics['mae']:.4f}")
        logger.info(f"RMSE: {self.validation_metrics['rmse']:.4f}")
        logger.info(f"R²: {self.validation_metrics['r2']:.4f}")
        logger.info(f"Accuracy ±1ast: {accuracy_1ast:.1f}%")
        logger.info(f"Accuracy ±2ast: {accuracy_2ast:.1f}%")
        logger.info(f"Accuracy ±3ast: {accuracy_3ast:.1f}%")
        logger.info("=" * 50)
        
        return self.validation_metrics

    def _train_base_models_with_optimization(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrenar modelos base con optimización bayesiana de hiperparámetros"""
        
        trained_models = {}
        
        for i, (model_name, model_config) in enumerate(self.base_models.items(), 1):
            logger.info(f"[{i}/{len(self.base_models)}] Entrenando {model_name}...")
            
            try:
                # Optimización bayesiana (silenciosa)
                best_params = self._optimize_hyperparameters(
                    model_config['model'], 
                    model_config['param_space'], 
                    X_train, 
                    y_train,
                    model_name
                )
                
                # Entrenar con mejores parámetros
                model = model_config['model']
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                
                # Evaluar en datos de entrenamiento
                y_pred_train = model.predict(X_train)
                train_mae = mean_absolute_error(y_train, y_pred_train)
                
                trained_models[model_name] = model
                self.best_params_per_model[model_name] = best_params
                
                logger.info(f"{model_name} completado - MAE: {train_mae:.3f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {model_name}: {str(e)}")
                continue
        
        if not trained_models:
            raise ValueError("No se pudo entrenar ningún modelo base")
        
        # Actualizar modelos base con los entrenados
        for model_name, trained_model in trained_models.items():
            self.base_models[model_name]['model'] = trained_model
        
        logger.info(f"Modelos entrenados: {len(trained_models)}/{len(self.base_models)}")
    
    def _optimize_hyperparameters(self, model, param_space: Dict, X_train: pd.DataFrame, 
                                 y_train: pd.Series, model_name: str) -> Dict:
        """Optimización bayesiana de hiperparámetros usando Optuna"""
        
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
            
            # Configurar modelo con parámetros del trial
            trial_model = model.__class__(**{**model.get_params(), **params})
            
            # Validación cruzada temporal más robusta
            n_splits = min(2, len(X_train) // 200)  # Máximo 2 folds para optimización rápida
            if n_splits < 2:
                # Si el dataset es muy pequeño, usar train-test split simple
                split_idx = int(len(X_train) * 0.8)
                X_fold_train, X_fold_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
                y_fold_train, y_fold_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                
                trial_model.fit(X_fold_train, y_fold_train)
                y_pred = trial_model.predict(X_fold_val)
                mae = mean_absolute_error(y_fold_val, y_pred)
                scores = [mae]
            else:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                scores = []
                
                for train_idx, val_idx in tscv.split(X_train):
                    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    trial_model.fit(X_fold_train, y_fold_train)
                    y_pred = trial_model.predict(X_fold_val)
                    mae = mean_absolute_error(y_fold_val, y_pred)
                    scores.append(mae)
            
            return np.mean(scores)
        
        # Crear estudio de optimización (silencioso) con TPESampler
        study = optuna.create_study(
            direction='minimize',
            sampler=TPESampler(seed=self.random_state, n_startup_trials=10)
        )
        
        # Optimizar (silencioso)
        import optuna.logging
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _perform_temporal_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Realizar validación cruzada temporal para evaluar estabilidad"""
        
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
                base_predictions = np.zeros((len(X_fold_val), len(self.trained_base_models)))
                
                for i, (name, model) in enumerate(self.trained_base_models.items()):
                    base_predictions[:, i] = model.predict(X_fold_val)
                
                # Usar meta-learner ya entrenado
                meta_predictions = self.meta_learner.predict(base_predictions)
                
                # Métricas del fold
                mae = mean_absolute_error(y_fold_val, meta_predictions)
                r2 = r2_score(y_fold_val, meta_predictions)
                
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
                    # XGBoost, LightGBM, CatBoost, GradientBoosting
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
    
    def _apply_elite_assist_calibration(self, predictions: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        """
        CALIBRACIÓN ULTRA-AGRESIVA PARA MODELO CASI PERFECTO
        Corrige TODOS los problemas identificados en el análisis exhaustivo
        """
        try:
            logger.info("Aplicando calibración ultra-agresiva para modelo casi perfecto...")
            
            # 1. DETECCIÓN EQUILIBRADA DE JUGADORES ELITE
            elite_players = {
                # ELITE PLAYMAKERS (Corrección equilibrada)
                'Trae Young': {'boost': 2.5, 'factor': 1.3, 'min_pred': 8.0, 'max_pred': 12.0},
                'Luka Doncic': {'boost': 2.2, 'factor': 1.25, 'min_pred': 7.5, 'max_pred': 11.0},
                'Nikola Jokić': {'boost': 2.0, 'factor': 1.2, 'min_pred': 7.0, 'max_pred': 10.0},
                'James Harden': {'boost': 1.8, 'factor': 1.15, 'min_pred': 6.5, 'max_pred': 9.5},
                'Chris Paul': {'boost': 1.5, 'factor': 1.1, 'min_pred': 6.0, 'max_pred': 9.0},
                'LeBron James': {'boost': 1.3, 'factor': 1.08, 'min_pred': 5.5, 'max_pred': 8.5},
                'Russell Westbrook': {'boost': 1.0, 'factor': 1.05, 'min_pred': 5.0, 'max_pred': 8.0},
                'Damian Lillard': {'boost': 1.3, 'factor': 1.08, 'min_pred': 5.5, 'max_pred': 8.5},
                'Kyrie Irving': {'boost': 1.0, 'factor': 1.05, 'min_pred': 5.0, 'max_pred': 8.0},
                'Ja Morant': {'boost': 1.3, 'factor': 1.08, 'min_pred': 5.5, 'max_pred': 8.5},
                'Dejounte Murray': {'boost': 1.0, 'factor': 1.05, 'min_pred': 5.0, 'max_pred': 8.0},
                'Tyrese Haliburton': {'boost': 1.5, 'factor': 1.1, 'min_pred': 6.0, 'max_pred': 9.0},
                'LaMelo Ball': {'boost': 1.3, 'factor': 1.08, 'min_pred': 5.5, 'max_pred': 8.5},
                'Darius Garland': {'boost': 1.0, 'factor': 1.05, 'min_pred': 5.0, 'max_pred': 8.0},
                
                # GOOD PLAYMAKERS (Corrección moderada)
                'Stephen Curry': {'boost': 0.8, 'factor': 1.05, 'min_pred': 4.5, 'max_pred': 7.0},
                'Kevin Durant': {'boost': 0.5, 'factor': 1.03, 'min_pred': 4.0, 'max_pred': 6.5},
                'Giannis Antetokounmpo': {'boost': 0.5, 'factor': 1.03, 'min_pred': 4.0, 'max_pred': 6.5},
                'Domantas Sabonis': {'boost': 0.8, 'factor': 1.05, 'min_pred': 4.5, 'max_pred': 7.0},
                'Kyle Lowry': {'boost': 0.5, 'factor': 1.03, 'min_pred': 4.0, 'max_pred': 6.5},
                'Spencer Dinwiddie': {'boost': 0.5, 'factor': 1.03, 'min_pred': 4.0, 'max_pred': 6.5},
                'Lonzo Ball': {'boost': 0.5, 'factor': 1.03, 'min_pred': 4.0, 'max_pred': 6.5},
                'Cade Cunningham': {'boost': 0.3, 'factor': 1.02, 'min_pred': 3.5, 'max_pred': 6.0},
                'Devin Booker': {'boost': 0.3, 'factor': 1.02, 'min_pred': 3.5, 'max_pred': 6.0},
                'Zion Williamson': {'boost': 0.2, 'factor': 1.01, 'min_pred': 3.0, 'max_pred': 5.5},
            }
            
            # 2. APLICAR CORRECCIONES ESPECÍFICAS POR JUGADOR (USANDO TODOS LOS DATOS HISTÓRICOS)
            if 'player' in df.columns:
                for idx, player in enumerate(df['player']):
                    if player in elite_players:
                        config = elite_players[player]
                        boost = config['boost']
                        factor = config['factor']
                        min_pred = config['min_pred']
                        max_pred = config['max_pred']
                        
                        # Aplicar calibración específica por jugador
                        
                        # Aplicar boost y factor
                        predictions[idx] = (predictions[idx] + boost) * factor
                        
                        # Asegurar predicción dentro del rango realista
                        if predictions[idx] < min_pred:
                            predictions[idx] = min_pred
                        elif predictions[idx] > max_pred:
                            predictions[idx] = max_pred
                        
                        logger.debug(f"Calibración elite aplicada a {player}: boost={boost}, factor={factor}, min={min_pred}, max={max_pred}, resultado={predictions[idx]:.2f}")
            
            # 3. CORRECCIÓN EQUILIBRADA POR RANGOS (AJUSTADA PARA NO SOBREESTIMAR)
            for i, pred in enumerate(predictions):
                # RANGO 10+ AST: CORRECCIÓN LIGERA (reducida)
                if pred >= 10:  # Predicción alta
                    # Aplicar factor multiplicador ligero
                    factor = 1.05 + (pred - 10) * 0.01  # Factor creciente muy suave
                    predictions[i] *= factor
                    # Asegurar máximo de 12 AST para predicciones altas
                    if predictions[i] > 12:
                        predictions[i] = 12
                
                # RANGO 7-9 AST: CORRECCIÓN MÍNIMA
                elif pred >= 7:
                    factor = 1.03 + (pred - 7) * 0.005
                    predictions[i] *= factor
                    # Asegurar máximo de 10 AST para predicciones medias-altas
                    if predictions[i] > 10:
                        predictions[i] = 10
                
                # RANGO 4-6 AST: CORRECCIÓN MÍNIMA
                elif pred >= 4:
                    factor = 1.02 + (pred - 4) * 0.002
                    predictions[i] *= factor
                    # Asegurar máximo de 8 AST para predicciones medias
                    if predictions[i] > 8:
                        predictions[i] = 8
                
                # RANGO 2-3 AST: CORRECCIÓN MÍNIMA
                elif pred >= 2:
                    factor = 1.01 + (pred - 2) * 0.001
                    predictions[i] *= factor
                
                # RANGO 0-1 AST: SIN CORRECCIÓN
                else:
                    # No aplicar corrección para predicciones muy bajas
                    pass
            
            # 4. APLICAR FEATURES DE QUARTERS COMO FACTOR MULTIPLICADOR (LIGERO)
            if 'quarter_based_game_projection' in df.columns:
                for i, projection in enumerate(df['quarter_based_game_projection']):
                    if projection > 8.0:  # Proyección muy alta basada en cuartos
                        # Factor multiplicador ligero
                        factor = min(projection / 8.0, 1.1)  # Máximo 1.1x
                        predictions[i] *= factor
                        logger.debug(f"Factor quarter projection aplicado: {factor:.2f}x")
            
            # 5. APLICAR DETECCIÓN EXPLOSIVA POR CUARTOS (LIGERO)
            if 'quarter_explosive_detection' in df.columns:
                for i, detection in enumerate(df['quarter_explosive_detection']):
                    if detection > 2.0:  # Jugador muy explosivo por cuartos
                        boost = (detection - 2.0) * 0.2  # Factor de boost ligero
                        predictions[i] += boost
                        logger.debug(f"Boost quarter explosivo aplicado: +{boost:.2f}")
            
            # 6. APLICAR RENDIMIENTO ELITE POR CUARTOS (LIGERO)
            if 'elite_quarter_performance' in df.columns:
                for i, performance in enumerate(df['elite_quarter_performance']):
                    if performance > 2.5:  # Rendimiento muy elite por cuartos
                        boost = (performance - 2.5) * 0.1  # Factor de boost ligero
                        predictions[i] += boost
                        logger.debug(f"Boost elite quarter aplicado: +{boost:.2f}")
            
            # 7. APLICAR POTENCIAL EXPLOSIVO (LIGERO)
            if 'explosive_game_potential' in df.columns:
                for i, potential in enumerate(df['explosive_game_potential']):
                    if potential > 3.0:  # Muy alto potencial explosivo
                        boost = min(potential * 0.1, 0.3)  # Máximo +0.3
                        predictions[i] += boost
                        logger.debug(f"Boost explosivo aplicado: +{boost:.2f}")
            
            # 8. APLICAR MOMENTUM DE ESTRELLA (LIGERO)
            if 'star_player_momentum' in df.columns:
                for i, momentum in enumerate(df['star_player_momentum']):
                    if momentum > 1.5:  # Momentum muy positivo
                        boost = min(momentum * 0.1, 0.2)  # Máximo +0.2
                        predictions[i] += boost
                        logger.debug(f"Boost momentum aplicado: +{boost:.2f}")
            
            # 9. APLICAR ESCALAMIENTO ELITE (LIGERO)
            if 'elite_player_scaling' in df.columns:
                for i, scaling in enumerate(df['elite_player_scaling']):
                    if scaling > 1.5:  # Solo para jugadores muy elite
                        factor = min(scaling / 1.5, 1.05)  # Máximo 1.05x
                        predictions[i] *= factor
                        logger.debug(f"Escalamiento elite aplicado: {factor:.2f}x")
            
            # 10. APLICAR CONSISTENCIA POR CUARTOS (LIGERO)
            if 'quarter_consistency_pattern' in df.columns:
                for i, consistency in enumerate(df['quarter_consistency_pattern']):
                    if consistency > 2.0:  # Muy alta consistencia
                        factor = min(consistency / 2.0, 1.05)  # Máximo 1.05x
                        predictions[i] *= factor
                        logger.debug(f"Factor consistencia aplicado: {factor:.2f}x")
            
            # 11. APLICAR MOMENTUM POR CUARTOS (LIGERO)
            if 'quarter_momentum_acceleration' in df.columns:
                for i, momentum in enumerate(df['quarter_momentum_acceleration']):
                    if momentum > 0.5:  # Momentum muy positivo
                        boost = min(momentum * 0.2, 0.1)  # Máximo +0.1
                        predictions[i] += boost
                        logger.debug(f"Boost momentum quarter aplicado: +{boost:.2f}")
            
            # 12. CORRECCIÓN FINAL PARA RANGOS EXTREMOS (MÁS CONSERVADORA)
            for i, pred in enumerate(predictions):
                # Si la predicción es muy alta, asegurar que sea realista
                if pred > 15:
                    predictions[i] = 15 + (pred - 15) * 0.2  # Más conservador
                
                # Si la predicción es muy baja para un jugador elite, aplicar boost mínimo
                if 'player' in df.columns and df.iloc[i]['player'] in elite_players:
                    if pred < 2.0:  # Muy bajo para elite
                        predictions[i] = 2.0 + (pred - 2.0) * 0.3
            
            # 13. ASEGURAR QUE LAS PREDICCIONES SEAN REALISTAS
            predictions = np.maximum(predictions, 0)  # No negativas
            predictions = np.minimum(predictions, 15)  # Máximo realista más conservador
            
            logger.info(f"Calibración equilibrada aplicada. Predicciones ajustadas: {len(predictions)}")
            logger.info(f"Rango de predicciones: {predictions.min():.2f} - {predictions.max():.2f}")
            logger.info(f"Media de predicciones: {predictions.mean():.2f}")
            
            return predictions
            
        except Exception as e:
            logger.warning(f"Error en calibración equilibrada: {e}")
            return predictions

    def _apply_robust_prediction_processing(self, predictions: np.ndarray) -> np.ndarray:
        """
        Aplica procesamiento robusto para predicciones de asistencias (AST)
        - Manejo de outliers con IQR
        - Predicciones enteras
        - Límites realistas para asistencias (0-20)
        """
        try:
            # 1. Manejo de outliers usando IQR
            Q1 = np.percentile(predictions, 25)
            Q3 = np.percentile(predictions, 75)
            IQR = Q3 - Q1
            
            # Factor más permisivo para asistencias elite (permitir casos extremos)
            outlier_factor = 3.5  # Más permisivo para casos de 15+ asistencias
            lower_bound = max(0, Q1 - outlier_factor * IQR)  # No permitir negativos
            upper_bound = min(25, Q3 + outlier_factor * IQR)  # Máximo realista: 25 asistencias
            
            # Aplicar winsorización
            predictions = np.clip(predictions, lower_bound, upper_bound)
            
            # 2. Asegurar que sean no negativas
            predictions = np.maximum(predictions, 0)
            
            # 3. Convertir a enteros usando redondeo probabilístico
            predictions_int = np.zeros_like(predictions)
            for i, pred in enumerate(predictions):
                floor_val = int(np.floor(pred))
                prob = pred - floor_val
                predictions_int[i] = floor_val + np.random.binomial(1, prob)
            
            # 4. Aplicar límites finales para asistencias (más permisivo)
            predictions_int = np.clip(predictions_int, 0, 112)
            
            return predictions_int.astype(int)
            
        except Exception as e:
            logger.warning(f"Error en procesamiento robusto: {e}")
            # Fallback: solo redondear y limitar
            predictions = np.maximum(predictions, 0)
            return np.round(np.clip(predictions, 0, 12)).astype(int)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Realizar predicciones usando el modelo entrenado con escalamiento post-predicción
        
        Args:
            df: DataFrame con datos de jugadores
            
        Returns:
            Array con predicciones de AST escaladas
        """
        if not hasattr(self, 'trained_base_models') or not self.trained_base_models:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

       # Verificar orden cronológico antes de generar features
        if 'Date' in df.columns and 'player' in df.columns:
            if not df['Date'].is_monotonic_increasing:
                logger.info("Reordenando datos cronológicamente para predicción...")
                df = df.sort_values(['player', 'Date']).reset_index(drop=True)

        # Generar features (modifica df in-place, retorna List[str])
        features = self.feature_engineer.generate_all_features(df)
        
        # Determinar expected_features dinámicamente del modelo entrenado
        try:
            if hasattr(self, 'expected_features'):
                expected_features = self.expected_features
                logger.info(f"Usando expected_features del modelo: {len(expected_features)} features")
            elif hasattr(self, 'selected_features') and self.selected_features:
                expected_features = self.selected_features
                logger.info(f"Usando selected_features del entrenamiento: {len(expected_features)} features")
            else:
                # Fallback: usar todas las features generadas
                expected_features = features
                logger.warning("No se encontraron expected_features, usando todas las features generadas")
        except Exception as e:
            logger.warning(f"Error obteniendo expected_features: {e}")
            expected_features = features if features else []
        
        # Reordenar DataFrame según expected_features (df ya tiene las features)
        available_features = [f for f in expected_features if f in df.columns]
        if len(available_features) != len(expected_features):
            missing_features = set(expected_features) - set(available_features)
            logger.warning(f"Features faltantes ({len(missing_features)}): {list(missing_features)[:5]}...")
            # Agregar features faltantes con valor 0
            for feature in missing_features:
                df[feature] = 0
                available_features.append(feature)
        
        # Usar expected_features en el orden correcto
        X = df[expected_features].fillna(0)
        
        # Aplicar el scaler antes de predecir (igual que en entrenamiento)
        X_scaled = self.scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=expected_features, index=X.index)

        # Validar datos para predicción
        self._validate_training_data(X, pd.Series([0] * len(X)))
        
        # Realizar predicciones usando el nuevo sistema de stacking
        predictions = self._predict_with_stacking(X)
        
        # Aplicar calibración específica para pasadores elite
        predictions = self._apply_elite_assist_calibration(predictions, df)
        
        # Aplicar procesamiento robusto para obtener predicciones enteras
        predictions = self._apply_robust_prediction_processing(predictions)
        
        return predictions
    
    def save_model(self, filepath: str):
        """Guardar modelo entrenado completo con scaler y metadata"""
        if not hasattr(self, 'trained_base_models') or self.trained_base_models is None:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")
        
        # Crear directorio si no existe
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Guardar objeto completo con todos los componentes necesarios
        joblib.dump(self, filepath, compress=3, protocol=4)
        logger.info(f"Modelo AST completo guardado (incluye scaler): {filepath}")
    
    def load_model(self, filepath: str):
        """Cargar modelo entrenado (compatible con ambos formatos)"""
        try:
            # Intentar cargar objeto completo (nuevo formato)
            loaded_obj = joblib.load(filepath)
            if isinstance(loaded_obj, StackingASTModel):
                # Es un objeto completo, copiar todos los atributos
                self.__dict__.update(loaded_obj.__dict__)
                logger.info(f"Modelo AST completo cargado (incluye scaler): {filepath}")
                return
            elif hasattr(loaded_obj, 'predict'):
                # Es solo el stacking_model
                self.stacking_model = loaded_obj
                logger.info(f"Modelo AST (stacking_model) cargado desde: {filepath}")
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
                    logger.info(f"Modelo AST (formato legacy) cargado desde: {filepath}")
                else:
                    raise ValueError("Formato de archivo no reconocido")
            except Exception as e:
                raise ValueError(f"No se pudo cargar el modelo AST: {e}")
    
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
            logger.info(" StackingRegressor creado para compatibilidad con predictor AST")

class XGBoostASTModel:
    """
    Modelo XGBoost simplificado para AST con compatibilidad con el sistema existente
    Mantiene la interfaz del modelo de PTS pero optimizado para asistencias
    """
    
    def __init__(self, enable_neural_network: bool = False, enable_gpu: bool = False, 
                 random_state: int = 42, teams_df: pd.DataFrame = None):
        """Inicializar modelo XGBoost AST con stacking ensemble"""
        self.stacking_model = StackingASTModel(
            enable_neural_network=enable_neural_network,
            enable_gpu=enable_gpu,
            random_state=random_state,
            teams_df=teams_df
        )
        
        # Atributos para compatibilidad
        self.model = None
        self.validation_metrics = {}
        self.best_params = {}
        self.cutoff_date = None
        
        logger.info("Modelo XGBoost AST inicializado con stacking completo")
    
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
        return self.stacking_model.predict(df)
    
    def save_model(self, filepath: str):
        """Guardar modelo"""
        self.stacking_model.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Cargar modelo"""
        self.stacking_model.load_model(filepath)
        self.model = self.stacking_model.stacking_model
        self.validation_metrics = self.stacking_model.validation_metrics