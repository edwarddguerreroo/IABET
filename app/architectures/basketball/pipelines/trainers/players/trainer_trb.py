"""
Trainer Completo para Modelo XGBoost TRB
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de rebotes NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para rebotes
- Análisis de feature importance
- Validación cruzada cronológica
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Imports directos
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports del proyecto
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.trb.model_trb import XGBoostTRBModel

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoostTRBTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de rebotes NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 players_total_path: str,
                 players_quarters_path: str,
                 biometrics_path: str,
                 teams_total_path: str,
                 teams_quarters_path: str,
                 output_dir: str = "app/architectures/basketball/results/trb_model",
                 n_trials: int = 25,  # Menos trials para TRB
                 cv_folds: int = 3,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para TRB.
        
        Args:
            players_total_path: Ruta a datos de partidos de jugadores
            players_quarters_path: Ruta a datos de partidos de jugadores
            biometrics_path: Ruta a datos biométricos
            teams_total_path: Ruta a datos de equipos
            teams_quarters_path: Ruta a datos de equipos
            output_dir: Directorio de salida para resultados
            n_trials: Trials para optimización bayesiana
            cv_folds: Folds para validación cruzada
            random_state: Semilla para reproducibilidad
        """
        self.players_total_path = players_total_path
        self.players_quarters_path = players_quarters_path
        self.biometrics_path = biometrics_path
        self.teams_total_path = teams_total_path
        self.teams_quarters_path = teams_quarters_path
        self.output_dir = os.path.normpath(output_dir)
        self.random_state = random_state
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_trb_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            players_total_path, players_quarters_path, teams_total_path, teams_quarters_path, biometrics_path
        )
        
        # Inicializar modelo TRB (datos se cargarán en load_and_prepare_data)
        logger.info(" Inicializando modelo TRB...")
        
        self.model = XGBoostTRBModel(
            enable_neural_network=False,  
            enable_gpu=False,
            random_state=random_state,
            teams_df=None,  # Se asignará en load_and_prepare_data
            players_df=None,  # Se asignará en load_and_prepare_data
            players_quarters_df=None  # Se asignará en load_and_prepare_data
        )
        
        # Configurar parámetros de optimización
        self.model.stacking_model.n_trials = n_trials
        self.model.stacking_model.cv_folds = cv_folds
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
            
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        
        # Cargar datos usando el data loader (UNA SOLA VEZ)
        self.df, self.teams_df = self.data_loader.load_data(use_quarters=False)
        
        # Cargar datos por cuartos para el modelo
        players_quarters, _ = self.data_loader.load_data(use_quarters=True)
        
        # ASIGNAR DATOS AL MODELO
        self.model.stacking_model.teams_df = self.teams_df
        self.model.stacking_model.feature_engineer.teams_df = self.teams_df
        self.model.stacking_model.players_df = self.df
        self.model.stacking_model.players_quarters_df = players_quarters
        logger.info("Datos de equipos asignados al modelo para features avanzadas")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Verificar target
        if 'rebounds' not in self.df.columns:
            raise ValueError("Columna 'rebounds' no encontrada en los datos")
        
        # Estadísticas del target
        trb_stats = self.df['rebounds'].describe()
        logger.info(f"Estadísticas rebounds - Media: {trb_stats['mean']:.2f}, "
                   f"Mediana: {trb_stats['50%']:.2f}, "
                   f"Max: {trb_stats['max']:.0f}")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info("Entrenando modelo TRB con stacking ensemble")
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo TRB completado | Duración: {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO TRB")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.training_results.get('mae', 0):.4f}")
        logger.info(f"RMSE: {self.training_results.get('rmse', 0):.4f}")
        logger.info(f"R²: {self.training_results.get('r2', 0):.4f}")
        logger.info(f"Accuracy ±1reb: {self.training_results.get('accuracy_1reb', 0):.1f}%")
        logger.info(f"Accuracy ±2reb: {self.training_results.get('accuracy_2reb', 0):.1f}%")
        logger.info(f"Accuracy ±3reb: {self.training_results.get('accuracy_3reb', 0):.1f}%")
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones")
        self.predictions = self.model.predict(self.df)
        
        # Calcular métricas finales en datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) > 0:
            # Obtener índices de los datos de prueba para alinear predicciones
            test_indices = test_data.index
            
            y_true = test_data['rebounds'].values
            # Alinear predicciones con los datos de prueba usando los índices
            y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
            
            # Verificar que las dimensiones coincidan
            if len(y_true) != len(y_pred):
                logger.warning(f"Dimensiones no coinciden: y_true={len(y_true)}, y_pred={len(y_pred)}")
                # Ajustar al tamaño menor para evitar errores
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                logger.info(f"Ajustado a dimensión común: {min_len}")
            
            mae = np.mean(np.abs(y_true - y_pred))
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
            
            # Métricas específicas para rebotes
            accuracy_1reb = np.mean(np.abs(y_true - y_pred) <= 1) * 100
            accuracy_2reb = np.mean(np.abs(y_true - y_pred) <= 2) * 100
            accuracy_3reb = np.mean(np.abs(y_true - y_pred) <= 3) * 100
            
            logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            logger.info(f"Accuracy ±1reb: {accuracy_1reb:.1f}%, ±2reb: {accuracy_2reb:.1f}%, ±3reb: {accuracy_3reb:.1f}%")
            
            self.training_results.update({
                'final_mae': mae,
                'final_rmse': rmse,
                'final_r2': r2,
                'final_accuracy_1reb': accuracy_1reb,
                'final_accuracy_2reb': accuracy_2reb,
                'final_accuracy_3reb': accuracy_3reb
            })
        
        return self.training_results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA TRB Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo (esquina superior izquierda)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance (esquina superior derecha)
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución del target (esquina superior derecha)
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_target_distribution_compact(ax3)
        
        # 4. Predicciones vs Reales (segunda fila, izquierda)
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_predictions_vs_actual_compact(ax4)
        
        # 5. Residuos (segunda fila, derecha)
        ax5 = fig.add_subplot(gs[1, 2:4])
        self._plot_residuals_compact(ax5)
        
        # 6. Validación cruzada (tercera fila, izquierda)
        ax6 = fig.add_subplot(gs[2, 0:2])
        self._plot_cv_results_compact(ax6)
        
        # 7. Análisis por rangos de rebotes (tercera fila, derecha)
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_rebounds_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (cuarta fila, izquierda)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top reboteadores predicciones (cuarta fila, derecha)
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_rebounders_analysis_compact(ax9)
        
        # Guardar como PNG con ruta normalizada
        png_path = os.path.normpath(os.path.join(self.output_dir, 'model_dashboard_complete.png'))
        
        try:
            plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Dashboard completo guardado en: {png_path}")
        except Exception as e:
            logger.error(f"Error guardando PNG: {e}")
            # Intentar con ruta absoluta
            abs_png_path = os.path.abspath(png_path)
            logger.info(f"Intentando con ruta absoluta: {abs_png_path}")
            plt.savefig(abs_png_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Dashboard guardado exitosamente en: {abs_png_path}")
        finally:
            plt.close()
    
    def _plot_model_metrics_summary(self, ax):
        """Resumen de métricas principales del modelo."""
        ax.axis('off')
        
        # Obtener métricas
        mae = self.training_results.get('mae', 0)
        rmse = self.training_results.get('rmse', 0)
        r2 = self.training_results.get('r2', 0)
        accuracy_1reb = self.training_results.get('accuracy_1reb', 0)
        accuracy_2reb = self.training_results.get('accuracy_2reb', 0)
        accuracy_3reb = self.training_results.get('accuracy_3reb', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO TRB

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY REBOTES:
±1 reb: {accuracy_1reb:.1f}%
±2 reb: {accuracy_2reb:.1f}%
±3 reb: {accuracy_3reb:.1f}%

MODELOS BASE:
• XGBoost
• LightGBM  
• CatBoost
• Gradient Boosting
• Ridge
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        if not hasattr(self.model.stacking_model, 'feature_importance') or not self.model.stacking_model.feature_importance:
            ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
            return
        
        # Obtener top 15 features
        importance_dict = self.model.stacking_model.feature_importance
        top_features = dict(list(importance_dict.items())[:15])
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Crear gráfico horizontal
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='skyblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title()[:20] for f in features], fontsize=8)
        ax.set_xlabel('Importancia')
        ax.set_title('Top 15 Features Más Importantes', fontweight='bold')
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, importances)):
            ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=7)
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target TRB."""
        trb_values = self.df['rebounds']
        
        # Histograma
        ax.hist(trb_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        
        # Estadísticas
        mean_trb = trb_values.mean()
        median_trb = trb_values.median()
        
        ax.axvline(mean_trb, color='red', linestyle='--', label=f'Media: {mean_trb:.1f}')
        ax.axvline(median_trb, color='blue', linestyle='--', label=f'Mediana: {median_trb:.1f}')
        
        ax.set_xlabel('Rebotes Totales (TRB)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de TRB', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Gráfico compacto de predicciones vs valores reales."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['rebounds'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='blue')
        
        # Línea perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        ax.set_xlabel('TRB Real')
        ax.set_ylabel('TRB Predicho')
        ax.set_title('Predicciones vs Valores Reales', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Agregar R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_residuals_compact(self, ax):
        """Gráfico compacto de residuos."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Residuos', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Residuos', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['rebounds'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='green')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('TRB Predicho')
        ax.set_ylabel('Residuos (Real - Predicho)')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Agregar estadísticas de residuos
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'MAE = {mae:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_cv_results_compact(self, ax):
        """Gráfico compacto de resultados de validación cruzada."""
        cv_scores = self.model.stacking_model.cv_scores
        
        if not cv_scores or 'cv_scores' not in cv_scores:
            ax.text(0.5, 0.5, 'Resultados CV\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validación Cruzada', fontweight='bold')
            return
        
        # Extraer MAE de cada fold
        fold_scores = cv_scores['cv_scores']
        mae_scores = [fold['mae'] for fold in fold_scores]
        r2_scores = [fold['r2'] for fold in fold_scores]
        
        folds = range(1, len(mae_scores) + 1)
        
        # Crear gráfico de barras
        x = np.arange(len(folds))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8, color='orange')
        
        # Crear segundo eje para R²
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, r2_scores, width, label='R²', alpha=0.8, color='purple')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel('MAE', color='orange')
        ax2.set_ylabel('R²', color='purple')
        ax.set_title('Validación Cruzada por Fold', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {i}' for i in folds])
        
        # Agregar valores en las barras
        for bar, val in zip(bars1, mae_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, r2_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Agregar promedios
        avg_mae = cv_scores.get('mae_mean', 0)
        avg_r2 = cv_scores.get('r2_mean', 0)
        ax.text(0.02, 0.98, f'Promedio MAE: {avg_mae:.3f}\nPromedio R²: {avg_r2:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(alpha=0.3)
    
    def _plot_rebounds_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de rebotes."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Rebotes', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Rebotes', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['rebounds'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Definir rangos de rebotes
        ranges = [
            (0, 3, 'Bajo (0-3)'),
            (4, 7, 'Medio (4-7)'),
            (8, 12, 'Alto (8-12)'),
            (13, 25, 'Elite (13+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_reb, max_reb, name in ranges:
            mask = (y_true >= min_reb) & (y_true <= max_reb)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Rebotes', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae, count in zip(bars, range_maes, range_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{mae:.3f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['rebounds'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        test_data = test_data.iloc[:min_len]
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Agrupar por mes
        test_data_copy = test_data.copy()
        test_data_copy['month'] = pd.to_datetime(test_data_copy['Date']).dt.to_period('M')
        test_data_copy['y_true'] = y_true
        test_data_copy['y_pred'] = y_pred
        test_data_copy['abs_error'] = np.abs(y_true - y_pred)
        
        monthly_stats = test_data_copy.groupby('month').agg({
            'abs_error': 'mean',
            'y_true': 'count'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            mae_by_month = monthly_stats['abs_error']
            
            bars = ax.bar(months, mae_by_month, alpha=0.8, color='lightcoral')
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Mes', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae in zip(bars, mae_by_month):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                       f'{mae:.3f}', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_rebounders_analysis_compact(self, ax):
        """Análisis compacto de top reboteadores."""
        # Obtener top reboteadores por promedio
        player_stats = self.df.groupby('player').agg({
            'rebounds': ['mean', 'count']
        }).reset_index()
        
        player_stats.columns = ['player', 'mean', 'count']
        
        # Filtrar jugadores con al menos 10 juegos
        player_stats = player_stats[player_stats['count'] >= 10]
        
        # Top 10 reboteadores
        top_rebounders = player_stats.nlargest(10, 'mean')
        
        if len(top_rebounders) > 0:
            players = [p[:15] + '' if len(p) > 15 else p for p in top_rebounders['player']]
            means = top_rebounders['mean']
            
            bars = ax.barh(players, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio TRB')
            ax.set_title('Top 10 Reboteadores', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
    
    def _generate_detailed_training_report(self):
        """Genera un reporte SÚPER DETALLADO del entrenamiento con toda la información posible."""
        
        # Información básica del modelo
        model_info = {
            'model_type': 'XGBoost TRB Stacking Ensemble',
            'architecture': 'Out-of-Fold Stacking with Meta-Learner',
            'base_models': ['XGBoost', 'LightGBM', 'CatBoost', 'Ridge'],
            'meta_learner': 'Ridge (optimized)',
            'feature_selection': 'Intelligent (50 features)',
            'cross_validation': 'TimeSeriesSplit (5 folds)',
            'scaling': 'StandardScaler',
            'timestamp': datetime.now().isoformat()
        }
        
        # Datos del dataset
        dataset_info = {
            'total_records': len(self.df),
            'unique_players': self.df['player'].nunique(),
            'unique_teams': self.df['Team'].nunique(),
            'date_range': {
                'start': self.df['Date'].min().isoformat(),
                'end': self.df['Date'].max().isoformat()
            },
            'target_statistics': {
                'mean': float(self.df['rebounds'].mean()),
                'median': float(self.df['rebounds'].median()),
                'std': float(self.df['rebounds'].std()),
                'min': float(self.df['rebounds'].min()),
                'max': float(self.df['rebounds'].max()),
                'q25': float(self.df['rebounds'].quantile(0.25)),
                'q75': float(self.df['rebounds'].quantile(0.75))
            }
        }
        
        # División temporal
        train_data, test_data = self.model._temporal_split(self.df)
        temporal_split_info = {
            'train_size': len(train_data),
            'test_size': len(test_data),
            'train_percentage': len(train_data) / len(self.df) * 100,
            'test_percentage': len(test_data) / len(self.df) * 100,
            'cutoff_date': train_data['Date'].max().isoformat()
        }
        
        # Features generadas
        features_info = {
            'total_features_generated': len(self.model.stacking_model.selected_features) if hasattr(self.model.stacking_model, 'selected_features') else 0,
            'features_used': list(self.model.stacking_model.selected_features) if hasattr(self.model.stacking_model, 'selected_features') else [],
            'feature_importance': self.model.stacking_model.feature_importance if hasattr(self.model.stacking_model, 'feature_importance') else {}
        }
        
        # Métricas de entrenamiento
        training_metrics = {
            'final_metrics': self.training_results,
            'base_models_performance': {},
            'meta_learner_performance': {},
            'cross_validation_scores': self.model.cv_scores if hasattr(self.model, 'cv_scores') else {}
        }
        
        # Obtener métricas de modelos base si están disponibles
        if hasattr(self.model.stacking_model, 'trained_base_models'):
            for name, model in self.model.stacking_model.trained_base_models.items():
                try:
                    # Predecir en datos de prueba para obtener métricas
                    X_test = test_data[self.model.stacking_model.selected_features].fillna(0)
                    y_test = test_data['rebounds']
                    
                    # Aplicar escalado si está disponible
                    if hasattr(self.model.stacking_model, 'scaler') and self.model.stacking_model.scaler:
                        X_test_scaled = self.model.stacking_model.scaler.transform(X_test)
                        X_test = pd.DataFrame(X_test_scaled, columns=self.model.stacking_model.selected_features, index=X_test.index)
                    
                    predictions = model.predict(X_test)
                    
                    training_metrics['base_models_performance'][name] = {
                        'mae': float(mean_absolute_error(y_test, predictions)),
                        'rmse': float(np.sqrt(mean_squared_error(y_test, predictions))),
                        'r2': float(r2_score(y_test, predictions)),
                        'mean_prediction': float(np.mean(predictions)),
                        'std_prediction': float(np.std(predictions)),
                        'min_prediction': float(np.min(predictions)),
                        'max_prediction': float(np.max(predictions))
                    }
                except Exception as e:
                    training_metrics['base_models_performance'][name] = {'error': str(e)}
        
        # Métricas del meta-learner
        if hasattr(self.model.stacking_model, 'meta_learner') and self.model.stacking_model.meta_learner:
            try:
                X_test = test_data[self.model.stacking_model.selected_features].fillna(0)
                y_test = test_data['rebounds']
                
                if hasattr(self.model.stacking_model, 'scaler') and self.model.stacking_model.scaler:
                    X_test_scaled = self.model.stacking_model.scaler.transform(X_test)
                    X_test = pd.DataFrame(X_test_scaled, columns=self.model.stacking_model.selected_features, index=X_test.index)
                
                # Obtener predicciones de modelos base
                base_predictions = np.zeros((len(X_test), len(self.model.stacking_model.trained_base_models)))
                for i, (name, model) in enumerate(self.model.stacking_model.trained_base_models.items()):
                    base_predictions[:, i] = model.predict(X_test)
                
                meta_predictions = self.model.stacking_model.meta_learner.predict(base_predictions)
                
                training_metrics['meta_learner_performance'] = {
                    'mae': float(mean_absolute_error(y_test, meta_predictions)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, meta_predictions))),
                    'r2': float(r2_score(y_test, meta_predictions)),
                    'mean_prediction': float(np.mean(meta_predictions)),
                    'std_prediction': float(np.std(meta_predictions)),
                    'min_prediction': float(np.min(meta_predictions)),
                    'max_prediction': float(np.max(meta_predictions))
                }
            except Exception as e:
                training_metrics['meta_learner_performance'] = {'error': str(e)}
        
        # Análisis de predicciones
        if self.predictions is not None:
            prediction_analysis = {
                'total_predictions': len(self.predictions),
                'prediction_statistics': {
                    'mean': float(np.mean(self.predictions)),
                    'median': float(np.median(self.predictions)),
                    'std': float(np.std(self.predictions)),
                    'min': float(np.min(self.predictions)),
                    'max': float(np.max(self.predictions)),
                    'q25': float(np.percentile(self.predictions, 25)),
                    'q75': float(np.percentile(self.predictions, 75))
                },
                'accuracy_analysis': {
                    'within_1_rebound': float(np.mean(np.abs(self.predictions - self.df['rebounds']) <= 1) * 100),
                    'within_2_rebounds': float(np.mean(np.abs(self.predictions - self.df['rebounds']) <= 2) * 100),
                    'within_3_rebounds': float(np.mean(np.abs(self.predictions - self.df['rebounds']) <= 3) * 100)
                },
                'error_analysis': {
                    'mean_absolute_error': float(np.mean(np.abs(self.predictions - self.df['rebounds']))),
                    'mean_squared_error': float(np.mean((self.predictions - self.df['rebounds'])**2)),
                    'root_mean_squared_error': float(np.sqrt(np.mean((self.predictions - self.df['rebounds'])**2))),
                    'mean_error': float(np.mean(self.predictions - self.df['rebounds'])),
                    'error_std': float(np.std(self.predictions - self.df['rebounds']))
                }
            }
        else:
            prediction_analysis = {'error': 'No predictions available'}
        
        # Configuración de hiperparámetros
        hyperparameters = {
            'base_models': {},
            'meta_learner': {},
            'cross_validation': {
                'n_splits': 5,
                'test_size': 0.2,
                'random_state': 42
            },
            'scaling': {
                'method': 'StandardScaler',
                'fit_on_train_only': True
            }
        }
        
        # Obtener hiperparámetros de modelos base si están disponibles
        if hasattr(self.model.stacking_model, 'base_models'):
            for name, model_config in self.model.stacking_model.base_models.items():
                try:
                    model = model_config['model']
                    hyperparameters['base_models'][name] = model.get_params()
                except:
                    hyperparameters['base_models'][name] = {'error': 'Could not extract parameters'}
        
        # Hiperparámetros del meta-learner
        if hasattr(self.model.stacking_model, 'meta_learner') and self.model.stacking_model.meta_learner:
            try:
                hyperparameters['meta_learner'] = self.model.stacking_model.meta_learner.get_params()
            except:
                hyperparameters['meta_learner'] = {'error': 'Could not extract parameters'}
        
        # Reporte final súper detallado
        detailed_report = {
            'model_information': model_info,
            'dataset_information': dataset_info,
            'temporal_split': temporal_split_info,
            'features_information': features_info,
            'training_metrics': training_metrics,
            'prediction_analysis': prediction_analysis,
            'hyperparameters': hyperparameters,
            'model_summary': self.model.stacking_model.get_model_summary() if hasattr(self.model.stacking_model, 'get_model_summary') else {},
            'training_results': self.training_results,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        return detailed_report

    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/
        model_path = os.path.normpath(os.path.join('app/architectures/basketball/.joblib', 'trb_model.joblib'))
        os.makedirs('app/architectures/basketball/.joblib', exist_ok=True)
        self.model.save_model(model_path)
        
        # Guardar reporte SÚPER DETALLADO
        report = self._generate_detailed_training_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['player', 'Date', 'Team', 'rebounds']].copy()
            predictions_df['rebounds_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['rebounds']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance COMPLETA (todas las features como en PTS)
        if hasattr(self.model.stacking_model, 'feature_importance') and self.model.stacking_model.feature_importance:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v} 
                for k, v in self.model.stacking_model.feature_importance.items()
            ]).sort_values('importance', ascending=False)
            
            # Agregar información adicional sobre las features (como en PTS)
            total_features = len(importance_df)
            importance_df['rank'] = range(1, total_features + 1)
            importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(4)
            importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum().round(4)
            
            importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"✅ Feature importance exportada: {total_features} features completas en {importance_path}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'app/architectures/basketball/.joblib/trb_model.joblib',
            'dashboard_image': 'model_dashboard_complete.png',
            'training_report': 'training_report.json',
            'predictions': 'predictions.csv',
            'feature_importance': 'feature_importance.csv',
            'output_directory': os.path.normpath(self.output_dir),
            'generation_timestamp': datetime.now().isoformat()
        }
        
        summary_path = os.path.normpath(os.path.join(self.output_dir, 'files_summary.json'))
        with open(summary_path, 'w') as f:
            json.dump(files_summary, f, indent=2)
        
        logger.info(f"Resultados guardados en: {os.path.normpath(self.output_dir)}")
        logger.info("Archivos generados:")
        logger.info(f"  • Modelo: {model_path}")
        logger.info(f"  • Dashboard PNG: {os.path.normpath(os.path.join(self.output_dir, 'model_dashboard_complete.png'))}")
        logger.info(f"  • Reporte: {report_path}")
        if self.predictions is not None:
            logger.info(f"  • Predicciones: {predictions_path}")
        logger.info(f"  • Resumen: {summary_path}")
    
    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict: Resultados completos del entrenamiento
        """
        
        try:
            # 1. Cargar datos
            self.load_and_prepare_data()
            
            # 2. Entrenar modelo
            results = self.train_model()
            
            # 3. Generar visualizaciones
            self.generate_all_visualizations()
            
            # 4. Guardar resultados
            self.save_results()
            
            logger.info("Pipeline ejecutado exitosamente!")
            return results
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo de TRB.
    """
    # Configurar logging ultra-silencioso
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configurar logging para el trainer principal
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    # Configurar logging general para mostrar todos los mensajes importantes
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True  # Forzar reconfiguración
    )
    
    # Silenciar librerías externas
    logging.getLogger('sklearn').setLevel(logging.ERROR)
    logging.getLogger('xgboost').setLevel(logging.ERROR)
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    logging.getLogger('catboost').setLevel(logging.ERROR)
    logging.getLogger('optuna').setLevel(logging.ERROR)
    
    # Rutas de datos (ajustar según tu configuración)
    players_total_path = "app/architectures/basketball/data/players_total.csv"
    players_quarters_path = "app/architectures/basketball/data/players_quarters.csv"
    teams_total_path = "app/architectures/basketball/data/teams_total.csv"
    teams_quarters_path = "app/architectures/basketball/data/teams_quarters.csv"
    biometrics_path = "app/architectures/basketball/data/biometrics.csv"
    
    # Crear y ejecutar trainer
    trainer = XGBoostTRBTrainer(
        players_total_path=players_total_path,
        players_quarters_path=players_quarters_path,
        teams_total_path=teams_total_path,
        teams_quarters_path=teams_quarters_path,
        biometrics_path=biometrics_path,
        output_dir="app/architectures/basketball/results/trb_model",
        n_trials=25,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento TRB Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 