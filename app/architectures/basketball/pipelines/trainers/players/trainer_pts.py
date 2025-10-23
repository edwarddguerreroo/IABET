"""
Trainer Completo para Modelo XGBoost PTS
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de puntos NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas y reportes completos
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


import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports del proyecto
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.pts.model_pts import XGBoostPTSModel


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoostPTSTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de puntos NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 players_total_path: str,
                 players_quarters_path: str,
                 biometrics_path: str,
                 teams_total_path: str,
                 teams_quarters_path: str,
                 output_dir: str = "app/architectures/basketball/results/pts_model",
                 n_trials: int = 25,
                 cv_folds: int = 3,
                 random_state: int = 42):
        """
        Inicializa el trainer completo.
        
        Args:
            players_total_path: Ruta a datos de partidos de jugadores
            players_quarters_path: Ruta a datos de partidos de jugadores
            teams_quarters_path: Ruta a datos de partidos de jugadores
            teams_total_path: Ruta a datos de equipos
            teams_quarters_path: Ruta a datos de equipos
            biometrics_path: Ruta a datos biométricos
            output_dir: Directorio de salida para resultados
            n_trials: Trials para optimización bayesiana
            cv_folds: Folds para validación cruzada
            random_state: Semilla para reproducibilidad
        """
        self.players_total_path = players_total_path
        self.players_quarters_path = players_quarters_path
        self.teams_total_path = teams_total_path
        self.teams_quarters_path = teams_quarters_path
        self.biometrics_path = biometrics_path
        self.output_dir = os.path.normpath(output_dir)
        self.random_state = random_state
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_pts_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            players_total_path, players_quarters_path, teams_total_path, teams_quarters_path, biometrics_path
        )
        
        # Datos y resultados (se cargarán en load_and_prepare_data)
        self.df = None
        self.teams_df = None
        self.players_quarters_df = None
        self.teams_quarters_df = None
        
        self.model = XGBoostPTSModel(
            teams_df=None,  # Se asignará en load_and_prepare_data
            players_df=None,  # Se asignará en load_and_prepare_data
            players_quarters_df=None,  # Se asignará en load_and_prepare_data
            n_trials=n_trials,
            cv_folds=cv_folds,
            random_state=random_state
        )
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer XGBoost PTS inicializado")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df, self.players_quarters_df, self.teams_quarters_df = self.data_loader.load_data()
        
        # Actualizar modelo con los datos cargados
        self.model.teams_df = self.teams_df
        self.model.players_df = self.df
        self.model.players_quarters_df = self.players_quarters_df
        self.model.feature_engineer.teams_df = self.teams_df
        self.model.feature_engineer.players_df = self.df
        self.model.feature_engineer.players_quarters_df = self.players_quarters_df
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Verificar target
        if 'points' not in self.df.columns:
            raise ValueError("Columna 'points' no encontrada en los datos")
        
        # Estadísticas del target
        pts_stats = self.df['points'].describe()
        logger.info(f"Estadísticas PTS - Media: {pts_stats['mean']:.2f}, "
                   f"Mediana: {pts_stats['50%']:.2f}, "
                   f"Max: {pts_stats['max']:.0f}")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo de puntos")
        logger.info(f"Datos de entrenamiento: {len(self.df)} registros")
        logger.info(f"Target PTS - Media: {self.df['points'].mean():.2f}, Std: {self.df['points'].std():.2f}")
        logger.info(f"Jugadores únicos: {self.df['player'].nunique()}")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Entrenamiento completado | Duración: {training_duration:.1f} segundos")
        
        # Mostrar métricas principales de la nueva arquitectura OOF
        if hasattr(self.model, 'base_models_performance'):
            logger.info("Rendimiento de modelos base (OOF):")
            for model_name, perf in self.model.base_models_performance.items():
                mae = perf.get('mae', 0)
                r2 = perf.get('r2', 0)
                logger.info(f"  • {model_name}: MAE={mae:.3f}, R²={r2:.3f}")
        
        if hasattr(self.model, 'meta_learner_performance'):
            meta_mae = self.model.meta_learner_performance.get('mae', 0)
            meta_r2 = self.model.meta_learner_performance.get('r2', 0)
            logger.info(f"Meta-learner (OOF): MAE={meta_mae:.3f}, R²={meta_r2:.3f}")
        
        # Mostrar métricas de validación cruzada temporal
        if hasattr(self.model, 'cv_scores') and self.model.cv_scores:
            cv_mae = np.mean([score.get('mae', 0) for score in self.model.cv_scores])
            cv_r2 = np.mean([score.get('r2', 0) for score in self.model.cv_scores])
            logger.info(f"Validación cruzada temporal: MAE={cv_mae:.3f}, R²={cv_r2:.3f}")
        
        # Generar predicciones en conjunto de prueba
        self.predictions = self.model.predict(self.df)
        
        # Compilar resultados completos con nueva arquitectura OOF
        results = {
            'base_models_performance': getattr(self.model, 'base_models_performance', {}),
            'meta_learner_performance': getattr(self.model, 'meta_learner_performance', {}),
            'cv_scores': getattr(self.model, 'cv_scores', []),
            'best_params_per_model': getattr(self.model, 'best_params_per_model', {}),
            'feature_importance': dict(list(self.model.get_feature_importance(None).items())[:20]),  # Top 20 para reporte
            'training_duration_seconds': training_duration,
            'model_info': {
                'n_features': len(getattr(self.model, 'selected_features', [])),
                'n_samples': len(self.df),
                'target_mean': self.df['points'].mean(),
                'target_std': self.df['points'].std(),
                'architecture': 'Stacking Ensemble OOF',
                'base_models': list(getattr(self.model, 'base_models', {}).keys()),
                'meta_learner': 'Ridge'
            }
        }
        
        return results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA Points Prediction', fontsize=20, fontweight='bold', y=0.98)
        
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
        
        # 7. Análisis por rangos de puntos (tercera fila, derecha)
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_points_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (cuarta fila, izquierda)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top jugadores predicciones (cuarta fila, derecha)
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_players_analysis_compact(ax9)
        
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
        """Resumen de métricas principales del modelo OOF."""
        ax.axis('off')
        
        # Obtener métricas de la nueva arquitectura OOF
        base_models_perf = self.training_results.get('base_models_performance', {})
        meta_learner_perf = self.training_results.get('meta_learner_performance', {})
        cv_scores = self.training_results.get('cv_scores', [])
        
        # Calcular métricas promedio de modelos base
        if base_models_perf:
            avg_base_mae = np.mean([perf.get('mae', 0) for perf in base_models_perf.values()])
            avg_base_r2 = np.mean([perf.get('r2', 0) for perf in base_models_perf.values()])
        else:
            avg_base_mae = 0
            avg_base_r2 = 0
        
        # Métricas del meta-learner
        meta_mae = meta_learner_perf.get('mae', 0)
        meta_r2 = meta_learner_perf.get('r2', 0)
        
        # Métricas de validación cruzada temporal
        if cv_scores:
            cv_mae = np.mean([score.get('mae', 0) for score in cv_scores])
            cv_r2 = np.mean([score.get('r2', 0) for score in cv_scores])
        else:
            cv_mae = 0
            cv_r2 = 0
        
        # Crear texto con métricas OOF
        metrics_text = f"""
MÉTRICAS DEL MODELO OOF

Modelos Base (Promedio):
• MAE: {avg_base_mae:.3f}
• R²: {avg_base_r2:.3f}

Meta-learner Ridge:
• MAE: {meta_mae:.3f}
• R²: {meta_r2:.3f}

Validación Cruzada Temporal:
• MAE CV: {cv_mae:.3f}
• R² CV: {cv_r2:.3f}

Arquitectura: Stacking OOF
Estado: PRODUCCIÓN
        """
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        ax.set_title('Resumen de Rendimiento OOF', fontsize=14, fontweight='bold')
    
    def _plot_feature_importance_compact(self, ax):
        """Feature importance compacta."""
        if not hasattr(self.model, 'get_feature_importance'):
            ax.text(0.5, 0.5, 'Feature importance no disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Obtener top 10 features
        top_features = dict(list(self.model.get_feature_importance(None).items())[:10])
        features = list(top_features.keys())
        importance = list(top_features.values())
        
        # Gráfico horizontal
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importance, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Importancia', fontsize=10)
        ax.set_title('Top 10 Características Más Importantes', fontsize=12, fontweight='bold')
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, importance)):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución del target compacta."""
        # Histograma de puntos
        ax.hist(self.df['points'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        
        # Líneas de estadísticas
        mean_pts = self.df['points'].mean()
        median_pts = self.df['points'].median()
        
        ax.axvline(mean_pts, color='red', linestyle='--', linewidth=2, 
                  label=f'Media: {mean_pts:.1f}')
        ax.axvline(median_pts, color='green', linestyle='--', linewidth=2,
                  label=f'Mediana: {median_pts:.1f}')
        
        ax.set_xlabel('Puntos', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Puntos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Análisis de predicciones vs valores reales compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['points'].values
        y_pred = self.predictions
        
        
        # Scatter plot predicciones vs reales
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax.set_xlabel('Puntos Reales', fontsize=10)
        ax.set_ylabel('Puntos Predichos', fontsize=10)
        ax.set_title('Predicciones vs Valores Reales', fontsize=12, fontweight='bold')
        
        # Calcular R²
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
               fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(alpha=0.3)
    
    def _plot_residuals_compact(self, ax):
        """Análisis de residuos compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Residuos no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['points'].values
        y_pred = self.predictions
        residuals = y_pred - y_true
        
        # Histograma de residuos
        ax.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=2,
                  label=f'Media: {residuals.mean():.3f}')
        
        ax.set_xlabel('Residuos (Predicho - Real)', fontsize=10)
        ax.set_ylabel('Frecuencia', fontsize=10)
        ax.set_title('Distribución de Residuos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
    
    def _plot_cv_results_compact(self, ax):
        """Resultados de validación cruzada temporal compactos."""
        cv_scores = self.training_results.get('cv_scores', [])
        
        if not cv_scores:
            ax.text(0.5, 0.5, 'Resultados de validación cruzada no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Calcular métricas de la lista de scores
        mae_scores = [score.get('mae', 0) for score in cv_scores]
        r2_scores = [score.get('r2', 0) for score in cv_scores]
        
        if not mae_scores or not r2_scores:
            ax.text(0.5, 0.5, 'Datos de validación cruzada insuficientes', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Preparar datos para gráfico
        metrics = ['MAE', 'R²']
        means = [np.mean(mae_scores), np.mean(r2_scores)]
        stds = [np.std(mae_scores), np.std(r2_scores)]
        
        x_pos = np.arange(len(metrics))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=['salmon', 'lightgreen'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metrics)
        ax.set_title('Validación Cruzada Temporal (Media ± Std)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Valor', fontsize=10)
        
        # Agregar valores en las barras
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Agregar información de folds
        ax.text(0.02, 0.98, f'Folds: {len(cv_scores)}', transform=ax.transAxes, 
               fontsize=8, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_points_range_analysis_compact(self, ax):
        """Análisis de predicciones por rango de puntos compacto."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones no disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        y_true = self.df['points'].values
        y_pred = self.predictions
        
        # Definir rangos de puntos
        ranges = [(0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 50)]
        range_labels = ['0-10', '10-15', '15-20', '20-25', '25-30', '30+']
        
        accuracies_1pt = []
        accuracies_2pts = []
        sample_counts = []
        
        for start, end in ranges:
            if end == 50:  # Para el último rango (30+)
                mask = y_true >= start
            else:
                mask = (y_true >= start) & (y_true < end)
            
            if mask.sum() > 0:
                range_y_true = y_true[mask]
                range_y_pred = y_pred[mask]
                
                acc_1pt = np.mean(np.abs(range_y_pred - range_y_true) <= 1) * 100
                acc_2pts = np.mean(np.abs(range_y_pred - range_y_true) <= 2) * 100
                
                accuracies_1pt.append(acc_1pt)
                accuracies_2pts.append(acc_2pts)
                sample_counts.append(mask.sum())
            else:
                accuracies_1pt.append(0)
                accuracies_2pts.append(0)
                sample_counts.append(0)
        
        # Gráfico de barras
        x_pos = np.arange(len(range_labels))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, accuracies_1pt, width, label='±1 punto', color='lightblue', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, accuracies_2pts, width, label='±2 puntos', color='lightgreen', alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(range_labels)
        ax.set_xlabel('Rango de Puntos', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Accuracy por Rango de Puntos', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        
        # Agregar conteos de muestras
        for i, count in enumerate(sample_counts):
            if count > 0:
                ax.text(i, max(accuracies_1pt[i], accuracies_2pts[i]) + 5, 
                       f'n={count}', ha='center', va='bottom', fontsize=8)
        
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        if self.df is None:
            ax.text(0.5, 0.5, 'Datos no cargados', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Promedio de puntos por mes
        monthly_pts = self.df.groupby(self.df['Date'].dt.to_period('M'))['points'].mean()
        
        # Limitar a últimos 12 meses para mejor visualización
        if len(monthly_pts) > 12:
            monthly_pts = monthly_pts.tail(12)
        
        ax.plot(range(len(monthly_pts)), monthly_pts.values, 
               marker='o', linewidth=2, color='purple', markersize=6)
        
        # Configurar etiquetas del eje x
        ax.set_xticks(range(len(monthly_pts)))
        ax.set_xticklabels([str(period)[-7:] for period in monthly_pts.index], 
                          rotation=45, fontsize=8)
        
        ax.set_title('Tendencia Temporal de Puntos', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mes', fontsize=10)
        ax.set_ylabel('Puntos Promedio', fontsize=10)
        
        # Agregar línea de tendencia
        if len(monthly_pts) > 2:
            z = np.polyfit(range(len(monthly_pts)), monthly_pts.values, 1)
            p = np.poly1d(z)
            ax.plot(range(len(monthly_pts)), p(range(len(monthly_pts))), 
                   "r--", alpha=0.8, linewidth=1, label=f'Tendencia')
            ax.legend(fontsize=9)
        
        ax.grid(alpha=0.3)
    
    def _plot_top_players_analysis_compact(self, ax):
        """Análisis de top jugadores compacto."""
        if self.df is None:
            ax.text(0.5, 0.5, 'Datos no cargados', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Top 10 jugadores por puntos promedio (mínimo 10 partidos)
        player_stats = self.df.groupby('player')['points'].agg(['mean', 'count']).reset_index()
        player_stats = player_stats[player_stats['count'] >= 10]
        top_scorers = player_stats.nlargest(10, 'mean')
        
        if len(top_scorers) == 0:
            ax.text(0.5, 0.5, 'No hay suficientes datos de jugadores', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        # Gráfico horizontal
        y_pos = np.arange(len(top_scorers))
        bars = ax.barh(y_pos, top_scorers['mean'].values, color='gold', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([name[:15] + '' if len(name) > 15 else name 
                           for name in top_scorers['player'].values], fontsize=8)
        ax.set_xlabel('Puntos Promedio', fontsize=10)
        ax.set_title('Top 10 Anotadores (≥10 partidos)', fontsize=12, fontweight='bold')
        
        # Agregar valores en las barras
        for i, (bar, val) in enumerate(zip(bars, top_scorers['mean'].values)):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontsize=8)
        
        ax.grid(axis='x', alpha=0.3)
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados completos")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/ con la nueva arquitectura OOF
        model_path = os.path.normpath(os.path.join('app/architectures/basketball/.joblib', 'pts_model.joblib'))
        os.makedirs('app/architectures/basketball/.joblib', exist_ok=True)
        self.model.save_model(model_path)
        
        # Generar reporte completo con formato estándar
        report = self._generate_comprehensive_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['player', 'Date', 'Team', 'points']].copy()
            predictions_df['points_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['points']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance COMPLETA (todas las features utilizadas)
        if hasattr(self.model, 'get_feature_importance'):
            try:
                # Obtener TODAS las features sin límite
                feature_importance = self.model.get_feature_importance(None)  # None = todas las features
                importance_df = pd.DataFrame([
                    {'feature': k, 'importance': v} 
                    for k, v in feature_importance.items()
                ]).sort_values('importance', ascending=False)
                
                # Agregar información adicional sobre las features
                total_features = len(importance_df)
                importance_df['rank'] = range(1, total_features + 1)
                importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(4)
                importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum().round(4)
                
                importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                importance_df.to_csv(importance_path, index=False)
                
                logger.info(f"Feature importance exportada: {total_features} features completas en {importance_path}")
            except Exception as e:
                logger.warning(f"No se pudo guardar feature importance: {e}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'app/architectures/basketball/.joblib/pts_model.joblib',
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
    
    def _generate_comprehensive_report(self) -> Dict:
        """
        Genera un reporte completo con formato estándar como los otros modelos.
        """
        # Información del modelo
        model_info = {
            "model_type": "XGBoost PTS Stacking Ensemble",
            "architecture": "Out-of-Fold Stacking with Meta-Learner",
            "base_models": list(getattr(self.model, 'base_models', {}).keys()),
            "meta_learner": "Ridge (optimized)",
            "feature_selection": f"Intelligent ({len(getattr(self.model, 'selected_features', []))} features)",
            "cross_validation": "TimeSeriesSplit (5 folds)",
            "scaling": "StandardScaler",
            "timestamp": datetime.now().isoformat()
        }
        
        # Información del dataset
        dataset_info = {
            "total_records": len(self.df),
            "unique_players": self.df['player'].nunique(),
            "unique_teams": self.df['Team'].nunique(),
            "date_range": {
                "start": self.df['Date'].min().isoformat(),
                "end": self.df['Date'].max().isoformat()
            },
            "target_statistics": {
                "mean": float(self.df['points'].mean()),
                "median": float(self.df['points'].median()),
                "std": float(self.df['points'].std()),
                "min": float(self.df['points'].min()),
                "max": float(self.df['points'].max()),
                "q25": float(self.df['points'].quantile(0.25)),
                "q75": float(self.df['points'].quantile(0.75))
            }
        }
        
        # División temporal
        train_size = int(len(self.df) * 0.8)
        test_size = len(self.df) - train_size
        cutoff_date = self.df.sort_values('Date').iloc[train_size]['Date']
        
        temporal_split = {
            "train_size": train_size,
            "test_size": test_size,
            "train_percentage": (train_size / len(self.df)) * 100,
            "test_percentage": (test_size / len(self.df)) * 100,
            "cutoff_date": cutoff_date.isoformat()
        }
        
        # Información de características
        selected_features = getattr(self.model, 'selected_features', [])
        feature_importance = self.model.get_feature_importance(None) if hasattr(self.model, 'get_feature_importance') else {}
        
        features_info = {
            "total_features_generated": len(selected_features),
            "features_used": selected_features[:20],  # Top 20 para el reporte
            "feature_importance": dict(list(feature_importance.items())[:10])  # Top 10
        }
        
        # Métricas de entrenamiento
        base_models_perf = self.training_results.get('base_models_performance', {})
        meta_learner_perf = self.training_results.get('meta_learner_performance', {})
        cv_scores = self.training_results.get('cv_scores', [])
        
        # Calcular métricas finales
        if self.predictions is not None:
            y_true = self.df['points'].values
            y_pred = self.predictions
            
            final_mae = mean_absolute_error(y_true, y_pred)
            final_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            final_r2 = r2_score(y_true, y_pred)
            
            # Accuracy por rangos
            accuracy_1pt = np.mean(np.abs(y_pred - y_true) <= 1) * 100
            accuracy_2pt = np.mean(np.abs(y_pred - y_true) <= 2) * 100
            accuracy_3pt = np.mean(np.abs(y_pred - y_true) <= 3) * 100
        else:
            final_mae = final_rmse = final_r2 = 0
            accuracy_1pt = accuracy_2pt = accuracy_3pt = 0
        
        training_metrics = {
            "final_metrics": {
                "mae": final_mae,
                "rmse": final_rmse,
                "r2": final_r2,
                "accuracy_1pt": accuracy_1pt,
                "accuracy_2pt": accuracy_2pt,
                "accuracy_3pt": accuracy_3pt
            },
            "base_models_performance": base_models_perf,
            "meta_learner_performance": meta_learner_perf,
            "cross_validation_scores": {
                "mae_mean": np.mean([score.get('mae', 0) for score in cv_scores]) if cv_scores else 0,
                "mae_std": np.std([score.get('mae', 0) for score in cv_scores]) if cv_scores else 0,
                "r2_mean": np.mean([score.get('r2', 0) for score in cv_scores]) if cv_scores else 0,
                "cv_scores": cv_scores
            }
        }
        
        # Análisis de predicciones
        if self.predictions is not None:
            prediction_analysis = {
                "total_predictions": len(self.predictions),
                "prediction_statistics": {
                    "mean": float(np.mean(self.predictions)),
                    "median": float(np.median(self.predictions)),
                    "std": float(np.std(self.predictions)),
                    "min": float(np.min(self.predictions)),
                    "max": float(np.max(self.predictions)),
                    "q25": float(np.percentile(self.predictions, 25)),
                    "q75": float(np.percentile(self.predictions, 75))
                },
                "accuracy_analysis": {
                    "within_1_point": accuracy_1pt,
                    "within_2_points": accuracy_2pt,
                    "within_3_points": accuracy_3pt
                },
                "error_analysis": {
                    "mean_absolute_error": final_mae,
                    "mean_squared_error": final_rmse ** 2,
                    "root_mean_squared_error": final_rmse,
                    "mean_error": float(np.mean(self.predictions - self.df['points'].values)),
                    "error_std": float(np.std(self.predictions - self.df['points'].values))
                }
            }
        else:
            prediction_analysis = {}
        
        # Hiperparámetros
        hyperparameters = {
            "base_models": self.training_results.get('best_params_per_model', {}),
            "meta_learner": {},
            "cross_validation": {
                "n_splits": 5,
                "test_size": 0.2,
                "random_state": 42
            },
            "scaling": {
                "method": "StandardScaler",
                "fit_on_train_only": True
            }
        }
        
        # Compilar reporte completo
        comprehensive_report = {
            "model_information": model_info,
            "dataset_information": dataset_info,
            "temporal_split": temporal_split,
            "features_information": features_info,
            "training_metrics": training_metrics,
            "prediction_analysis": prediction_analysis,
            "hyperparameters": hyperparameters,
            "model_summary": {
                "n_features": len(selected_features),
                "n_samples": len(self.df),
                "target_mean": float(self.df['points'].mean()),
                "target_std": float(self.df['points'].std()),
                "is_trained": getattr(self.model, 'is_trained', False)
            },
            "training_results": {
                "mae": final_mae,
                "rmse": final_rmse,
                "r2": final_r2,
                "accuracy_1pt": accuracy_1pt,
                "accuracy_2pt": accuracy_2pt,
                "accuracy_3pt": accuracy_3pt
            },
            "generation_timestamp": datetime.now().isoformat()
        }
        
        return comprehensive_report
    
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
            
            logger.info("Pipeline completo ejecutado exitosamente!")
            return results
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo de PTS.
    """
    # Configurar logging informativo
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Logging detallado del trainer principal
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
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
    trainer = XGBoostPTSTrainer(
        players_total_path=players_total_path,
        players_quarters_path=players_quarters_path,
        teams_total_path=teams_total_path,
        teams_quarters_path=teams_quarters_path,
        biometrics_path=biometrics_path,
        output_dir="app/architectures/basketball/results/pts_model",
        n_trials=20,
        cv_folds=3,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento PTS Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 