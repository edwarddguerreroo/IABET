"""
Trainer Completo para Modelo XGBoost AST
========================================

Trainer que integra carga de datos, entrenamiento del modelo XGBoost
y generación completa de métricas y visualizaciones para predicción de asistencias NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para asistencias
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

# Agregar el directorio raíz al PYTHONPATH para resolver las importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..')))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Imports del proyecto - rutas corregidas
import sys
import os

# Configurar rutas correctamente
current_file = os.path.abspath(__file__)
basketball_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))  # app/architectures/basketball
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(basketball_dir))))  # raíz del proyecto

# Agregar rutas al path
sys.path.insert(0, project_root)
sys.path.insert(0, basketball_dir)

# Imports directos
from app.architectures.basketball.src.preprocessing.data_loader import NBADataLoader
from app.architectures.basketball.src.models.players.ast.model_ast import XGBoostASTModel

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class XGBoostASTTrainer:
    """
    Trainer completo para modelo XGBoost de predicción de asistencias NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                 players_total_path: str,
                 players_quarters_path: str,
                 biometrics_path: str,
                 teams_total_path: str,
                 teams_quarters_path: str,
                 output_dir: str = "app/architectures/basketball/results/ast_model",
                 n_trials: int = 25,  # Trials para optimización bayesiana 
                 cv_folds: int = 5,
                 random_state: int = 42,
                 max_features: int = 40):  # Features optimizadas centralizadas
        """
        Inicializa el trainer completo para AST.
        
        Args:
            players_total_path: Ruta a datos de partidos de jugadores
            players_quarters_path: Ruta a datos de partidos de jugadores
            teams_total_path: Ruta a datos de equipos
            teams_quarters_path: Ruta a datos de equipos
            biometrics_path: Ruta a datos biométricos
            output_dir: Directorio de salida para resultados
            n_trials: Trials para optimización bayesiana
            cv_folds: Folds para validación cruzada
            random_state: Semilla para reproducibilidad
            max_features: Número máximo de features a seleccionar (optimizado: 75)
        """
        self.players_total_path = players_total_path
        self.players_quarters_path = players_quarters_path
        self.teams_total_path = teams_total_path
        self.teams_quarters_path = teams_quarters_path
        self.biometrics_path = biometrics_path
        self.output_dir = os.path.normpath(output_dir)
        self.random_state = random_state
        self.max_features = max_features
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_ast_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            players_total_path, players_quarters_path, teams_total_path, teams_quarters_path, biometrics_path
        )
        self.model = XGBoostASTModel(
            enable_neural_network=False,  # Deshabilitado para mayor velocidad
            enable_gpu=False,
            random_state=random_state,
            teams_df=None  # Se asignará después de cargar datos
        )
        
        # Configurar parámetros de optimización
        self.model.stacking_model.n_trials = n_trials
        self.model.stacking_model.cv_folds = cv_folds
        
        logger.info(f"Configuración optimización: {n_trials} trials, {cv_folds} folds, {max_features} features optimizadas")
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer XGBoost AST inicializado | Output: {self.output_dir}")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df, players_quarters, teams_quarters = self.data_loader.load_data()
        
        # Cargar datos de quarters para features avanzadas
        self.players_quarters_df = pd.read_csv(self.players_quarters_path)
        logger.info(f"Datos de quarters cargados: {len(self.players_quarters_df)} registros")
        
        # ASIGNAR DATOS DE EQUIPOS Y QUARTERS AL MODELO
        self.model.stacking_model.teams_df = self.teams_df
        self.model.stacking_model.feature_engineer.teams_df = self.teams_df
        self.model.stacking_model.feature_engineer.players_quarters_df = self.players_quarters_df
        logger.info("Datos de equipos y quarters asignados al modelo para features avanzadas")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(self.df)} registros de jugadores")
        logger.info(f"Datos de equipos: {len(self.teams_df)} registros")
        logger.info(f"Jugadores únicos: {self.df['player'].nunique()}")
        logger.info(f"Equipos únicos: {self.df['Team'].nunique()}")
        logger.info(f"Rango de fechas: {self.df['Date'].min()} a {self.df['Date'].max()}")
        
        # Verificar target
        if 'assists' not in self.df.columns:
            raise ValueError("Columna 'assists' no encontrada en los datos")
        
        # Estadísticas del target
        ast_stats = self.df['assists'].describe()
        logger.info(f"Estadísticas assists - Media: {ast_stats['mean']:.2f}, "
                   f"Mediana: {ast_stats['50%']:.2f}, "
                   f"Max: {ast_stats['max']:.0f}")
        
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo AST")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo AST completado | Duración: {training_duration:.1f} segundos")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO AST")
        logger.info("=" * 50)
        logger.info(f"MAE: {self.training_results.get('mae', 0):.4f}")
        logger.info(f"RMSE: {self.training_results.get('rmse', 0):.4f}")
        logger.info(f"R²: {self.training_results.get('r2', 0):.4f}")
        logger.info(f"Accuracy ±1ast: {self.training_results.get('accuracy_1ast', 0):.1f}%")
        logger.info(f"Accuracy ±2ast: {self.training_results.get('accuracy_2ast', 0):.1f}%")
        logger.info(f"Accuracy ±3ast: {self.training_results.get('accuracy_3ast', 0):.1f}%")
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones")
        self.predictions = self.model.predict(self.df)
        
        # Calcular métricas finales en datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) > 0:
            # Obtener índices de los datos de prueba para alinear predicciones
            test_indices = test_data.index
            
            y_true = test_data['assists'].values
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
            
            # Métricas específicas para asistencias
            accuracy_1ast = np.mean(np.abs(y_true - y_pred) <= 1) * 100
            accuracy_2ast = np.mean(np.abs(y_true - y_pred) <= 2) * 100
            accuracy_3ast = np.mean(np.abs(y_true - y_pred) <= 3) * 100
            
            logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            logger.info(f"Accuracy ±1ast: {accuracy_1ast:.1f}%, ±2ast: {accuracy_2ast:.1f}%, ±3ast: {accuracy_3ast:.1f}%")
            
            self.training_results.update({
                'final_mae': mae,
                'final_rmse': rmse,
                'final_r2': r2,
                'final_accuracy_1ast': accuracy_1ast,
                'final_accuracy_2ast': accuracy_2ast,
                'final_accuracy_3ast': accuracy_3ast
            })
        
        # Mostrar TODAS las features con su importancia
        logger.info("Registrando importancia de TODAS las features...")
        self._log_all_features_importance()
        
        return self.training_results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando visualización completa en PNG")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(28, 20))
        fig.suptitle('Dashboard Completo - Modelo NBA AST Prediction', fontsize=22, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (5 filas x 4 columnas)
        gs = fig.add_gridspec(5, 4, hspace=0.35, wspace=0.3)
        
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
        
        # 7. Análisis por rangos de asistencias (tercera fila, derecha)
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_assists_range_analysis_compact(ax7)
        
        # 8. Análisis temporal (cuarta fila, izquierda)
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Top pasadores predicciones (cuarta fila, derecha)
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_top_passers_analysis_compact(ax9)
        
        # 10. Correlation plot (quinta fila, completa)
        ax10 = fig.add_subplot(gs[4, :])
        self._plot_correlation_analysis(ax10)
        
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
    
    def _plot_correlation_analysis(self, ax):
        """
        Gráfico de análisis de correlación entre feature importance y correlación con el target.
        Similar al plot de PTS.
        """
        if not hasattr(self.model.stacking_model, 'feature_importance') or not self.model.stacking_model.feature_importance:
            ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Correlación', fontweight='bold')
            return
        
        # Obtener feature importance
        importance_dict = self.model.stacking_model.feature_importance
        
        # Calcular correlaciones
        logger.info("Calculando correlaciones para el plot...")
        correlations = {}
        train_data = self.df[self.df['Date'] < self.model.cutoff_date].copy()
        y_train = train_data['assists']
        
        for feature in importance_dict.keys():
            if feature in train_data.columns:
                try:
                    feature_values = train_data[feature]
                    valid_mask = ~(feature_values.isna() | y_train.isna())
                    if valid_mask.sum() > 10:
                        corr = feature_values[valid_mask].corr(y_train[valid_mask])
                        correlations[feature] = corr
                    else:
                        correlations[feature] = np.nan
                except:
                    correlations[feature] = np.nan
            else:
                correlations[feature] = np.nan
        
        # Crear DataFrame para el plot
        plot_data = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values()),
            'correlation': [correlations.get(f, np.nan) for f in importance_dict.keys()]
        })
        
        # Filtrar features con datos válidos
        plot_data = plot_data.dropna(subset=['correlation'])
        
        if len(plot_data) == 0:
            ax.text(0.5, 0.5, 'No hay datos\nde correlación', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis de Correlación', fontweight='bold')
            return
        
        # Tomar top 30 features por importance
        plot_data = plot_data.nlargest(30, 'importance')
        
        # Calcular correlación absoluta
        plot_data['abs_correlation'] = plot_data['correlation'].abs()
        
        # Crear scatter plot
        scatter = ax.scatter(
            plot_data['importance'], 
            plot_data['abs_correlation'],
            s=100, 
            alpha=0.6,
            c=plot_data['abs_correlation'],
            cmap='RdYlGn',
            edgecolors='black',
            linewidth=0.5
        )
        
        # Agregar labels para las top 10 features
        top_10 = plot_data.nlargest(10, 'importance')
        for _, row in top_10.iterrows():
            ax.annotate(
                row['feature'][:15],  # Truncar nombres largos
                (row['importance'], row['abs_correlation']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=7,
                alpha=0.8
            )
        
        ax.set_xlabel('Feature Importance', fontsize=11, fontweight='bold')
        ax.set_ylabel('Correlación Absoluta con Assists', fontsize=11, fontweight='bold')
        ax.set_title('Análisis: Importance vs Correlación (Top 30 Features)', fontweight='bold', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')
        
        # Agregar colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Correlación Absoluta', fontsize=9)
        
        # Agregar líneas de referencia
        ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Correlación Moderada (0.3)')
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Correlación Fuerte (0.5)')
        ax.legend(fontsize=8, loc='upper right')
        
        # Estadísticas en el plot
        mean_corr = plot_data['abs_correlation'].mean()
        median_corr = plot_data['abs_correlation'].median()
        stats_text = f'Media: {mean_corr:.3f}\nMediana: {median_corr:.3f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))
        
        logger.info(f"✅ Plot de correlación generado con {len(plot_data)} features")
    
    def _plot_model_metrics_summary(self, ax):
        """Resumen de métricas principales del modelo."""
        ax.axis('off')
        
        # Obtener métricas
        mae = self.training_results.get('mae', 0)
        rmse = self.training_results.get('rmse', 0)
        r2 = self.training_results.get('r2', 0)
        accuracy_1ast = self.training_results.get('accuracy_1ast', 0)
        accuracy_2ast = self.training_results.get('accuracy_2ast', 0)
        accuracy_3ast = self.training_results.get('accuracy_3ast', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO AST

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY ASISTENCIAS:
±1 ast: {accuracy_1ast:.1f}%
±2 ast: {accuracy_2ast:.1f}%
±3 ast: {accuracy_3ast:.1f}%

MODELOS BASE:
        • XGBoost
        • LightGBM
        • CatBoost
        • Ridge (meta-learner: LightGBM)
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features - TODAS LAS FEATURES."""
        if not hasattr(self.model.stacking_model, 'feature_importance') or not self.model.stacking_model.feature_importance:
            ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
            return
        
        # Obtener TODAS las features (no solo top 15)
        importance_dict = self.model.stacking_model.feature_importance
        
        # Ordenar por importancia descendente
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Mostrar todas las features (máximo 50 para legibilidad)
        max_features_to_show = min(50, len(sorted_features))
        top_features = dict(sorted_features[:max_features_to_show])
        
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        # Crear gráfico horizontal
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, color='lightcoral', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f.replace('_', ' ').title()[:25] for f in features], fontsize=7)
        ax.set_xlabel('Importancia')
        ax.set_title(f'Feature Importance - Top {max_features_to_show} de {len(importance_dict)}', fontweight='bold')
        
        # Agregar valores en las barras (solo para las primeras 20 para evitar saturación)
        for i, (bar, val) in enumerate(zip(bars, importances)):
            if i < 20:  # Solo mostrar valores para las primeras 20
                ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=6)
        
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()

    def _log_all_features_importance(self):
        """Registra todas las features con su importancia en el log."""
        importance_dict = getattr(self.model.stacking_model, 'feature_importance', {})
        if not importance_dict:
            logger.warning("Feature importance no disponible")
            return
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        total = sum(importance_dict.values()) or 1.0
        logger.info(f"Total features: {len(sorted_features)}")
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target AST."""
        ast_values = self.df['assists']
        
        # Histograma
        ax.hist(ast_values, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Estadísticas
        mean_ast = ast_values.mean()
        median_ast = ast_values.median()
        
        ax.axvline(mean_ast, color='red', linestyle='--', label=f'Media: {mean_ast:.1f}')
        ax.axvline(median_ast, color='blue', linestyle='--', label=f'Mediana: {median_ast:.1f}')
        
        ax.set_xlabel('Asistencias (assists)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de AST', fontweight='bold')
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
        y_true = test_data['assists'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='coral')
        
        # Línea perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        ax.set_xlabel('assists Real')
        ax.set_ylabel('assists Predicho')
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
        y_true = test_data['assists'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='orange')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('assists Predicho')
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
    
    def _plot_assists_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de asistencias."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Asistencias', fontweight='bold')
            return
        
        # Usar datos de prueba
        test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
        if len(test_data) == 0:
            ax.text(0.5, 0.5, 'Datos de prueba\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Asistencias', fontweight='bold')
            return
        
        test_indices = test_data.index
        y_true = test_data['assists'].values
        y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
        
        # Ajustar dimensiones si es necesario
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        
        # Definir rangos de asistencias
        ranges = [
            (0, 2, 'Bajo (0-2)'),
            (3, 5, 'Medio (3-5)'),
            (6, 8, 'Alto (6-8)'),
            (9, 20, 'Elite (9+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_ast, max_ast, name in ranges:
            mask = (y_true >= min_ast) & (y_true <= max_ast)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Asistencias', fontweight='bold')
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
        y_true = test_data['assists'].values
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
    
    def _plot_top_passers_analysis_compact(self, ax):
        """Análisis compacto de top pasadores."""
        # Obtener top pasadores por promedio
        player_stats = self.df.groupby('player').agg({
            'assists': ['mean', 'count']
        }).reset_index()
        
        player_stats.columns = ['player', 'mean', 'count']
        
        # Filtrar jugadores con al menos 10 juegos
        player_stats = player_stats[player_stats['count'] >= 10]
        
        # Top 10 pasadores
        top_passers = player_stats.nlargest(10, 'mean')
        
        if len(top_passers) > 0:
            players = [p[:15] + '' if len(p) > 15 else p for p in top_passers['player']]
            means = top_passers['mean']
            
            bars = ax.barh(players, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio assists')
            ax.set_title('Top 10 Pasadores', fontweight='bold')
            
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
            'model_type': 'XGBoost AST Stacking Ensemble',
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
                'mean': float(self.df['assists'].mean()),
                'median': float(self.df['assists'].median()),
                'std': float(self.df['assists'].std()),
                'min': float(self.df['assists'].min()),
                'max': float(self.df['assists'].max()),
                'q25': float(self.df['assists'].quantile(0.25)),
                'q75': float(self.df['assists'].quantile(0.75))
            }
        }
        
        # División temporal
        train_data, test_data = self.model.stacking_model._temporal_split(self.df)
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
                    y_test = test_data['assists']
                    
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
                y_test = test_data['assists']
                
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
                    'within_1_assist': float(np.mean(np.abs(self.predictions - self.df['assists']) <= 1) * 100),
                    'within_2_assists': float(np.mean(np.abs(self.predictions - self.df['assists']) <= 2) * 100),
                    'within_3_assists': float(np.mean(np.abs(self.predictions - self.df['assists']) <= 3) * 100)
                },
                'error_analysis': {
                    'mean_absolute_error': float(np.mean(np.abs(self.predictions - self.df['assists']))),
                    'mean_squared_error': float(np.mean((self.predictions - self.df['assists'])**2)),
                    'root_mean_squared_error': float(np.sqrt(np.mean((self.predictions - self.df['assists'])**2))),
                    'mean_error': float(np.mean(self.predictions - self.df['assists'])),
                    'error_std': float(np.std(self.predictions - self.df['assists']))
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
        model_path = os.path.normpath(os.path.join('app/architectures/basketball/.joblib', 'ast_model.joblib'))
        os.makedirs('app/architectures/basketball/.joblib', exist_ok=True)
        self.model.save_model(model_path)
        
        # Guardar reporte SÚPER DETALLADO
        report = self._generate_detailed_training_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['player', 'Date', 'Team', 'assists']].copy()
            predictions_df['assists_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['assists']
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
            
            # Calcular correlación con el target (assists)
            logger.info("Calculando correlaciones con el target (assists)...")
            correlations = {}
            
            # Obtener datos de entrenamiento
            train_data = self.df[self.df['Date'] < self.model.cutoff_date].copy()
            y_train = train_data['assists']
            
            for feature in importance_df['feature']:
                if feature in train_data.columns:
                    try:
                        # Calcular correlación de Pearson
                        feature_values = train_data[feature]
                        
                        # Eliminar NaN para el cálculo
                        valid_mask = ~(feature_values.isna() | y_train.isna())
                        if valid_mask.sum() > 10:  # Mínimo 10 valores válidos
                            corr = feature_values[valid_mask].corr(y_train[valid_mask])
                            correlations[feature] = corr
                        else:
                            correlations[feature] = np.nan
                    except Exception as e:
                        logger.warning(f"Error calculando correlación para {feature}: {e}")
                        correlations[feature] = np.nan
                else:
                    correlations[feature] = np.nan
            
            # Agregar columna de correlación
            importance_df['correlation'] = importance_df['feature'].map(correlations)
            importance_df['abs_correlation'] = importance_df['correlation'].abs()
            
            importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
            importance_df.to_csv(importance_path, index=False)
            
            logger.info(f"✅ Feature importance exportada: {total_features} features completas en {importance_path}")
            logger.info(f"✅ Correlaciones calculadas: {importance_df['correlation'].notna().sum()} features con correlación válida")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': 'app/architectures/basketball/.joblib/ast_model.joblib',
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
        logger.info("Iniciando pipeline de entrenamiento AST")
        
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
    Función principal para ejecutar el entrenamiento completo de AST.
    """
    # Configurar logging ultra-silencioso
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Solo mensajes críticos del trainer principal
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    # Silenciar librerías externas
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('xgboost').setLevel(logging.WARNING)
    logging.getLogger('lightgbm').setLevel(logging.WARNING)
    logging.getLogger('catboost').setLevel(logging.WARNING)
    logging.getLogger('optuna').setLevel(logging.WARNING)
    
    # Rutas de datos (ajustar según tu configuración)
    players_total_path = "app/architectures/basketball/data/players_total.csv"
    players_quarters_path = "app/architectures/basketball/data/players_quarters.csv"
    teams_total_path = "app/architectures/basketball/data/teams_total.csv"
    teams_quarters_path = "app/architectures/basketball/data/teams_quarters.csv"
    biometrics_path = "app/architectures/basketball/data/biometrics.csv"
    
    # Crear y ejecutar trainer
    trainer = XGBoostASTTrainer(
        players_total_path=players_total_path,
        players_quarters_path=players_quarters_path,
        teams_total_path=teams_total_path,
        teams_quarters_path=teams_quarters_path,
        biometrics_path=biometrics_path,
        output_dir="app/architectures/basketball/results/ast_model",
        n_trials=25,
        cv_folds=3,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento AST Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 