"""
Trainer Completo para Modelo NBA Teams Points
=============================================

Trainer que integra carga de datos, entrenamiento del modelo de puntos por equipo
y generación completa de métricas y visualizaciones para predicción de puntos NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para predicción de puntos por equipo
- Análisis de feature importance
- Validación cruzada temporal
"""

import json
import logging
import os
import sys
import warnings

# Suprimir warnings de matplotlib
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
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
from app.architectures.basketball.src.models.teams.teams_points.model_teams_points import TeamPointsModel

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class TeamsPointsTrainer:
    """
    Trainer completo para modelo de predicción de puntos por equipo NBA.
    
    Integra carga de datos, entrenamiento, evaluación y visualizaciones.
    """
    
    def __init__(self,
                players_total_path: str,
                players_quarters_path: str,
                 biometrics_path: str,
                teams_total_path: str,
                teams_quarters_path: str,
                 output_dir: str = "app/architectures/basketball/results/teams_points_model",
                 n_trials: int = 25,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para predicción de puntos por equipo.
        
        Args:
            players_total_path: Ruta a datos de partidos
            players_quarters_path: Ruta a datos de partidos
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
            self.output_dir = os.path.normpath("results_teams_points_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            players_total_path=players_total_path,
            players_quarters_path=players_quarters_path,
            teams_total_path=teams_total_path,
            teams_quarters_path=teams_quarters_path,
            biometrics_path=biometrics_path
        )
        self.model = TeamPointsModel(
            optimize_hyperparams=True,
            bayesian_n_trials=25
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
        
        logger.info(f"Trainer Teams Points inicializado")
    
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos necesarios.
        
        Returns:
            pd.DataFrame: Datos preparados para entrenamiento
        """
        logger.info("Cargando datos NBA")
        
        # Cargar datos usando el data loader
        self.df, self.teams_df, players_quarters, self.teams_quarters_df = self.data_loader.load_data()
        
        # Cargar datasets adicionales de equipos
        self.teams_total_df = self.teams_df
        
        # Actualizar el modelo con todos los datasets
        self.model = TeamPointsModel(
            optimize_hyperparams=True,
            bayesian_n_trials=25,
            df_players=self.df,  # Datos de jugadores
            teams_total_df=self.teams_total_df,  # Datos totales de equipos
            teams_quarters_df=self.teams_quarters_df  # Datos por cuartos de equipos
        )
        
        # Usar datos de equipos para predicción de puntos
        teams_data = self.teams_df.copy()
        
        # Verificar que existe la columna target
        if 'points' not in teams_data.columns:
            raise ValueError("Columna 'points' no encontrada en datos de equipos")
        
        # Estadísticas básicas de los datos
        logger.info(f"Datos cargados: {len(teams_data)} registros de equipos")
        logger.info(f"Equipos únicos: {teams_data['Team'].nunique()}")
        logger.info(f"Rango de fechas: {teams_data['Date'].min()} a {teams_data['Date'].max()}")
        
        # Estadísticas del target
        pts_stats = teams_data['points'].describe()
        logger.info(f"Estadísticas PTS por equipo:")
        logger.info(f"  | Media: {pts_stats['mean']:.1f}")
        logger.info(f"  | Mediana: {pts_stats['50%']:.1f}")
        logger.info(f"  | Min/Max: {pts_stats['min']:.0f}/{pts_stats['max']:.0f}")
        logger.info(f"  | Desv. Estándar: {pts_stats['std']:.1f}")
        
        self.df = teams_data
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento
        """
        logger.info("Iniciando entrenamiento del modelo Teams Points")
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        self.training_results = self.model.train(self.df, validation_split=0.2)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo Teams Points completado")
        
        # Mostrar resultados del entrenamiento
        logger.info("=" * 50)
        logger.info("RESULTADOS DEL ENTRENAMIENTO TEAMS POINTS")
        logger.info("=" * 50)
        
        if 'mae' in self.training_results:
            logger.info(f"MAE: {self.training_results['mae']:.3f}")
        if 'rmse' in self.training_results:
            logger.info(f"RMSE: {self.training_results['rmse']:.3f}")
        if 'r2' in self.training_results:
            logger.info(f"R²: {self.training_results['r2']:.3f}")
        
        # Métricas específicas para puntos
        if 'accuracy_5pts' in self.training_results:
            logger.info(f"Accuracy ±5pts: {self.training_results['accuracy_5pts']:.1f}%")
        if 'accuracy_10pts' in self.training_results:
            logger.info(f"Accuracy ±10pts: {self.training_results['accuracy_10pts']:.1f}%")
        
        logger.info("=" * 50)
        
        # Generar predicciones
        logger.info("Generando predicciones")
        # Aplicar los mismos filtros que en entrenamiento para consistencia
        df_filtered = self.model._apply_data_quality_filters(self.df)
        self.predictions = self.model.predict(self.df)
        # Guardar también los datos filtrados para visualizaciones
        self.df_filtered = df_filtered
        
        # Calcular métricas finales
        if hasattr(self.model, 'cutoff_date'):
            test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
            if len(test_data) > 0:
                test_indices = test_data.index
                y_true = test_data['points'].values
                y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
                
                # Ajustar dimensiones si es necesario
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                
                # Métricas específicas para puntos
                accuracy_5pts = np.mean(np.abs(y_true - y_pred) <= 5) * 100
                accuracy_10pts = np.mean(np.abs(y_true - y_pred) <= 10) * 100
                
                logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
                logger.info(f"Accuracy ±5pts: {accuracy_5pts:.1f}%, ±10pts: {accuracy_10pts:.1f}%")
                
                self.training_results.update({
                    'final_mae': mae,
                    'final_rmse': rmse,
                    'final_r2': r2,
                    'final_accuracy_5pts': accuracy_5pts,
                    'final_accuracy_10pts': accuracy_10pts
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
        fig.suptitle('Dashboard Completo - Modelo NBA Teams Points Prediction', fontsize=20, fontweight='bold', y=0.98)
        
        # Crear grid de subplots (4 filas x 4 columnas)
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Métricas principales del modelo
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_model_metrics_summary(ax1)
        
        # 2. Feature importance
        ax2 = fig.add_subplot(gs[0, 1:3])
        self._plot_feature_importance_compact(ax2)
        
        # 3. Distribución del target
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_target_distribution_compact(ax3)
        
        # 4. Predicciones vs Reales
        ax4 = fig.add_subplot(gs[1, 0:2])
        self._plot_predictions_vs_actual_compact(ax4)
        
        # 5. Residuos
        ax5 = fig.add_subplot(gs[1, 2:4])
        self._plot_residuals_compact(ax5)
        
        # 6. Análisis por rangos de puntos
        ax6 = fig.add_subplot(gs[2, 0:2])
        self._plot_points_range_analysis_compact(ax6)
        
        # 7. Top equipos ofensivos
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_top_offensive_teams_compact(ax7)
        
        # 8. Análisis temporal
        ax8 = fig.add_subplot(gs[3, 0:2])
        self._plot_temporal_analysis_compact(ax8)
        
        # 9. Validación cruzada
        ax9 = fig.add_subplot(gs[3, 2:4])
        self._plot_cv_results_compact(ax9)
        
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
        accuracy_5pts = self.training_results.get('accuracy_5pts', 0)
        accuracy_10pts = self.training_results.get('accuracy_10pts', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO TEAMS POINTS

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY PUNTOS:
±5 pts: {accuracy_5pts:.1f}%
±10 pts: {accuracy_10pts:.1f}%

MODELOS BASE:
• XGBoost (Offense Engine)
• LightGBM (Tempo Control)
• CatBoost (Team Chemistry)
• Neural Network (Patterns)
"""
        
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        ax.set_title('Resumen del Modelo', fontweight='bold', fontsize=12)
    
    def _plot_feature_importance_compact(self, ax):
        """Gráfico compacto de importancia de features."""
        try:
            # Debug logging
            # Iniciando plot de feature importance
            
            # Obtener feature importance del modelo
            top_features = {}
            
            if hasattr(self.model, 'get_feature_importance'):
                importance_dict = self.model.get_feature_importance(top_n=20)
                
                # Manejar diferentes tipos de importance_dict
                if hasattr(importance_dict, 'empty') and not importance_dict.empty:
                    # Es un DataFrame
                    if 'feature' in importance_dict.columns and 'importance' in importance_dict.columns:
                        top_features_list = importance_dict.head(15).to_dict('records')  # CAMBIADO A 15
                        top_features = {item['feature']: item['importance'] for item in top_features_list}
                    else:
                        logger.warning("DataFrame no tiene columnas esperadas")
                        top_features = {}
                elif isinstance(importance_dict, dict) and 'feature_importance' in importance_dict:
                    # Usar el DataFrame directamente y tomar solo las top 15
                    feature_df = importance_dict['feature_importance']
                    if not feature_df.empty and 'feature' in feature_df.columns and 'importance' in feature_df.columns:
                        top_features_list = feature_df.head(15).to_dict('records')  # SOLO TOP 15 PARA DASHBOARD
                        top_features = {item['feature']: item['importance'] for item in top_features_list}
                    else:
                        top_features = {}
                elif isinstance(importance_dict, dict) and 'top_features' in importance_dict and importance_dict['top_features']:
                    # Convertir la lista de diccionarios a un diccionario simple - SOLO TOP 15
                    top_features_list = importance_dict['top_features'][:15]  # CAMBIADO DE 20 A 15
                    top_features = {item['feature']: item['importance'] for item in top_features_list}
                elif importance_dict and isinstance(importance_dict, dict):
                    top_features = dict(list(importance_dict.items())[:15])  # CAMBIADO A 15
                else:
                    logger.warning("No se pudo procesar importance_dict")
            else:
                logger.warning("Modelo no tiene get_feature_importance")
            
            # Si no se obtuvo feature importance, crear uno básico
            if not top_features and hasattr(self.model, 'feature_columns') and len(self.model.feature_columns) > 0:
                features = self.model.feature_columns[:20]
                importances = [1.0/len(features)] * len(features)
                top_features = {f: imp for f, imp in zip(features, importances)}
            
            # Si aún no hay datos, mostrar mensaje
            if not top_features:
                logger.warning("No hay top_features, mostrando mensaje de no disponible")
                ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
                return
            
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            # Crear gráfico horizontal
            y_pos = np.arange(len(features))
            bars = ax.barh(y_pos, importances, color='lightcoral', alpha=0.8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f.replace('_', ' ').title()[:25] for f in features], fontsize=7)
            ax.set_xlabel('Importancia')
            ax.set_title('Top 15 Features Más Importantes', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, importances)):
                ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{val:.3f}', va='center', fontsize=7)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
            
        except Exception as e:
            logger.error(f"Error en _plot_feature_importance_compact: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            ax.text(0.5, 0.5, f'Error cargando\nfeature importance:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target PTS."""
        pts_values = self.df['points']
        
        # Histograma
        ax.hist(pts_values, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Estadísticas
        mean_pts = pts_values.mean()
        median_pts = pts_values.median()
        
        ax.axvline(mean_pts, color='red', linestyle='--', label=f'Media: {mean_pts:.1f}')
        ax.axvline(median_pts, color='blue', linestyle='--', label=f'Mediana: {median_pts:.1f}')
        
        ax.set_xlabel('Puntos por Equipo (PTS)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de PTS por Equipo', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Gráfico compacto de predicciones vs valores reales."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        # Usar datos filtrados para consistencia con las predicciones
        if hasattr(self, 'df_filtered'):
            y_true = self.df_filtered['points'].values
            y_pred = self.predictions
        else:
            # Fallback: usar datos originales con ajuste de dimensiones
            y_true = self.df['points'].values
            y_pred = self.predictions
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
        
        ax.set_xlabel('PTS Real')
        ax.set_ylabel('PTS Predicho')
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
        
        # Calcular residuos usando datos filtrados
        if hasattr(self, 'df_filtered'):
            y_true = self.df_filtered['points'].values
            y_pred = self.predictions
        else:
            # Fallback: usar datos originales con ajuste de dimensiones
            y_true = self.df['points'].values
            y_pred = self.predictions
            # Ajustar dimensiones si es necesario
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='orange')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('PTS Predicho')
        ax.set_ylabel('Residuos (Real - Predicho)')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Agregar estadísticas de residuos
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'MAE = {mae:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_points_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de puntos."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Puntos', fontweight='bold')
            return
        
        # Obtener datos usando datos filtrados
        if hasattr(self, 'df_filtered'):
            y_true = self.df_filtered['points'].values
            y_pred = self.predictions
        else:
            # Fallback: usar datos originales con ajuste de dimensiones
            y_true = self.df['points'].values
            y_pred = self.predictions
            # Ajustar dimensiones si es necesario
            min_len = min(len(y_true), len(y_pred))
            y_true = y_true[:min_len]
            y_pred = y_pred[:min_len]
        
        # Definir rangos de puntos
        ranges = [
            (70, 90, 'Bajo (70-90)'),
            (91, 110, 'Medio (91-110)'),
            (111, 130, 'Alto (111-130)'),
            (131, 160, 'Elite (131+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_pts, max_pts, name in ranges:
            mask = (y_true >= min_pts) & (y_true <= max_pts)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Puntos', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae, count in zip(bars, range_maes, range_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{mae:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
            ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_offensive_teams_compact(self, ax):
        """Análisis compacto de top equipos ofensivos."""
        # Obtener promedio de puntos por equipo
        team_stats = self.df.groupby('Team').agg({
            'points': ['mean', 'count']
        }).reset_index()
        
        team_stats.columns = ['Team', 'mean', 'count']
        
        # Filtrar equipos con al menos 20 juegos
        team_stats = team_stats[team_stats['count'] >= 20]
        
        # Top 10 equipos ofensivos
        top_teams = team_stats.nlargest(10, 'mean')
        
        if len(top_teams) > 0:
            teams = [t[:3] for t in top_teams['Team']]  # Abreviar nombres
            means = top_teams['mean']
            
            bars = ax.barh(teams, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio PTS')
            ax.set_title('Top 10 Equipos Ofensivos', fontweight='bold')
            
            # Agregar valores en las barras
            for i, (bar, val) in enumerate(zip(bars, means)):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontsize=8)
            
            ax.grid(axis='x', alpha=0.3)
            ax.invert_yaxis()
    
    def _plot_temporal_analysis_compact(self, ax):
        """Análisis temporal compacto."""
        # Agrupar por mes
        df_copy = self.df.copy()
        df_copy['month'] = pd.to_datetime(df_copy['Date']).dt.to_period('M')
        
        monthly_stats = df_copy.groupby('month').agg({
            'points': 'mean'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            avg_pts = monthly_stats['points']
            
            ax.plot(months, avg_pts, marker='o', linewidth=2, markersize=4)
            
            ax.set_ylabel('Promedio PTS')
            ax.set_title('Promedio de Puntos por Mes', fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45)
            ax.grid(alpha=0.3)
    
    def _plot_cv_results_compact(self, ax):
        """Gráfico compacto de resultados de validación cruzada."""
        try:
            # Intentar obtener resultados de CV del modelo
            cv_results = getattr(self.model, '_cv_results', None)
            
            if cv_results is None:
                ax.text(0.5, 0.5, 'Resultados CV\nno disponibles', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validación Cruzada', fontweight='bold')
                return
            
            # Extraer métricas de CV
            if isinstance(cv_results, dict) and 'cv_scores' in cv_results:
                fold_scores = cv_results['cv_scores']
                mae_scores = [fold.get('mae', 0) for fold in fold_scores]
                
                folds = range(1, len(mae_scores) + 1)
                
                bars = ax.bar(folds, mae_scores, alpha=0.8, color='purple')
                
                # Agregar valores en las barras
                for bar, mae in zip(bars, mae_scores):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           f'{mae:.2f}', ha='center', va='bottom', fontsize=8)
                
                ax.set_xlabel('Fold')
                ax.set_ylabel('MAE')
                ax.set_title('Validación Cruzada por Fold', fontweight='bold')
                ax.set_xticks(folds)
                ax.set_xticklabels([f'Fold {i}' for i in folds])
                
                # Agregar promedio
                avg_mae = np.mean(mae_scores)
                ax.axhline(y=avg_mae, color='red', linestyle='--', 
                          label=f'Promedio: {avg_mae:.3f}')
                ax.legend()
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'Formato CV\nno compatible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validación Cruzada', fontweight='bold')
                
        except Exception as e:
            ax.text(0.5, 0.5, f'Error en CV:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validación Cruzada', fontweight='bold')
    
    def _generate_comprehensive_report(self) -> Dict:
        """
        Genera un reporte comprensivo con métricas detalladas por modelo.
        
        Returns:
            Dict: Reporte completo del entrenamiento
        """
        
        # Obtener métricas individuales de cada modelo (si están disponibles)
        individual_models_metrics = {}
        if hasattr(self.model, 'model_metrics') and self.model.model_metrics:
            for model_name, metrics in self.model.model_metrics.items():
                individual_models_metrics[model_name] = {
                    'validation_metrics': metrics,
                    'model_type': self._get_model_type_info(model_name)
                }
        
        # Información de hyperparameters optimizados
        hyperparameters_info = {}
        if hasattr(self.model, 'bayesian_optimizer') and self.model.bayesian_optimizer:
            if hasattr(self.model.bayesian_optimizer, 'optimization_results'):
                hyperparameters_info = self.model.bayesian_optimizer.optimization_results
        
        # Análisis de overfitting
        overfitting_analysis = self._analyze_overfitting()
        
        # Feature importance detallada
        feature_analysis = self._analyze_features()
        
        # Cross-validation analysis detallado
        cv_analysis = self._analyze_cross_validation()
        
        # Reporte principal
        report = {
            'metadata': {
                'model_type': 'NBA Teams Points Ensemble (Enhanced)',
                'training_methodology': 'Bayesian Optimization + TPESampler',
                'feature_engineering': 'Four Factors + Differential Features',
                'timestamp': datetime.now().isoformat(),
                'total_features': len(self.model.feature_columns) if hasattr(self.model, 'feature_columns') else 'Unknown'
            },
            
            'ensemble_results': {
                'best_model': self.training_results.get('best_model', 'Unknown'),
                'training_metrics': self.training_results.get('train', {}),
                'validation_metrics': self.training_results.get('validation', {}),
                'stacking_metrics': self.training_results.get('stacking', {}),
                'cross_validation_summary': self.training_results.get('cross_validation', {})
            },
            
            'individual_models': individual_models_metrics,
            
            'optimization_details': {
                'hyperparameter_optimization': hyperparameters_info,
                'bayesian_calls': getattr(self.model, 'bayesian_n_calls', 20),
                'optimization_time_estimate': 'Disponible en logs'
            },
            
            'performance_analysis': {
                'overfitting_analysis': overfitting_analysis,
                'feature_analysis': feature_analysis,
                'cross_validation_analysis': cv_analysis
            },
            
            'data_summary': {
                'total_samples': len(self.df) if self.df is not None else 'Unknown',
                'training_split': 0.8,
                'validation_split': 0.2,
                'target_variable': 'PTS (Team Points)',
                'data_time_range': self._get_data_time_range()
            },
            
            'recommendations': self._generate_recommendations(overfitting_analysis)
        }
        
        return report
    
    def _get_model_type_info(self, model_name: str) -> str:
        """Obtiene información del tipo de modelo."""
        model_types = {
            'xgboost': 'Gradient Boosting (XGBoost)',
            'lightgbm': 'Gradient Boosting (LightGBM)', 
            'catboost': 'Gradient Boosting (CatBoost)',
            'stacking': 'Ensemble Stacking'
        }
        return model_types.get(model_name, 'Unknown Model Type')
    
    def _analyze_overfitting(self) -> Dict:
        """Analiza el nivel de overfitting del modelo."""
        # CORREGIR: Usar las claves correctas de métricas
        train_metrics = self.training_results.get('training_metrics', {})
        val_metrics = self.training_results.get('validation_metrics', {})
        
        # Fallback a las claves alternativas si no existen
        if not train_metrics:
            train_metrics = self.training_results.get('train', {})
        if not val_metrics:
            val_metrics = self.training_results.get('validation', {})
        
        # Calcular diferencias
        train_r2 = train_metrics.get('r2', 0)
        val_r2 = val_metrics.get('r2', 0)
        train_mae = train_metrics.get('mae', 0)
        val_mae = val_metrics.get('mae', 0)
        
        r2_diff = train_r2 - val_r2
        mae_diff = val_mae - train_mae
        
        # Clasificar nivel de overfitting basado en R² y MAE
        if r2_diff < 0.05 and mae_diff < 1.0:
            overfitting_level = "Mínimo"
            severity = "low"
        elif r2_diff < 0.10 and mae_diff < 2.0:
            overfitting_level = "Moderado"
            severity = "medium"
        elif r2_diff < 0.15 and mae_diff < 3.0:
            overfitting_level = "Alto"
            severity = "high"
        else:
            overfitting_level = "Severo"
            severity = "critical"
        
        return {
            'r2_difference': round(r2_diff, 4),
            'mae_difference': round(mae_diff, 4),
            'overfitting_level': overfitting_level,
            'severity': severity,
            'train_r2': train_r2,
            'validation_r2': val_r2,
            'train_mae': train_mae,
            'validation_mae': val_mae,
            'interpretation': self._interpret_overfitting(severity, r2_diff)
        }
    
    def _interpret_overfitting(self, severity: str, r2_diff: float) -> str:
        """Proporciona interpretación del overfitting."""
        interpretations = {
            "low": "Modelo bien balanceado con generalización adecuada.",
            "medium": "Overfitting moderado. Considerar más regularización.",
            "high": "Overfitting significativo. Requiere reducción de complejidad.",
            "critical": "Overfitting severo. Modelo no apto para producción."
        }
        return interpretations.get(severity, "Sin interpretación disponible")
    
    def _analyze_features(self) -> Dict:
        """Analiza el rendimiento y distribución de features."""
        try:
            if hasattr(self.model, 'get_feature_importance'):
                importance_dict = self.model.get_feature_importance(top_n=None)
                if importance_dict and 'feature_importance' in importance_dict:
                    # CORREGIR: Usar el DataFrame directamente, no comparación ambigua
                    features_df = importance_dict['feature_importance']
                    
                    # Verificar que no esté vacío
                    if not features_df.empty and 'importance' in features_df.columns:
                    # Análisis de concentración de importancia
                        total_importance = features_df['importance'].sum()
                        if total_importance > 0:
                            top_5_pct = features_df.head(5)['importance'].sum() / total_importance * 100
                            top_10_pct = features_df.head(10)['importance'].sum() / total_importance * 100
                        else:
                            top_5_pct = 0
                            top_10_pct = 0
                    
                    # Categorizar features
                    feature_categories = self._categorize_features(features_df)
                    
                    return {
                        'total_features': len(features_df),
                        'top_5_concentration': round(top_5_pct, 2),
                        'top_10_concentration': round(top_10_pct, 2),
                        'feature_categories': feature_categories,
                        'top_features': features_df.head(15).to_dict('records')
                    }
            return {'status': 'Feature importance not available'}
        except Exception as e:
            logger.error(f"Error en análisis de features: {e}")
            return {'error': str(e)}
    
    def _categorize_features(self, features_df: pd.DataFrame) -> Dict:
        """Categoriza las features por tipo."""
        categories = {
            'four_factors': 0,
            'differential': 0,
            'historical_points': 0,
            'efficiency': 0,
            'opponent': 0,
            'other': 0
        }
        
        # CORREGIR: Usar la columna correcta 'feature' no 'Feature'
        feature_column = 'feature' if 'feature' in features_df.columns else 'Feature'
        
        for _, row in features_df.iterrows():
            feature_name = str(row[feature_column]).lower()  # Convertir a string y lowercase
            if 'four_factors' in feature_name or 'efg' in feature_name:
                categories['four_factors'] += 1
            elif 'advantage' in feature_name or 'projected_pace' in feature_name:
                categories['differential'] += 1
            elif 'pts_' in feature_name:
                categories['historical_points'] += 1
            elif 'pct' in feature_name or 'efficiency' in feature_name:
                categories['efficiency'] += 1
            elif 'opp_' in feature_name:
                categories['opponent'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _analyze_cross_validation(self) -> Dict:
        """Analiza los resultados de validación cruzada."""
        cv_results = self.training_results.get('cross_validation', {})
        if not cv_results:
            return {'status': 'Cross-validation results not available'}
        
        # CORREGIR: Calcular estabilidad correctamente
        mae_scores = cv_results.get('mae_scores', [])
        if len(mae_scores) > 1:
            mae_range = float(np.max(mae_scores) - np.min(mae_scores))
            mean_mae = np.mean(mae_scores)
            # Estabilidad normalizada: menor rango relativo = mayor estabilidad
            stability_score = max(0, 1 - (mae_range / mean_mae)) if mean_mae > 0 else 0
        else:
            mae_range = 0
            stability_score = 0
        
        # Clasificar nivel de estabilidad
        if stability_score > 0.8:
            stability_level = "Excelente"
        elif stability_score > 0.6:
            stability_level = "Buena"
        elif stability_score > 0.4:
            stability_level = "Moderada"
        else:
            stability_level = "Pobre"
        
        return {
            'stability_score': round(stability_score, 4),
            'stability_level': stability_level,
            'mean_mae': cv_results.get('mean_mae', 0),
            'std_mae': cv_results.get('std_mae', 0),
            'mean_r2': cv_results.get('mean_r2', 0),
            'std_r2': cv_results.get('std_r2', 0),
            'mae_range': round(mae_range, 4)
        }
    
    def _get_data_time_range(self) -> Dict:
        """Obtiene el rango temporal de los datos."""
        try:
            if self.df is not None and 'Date' in self.df.columns:
                min_date = self.df['Date'].min()
                max_date = self.df['Date'].max()
                return {
                    'start_date': str(min_date),
                    'end_date': str(max_date),
                    'total_days': str((max_date - min_date).days)
                }
            return {'status': 'Date information not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def _generate_recommendations(self, overfitting_analysis: Dict) -> List[str]:
        """Genera recomendaciones basadas en el análisis."""
        recommendations = []
        
        # Recomendaciones para overfitting
        if overfitting_analysis['severity'] in ['high', 'critical']:
            recommendations.extend([
                "🚨 CRÍTICO: Reducir complejidad del modelo",
                "📉 Incrementar regularización en hiperparámetros",
                "🔧 Considerar feature selection más agresiva",
                " Probar ensemble más simple (menos modelos base)"
            ])
        elif overfitting_analysis['severity'] == 'medium':
            recommendations.extend([
                " Ajustar regularización de hiperparámetros",
                " Evaluar feature selection",
                " Monitorear en producción"
            ])
        
        # Recomendaciones generales
        recommendations.extend([
            "📈 Implementar features de player availability",
            "😴 Agregar features de fatiga (back-to-back games)",
            " Incorporar features de estilo de juego del oponente",
            "📉 Considerar target engineering (log transform, etc.)"
        ])
        
        return recommendations
    
    def save_results(self):
        """Guarda todos los resultados del entrenamiento."""
        logger.info("Guardando resultados")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/ con ruta absoluta
        joblib_dir = os.path.abspath('app/architectures/basketball/.joblib')
        os.makedirs(joblib_dir, exist_ok=True)
        model_path = os.path.join(joblib_dir, 'teams_points_model.joblib')
        if hasattr(self.model, 'save_model'):
            self.model.save_model(model_path)
        else:
            # Backup: usar joblib
            joblib.dump(self.model, model_path)
        
        # Generar reporte completo y detallado
        report = self._generate_comprehensive_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones usando datos filtrados
        if self.predictions is not None:
            if hasattr(self, 'df_filtered'):
                predictions_df = self.df_filtered[['Team', 'Date', 'Opp', 'points']].copy()
            else:
                predictions_df = self.df[['Team', 'Date', 'Opp', 'points']].copy()
                # Ajustar dimensiones si es necesario
                min_len = min(len(predictions_df), len(self.predictions))
                predictions_df = predictions_df.iloc[:min_len]
                self.predictions = self.predictions[:min_len]
            
            predictions_df['points_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - predictions_df['points']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance COMPLETA (todas las features como en PTS)
        try:
            if hasattr(self.model, 'get_feature_importance'):
                # EXPORTAR TODAS LAS FEATURES - sin límite de top_n
                importance_result = self.model.get_feature_importance(top_n=None)  # SIN LÍMITE para exportar TODAS
                if importance_result and 'feature_importance' in importance_result:
                    # Usar el DataFrame directamente del nuevo formato
                    importance_df = importance_result['feature_importance'].copy()
                    
                    # Agregar información adicional sobre las features (como en PTS)
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    total_features = len(importance_df)
                    importance_df['rank'] = range(1, total_features + 1)
                    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(4)
                    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum().round(4)
                    
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    importance_df.to_csv(importance_path, index=False)
                else:
                    logger.warning("Feature importance está vacío")
            else:
                logger.warning("El modelo no tiene método 'get_feature_importance'")
        except Exception as e:
            logger.error(f"Error al guardar feature importance: {e}")
            # Intentar crear un archivo básico de feature importance
            try:
                if hasattr(self.model, 'selected_features') and self.model.selected_features:
                    basic_df = pd.DataFrame({
                        'feature': self.model.selected_features,
                        'importance': [1.0/len(self.model.selected_features)] * len(self.model.selected_features),
                        'rank': range(1, len(self.model.selected_features) + 1),
                        'importance_pct': [100.0/len(self.model.selected_features)] * len(self.model.selected_features)
                    })
                    basic_df['cumulative_pct'] = basic_df['importance_pct'].cumsum().round(4)
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    basic_df.to_csv(importance_path, index=False)
                elif hasattr(self.model, 'feature_columns') and self.model.feature_columns:
                    basic_df = pd.DataFrame({
                        'feature': self.model.feature_columns,
                        'importance': [1.0/len(self.model.feature_columns)] * len(self.model.feature_columns),
                        'rank': range(1, len(self.model.feature_columns) + 1),
                        'importance_pct': [100.0/len(self.model.feature_columns)] * len(self.model.feature_columns)
                    })
                    basic_df['cumulative_pct'] = basic_df['importance_pct'].cumsum().round(4)
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    basic_df.to_csv(importance_path, index=False)
            except Exception as e2:
                logger.error(f"No se pudo crear feature importance básico: {e2}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': os.path.abspath('app/architectures/basketball/.joblib/teams_points_model.joblib'),
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
        logger.info(f"  Modelo: {model_path}")
        logger.info(f"  Dashboard PNG: {os.path.normpath(os.path.join(self.output_dir, 'model_dashboard_complete.png'))}")
        logger.info(f"  Reporte: {report_path}")
        if self.predictions is not None:
            logger.info(f"  Predicciones: {predictions_path}")
        logger.info(f"  Feature Importance: {os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))}")
        logger.info(f"  Resumen: {summary_path}")
    
    def run_complete_training(self):
        """
        Ejecuta el pipeline completo de entrenamiento.
        
        Returns:
            Dict: Resultados completos del entrenamiento
        """
        logger.info("Iniciando pipeline de entrenamiento Teams Points")
        
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
    Función principal para ejecutar el entrenamiento completo de TEAMS_POINTS.
    """
    # Configurar logging detallado
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
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.category').setLevel(logging.ERROR)
    
    # Rutas de datos (ajustar según tu configuración)
    players_total_path = "app/architectures/basketball/data/players_total.csv"
    players_quarters_path = "app/architectures/basketball/data/players_quarters.csv"
    biometrics_path = "app/architectures/basketball/data/biometrics.csv"
    teams_total_path = "app/architectures/basketball/data/teams_total.csv"
    teams_quarters_path = "app/architectures/basketball/data/teams_quarters.csv"


    trainer = TeamsPointsTrainer(
        players_total_path=players_total_path,
        players_quarters_path=players_quarters_path,
        teams_total_path=teams_total_path,
        teams_quarters_path=teams_quarters_path,
        biometrics_path=biometrics_path,
        output_dir="app/architectures/basketball/results/teams_points_model",
        n_trials=25,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento Teams Points Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 