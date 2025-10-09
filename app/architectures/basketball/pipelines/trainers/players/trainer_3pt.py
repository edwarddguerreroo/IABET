"""
Trainer Completo para Modelo NBA Player Triples
=============================================

Trainer que integra carga de datos, entrenamiento del modelo de triples por jugador
y generación completa de métricas y visualizaciones para predicción de triples NBA.

Características:
- Integración completa con data loader
- Entrenamiento automatizado con optimización bayesiana
- Generación de dashboard PNG unificado con todas las métricas
- Métricas detalladas específicas para predicción de triples por jugador
- Análisis de feature importance
- Validación cruzada temporal
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
from app.architectures.basketball.src.models.players.triples.model_triples import Stacking3PTModel


warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Configurar estilo de visualizaciones optimizado para PNG
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


class ThreePointsTrainer:
    """
    Trainer completo para modelo de predicción de triples (3PT) por jugador NBA.
    
    Integra carga de datos de jugadores, entrenamiento, evaluación y visualizaciones específicas para triples.
    """
    
    def __init__(self,
                 players_total_path: str,
                 players_quarters_path: str,
                 biometrics_path: str,
                 teams_total_path: str,
                 teams_quarters_path: str,
                 output_dir: str = "app/architectures/basketball/results/3pt_model",
                 n_trials: int = 25,  # Trials para optimización bayesiana
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Inicializa el trainer completo para predicción de triples (3PT) por jugador.
        
        Args:
            players_total_path: Ruta a datos de partidos de jugadores
            players_quarters_path: Ruta a datos de partidos de jugadores
            biometrics_path: Ruta a datos biométricos de jugadores
            teams_total_path: Ruta a datos de equipos (para contexto)
            teams_quarters_path: Ruta a datos de equipos (para contexto)
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
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Crear directorio de salida con manejo robusto
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Directorio de salida creado/verificado: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error creando directorio {self.output_dir}: {e}")
            # Crear directorio alternativo en caso de error
            self.output_dir = os.path.normpath("results_3pt_model")
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Usando directorio alternativo: {self.output_dir}")
        
        # Componentes principales
        self.data_loader = NBADataLoader(
            players_total_path, players_quarters_path, teams_total_path, teams_quarters_path, biometrics_path
        )
        
        # Cargar datasets para pasarlos al modelo TRIPLES
        players_data, teams_data = self.data_loader.load_data(use_quarters=False)
        players_quarters, _ = self.data_loader.load_data(use_quarters=True)
        
        self.model = Stacking3PTModel(
            optimize_hyperparams=True,
            bayesian_n_trials=25,
            teams_df=teams_data,
            players_df=players_data,
            players_quarters_df=players_quarters
        )
        
        # Datos y resultados
        self.df = None
        self.teams_df = None
        self.training_results = None
        self.predictions = None
            
    def load_and_prepare_data(self) -> pd.DataFrame:
        """
        Carga y prepara todos los datos de jugadores necesarios para predicción de triples.
        
        Returns:
            pd.DataFrame: Datos de jugadores preparados para entrenamiento de triples
        """
        # Cargar datos usando el data loader - solo necesitamos los datos de entrenamiento
        players_data, teams_data = self.data_loader.load_data()
        players_quarters, _ = self.data_loader.load_data(use_quarters=True)
        
        # Verificar que existe la columna target de triples
        if 'three_points_made' not in players_data.columns:
            raise ValueError("Columna 'three_points_made' no encontrada en datos de jugadores")
        
        # Estadísticas básicas de los datos de jugadores
        logger.info(f"Datos cargados: {len(players_data):,} registros, {players_data['player'].nunique():,} jugadores únicos")
        
        # Filtrar solo jugadores con minutos jugados > 0
        players_data = players_data[players_data['minutes'] > 0].copy()
        logger.info(f"Registros filtrados (minutes > 0): {len(players_data)}")
        
        self.df = players_data
        self.teams_df = teams_data
        return self.df
    
    def train_model(self) -> Dict:
        """
        Entrena el modelo completo de triples con optimización y validación.
        
        Returns:
            Dict: Resultados del entrenamiento de triples
        """
        
        if self.df is None:
            raise ValueError("Datos no cargados. Ejecutar load_and_prepare_data() primero")
        
        # Entrenar modelo
        start_time = datetime.now()
        logger.info(f"Entrenando modelo 3PT: {len(self.df):,} registros, {self.n_trials} trials")
        
        self.training_results = self.model.train(self.df)
        training_duration = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Modelo 3PT completado en {training_duration:.1f}s")
        
        # Mostrar métricas principales
        if 'mae' in self.training_results and 'r2' in self.training_results:
            logger.info(f"MAE: {self.training_results['mae']:.3f}, R²: {self.training_results['r2']:.3f}")
        
        # Métricas de precisión
        if 'accuracy_1pt' in self.training_results and 'accuracy_2pt' in self.training_results:
            logger.info(f"Precisión ±1: {self.training_results['accuracy_1pt']:.1f}%, ±2: {self.training_results['accuracy_2pt']:.1f}%")
        
        # Generar predicciones
        logger.info("Generando predicciones")
        self.predictions = self.model.predict(self.df)
        
        # Calcular métricas finales
        if hasattr(self.model, 'cutoff_date'):
            test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
            if len(test_data) > 0:
                test_indices = test_data.index
                y_true = test_data['three_points_made'].values
                y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
                
                # Ajustar dimensiones si es necesario
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
                
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                
                # Métricas específicas para TRIPLES
                accuracy_1triple = np.mean(np.abs(y_true - y_pred) <= 1) * 100
                accuracy_2triples = np.mean(np.abs(y_true - y_pred) <= 2) * 100
                
                logger.info(f"Métricas finales | MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
                logger.info(f"Accuracy ±1 triple: {accuracy_1triple:.1f}%, ±2 triples: {accuracy_2triples:.1f}%")
                
                self.training_results.update({
                    'final_mae': mae,
                    'final_rmse': rmse,
                    'final_r2': r2,
                    'final_accuracy_1triple': accuracy_1triple,
                    'final_accuracy_2triples': accuracy_2triples
                })
        
        return self.training_results
    
    def generate_all_visualizations(self):
        """
        Genera una visualización completa en PNG con todas las métricas principales.
        """
        logger.info("Generando dashboard")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Crear figura principal con subplots organizados
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle('Dashboard Completo - Modelo NBA Player Triples Prediction', fontsize=20, fontweight='bold', y=0.98)
        
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
        self._plot_triples_range_analysis_compact(ax6)
        
        # 7. Top tiradores de triples
        ax7 = fig.add_subplot(gs[2, 2:4])
        self._plot_top_shooters_compact(ax7)
        
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
        accuracy_1pt = self.training_results.get('accuracy_1pt', 0)
        accuracy_2pt = self.training_results.get('accuracy_2pt', 0)
        
        # Crear texto de métricas
        metrics_text = f"""
MÉTRICAS DEL MODELO PLAYER TRIPLES

MAE: {mae:.3f}
RMSE: {rmse:.3f}
R²: {r2:.3f}

ACCURACY TRIPLES:
±1 triple: {accuracy_1pt:.1f}%
±2 triples: {accuracy_2pt:.1f}%

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
            logger.info("DEBUG: Iniciando _plot_feature_importance_compact")
            logger.info(f"DEBUG: Modelo entrenado: {getattr(self.model, 'is_trained', False)}")
            logger.info(f"DEBUG: Tiene get_feature_importance: {hasattr(self.model, 'get_feature_importance')}")
            
            # Obtener feature importance del modelo
            top_features = {}
            
            if hasattr(self.model, 'get_feature_importance'):
                logger.info("DEBUG: Llamando get_feature_importance...")
                importance_dict = self.model.get_feature_importance(top_n=20)
                logger.info(f"DEBUG: Resultado importance_dict: {type(importance_dict)}")
                # Verificar si importance_dict es un DataFrame antes de usar keys()
                if hasattr(importance_dict, 'empty'):
                    logger.info(f"DEBUG: importance_dict es DataFrame, shape: {importance_dict.shape}")
                elif importance_dict is not None:
                    logger.info(f"DEBUG: Claves en importance_dict: {list(importance_dict.keys())}")
                else:
                    logger.info("DEBUG: importance_dict es None")
                
                # Manejar diferentes tipos de importance_dict
                if hasattr(importance_dict, 'empty') and not importance_dict.empty:
                    # Es un DataFrame
                    logger.info(f"DEBUG: Procesando DataFrame con {len(importance_dict)} features")
                    if 'feature' in importance_dict.columns and 'importance' in importance_dict.columns:
                        top_features_list = importance_dict.head(15).to_dict('records')  # CAMBIADO A 15
                        top_features = {item['feature']: item['importance'] for item in top_features_list}
                    else:
                        logger.warning("DEBUG: DataFrame no tiene columnas esperadas")
                        top_features = {}
                elif isinstance(importance_dict, dict) and 'feature_importance' in importance_dict:
                    logger.info("DEBUG: Usando nuevo formato feature_importance DataFrame")
                    # Usar el DataFrame directamente y tomar solo las top 15
                    feature_df = importance_dict['feature_importance']
                    if not feature_df.empty and 'feature' in feature_df.columns and 'importance' in feature_df.columns:
                        top_features_list = feature_df.head(15).to_dict('records')  # SOLO TOP 15 PARA DASHBOARD
                        top_features = {item['feature']: item['importance'] for item in top_features_list}
                        logger.info(f"DEBUG: top_features convertido: {len(top_features)} items")
                    else:
                        top_features = {}
                elif isinstance(importance_dict, dict) and 'top_features' in importance_dict and importance_dict['top_features']:
                    logger.info(f"DEBUG: top_features encontrado, longitud: {len(importance_dict['top_features'])}")
                    # Convertir la lista de diccionarios a un diccionario simple - SOLO TOP 15
                    top_features_list = importance_dict['top_features'][:15]  # CAMBIADO DE 20 A 15
                    top_features = {item['feature']: item['importance'] for item in top_features_list}
                    logger.info(f"DEBUG: top_features convertido: {len(top_features)} items")
                elif importance_dict and isinstance(importance_dict, dict):
                    logger.info("DEBUG: Usando dict directo")
                    top_features = dict(list(importance_dict.items())[:15])  # CAMBIADO A 15
                else:
                    logger.warning("DEBUG: No se pudo procesar importance_dict")
            else:
                logger.warning("DEBUG: Modelo no tiene get_feature_importance")
            
            # Si no se obtuvo feature importance, crear uno básico
            if not top_features and hasattr(self.model, 'feature_columns') and len(self.model.feature_columns) > 0:
                logger.info("DEBUG: Creando feature importance básico")
                features = self.model.feature_columns[:20]
                importances = [1.0/len(features)] * len(features)
                top_features = {f: imp for f, imp in zip(features, importances)}
                logger.info(f"DEBUG: Feature importance básico creado: {len(top_features)} items")
            
            # Si aún no hay datos, mostrar mensaje
            if not top_features:
                logger.warning("DEBUG: No hay top_features, mostrando mensaje de no disponible")
                ax.text(0.5, 0.5, 'Feature importance\nno disponible', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Feature Importance', fontweight='bold')
            else:
                logger.info(f"DEBUG: Creando plot con {len(top_features)} features")
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
                logger.info("DEBUG: Plot de feature importance completado exitosamente")
        except Exception as e:
            logger.error(f"DEBUG: Error en _plot_feature_importance_compact: {e}")
            import traceback
            logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            ax.text(0.5, 0.5, f'Error cargando\nfeature importance:\n{str(e)}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance', fontweight='bold')
    
    def _plot_target_distribution_compact(self, ax):
        """Distribución compacta del target 3P."""
        pts_values = self.df['three_points_made']
        
        # Histograma
        ax.hist(pts_values, bins=25, alpha=0.7, color='lightcoral', edgecolor='black')
        
        # Estadísticas
        mean_pts = pts_values.mean()
        median_pts = pts_values.median()
        
        ax.axvline(mean_pts, color='red', linestyle='--', label=f'Media: {mean_pts:.1f}')
        ax.axvline(median_pts, color='blue', linestyle='--', label=f'Mediana: {median_pts:.1f}')
        
        ax.set_xlabel('Triples por Jugador (3P)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de 3P por Jugador', fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    
    def _plot_predictions_vs_actual_compact(self, ax):
        """Gráfico compacto de predicciones vs valores reales."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Predicciones vs Reales', fontweight='bold')
            return
        
        # Usar datos de prueba si están disponibles
        if hasattr(self.model, 'cutoff_date'):
            test_data = self.df[self.df['Date'] >= self.model.cutoff_date].copy()
            if len(test_data) > 0:
                test_indices = test_data.index
                y_true = test_data['three_points_made'].values
                y_pred = self.predictions[test_indices] if len(self.predictions) == len(self.df) else self.predictions[:len(y_true)]
                
                # Ajustar dimensiones si es necesario
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            else:
                # Usar todos los datos
                y_true = self.df['three_points_made'].values
                y_pred = self.predictions
        else:
            # Usar todos los datos
            y_true = self.df['three_points_made'].values
            y_pred = self.predictions
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, s=20, color='coral')
        
        # Línea perfecta
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Predicción Perfecta')
        
        ax.set_xlabel('3P Real')
        ax.set_ylabel('3P Predicho')
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
        
        # Calcular residuos
        y_true = self.df['three_points_made'].values
        y_pred = self.predictions
        residuals = y_true - y_pred
        
        # Scatter plot de residuos
        ax.scatter(y_pred, residuals, alpha=0.6, s=20, color='orange')
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        
        ax.set_xlabel('3P Predicho')
        ax.set_ylabel('Residuos (Real - Predicho)')
        ax.set_title('Análisis de Residuos', fontweight='bold')
        ax.grid(alpha=0.3)
    
        # Agregar estadísticas de residuos
        mae = np.mean(np.abs(residuals))
        ax.text(0.05, 0.95, f'MAE = {mae:.3f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    def _plot_triples_range_analysis_compact(self, ax):
        """Análisis compacto por rangos de triples."""
        if self.predictions is None:
            ax.text(0.5, 0.5, 'Predicciones\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis por Rangos de Triples', fontweight='bold')
            return
        
        y_true = self.df['three_points_made'].values
        y_pred = self.predictions
        
        # Definir rangos de triples
        ranges = [
            (0, 1, 'Bajo (0-1)'),
            (2, 3, 'Medio (2-3)'),
            (4, 6, 'Alto (4-6)'),
            (7, 15, 'Elite (7+)')
        ]
        
        range_names = []
        range_maes = []
        range_counts = []
        
        for min_triples, max_triples, name in ranges:
            mask = (y_true >= min_triples) & (y_true <= max_triples)
            if np.sum(mask) > 0:
                range_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                range_names.append(name)
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
        
        if range_names:
            # Crear gráfico de barras
            bars = ax.bar(range_names, range_maes, alpha=0.8, color=['lightblue', 'lightgreen', 'orange', 'red'])
            
            ax.set_ylabel('MAE')
            ax.set_title('MAE por Rango de Triples', fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            # Agregar valores en las barras
            for bar, mae, count in zip(bars, range_maes, range_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       f'{mae:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
            
        ax.grid(axis='y', alpha=0.3)
    
    def _plot_top_shooters_compact(self, ax):
        """Análisis compacto de top tiradores de triples."""
        # Obtener promedio de triples por jugador
        team_stats = self.df.groupby('player').agg({
            'three_points_made': ['mean', 'count']
        }).reset_index()
        
        team_stats.columns = ['player', 'mean', 'count']
        
        # Filtrar jugadores con al menos 20 juegos
        team_stats = team_stats[team_stats['count'] >= 20]
        
        # Top 10 jugadores en triples
        top_players = team_stats.nlargest(10, 'mean')
        
        if len(top_players) > 0:
            players = [p[:15] + '...' if len(p) > 15 else p for p in top_players['player']]  # Abreviar nombres
            means = top_players['mean']
            
            bars = ax.barh(players, means, alpha=0.8, color='lightsteelblue')
            
            ax.set_xlabel('Promedio 3P')
            ax.set_title('Top 10 Tiradores de Triples', fontweight='bold')
            
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
            'three_points_made': 'mean'
        }).reset_index()
        
        if len(monthly_stats) > 0:
            months = [str(m) for m in monthly_stats['month']]
            avg_pts = monthly_stats['three_points_made']
            
            ax.plot(months, avg_pts, marker='o', linewidth=2, markersize=4)
            ax.set_ylabel('Promedio 3P')
            ax.set_title('Promedio de Triples por Mes', fontweight='bold')
            plt.setp(ax.get_xticklabels(), rotation=45)
        else:
            ax.text(0.5, 0.5, 'Datos insuficientes\npara análisis temporal', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Análisis Temporal', fontweight='bold')
            ax.grid(alpha=0.3)
        
    def _plot_cv_results_compact(self, ax):
        """Gráfico compacto de resultados de validación cruzada."""
        try:
            # CORREGIDO: Usar cv_scores directamente del modelo
            cv_scores = getattr(self.model, 'cv_scores', None)
            
            if cv_scores is None or not cv_scores:
                ax.text(0.5, 0.5, 'Resultados CV\nno disponibles', 
                   ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Validación Cruzada', fontweight='bold')
                return
        
            # Extraer métricas de CV - cv_scores es directamente la lista
            if isinstance(cv_scores, list) and len(cv_scores) > 0:
                mae_scores = [fold.get('mae', 0) for fold in cv_scores if isinstance(fold, dict)]
                
                if not mae_scores:
                    ax.text(0.5, 0.5, 'Métricas CV\nno válidas', 
                           ha='center', va='center', transform=ax.transAxes)
                    ax.set_title('Validación Cruzada', fontweight='bold')
                    return
                
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
            ax.text(0.5, 0.5, f'Error en CV:\n{str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Validación Cruzada', fontweight='bold')
    
    def _generate_detailed_training_report(self):
        """Genera un reporte SÚPER DETALLADO del entrenamiento con toda la información posible."""
        
        # Información básica del modelo
        model_info = {
            'model_type': 'XGBoost 3PT Stacking Ensemble',
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
                'mean': float(self.df['three_points_made'].mean()),
                'median': float(self.df['three_points_made'].median()),
                'std': float(self.df['three_points_made'].std()),
                'min': float(self.df['three_points_made'].min()),
                'max': float(self.df['three_points_made'].max()),
                'q25': float(self.df['three_points_made'].quantile(0.25)),
                'q75': float(self.df['three_points_made'].quantile(0.75))
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
            'total_features_generated': len(self.model.selected_features) if hasattr(self.model, 'selected_features') else 0,
            'features_used': list(self.model.selected_features) if hasattr(self.model, 'selected_features') else [],
            'feature_importance': dict(list(self.model.feature_importance.items())[:10]) if hasattr(self.model, 'feature_importance') and self.model.feature_importance else {}
        }
        
        # Métricas de entrenamiento
        training_metrics = {
            'final_metrics': self.training_results,
            'base_models_performance': getattr(self.model, "base_models_performance", {}),
            'meta_learner_performance': getattr(self.model, "meta_learner_performance", {}),
            'cross_validation_scores': getattr(self.model, "cv_scores", {})
        }
        
        # Obtener métricas de modelos base si están disponibles
        if hasattr(self.model, 'trained_base_models'):
            for name, model in self.model.trained_base_models.items():
                try:
                    # Predecir en datos de prueba para obtener métricas
                    X_test = test_data[self.model.selected_features].fillna(0)
                    y_test = test_data['three_points_made']
                    
                    # Aplicar escalado si está disponible
                    if hasattr(self.model, 'scaler') and self.model.scaler:
                        X_test_scaled = self.model.scaler.transform(X_test)
                        X_test = pd.DataFrame(X_test_scaled, columns=self.model.selected_features, index=X_test.index)
                    
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
                y_test = test_data['three_points_made']
                
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
                    'within_1_triple': float(np.mean(np.abs(self.predictions - self.df['three_points_made']) <= 1) * 100),
                    'within_2_triples': float(np.mean(np.abs(self.predictions - self.df['three_points_made']) <= 2) * 100),
                    'within_3_triples': float(np.mean(np.abs(self.predictions - self.df['three_points_made']) <= 3) * 100)
                },
                'error_analysis': {
                    'mean_absolute_error': float(np.mean(np.abs(self.predictions - self.df['three_points_made']))),
                    'mean_squared_error': float(np.mean((self.predictions - self.df['three_points_made'])**2)),
                    'root_mean_squared_error': float(np.sqrt(np.mean((self.predictions - self.df['three_points_made'])**2))),
                    'mean_error': float(np.mean(self.predictions - self.df['three_points_made'])),
                    'error_std': float(np.std(self.predictions - self.df['three_points_made']))
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
        if hasattr(self.model.stacking_model, 'models'):
            for name, model in self.model.stacking_model.models.items():
                try:
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
        logger.info("Guardando modelo y resultados")
        
        # Asegurar que el directorio de salida existe
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Guardar modelo en .joblib/ con ruta absoluta
        joblib_dir = os.path.abspath('app/architectures/basketball/.joblib')
        os.makedirs(joblib_dir, exist_ok=True)
        model_path = os.path.join(joblib_dir, '3pt_model.joblib')
        if hasattr(self.model, 'save_model'):
            self.model.save_model(model_path)
        else:
            # Backup: usar joblib
            joblib.dump(self.model, model_path)
        
        # Generar reporte SÚPER DETALLADO
        report = self._generate_detailed_training_report()
        report_path = os.path.normpath(os.path.join(self.output_dir, 'training_report.json'))
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Guardar predicciones
        if self.predictions is not None:
            predictions_df = self.df[['player', 'Date', 'Team', 'three_points_made']].copy()
            predictions_df['three_points_made_predicted'] = self.predictions
            predictions_df['error'] = self.predictions - self.df['three_points_made']
            predictions_df['abs_error'] = np.abs(predictions_df['error'])
            
            predictions_path = os.path.normpath(os.path.join(self.output_dir, 'predictions.csv'))
            predictions_df.to_csv(predictions_path, index=False)
        
        # Guardar feature importance COMPLETA (todas las features como en PTS)
        try:
            if hasattr(self.model, 'get_feature_importance'):
                # EXPORTAR TODAS LAS FEATURES - sin límite de top_n
                importance_result = self.model.get_feature_importance(top_n=None)  # SIN LÍMITE para exportar TODAS
                if importance_result is not None and not importance_result.empty:
                    # Usar el DataFrame directamente
                    importance_df = importance_result.copy()
            
                    # Agregar información adicional sobre las features (como en PTS)
                    importance_df = importance_df.sort_values('importance', ascending=False)
                    total_features = len(importance_df)
                    importance_df['rank'] = range(1, total_features + 1)
                    importance_df['importance_pct'] = (importance_df['importance'] / importance_df['importance'].sum() * 100).round(4)
                    importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum().round(4)
                    
                    importance_path = os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))
                    importance_df.to_csv(importance_path, index=False)
                    logger.info(f"Feature importance exportada: {total_features} features")
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
                    logger.info(f"Feature importance básico exportado: {len(self.model.selected_features)} features")
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
                    logger.info(f"Feature importance básico exportado: {len(self.model.feature_columns)} features")
            except Exception as e2:
                logger.error(f"No se pudo crear feature importance básico: {e2}")
        
        # Crear resumen de archivos generados
        files_summary = {
            'model_file': os.path.abspath('app/architectures/basketball/.joblib/3pt_triples_model.joblib'),
            'dashboard_image': 'model_dashboard_complete.png',
            'training_report': 'training_report.json',
            'predictions': 'predictions.csv',
            'feature_importance': 'feature_importance.csv',
            'error_analysis_script': 'analyze_prediction_errors.py',
            'output_directory': os.path.normpath(self.output_dir),
            'generation_timestamp': datetime.now().isoformat(),
            'model_improvements': {
                'outlier_handling': 'Procesamiento robusto de outliers usando IQR y winsorización',
                'integer_predictions': 'Predicciones convertidas a valores enteros mediante redondeo probabilístico',
                'prediction_smoothing': 'Suavizado de predicciones consecutivas para evitar cambios abruptos',
                'robust_limits': 'Límites adaptativos basados en percentiles de datos reales'
            }
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
        logger.info(f"  • Feature Importance: {os.path.normpath(os.path.join(self.output_dir, 'feature_importance.csv'))}")
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
            
            return results
            
        except Exception as e:
            logger.error(f"Error en pipeline de entrenamiento: {str(e)}")
            raise


def main():
    """
    Función principal para ejecutar el entrenamiento completo de TRIPLES (3PT).
    """
    # Configurar logging detallado para debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Mostrar mensajes informativos del trainer principal
    main_logger = logging.getLogger(__name__)
    main_logger.setLevel(logging.INFO)
    
    # Silenciar solo las librerías más verbosas
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
    trainer = ThreePointsTrainer(
        players_total_path=players_total_path,
        players_quarters_path=players_quarters_path,
        teams_total_path=teams_total_path,
        teams_quarters_path=teams_quarters_path,
        biometrics_path=biometrics_path,
        output_dir="app/architectures/basketball/results/3pt_model",
        n_trials=25,
        cv_folds=5,
        random_state=42
    )
    
    # Ejecutar pipeline completo
    results = trainer.run_complete_training()
    
    print("Entrenamiento Triples (3PT) Model completado!")
    print(f"Resultados: {trainer.output_dir}")
    
    return results


if __name__ == "__main__":
    main() 
