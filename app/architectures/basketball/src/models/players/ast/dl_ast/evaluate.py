"""
Script de Evaluación y Comparación de Modelos DL AST
===================================================

Evalúa y compara el rendimiento de todos los modelos de Deep Learning
implementados para predicción de asistencias.
"""

import pandas as pd
import numpy as np
import torch
import logging
from pathlib import Path
from typing import Dict, List
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .trainer import DLTrainer
from .config import DLConfig

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluador y comparador de modelos de Deep Learning.
    
    Funcionalidades:
    - Entrenamiento de múltiples modelos
    - Comparación de rendimiento
    - Análisis de características
    - Generación de reportes
    - Visualización de resultados
    """
    
    def __init__(self, config: DLConfig, results_dir: str = "results/dl_models"):
        """
        Inicializa el evaluador.
        
        Args:
            config: Configuración de Deep Learning
            results_dir: Directorio para guardar resultados
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Modelos a evaluar
        self.model_types = [
            "transformer",
            "multiscale_transformer", 
            "lstm",
            "hierarchical_lstm",
            "gnn",
            "vae",
            "ensemble",
            "hybrid",
            "adaptive_hybrid"
        ]
        
        # Resultados de evaluación
        self.evaluation_results = {}
        
        logger.info(f"ModelEvaluator inicializado: {len(self.model_types)} modelos a evaluar")
    
    def evaluate_all_models(self, df: pd.DataFrame, 
                           quick_eval: bool = False) -> Dict:
        """
        Evalúa todos los modelos implementados.
        
        Args:
            df: DataFrame con datos de entrenamiento
            quick_eval: Si True, usa configuración rápida para pruebas
            
        Returns:
            Diccionario con resultados de todos los modelos
        """
        logger.info("Iniciando evaluación de todos los modelos")
        
        # Usar configuración rápida si se especifica
        if quick_eval:
            config = self.config.get_quick_config()
        else:
            config = self.config
        
        results = {}
        
        for model_type in self.model_types:
            try:
                logger.info(f"Evaluando modelo: {model_type}")
                
                # Crear trainer
                trainer = DLTrainer(config, model_type)
                
                # Entrenar modelo
                start_time = datetime.now()
                
                if quick_eval:
                    # Entrenamiento rápido
                    model_results = trainer.train(
                        df.sample(min(1000, len(df))),  # Muestra pequeña
                        validation_split=0.2,
                        save_path=str(self.results_dir / f"{model_type}_quick.pth")
                    )
                else:
                    # Entrenamiento completo con CV
                    cv_results = trainer.cross_validate(df, cv_folds=3)
                    
                    # Entrenamiento final
                    model_results = trainer.train(
                        df,
                        validation_split=0.2,
                        save_path=str(self.results_dir / f"{model_type}_best.pth")
                    )
                    
                    model_results['cv_results'] = cv_results
                
                training_time = (datetime.now() - start_time).total_seconds()
                model_results['training_time'] = training_time
                
                results[model_type] = model_results
                
                logger.info(f"Modelo {model_type} completado: "
                           f"MAE={model_results['final_metrics']['mae']:.4f}, "
                           f"tiempo={training_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Error evaluando modelo {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        self.evaluation_results = results
        
        # Guardar resultados
        self._save_results(results)
        
        # Generar reporte
        self._generate_report(results)
        
        logger.info("Evaluación de todos los modelos completada")
        
        return results
    
    def compare_models(self, results: Dict) -> pd.DataFrame:
        """
        Compara el rendimiento de los modelos.
        
        Args:
            results: Resultados de evaluación
            
        Returns:
            DataFrame con comparación de modelos
        """
        comparison_data = []
        
        for model_type, result in results.items():
            if 'error' in result:
                continue
            
            metrics = result['final_metrics']
            
            row = {
                'Model': model_type,
                'MAE': metrics.get('mae', np.nan),
                'RMSE': metrics.get('rmse', np.nan),
                'R²': metrics.get('r2', np.nan),
                'Accuracy_±1': metrics.get('accuracy_1ast', np.nan),
                'Accuracy_±2': metrics.get('accuracy_2ast', np.nan),
                'Accuracy_±3': metrics.get('accuracy_3ast', np.nan),
                'Training_Time': result.get('training_time', np.nan),
                'Epochs': result.get('epochs_trained', np.nan)
            }
            
            # Agregar resultados de CV si están disponibles
            if 'cv_results' in result:
                cv_summary = result['cv_results']['cv_summary']
                row['CV_MAE_Mean'] = cv_summary.get('mae_mean', np.nan)
                row['CV_MAE_Std'] = cv_summary.get('mae_std', np.nan)
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Ordenar por MAE
        comparison_df = comparison_df.sort_values('MAE')
        
        return comparison_df
    
    def analyze_best_model(self, results: Dict) -> Dict:
        """
        Analiza el mejor modelo en detalle.
        
        Args:
            results: Resultados de evaluación
            
        Returns:
            Análisis detallado del mejor modelo
        """
        # Encontrar mejor modelo por MAE
        best_model = None
        best_mae = float('inf')
        
        for model_type, result in results.items():
            if 'error' in result:
                continue
            
            mae = result['final_metrics'].get('mae', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_model = model_type
        
        if best_model is None:
            return {'error': 'No se encontró modelo válido'}
        
        best_result = results[best_model]
        
        analysis = {
            'best_model': best_model,
            'performance': best_result['final_metrics'],
            'training_info': {
                'epochs': best_result.get('epochs_trained'),
                'time': best_result.get('training_time'),
                'model_summary': best_result.get('model_summary')
            }
        }
        
        # Análisis de CV si está disponible
        if 'cv_results' in best_result:
            analysis['cross_validation'] = best_result['cv_results']['cv_summary']
        
        logger.info(f"Mejor modelo: {best_model} con MAE={best_mae:.4f}")
        
        return analysis
    
    def create_visualizations(self, results: Dict):
        """Crea visualizaciones de los resultados."""
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Preparar datos para visualización
        comparison_df = self.compare_models(results)
        
        # 1. Comparación de MAE
        axes[0, 0].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[0, 0].set_title('Comparación de MAE por Modelo')
        axes[0, 0].set_ylabel('MAE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Accuracy ±2 AST
        axes[0, 1].bar(comparison_df['Model'], comparison_df['Accuracy_±2'])
        axes[0, 1].set_title('Accuracy ±2 AST por Modelo')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Tiempo de entrenamiento vs MAE
        axes[1, 0].scatter(comparison_df['Training_Time'], comparison_df['MAE'])
        axes[1, 0].set_xlabel('Tiempo de Entrenamiento (s)')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('Eficiencia: Tiempo vs Precisión')
        
        # Anotar puntos
        for i, model in enumerate(comparison_df['Model']):
            axes[1, 0].annotate(model, 
                              (comparison_df['Training_Time'].iloc[i], 
                               comparison_df['MAE'].iloc[i]),
                              fontsize=8)
        
        # 4. Heatmap de métricas normalizadas
        metrics_cols = ['MAE', 'RMSE', 'Accuracy_±1', 'Accuracy_±2', 'Accuracy_±3']
        heatmap_data = comparison_df[['Model'] + metrics_cols].set_index('Model')
        
        # Normalizar métricas (0-1)
        heatmap_normalized = heatmap_data.copy()
        for col in metrics_cols:
            if col.startswith('Accuracy'):
                # Para accuracy, mayor es mejor
                heatmap_normalized[col] = heatmap_data[col] / 100
            else:
                # Para MAE/RMSE, menor es mejor (invertir)
                heatmap_normalized[col] = 1 - (heatmap_data[col] / heatmap_data[col].max())
        
        sns.heatmap(heatmap_normalized.T, annot=True, cmap='RdYlGn', 
                   ax=axes[1, 1], cbar_kws={'label': 'Rendimiento Normalizado'})
        axes[1, 1].set_title('Heatmap de Rendimiento')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizaciones guardadas")
    
    def _save_results(self, results: Dict):
        """Guarda los resultados en archivo JSON."""
        
        # Convertir tensors y objetos no serializables
        serializable_results = {}
        
        for model_type, result in results.items():
            if 'error' in result:
                serializable_results[model_type] = result
                continue
            
            serializable_result = {}
            
            for key, value in result.items():
                if isinstance(value, dict):
                    # Convertir métricas
                    serializable_value = {}
                    for k, v in value.items():
                        if isinstance(v, (torch.Tensor, np.ndarray)):
                            serializable_value[k] = float(v)
                        elif isinstance(v, (int, float, str, bool)):
                            serializable_value[k] = v
                        else:
                            serializable_value[k] = str(v)
                    serializable_result[key] = serializable_value
                elif isinstance(value, (int, float, str, bool)):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)
            
            serializable_results[model_type] = serializable_result
        
        # Guardar
        with open(self.results_dir / 'evaluation_results.json', 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info("Resultados guardados en evaluation_results.json")
    
    def _generate_report(self, results: Dict):
        """Genera reporte detallado en texto."""
        
        report_lines = [
            "=" * 80,
            "REPORTE DE EVALUACIÓN DE MODELOS DEEP LEARNING AST",
            "=" * 80,
            f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Modelos evaluados: {len([r for r in results.values() if 'error' not in r])}",
            ""
        ]
        
        # Comparación de modelos
        comparison_df = self.compare_models(results)
        
        report_lines.extend([
            "RANKING DE MODELOS (por MAE):",
            "-" * 40
        ])
        
        for i, row in comparison_df.iterrows():
            report_lines.append(
                f"{i+1:2d}. {row['Model']:20s} | "
                f"MAE: {row['MAE']:6.4f} | "
                f"Acc±2: {row['Accuracy_±2']:5.1f}% | "
                f"R²: {row['R²']:6.4f}"
            )
        
        report_lines.extend(["", ""])
        
        # Análisis del mejor modelo
        best_analysis = self.analyze_best_model(results)
        
        if 'error' not in best_analysis:
            report_lines.extend([
                "ANÁLISIS DEL MEJOR MODELO:",
                "-" * 40,
                f"Modelo: {best_analysis['best_model']}",
                f"MAE: {best_analysis['performance']['mae']:.4f}",
                f"RMSE: {best_analysis['performance']['rmse']:.4f}",
                f"R²: {best_analysis['performance']['r2']:.4f}",
                f"Accuracy ±1 AST: {best_analysis['performance']['accuracy_1ast']:.1f}%",
                f"Accuracy ±2 AST: {best_analysis['performance']['accuracy_2ast']:.1f}%",
                f"Accuracy ±3 AST: {best_analysis['performance']['accuracy_3ast']:.1f}%",
                ""
            ])
            
            if 'cross_validation' in best_analysis:
                cv = best_analysis['cross_validation']
                report_lines.extend([
                    "Validación Cruzada:",
                    f"  MAE: {cv['mae_mean']:.4f} ± {cv['mae_std']:.4f}",
                    f"  R²: {cv['r2_mean']:.4f} ± {cv['r2_std']:.4f}",
                    ""
                ])
        
        # Recomendaciones
        report_lines.extend([
            "RECOMENDACIONES:",
            "-" * 40
        ])
        
        if len(comparison_df) > 0:
            best_model = comparison_df.iloc[0]['Model']
            best_mae = comparison_df.iloc[0]['MAE']
            
            if best_mae < 1.0:
                report_lines.append(" EXCELENTE: MAE < 1.0 - Modelo listo para producción")
            elif best_mae < 1.3:
                report_lines.append(" BUENO: MAE < 1.3 - Rendimiento competitivo")
            else:
                report_lines.append("  MEJORABLE: MAE > 1.3 - Considerar más datos o features")
            
            # Comparar con baseline actual (1.313)
            baseline_mae = 1.313
            if best_mae < baseline_mae:
                improvement = ((baseline_mae - best_mae) / baseline_mae) * 100
                report_lines.append(f" MEJORA: {improvement:.1f}% mejor que baseline actual")
            else:
                report_lines.append(" BASELINE: Modelo actual sigue siendo competitivo")
        
        report_lines.extend([
            "",
            "PRÓXIMOS PASOS:",
            "1. Implementar el mejor modelo en producción",
            "2. Realizar A/B testing con modelo actual",
            "3. Monitorear rendimiento en datos nuevos",
            "4. Considerar ensemble con modelo actual",
            "",
            "=" * 80
        ])
        
        # Guardar reporte
        with open(self.results_dir / 'evaluation_report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        # También imprimir en consola
        print('\n'.join(report_lines))
        
        logger.info("Reporte generado en evaluation_report.txt")


def run_quick_evaluation(data_path: str):
    """
    Ejecuta una evaluación rápida de todos los modelos.
    
    Args:
        data_path: Ruta al archivo de datos
    """
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Crear configuración
    config = DLConfig()
    
    # Crear evaluador
    evaluator = ModelEvaluator(config)
    
    # Ejecutar evaluación rápida
    results = evaluator.evaluate_all_models(df, quick_eval=True)
    
    # Crear visualizaciones
    evaluator.create_visualizations(results)
    
    return results


def run_full_evaluation(data_path: str):
    """
    Ejecuta una evaluación completa de todos los modelos.
    
    Args:
        data_path: Ruta al archivo de datos
    """
    # Configurar logging
    logging.basicConfig(level=logging.INFO)
    
    # Cargar datos
    df = pd.read_csv(data_path)
    
    # Crear configuración de producción
    config = DLConfig()
    
    # Crear evaluador
    evaluator = ModelEvaluator(config)
    
    # Ejecutar evaluación completa
    results = evaluator.evaluate_all_models(df, quick_eval=False)
    
    # Crear visualizaciones
    evaluator.create_visualizations(results)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python evaluate.py <data_path> [quick|full]")
        sys.exit(1)
    
    data_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "quick"
    
    if mode == "quick":
        results = run_quick_evaluation(data_path)
    else:
        results = run_full_evaluation(data_path)
    
    print(f"\nEvaluación completada. Resultados guardados en results/dl_models/") 