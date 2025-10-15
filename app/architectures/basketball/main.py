"""
Sistema Unificado de Predicción NBA - Orquestador Principal
=========================================================

Sistema maestro que integra y orquesta todos los modelos de predicción NBA:
- Modelos de jugadores: PTS, AST, TRB, 3PT, Double-Double
- Modelos de equipos: Teams Points (con predicción de victoria integrada)
- Análisis avanzado integrado para todos los modelos
- Pipeline completo automatizado
- Métricas unificadas y reportes consolidados

Arquitectura:
- Carga de datos centralizada
- Entrenamiento paralelo de modelos
- Análisis avanzado automático
- Reportes consolidados
- Visualizaciones unificadas
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Agregar el directorio raíz al PYTHONPATH para resolver las importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

import pandas as pd
from app.utils.helpers import convert_numpy_types
import numpy as np
from tqdm import tqdm

# Imports de trainers de jugadores
from app.models.prediction import PredictionResponse
from app.architectures.basketball.pipelines.trainers.players.trainer_pts import XGBoostPTSTrainer
from app.architectures.basketball.pipelines.trainers.players.trainer_ast import XGBoostASTTrainer
from app.architectures.basketball.pipelines.trainers.players.trainer_trb import XGBoostTRBTrainer
from app.architectures.basketball.pipelines.trainers.players.trainer_3pt import ThreePointsTrainer
from app.architectures.basketball.pipelines.trainers.players.trainer_dd import DoubleDoubleTrainer

# Imports de trainers de equipos
from app.architectures.basketball.pipelines.trainers.teams.trainer_teams_points import TeamsPointsTrainer

# Import del predictor unificado
from app.architectures.basketball.pipelines.predict.unified_predictor.unified_predictor import UnifiedPredictor

# Import locales
from app.models.game import Game


# Configurar logging unificado
warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class NBATrainers:
    """
    Sistema Unificado de Predicción NBA
    
    Orquesta todos los modelos de predicción NBA con análisis avanzado integrado.
    """
    
    def __init__(self,
                 players_total_path: str = "app/architectures/basketball/data/players_total.csv",
                 players_quarters_path: str = "app/architectures/basketball/data/players_quarters.csv",
                 teams_total_path: str = "app/architectures/basketball/data/teams_total.csv",
                 teams_quarters_path: str = "app/architectures/basketball/data/teams_quarters.csv",
                 biometrics_path: str = "app/architectures/basketball/data/biometrics.csv",
                 output_base_dir: str = "results",
                 n_trials: int = 50,
                 enable_parallel: bool = True,
                 enable_advanced_analysis: bool = True,
                 random_state: int = 42):
        """
        Inicializa el sistema unificado NBA.
        
        Args:
            game_data_path: Ruta a datos de partidos
            biometrics_path: Ruta a datos biométricos
            teams_path: Ruta a datos de equipos
            output_base_dir: Directorio base para resultados
            n_trials: Trials para optimización bayesiana
            enable_parallel: Si habilitar entrenamiento paralelo
            enable_advanced_analysis: Si habilitar análisis avanzado
            random_state: Semilla para reproducibilidad
        """
        self.players_total_path = players_total_path
        self.players_quarters_path = players_quarters_path
        self.teams_total_path = teams_total_path
        self.teams_quarters_path = teams_quarters_path
        self.biometrics_path = biometrics_path
        self.output_base_dir = output_base_dir
        self.n_trials = n_trials
        self.enable_parallel = enable_parallel
        self.enable_advanced_analysis = enable_advanced_analysis
        self.random_state = random_state
        
        # Crear directorio base
        os.makedirs(output_base_dir, exist_ok=True)
        
        # Configuración de modelos con estructura organizada
        self.player_models_config = {
            'pts': {
                'trainer_class': XGBoostPTSTrainer,
                'output_dir': os.path.join(output_base_dir, 'players', 'pts_model'),
                'description': 'Predicción de Puntos'
            },
            'ast': {
                'trainer_class': XGBoostASTTrainer,
                'output_dir': os.path.join(output_base_dir, 'players', 'ast_model'),
                'description': 'Predicción de Asistencias'
            },
            'trb': {
                'trainer_class': XGBoostTRBTrainer,
                'output_dir': os.path.join(output_base_dir, 'players', 'trb_model'),
                'description': 'Predicción de Rebotes'
            },
            '3pt': {
                'trainer_class': ThreePointsTrainer,
                'output_dir': os.path.join(output_base_dir, 'players', '3pt_model'),
                'description': 'Predicción de Triples'
            },
            'double_double': {
                'trainer_class': DoubleDoubleTrainer,
                'output_dir': os.path.join(output_base_dir, 'players', 'double_double_model'),
                'description': 'Predicción de Doble-Dobles'
            }
        }
        
        self.team_models_config = {
            'teams_points': {
                'trainer_class': TeamsPointsTrainer,
                'output_dir': os.path.join(output_base_dir, 'teams', 'teams_points_model'),
                'description': 'Predicción de Puntos por Equipo'
            }
        }
        
        # Resultados del entrenamiento
        self.training_results = {}
        self.advanced_analysis_results = {}
        self.trained_models = {}
        
        logger.info("Sistema Unificado NBA inicializado")
        logger.info(f"Modelos de jugadores: {len(self.player_models_config)}")
        logger.info(f"Modelos de equipos: {len(self.team_models_config)}")
        logger.info(f"Entrenamiento paralelo: {enable_parallel}")
    
    def train_single_model(self, model_type: str, model_name: str, config: Dict) -> Dict[str, Any]:
        """
        Entrena un modelo individual.
        
        Args:
            model_type: Tipo de modelo ('player' o 'team')
            model_name: Nombre del modelo
            config: Configuración del modelo
            
        Returns:
            Dict con resultados del entrenamiento
        """
        try:
            logger.info(f"Iniciando {model_name} ({config['description']})")
            
            # Inicializar trainer con parámetros específicos por tipo
            if model_type == 'team':
                # Los trainers de equipos tienen parámetros estándar
                trainer = config['trainer_class'](
                    players_total_path=self.players_total_path,
                    players_quarters_path=self.players_quarters_path,
                    teams_total_path=self.teams_total_path,
                    teams_quarters_path=self.teams_quarters_path,
                    biometrics_path=self.biometrics_path,
                    output_dir=config['output_dir'],
                    n_optimization_trials=self.n_trials,
                    random_state=self.random_state
                )
            else:
                # Trainers estándar
                trainer = config['trainer_class'](
                    players_total_path=self.players_total_path,
                    players_quarters_path=self.players_quarters_path,
                    teams_total_path=self.teams_total_path,
                    teams_quarters_path=self.teams_quarters_path,
                    biometrics_path=self.biometrics_path,
                    output_dir=config['output_dir'],
                    n_trials=self.n_trials,
                    random_state=self.random_state
                )
            
            # Ejecutar entrenamiento completo
            start_time = datetime.now()
            results = trainer.run_complete_training()
            training_duration = (datetime.now() - start_time).total_seconds()
            
            # Guardar modelo entrenado para análisis avanzado
            self.trained_models[model_name] = {
                'trainer': trainer,
                'model': trainer.model,
                'type': model_type,
                'results': results
            }
            
            # Extraer métricas principales para logging
            metrics = {}
            if hasattr(results, 'get') and results.get('metrics'):
                result_metrics = results['metrics']
                if 'mae' in result_metrics:
                    metrics['MAE'] = result_metrics['mae']
                if 'r2' in result_metrics:
                    metrics['R2'] = result_metrics['r2']
                if 'accuracy' in result_metrics:
                    metrics['ACC'] = result_metrics['accuracy']
            
            logger.info(f"Completado {model_name} | Duración: {training_duration:.1f}s | {metrics}")
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'status': 'success',
                'training_duration': training_duration,
                'results': results,
                'output_dir': config['output_dir']
            }
            
        except Exception as e:
            error_msg = f"Error entrenando {model_name}: {str(e)}"
            logger.error(f"Error en entrenamiento de {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            
            return {
                'model_name': model_name,
                'model_type': model_type,
                'status': 'error',
                'error': error_msg,
                'traceback': traceback.format_exc()
            }
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Entrena todos los modelos del sistema.
        
        Returns:
            Dict con resultados de todos los entrenamientos
        """
        logger.info("="*80)
        logger.info("INICIANDO ENTRENAMIENTO COMPLETO DEL SISTEMA NBA")
        logger.info("="*80)
        
        all_results = {
            'player_models': {},
            'team_models': {},
            'summary': {},
            'errors': []
        }
        
        start_time = datetime.now()
        
        if self.enable_parallel:
            # Entrenamiento paralelo
            logger.info("Ejecutando entrenamiento paralelo")
            
            with ThreadPoolExecutor(max_workers=4) as executor:
                # Enviar trabajos de modelos de jugadores
                future_to_model = {}
                
                for model_name, config in self.player_models_config.items():
                    future = executor.submit(
                        self.train_single_model, 'player', model_name, config
                    )
                    future_to_model[future] = ('player', model_name)
                
                # Enviar trabajos de modelos de equipos
                for model_name, config in self.team_models_config.items():
                    future = executor.submit(
                        self.train_single_model, 'team', model_name, config
                    )
                    future_to_model[future] = ('team', model_name)
                
                # Recoger resultados
                for future in tqdm(as_completed(future_to_model), 
                                 total=len(future_to_model),
                                 desc="Entrenando modelos"):
                    model_type, model_name = future_to_model[future]
                    result = future.result()
                    
                    if model_type == 'player':
                        all_results['player_models'][model_name] = result
                    else:
                        all_results['team_models'][model_name] = result
                    
                    if result['status'] == 'error':
                        all_results['errors'].append(result)
        
        else:
            # Entrenamiento secuencial
            logger.info("Ejecutando entrenamiento secuencial")
            
            # Entrenar modelos de jugadores
            logger.info("Entrenando modelos de jugadores")
            for model_name, config in tqdm(self.player_models_config.items(), 
                                         desc="Modelos de jugadores"):
                result = self.train_single_model('player', model_name, config)
                all_results['player_models'][model_name] = result
                
                if result['status'] == 'error':
                    all_results['errors'].append(result)
            
            # Entrenar modelos de equipos
            logger.info("Entrenando modelos de equipos")
            for model_name, config in tqdm(self.team_models_config.items(),
                                         desc="Modelos de equipos"):
                result = self.train_single_model('team', model_name, config)
                all_results['team_models'][model_name] = result
                
                if result['status'] == 'error':
                    all_results['errors'].append(result)
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Generar resumen
        successful_models = []
        failed_models = []
        
        for model_type in ['player_models', 'team_models']:
            for model_name, result in all_results[model_type].items():
                if result['status'] == 'success':
                    successful_models.append(f"{model_name} ({model_type.replace('_models', '')})")
                else:
                    failed_models.append(f"{model_name} ({model_type.replace('_models', '')})")
        
        all_results['summary'] = {
            'total_duration': total_duration,
            'total_models': len(self.player_models_config) + len(self.team_models_config),
            'successful_models': len(successful_models),
            'failed_models': len(failed_models),
            'success_rate': len(successful_models) / (len(successful_models) + len(failed_models)) * 100,
            'successful_list': successful_models,
            'failed_list': failed_models
        }
        
        self.training_results = all_results
        
        logger.info("="*80)
        logger.info("RESUMEN DE ENTRENAMIENTO COMPLETO")
        logger.info("="*80)
        logger.info(f"Duración total: {total_duration:.1f} segundos")
        logger.info(f"Modelos exitosos: {len(successful_models)}/{len(successful_models) + len(failed_models)}")
        logger.info(f"Tasa de éxito: {all_results['summary']['success_rate']:.1f}%")
        
        if successful_models:
            logger.info("Modelos exitosos:")
            for model in successful_models:
                logger.info(f"  - {model}")
        
        if failed_models:
            logger.info("Modelos fallidos:")
            for model in failed_models:
                logger.info(f"  - {model}")
        
        return all_results
    
    def generate_consolidated_report(self):
        """
        Genera un reporte consolidado de todos los modelos.
        """
        logger.info("Generando reporte consolidado")
        
        report = {
            'system_info': {
                'timestamp': datetime.now().isoformat(),
                'total_models': len(self.player_models_config) + len(self.team_models_config),
                'parallel_training': self.enable_parallel,
                'n_trials': self.n_trials
            },
            'training_summary': self.training_results.get('summary', {}),
            'model_performance': {}
        }
        
        # Compilar rendimiento de modelos
        for model_type in ['player_models', 'team_models']:
            if model_type in self.training_results:
                for model_name, result in self.training_results[model_type].items():
                    if result['status'] == 'success':
                        report['model_performance'][model_name] = {
                            'type': model_type.replace('_models', ''),
                            'status': result['status'],
                            'training_duration': result.get('training_duration', None),
                            'output_dir': result.get('output_dir', '')
                        }
        
        # Guardar reporte
        report_path = os.path.join(self.output_base_dir, 'consolidated_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Reporte consolidado guardado en: {report_path}")
        
        # Mostrar resumen en consola
        logger.info("="*80)
        logger.info("REPORTE CONSOLIDADO DEL SISTEMA NBA")
        logger.info("="*80)
        logger.info(f"Modelos entrenados exitosamente: {report['training_summary'].get('successful_models', 0)}")
        logger.info(f"Tasa de éxito en entrenamiento: {report['training_summary'].get('success_rate', 0):.1f}%")
        logger.info(f"Resultados guardados en: {self.output_base_dir}")
        logger.info("="*80)
    
    def run_complete_system(self) -> Dict[str, Any]:
        """
        Ejecuta el sistema completo: entrenamiento + análisis avanzado.
        
        Returns:
            Dict con todos los resultados del sistema
        """
        logger.info("INICIANDO SISTEMA UNIFICADO NBA COMPLETO")
        
        system_start_time = datetime.now()
        
        # Paso 1: Entrenar todos los modelos
        logger.info("PASO 1: ENTRENAMIENTO DE MODELOS")
        training_results = self.train_all_models()
        
        # Paso 2: Reporte final
        logger.info("PASO 2: GENERACION DE REPORTES")
        self.generate_consolidated_report()
        
        total_system_duration = (datetime.now() - system_start_time).total_seconds()
        
        final_results = {
            'training_results': training_results,
            'system_duration': total_system_duration,
            'output_directory': self.output_base_dir
        }
        
        logger.info("="*80)
        logger.info("SISTEMA UNIFICADO NBA COMPLETADO EXITOSAMENTE")
        logger.info("="*80)
        logger.info(f"Duración total del sistema: {total_system_duration:.1f} segundos")
        logger.info(f"Resultados guardados en: {self.output_base_dir}")
        logger.info(f"Modelos entrenados: {training_results['summary']['successful_models']}")
        logger.info("="*80)
        
        return final_results

class NBAPredict:
    """
    Sistema Unificado de Predicción NBA - Punto de Inicialización
    
    Clase principal para recibir el insumo del juego (Game) y orquestar
    todas las predicciones del sistema NBA usando el UnifiedPredictor.
    """
    
    def __init__(self, auto_load: bool = True):
        """
        Inicializa el sistema de predicciones NBA.
        
        Args:
            auto_load: Si cargar automáticamente los modelos al inicializar
        """
        logger.info("🚀 Inicializando NBAPredict...")
        
        # Inicializar predictor unificado
        self.unified_predictor = UnifiedPredictor()
        self.predictors_loaded = False
        
        # Cargar modelos si se solicita
        if auto_load:
            self.load_models()
    
    def load_models(self) -> bool:
        """
        Carga todos los modelos del sistema.
        
        Returns:
            bool: True si todos los modelos se cargaron exitosamente
        """
        try:
            logger.info("📂 Cargando todos los modelos del sistema...")
            
            if self.unified_predictor.load_all_models():
                self.predictors_loaded = True
                logger.info("✅ Todos los modelos cargados exitosamente")
                return True
            else:
                logger.error("❌ Error cargando algunos modelos")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")
            return False
    
    def predict(self, data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Punto de entrada principal para predicciones.
        Maneja tanto un solo juego como múltiples juegos usando UnifiedPredictor.
        Espera datos ya convertidos a formato SportRadar.
        """
        try:
            # Validar que los modelos estén cargados
            if not self.predictors_loaded:
                logger.warning("⚠️ Modelos no cargados. Intentando cargar...")
                if not self.load_models():
                    return {'error': 'No se pudieron cargar los modelos', 'status': 'error'}
            
            # Usar el método unificado de UnifiedPredictor
            result = self.unified_predictor.predict(data)
            
            # Limpiar datos numpy para serialización JSON usando función existente
            return convert_numpy_types(result)
            
        except Exception as e:
            logger.error(f"❌ Error en predicción: {e}")
            return {'error': str(e), 'status': 'error'}
    
    


def main():
    """
    Función principal para ejecutar el sistema unificado NBA.
    """
    try:
        # Configuración del sistema
        system = NBATrainers(
            players_total_path="app/architectures/basketball/data/players_total.csv",
            players_quarters_path="app/architectures/basketball/data/players_quarters.csv",
            teams_total_path="app/architectures/basketball/data/teams_total.csv",
            teams_quarters_path="app/architectures/basketball/data/teams_quarters.csv",
            biometrics_path="app/architectures/basketball/data/biometrics.csv",
            output_base_dir="app/architectures/basketball/results",
            n_trials=50,
            enable_parallel=True,
            enable_advanced_analysis=True,
            random_state=42
        )
        
        # Ejecutar sistema completo
        results = system.run_complete_system()
        
        return results
        
    except KeyboardInterrupt:
        logger.info("Sistema interrumpido por el usuario")
        return None
    except Exception as e:
        logger.error(f"Error crítico en el sistema: {str(e)}")
        logger.error(traceback.format_exc())
        return None


if __name__ == "__main__":
    results = main()
    if results:
        print("\nSistema ejecutado exitosamente")
        print(f"Revisa los resultados en: {results['output_directory']}")
    else:
        print("\nSistema falló o fue interrumpido")
        sys.exit(1) 