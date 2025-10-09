"""
Sistema de Logging Unificado NBA
===============================

Configuración centralizada de logging para todos los modelos NBA.
Diseñado para ser eficiente, consistente y sin verbosidad excesiva.

Características:
- Formato uniforme para todos los modelos
- Niveles de logging optimizados
- Sin emojis ni caracteres especiales
- Logging eficiente para orquestación
- Configuración centralizada
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class NBALogger:
    """Sistema de logging unificado para modelos NBA"""
    
    _loggers = {}
    _configured = False
    
    @classmethod
    def get_logger(cls, name: str, level: str = "INFO") -> logging.Logger:
        """
        Obtener logger configurado de forma uniforme.
        
        Args:
            name: Nombre del logger
            level: Nivel de logging (INFO, WARNING, ERROR)
            
        Returns:
            Logger configurado
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Configurar sistema de logging si no está configurado
        if not cls._configured:
            cls._setup_logging_system()
            cls._configured = True
        
        # Crear logger específico
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.propagate = False
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _setup_logging_system(cls):
        """Configurar sistema de logging centralizado"""
        
        # Formato unificado sin emojis
        formatter = logging.Formatter(
            fmt='%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Handler para consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Handler para archivo de errores
        log_dir = Path("app/architectures/basketball/logs")
        log_dir.mkdir(exist_ok=True)
        
        error_handler = logging.FileHandler(
            log_dir / f"nba_errors_{datetime.now().strftime('%Y%m%d')}.log",
            mode='a',
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        
        # Configurar logger raíz
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(error_handler)
        root_logger.setLevel(logging.INFO)
        
        # Silenciar librerías externas
        external_libs = [
            'sklearn', 'xgboost', 'lightgbm', 'catboost', 
            'optuna', 'matplotlib', 'seaborn', 'torch'
        ]
        for lib in external_libs:
            logging.getLogger(lib).setLevel(logging.ERROR)
    
    @classmethod
    def log_model_start(cls, logger: logging.Logger, model_name: str, model_type: str):
        """Log estandarizado para inicio de modelo"""
        logger.info(f"Iniciando {model_name} ({model_type})")
    
    @classmethod
    def log_model_complete(cls, logger: logging.Logger, model_name: str, 
                          duration: float, metrics: Dict[str, float]):
        """Log estandarizado para finalización de modelo"""
        metric_str = " | ".join([f"{k}={v:.3f}" for k, v in metrics.items()])
        logger.info(f"Completado {model_name} | Duración: {duration:.1f}s | {metric_str}")
    
    @classmethod
    def log_data_info(cls, logger: logging.Logger, data_info: Dict[str, Any]):
        """Log estandarizado para información de datos"""
        logger.info(f"Datos cargados: {data_info.get('total_records', 0)} registros")
        if 'date_range' in data_info:
            logger.info(f"Rango temporal: {data_info['date_range']}")
    
    @classmethod
    def log_training_progress(cls, logger: logging.Logger, phase: str, 
                             progress: Optional[str] = None):
        """Log estandarizado para progreso de entrenamiento"""
        if progress:
            logger.info(f"{phase}: {progress}")
        else:
            logger.info(f"{phase}")
    
    @classmethod
    def log_optimization_result(cls, logger: logging.Logger, method: str, 
                               best_score: float, trials: int):
        """Log estandarizado para resultados de optimización"""
        logger.info(f"Optimización {method} completada | Mejor score: {best_score:.4f} | Trials: {trials}")
    
    @classmethod
    def log_validation_results(cls, logger: logging.Logger, cv_results: Dict[str, float]):
        """Log estandarizado para resultados de validación cruzada"""
        results_str = " | ".join([f"{k}={v:.3f}" for k, v in cv_results.items()])
        logger.info(f"Validación cruzada: {results_str}")
    
    @classmethod
    def log_feature_selection(cls, logger: logging.Logger, original: int, 
                             selected: int, method: str):
        """Log estandarizado para selección de features"""
        logger.info(f"Selección de features ({method}): {original} -> {selected}")
    
    @classmethod
    def log_error(cls, logger: logging.Logger, error_msg: str, context: str = ""):
        """Log estandarizado para errores"""
        if context:
            logger.error(f"Error en {context}: {error_msg}")
        else:
            logger.error(f"Error: {error_msg}")
    
    @classmethod
    def log_warning(cls, logger: logging.Logger, warning_msg: str):
        """Log estandarizado para warnings"""
        logger.warning(f"Advertencia: {warning_msg}")


class TrainingProgressLogger:
    """Logger especializado para progreso de entrenamiento sin verbosidad"""
    
    def __init__(self, model_name: str, total_epochs: Optional[int] = None):
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.logger = NBALogger.get_logger(f"training.{model_name}")
        self.last_log_time = datetime.now()
        self.log_interval = 30  # Segundos entre logs de progreso
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                  is_best: bool = False):
        """Log de época con throttling para evitar spam"""
        current_time = datetime.now()
        time_diff = (current_time - self.last_log_time).total_seconds()
        
        # Solo log cada cierto intervalo o si es el mejor modelo
        if time_diff >= self.log_interval or is_best or epoch == 1:
            status = "MEJOR" if is_best else ""
            self.logger.info(f"Época {epoch}: Train={train_loss:.4f} | Val={val_loss:.4f} {status}")
            self.last_log_time = current_time
    
    def log_early_stopping(self, epoch: int, reason: str):
        """Log para early stopping"""
        self.logger.info(f"Early stopping en época {epoch}: {reason}")
    
    def log_training_complete(self, final_epoch: int, best_score: float):
        """Log para finalización de entrenamiento"""
        self.logger.info(f"Entrenamiento completado | Épocas: {final_epoch} | Mejor score: {best_score:.4f}")


def configure_model_logging(model_name: str, verbose: bool = False) -> logging.Logger:
    """
    Configurar logging para un modelo específico.
    
    Args:
        model_name: Nombre del modelo
        verbose: Si habilitar logging verboso (solo para debugging)
        
    Returns:
        Logger configurado
    """
    level = "DEBUG" if verbose else "INFO"
    return NBALogger.get_logger(f"model.{model_name}", level)


def configure_trainer_logging(trainer_name: str) -> logging.Logger:
    """
    Configurar logging para un trainer específico.
    
    Args:
        trainer_name: Nombre del trainer
        
    Returns:
        Logger configurado
    """
    return NBALogger.get_logger(f"trainer.{trainer_name}")


def configure_system_logging() -> logging.Logger:
    """
    Configurar logging para el sistema principal.
    
    Returns:
        Logger configurado para el sistema
    """
    return NBALogger.get_logger("system.main") 