import logging
import sys

from app.core.config import settings

def configure_logging():
    """Configura el sistema de logging centralizado
    
    Establece una configuración unificada para todos los loggers de la aplicación
    con salida exclusivamente a consola.
    """
    # Determinar nivel de logging desde configuración
    log_level_name = settings.LOG_LEVEL.upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    
    # Formato detallado para los logs
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # Configuración básica para salida a consola
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    # Reducir verbosidad de algunos loggers externos
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    
    # Unificar el logging de Uvicorn con nuestra configuración
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logger = logging.getLogger(logger_name)
        logger.handlers = logging.root.handlers
        logger.setLevel(log_level)
    
    # Log inicial confirmando la configuración
    logging.getLogger(__name__).info(
        f"Sistema de logging configurado (nivel={log_level_name}, solo en consola)"
    )
