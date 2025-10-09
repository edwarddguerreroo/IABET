"""
Excepciones Personalizadas del Módulo Bookmakers
===============================================

Excepciones específicas para manejo de errores en APIs,
análisis de datos y operaciones de betting.
"""

from typing import Optional, Dict, Any


class BookmakersAPIError(Exception):
    """Excepción base para errores del módulo bookmakers."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 details: Optional[Dict[str, Any]] = None):
        """
        Inicializa la excepción.
        
        Args:
            message: Mensaje de error
            error_code: Código de error específico
            details: Detalles adicionales del error
        """
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self) -> str:
        base_msg = super().__str__()
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            base_msg = f"{base_msg} | {details_str}"
        return base_msg


class SportradarAPIError(BookmakersAPIError):
    """Excepción específica para errores de la API de Sportradar."""
    
    def __init__(self, message: str, status_code: Optional[int] = None,
                 response_data: Optional[Dict] = None, endpoint: Optional[str] = None):
        """
        Inicializa la excepción de Sportradar.
        
        Args:
            message: Mensaje de error
            status_code: Código de estado HTTP
            response_data: Datos de respuesta de la API
            endpoint: Endpoint que causó el error
        """
        details = {}
        if status_code:
            details['status_code'] = status_code
        if endpoint:
            details['endpoint'] = endpoint
        if response_data:
            details['response'] = str(response_data)[:200]  # Limitar longitud
        
        super().__init__(message, f"SPORTRADAR_{status_code}", details)
        self.status_code = status_code
        self.response_data = response_data
        self.endpoint = endpoint


class RateLimitError(SportradarAPIError):
    """Excepción para errores de límite de tasa de API."""
    
    def __init__(self, retry_after: Optional[int] = None):
        message = "Límite de tasa de API excedido"
        if retry_after:
            message += f". Reintentar después de {retry_after} segundos"
        
        super().__init__(message, 429)
        self.retry_after = retry_after


class AuthenticationError(SportradarAPIError):
    """Excepción para errores de autenticación."""
    
    def __init__(self, message: str = "Error de autenticación con la API"):
        super().__init__(message, 401)


class InsufficientDataError(BookmakersAPIError):
    """Excepción para cuando no hay suficientes datos para análisis."""
    
    def __init__(self, message: str, data_type: Optional[str] = None,
                 minimum_required: Optional[int] = None, available: Optional[int] = None):
        """
        Inicializa la excepción de datos insuficientes.
        
        Args:
            message: Mensaje de error
            data_type: Tipo de datos insuficientes
            minimum_required: Cantidad mínima requerida
            available: Cantidad disponible
        """
        details = {}
        if data_type:
            details['data_type'] = data_type
        if minimum_required is not None:
            details['minimum_required'] = minimum_required
        if available is not None:
            details['available'] = available
        
        super().__init__(message, "INSUFFICIENT_DATA", details)
        self.data_type = data_type
        self.minimum_required = minimum_required
        self.available = available


class ConfigurationError(BookmakersAPIError):
    """Excepción para errores de configuración."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        """
        Inicializa la excepción de configuración.
        
        Args:
            message: Mensaje de error
            config_key: Clave de configuración problemática
        """
        details = {}
        if config_key:
            details['config_key'] = config_key
        
        super().__init__(message, "CONFIG_ERROR", details)
        self.config_key = config_key


class DataValidationError(BookmakersAPIError):
    """Excepción para errores de validación de datos."""
    
    def __init__(self, message: str, field: Optional[str] = None, 
                 expected_type: Optional[str] = None, actual_value: Optional[Any] = None):
        """
        Inicializa la excepción de validación.
        
        Args:
            message: Mensaje de error
            field: Campo que falló la validación
            expected_type: Tipo esperado
            actual_value: Valor actual
        """
        details = {}
        if field:
            details['field'] = field
        if expected_type:
            details['expected_type'] = expected_type
        if actual_value is not None:
            details['actual_value'] = str(actual_value)[:100]
        
        super().__init__(message, "VALIDATION_ERROR", details)
        self.field = field
        self.expected_type = expected_type
        self.actual_value = actual_value


class BettingAnalysisError(BookmakersAPIError):
    """Excepción para errores en análisis de apuestas."""
    
    def __init__(self, message: str, analysis_type: Optional[str] = None,
                 target: Optional[str] = None):
        """
        Inicializa la excepción de análisis.
        
        Args:
            message: Mensaje de error
            analysis_type: Tipo de análisis que falló
            target: Target de predicción
        """
        details = {}
        if analysis_type:
            details['analysis_type'] = analysis_type
        if target:
            details['target'] = target
        
        super().__init__(message, "ANALYSIS_ERROR", details)
        self.analysis_type = analysis_type
        self.target = target


class CacheError(BookmakersAPIError):
    """Excepción para errores de cache."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None):
        """
        Inicializa la excepción de cache.
        
        Args:
            message: Mensaje de error
            cache_key: Clave de cache problemática
            operation: Operación que falló (read, write, delete)
        """
        details = {}
        if cache_key:
            details['cache_key'] = cache_key
        if operation:
            details['operation'] = operation
        
        super().__init__(message, "CACHE_ERROR", details)
        self.cache_key = cache_key
        self.operation = operation


class NetworkError(BookmakersAPIError):
    """Excepción para errores de red."""
    
    def __init__(self, message: str, url: Optional[str] = None,
                 timeout: Optional[float] = None):
        """
        Inicializa la excepción de red.
        
        Args:
            message: Mensaje de error
            url: URL que causó el error
            timeout: Timeout usado
        """
        details = {}
        if url:
            details['url'] = url
        if timeout:
            details['timeout'] = timeout
        
        super().__init__(message, "NETWORK_ERROR", details)
        self.url = url
        self.timeout = timeout


# Mapping de códigos de estado HTTP a excepciones
HTTP_ERROR_MAPPING = {
    400: DataValidationError,
    401: AuthenticationError,
    403: AuthenticationError,
    404: SportradarAPIError,
    429: RateLimitError,
    500: SportradarAPIError,
    502: NetworkError,
    503: NetworkError,
    504: NetworkError
}


def create_http_error(status_code: int, message: str, **kwargs) -> BookmakersAPIError:
    """
    Crea una excepción apropiada basada en el código de estado HTTP.
    
    Args:
        status_code: Código de estado HTTP
        message: Mensaje de error
        **kwargs: Argumentos adicionales para la excepción
        
    Returns:
        Instancia de excepción apropiada
    """
    error_class = HTTP_ERROR_MAPPING.get(status_code, SportradarAPIError)
    
    if error_class == RateLimitError:
        return error_class(kwargs.get('retry_after'))
    elif error_class == AuthenticationError:
        return error_class(message)
    elif error_class in (DataValidationError, NetworkError):
        return error_class(message, **kwargs)
    else:
        return error_class(message, status_code, **kwargs) 