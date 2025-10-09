#!/usr/bin/env python3
"""
HELPERS CENTRALIZADOS - Utilidades para JSON y tipos NumPy
========================================================

Funciones centralizadas para manejo seguro de JSON y conversión de tipos NumPy.
"""

import json
import numpy as np
from typing import Any, Dict, List, Union, Optional

def convert_numpy_types(obj: Any) -> Any:
    """
    Convierte recursivamente todos los tipos NumPy a tipos nativos de Python.
    
    Args:
        obj: Objeto que puede contener tipos NumPy
        
    Returns:
        Objeto con tipos NumPy convertidos a tipos nativos de Python
        
    Examples:
        >>> convert_numpy_types(np.int64(5))
        5
        >>> convert_numpy_types(np.float32(3.14))
        3.14
        >>> convert_numpy_types({'value': np.array([1, 2, 3])})
        {'value': [1, 2, 3]}
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Serializa un objeto a JSON con configuración automática para Unicode y tipos NumPy.
    
    Args:
        obj: Objeto a serializar
        **kwargs: Argumentos adicionales para json.dumps
        
    Returns:
        String JSON serializado
        
    Examples:
        >>> safe_json_dumps({'name': 'José', 'value': np.int64(5)})
        '{"name": "José", "value": 5}'
    """
    default_kwargs = {
        'ensure_ascii': False,
        'indent': 2
    }
    final_kwargs = {**default_kwargs, **kwargs}
    clean_obj = convert_numpy_types(obj)
    return json.dumps(clean_obj, **final_kwargs)

def safe_json_response(obj: Any, **kwargs) -> Dict[str, Any]:
    """
    Prepara un objeto para respuestas JSON en APIs de producción (FastAPI).
    
    Nota: FastAPI maneja automáticamente la serialización JSON, pero esta función
    limpia los tipos NumPy para evitar errores de serialización.
    
    Args:
        obj: Objeto a preparar
        **kwargs: Argumentos adicionales (no usados en FastAPI)
        
    Returns:
        Objeto con tipos NumPy convertidos a tipos nativos
        
    Examples:
        >>> safe_json_response({'value': np.int64(5)})
        {'value': 5}
    """
    return convert_numpy_types(obj)
