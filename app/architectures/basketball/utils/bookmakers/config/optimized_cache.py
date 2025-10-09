"""
Sistema de Cache Optimizado para Sportradar API
==============================================

Implementa un sistema de cache inteligente que combina:
- Cache en memoria para acceso ultra-rápido
- Cache en disco para persistencia
- Compresión de datos para optimizar espacio
- Limpieza automática y gestión de memoria
- Estadísticas de rendimiento

Optimizado específicamente para las necesidades del sistema NBA.
"""

import os
import json
import pickle
import gzip
import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import time

logger = logging.getLogger(__name__)


class OptimizedCache:
    """
    Sistema de cache optimizado para APIs de Sportradar.
    
    Características:
    - Cache híbrido memoria + disco
    - Compresión automática
    - TTL por entrada
    - Limpieza automática
    - Estadísticas de rendimiento
    - Thread-safe
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Inicializa el sistema de cache optimizado.
        
        Args:
            config: Configuración del cache desde BookmakersConfig
        """
        self.config = config
        self.cache_enabled = config.get('cache_enabled', True)
        self.cache_type = config.get('cache_type', 'memory_disk')
        self.default_ttl = config.get('cache_duration', 300)  # 5 minutos
        self.max_size_mb = config.get('cache_max_size_mb', 100)
        self.cache_dir = Path(config.get('cache_directory', 'data/cache/bookmakers'))
        self.compression_enabled = config.get('cache_compression', True)
        self.persistence_enabled = config.get('cache_persistence', True)
        self.max_entries = config.get('cache_max_entries', 10000)
        self.stats_enabled = config.get('cache_stats_enabled', True)
        
        # Cache en memoria
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        
        # Lock para thread safety
        self._lock = threading.RLock()
        
        # Estadísticas de rendimiento
        self._stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'writes': 0,
            'disk_writes': 0,
            'evictions': 0,
            'errors': 0,
            'total_size_bytes': 0,
            'last_cleanup': None
        } if self.stats_enabled else {}
        
        # Crear directorio de cache si no existe
        if self.persistence_enabled:
            self._setup_cache_directory()
        
        # Cargar cache persistente al inicializar
        if self.persistence_enabled and self.cache_enabled:
            self._load_persistent_cache()
        
        logger.info(f"OptimizedCache inicializada | Tipo: {self.cache_type} | "
                   f"TTL: {self.default_ttl}s | Max: {self.max_size_mb}MB")
    
    def _setup_cache_directory(self):
        """Configura el directorio de cache en disco."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Directorio de cache configurado: {self.cache_dir}")
        except Exception as e:
            logger.error(f"Error configurando directorio cache: {e}")
            self.persistence_enabled = False
    
    def _generate_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """
        Genera una clave única para el cache.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            
        Returns:
            Clave única de cache
        """
        # Crear string determinístico de parámetros
        params_str = json.dumps(params, sort_keys=True, separators=(',', ':'))
        
        # Combinar endpoint y parámetros
        combined = f"{endpoint}:{params_str}"
        
        # Generar hash SHA256 para clave única
        cache_key = hashlib.sha256(combined.encode('utf-8')).hexdigest()[:32]
        
        return f"sr_{cache_key}"
    
    def _get_disk_path(self, cache_key: str) -> Path:
        """Obtiene la ruta del archivo en disco para una clave."""
        # Organizar archivos en subdirectorios para mejor rendimiento
        subdir = cache_key[:2]
        disk_dir = self.cache_dir / subdir
        disk_dir.mkdir(exist_ok=True)
        
        extension = '.gz' if self.compression_enabled else '.pkl'
        return disk_dir / f"{cache_key}{extension}"
    
    def _serialize_data(self, data: Any) -> bytes:
        """
        Serializa datos para almacenamiento.
        
        Args:
            data: Datos a serializar
            
        Returns:
            Datos serializados en bytes
        """
        try:
            # Serializar con pickle
            serialized = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Comprimir si está habilitado
            if self.compression_enabled:
                serialized = gzip.compress(serialized)
            
            return serialized
        except Exception as e:
            logger.error(f"Error serializando datos: {e}")
            raise
    
    def _deserialize_data(self, data: bytes) -> Any:
        """
        Deserializa datos desde almacenamiento.
        
        Args:
            data: Datos serializados
            
        Returns:
            Datos deserializados
        """
        try:
            # Descomprimir si es necesario
            if self.compression_enabled:
                data = gzip.decompress(data)
            
            # Deserializar con pickle
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error deserializando datos: {e}")
            raise
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Verifica si una entrada del cache ha expirado.
        
        Args:
            entry: Entrada del cache con metadata
            
        Returns:
            True si ha expirado, False en caso contrario
        """
        ttl = entry.get('ttl', self.default_ttl)
        created_at = entry.get('created_at', 0)
        return (time.time() - created_at) > ttl
    
    def _update_stats(self, operation: str, **kwargs):
        """Actualiza estadísticas de rendimiento."""
        if not self.stats_enabled:
            return
        
        with self._lock:
            if operation in self._stats:
                self._stats[operation] += 1
            
            # Actualizar estadísticas adicionales
            for key, value in kwargs.items():
                if key in self._stats:
                    self._stats[key] += value
    
    def get(self, endpoint: str, params: Dict[str, Any], 
           ttl: Optional[int] = None) -> Optional[Any]:
        """
        Obtiene datos del cache.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            ttl: TTL personalizado para esta entrada
            
        Returns:
            Datos en cache o None si no existe/expirado
        """
        if not self.cache_enabled:
            return None
        
        cache_key = self._generate_cache_key(endpoint, params)
        
        with self._lock:
            try:
                # 1. Buscar en cache de memoria primero
                if cache_key in self._memory_cache:
                    entry = self._memory_cache[cache_key]
                    
                    if not self._is_expired(entry):
                        self._update_stats('hits', memory_hits=1)
                        logger.debug(f"Cache hit (memoria): {cache_key[:16]}...")
                        return entry['data']
                    else:
                        # Eliminar entrada expirada
                        del self._memory_cache[cache_key]
                
                # 2. Buscar en cache de disco si está habilitado
                if self.persistence_enabled and self.cache_type in ['disk', 'memory_disk']:
                    disk_path = self._get_disk_path(cache_key)
                    
                    if disk_path.exists():
                        try:
                            with open(disk_path, 'rb') as f:
                                entry_data = f.read()
                            
                            entry = self._deserialize_data(entry_data)
                            
                            if not self._is_expired(entry):
                                # Cargar de vuelta a memoria para próximos accesos
                                if self.cache_type == 'memory_disk':
                                    self._memory_cache[cache_key] = entry
                                
                                self._update_stats('hits', disk_hits=1)
                                logger.debug(f"Cache hit (disco): {cache_key[:16]}...")
                                return entry['data']
                            else:
                                # Eliminar archivo expirado
                                disk_path.unlink(missing_ok=True)
                        
                        except Exception as e:
                            logger.error(f"Error leyendo cache disco {cache_key}: {e}")
                            disk_path.unlink(missing_ok=True)  # Eliminar archivo corrupto
                
                # No encontrado o expirado
                self._update_stats('misses')
                return None
                
            except Exception as e:
                logger.error(f"Error obteniendo del cache {cache_key}: {e}")
                self._update_stats('errors')
                return None
    
    def set(self, endpoint: str, params: Dict[str, Any], data: Any, 
           ttl: Optional[int] = None) -> bool:
        """
        Almacena datos en el cache.
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            data: Datos a almacenar
            ttl: TTL personalizado para esta entrada
            
        Returns:
            True si se almacenó correctamente, False en caso contrario
        """
        if not self.cache_enabled:
            return False
        
        cache_key = self._generate_cache_key(endpoint, params)
        ttl = ttl or self.default_ttl
        
        with self._lock:
            try:
                # Crear entrada de cache
                entry = {
                    'data': data,
                    'created_at': time.time(),
                    'ttl': ttl,
                    'endpoint': endpoint,
                    'size_bytes': len(str(data))  # Estimación aproximada
                }
                
                # Verificar límites de memoria antes de almacenar
                self._enforce_memory_limits()
                
                # 1. Almacenar en memoria
                if self.cache_type in ['memory', 'memory_disk']:
                    self._memory_cache[cache_key] = entry
                    self._update_stats('writes')
                
                # 2. Almacenar en disco si está habilitado
                if (self.persistence_enabled and 
                    self.cache_type in ['disk', 'memory_disk']):
                    
                    try:
                        disk_path = self._get_disk_path(cache_key)
                        serialized_entry = self._serialize_data(entry)
                        
                        with open(disk_path, 'wb') as f:
                            f.write(serialized_entry)
                        
                        self._update_stats('disk_writes')
                        
                    except Exception as e:
                        logger.error(f"Error escribiendo cache disco {cache_key}: {e}")
                
                logger.debug(f"Cache set: {cache_key[:16]}... | TTL: {ttl}s")
                return True
                
            except Exception as e:
                logger.error(f"Error almacenando en cache {cache_key}: {e}")
                self._update_stats('errors')
                return False
    
    def _enforce_memory_limits(self):
        """Aplica límites de memoria y elimina entradas antiguas si es necesario."""
        if len(self._memory_cache) <= self.max_entries:
            return
        
        # Ordenar por tiempo de creación (más antiguo primero)
        sorted_entries = sorted(
            self._memory_cache.items(),
            key=lambda x: x[1]['created_at']
        )
        
        # Eliminar 20% de las entradas más antiguas
        entries_to_remove = len(sorted_entries) // 5
        
        for cache_key, _ in sorted_entries[:entries_to_remove]:
            del self._memory_cache[cache_key]
            self._update_stats('evictions')
        
        logger.debug(f"Cache memory cleanup: removidas {entries_to_remove} entradas")
    
    def _load_persistent_cache(self):
        """Carga cache persistente desde disco al inicializar."""
        if not self.cache_dir.exists():
            return
        
        try:
            loaded_count = 0
            
            for cache_file in self.cache_dir.rglob('sr_*'):
                if cache_file.is_file():
                    try:
                        with open(cache_file, 'rb') as f:
                            entry_data = f.read()
                        
                        entry = self._deserialize_data(entry_data)
                        
                        # Solo cargar si no ha expirado
                        if not self._is_expired(entry):
                            cache_key = cache_file.stem.replace('.gz', '').replace('.pkl', '')
                            
                            if self.cache_type == 'memory_disk':
                                self._memory_cache[cache_key] = entry
                            
                            loaded_count += 1
                        else:
                            # Eliminar archivo expirado
                            cache_file.unlink(missing_ok=True)
                    
                    except Exception as e:
                        logger.debug(f"Error cargando cache {cache_file}: {e}")
                        cache_file.unlink(missing_ok=True)  # Eliminar archivo corrupto
            
            if loaded_count > 0:
                logger.info(f"Cache persistente cargado: {loaded_count} entradas")
                
        except Exception as e:
            logger.error(f"Error cargando cache persistente: {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """
        Limpia el cache.
        
        Args:
            pattern: Patrón opcional para limpiar solo ciertas entradas
        """
        with self._lock:
            if pattern:
                # Limpiar entradas que coincidan con el patrón
                keys_to_remove = [
                    key for key in self._memory_cache.keys()
                    if pattern in key
                ]
                for key in keys_to_remove:
                    del self._memory_cache[key]
                
                logger.info(f"Cache limpiado: {len(keys_to_remove)} entradas con patrón '{pattern}'")
            else:
                # Limpiar todo
                self._memory_cache.clear()
                
                # Limpiar archivos en disco
                if self.persistence_enabled:
                    try:
                        for cache_file in self.cache_dir.rglob('sr_*'):
                            cache_file.unlink(missing_ok=True)
                    except Exception as e:
                        logger.error(f"Error limpiando cache disco: {e}")
                
                logger.info("Cache completamente limpiado")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del cache.
        
        Returns:
            Diccionario con estadísticas de rendimiento
        """
        if not self.stats_enabled:
            return {}
        
        with self._lock:
            stats = self._stats.copy()
            
            # Agregar estadísticas adicionales
            stats.update({
                'memory_entries': len(self._memory_cache),
                'hit_ratio': (stats['hits'] / (stats['hits'] + stats['misses'])) * 100 
                           if (stats['hits'] + stats['misses']) > 0 else 0,
                'memory_hit_ratio': (stats['memory_hits'] / stats['hits']) * 100 
                                   if stats['hits'] > 0 else 0,
                'cache_enabled': self.cache_enabled,
                'cache_type': self.cache_type,
                'max_entries': self.max_entries,
                'compression_enabled': self.compression_enabled
            })
            
            return stats
    
    def cleanup_expired(self):
        """Limpia entradas expiradas del cache."""
        with self._lock:
            # Limpiar memoria
            expired_keys = [
                key for key, entry in self._memory_cache.items()
                if self._is_expired(entry)
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
            
            # Limpiar disco
            if self.persistence_enabled:
                try:
                    disk_expired = 0
                    for cache_file in self.cache_dir.rglob('sr_*'):
                        try:
                            with open(cache_file, 'rb') as f:
                                entry_data = f.read()
                            
                            entry = self._deserialize_data(entry_data)
                            
                            if self._is_expired(entry):
                                cache_file.unlink(missing_ok=True)
                                disk_expired += 1
                        
                        except Exception:
                            # Archivo corrupto, eliminar
                            cache_file.unlink(missing_ok=True)
                            disk_expired += 1
                    
                    logger.debug(f"Cleanup: {len(expired_keys)} memoria, {disk_expired} disco")
                
                except Exception as e:
                    logger.error(f"Error en cleanup disco: {e}")
            
            # Actualizar estadísticas
            self._stats['last_cleanup'] = datetime.now().isoformat()
    
    def get_size_info(self) -> Dict[str, Any]:
        """
        Obtiene información sobre el tamaño del cache.
        
        Returns:
            Información de tamaño y uso
        """
        size_info = {
            'memory_entries': len(self._memory_cache),
            'estimated_memory_mb': 0,
            'disk_files': 0,
            'disk_size_mb': 0
        }
        
        # Estimar tamaño en memoria
        try:
            memory_size = sum(entry.get('size_bytes', 0) for entry in self._memory_cache.values())
            size_info['estimated_memory_mb'] = memory_size / (1024 * 1024)
        except Exception:
            pass
        
        # Calcular tamaño en disco
        if self.persistence_enabled and self.cache_dir.exists():
            try:
                disk_size = 0
                file_count = 0
                
                for cache_file in self.cache_dir.rglob('sr_*'):
                    if cache_file.is_file():
                        disk_size += cache_file.stat().st_size
                        file_count += 1
                
                size_info['disk_files'] = file_count
                size_info['disk_size_mb'] = disk_size / (1024 * 1024)
            
            except Exception as e:
                logger.error(f"Error calculando tamaño disco: {e}")
        
        return size_info 