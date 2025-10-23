import redis
from typing import Optional

class RedisClient:
    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
   
    def connect(self):
        """Conectar a Redis"""
        self.redis_client = redis.Redis(
            host="173.212.203.175",
            port=16379,
            db=0,
            password="q7Vx9rT2bJm4NcYzLwKdQ3AeRuHpXsCg",  # si tienes contraseña
            decode_responses=True  # Para obtener strings en lugar de bytes
        )
        # Probar conexión
        self.redis_client.ping()
        print("OK Conectado a Redis exitosamente")
   
    def disconnect(self):
        """Cerrar conexión"""
        if self.redis_client:
            self.redis_client.close()
            print("❌ Desconectado de Redis")
   
    def get_client(self) -> redis.Redis:
        """Obtener cliente Redis"""
        if not self.redis_client:
            self.connect()
        return self.redis_client
