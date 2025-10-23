# app/main.py

from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError

import logging
import traceback
import time

from app.core.logging import configure_logging
from app.core.config import settings
from app.api import api_router  # nuestro agregador de routers automáticos
from app.core.redis_config import RedisClient

# 1) Carga variables de entorno
load_dotenv()

# 2) Configura logging centralizado
configure_logging()
logger = logging.getLogger(__name__)

# 3) Define el lifespan para startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestiona los recursos durante el ciclo de vida de la aplicación
    
    Inicializa las conexiones a BD, modelos de IA y otros recursos cuando
    la aplicación arranca, y asegura que se liberen correctamente al finalizar.
    """
    # Inicio de recursos
    logger.info(" Startup: cargando recursos (modelos, BD...)")
    start_time = time.time()
    
    # Aquí podrías, por ejemplo:
    # await some_database.connect()
    # model = await load_model(settings.MODEL_PATH)

    logger.info(f"Recursos cargados en {time.time() - start_time:.2f} segundos")
    yield  # La aplicación se ejecuta aquí

    # Limpieza de recursos
    logger.info(" Shutdown: liberando recursos")
    # await some_database.disconnect()

# 4) Instancia FastAPI usando lifespan (no on_event)
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# 5) Middlewares transversales
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6) Manejo global de excepciones
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Captura excepciones no manejadas para evitar caídas del servicio"""
    logger.error(
        f"Error no capturado en {request.url.path}: {str(exc)}", 
        exc_info=True
    )
    
    # En modo debug, incluir el traceback completo
    error_detail = "Error interno del servidor"
    if settings.DEBUG:
        error_detail = f"{str(exc)}\n{''.join(traceback.format_tb(exc.__traceback__))}"
        
    return JSONResponse(
        status_code=500,
        content={"detail": error_detail},
    )

# Manejador específico para errores de validación
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Mejora los mensajes de error de validación de datos"""
    logger.warning(f"Error de validación: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Error de validación en los datos enviados",
            "errors": exc.errors(),
        },
    )

# 7) Middleware para registro de tiempos de respuesta
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Registra el tiempo de procesamiento de cada petición"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    if process_time > 1.0:  # Alerta si tarda más de 1 segundo
        logger.warning(f"Petición lenta ({process_time:.2f}s): {request.method} {request.url.path}")
    return response

# 8) Monta automáticamente todos los routers desde app/api/routers/
app.include_router(api_router, prefix=settings.API_PREFIX)

# 9) Endpoint raíz para chequeo rápido
@app.get("/", tags=["Root"])
async def read_root():
    """Endpoint principal para verificar que la API está funcionando"""
    return {
        "message": f"{settings.PROJECT_NAME} v{settings.VERSION} ejecutándose ",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs"
    }
