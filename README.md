# API_MODELS_IA

API para la integración de modelos de Inteligencia Artificial de predicción deportiva.

![FastAPI](https://img.shields.io/badge/FastAPI-0.95.2-009688.svg?style=flat&logo=fastapi)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?style=flat&logo=python&logoColor=white)
![Licencia: MIT](https://img.shields.io/badge/Licencia-MIT-yellow.svg)

## Descripción

Este proyecto implementa una API REST moderna para servir modelos de inteligencia artificial enfocados en predicciones deportivas. Utiliza FastAPI como framework web, proporcionando validación automática de datos, documentación interactiva y un alto rendimiento gracias a sus características asíncronas.

### Características

- ✅ Arquitectura modular y escalable
- ✅ Validación de datos con Pydantic
- ✅ Documentación OpenAPI automática
- ✅ Sistema de logging centralizado
- ✅ Manejo de excepciones global
- ✅ Carga automática de módulos (routers)

## Requisitos

- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Entorno virtual (venv o conda)

## Instalación

### 1. Clonar el repositorio

```bash
git clone https://github.com/Alvaro747/API_MODELS_IA.git
cd API_MODELS_IA
```

### 2. Crear y activar entorno virtual

```bash
# En Windows (PowerShell)
python -m venv venv
venv\Scripts\activate

# En macOS/Linux/GitBash
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
source venv/Scripts/activate  # GitBash en Windows
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configuración del entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables (ajusta según tu entorno):

```env
ENVIRONMENT=development
LOG_LEVEL=INFO
# MODEL_PATH=/ruta/a/tu/modelo (opcional)
```

## Ejecución

### Modo desarrollo

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en http://localhost:8000

- Documentación interactiva: http://localhost:8000/docs
- Documentación alternativa: http://localhost:8000/redoc

### Modo producción

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Estructura del Proyecto

```
API_MODELS_IA/
├── app/
│   ├── api/
│   │   ├── routers/      # Endpoints organizados por recurso
│   │   │   ├── health.py
│   │   │   └── prediction.py
│   │   ├── __init__.py   # Autoregistro de routers
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py     # Configuración centralizada
│   │   └── logging.py    # Configuración de logs
│   ├── models/
│   │   ├── game.py       # Modelos de datos (Pydantic)
│   │   └── prediction.py
│   ├── services/
│   │   └── model_service.py  # Lógica de negocio
│   ├── utils/
│   │   └── helpers.py
│   └── main.py           # Punto de entrada de la aplicación
├── tests/                # Pruebas automatizadas
├── .env                  # Variables de entorno (no incluido en repo)
├── .gitignore
├── requirements.txt
└── README.md
```

## Documentación API

La API proporciona los siguientes endpoints principales:

- `GET /api/health` - Verifica el estado de la API
- `POST /api/prediction` - Genera una predicción para un juego deportivo

Para una documentación completa, consulta la documentación OpenAPI generada automáticamente en `/docs`.

## Desarrollo

### Crear un nuevo endpoint

Crea un nuevo archivo en `app/api/routers/`, por ejemplo `app/api/routers/analysis.py`:

```python
from fastapi import APIRouter, Depends

router = APIRouter(
    prefix="/analysis",
    tags=["Analysis"],
)

@router.get("/")
async def get_analysis():
    return {"message": "Análisis disponible"}
```

El router se registrará automáticamente en la API.

## Licencia

Este proyecto está bajo la Licencia MIT.
