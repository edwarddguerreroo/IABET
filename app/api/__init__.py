# app/api/__init__.py
import pkgutil
import importlib
from fastapi import APIRouter

api_router = APIRouter()

# Busca todos los módulos de app/api/routers y, si exponen `router`, los añade
def include_all_routers():
    import app.api.routers as routers_pkg
    for _, module_name, _ in pkgutil.iter_modules(routers_pkg.__path__):
        module = importlib.import_module(f"app.api.routers.{module_name}")
        if hasattr(module, "router"):
            api_router.include_router(module.router)

include_all_routers()
