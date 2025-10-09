from fastapi import APIRouter, Depends
from logging import Logger

from app.api.dependencies import get_model_service, get_logger
from app.models.game import Game
from app.models.prediction import PredictionResponse
from app.services.model_service import ModelService

router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"],
)

@router.post(
    "/", 
    response_model=PredictionResponse,
    summary="Genera una predicción para un juego dado",
)
async def prediction(
    game: Game,
    svc: ModelService = Depends(get_model_service),
    logger: Logger     = Depends(get_logger),
) -> PredictionResponse:
    """
    - Recibe la información completa de un juego (Game).
    - Pasa ese objeto al servicio de IA para generar la predicción.
    - Devuelve un PredictionResponse tipado y validado.
    """
    logger.info(f"Recibiendo predicción para gameId={game.gameId}")
    # Aquí llamas a tu lógica de inferencia:
    # prediction = svc.predict(game)
    # Por ahora devolvemos un stub:
    prediction = svc.predict(game)
    logger.info(f"Predicción generada para gameId={game.gameId}: {prediction}")
    return prediction
