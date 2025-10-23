from fastapi import APIRouter, Depends
from logging import Logger

from app.api.dependencies import get_model_service, get_logger, get_redis
from app.models.game import Game
from app.models.prediction import PredictionResponse
from app.services.model_service import ModelService
from app.core.redis_config import RedisClient

mock_game = [
        {
            "gameId": "f71cb64f-4d52-4e2b-a3db-7436b798476d",
            "status": "scheduled",
            "scheduled": "2025-05-18T19:30:00+00:00",
            "venue": {
                "id": "a13af216-4409-5021-8dd5-255cc71bffc3",
                "name": "Paycom Center",
                "city": "Oklahoma City",
                "state": "OK",
                "capacity": 18203,
            },
            "homeTeam": {
                "teamId": "583ecfff-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Oklahoma City Thunder",
                "alias": "OKC",
                "conference": None,
                "division": None,
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "d9ea4a8f-ff51-408d-b518-980efc2a35a1",
                        "fullName": "Shai Gilgeous-Alexander",
                        "jerseyNumber": 2,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "eb91a4c8-1a8a-46bf-86e6-e16950b67ef6",
                        "fullName": "Chet Holmgren",
                        "jerseyNumber": 7,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "62c44a90-f280-438a-9c7e-252f4f376283",
                        "fullName": "Jalen Williams",
                        "jerseyNumber": 8,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "3f7e2350-e208-4791-98c2-684b53bb5a9a",
                        "fullName": "Luguentz Dort",
                        "jerseyNumber": 5,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "38745a56-7472-4844-a2dc-f61d3bcd941f",
                        "fullName": "Isaiah Hartenstein",
                        "jerseyNumber": 55,
                        "position": "C",
                        "starter": False,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "awayTeam": {
                "teamId": "583ecb3a-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Houston Rockets",
                "alias": "HOU",
                "conference": None,
                "division": None,
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "9a331092-35db-456c-a44a-d5b80a02ebe9",
                        "fullName": "Jalen Green",
                        "jerseyNumber": 0,
                        "position": "SG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "45f17314-918c-49bd-a482-adc171859025",
                        "fullName": "Fred VanVleet",
                        "jerseyNumber": 5,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "47ff78e7-5607-48d6-9c1a-bddb075dbe70",
                        "fullName": "Alperen Sengun",
                        "jerseyNumber": 28,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "293be24b-3a94-40b2-a7a4-a1dd788302e9",
                        "fullName": "Jabari Smith Jr.",
                        "jerseyNumber": 1,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "aeba1657-a0df-427b-ba9a-6db9e9147304",
                        "fullName": "Cam Whitmore",
                        "jerseyNumber": 7,
                        "position": "SF",
                        "starter": False,
                        "status": "ACT",
                        "injuries": [],
                    },
                ],
            },
            "coverage": {
                "broadcasters": [
                    {
                        "name": "ABC",
                        "type": "tv",
                    },
                ],
            },
        }
    ]
    
router = APIRouter(
    prefix="/prediction",
    tags=["Prediction"],
)

@router.post(
    "/", 
    summary="Genera una predicción completa para un juego dado",
)
async def prediction(
    game: Game = None,
    svc: ModelService = Depends(get_model_service),
    logger: Logger     = Depends(get_logger),
    redis: RedisClient = Depends(get_redis),
):
    """
    - Usa el mock_game predefinido para generar predicciones.
    - Devuelve una estructura completa con predicciones de equipos y jugadores.
    
    La respuesta incluye:
    - game_info: Información básica del juego
    - team_predictions: Predicciones de equipos (is_win, total_points, teams_points, ht_teams_points, ht_total_points)
    - player_predictions: Predicciones de jugadores (points, three_points, assists, rebounds, double_double)
    - summary: Resumen de conteos de predicciones
    """
    # Usar el mock_game predefinido
    mock_data = mock_game[0]  # Tomar el primer (y único) juego del mock
    
    # Convertir el mock_game a objeto Game de Pydantic
    from app.models.game import Game
    mock_game_obj = Game(**mock_data)
    
    prediction = svc.predict(mock_game_obj, redis)
    return prediction
