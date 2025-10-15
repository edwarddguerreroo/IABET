from fastapi import APIRouter, Depends
from logging import Logger

from app.api.dependencies import get_model_service, get_logger
from app.models.game import Game
from app.models.prediction import PredictionResponse
from app.services.model_service import ModelService

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
                "teamId": "583ed102-fb46-11e1-82cb-f4ce4684ea4c",
                "name": "Denver Nuggets",
                "alias": "DEN",
                "conference": None,
                "division": None,
                "score": 0,
                "record": None,
                "players": [
                    {
                        "playerId": "f2625432-3903-4f90-9b0b-2e4f63856bb0",
                        "fullName": "Nikola Jokić",
                        "jerseyNumber": 15,
                        "position": "C",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "685576ef-ea6c-4ccf-affd-18916baf4e60",
                        "fullName": "Jamal Murray",
                        "jerseyNumber": 27,
                        "position": "PG",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "3a7d6510-00e9-4265-81df-864a1f547269",
                        "fullName": "Michael Porter Jr.",
                        "jerseyNumber": 1,
                        "position": "SF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [],
                    },
                    {
                        "playerId": "20f85838-0bd5-4c1f-ab85-a308bafaf5bc",
                        "fullName": "Aaron Gordon",
                        "jerseyNumber": 32,
                        "position": "PF",
                        "starter": True,
                        "status": "ACT",
                        "injuries": [
                            {
                                "id": "638efe6b-bb02-4c41-a88c-b6bc09f97155",
                                "type": "Hamstring",
                                "location": "Hamstring",
                                "comment": "The Nuggets announced that Gordon is expected to be listed as Questionable for Sunday's (May. 18) Game 7 against the Thunder, per Tony Jones of The Athletic.",
                                "startDate": "2025-05-16",
                                "expectedReturn": None,
                            },
                        ],
                    },
                    {
                        "playerId": "74a45eed-f2b0-4886-ae71-d04cf7d59528",
                        "fullName": "Russell Westbrook",
                        "jerseyNumber": 4,
                        "position": "PG",
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
    logger.info(f"🎯 Usando mock_game para predicción")
    logger.info(f"🔍 Mock Game - homeTeam.name: '{mock_data['homeTeam']['name']}'")
    logger.info(f"🔍 Mock Game - awayTeam.name: '{mock_data['awayTeam']['name']}'")
    
    # Convertir el mock_game a objeto Game de Pydantic
    from app.models.game import Game
    mock_game_obj = Game(**mock_data)
    
    prediction = svc.predict(mock_game_obj)
    logger.info(f"Predicción generada para mock_game: {prediction}")
    return prediction
