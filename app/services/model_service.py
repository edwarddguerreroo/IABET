# app/services/model_service.py
import logging
import uuid
from datetime import datetime

from app.models.game import Game
from app.models.prediction import PredictionResponse
from app.architectures.basketball.main import NBAPredict

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        # Carga tu modelo una sola vez al instanciar el servicio
        logger.info("🔄 Cargando modelo de IA en ModelService...")
        # self.model = load_your_model(path=settings.MODEL_PATH)
        self.model = None

    def predict(self, game: Game) -> PredictionResponse:
      

      models = NBAPredict()
      models.predict(game)

    """  # 
        prediction = PredictionResponse(
            id=uuid.uuid4(),
            sport_id=uuid.uuid4(),
            league_id=uuid.uuid4(),
            home_team=home_team,
            away_team=away_team,
            target_type="team",
            target_name=home_team if home_score > away_score else away_team,
            bet_line=f"{home_team} vs {away_team}",
            bet_type="Moneyline",
            prediction_type="winner",
            confidence_percentage=confidence,
            analysis=f"Basado en el análisis del juego entre {home_team} y {away_team}, "
                    f"hay un {confidence*100:.1f}% de confianza en que "
                    f"{home_team if home_score > away_score else away_team} ganará el partido.",
            event_datetime=game.scheduled,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        logger.info(f"Predicción generada para gameId={game.gameId}: {prediction.id}")
        return prediction """
