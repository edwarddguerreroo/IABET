import pytest
from datetime import datetime, timezone
import uuid

from app.models.game import Game, Team, Venue
from app.models.prediction import PredictionResponse
from app.services.model_service import ModelService

@pytest.fixture
def model_service():
    """Fixture para obtener una instancia del servicio de modelo"""
    return ModelService()

@pytest.fixture
def sample_game():
    """Genera un juego de ejemplo para pruebas"""
    return Game(
        gameId="test-game-123",
        status="scheduled",
        scheduled=datetime.now(timezone.utc),
        venue=Venue(
            id="venue-test",
            name="Estadio de Prueba",
            city="Ciudad Test",
            state="Estado Test",
            capacity=30000
        ),
        homeTeam=Team(
            teamId="home-test",
            name="Equipo Local",
            alias="Local",
            score=2,
            players=[]
        ),
        awayTeam=Team(
            teamId="away-test",
            name="Equipo Visitante",
            alias="Visitante",
            score=0,
            players=[]
        ),
        coverage=None
    )

def test_model_service_initialization(model_service):
    """Verifica que el servicio de modelo se inicialice correctamente"""
    assert model_service is not None
    assert model_service.model is None  # En nuestra implementación de prueba

def test_predict_returns_prediction_response(model_service, sample_game):
    """Verifica que el método predict devuelve un objeto PredictionResponse"""
    result = model_service.predict(sample_game)
    assert isinstance(result, PredictionResponse)
    assert isinstance(result.id, uuid.UUID)
    assert result.home_team == sample_game.homeTeam.name
    assert result.away_team == sample_game.awayTeam.name
    
def test_confidence_calculation(model_service):
    """Prueba que el cálculo de confianza funciona con diferentes puntuaciones"""
    # Caso 1: Equipo local con mayor puntuación
    game1 = Game(
        gameId="game1",
        status="inprogress",
        scheduled=datetime.now(timezone.utc),
        venue=Venue(id="v1", name="V1", city="C1", state="S1", capacity=1000),
        homeTeam=Team(teamId="h1", name="Home1", alias="H1", score=3, players=[]),
        awayTeam=Team(teamId="a1", name="Away1", alias="A1", score=1, players=[]),
        coverage=None
    )
    pred1 = model_service.predict(game1)
    assert pred1.target_name == "Home1"  # El equipo local debería ser el favorito
    
    # Caso 2: Equipo visitante con mayor puntuación
    game2 = Game(
        gameId="game2",
        status="inprogress",
        scheduled=datetime.now(timezone.utc),
        venue=Venue(id="v1", name="V1", city="C1", state="S1", capacity=1000),
        homeTeam=Team(teamId="h1", name="Home1", alias="H1", score=1, players=[]),
        awayTeam=Team(teamId="a1", name="Away1", alias="A1", score=4, players=[]),
        coverage=None
    )
    pred2 = model_service.predict(game2)
    assert pred2.target_name == "Away1"  # El equipo visitante debería ser el favorito
    
    # Verificar que la diferencia de puntuación afecta la confianza
    assert pred2.confidence_percentage > pred1.confidence_percentage
