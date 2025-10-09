from fastapi.testclient import TestClient
import pytest
from datetime import datetime, timezone

from app.main import app
from app.models.game import Game, Team, Venue
from app.models.prediction import PredictionResponse

client = TestClient(app)

def test_read_root():
    """Prueba el endpoint principal"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    
def test_health_check():
    """Prueba el endpoint de salud"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
@pytest.fixture
def sample_game():
    """Genera un juego de ejemplo para usar en pruebas"""
    return Game(
        gameId="123456",
        status="scheduled",
        scheduled=datetime.now(timezone.utc),
        venue=Venue(
            id="venue1",
            name="Estadio Ejemplo",
            city="Ciudad",
            state="Estado",
            capacity=50000
        ),
        homeTeam=Team(
            teamId="home1",
            name="Equipo Local",
            alias="Local",
            score=3,
            players=[]
        ),
        awayTeam=Team(
            teamId="away1",
            name="Equipo Visitante",
            alias="Visitante",
            score=1,
            players=[]
        ),
        coverage=None
    )

def test_prediction_endpoint(sample_game):
    """Prueba el endpoint de predicción con un juego de muestra"""
    response = client.post(
        "/api/prediction/",
        json=sample_game.dict(),
    )
    assert response.status_code == 200
    
    # Verificar que la respuesta tiene el formato correcto
    data = response.json()
    assert "id" in data
    assert data["home_team"] == "Equipo Local"
    assert data["away_team"] == "Equipo Visitante"
    
    # El equipo local tiene mayor puntaje, debería ser el ganador predicho
    assert data["target_name"] == "Equipo Local"
    
    # Verificar que el porcentaje de confianza está en el rango correcto
    assert 0.5 <= data["confidence_percentage"] <= 1.0
