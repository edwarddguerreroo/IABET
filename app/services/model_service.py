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
        logger.info("Cargando modelo de IA en ModelService...")
        # self.model = load_your_model(path=settings.MODEL_PATH)
        self.model = None

    def predict(self, game: Game):
        """
        Genera predicciones usando NBAPredict.
        Convierte el objeto Game a formato SportRadar y pasa los datos a NBAPredict.
        """
        try:
            # Debug: verificar los valores del objeto Game
            logger.info(f"🔍 Debug ModelService - gameId: {game.gameId}")
            logger.info(f"🔍 Debug ModelService - homeTeam.name: '{game.homeTeam.name}' (tipo: {type(game.homeTeam.name)})")
            logger.info(f"🔍 Debug ModelService - awayTeam.name: '{game.awayTeam.name}' (tipo: {type(game.awayTeam.name)})")
            logger.info(f"🔍 Debug ModelService - homeTeam.record: {game.homeTeam.record}")
            logger.info(f"🔍 Debug ModelService - awayTeam.record: {game.awayTeam.record}")
            
            # Convertir objeto Game a formato SportRadar
            game_data = self._convert_game_to_sportradar(game)
            logger.info(f"🔄 Datos convertidos: {type(game_data)}")
            logger.info(f"🔍 Debug convertido - homeTeam.name: '{game_data['homeTeam']['name']}'")
            logger.info(f"🔍 Debug convertido - awayTeam.name: '{game_data['awayTeam']['name']}'")
            
            # Inicializar el predictor NBA y pasar los datos convertidos
            models = NBAPredict()
            result = models.predict(game_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Error en ModelService.predict: {e}")
            return {'error': str(e), 'status': 'error'}
    
    def _convert_game_to_sportradar(self, data: Game) -> dict:
        """
        Convierte objeto Game de Pydantic a formato SportRadar.
        """
        logger.info("🔄 Convirtiendo objeto Game de Pydantic a formato SportRadar")
        logger.info(f"🔍 Debug - homeTeam.name: {data.homeTeam.name}")
        logger.info(f"🔍 Debug - awayTeam.name: {data.awayTeam.name}")
        
        return {
            'gameId': data.gameId,
            'status': data.status,
            'scheduled': data.scheduled.isoformat() if hasattr(data.scheduled, 'isoformat') else str(data.scheduled),
            'venue': {
                'id': data.venue.id,
                'name': data.venue.name,
                'city': data.venue.city,
                'state': data.venue.state,
                'capacity': data.venue.capacity
            },
            'homeTeam': {
                'teamId': data.homeTeam.teamId,
                'name': data.homeTeam.name,
                'alias': data.homeTeam.alias,
                'conference': data.homeTeam.conference,
                'division': data.homeTeam.division,
                'score': data.homeTeam.score,
                'record': data.homeTeam.record,
                'players': [{
                    'playerId': player.playerId,
                    'fullName': player.fullName,
                    'jerseyNumber': player.jerseyNumber,
                    'position': player.position,
                    'starter': player.starter,
                    'status': player.status,
                    'injuries': [{
                        'id': injury.id,
                        'type': injury.type,
                        'location': injury.location,
                        'comment': injury.comment,
                        'startDate': injury.startDate.isoformat() if hasattr(injury.startDate, 'isoformat') else str(injury.startDate),
                        'expectedReturn': injury.expectedReturn.isoformat() if injury.expectedReturn and hasattr(injury.expectedReturn, 'isoformat') else str(injury.expectedReturn) if injury.expectedReturn else None
                    } for injury in player.injuries]
                } for player in data.homeTeam.players]
            },
            'awayTeam': {
                'teamId': data.awayTeam.teamId,
                'name': data.awayTeam.name,
                'alias': data.awayTeam.alias,
                'conference': data.awayTeam.conference,
                'division': data.awayTeam.division,
                'score': data.awayTeam.score,
                'record': data.awayTeam.record,
                'players': [{
                    'playerId': player.playerId,
                    'fullName': player.fullName,
                    'jerseyNumber': player.jerseyNumber,
                    'position': player.position,
                    'starter': player.starter,
                    'status': player.status,
                    'injuries': [{
                        'id': injury.id,
                        'type': injury.type,
                        'location': injury.location,
                        'comment': injury.comment,
                        'startDate': injury.startDate.isoformat() if hasattr(injury.startDate, 'isoformat') else str(injury.startDate),
                        'expectedReturn': injury.expectedReturn.isoformat() if injury.expectedReturn and hasattr(injury.expectedReturn, 'isoformat') else str(injury.expectedReturn) if injury.expectedReturn else None
                    } for injury in player.injuries]
                } for player in data.awayTeam.players]
            },
            'coverage': {
                'broadcasters': [{
                    'name': broadcaster.name,
                    'type': broadcaster.type
                } for broadcaster in data.coverage.broadcasters] if data.coverage else []
            }
        }








