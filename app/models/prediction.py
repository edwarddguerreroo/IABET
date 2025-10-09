from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

class ValueBet(BaseModel):
    """Información de una apuesta de valor identificada"""
    target: str
    player_name: Optional[str] = None
    team_name: Optional[str] = None
    line: float
    side: Literal["over", "under"]
    predicted_value: float
    market_odds: float
    edge_percentage: float
    confidence: float
    kelly_fraction: float
    bookmaker: str
    recommended_stake: Optional[float] = None
    expected_roi: Optional[float] = None

class BookmakerAnalysis(BaseModel):
    """Análisis completo de bookmakers para una predicción"""
    analysis_enabled: bool = False
    total_opportunities: int = 0
    value_bets_found: int = 0
    average_edge: float = 0.0
    best_value_bets: List[ValueBet] = []
    market_summary: Dict[str, Any] = {}
    sportradar_status: str = "not_checked"
    analysis_timestamp: Optional[datetime] = None
    error_message: Optional[str] = None

class PredictionResponse(BaseModel):
    id: UUID
    sport_id: UUID
    league_id: UUID
    home_team: str
    away_team: str
    target_type: Literal["team", "match", "player"]
    target_name: str
    bet_line: str
    bet_type: str
    prediction_type: str
    confidence_percentage: float
    analysis: str
    event_datetime: datetime
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None
    
    # Nuevos campos para integración con bookmakers
    bookmaker_analysis: Optional[BookmakerAnalysis] = None
    has_value_bets: bool = False
    total_value_bets: int = 0
    best_edge_percentage: Optional[float] = None
