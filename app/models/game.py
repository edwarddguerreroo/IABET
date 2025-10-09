# app/models/game.py
from __future__ import annotations
from typing import List, Optional
from datetime import datetime, date
from pydantic import BaseModel, Field

class Injury(BaseModel):
    id: str
    type: str
    location: str
    comment: str
    startDate: date
    expectedReturn: Optional[date] = None

class Player(BaseModel):
    playerId: str
    fullName: str
    jerseyNumber: int
    position: str
    starter: bool
    status: str
    injuries: List[Injury] = Field(default_factory=list)

class Team(BaseModel):
    teamId: str
    name: str
    alias: str
    conference: Optional[str] = None
    division: Optional[str] = None
    score: int
    record: Optional[str] = None
    players: List[Player] = Field(default_factory=list)

class Venue(BaseModel):
    id: str
    name: str
    city: str
    state: str
    capacity: int

class Broadcaster(BaseModel):
    name: str
    type: str

class Coverage(BaseModel):
    broadcasters: List[Broadcaster] = Field(default_factory=list)

class Game(BaseModel):
    gameId: str
    status: str
    scheduled: datetime
    venue: Venue
    homeTeam: Team
    awayTeam: Team
    coverage: Optional[Coverage] = None
