"""
Database Schemas for MVP

Each Pydantic model name maps to a MongoDB collection with the lowercase name.
Example: class User -> collection "user"
"""
from typing import List, Optional, Literal
from pydantic import BaseModel, Field, EmailStr

# Core users
class User(BaseModel):
    role: Literal["parent", "child"] = Field(..., description="User role")
    name: str = Field(..., description="Display name")
    email: Optional[EmailStr] = Field(None, description="Email for parent or older children")
    parentId: Optional[str] = Field(None, description="Parent user id for child accounts")
    children: Optional[List[str]] = Field(default=None, description="Child user ids for parent accounts")
    pin: Optional[str] = Field(None, description="Simple PIN for child login (MVP)")

# Messages from users and assistant with analysis
class Message(BaseModel):
    userId: str
    role: Literal["user", "assistant"]
    text: str
    emotions: Optional[List[str]] = None
    sentiment: Optional[str] = None
    risk: Optional[float] = None

# Rolling risk state per user
class RiskState(BaseModel):
    userId: str
    score: float = 0.0
    history: Optional[List[float]] = None
    lastAlertAt: Optional[str] = None

# Timers and usage
class Timer(BaseModel):
    childId: str
    dailyLimit: int = Field(60, ge=0, description="Daily limit in minutes")
    sessionLimit: int = Field(20, ge=0, description="Per-session limit in minutes")
    windows: Optional[List[dict]] = None
    activeSession: Optional[dict] = None
    history: Optional[List[dict]] = None

# Alerts sent to parents
class Alert(BaseModel):
    userId: str
    level: Literal["info", "concern", "critical"]
    summary: str
    channels: Optional[List[str]] = None
    sentAt: Optional[str] = None
    acknowledged: bool = False
