from __future__ import annotations

from datetime import datetime
from typing import Optional
from sqlmodel import SQLModel, Field


class User(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  email: str = Field(index=True, unique=True)
  name: Optional[str] = None
  password_hash: str
  created_at: datetime = Field(default_factory=datetime.utcnow)

class Analysis(SQLModel, table=True):
  id: Optional[int] = Field(default=None, primary_key=True)
  user_id: Optional[int] = Field(default=None, foreign_key="user.id")
  image_path: str
  gradcam_path: Optional[str] = None
  diagnosis: str
  probability: float
  risk_level: str
  recommendations: str
  referral: str
  created_at: datetime = Field(default_factory=datetime.utcnow)
  notes: Optional[str] = None
