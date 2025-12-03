from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class Token(BaseModel):
  access_token: str
  token_type: str = "bearer"


class UserCreate(BaseModel):
  email: EmailStr
  password: str
  name: Optional[str] = None


class UserRead(BaseModel):
  id: int
  email: EmailStr
  name: Optional[str]
  created_at: datetime

  model_config = {"from_attributes": True}


class AnalysisResponse(BaseModel):
  id: int
  diagnosis: str
  probability: float
  risk_level: str
  gradcam_url: Optional[str]
  recommendations: str
  referral: str
  notes: Optional[str] = None
  created_at: datetime

  model_config = {"from_attributes": True}
