from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr


class Token(BaseModel):
  access_token: str
  token_type: str = "bearer"


class TokenData(BaseModel):
  sub: Optional[str] = None


class UserCreate(BaseModel):
  email: EmailStr
  password: str
  name: Optional[str] = None


class UserRead(BaseModel):
  id: int
  email: EmailStr
  name: Optional[str] = None

  class Config:
    from_attributes = True


class AnalysisResponse(BaseModel):
  id: int
  diagnosis: str
  probability: float
  risk_level: str
  gradcam_url: Optional[str] = None
  recommendations: str
  referral: str
  created_at: datetime
  notes: Optional[str] = None

  class Config:
    from_attributes = True
