from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from sqlalchemy import Integer, String, DateTime, ForeignKey, Float, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .session import Base


class User(Base):
  __tablename__ = "users"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
  email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
  name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
  password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

  analyses: Mapped[List["Analysis"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Analysis(Base):
  __tablename__ = "analyses"

  id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
  user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
  image_path: Mapped[str] = mapped_column(String(512))
  gradcam_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
  diagnosis: Mapped[str] = mapped_column(String(255))
  probability: Mapped[float] = mapped_column(Float)
  risk_level: Mapped[str] = mapped_column(String(32))
  recommendations: Mapped[str] = mapped_column(Text)
  referral: Mapped[str] = mapped_column(Text)
  notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
  created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

  user: Mapped[Optional[User]] = relationship(back_populates="analyses")
