from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from ..database import get_session
from ..models import User
from ..schemas import Token, UserCreate, UserRead
from ..config import get_settings
from ..deps import hash_password, verify_password, create_access_token

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


@router.post("/register", response_model=UserRead)
def register(payload: UserCreate, session: Session = Depends(get_session)):
  existing = session.exec(select(User).where(User.email == payload.email)).first()
  if existing:
    raise HTTPException(status_code=400, detail="이미 가입된 이메일입니다.")
  user = User(email=payload.email, name=payload.name, password_hash=hash_password(payload.password))
  session.add(user)
  session.commit()
  session.refresh(user)
  return user


@router.post("/login", response_model=Token)
def login(payload: UserCreate, session: Session = Depends(get_session)):
  user = session.exec(select(User).where(User.email == payload.email)).first()
  if not user or not verify_password(payload.password, user.password_hash):
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
  access_token = create_access_token({"sub": user.email}, expires_delta=timedelta(minutes=settings.access_token_expire_minutes))
  return Token(access_token=access_token)
