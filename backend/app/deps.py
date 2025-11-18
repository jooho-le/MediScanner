from __future__ import annotations

from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlmodel import Session, select

from .config import get_settings
from .database import get_session
from .models import User
from .schemas import TokenData

pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
settings = get_settings()


def verify_password(plain: str, hashed: str) -> bool:
  return pwd_context.verify(plain, hashed)


def hash_password(password: str) -> str:
  return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
  to_encode = data.copy()
  expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
  to_encode.update({"exp": expire})
  encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm="HS256")
  return encoded_jwt


def get_current_user(token: Annotated[str, Depends(oauth2_scheme)], session: Annotated[Session, Depends(get_session)]) -> User:
  credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
  )
  try:
    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    sub = payload.get("sub")
    if sub is None:
      raise credentials_exception
    token_data = TokenData(sub=sub)
  except JWTError:
    raise credentials_exception

  user = session.exec(select(User).where(User.email == token_data.sub)).first()
  if user is None:
    raise credentials_exception
  return user
