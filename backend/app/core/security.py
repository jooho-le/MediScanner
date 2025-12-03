from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..db.session import get_session
from ..db.models import User

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
settings = get_settings()


def hash_password(password: str) -> str:
  return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
  return pwd_context.verify(password, hashed)


def create_access_token(subject: str, expires_delta: Optional[timedelta] = None) -> str:
  expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.access_token_expire_minutes))
  payload = {"sub": subject, "exp": expire}
  return jwt.encode(payload, settings.secret_key, algorithm="HS256")


def get_current_user(token: Optional[str] = Depends(oauth2_scheme), session: Session = Depends(get_session)) -> User:
  credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
  )
  if token is None:
    raise credentials_exception
  try:
    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    email = payload.get("sub")
    if email is None:
      raise credentials_exception
  except JWTError:
    raise credentials_exception

  user = session.query(User).filter(User.email == email).first()
  if user is None:
    raise credentials_exception
  return user


from typing import Optional
...
def get_current_user_optional(
  token: Optional[str] = Depends(oauth2_scheme), session: Session = Depends(get_session)
) -> Optional[User]:
  if not token:
    return None
  try:
    payload = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
    email = payload.get("sub")
  except JWTError:
    return None
  if not email:
    return None
  return session.query(User).filter(User.email == email).first()
