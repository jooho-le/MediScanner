from __future__ import annotations

from sqlmodel import SQLModel, create_engine, Session

from .config import get_settings

settings = get_settings()
engine = create_engine(settings.sqlite_url, echo=False, connect_args={"check_same_thread": False})


def init_db() -> None:
  SQLModel.metadata.create_all(engine)


def get_session() -> Session:
  with Session(engine) as session:
    yield session
