from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    protected_namespaces=("settings_",),
    extra="ignore",
  )

  api_prefix: str = "/api"
  secret_key: str
  access_token_expire_minutes: int = 60 * 24
  database_url: str = "sqlite:///./mediscanner.db"
  media_dir: Path = Path("media")
  uploads_dir: Path = media_dir / "uploads"
  gradcam_dir: Path = media_dir / "gradcam"

  model_weights: Path
  class_names_path: Path
  reject_threshold: float = 0.6
  entropy_threshold: float = 1.4
  temperature: float = 1.8
  hira_service_key: Optional[str] = None


@lru_cache()
def get_settings() -> Settings:
  settings = Settings()
  # Resolve to absolute paths for robustness
  settings.media_dir = settings.media_dir.resolve()
  settings.uploads_dir = settings.uploads_dir.resolve()
  settings.gradcam_dir = settings.gradcam_dir.resolve()
  settings.model_weights = settings.model_weights.resolve()
  settings.class_names_path = settings.class_names_path.resolve()

  settings.media_dir.mkdir(parents=True, exist_ok=True)
  settings.uploads_dir.mkdir(parents=True, exist_ok=True)
  settings.gradcam_dir.mkdir(parents=True, exist_ok=True)
  return settings
