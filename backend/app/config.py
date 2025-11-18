from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
  model_config = {
    "env_file": ".env",
    "env_file_encoding": "utf-8",
    "protected_namespaces": ("settings_",),
  }
  api_prefix: str = "/api"
  secret_key: str = "change-me"
  access_token_expire_minutes: int = 60 * 24
  sqlite_url: str = "sqlite:///./mediscanner.db"
  media_dir: Path = Path("media")
  uploads_dir: Path = media_dir / "uploads"
  gradcam_dir: Path = media_dir / "gradcam"
  model_weights: Path = Path("outputs_merge_e5/checkpoints/last_model.pt")
  class_names_path: Path = Path("outputs_merge_e5/class_names.json")
  reject_threshold: float = 0.6
  entropy_threshold: float = 1.4
  temperature: float = 1.8
  hira_service_key: Optional[str] = None

@lru_cache()
def get_settings() -> Settings:
  settings = Settings()
  settings.media_dir.mkdir(exist_ok=True, parents=True)
  settings.uploads_dir.mkdir(exist_ok=True, parents=True)
  settings.gradcam_dir.mkdir(exist_ok=True, parents=True)
  return settings
