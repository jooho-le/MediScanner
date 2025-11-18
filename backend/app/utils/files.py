from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import UploadFile

from ..config import get_settings

settings = get_settings()


async def save_upload(file: UploadFile) -> Path:
  ext = Path(file.filename or "image.jpg").suffix or ".jpg"
  dest = settings.uploads_dir / f"{uuid.uuid4().hex}{ext}"
  contents = await file.read()
  dest.write_bytes(contents)
  return dest
