from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query

from ..services.hira import list_clinics

router = APIRouter(prefix="/clinics", tags=["clinics"])


@router.get("/")
def clinics(region: Optional[str] = Query(default=None, description="예: 서울, 부산")):
  return list_clinics(region)
