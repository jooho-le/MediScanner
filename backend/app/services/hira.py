from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from ..core.config import get_settings

settings = get_settings()


@lru_cache()
def _seed_clinics() -> List[dict]:
  return [
    {"name": "서울테라피 피부과", "specialty": "피부종양 · 레이저", "address": "서울 강남구", "region": "서울", "distance": "2.1km"},
    {"name": "바른빛 피부센터", "specialty": "색소·광선치료", "address": "서울 서초구", "region": "서울", "distance": "4.0km"},
    {"name": "해안 피부클리닉", "specialty": "모반 · 종양 수술", "address": "부산 해운대구", "region": "부산", "distance": "부산"},
  ]


def list_clinics(region: Optional[str] = None) -> List[dict]:
  clinics = _seed_clinics()
  if region and region != "전체":
    return [c for c in clinics if region in (c.get("region") or "") or region in c.get("address", "")]
  return clinics
