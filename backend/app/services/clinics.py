from __future__ import annotations

from typing import List, Optional

fake_clinics = [
  {"name": "서울테라피 피부과", "specialty": "피부종양 · 레이저", "address": "서울 강남구", "distance": "2.1km"},
  {"name": "바른빛 피부센터", "specialty": "색소질환 · 광선치료", "address": "서울 서초구", "distance": "4.0km"},
  {"name": "해안 피부클리닉", "specialty": "모반 · 종양 수술", "address": "부산 해운대", "distance": "부산"},
]


def list_clinics(region: Optional[str] = None) -> List[dict]:
  if region and region != "전체":
    return [c for c in fake_clinics if region in c["address"]]
  return fake_clinics
