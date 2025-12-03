from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.security import get_current_user, get_current_user_optional
from ..db.session import get_session
from ..db.models import Analysis, User
from ..schemas import AnalysisResponse
from ..services import inference, gradcam
from ..services.storage import save_upload

router = APIRouter(prefix="/analysis", tags=["analysis"])
settings = get_settings()


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
  image: UploadFile = File(...),
  notes: Optional[str] = Form(default=None),
  session: Session = Depends(get_session),
  current_user: Optional[User] = Depends(get_current_user_optional),
):
  # 'image' 필드로 받은 파일 저장
  image_path = await save_upload(image)
  try:
    pred = inference.predict(image_path)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"분석 실패: {exc}") from exc

  gradcam_path = gradcam.generate_gradcam(image_path, pred.get("target_index"))
  risk_level = "uncertain"
  prob = pred["prob"]
  if not pred["uncertain"]:
    risk_level = "high" if prob >= 0.8 else "medium" if prob >= 0.6 else "low"

  recommendations = "전문의 상담을 권장합니다." if risk_level != "low" else "경과 관찰을 권장합니다."
  referral = "가까운 피부과 방문 권유"

  created_at = datetime.utcnow()
  analysis = Analysis(
    user_id=current_user.id if current_user else None,
    image_path=str(image_path),
    gradcam_path=str(gradcam_path),
    diagnosis=pred["prediction"],
    probability=prob,
    risk_level=risk_level,
    recommendations=recommendations,
    referral=referral,
    notes=notes,
    created_at=created_at,
  )

  if current_user:
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
  else:
    analysis.id = -1
    analysis.created_at = created_at

  return AnalysisResponse(
    id=analysis.id,
    diagnosis=analysis.diagnosis,
    probability=analysis.probability,
    risk_level=analysis.risk_level,
    gradcam_url=f"{settings.api_prefix}/static/gradcam/{Path(gradcam_path).name}",
    recommendations=recommendations,
    referral=referral,
    notes=notes,
    created_at=analysis.created_at,
  )


@router.get("/history", response_model=List[AnalysisResponse])
def history(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
  rows = (
    session.query(Analysis)
    .filter(Analysis.user_id == current_user.id)
    .order_by(Analysis.created_at.desc())
    .all()
  )
  responses: List[AnalysisResponse] = []
  for row in rows:
    responses.append(
      AnalysisResponse(
        id=row.id,
        diagnosis=row.diagnosis,
        probability=row.probability,
        risk_level=row.risk_level,
        gradcam_url=f"{settings.api_prefix}/static/gradcam/{Path(row.gradcam_path).name}" if row.gradcam_path else None,
        recommendations=row.recommendations,
        referral=row.referral,
        notes=row.notes,
        created_at=row.created_at,
      )
    )
  return responses
