from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, UploadFile, Form, HTTPException
from sqlmodel import Session

from sqlmodel import select

from ..database import get_session
from ..models import Analysis, User
from ..schemas import AnalysisResponse
from ..deps import get_current_user
from ..utils.files import save_upload
from ..services import inference, gradcam

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
  file: UploadFile = File(...),
  notes: Optional[str] = Form(default=None),
  session: Session = Depends(get_session),
  current_user: Optional[User] = Depends(get_current_user),
):
  image_path = await save_upload(file)
  try:
    result = inference.predict(image_path)
  except Exception as exc:
    raise HTTPException(status_code=500, detail=f"분석 실패: {exc}")

  gradcam_path = gradcam.generate_gradcam(image_path)

  diagnosis = result["prediction"]
  prob = result["prob"]
  risk_level = "uncertain" if result["uncertain"] else ("high" if prob >= 0.8 else "medium" if prob >= 0.6 else "low")
  recommendations = "전문의 상담을 권장합니다." if risk_level != "low" else "경과관찰을 권장합니다."
  referral = "가까운 피부과 내원 권장"

  analysis = Analysis(
    user_id=current_user.id if current_user else None,
    image_path=str(image_path),
    gradcam_path=str(gradcam_path),
    diagnosis=diagnosis,
    probability=prob,
    risk_level=risk_level,
    recommendations=recommendations,
    referral=referral,
    notes=notes,
  )

  if current_user:
    session.add(analysis)
    session.commit()
    session.refresh(analysis)
  else:
    analysis.id = -1  # transient id

  gradcam_url = f"/static/gradcam/{gradcam_path.name}"
  return AnalysisResponse(
    id=analysis.id or -1,
    diagnosis=diagnosis,
    probability=prob,
    risk_level=risk_level,
    gradcam_url=gradcam_url,
    recommendations=recommendations,
    referral=referral,
    created_at=analysis.created_at,
    notes=notes,
  )


@router.get("/history", response_model=list[AnalysisResponse])
def get_history(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
  statement = select(Analysis).where(Analysis.user_id == current_user.id).order_by(Analysis.created_at.desc())
  analyses = session.exec(statement).all()
  responses = []
  for a in analyses:
    responses.append(
      AnalysisResponse(
        id=a.id,
        diagnosis=a.diagnosis,
        probability=a.probability,
        risk_level=a.risk_level,
        gradcam_url=f"/static/gradcam/{Path(a.gradcam_path).name}" if a.gradcam_path else None,
        recommendations=a.recommendations,
        referral=a.referral,
        created_at=a.created_at,
        notes=a.notes,
      )
    )
  return responses
