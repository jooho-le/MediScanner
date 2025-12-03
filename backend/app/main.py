from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .core.config import get_settings
from .db.init_db import init_db
from .routers import auth, analysis, clinics

settings = get_settings()
app = FastAPI(title="MediScanner API")

app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"],
  allow_methods=["*"],
  allow_headers=["*"],
  allow_credentials=True,
)

# Serve Grad-CAM under the API prefix so front-end proxy works (e.g., /api/static/gradcam/...)
app.mount(f"{settings.api_prefix}/static/gradcam", StaticFiles(directory=settings.gradcam_dir), name="gradcam")


@app.on_event("startup")
def startup_event():
  init_db()


app.include_router(auth.router, prefix=settings.api_prefix)
app.include_router(analysis.router, prefix=settings.api_prefix)
app.include_router(clinics.router, prefix=settings.api_prefix)


@app.get("/health")
def health():
  return {"status": "ok"}
