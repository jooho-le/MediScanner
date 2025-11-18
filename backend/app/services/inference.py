from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.append(str(PROJECT_ROOT))

from src.mediscanner.model import build_model  # type: ignore
from src.mediscanner.calibration import TemperatureScaler  # type: ignore

from ..config import get_settings

settings = get_settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names: list[str] = []


def load_resources() -> None:
  global _model, _class_names
  if _model is not None:
    return
  class_path = settings.class_names_path
  if class_path.exists():
    _class_names = json.loads(class_path.read_text(encoding="utf-8"))
  else:
    raise FileNotFoundError(f"Class names file not found at {class_path}")
  model, _ = build_model("efficientnet_v2_m", num_classes=len(_class_names))
  state = torch.load(settings.model_weights, map_location="cpu")
  if isinstance(state, dict) and "model_state" in state:
    state = state["model_state"]
  model.load_state_dict(state)
  model.to(DEVICE)
  model.eval()
  _model = model


def _transform(image: Image.Image) -> torch.Tensor:
  tfm = transforms.Compose(
    [
      transforms.Resize((256, 256)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
  )
  return tfm(image).unsqueeze(0)


def predict(image_path: Path) -> dict[str, Any]:
  load_resources()
  assert _model is not None
  image = Image.open(image_path).convert("RGB")
  tensor = _transform(image).to(DEVICE)
  with torch.no_grad():
    logits = _model(tensor)
    if settings.temperature:
      scaler = TemperatureScaler(init_temperature=settings.temperature)
      probs = scaler.softmax(logits).cpu().numpy()[0]
    else:
      probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

  top_idx = int(np.argmax(probs))
  top_prob = float(probs[top_idx])
  entropy = float(-(probs * (np.log(probs + 1e-12))).sum())
  uncertain = top_prob < settings.reject_threshold or entropy > settings.entropy_threshold

  return {
    "class_names": _class_names,
    "probabilities": probs.tolist(),
    "prediction": _class_names[top_idx],
    "prob": top_prob,
    "uncertain": uncertain,
    "entropy": entropy,
  }
