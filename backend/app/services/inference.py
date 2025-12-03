from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
  sys.path.append(str(PROJECT_ROOT))

from src.mediscanner.model import build_model  # type: ignore  # noqa: E402
from src.mediscanner.calibration import TemperatureScaler  # type: ignore  # noqa: E402

from ..core.config import get_settings

settings = get_settings()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_class_names: list[str] = []
_transform = transforms.Compose(
  [
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ]
)


def load_model() -> None:
  global _model, _class_names
  if _model is not None:
    return
  if not settings.class_names_path.exists():
    raise FileNotFoundError(f"class_names_path not found: {settings.class_names_path}")
  if not settings.model_weights.exists():
    raise FileNotFoundError(f"model_weights not found: {settings.model_weights}")
  _class_names = json.loads(settings.class_names_path.read_text(encoding="utf-8"))
  model, _ = build_model("efficientnet_v2_m", num_classes=len(_class_names))
  state = torch.load(settings.model_weights, map_location="cpu")
  if isinstance(state, dict) and "model_state" in state:
    state = state["model_state"]
  model.load_state_dict(state)
  model.to(DEVICE)
  model.eval()
  _model = model


def predict(image_path: Path) -> Dict[str, Any]:
  load_model()
  assert _model is not None
  tensor = _transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(DEVICE)
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
    "target_index": top_idx,
  }
