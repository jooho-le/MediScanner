from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..core.config import get_settings
from . import inference

settings = get_settings()


def _resolve_layer(model: torch.nn.Module, path: str):
  module = model
  for attr in path.split("."):
    if attr.isdigit():
      module = module[int(attr)]  # type: ignore[index]
    else:
      module = getattr(module, attr)
  return module


def generate_gradcam(image_path: Path, target_index: Optional[int] = None) -> Path:
  # Ensure model is loaded (handles cases where predict wasn't called or different process)
  inference.load_model()
  model = inference._model  # type: ignore[attr-defined]
  if model is None:
    raise RuntimeError("Model not loaded; check MODEL_WEIGHTS path and load_model()")
  tensor = inference._transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(next(model.parameters()).device)

  activations = []
  gradients = []
  target_layer = _resolve_layer(model, "features")

  def forward_hook(_, __, output):
    activations.append(output.detach())

  def backward_hook(_, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

  h1 = target_layer.register_forward_hook(forward_hook)
  h2 = target_layer.register_full_backward_hook(backward_hook)

  model.zero_grad(set_to_none=True)
  output = model(tensor)
  if target_index is None:
    target_index = int(output.argmax())
  score = output[0, target_index]
  score.backward()
  h1.remove()
  h2.remove()

  activation = activations[0][0].cpu().numpy()
  gradient = gradients[0][0].cpu().numpy()
  weights = gradient.mean(axis=(1, 2))
  cam = np.zeros(activation.shape[1:], dtype=np.float32)
  for w, act in zip(weights, activation):
    cam += w * act
  cam = np.maximum(cam, 0)
  if cam.max() > 0:
    cam = cam / cam.max()

  base_image = Image.open(image_path).convert("RGB").resize(cam.shape[::-1])
  heatmap = Image.fromarray(np.uint8(255 * cam)).convert("RGB").resize(base_image.size)
  blended = Image.blend(base_image, heatmap, alpha=0.5)

  dest = settings.gradcam_dir / f"{image_path.stem}_gradcam.jpg"
  blended.save(dest)
  return dest
