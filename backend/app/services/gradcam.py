from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from ..config import get_settings
from .inference import _model, _transform, load_resources

settings = get_settings()


def _resolve_target_layer(model: torch.nn.Module, path: str):
  module = model
  for attr in path.split("."):
    if attr.isdigit():
      module = module[int(attr)]  # type: ignore[index]
    else:
      module = getattr(module, attr)
  return module


def generate_gradcam(image_path: Path, target_class: Optional[int] = None) -> Path:
  load_resources()
  assert _model is not None
  tensor = _transform(Image.open(image_path).convert("RGB")).to(next(_model.parameters()).device)

  activations = []
  gradients = []
  target_layer = _resolve_target_layer(_model, "features")

  def forward_hook(_, __, output):
    activations.append(output.detach())

  def backward_hook(_, grad_input, grad_output):
    gradients.append(grad_output[0].detach())

  handle_f = target_layer.register_forward_hook(forward_hook)
  handle_b = target_layer.register_full_backward_hook(backward_hook)

  _model.zero_grad(set_to_none=True)
  output = _model(tensor)
  if target_class is None:
    target_class = int(output.argmax())
  score = output[0, target_class]
  score.backward()
  handle_f.remove()
  handle_b.remove()

  activation = activations[0][0].cpu().numpy()
  gradient = gradients[0][0].cpu().numpy()
  weights = gradient.mean(axis=(1, 2))
  cam = np.zeros(activation.shape[1:], dtype=np.float32)
  for w, act in zip(weights, activation):
    cam += w * act
  cam = np.maximum(cam, 0)
  cam = (cam - cam.min()) / (cam.max() + 1e-8)
  cam_img = Image.open(image_path).convert("RGB").resize(cam.shape[::-1])
  heatmap = Image.fromarray(np.uint8(255 * cam)).resize(cam_img.size)
  heatmap = heatmap.convert("RGB")
  blended = Image.blend(cam_img, heatmap, alpha=0.5)

  dest = settings.gradcam_dir / f"{image_path.stem}_gradcam.jpg"
  blended.save(dest)
  return dest
