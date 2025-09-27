import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mediscanner.model import build_model
from mediscanner.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for an image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Checkpoint file from training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnext_base",
        help="Model backbone name (must match training).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="outputs/class_names.json",
        help="JSON file listing class names.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Image resolution used during training.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional class index to explain. Defaults to the model's prediction.",
    )
    parser.add_argument(
        "--target-layer",
        type=str,
        default="features.3.2",
        help="Module path to use for Grad-CAM hooks (e.g., features.3.2).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/gradcam_overlay.jpg",
        help="Where to save the overlay image.",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> List[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def resolve_module(model: torch.nn.Module, path: str) -> torch.nn.Module:
    module = model
    for attr in path.split("."):
        if attr.isdigit():
            module = module[int(attr)]  # type: ignore[index]
        else:
            module = getattr(module, attr)
    return module


def make_heatmap(cam: np.ndarray, image: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    heat_uint8 = np.uint8(255 * cam)
    heatmap = np.stack([
        heat_uint8,
        np.zeros_like(heat_uint8),
        255 - heat_uint8,
    ], axis=-1)
    overlay = (0.6 * image + 0.4 * heatmap).clip(0, 255)
    return overlay.astype(np.uint8)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image {image_path} does not exist.")

    class_names = load_class_names(Path(args.class_names))
    model, _ = build_model(args.model, num_classes=len(class_names))

    checkpoint_path = Path(args.weights)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        model.load_state_dict(payload["model_state"])
    else:
        model.load_state_dict(payload)

    device = select_device()
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    activations = []
    gradients = []
    target_layer = resolve_module(model, args.target_layer)

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    with torch.no_grad():
        logits = model(tensor)
    if args.target_class is None:
        target_class = int(torch.softmax(logits, dim=1).argmax().item())
    else:
        target_class = args.target_class

    activations.clear()
    gradients.clear()

    output = model(tensor)
    score = output[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()

    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0][0].cpu().numpy()
    gradient = gradients[0][0].cpu().numpy()

    weights = gradient.mean(axis=(1, 2))
    cam = np.zeros(activation.shape[1:], dtype=np.float32)
    for w, act in zip(weights, activation):
        cam += w * act

    original = np.array(image.resize((args.image_size, args.image_size)))
    overlay = make_heatmap(cam, original)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    print(f"Saved Grad-CAM overlay to {output_path}")

    print("Class indices:")
    for idx, name in enumerate(class_names):
        marker = "<- target" if idx == target_class else ""
        print(f"  {idx}: {name} {marker}")


if __name__ == "__main__":
    main()
