import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from mediscanner.model import build_model
from mediscanner.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on a single image.")
    parser.add_argument("image", type=str, help="Path to the input image.")
    parser.add_argument(
        "--weights",
        type=str,
        default="outputs/checkpoints/best.pt",
        help="Checkpoint file produced during training.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="convnext_base",
        help="Model backbone (must match training).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Image resolution expected by the model (must match training).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="outputs/class_names.json",
        help="JSON file containing class names in order.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, e.g. cpu.",
    )
    return parser.parse_args()


def load_class_names(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image {image_path} does not exist.")

    class_names = load_class_names(Path(args.class_names))
    model, _ = build_model(args.model, num_classes=len(class_names))

    checkpoint_path = Path(args.weights)
    if checkpoint_path.suffix == ".pt":
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)
    else:
        raise ValueError("Weights must be a .pt checkpoint file.")

    device = torch.device(args.device) if args.device else select_device()
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = probs.argmax()
    for cls, prob in zip(class_names, probs):
        print(f"{cls}: {prob:.3f}")
    print(f"Predicted class: {class_names[top_idx]}")


if __name__ == "__main__":
    main()
