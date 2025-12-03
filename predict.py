import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from mediscanner.model import build_model, build_multitask_model
from mediscanner.calibration import TemperatureScaler
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
        "--multitask",
        action="store_true",
        help="Enable multitask inference (disease + optional severity/infectious/urgent).",
    )
    parser.add_argument(
        "--severity-classes",
        type=int,
        default=3,
        help="Number of severity classes if multitask is enabled.",
    )
    parser.add_argument(
        "--apply-temperature",
        type=float,
        default=None,
        help="Optional temperature value for softmax calibration (e.g., 1.5).",
    )
    parser.add_argument(
        "--reject-threshold",
        type=float,
        default=None,
        help="If top probability is below this, output 'uncertain/OOD' instead of a class.",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=None,
        help="If predictive entropy (natural log) exceeds this, mark as uncertain.",
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
    if args.multitask:
        model, _ = build_multitask_model(
            args.model,
            disease_classes=len(class_names),
            severity_classes=args.severity_classes,
            enable_infectious=True,
            enable_urgent=True,
        )
    else:
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
        out = model(tensor)

    if not args.multitask:
        logits = out
        if args.apply_temperature:
            scaler = TemperatureScaler(init_temperature=args.apply_temperature)
            with torch.no_grad():
                probs = scaler.softmax(logits).detach().cpu().numpy()[0]
        else:
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        top_idx = probs.argmax()
        top_prob = float(probs[top_idx])
        uncertain = False
        if args.reject_threshold is not None and top_prob < args.reject_threshold:
            uncertain = True
        if args.entropy_threshold is not None:
            import numpy as np
            ent = float(-(probs * (np.log(probs + 1e-12))).sum())
            if ent > args.entropy_threshold:
                uncertain = True
        for cls, prob in zip(class_names, probs):
            print(f"{cls}: {prob:.3f}")
        if uncertain:
            print("Result: uncertain/OOD — possible no lesion or distribution shift")
        else:
            print(f"Predicted class: {class_names[top_idx]}")
        return

    # Multitask formatting
    disease_logits = out["disease"]
    if args.apply_temperature:
        scaler = TemperatureScaler(init_temperature=args.apply_temperature)
        with torch.no_grad():
            disease_probs = scaler.softmax(disease_logits).detach().cpu().numpy()[0]
    else:
        disease_probs = torch.softmax(disease_logits, dim=1).detach().cpu().numpy()[0]
    disease_idx = int(disease_probs.argmax())
    top_prob = float(disease_probs[disease_idx])
    uncertain = False
    if args.reject_threshold is not None and top_prob < args.reject_threshold:
        uncertain = True
    if args.entropy_threshold is not None:
        import numpy as np
        ent = float(-(disease_probs * (np.log(disease_probs + 1e-12))).sum())
        if ent > args.entropy_threshold:
            uncertain = True

    severity_probs = None
    if "severity" in out:
        sev_logits = out["severity"]
        if args.apply_temperature:
            scaler_s = TemperatureScaler(init_temperature=args.apply_temperature)
            with torch.no_grad():
                severity_probs = scaler_s.softmax(sev_logits).detach().cpu().numpy()[0]
        else:
            severity_probs = torch.softmax(sev_logits, dim=1).detach().cpu().numpy()[0]

    infectious_prob = None
    if "infectious" in out:
        infectious_prob = torch.sigmoid(out["infectious"]).detach().cpu().numpy()[0].item()

    urgent_prob = None
    if "urgent" in out:
        urgent_prob = torch.sigmoid(out["urgent"]).detach().cpu().numpy()[0].item()

    # Simple risk/triage heuristics
    top_class = class_names[disease_idx]
    sev_idx = int(severity_probs.argmax()) if severity_probs is not None else None
    sev_names = ["low", "moderate", "high"][: args.severity_classes]
    sev_name = (sev_names[sev_idx] if sev_idx is not None else None)

    risk = "low"
    reasons = []
    if urgent_prob is not None and urgent_prob >= 0.5:
        risk = "high"; reasons.append("urgent pattern")
    if infectious_prob is not None and infectious_prob >= 0.5:
        risk = "high"; reasons.append("infectious suspicion")
    if sev_name == "high":
        risk = "high"; reasons.append("severity=high")
    elif sev_name == "moderate" and risk != "high":
        risk = "medium"; reasons.append("severity=moderate")
    if top_prob >= 0.5:
        if risk != "high":
            risk = "medium"
        reasons.append(f"high confidence in {top_class}")

    if uncertain:
        referral = "Uncertain: re-capture image or seek clinical evaluation"
    elif risk == "high":
        referral = "Immediate dermatology visit recommended"
    elif risk == "medium":
        referral = "Dermatology visit within 1 week recommended"
    else:
        referral = "Self-monitor; seek care if changes occur"

    # Output
    print("Disease probabilities:")
    for cls, prob in zip(class_names, disease_probs):
        print(f"- {cls}: {prob:.3f}")
    if uncertain:
        print("Predicted disease: uncertain/OOD")
    else:
        print(f"Predicted disease: {top_class} (p={top_prob:.2f})")
    if severity_probs is not None:
        for i, p in enumerate(severity_probs):
            name = sev_names[i] if i < len(sev_names) else f"sev_{i}"
            print(f"Severity {name}: {p:.3f}")
        print(f"Predicted severity: {sev_name}")
    if infectious_prob is not None:
        print(f"Infectious probability: {infectious_prob:.3f}")
    if urgent_prob is not None:
        print(f"Urgent probability: {urgent_prob:.3f}")
    print(f"Risk level: {risk}")
    if reasons:
        print("Reasons: " + ", ".join(reasons))
    print(f"Referral: {referral}")


if __name__ == "__main__":
    main()
