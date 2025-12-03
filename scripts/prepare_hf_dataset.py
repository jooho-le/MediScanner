from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from PIL import Image


def infer_label_name(label_value, label_feature) -> str:
    if label_feature is not None and hasattr(label_feature, "names"):
        # integer label mapped to class name
        if isinstance(label_value, int):
            return label_feature.names[label_value]
    return str(label_value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset split and export images into class folders."
    )
    parser.add_argument("dataset", type=str, help="Dataset name, e.g. 'nateraw/skin_cancer_mnist'")
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name (if the dataset exposes multiple configs).",
    )
    parser.add_argument("--split", type=str, default="train", help="Dataset split to fetch.")
    parser.add_argument(
        "--out",
        type=str,
        default="data/hf_export",
        help="Output directory (will contain class subfolders).",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="image",
        help="Column holding images (defaults to 'image').",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Column holding labels (defaults to 'label').",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap to limit how many rows are exported.",
    )
    args = parser.parse_args()

    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.image_column not in ds.column_names:
        raise KeyError(f"Image column '{args.image_column}' not found. Available: {ds.column_names}")
    if args.label_column not in ds.column_names:
        raise KeyError(f"Label column '{args.label_column}' not found. Available: {ds.column_names}")

    label_feature = ds.features.get(args.label_column)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    total = 0
    for idx, row in enumerate(ds):
        if args.max_samples is not None and idx >= args.max_samples:
            break

        image_value = row[args.image_column]
        label_value = row[args.label_column]
        if image_value is None:
            continue

        label_name = infer_label_name(label_value, label_feature)
        class_dir = out_root / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # convert to PIL Image (HF datasets may provide dict or PIL already)
        if isinstance(image_value, Image.Image):
            pil_img = image_value
        else:
            pil_img = Image.fromarray(image_value)

        filename = f"{uuid.uuid4().hex}.jpg"
        save_path = class_dir / filename
        pil_img.save(save_path)
        total += 1

    print(f"Exported {total} samples from {args.dataset}::{args.split} to {out_root}")


if __name__ == "__main__":
    main()
