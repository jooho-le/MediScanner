from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import pandas as pd


def find_paths(raw_root: Path) -> tuple[Path, list[Path]]:
    # Find metadata CSV
    meta = raw_root / "HAM10000_metadata.csv"
    if not meta.exists():
        cands = list(raw_root.glob("*metadata*.csv"))
        if not cands:
            raise FileNotFoundError("Could not find HAM10000_metadata.csv under data/raw")
        meta = cands[0]

    # Find image folders
    img_dirs: list[Path] = []
    for name in ["HAM10000_images_part_1", "HAM10000_images_part_2"]:
        p = raw_root / name
        if p.exists():
            img_dirs.append(p)
    if not img_dirs:
        img_dirs = [d for d in raw_root.iterdir() if d.is_dir() and "images_part" in d.name.lower()]
    if not img_dirs:
        raise FileNotFoundError("Could not find image folders (HAM10000_images_part_*) under data/raw")
    return meta, img_dirs


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare HAM10000 into data/train/<class>/ image folders")
    ap.add_argument("--raw", type=str, default="data/raw", help="Path to raw HAM10000 download root")
    ap.add_argument("--out", type=str, default="data/train", help="Output folder root")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_csv, img_dirs = find_paths(raw_root)
    meta = pd.read_csv(meta_csv)

    count = 0
    per_class: dict[str, int] = {}
    for _, r in meta.iterrows():
        img_id = str(r["image_id"])  # e.g., ISIC_002xxxx
        dx = str(r["dx"])            # class name

        src = None
        for d in img_dirs:
            for ext in (".jpg", ".JPG", ".png"):
                p = d / f"{img_id}{ext}"
                if p.exists():
                    src = p
                    break
            if src is not None:
                break
        if src is None:
            continue

        cls_dir = out_root / dx
        cls_dir.mkdir(parents=True, exist_ok=True)
        dst = cls_dir / f"{img_id}.jpg"
        if not dst.exists():
            shutil.copy(src, dst)
            count += 1
            per_class[dx] = per_class.get(dx, 0) + 1

    print(f"Done. Copied {count} files into {out_root}")
    for k in sorted(per_class):
        print(f"- {k}: {per_class[k]}")


if __name__ == "__main__":
    main()

