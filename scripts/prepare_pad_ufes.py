from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def find_metadata(raw_root: Path) -> Path:
    cands = list(raw_root.glob("*.csv")) + list(raw_root.glob("**/metadata*.csv"))
    if not cands:
        raise FileNotFoundError("metadata CSV not found under raw root")
    return cands[0]


def find_image_dir(raw_root: Path) -> Path:
    for name in ["images", "imgs", "skin_images", "pad_ufes_images"]:
        p = raw_root / name
        if p.exists() and p.is_dir():
            return p
    # fallback: first dir with jpg files
    for p in raw_root.iterdir():
        if p.is_dir() and list(p.glob("*.jpg")):
            return p
    raise FileNotFoundError("image directory not found under raw root")


def build_mapping(include_normal: bool = True) -> Dict[str, str]:
    mapping = {
        "basal cell carcinoma": "bcc",
        "bcc": "bcc",
        "benign keratosis-like lesions": "bkl",
        "bkl": "bkl",
        "dermatofibroma": "df",
        "df": "df",
        "melanoma": "mel",
        "mel": "mel",
        "nevus": "nv",
        "nv": "nv",
        "squamous cell carcinoma": "scc",
        "scc": "scc",
        "vascular lesion": "vasc",
        "vasc": "vasc",
        "actinic keratosis": "akiec",
        "akiec": "akiec",
        "bowen disease": "akiec",
    }
    if include_normal:
        mapping["normal"] = "normal"
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare PAD-UFES-20 dataset into class folders (incl. normal).")
    ap.add_argument("--raw", type=str, default="data/pad_ufes", help="Root containing metadata.csv and images/")
    ap.add_argument("--out", type=str, default="data/train_pad", help="Output root for class folders")
    ap.add_argument("--include-normal", action="store_true", help="Include rows labeled normal (default off)")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    meta_path = find_metadata(raw_root)
    img_dir = find_image_dir(raw_root)

    df = pd.read_csv(meta_path)
    # detect label column
    label_col = None
    for cand in ["diagnostic", "diagnosis", "dx"]:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        raise KeyError(f"Label column not found in {meta_path}. Expected one of diagnostic/diagnosis/dx")

    # detect image column
    img_col = None
    for cand in ["image_id", "img_id", "image", "image_name", "file"]:
        if cand in df.columns:
            img_col = cand
            break
    if img_col is None:
        raise KeyError(f"Image column not found in {meta_path}. Expected image_id/img_id/image/image_name/file")

    mapping = build_mapping(include_normal=args.include_normal)
    copied = 0
    skipped = 0

    for _, row in df.iterrows():
        raw_label = str(row[label_col]).strip().lower()
        cls = mapping.get(raw_label)
        if cls is None:
            skipped += 1
            continue
        fname = str(row[img_col])
        # ensure extension
        if not Path(fname).suffix:
            fname = f"{fname}.jpg"
        src = img_dir / fname
        if not src.exists():
            # try png
            alt = src.with_suffix(".png")
            if alt.exists():
                src = alt
            else:
                skipped += 1
                continue
        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            copied += 1

    print(f"Copied {copied} files into {out_root}, skipped {skipped} (missing/unknown labels).")
    print(f"Classes created: {[p.name for p in out_root.iterdir() if p.is_dir()]}")


if __name__ == "__main__":
    main()
