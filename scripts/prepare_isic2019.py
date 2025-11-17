from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd


def find_paths(raw_root: Path) -> tuple[Path, Path]:
    # Locate ground-truth CSV
    csv_candidates = [
        raw_root / "ISIC_2019_Training_GroundTruth.csv",
        raw_root / "ISIC_2019_Training_GroundTruth_v2.csv",
    ] + list(raw_root.glob("*GroundTruth*.csv"))
    gt_csv = next((p for p in csv_candidates if p.exists()), None)
    if gt_csv is None:
        raise FileNotFoundError("Could not find ISIC_2019_Training_GroundTruth.csv under raw root")

    # Locate image directory
    img_dir = raw_root / "ISIC_2019_Training_Input"
    if not img_dir.exists():
        # try to find a folder with 'Training_Input'
        alts = [d for d in raw_root.iterdir() if d.is_dir() and "training_input" in d.name.lower()]
        if not alts:
            raise FileNotFoundError("Could not find ISIC_2019_Training_Input directory under raw root")
        img_dir = alts[0]
    return gt_csv, img_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ISIC2019 into 7-class folders (SCC excluded)")
    ap.add_argument("--raw", type=str, default="data/isic2019_raw", help="Path to raw ISIC2019 root")
    ap.add_argument("--out", type=str, default="data/train_isic2019", help="Output folder root")
    ap.add_argument("--include-scc", action="store_true", help="Include SCC as 8th class (default: exclude)")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    gt_csv, img_dir = find_paths(raw_root)
    df = pd.read_csv(gt_csv)

    # Expected one-hot columns
    cols = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}' in {gt_csv}")

    map7 = {"MEL": "mel", "NV": "nv", "BCC": "bcc", "AK": "akiec", "BKL": "bkl", "DF": "df", "VASC": "vasc", "SCC": "scc"}

    copied = defaultdict(int)
    missed = 0

    for _, r in df.iterrows():
        # pick active label
        labs = [c for c in cols if int(r[c]) == 1]
        if not labs:
            continue
        lab = labs[0]
        if lab == "SCC" and not args.include_scc:
            continue

        cls = map7[lab]
        img_id = str(r["image"]) if "image" in r else str(r["Image"])
        src = None
        for ext in (".jpg", ".JPG", ".png"):
            p = img_dir / f"{img_id}{ext}"
            if p.exists():
                src = p
                break
        if src is None:
            missed += 1
            continue

        dst_dir = out_root / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if not dst.exists():
            shutil.copy(src, dst)
            copied[cls] += 1

    print(f"Done. Copied: {dict(copied)} | missed: {missed} | out: {out_root}")


if __name__ == "__main__":
    main()

