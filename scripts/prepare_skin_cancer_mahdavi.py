from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict


def parse_mapping(items: list[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            continue
        src, dst = item.split("=", 1)
        mapping[src.strip().lower()] = dst.strip().lower()
    return mapping


def main() -> None:
    ap = argparse.ArgumentParser(description="Flatten mahdavi1202/skin-cancer dataset into class folders.")
    ap.add_argument("--raw", type=str, default="data/skin_cancer_raw", help="Root path after unzip")
    ap.add_argument("--out", type=str, default="data/train_skin_cancer", help="Output root")
    ap.add_argument("--exts", type=str, default=".jpg,.png,.jpeg", help="Comma-separated extensions to include")
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        help="Optional label remap in form src=dst (e.g., malignant=mel, benign=normal)",
    )
    args = ap.parse_args()

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    exts = {e.strip().lower() for e in args.exts.split(",")}
    mapping = parse_mapping(args.map)

    copied = 0
    skipped = 0
    classes = set()

    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in exts:
            continue
        label = p.parent.name.lower()
        label = mapping.get(label, label)
        classes.add(label)
        dst_dir = out_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / p.name
        if dst.exists():
            continue
        shutil.copy(p, dst)
        copied += 1

    print(f"Copied {copied} files into {out_root}, skipped {skipped}")
    print("Classes found:", sorted(classes))


if __name__ == "__main__":
    main()
