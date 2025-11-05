import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from mediscanner.model import build_multitask_model
from mediscanner.utils import ensure_dir, seed_everything, select_device


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        root: Path,
        image_col: str,
        disease_col: str,
        disease_class_to_idx: Dict[str, int],
        severity_col: Optional[str] = None,
        infectious_col: Optional[str] = None,
        urgent_col: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.root = root
        self.image_col = image_col
        self.disease_col = disease_col
        self.severity_col = severity_col
        self.infectious_col = infectious_col
        self.urgent_col = urgent_col
        self.transform = transform
        self.disease_class_to_idx = disease_class_to_idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = (self.root / str(row[self.image_col])).resolve()
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        disease_name = str(row[self.disease_col])
        disease_label = self.disease_class_to_idx[disease_name]

        target: Dict[str, torch.Tensor] = {"disease": torch.tensor(disease_label, dtype=torch.long)}
        if self.severity_col:
            target["severity"] = torch.tensor(int(row[self.severity_col]), dtype=torch.long)
        if self.infectious_col:
            target["infectious"] = torch.tensor(int(row[self.infectious_col]), dtype=torch.float32)
        if self.urgent_col:
            target["urgent"] = torch.tensor(int(row[self.urgent_col]), dtype=torch.float32)

        return image, target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multitask skin lesion model (disease/severity/flags)")
    p.add_argument("data_root", type=str, help="Root directory for image paths in CSV")
    p.add_argument("csv", type=str, help="CSV with labels")
    p.add_argument("--model", type=str, default="convnext_base")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--project-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true")

    # columns
    p.add_argument("--image-column", type=str, default="image_path")
    p.add_argument("--disease-column", type=str, default="label")
    p.add_argument("--severity-column", type=str, default=None)
    p.add_argument("--infectious-column", type=str, default=None)
    p.add_argument("--urgent-column", type=str, default=None)

    # loss weights
    p.add_argument("--w-disease", type=float, default=1.0)
    p.add_argument("--w-severity", type=float, default=0.5)
    p.add_argument("--w-infectious", type=float, default=0.25)
    p.add_argument("--w-urgent", type=float, default=0.25)

    # class names IO
    p.add_argument("--disease-class-names", type=str, default="class_names.json")
    p.add_argument("--severity-classes", type=int, default=3)

    return p.parse_args()


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomRotation(degrees=20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tfms, val_tfms


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    project_dir = Path(args.project_dir)
    ensure_dir(project_dir)
    checkpoints_dir = project_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    # Load CSV
    df = pd.read_csv(args.csv)
    # train/val split
    val_frac = args.val_split
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_size = int(len(df) * val_frac)
    df_val = df.iloc[:val_size]
    df_train = df.iloc[val_size:]

    # Disease classes
    disease_names = sorted(df[args.disease_column].unique().tolist())
    disease_to_idx = {n: i for i, n in enumerate(disease_names)}
    with open(project_dir / args.disease_class_names, "w", encoding="utf-8") as fp:
        json.dump(disease_names, fp, indent=2)

    train_tfms, val_tfms = build_transforms(args.image_size)

    train_ds = MultiTaskDataset(
        df_train,
        root=Path(args.data_root),
        image_col=args.image_column,
        disease_col=args.disease_column,
        disease_class_to_idx=disease_to_idx,
        severity_col=args.severity_column,
        infectious_col=args.infectious_column,
        urgent_col=args.urgent_column,
        transform=train_tfms,
    )
    val_ds = MultiTaskDataset(
        df_val,
        root=Path(args.data_root),
        image_col=args.image_column,
        disease_col=args.disease_column,
        disease_class_to_idx=disease_to_idx,
        severity_col=args.severity_column,
        infectious_col=args.infectious_column,
        urgent_col=args.urgent_column,
        transform=val_tfms,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model, _ = build_multitask_model(
        model_name=args.model,
        disease_classes=len(disease_names),
        dropout=args.dropout,
        severity_classes=(args.severity_classes if args.severity_column else None),
        enable_infectious=bool(args.infectious_column),
        enable_urgent=bool(args.urgent_column),
    )

    device = select_device()
    model = model.to(device)
    mixed_precision = not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    def step(batch, train: bool) -> Tuple[float, Dict[str, float]]:
        images, targets = batch
        images = images.to(device, non_blocking=True)
        t_disease = targets["disease"].to(device)
        t_sev = targets.get("severity")
        t_inf = targets.get("infectious")
        t_urg = targets.get("urgent")
        if t_sev is not None:
            t_sev = t_sev.to(device)
        if t_inf is not None:
            t_inf = t_inf.to(device)
        if t_urg is not None:
            t_urg = t_urg.to(device)

        with torch.cuda.amp.autocast(enabled=mixed_precision):
            outputs = model(images)
            loss = args.w_disease * ce(outputs["disease"], t_disease)
            if t_sev is not None and "severity" in outputs:
                loss = loss + args.w_severity * ce(outputs["severity"], t_sev)
            if t_inf is not None and "infectious" in outputs:
                loss = loss + args.w_infectious * bce(outputs["infectious"], t_inf)
            if t_urg is not None and "urgent" in outputs:
                loss = loss + args.w_urgent * bce(outputs["urgent"], t_urg)

        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # simple accuracies
        with torch.no_grad():
            metrics: Dict[str, float] = {}
            preds = outputs["disease"].argmax(dim=1)
            metrics["acc_disease"] = (preds == t_disease).float().mean().item()
            if t_sev is not None and "severity" in outputs:
                preds_sev = outputs["severity"].argmax(dim=1)
                metrics["acc_severity"] = (preds_sev == t_sev).float().mean().item()
            if t_inf is not None and "infectious" in outputs:
                metrics["acc_infectious"] = (((outputs["infectious"] > 0).float() == t_inf).float().mean().item())
            if t_urg is not None and "urgent" in outputs:
                metrics["acc_urgent"] = (((outputs["urgent"] > 0).float() == t_urg).float().mean().item())

        return loss.item(), metrics

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot_loss = 0.0
        count = 0
        agg: Dict[str, float] = {}
        for batch in train_loader:
            loss, m = step(batch, train=True)
            bs = batch[0].size(0)
            tot_loss += loss * bs
            count += bs
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v * bs
        train_log = {k: v / count for k, v in agg.items()}
        train_log["loss"] = tot_loss / count

        model.eval()
        tot_loss = 0.0
        count = 0
        agg = {}
        with torch.no_grad():
            for batch in val_loader:
                loss, m = step(batch, train=False)
                bs = batch[0].size(0)
                tot_loss += loss * bs
                count += bs
                for k, v in m.items():
                    agg[k] = agg.get(k, 0.0) + v * bs
        val_log = {k: v / count for k, v in agg.items()}
        val_log["loss"] = tot_loss / count

        msg = (
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_log['loss']:.4f} "
            + " ".join([f"train_{k}={v:.3f}" for k, v in train_log.items() if k != 'loss'])
            + " | "
            + f"val_loss={val_log['loss']:.4f} "
            + " ".join([f"val_{k}={v:.3f}" for k, v in val_log.items() if k != 'loss'])
        )
        print(msg)

        # simple checkpoint on disease acc
        acc = val_log.get("acc_disease", 0.0)
        if acc > best_acc:
            best_acc = acc
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "best_acc_disease": best_acc,
            }
            path = checkpoints_dir / "best_multitask.pt"
            torch.save(ckpt, path)
            print(f"Saved best checkpoint to {path}")

    # save last
    last = checkpoints_dir / "last_multitask.pt"
    torch.save(model.state_dict(), last)
    print(f"Saved last weights to {last}")


if __name__ == "__main__":
    main()

