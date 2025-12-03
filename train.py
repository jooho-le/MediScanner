import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mediscanner.data import DatasetConfig, compute_class_weights, create_dataloaders
from mediscanner.engine import Trainer
from mediscanner.model import build_model
from mediscanner.utils import TrainingConfig, ensure_dir, seed_everything, select_device


def parse_args() -> TrainingConfig:
    parser = argparse.ArgumentParser(description="Train a skin lesion classifier.")
    parser.add_argument("data_dir", type=str, help="Directory with class subfolders or CSV references.")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV file with columns image_path,label for custom folder layouts.",
    )
    parser.add_argument("--model", type=str, default="convnext_base", help="Model backbone to use.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--project-dir",
        type=str,
        default="outputs",
        help="Directory where checkpoints and logs will be stored.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (AMP).",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        default="class_names.json",
        help="Filename to store class names JSON inside project-dir.",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="Enable class-balanced weights for the loss function.",
    )

    args = parser.parse_args()
    config = TrainingConfig(
        data_dir=Path(args.data_dir),
        csv_path=Path(args.csv) if args.csv else None,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        val_split=args.val_split,
        num_workers=args.num_workers,
        image_size=args.image_size,
        project_dir=Path(args.project_dir),
        class_names_path=Path(args.project_dir) / args.class_names,
        mixed_precision=not args.no_amp,
        use_class_weights=args.use_class_weights,
    )
    seed_everything(args.seed)
    return config


def main() -> None:
    config = parse_args()
    ensure_dir(config.project_dir)
    checkpoints_dir = config.project_dir / "checkpoints"
    ensure_dir(checkpoints_dir)

    dataset_cfg = DatasetConfig(
        root=config.data_dir,
        batch_size=config.batch_size,
        val_split=config.val_split,
        num_workers=config.num_workers,
        image_size=config.image_size,
        csv_path=config.csv_path,
    )

    train_loader, val_loader, class_names, train_labels, _ = create_dataloaders(dataset_cfg)
    num_classes = len(class_names)

    model, _ = build_model(config.model_name, num_classes=num_classes, dropout=config.dropout)
    device = select_device()
    model = model.to(device)

    if config.use_class_weights:
        weights = compute_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        mixed_precision=config.mixed_precision,
        scheduler=scheduler,
    )

    best_auc = 0.0
    history = []

    for epoch in range(1, config.epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
                "val_auc": val_metrics.auc,
            }
        )

        msg = (
            f"Epoch {epoch:02d}/{config.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.3f} | "
            f"val_loss={val_metrics.loss:.4f} val_acc={val_metrics.accuracy:.3f}"
        )
        if val_metrics.auc is not None:
            msg += f" val_auc={val_metrics.auc:.3f}"
        print(msg)

        if val_metrics.auc is not None and val_metrics.auc > best_auc:
            best_auc = val_metrics.auc
            best_path = checkpoints_dir / "best.pt"
            trainer.save_checkpoint(best_path, epoch=epoch, best_auc=best_auc)
            print(f"Saved best checkpoint to {best_path}")

    final_path = checkpoints_dir / "last_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Saved final weights to {final_path}")

    with open(config.class_names_path, "w", encoding="utf-8") as fp:
        json.dump(class_names, fp, indent=2)
    print(f"Stored class names at {config.class_names_path}")

    history_path = config.project_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as fp:
        json.dump(history, fp, indent=2)
    print(f"Saved training history to {history_path}")


if __name__ == "__main__":
    main()
