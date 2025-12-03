from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


@dataclass
class DatasetConfig:
    root: Path
    batch_size: int = 16
    val_split: float = 0.2
    random_state: int = 42
    num_workers: int = 4
    image_size: int = 384
    csv_path: Optional[Path] = None
    label_column: str = "label"
    image_column: str = "image_path"


class SkinLesionDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Simple dataset that loads RGB images and integer labels."""

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError("Paths and labels must have identical lengths.")
        self.paths: List[Path] = list(paths)
        self.labels: List[int] = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def _default_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
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


def _gather_from_folders(root: Path) -> Tuple[List[Path], List[int], List[str]]:
    paths: List[Path] = []
    labels: List[int] = []
    class_names: List[str] = []

    for idx, class_dir in enumerate(sorted(p for p in root.iterdir() if p.is_dir())):
        class_names.append(class_dir.name)
        for image_path in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
            paths.append(image_path)
            labels.append(idx)

    if not paths:
        raise FileNotFoundError(
            f"No images found under {root}. Expected structure root/class_name/image.jpg"
        )
    return paths, labels, class_names


def _gather_from_csv(config: DatasetConfig) -> Tuple[List[Path], List[int], List[str]]:
    if config.csv_path is None:
        raise ValueError("csv_path must be provided when using CSV metadata.")

    df = pd.read_csv(config.csv_path)
    if config.image_column not in df.columns or config.label_column not in df.columns:
        raise KeyError(
            f"CSV must contain '{config.image_column}' and '{config.label_column}' columns."
        )

    class_names = sorted(df[config.label_column].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    paths = []
    labels = []
    for _, row in df.iterrows():
        path = (config.root / str(row[config.image_column])).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file {path} referenced in CSV does not exist.")
        paths.append(path)
        labels.append(class_to_idx[row[config.label_column]])

    return paths, labels, class_names


def create_dataloaders(config: DatasetConfig) -> Tuple[
    DataLoader,
    DataLoader,
    List[str],
    List[int],
    List[int],
]:
    if config.csv_path:
        paths, labels, class_names = _gather_from_csv(config)
    else:
        paths, labels, class_names = _gather_from_folders(config.root)

    train_tfms, val_tfms = _default_transforms(config.image_size)

    train_idx, val_idx = train_test_split(
        range(len(paths)),
        test_size=config.val_split,
        stratify=labels,
        random_state=config.random_state,
    )

    train_dataset = SkinLesionDataset(
        [paths[i] for i in train_idx],
        [labels[i] for i in train_idx],
        transform=train_tfms,
    )
    val_dataset = SkinLesionDataset(
        [paths[i] for i in val_idx],
        [labels[i] for i in val_idx],
        transform=val_tfms,
    )

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin,
    )

    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    return train_loader, val_loader, class_names, train_labels, val_labels


def compute_class_weights(labels: Iterable[int]) -> torch.Tensor:
    labels_list = list(labels)
    unique = torch.tensor(sorted(set(labels_list)), dtype=torch.long)
    counts = torch.tensor([labels_list.count(int(cls)) for cls in unique], dtype=torch.float32)
    weights = counts.sum() / (len(unique) * counts)
    return weights
