from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch


@dataclass
class TrainingConfig:
    data_dir: Path
    csv_path: Optional[Path]
    model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    dropout: float
    val_split: float
    num_workers: int
    image_size: int
    project_dir: Path
    class_names_path: Path
    mixed_precision: bool = True
    use_class_weights: bool = False


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
