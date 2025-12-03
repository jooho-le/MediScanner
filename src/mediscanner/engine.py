from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .metrics import EvaluationResult, compute_metrics


@dataclass
class Checkpoint:
    epoch: int
    model_state: Dict[str, torch.Tensor]
    optimizer_state: Dict[str, torch.Tensor]
    scaler_state: Optional[Dict[str, torch.Tensor]]
    best_auc: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        mixed_precision: bool = True,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        # Enable AMP only on CUDA
        self.mixed_precision = bool(mixed_precision and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            # autocast is only meaningful on CUDA here
            with torch.cuda.amp.autocast(enabled=self.mixed_precision):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            preds = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += images.size(0)

        if self.scheduler:
            self.scheduler.step()

        return {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
        }

    def evaluate(self, loader: DataLoader) -> EvaluationResult:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        predictions = []
        probabilities = []
        labels_list = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(dim=1)

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        loss = total_loss / total_samples
        eval_result = compute_metrics(
            labels=np.array(labels_list),
            predictions=np.array(predictions),
            probabilities=np.array(probabilities),
            loss=loss,
        )
        return eval_result

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        best_auc: float,
    ) -> None:
        checkpoint = Checkpoint(
            epoch=epoch,
            model_state=self.model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            scaler_state=self.scaler.state_dict() if self.mixed_precision else None,
            best_auc=best_auc,
        )
        torch.save(checkpoint.__dict__, path)

    def load_checkpoint(self, path: Path) -> Checkpoint:
        payload = torch.load(path, map_location=self.device)
        checkpoint = Checkpoint(**payload)
        self.model.load_state_dict(checkpoint.model_state)
        self.optimizer.load_state_dict(checkpoint.optimizer_state)
        if self.mixed_precision and checkpoint.scaler_state:
            self.scaler.load_state_dict(checkpoint.scaler_state)
        return checkpoint
