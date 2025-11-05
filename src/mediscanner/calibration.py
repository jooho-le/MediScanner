from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TemperatureState:
    temperature: float = 1.0


class TemperatureScaler(nn.Module):
    """Post-hoc temperature scaling for logits calibration.

    Usage:
      scaler = TemperatureScaler()
      scaler.fit(val_logits, val_labels)  # CrossEntropy on scaled logits
      probs = scaler.softmax(logits)      # apply at inference
    """

    def __init__(self, init_temperature: float = 1.5) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_t.exp()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        t = self.temperature
        return logits / t

    def softmax(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(logits), dim=1)

    @torch.no_grad()
    def get_state(self) -> TemperatureState:
        return TemperatureState(temperature=float(self.temperature.item()))

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, max_iter: int = 1000, lr: float = 0.01) -> None:
        self.train()
        opt = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)

        nll_criterion = nn.CrossEntropyLoss()

        def closure():
            opt.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        opt.step(closure)
        self.eval()

