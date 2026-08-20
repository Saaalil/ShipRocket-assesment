from __future__ import annotations

import torch
from torch import nn


class EndpointLoss(nn.Module):
    def __init__(self, pos_weight: float | None = None) -> None:
        super().__init__()
        weight = None if pos_weight is None else torch.tensor([pos_weight], dtype=torch.float32)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits.view(-1), labels.float().view(-1))
