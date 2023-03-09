from typing import Union

import torch
from torch import nn
from torch.nn import functional as F

from utils import LossReduction

__all__ = ["BinaryFocalLoss"]


class BinaryFocalLoss(nn.Module):
    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.NONE,
            alpha: float = 0.25,
            gamma: float = 2,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.float()

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        prob = torch.sigmoid(inputs)
        bce = F.binary_cross_entropy(inputs, targets, reduction=LossReduction.NONE)

        pt = prob * targets + (1 - prob) * (1 - targets)

        alpha_factor = 1.0
        modulating_factor = 1.0

        if self.alpha:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        if self.gamma:
            modulating_factor = (1 - pt) ** self.gamma

        loss = bce * alpha_factor * modulating_factor  # focal loss

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            pass
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction}, Supported options are: 'mean', 'sum', 'none'"
            )

        return loss


class SigmoidFocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        pass
