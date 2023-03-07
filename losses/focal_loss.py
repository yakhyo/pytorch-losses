from typing import Union

import torch

from _utils import ActivationFunction, LossReduction
from torch import nn

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module):
    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.NONE,
            alpha: float = 0.25,
            gamma: float = 2,
            activation: Union[ActivationFunction, str] = ActivationFunction.SIGMOID,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.activation = activation
        self.bce = nn.BCELoss(reduction=LossReduction.NONE)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        targets = targets.float()

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        prob = torch.sigmoid(inputs)
        ce_loss = self.bce(inputs, targets)

        pt = prob * targets + (1 - prob) * (1 - targets)

        alpha_factor = 1.0
        modulating_factor = 1.0

        if self.alpha:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        if self.gamma:
            modulating_factor = (1 - pt) ** self.gamma

        loss = ce_loss * alpha_factor * modulating_factor

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
