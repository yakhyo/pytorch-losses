from typing import Union

import torch

from _utils import ActivationFunction, LossReduction
from torch import nn
from torch.nn import functional as F

__all__ = ["DiceLoss"]


class DiceLoss(nn.Module):
    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            epsilon: float = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.activation = activation
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        if self.activation == ActivationFunction.SOFTMAX:
            inputs = torch.softmax(inputs, dim=1)
        if self.activation == ActivationFunction.SIGMOID:
            inputs = torch.sigmoid(inputs)

        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # flatten prediction and label tensors
        inputs = inputs.flatten()
        targets = targets.flatten()

        intersection = torch.sum(inputs * targets)
        denominator = torch.sum(inputs) + torch.sum(targets)

        # calculate the dice loss
        dice_coeff = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)
        loss = 1.0 - dice_coeff

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
