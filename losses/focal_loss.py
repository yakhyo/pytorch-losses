"""Focal loss implementation"""
from typing import Optional

import torch
from torch import nn

from losses.functional import sigmoid_focal_loss

__all__ = ["FocalLoss"]


class FocalLoss(nn.Module):
    """Sigmoid Focal Loss"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward method
        Args:
            inputs: input tensors
            targets: target tensors
            weight: loss weight
        Return:
            torch.Tensor
        """
        loss = self.loss_weight * sigmoid_focal_loss(
            inputs,
            targets,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )

        return loss
