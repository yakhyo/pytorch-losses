from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from losses.functional import dice_loss

__all__ = ["DiceLoss", "DiceCELoss"]


class DiceLoss(nn.Module):
    def __init__(
            self,
            reduction: str = "mean",
            loss_weight: Optional[float] = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * dice_loss(
            inputs, targets, weight=weight, reduction=self.reduction, eps=self.eps
        )

        return loss


class DiceCELoss(nn.Module):
    def __init__(
            self,
            reduction: str = "mean",
            dice_weight: float = 1.0,
            ce_weight: float = 1.0,
            eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(
            self,
            inputs: torch.Tensor,
            targets: torch.Tensor,
            weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # calculate dice loss
        dice = dice_loss(
            inputs, targets, weight=weight, reduction=self.reduction, eps=self.eps
        )
        # calculate cross entropy loss
        ce = F.cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)
        # accumulate loss according to given weights
        loss = self.dice_weight * dice + ce * self.ce_weight

        return loss
