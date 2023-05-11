"""Cross entropy loss implementation"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from losses._utils import LossReduction, weight_reduce_loss


def cross_entropy(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    class_weight: Optional[torch.Tensor] = None,
    reduction: LossReduction = "mean",
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross entropy loss calculation function
    Args:
        inputs: input tensor
        targets: target tensor
        weight: loss weight
        class_weight: class weights
        reduction: reduction mode
        ignore_index: cross entropy param
    Returns:
        torch.Tensor
    """
    loss = F.cross_entropy(
        inputs,
        targets,
        weight=class_weight,
        reduction="none",
        ignore_index=ignore_index,
    )

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction)

    return loss


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: LossReduction = "mean",
        loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.class_weight = class_weights
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ) -> torch.Tensor:
        """Forward method
        Args:
            inputs: input tensor
            targets: target tensor
            weight: loss weight
            ignore_index: cross entropy param
        Returns:
             torch.Tensor
        """
        loss = self.loss_weight * cross_entropy(
            inputs,
            targets,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            ignore_index=ignore_index,
        )

        return loss
