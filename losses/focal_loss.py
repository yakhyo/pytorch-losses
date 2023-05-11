"""Focal loss implementation"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from losses._utils import LossReduction, weight_reduce_loss

__all__ = ["sigmoid_focal_loss", "FocalLoss"]


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    alpha: float = 0.25,
    reduction: LossReduction = "mean",
) -> torch.Tensor:
    """Sigmoid focal loss calculation function
    Args:
        inputs: input tensor
        targets: target tensor
        weight: loss weight
        gamma: focal weight param
        alpha: focal weight param
        reduction: reduction mode
    Returns:
        torch.Tensor
    """
    probs = torch.sigmoid(inputs)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    inputs = inputs.float()
    targets = targets.float()

    if inputs.shape != targets.shape:
        raise AssertionError(
            f"Ground truth has different shape ({targets.shape})\
             from input ({inputs.shape})"
        )
    pt = (1 - probs) * targets + probs * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(
        gamma
    )

    loss = focal_weight * F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )

    if weight is not None:
        assert (
            weight.dim() == 1
        ), f"Weight dimension must be `weight.dim()=1`,\
         current dimension {weight.dim()}"
        weight = weight.float()
        if inputs.dim() > 1:
            weight = weight.view(-1, 1)

    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss


class FocalLoss(nn.Module):
    """Sigmoid Focal Loss"""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: LossReduction = "mean",
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
        Returns:
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
