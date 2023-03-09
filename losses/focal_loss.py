from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from utils import weight_reduce_loss

__all__ = ["FocalLoss"]


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean"
) -> torch.Tensor:
    probs = F.sigmoid(inputs)

    if inputs.shape != targets.shape:
        raise AssertionError(
            f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
        )
    pt = (1 - probs) * targets + probs * (1 - targets)
    focal_weight = (alpha * targets + (1 - alpha) * (1 - targets)) * pt.pow(gamma)

    loss = focal_weight * F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    if weight is not None:
        assert weight.dim() == 1, f"Weight dimension must be `weight.dim()=1`, current dimension {weight.dim()}"
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
            reduction: str = "mean",
            loss_weight: float = 1.0
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
    ):
        inputs = inputs.float()
        targets = targets.float()

        loss = self.loss_weight * sigmoid_focal_loss(
            inputs,
            targets,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction
        )

        return loss
