"""Helper function and classes"""
from enum import Enum
from typing import Optional

import torch


class LossReduction(Enum):
    """Alias for loss reduction"""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def weight_reduce_loss(
    loss: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: LossReduction = "mean",
):
    """Apply element-wise weight and reduce loss.
    Args:
        loss: element-wise loss
        weight: element-wise weight
        reduction: reduction mode
    Returns:
        torch.Tensor
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if reduction == LossReduction.MEAN:
        loss = torch.mean(loss)
    elif reduction == LossReduction.SUM:
        loss = torch.sum(loss)
    elif reduction == LossReduction.NONE:
        return loss

    return loss
