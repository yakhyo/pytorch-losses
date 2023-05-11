"""All loss functions"""
from typing import Optional

import torch
from torch.nn import functional as F

from losses._utils import LossReduction, weight_reduce_loss

__all__ = ["dice_loss", "sigmoid_focal_loss", "cross_entropy"]


def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: LossReduction = "none",
    eps: float = 1e-5,
) -> torch.Tensor:
    """Dice loss calculation function
    Args:
        inputs: input tensor
        targets: target tensor
        weight: loss weight
        reduction: reduction mode
        eps: epsilon to avoid zero division
    Return:
        torch.Tensor
    """
    inputs = F.softmax(inputs, dim=1)
    targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

    if inputs.shape != targets.shape:
        raise AssertionError(
            f"Ground truth has different shape ({targets.shape})\
             from input ({inputs.shape})"
        )

    # flatten prediction and label tensors
    inputs = inputs.flatten()
    targets = targets.flatten()

    intersection = torch.sum(inputs * targets)
    denominator = torch.sum(inputs) + torch.sum(targets)

    # calculate the dice loss
    dice_score = (2.0 * intersection + eps) / (denominator + eps)
    loss = 1 - dice_score

    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(inputs)
    loss = weight_reduce_loss(loss, weight, reduction=reduction)

    return loss


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
    Return:
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
    Return:
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
