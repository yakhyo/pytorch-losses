import torch
from torch import nn
from torch.nn import functional as F

from _utils import LossReduction, ActivationFunction
from typing import Union


class FocalLoss(nn.Module):
    def __init__(
            self,
            include_background: bool = True,
            alpha: float = 0.25,
            gamma: float = 2,
            reduction: Union[LossReduction, str] = LossReduction.NONE,
            activation: Union[ActivationFunction, str] = ActivationFunction.SIGMOID
    ):
        super().__init__()
        self.include_background = include_background
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.activation = activation

    def forwad(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.activation == ActivationFunction.SOFTMAX:
            inputs = torch.softmax(inputs, dim=1)
        if self.activation == ActivationFunction.SIGMOID:
            inputs = torch.sigmoid(inputs)

        targets = F.one_hot(targets, inputs.shape[1]).permute(0, 3, 1, 2)

        if not self.include_background:
            if inputs.shape[1] == 1:
                raise Warning(
                    "Single channel prediction, `include_background=False` ignored"
                )
            else:
                # if skipping background, removing first channel
                targets = targets[:, 1:]
                inputs = inputs[:, 1:]

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # TODO: Loss function should be implemented here

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            # If we are not computing voxel-wise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(loss.shape[0:2]) + [1] * (len(inputs.shape) - 2)
            loss = loss.view(broadcast_shape)
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction}, Supported options are: 'mean', 'sum', 'none'"
            )

        return loss

def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # Check reduction option and return loss accordingly
    if reduction == "none":
        pass
    elif reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )
    return loss
