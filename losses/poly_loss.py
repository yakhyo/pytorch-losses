"""Poly loss implementation"""
import torch
from torch import nn
from torch.nn import functional as F

from losses._utils import LossReduction

__all__ = ["PolyCELoss", "SmoothPolyCELoss"]


# TODO: Add focal poly loss


class PolyCELoss(nn.Module):
    """Poly Cross Entropy Loss"""

    def __init__(self, reduction: LossReduction, epsilon: float = 1.0) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward method
        Args:
            inputs: input tensor
            targets: target tensor
        Returns:
            torch.Tensor
        """
        inputs = torch.softmax(inputs, dim=1)
        targets = F.one_hot(targets, inputs.shape[1])

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape})\
                 from input ({inputs.shape})"
            )

        ce_loss = F.cross_entropy(inputs, targets, reduction=LossReduction.NONE)
        pt = torch.sum(targets * inputs, dim=-1)

        loss = ce_loss + self.epsilon * (1 - pt)

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            pass
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction},\
                 Supported options are: 'mean', 'sum', 'none'"
            )

        return loss


class SmoothPolyCELoss(nn.Module):
    """Smooth Poly Cross Entropy Loss, alpha=0.1"""

    def __init__(
        self,
        reduction: LossReduction,
        epsilon: float = 1.0,
        alpha: float = 0.1,
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Forward method
        Args:
            inputs: input tensor
            targets: target tensor
        Returns:
            torch.Tensor
        """
        inputs = torch.softmax(inputs, dim=-1)
        targets = F.one_hot(targets, inputs.shape[1])

        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape})\
                 from input ({inputs.shape})"
            )

        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction=LossReduction.NONE,
            label_smoothing=self.alpha,
        )
        smooth_labels = (
            targets * (1 - self.alpha) + self.alpha / inputs.shape[1]
        )
        pt = torch.sum(smooth_labels * (1 - inputs), dim=-1)

        loss = ce_loss + self.epsilon * pt

        if self.reduction == LossReduction.MEAN:
            loss = torch.mean(loss)
        elif self.reduction == LossReduction.SUM:
            loss = torch.sum(loss)
        elif self.reduction == LossReduction.NONE:
            pass
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction},\
                 Supported options are: 'mean', 'sum', 'none'"
            )

        return loss
