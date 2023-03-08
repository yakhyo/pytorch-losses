from typing import Union

import torch
import torch.nn as nn
from _utils import LossReduction

__all__ = ["AsymmetricLoss", "AsymmetricLossOptimized", "ASLSingleLabel"]


class AsymmetricLoss(nn.Module):
    def __init__(
            self,
            gamma_neg: int = 4,
            gamma_pos: int = 1,
            clip: float = 0.05,
            epsilon: float = 1e-8,
            disable_torch_grad_focal_loss: bool = True,
    ) -> None:
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = targets * torch.log(xs_pos.clamp(min=self.epsilon))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.epsilon))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * targets
            pt1 = xs_neg * (1 - targets)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -torch.sum(loss)


class AsymmetricLossOptimized(nn.Module):
    """Minimizes memory allocation and gpu uploading, favors inplace operations"""

    def __init__(
            self,
            gamma_neg: int = 4,
            gamma_pos: int = 1,
            clip: float = 0.05,
            epsilon: float = 1e-8,
            disable_torch_grad_focal_loss: bool = False,
    ) -> None:
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.epsilon = epsilon

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = (
            self.anti_targets
        ) = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.shape != inputs.shape:
            raise AssertionError(
                f"Ground truth has different shape ({targets.shape}) from input ({inputs.shape})"
            )

        self.targets = targets
        self.anti_targets = 1 - targets

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(inputs)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.epsilon))
        self.loss.add_(
            self.anti_targets * torch.log(self.xs_neg.clamp(min=self.epsilon))
        )

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(
                1 - self.xs_pos - self.xs_neg,
                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets,
            )
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -torch.sum(self.loss)


class ASLSingleLabel(nn.Module):
    """This loss is intended for single-label classification problems"""

    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            gamma_pos: int = 0,
            gamma_neg: int = 4,
            epsilon: float = 0.1,
    ):

        super(ASLSingleLabel, self).__init__()

        self.epsilon = epsilon
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):

        log_preds = torch.log_softmax(inputs, dim=-1)
        self.targets_classes = torch.zeros_like(inputs).scatter_(
            1, target.long().unsqueeze(1), 1
        )

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(
            1 - xs_pos - xs_neg,
            self.gamma_pos * targets + self.gamma_neg * anti_targets,
        )
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.epsilon).add(
                self.epsilon / inputs.size()[-1]
            )

        # loss calculation
        loss = -self.targets_classes.mul(log_preds)

        loss = torch.sum(loss, dim=-1)
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
