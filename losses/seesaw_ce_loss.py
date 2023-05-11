"""Seesaw cross entropy loss"""
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from losses._utils import LossReduction, weight_reduce_loss

__all__ = ["seesaw_ce_loss"]


def seesaw_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor,
    cum_samples: torch.Tensor,
    num_classes: int,
    p: float,
    q: float,
    eps: float,
    reduction: LossReduction = "mean",
) -> torch.Tensor:
    """The seesaw cross entropy loss
    Args:
        inputs: input tensor
        targets: target tensor
        weight: sample-wise loss weight
        cum_samples: cumulative samples for each category
        num_classes: number of classes
        p: mitigation factor
        q: compensation factor
        eps: avoid zero division
        reduction: reduction mode
    Returns:
        torch.Tensor
    """
    assert inputs.size(-1) == num_classes
    assert len(cum_samples) == num_classes

    onehot_labels = F.one_hot(targets, num_classes)
    seesaw_weights = inputs.new_ones(onehot_labels.size())

    # mitigation factor
    if p > 0:
        sample_ratio_matrix = cum_samples[None, :].clamp(min=1) / cum_samples[
            :, None
        ].clamp(min=1)
        index = (sample_ratio_matrix < 1.0).float()
        sample_weights = sample_ratio_matrix.pow(p) * index + (1 - index)
        mitigation_factor = sample_weights[targets.long(), :]
        seesaw_weights = seesaw_weights * mitigation_factor

    # compensation factor
    if q > 0:
        scores = F.softmax(inputs.detach(), dim=1)
        self_scores = scores[
            torch.arange(0, len(scores)).to(scores.device).long(),
            targets.long(),
        ]
        score_matrix = scores / self_scores[:, None].clamp(min=eps)
        index = (score_matrix > 1.0).float()
        compensation_factor = score_matrix.pow(q) * index + (1 - index)
        seesaw_weights = seesaw_weights * compensation_factor

    inputs = inputs + (seesaw_weights.log() * (1 - onehot_labels))

    loss = F.cross_entropy(inputs, targets, weight=None, reduction="none")

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction)
    return loss


class SeesawLoss(nn.Module):
    """Seesaw loss"""

    def __init__(
        self,
        p=0.8,
        q=2.0,
        num_classes=1000,
        eps=1e-2,
        reduction="mean",
        loss_weight=1.0,
    ) -> None:
        super().__init__()
        self.p = p
        self.q = q
        self.num_classes = num_classes
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

        # cumulative samples for each category
        self.register_buffer(
            "cum_samples", torch.zeros(self.num_classes, dtype=torch.float)
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        reduction: LossReduction = "none",
    ) -> torch.Tensor:
        """Forward method
        Args:
           inputs: input tensor
           targets: target tensor
           weight: sample-wise loss weight
           reduction: reduction mode
        Returns:
            torch.Tensor
        """

        assert inputs.size(0) == targets.view(-1).size(0), (
            f"Expected `labels` shape [{inputs.size(0)}], "
            f"but got {list(targets.size())}"
        )
        assert inputs.size(-1) == self.num_classes, (
            f"The channel number of output ({inputs.size(-1)}) does "
            f"not match the `num_classes` of seesaw loss ({self.num_classes})."
        )

        # accumulate the samples for each category
        unique_labels = targets.unique()
        for u_l in unique_labels:
            inds_ = targets == u_l.item()
            self.cum_samples[u_l] += inds_.sum()

        if weight is not None:
            weight = weight.float()
        else:
            weight = targets.new_ones(targets.size(), dtype=torch.float)

        # calculate loss_cls_classes
        loss_cls = self.loss_weight * seesaw_ce_loss(
            inputs,
            targets,
            weight,
            self.cum_samples,
            self.num_classes,
            self.p,
            self.q,
            self.eps,
            reduction,
        )

        return loss_cls
