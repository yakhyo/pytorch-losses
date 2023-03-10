from typing import Optional

import torch
from torch import nn

from losses.functional import cross_entropy


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss"""

    def __init__(
            self,
            class_weights: Optional[torch.Tensor] = None,
            reduction: str = "mean",
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
        loss = self.loss_weight * cross_entropy(
            inputs,
            targets,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            ignore_index=ignore_index
        )

        return loss
