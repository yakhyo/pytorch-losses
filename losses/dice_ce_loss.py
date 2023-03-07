from typing import Any, Dict, Tuple, Union

import torch
from _utils import ActivationFunction, LossReduction
from torch import nn

from torch.nn import functional as F

from losses.dice_loss import DiceLoss

__all__ = ["DiceCELoss"]


class DiceCELoss(nn.Module):
    """Cross Entropy Dice Loss"""

    def __init__(
            self,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            epsilon: float = 1e-5,
            activation: Union[ActivationFunction, str] = ActivationFunction.SOFTMAX,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss(reduction, epsilon, activation)

    def __call__(
            self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[Any, Dict[str, Any]]:
        ce_loss = F.cross_entropy(inputs, targets)
        dice_loss = self.dice(inputs, targets)

        return ce_loss + dice_loss
