from typing import Union

import torch

from _utils import _loss_inter_union, LossReduction
from torch import nn


class GIOULoss(nn.Module):
    """Generalized box IOU Loss"""

    def __init__(
        self,
        reduction: Union[LossReduction, str] = LossReduction.NONE,
        epsilon: float = 1e-7,
    ):
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon

    def forward(self, boxes1: torch.Tensor, boxes2: torch.Tensor):
        boxes1 = boxes1.float()
        boxes2 = boxes2.float()

        intsctk, unionk = _loss_inter_union(boxes1, boxes2)
        iouk = intsctk / (unionk + self.epsilon)

        x1, y1, x2, y2 = boxes1.unbind(dim=-1)
        x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

        # smallest enclosing box
        xc1 = torch.min(x1, x1g)
        yc1 = torch.min(y1, y1g)
        xc2 = torch.max(x2, x2g)
        yc2 = torch.max(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1)
        miouk = iouk - ((area_c - unionk) / (area_c + self.epsilon))

        loss = 1 - miouk

        # Check reduction option and return loss accordingly
        if self.reduction == LossReduction.NONE:
            pass
        elif self.reduction == LossReduction.MEAN:
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif self.reduction == LossReduction.SUM:
            loss = loss.sum()
        else:
            raise ValueError(
                f"Unsupported reduction: {self.reduction}, Supported options are: 'mean', 'sum', 'none'"
            )
        return loss
