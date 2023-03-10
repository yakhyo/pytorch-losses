import torch


class LossReduction:
    """Alias for loss reduction"""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
    Returns:
        Tensor: Processed loss values.
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
