"""
Cross-Lipschitz Regularizer
source: https://arxiv.org/abs/1705.08475
"""

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def cross_lipschitz_regulerizer(
    model: nn.Module, x: torch.Tensor, is_binary: bool = True, sample_size: float = None
) -> float:
    """
    Cross-Lipschitz Regulerization: controlling the magnitude of gradient wrt inputs

    Here we use 2-norm squarred
    :param model: pytorch Module
    :param X: inputs
    :param is_binary: indicator for binary classification (output dim = 1) or multiclass classification
    :param sample_size: None to keep full samples, otherwise downsampled
    """
    if sample_size is not None:
        # downsample
        x = x[torch.randperm(len(x))[: int(sample_size * len(x))]]

    to_logits = nn.Sigmoid() if is_binary else nn.Softmax(dim=-1)
    grad = jacobian(
        lambda x: to_logits(model(x)).sum(axis=0), x, create_graph=True
    ).flatten(start_dim=2)

    if is_binary:
        # already a difference between two logits -> return gradient norm
        reg_term = grad.square().sum()
    else:
        # K: number of output
        # n: number of inputs
        # d: input dimension (flattend)
        K, n, d = grad.shape

        # expand the squared two norm, we can get the following implementation
        reg_term = (
            2
            / (n * K**2)
            * (grad.square().sum() - torch.einsum("lij,mij->...", grad, grad))
        )
    return reg_term
