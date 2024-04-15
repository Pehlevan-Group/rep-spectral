"""
autograd version of geometric quantities computations
"""

# load packages
from typing import Tuple
import torch
import torch.nn as nn
import torch.autograd.functional as fnc


def batch_jacobian(f, x):
    """
    efficient jacobian computation of feature map f with respect to input x

    the output is of shape (feature_dim, batch_size, *input_dim)
    For example, if input x is (2, 10), then output is (feature_dim, 2, 10)
                 if input x is (2, 3, 32, 32), then the output is (feature_dim, 2, 3, 32, 32)
    """

    def f_sum(x):
        return torch.sum(f(x), axis=0)

    return fnc.jacobian(f_sum, x, create_graph=True)


def metric(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """
    compute metric tensor

    :param x: the query point
    :param feature_map: the feature map defined by the neural network
    """
    # computed directly from feature map
    J = batch_jacobian(feature_map, x).flatten(
        start_dim=2
    )  # starting from input dimensions
    met = J.permute(1, 2, 0) @ J.permute(1, 0, 2)
    return met


def determinant_autograd(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """(log) local expansion"""
    cur_metric = metric(x, feature_map)
    result = torch.log(torch.linalg.eigvalsh(cur_metric)).sum(dim=-1) / 2
    return result


def top_eig_autograd(x: torch.Tensor, feature_map: nn.Module) -> torch.Tensor:
    """the top eigenvalue of each metric tensor"""
    cur_metric = metric(x, feature_map)
    top_eigs, _ = torch.linalg.eigvalsh(cur_metric).max(dim=-1)
    return top_eigs


def determinant_and_eig_autograd(
    x: torch.Tensor, feature_map: nn.Module
) -> Tuple[torch.Tensor]:
    """jointly get determinant and top eigenvalue"""
    cur_metric = metric(x, feature_map)
    top_eigs, _ = torch.linalg.eigvalsh(cur_metric).max(dim=-1)
    log_det = torch.log(torch.linalg.eigvalsh(cur_metric)).sum(dim=-1) / 2
    return top_eigs, log_det
