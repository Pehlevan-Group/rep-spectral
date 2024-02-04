"""
collection of regularizations on transfer learning tasks
"""

# load packages
import torch
import torch.nn as nn
from .conv import top_eig_ub_regularizer_conv


def l2sp_transfer(
    model: nn.Module, alpha: float = 0.01, beta: float = 0.01
) -> torch.Tensor:
    """
    L2-SP: https://arxiv.org/abs/1802.01483
    regularize based on l2 deviation with the starting point parameter (pretrained weights)

    :param alpha: strength in deviation from the starting point
    :param beta: strength in the last layer linear head
    """
    deviation_loss = model.get_param_l2sp()
    last_layer_norm = torch.sum(model.fc.weight**2) + torch.sum(model.fc.bias**2)
    penalty = alpha * deviation_loss + beta * last_layer_norm
    return penalty


def bss_transfer(features: torch.Tensor, k: int = 1) -> torch.Tensor:
    """
    batch spectral shrinkage: https://proceedings.neurips.cc/paper/2019/hash/c6bff625bdb0393992c9d4db0c6bbe45-Abstract.html
    penalize bottom singular values of the feature representations

    :param features: the feature representations
    :param k: the number of bottom eigenvalues to penalize
    """
    # svd squared implemented through eigenvalues
    cov = features @ features.T
    eigvals = torch.lobpcg(cov, k, largest=False)
    penalty = sum(eigvals)
    return penalty


def top_eig_ub_transfer(model: nn.Module, max_layer: int = 4) -> torch.Tensor:
    """ours"""
    return top_eig_ub_regularizer_conv(model, max_layer)


def spectral_ub_transfer(model: nn.Module) -> torch.Tensor:
    """ours + last  layer spectral regularization"""
    reg_term = top_eig_ub_transfer(model)

    # last layer head
    W = model.fc.weight
    eig = torch.linalg.eigvals(W.T @ W).max()
    reg_term += eig
    return reg_term
