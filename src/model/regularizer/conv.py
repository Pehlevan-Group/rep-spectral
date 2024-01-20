"""
regularization for convolution layers, assume all strides are square
"""

# load packages
import torch
import torch.nn as nn


def top_eig_ub_regularizer_conv(model: nn.Module, max_layer: int=4) -> torch.Tensor:
    """
    regularize up to feature map
    :param max_layer: the maximum layers in ResNet to regularize
    """
    eigvals = model.get_conv_layer_eigvals(max_layer=max_layer)
    reg_term = sum(eigvals)
    return reg_term


def spectral_ub_regularizer_conv(model: nn.Module) -> torch.Tensor:
    """add final linear head regularization"""
    reg_term = top_eig_ub_regularizer_conv(model)

    # last linear head
    W = model.linear.weight
    eig = torch.linalg.eigvalsh(W.T @ W).max()
    reg_term += eig
    return reg_term
