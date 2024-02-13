"""
regularizing the spectral norm
from https://arxiv.org/pdf/1705.10941.pdf
"""

# load packages
import torch
import torch.nn as nn

from .utils import iterative_top_right_singular_vector


def spectral_ub_regularizer_autograd(
    model: nn.Module, iterative=True, v_init=None, max_update: int = 2, tol=1e-6
):
    """
    give spectral bound on also the last layer
    (served as benchmark of our noval method)

    :param iterative: True to us exact eigen-decomposition, False to use power iteration
    :param v_init: the initialization of top right singular vector
    :param max_update: the maximum number of updates to find the new singular vectors
    :param tol: the tolerance for convergence stopping criterion
    """
    # sequential passing in
    eigs = []

    # multi-hidden-layer
    if hasattr(model, "model"):
        layers = model.model
    # single-hidden-layer
    else:
        layers = [model.lin1, model.lin2]

    # sequential passing
    for layer in layers:
        # a linear layer
        if isinstance(layer, nn.Linear):
            W = layer.weight
            if iterative:
                v_new = iterative_top_right_singular_vector(
                    W, v_init[layer], tol=tol, max_update=max_update
                )

                # compute eigenvalue for backprop
                temp = W @ v_new
                eig = temp.T @ temp
                eig = eig[0][0]

                # update singular values
                v_init[layer] = v_new
            else:
                eig = torch.linalg.eigvalsh(W.T @ W).max()
            eigs.append(eig)

    # compute regularization
    reg_term = sum(eigs)

    return reg_term
