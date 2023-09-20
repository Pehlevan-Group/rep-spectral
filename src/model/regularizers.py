"""
some regularizers
"""

# load packages
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

# ========== reg 1: cross lipschitz =============
def cross_lipschitz_regulerizer(model: nn.Module, X: torch.Tensor, is_binary: bool=True) -> float:
    """
    Cross-Lipschitz Regulerization: controlling the magnitude of gradient wrt inputs
    source: https://arxiv.org/abs/1705.08475

    Here we use 2-norm squarred
    """
    sig = nn.Sigmoid()
    grad = jacobian(lambda x: sig(model(x)).sum(axis=0), X, create_graph=True).flatten(start_dim=2)

    if is_binary:
        # already a difference between two logits -> return gradient norm
        reg_term = grad.square().sum()
    else:
        # K: number of output
        # n: number of inputs
        # d: input dimension (flattend)
        K, n, d = grad.shape

        # expand the squared two norm, we can get the following implementation
        reg_term = 2 / (n * K ** 2) * (
            grad.square().sum()
            - torch.einsum('lij,mij->...', grad, grad)
        )
    return reg_term

# =========== reg 2: volume element =============
def determinant_analytic(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """
    the analytic determinant

    :param x: input tensor (d = 2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    """
    # prepare
    n = W.shape[0]
    # preactivation
    z = x @ W.T + b  # number of scans by n
    nl = nn.Sigmoid() # * fix sigmoid for now
    activated_z = nl(z) * (1 - nl(z))
    activated_square = activated_z.square()

    # precompute m
    m = W[:, [0]] @ W[:, [1]].T - W[:, [1]] @ W[:, [0]].T
    m_squared = m.square()

    # O(n^2) einsum enhanced (divided by two since each added twice and diagonal are zeros)
    results = (
        torch.einsum("jk,nj,nk->n", m_squared, activated_square, activated_square) / 2
    )
    results = results / n**2
    results = torch.sqrt(results)
    return results

def volume_element_regularizer(model: nn.Module, X: torch.Tensor) -> float:
    """
    sum of geometric quantities
    """
    W, b = model.lin1.parameters()
    reg_term = determinant_analytic(X, W, b).sum()
    return reg_term
