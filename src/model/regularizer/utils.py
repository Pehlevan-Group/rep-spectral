"""
some utility functions for regularizer computations
"""

# load packages
import warnings
from typing import Dict
import torch
import torch.nn as nn
import torch.autograd.functional as fnc


# ========================
# torch functional wrapper
# ========================
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


# ========================
# closed form derivatives
# ========================
def derivatives(x: torch.Tensor, nl_type: str) -> torch.Tensor:
    """
    closed-form computation of derivatives of common nonlinearity
    """
    if nl_type.lower() == "sigmoid":
        return _derivative_sigmoid(x)
    elif nl_type.lower() == "gelu":
        return _derivative_gelu(x)
    elif nl_type.lower() == "relu":
        return _derivative_relu(x)
    elif nl_type.lower() == 'elu':
        return _derivative_elu(x)
    else:
        raise NotImplementedError(f"nl type {nl_type} not available")


def _derivative_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """derivative of sigmoid"""
    nl = nn.Sigmoid()
    der = nl(x) * (1 - nl(x))
    return der


def _derivative_gelu(x: torch.Tensor) -> torch.Tensor:
    """derivative of gelu"""
    der = (1 + torch.erf(x / 2 ** (1 / 2))) / 2 + x * 1 / (2 * torch.pi) ** (
        1 / 2
    ) * torch.exp(-(x**2) / 2)
    return der

def _derivative_relu(x: torch.Tensor) -> torch.Tensor:
    """derivative of ReLU"""
    der = torch.maximum(x, torch.tensor(0, device=x.device)) / x.clamp(min=1e-16)
    return der

def _derivative_elu(x: torch.Tensor) -> torch.Tensor:
    """derivative of ELU"""
    ind = (x > 0).to(torch.int64)
    der = x / x * ind + torch.exp(x) * (1 - ind)
    return der


# ============================================
# Iterative update for top singular direction
# ============================================
@torch.no_grad()
def iterative_top_singular_pair(
    W: torch.Tensor, v: torch.Tensor = None, tol: float = 1e-6, max_update: int = None
):
    """
    power iteration applied to find the top singular pairs

    :param W: the parameter matrix
    :param v: the initial top right singular value guess
    :param tol: the convergence stopping criterion
    :param max_update: the max number of iterations

    :return the singular value bundle (sigma, u, v)
    """

    n, p = W.shape
    # random init
    if v is None:
        v = torch.normal(0, 1, (p, 1)).to(W.device)
        v /= v.norm()

    u_prev, v_prev = 0, v
    if max_update is None:
        max_update = max(n, p) * 2

    # power iteration
    for i in range(max_update):
        u = W @ v_prev
        u /= u.norm()
        v = W.T @ u
        v /= v.norm()

        diff = max(torch.norm(u - u_prev), torch.norm(v - v_prev))
        if diff < tol:
            break
        elif diff >= tol and i == max_update - 1:
            warnings.warn(
                f"iterative method did not converge with max_update {max_update}, diff={diff}"
            )
        else:
            u_prev, v_prev = u, v

    sigma = u.T @ W @ v
    return sigma, u, v


@torch.no_grad()
def iterative_top_right_singular_vector(
    W: torch.Tensor, v: torch.Tensor = None, tol: float = 1e-6, max_update: int = None
):
    """power method on W^T W to get the top right singular vector"""
    A = W.T @ W
    p = A.shape[0]
    # random init
    if v is None:
        v = torch.normal(0, 1, (p, 1)).to(W.device)
        v /= v.norm()

    v_prev = v
    if max_update is None:
        max_update = p * 2
    # power iteration
    for i in range(max_update):
        v = A @ v_prev
        v /= v.norm()

        diff = (v - v_prev).norm()
        if diff < tol:
            break
        elif diff >= tol and i == max_update - 1:
            warnings.warn(
                f"iterative method did not converge with max_update {max_update}, diff={diff}"
            )
        else:
            v_prev = v

    return v

@torch.no_grad()
def init_model_right_singular(
    feature_map: nn.Module, tol: float = 1e-6
) -> Dict[str, torch.Tensor]:
    """
    find the first right singular value for every linear layer in the feature map
    """
    v_init_by_layer = {}
    for layer in feature_map:
        if isinstance(layer, nn.Linear):
            W = layer.weight
            v_init = iterative_top_right_singular_vector(W, v=None, tol=tol)
            v_init_by_layer[layer] = v_init

    return v_init_by_layer
