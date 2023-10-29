"""
some utility functions for regularizer computations
"""

# load packages
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
    def f_sum(x): return torch.sum(f(x), axis=0)
    return fnc.jacobian(f_sum, x, create_graph=True)

# ========================
# closed form derivatives
# ========================
def derivatives(x: torch.Tensor, nl_type: str) -> torch.Tensor:
    """
    closed-form computation of derivatives of common nonlinearity
    """
    if nl_type.lower() == 'sigmoid':
        return _derivative_sigmoid(x)
    elif nl_type.lower() == 'gelu':
        return _derivative_gelu(x)
    else:
        raise NotImplementedError(f"nl type {nl_type} not available")

def _derivative_sigmoid(x: torch.Tensor):
    """derivative of sigmoid"""
    nl = nn.Sigmoid()
    der = nl(x) * (1 - nl(x))
    return der

def _derivative_gelu(x: torch.Tensor): 
    """derivative of gelu"""
    der = (1 + torch.erf(x / 2 ** (1/2))) / 2 + x * 1 / (2 * torch.pi) ** (1/2) * torch.exp(-x**2/2)
    return der
