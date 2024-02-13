"""
geometric elements (analytic solution)
"""

# load packages
from typing import Callable
import numpy as np

# import torch
# import torch.nn as nn


def _derivative_selector(nl: str) -> Callable:
    """select the derivative function by name of non-linearity (avoid eval)"""
    if nl == "Sigmoid":
        return _Sigmoid_analytic_derivative
    else:
        raise NotImplementedError(
            f"derivative of nl {nl} not implemented in closed-form"
        )


def _Sigmoid_analytic_derivative(x: np.ndarray) -> np.ndarray:
    """the analytic derivative of the sigmoid function"""
    nl = lambda x: 1 / (1 + np.exp(-x))
    nl_result = nl(x)
    der = nl_result * (1 - nl_result)
    return der


# ============== analytic metrics =================
def determinant_analytic(x: np.ndarray, W: np.ndarray, b: np.ndarray, nl: str):
    """
    the analytic determinant

    :param x: input tensor (d = 2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    :param nl: the nonlinearity, specified by a string
    """
    # prepare
    n = W.shape[0]
    # preactivation
    z = x @ W.T + b  # number of scans by n
    der_func = _derivative_selector(nl)
    activated_z = der_func(z)
    activated_square = activated_z**2

    # precompute m
    m = W[:, [0]] @ W[:, [1]].T - W[:, [1]] @ W[:, [0]].T
    m_squared = m**2

    # O(n^2) einsum enhanced (divided by two since each added twice and diagonal are zeros)
    results = (
        np.einsum("jk,nj,nk->n", m_squared, activated_square, activated_square) / 2
    )
    results = results / n**2
    results = np.sqrt(results)
    return results
