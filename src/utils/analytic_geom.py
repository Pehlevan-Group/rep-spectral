"""
geometric elements (analytic solution)
"""

# load packages
from typing import Callable

import torch
import torch.nn as nn


# utility functions
def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """convert erf to cdf of a standard normal"""
    x_erf = torch.erf(x / 2 ** (1 / 2))
    x_cdf = x_erf + (1 - x_erf) / 2
    return x_cdf


def _gaussian_pdf(x: torch.Tensor) -> torch.Tensor:
    """compute standard normal pdf"""
    x_pdf = torch.exp(-(x**2) / 2) / (2 * torch.pi) ** (1 / 2)
    return x_pdf


# ============= collection of closed form derivatives ============
def derivative_selector(nl: str) -> Callable:
    """select the derivative function by name of non-linearity (avoid eval)"""
    if nl == "Sigmoid":
        return Sigmoid_analytic_derivative
    elif nl == "Erf":
        return Erf_analytic_derivative
    elif nl == "ReLU":
        return lambda x: (x > 0).float()
    elif nl == "GELU":
        return GELU_analytic_derivative
    else:
        raise NotImplementedError(
            f"derivative of nl {nl} not implemented in closed-form"
        )


def hessian_selector(nl: str) -> Callable:
    """select the hessian function by name of the non-linearity"""
    if nl == "Sigmoid":
        return Sigmoid_analytic_hessian
    elif nl == "Erf":
        return Erf_analytic_hessian
    else:
        raise NotImplementedError(
            f"derivative of nl {nl} not implemented in closed-form"
        )


def GELU_analytic_derivative(x: torch.Tensor) -> torch.Tensor:
    """the analytic derivative of GELU function"""
    der = x * _gaussian_pdf(x) + _gaussian_cdf(x)
    return der


def Sigmoid_analytic_derivative(x: torch.Tensor) -> torch.Tensor:
    """the analytic derivative of the sigmoid function"""
    nl = nn.Sigmoid()
    nl_result = nl(x)
    der = nl_result * (1 - nl_result)
    return der


def Sigmoid_analytic_hessian(x: torch.Tensor) -> torch.Tensor:
    """the analytic hessian of the sigmoid function"""
    nl = nn.Sigmoid()
    nl_result = nl(x)
    hes = nl_result * (1 - nl_result) * (1 - 2 * nl_result)
    return hes


def Erf_analytic_derivative(x: torch.Tensor) -> torch.Tensor:
    """the analytic derivative of the error function"""
    der = (2 / torch.pi) ** (1 / 2) * torch.exp(-x.square() / 2)
    return der


def Erf_analytic_hessian(x: torch.Tensor) -> torch.Tensor:
    """the analytic hessian of the error function"""
    hes = (2 / torch.pi) ** (1 / 2) * (-x) * torch.exp(-x.square() / 2)
    return hes


# ============== analytic metrics =================
@torch.no_grad()
def determinant_analytic(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, nl: str
) -> torch.Tensor:
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
    der_func = derivative_selector(nl)
    activated_z = der_func(z)
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


@torch.no_grad()
def top_eig_analytic(
    x: torch.Tensor, W: torch.Tensor, b: torch.Tensor, nl: str
) -> torch.Tensor:
    """
    compute the top eigenvalue exactly from the metric tensor

    :param x: input tensor (d = 2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    :param nl: the nonlinearity, specified by a string
    """
    # prepare
    n = W.shape[0]
    z = x @ W.T + b  # number of scans by n
    der_func = derivative_selector(nl)
    activated_z = der_func(z)
    activated_square = activated_z.square()

    # compute gradient
    input_grad = torch.einsum("ni,...n,nj->...ij", W, activated_square, W)

    # get largest eigenvalue
    eigvals = torch.linalg.eigvalsh(input_grad)[:, -1]
    return eigvals
