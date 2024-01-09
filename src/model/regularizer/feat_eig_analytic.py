"""
hosts a list of feature map regularizers (analytic)
"""

# load packages
import torch
import torch.nn as nn

# TODO: change nonlinearity dependence


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
    nl = nn.Sigmoid()  # * fix sigmoid for now
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
    # results = torch.sqrt(results) # * for dimensional analysis, keep squared
    return results


def volume_element_regularizer(
    model: nn.Module, X: torch.Tensor, sample_size: float = None
) -> float:
    """
    sum of geometric quantities (with top ones selected)
    """
    W, b = model.lin1.parameters()
    reg_terms = determinant_analytic(X, W, b)
    if sample_size is None:
        reg_term = reg_terms.sum()
    else:
        reg_terms, _ = torch.topk(
            reg_terms, int(sample_size * len(reg_terms)), dim=0, sorted=False
        )
        reg_term = reg_terms.sum()

    return reg_term


# ========= reg 3: top eig =================
def top_eig_analytic(x: torch.Tensor, W: torch.Tensor, b: torch.Tensor):
    """
    it can be shown that a small top eigenvalue for a correct sample
    is a necessary condition for good input space perturbation

    :param x: input tensor (d=2)
    :param W: weight matrix n by d
    :param b: bias vector n by 1
    """
    # prepare
    n = W.shape[0]

    # compute three components
    z = x @ W.T + b  # number of scans by n
    nl = nn.Sigmoid()  # * fix sigmoid for now
    activated_z = nl(z) * (1 - nl(z))

    g11 = activated_z @ W[:, [0]].square() / n
    g22 = activated_z @ W[:, [1]].square() / n
    g12 = activated_z @ W.prod(dim=1, keepdim=True) / n

    # get top eigvalue
    lambda_max = (g11 + g22 + torch.sqrt((g11 - g22).square() + 4 * g12.square())) / 2
    return lambda_max


def top_eig_regularizer(
    model: nn.Module, X: torch.Tensor, sample_size: float = None
) -> float:
    """
    sum of top eigvalues

    :param sample_size: select top sample_size of them to sum
    """
    W, b = model.lin1.parameters()
    reg_terms = top_eig_analytic(X, W, b)
    if sample_size is None:
        reg_term = reg_terms.sum()
    else:
        reg_terms, _ = torch.topk(
            reg_terms, int(sample_size * len(reg_terms)), dim=0, sorted=False
        )
        reg_term = reg_terms.sum()

    return reg_term
