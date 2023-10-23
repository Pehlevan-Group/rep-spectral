"""
some regularizers
"""

# load packages
import numpy as np
import torch
import torch.nn as nn
import torch.autograd.functional as fnc
from torch.autograd.functional import jacobian

# ===============================================
# ================== analytic ===================
# ===============================================

# ========== reg 1: cross lipschitz =============
def cross_lipschitz_regulerizer(model: nn.Module, x: torch.Tensor, is_binary: bool=True, sample_size: float=None) -> float:
    """
    Cross-Lipschitz Regulerization: controlling the magnitude of gradient wrt inputs
    source: https://arxiv.org/abs/1705.08475

    Here we use 2-norm squarred
    :param model: pytorch Module
    :param X: inputs
    :param is_binary: indicator for binary classification (output dim = 1) or multiclass classification 
    :param sample_size: None to keep full samples, otherwise downsampled
    """
    if sample_size is not None:
        # downsample 
        x = x[torch.randperm(len(x))[:int(sample_size * len(x))]]
    
    to_logits = nn.Sigmoid() if is_binary else nn.Softmax(dim=-1)
    grad = jacobian(lambda x: to_logits(model(x)).sum(axis=0), x, create_graph=True).flatten(start_dim=2)

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
    # results = torch.sqrt(results) # * for dimensional analysis, keep squared
    return results

def volume_element_regularizer(model: nn.Module, X: torch.Tensor, sample_size: float=None) -> float:
    """
    sum of geometric quantities (with top ones selected)
    """
    W, b = model.lin1.parameters()
    reg_terms = determinant_analytic(X, W, b)
    if sample_size is None:
        reg_term = reg_terms.sum()
    else:
        reg_terms, _ = torch.topk(reg_terms, int(sample_size * len(reg_terms)), dim=0, sorted=False)
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
    nl = nn.Sigmoid() # * fix sigmoid for now
    activated_z = nl(z) * (1 - nl(z))
    
    g11 = activated_z @ W[:, [0]].square() / n
    g22 = activated_z @ W[:, [1]].square() / n
    g12 = activated_z @ W.prod(dim=1, keepdim=True) / n

    # get top eigvalue
    lambda_max = (g11 + g22 + torch.sqrt((g11 - g22).square() + 4 * g12.square())) / 2
    return lambda_max

def top_eig_regularizer(model: nn.Module, X: torch.Tensor, sample_size: float=None) -> float:
    """
    sum of top eigvalues

    :param sample_size: select top sample_size of them to sum
    """
    W, b = model.lin1.parameters()
    reg_terms = top_eig_analytic(X, W, b)
    if sample_size is None:
        reg_term = reg_terms.sum()
    else:
        reg_terms, _ = torch.topk(reg_terms, int(sample_size * len(reg_terms)), dim=0, sorted=False)
        reg_term = reg_terms.sum()

    return reg_term


# ========================================
# =========== autograd ===================
# ========================================

def batch_jacobian(f, x):
    """
    efficient jacobian computation of feature map f with respect to input x

    the output is of shape (feature_dim, batch_size, *input_dim)
    For example, if input x is (2, 10), then output is (feature_dim, 2, 10)
                 if input x is (2, 3, 32, 32), then the output is (feature_dim, 2, 3, 32, 32)
    """
    f_sum = lambda x: torch.sum(f(x), axis=0)
    return fnc.jacobian(f_sum, x, create_graph=True)

def volume_element_regularizer_autograd(x: torch.Tensor, feature_map: nn.Module, sample_size: float=None, m: int=None, scanbatchsize: int=20):
    """
    autograd computation of volume element (using SVD)
    for memory reasons, we loop through samller batch size for jacobian computations

    :param m: the number of singular values to keep at each sample
    :param sample_size: the proportion of samples to take volume element evaluations at
    :param scanbatchsize: the scan size for each batch jacobian computation
    """
    # downsample 
    x = x[torch.randperm(len(x))[:int(sample_size * len(x))]]

    # directly from jacobian (time efficient)
    num_loops = int(np.ceil(x.shape[0] / scanbatchsize))
    log_svdvals_list = []
    for l in range(num_loops):
        cur_batch = x[l * scanbatchsize: (l + 1) * scanbatchsize]
        # get batch jacobian
        J = batch_jacobian(feature_map, cur_batch).flatten(start_dim=2)  # flatten starting from the input dim
        width = J.shape[0]
        J = J.permute(1, 2, 0) / width ** (1 / 2)  # manual normalization
        
        # take log for numerical stability
        log_svdvals = torch.linalg.svdvals(J).log()
        
        # keep only the top m eigenvalues
        if m is not None:
            log_svdvals = log_svdvals[:, :m]

        log_svdvals_list.append(log_svdvals) 
    # concat
    log_svdvals = torch.concat(log_svdvals_list)

    # aggregate 
    # TODO: check numerical stability
    reg_terms = torch.exp(log_svdvals.sum(dim=-1) * 2)
    reg_term = reg_terms.sum()

    return reg_term

def top_eig_regularizer_autograd(x: torch.Tensor, feature_map: nn.Module, sample_size: float=None, scanbatchsize: int=20):
    """
    autograd computation of top eigenvalue (using lobpcg)
    for memory reasons, we split into smaller batch size for jacobian computationss
    
    :param sample_size: the proportion of samples to take eigenvalues at 
    :param scanbatchsize: the batch size for batched jacobian  
    """
    # downsample 
    x = x[torch.randperm(len(x))[:int(sample_size * len(x))]]

    num_loops = int(np.ceil(x.shape[0] / scanbatchsize))
    eig_list = []
    for l in range(num_loops): 
        # compute metric
        cur_scan = x[l*scanbatchsize : (l+1) * scanbatchsize]
        J = batch_jacobian(feature_map, cur_scan).flatten(start_dim=2)  # flatten starting from the input dim
        width = J.shape[0]
        met = J.permute(1, 2, 0) @ J.permute(1, 0, 2) / width # manual normalization
        # eigs, _ = torch.lobpcg(met, k=1, largest=True)
        eigs = torch.linalg.eigvalsh(met)[:, -1:] # * more numerically stable than lobpcg
        eig_list.append(eigs)

    reg_terms = torch.concat(eig_list)
    reg_term = reg_terms.sum()
    
    return reg_term
