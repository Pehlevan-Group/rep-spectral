"""
collection of regularizations on transfer learning tasks
"""

# load packages
import torch
import torch.nn as nn
import torch.optim as optim

from .conv import top_eig_ub_regularizer_conv
from .utils import batch_iterative_top_right_singular_vector, get_conv_fft2_blocks


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
    eigvals, _ = torch.topk(
        torch.linalg.eigvalsh(cov),
        k,
        largest=False,  # take bottom
        sorted=False,  # save time
    )
    penalty = sum(eigvals)
    return penalty


# ----- collecting and backprop is too memory expansive -------
# def top_eig_ub_transfer(model: nn.Module, max_layer: int = 4) -> torch.Tensor:
#     """ours"""
#     return top_eig_ub_regularizer_conv(model, max_layer)


# def spectral_ub_transfer(model: nn.Module) -> torch.Tensor:
#     """ours + last  layer spectral regularization"""
#     reg_term = top_eig_ub_transfer(model)

#     # last layer head
#     W = model.fc.weight
#     eig = torch.linalg.eigvalsh(W.T @ W).max()
#     reg_term += eig
#     return reg_term


# def top_eig_ub_transfer_update(
#     model: nn.Module, opt: optim, max_layer: int = 4, lam: float = 0.01
# ):
#     """compute eigenvalues and update for each convolution layers"""
#     funcs = model.get_conv_layer_eigvals_funcs()
#     for func in funcs:
#         opt.zero_grad()
#         eig_loss = func() * lam
#         eig_loss.backward()
#         opt.step()


# * testing alternative implementation, bypassing optim
def top_eig_ub_transfer_update(
    model: nn.Module,
    max_layer: int = 4,
    lam: float = 0.01,
    iterative=True,
    v_init=None,
    max_update=5,
    tol=1e-4,
):
    """
    eigenvalue upper bound regularization for convolution layer
    change parameter inside this functions

    :param model: ResNet50Pretrained
    :param max_layer: the maximum num blocks to regularize
    :param lam: the regularization strength
    :param iterative: True to use power iteration, False exact
    :param v_init: a Dict storing initial guesses right singular vectors in CPU
    :param max_update: the maximum number power iterations
    :param tol: stopping criterion for power iterations
    """
    conv_layers = model.get_conv_layers(max_layer=max_layer)
    for conv_layer in conv_layers:
        conv_layer.zero_grad()

        if iterative:
            # compute fft2 blocks
            h, w = conv_layer._input_shapes
            stride = conv_layer.wrap.stride[0]
            P = get_conv_fft2_blocks(conv_layer.wrap.weight, h, w, stride)

            # update right singular vectors
            v_new = batch_iterative_top_right_singular_vector(
                P, v_init[conv_layer].to(P.device), tol=tol, max_update=max_update
            )

            # compute eigenvalue for backprop
            u_sigma = torch.einsum("...ij,...j", P, v_new)
            eig_loss = (
                (u_sigma.conj() * u_sigma).sum(dim=-1).real.max()
            )  # sigma^2=lambda
            eig_loss *= lam
            eig_loss.backward()

            # update singular values
            v_init[conv_layer] = v_new.detach().cpu()
        else:
            # * not recommended, GPU eigvalsh takes a very long time for these sizes
            eig_loss = conv_layer._get_conv_layer_eigvals() * lam
            eig_loss.backward()

        conv_layer_grad = conv_layer.wrap.weight.grad

        # individual update, bypassing optimizer
        with torch.no_grad():
            conv_layer.wrap.weight.copy_(conv_layer.wrap.weight - conv_layer_grad)


def spectral_ub_transfer_update(
    model: nn.Module,
    opt_fc: optim,
    max_layer: int = 4,
    lam: float = 0.01,
    iterative=True,
    v_init=None,
    max_update: int = 5,
    tol: float = 1e-4,
):
    """
    compute eigenvalues and update for each convolution layers, and last connection layer
    :param opt_fc: the optimizer for the last layer
    """
    # eigenvalues
    top_eig_ub_transfer_update(
        model,
        max_layer=max_layer,
        lam=lam,
        iterative=iterative,
        v_init=v_init,
        max_update=max_update,
        tol=tol,
    )

    # last layer norm
    W = model.fc.weight
    eig = torch.linalg.eigvalsh(W.T @ W).max()
    eig_loss = eig * lam
    opt_fc.zero_grad()
    eig_loss.backward()
    opt_fc.step()
