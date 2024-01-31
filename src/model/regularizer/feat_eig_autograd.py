"""
hosts a list of feature map regularization (autograd)
"""

# load packages
import numpy as np
import torch
import torch.nn as nn

# load files
from .utils import batch_jacobian, derivatives, iterative_top_right_singular_vector


def top_eig_regularizer_autograd(
    x: torch.Tensor,
    feature_map: nn.Module,
    sample_size: float = None,
    scanbatchsize: int = 20,
    max_layer: int = None,
):
    """
    autograd computation of top eigenvalue (using lobpcg)
    for memory reasons, we split into smaller batch size for jacobian computationss
    # * highly time-costly

    :param sample_size: the proportion of samples to take eigenvalues at
    :param scanbatchsize: the batch size for batched jacobian
    :param max_layer: the maximum number of layers to pull information from and regularize
        None means all feature maps
    """
    # downsample
    x = x[torch.randperm(len(x))[: int(sample_size * len(x))]]

    # find feature map
    counter = 0
    for i, layer in enumerate(feature_map):
        if isinstance(layer, nn.Linear):
            counter += 1
        if max_layer is not None and counter == max_layer:
            break
    feature_map = nn.Sequential(*feature_map[: (i + 1)])

    # scan
    num_loops = int(np.ceil(x.shape[0] / scanbatchsize))
    eig_list = []
    for l in range(num_loops):
        # compute metric
        cur_scan = x[l * scanbatchsize : (l + 1) * scanbatchsize]
        J = batch_jacobian(feature_map, cur_scan).flatten(
            start_dim=2
        )  # flatten starting from the input dim
        width = J.shape[0]
        met = J.permute(1, 2, 0) @ J.permute(1, 0, 2) / width  # manual normalization
        # eigs, _ = torch.lobpcg(met, k=1, largest=True)
        # * more numerically stable than lobpcg
        eigs = torch.linalg.eigvalsh(met)[:, -1:]
        eig_list.append(eigs)

    reg_terms = torch.concat(eig_list)
    reg_term = reg_terms.mean()  # * use average

    return reg_term


def top_eig_ub_regularizer_autograd(
    x: torch.Tensor,
    feature_map: nn.Module,
    max_layer: int = None,
    iterative=True,
    v_init=None,
    max_update: int = 2,
    tol=1e-6,
):
    """
    give upper bound of the previous quantity
    only for linear MLP

    :param max_layer: the max number of layers to pull information from and regularize
        None means pulling from all
    :param iterative: True to us exact eigen-decomposition, False to use power iteration
    :param v_init: the initialization of top right singular vector
    :param max_update: the maximum number of updates to find the new singular vectors
    :param tol: the tolerance for convergence stopping criterion

    :return the regularization term
    """
    # sequential passing in
    activations, eigs = [], []
    counter = 0  # count current linear layers encountered
    for layer in feature_map:
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
            counter += 1

        # anything else
        else:
            layer_name = layer.__class__.__name__
            # a valid nonlinearity
            if layer_name.lower() in ["sigmoid", "gelu", "relu", "elu"]:
                z_der = derivatives(x, layer_name)
                z_der_sq = z_der.square()
                activation = z_der_sq.max(dim=-1).values.mean()  # * use average
                activations.append(activation)

        if max_layer is not None and counter == max_layer:
            break

        # update to next layer
        x = layer(x)

    # compute regularization
    reg_term = sum(activations) + sum(eigs)

    return reg_term


def top_eig_ub_pure_regularizer_autograd(
    x: torch.Tensor,
    feature_map: nn.Module,
    max_layer: int = None,
    iterative=True,
    v_init=None,
    max_update: int = 2,
    tol=1e-6,
):
    """
    give upper bound of the previous quantity
    only for linear MLP (activation pattern removed)

    :param max_layer: the max number of layers to pull information from and regularize
        None means pulling from all
    :param iterative: True to us exact eigen-decomposition, False to use power iteration
    :param v_init: the initialization of top right singular vector
    :param max_update: the maximum number of updates to find the new singular vectors
    :param tol: the tolerance for convergence stopping criterion

    :return the regularization term
    """
    # sequential passing in
    eigs = []
    counter = 0  # count current linear layers encountered
    for layer in feature_map:
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
            counter += 1

        if max_layer is not None and counter == max_layer:
            break

        # update to next layer
        x = layer(x)

    # compute regularization
    reg_term = sum(eigs)

    return reg_term


# def volume_element_regularizer_autograd(x: torch.Tensor, feature_map: nn.Module, sample_size: float = None, m: int = None, scanbatchsize: int = 20):
#     """
#     autograd computation of volume element (using SVD)
#     for memory reasons, we loop through samller batch size for jacobian computations

#     :param m: the number of singular values to keep at each sample
#     :param sample_size: the proportion of samples to take volume element evaluations at
#     :param scanbatchsize: the scan size for each batch jacobian computation
#     """
#     # downsample
#     x = x[torch.randperm(len(x))[:int(sample_size * len(x))]]

#     # directly from jacobian (time efficient)
#     num_loops = int(np.ceil(x.shape[0] / scanbatchsize))
#     log_svdvals_list = []
#     for l in range(num_loops):
#         cur_batch = x[l * scanbatchsize: (l + 1) * scanbatchsize]
#         # get batch jacobian
#         J = batch_jacobian(feature_map, cur_batch).flatten(
#             start_dim=2)  # flatten starting from the input dim
#         width = J.shape[0]
#         J = J.permute(1, 2, 0) / width ** (1 / 2)  # manual normalization

#         # take log for numerical stability
#         log_svdvals = torch.linalg.svdvals(J).log()

#         # keep only the top m eigenvalues
#         if m is not None:
#             log_svdvals = log_svdvals[:, :m]

#         log_svdvals_list.append(log_svdvals)
#     # concat
#     log_svdvals = torch.concat(log_svdvals_list)

#     # aggregate
#     # TODO: check numerical stability
#     reg_terms = torch.exp(log_svdvals.sum(dim=-1) * 2)
#     reg_term = reg_terms.sum()

#     return reg_term
