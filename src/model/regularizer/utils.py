"""
some utility functions for regularizer computations
"""

# load packages
import os
import warnings
from typing import Dict
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.autograd.functional as fnc
from torch.nn.functional import pad


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
    elif nl_type.lower() == "elu":
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
# @torch.no_grad()
# def iterative_top_singular_pair(
#     W: torch.Tensor, v: torch.Tensor = None, tol: float = 1e-6, max_update: int = None
# ):
#     """
#     power iteration applied to find the top singular pairs

#     :param W: the parameter matrix
#     :param v: the initial top right singular value guess
#     :param tol: the convergence stopping criterion
#     :param max_update: the max number of iterations

#     :return the singular value bundle (sigma, u, v)
#     """

#     n, p = W.shape
#     # random init
#     if v is None:
#         v = torch.normal(0, 1, (p, 1)).to(W.device)
#         v /= v.norm()

#     u_prev, v_prev = 0, v
#     if max_update is None:
#         max_update = max(n, p) * 2

#     # power iteration
#     for i in range(max_update):
#         u = W @ v_prev
#         u /= u.norm()
#         v = W.T @ u
#         v /= v.norm()

#         diff = max(torch.norm(u - u_prev), torch.norm(v - v_prev))
#         if diff < tol:
#             break
#         elif diff >= tol and i == max_update - 1:
#             warnings.warn(
#                 f"iterative method did not converge with max_update {max_update}, diff={diff}"
#             )
#         else:
#             u_prev, v_prev = u, v

#     sigma = u.T @ W @ v
#     return sigma, u, v


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
def get_batch_complex_norm(v: torch.Tensor, dim=-1):
    """get the norm of a complex vector"""
    norm = (v.conj() * v).sum(dim=dim, keepdim=True) ** 0.5
    # keep real only
    norm = norm.real
    return norm


@torch.no_grad()
def batch_iterative_top_right_singular_vector(
    kernel: torch.Tensor,
    v: torch.Tensor = None,
    tol: float = 1e-4,
    max_update: int = None,
) -> torch.Tensor:
    n, p = kernel.shape[-2], kernel.shape[-1]
    batch_dims = kernel.shape[:-2]

    # initialize complex vectors
    if v is None:
        v = torch.randn((*batch_dims, p), dtype=torch.cfloat, device=kernel.device)
        v /= get_batch_complex_norm(v)

    if max_update is None:
        max_update = p * 2
    u_prev, v_prev = 0, v
    kernel_conj_transpose = torch.conj(kernel).transpose(-1, -2)
    for i in range(max_update):
        u = torch.einsum("...ij,...j->...i", kernel, v)
        u /= get_batch_complex_norm(u)
        v = torch.einsum("...ij,...j->...i", kernel_conj_transpose, u)
        v /= get_batch_complex_norm(v)

        diff = max(
            get_batch_complex_norm(u - u_prev).max(),
            get_batch_complex_norm(v - v_prev).max(),
        )
        if diff < tol:
            break
        elif diff >= tol and i == max_update - 1:
            warnings.warn(
                f"iterative method did not converge with max_update {max_update}, diff={diff}"
            )
        else:
            u_prev, v_prev = u, v

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


@torch.no_grad()
def init_model_right_singular_conv(
    model: nn.Module,
    tol: float = 1e-4,
    h: int = 224,
    w: int = 224,
    max_layer=4,
    dump_path: str = None,
) -> Dict[nn.Module, torch.Tensor]:
    """
    find the top right singular value for convolution layer in ResNet models
    * currently only supporting the pretrained module (`ResNet50Pretrained` class)

    we load from dump_path if all initial guesses are available
    otherwise initialize

    :param model: ResNet50Pretrained
    :parma tol: the tolerance in consecutive singular direction deviation
    :param h, w: the height and width of input image
    :param max_layer: maximum number of blocks to regularize
    :param dump_path: the path to write/load precomputed right singular vectors
    """
    # register input shapes to each layer
    conv_layers = model.get_conv_layers(max_layer=max_layer)
    conv_layers_names = model.get_conv_layers_names(max_layer=max_layer)

    if conv_layers[0]._input_shapes is None:
        fake_input = torch.randn((1, 3, h, w))
        model(fake_input)  # register shapes

    v_init_by_conv = {}
    # if initial guesses already available, read
    if dump_path is not None and len(os.listdir(dump_path)) == 53:
        for name in conv_layers_names:
            v_load_path = os.path.join(dump_path, name + ".pt")
            v_init_by_conv[name] = torch.load(v_load_path, map_location="cpu")

    # if not available, initialize
    else:
        for name, layer in tqdm(zip(conv_layers_names, conv_layers)):
            kernel = layer.wrap.weight
            stride = layer.wrap.stride[0]  # * assume symmetric
            input_h, input_w = layer._input_shapes
            P = get_conv_fft2_blocks(kernel, input_h, input_w, stride)

            # get sufficiently good starting point and move to cpu
            v_init = (
                batch_iterative_top_right_singular_vector(P, v=None, tol=tol)
                .detach()
                .cpu()
            )

            # save to cpu
            v_init_by_conv[name] = v_init
            torch.cuda.empty_cache()

            # dump to path
            if dump_path is not None:
                v_write_path = os.path.join(dump_path, name + ".pt")
                torch.save(v_init, v_write_path)

    return v_init_by_conv


# ============================
# ---- for convolution -------
# =============================
@torch.no_grad()
def get_conv_fft2_blocks(
    kernel: torch.Tensor, h: int, w: int, stride: int
) -> torch.Tensor:
    """
    pad filters and precompute fft2 for getting initial guesses of singular values
    * code tested only for even n and stride = 1 or 2.

    code adapted from theorem 2 of https://openreview.net/forum?id=T5TtjbhlAZH

    :param kernel: the conv2d kernel, with shape (c_out, c_in, k, k)
    :param h: the image height
    :param w: the image width
    :param stride: the stride of convolution
    :return a matrix of shape (h//stride, w//stride, c_out, c_in * stride ** 2)
    """
    assert (
        len(kernel.shape) == 4
    ), f"kernel of shape {kernel.shape} is not for 2D convolution"
    # pad zeros to the kernel to make the same shape as input
    c_out, c_in, k_h, k_w = kernel.shape
    pad_height = h - k_h
    pad_width = w - k_w
    kernel_pad = pad(kernel, (0, pad_height, 0, pad_width), mode="constant", value=0)
    str_shape_height, str_shape_width = h // stride, w // stride

    # downsample the kernel
    transforms = torch.zeros(
        (c_out, c_in, stride**2, str_shape_height, str_shape_width)
    ).to(kernel.device)
    for i in range(stride):
        for j in range(stride):
            transforms[:, :, i * stride + j, :, :] = kernel_pad[
                :, :, i::stride, j::stride
            ]

    # fft2
    transforms = torch.fft.fft2(transforms)
    transforms = transforms.reshape(c_out, -1, str_shape_height, str_shape_width)

    # reorg (h // stride, w // stride, c_out, c_in * stride^2)
    P = transforms.permute(2, 3, 0, 1)
    return P
