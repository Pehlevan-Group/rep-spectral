"""
architecture helper functions
"""

# load packages
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import pad


def get_multi_channel_top_eigval_with_stride(
    kernel: torch.Tensor, h: int, w: int, stride: int
) -> torch.Tensor:
    """
    compute top eigen value of a convolution layer
    * code tested only for even n and stride = 1 or 2.

    code adapted from theorem 2 of https://openreview.net/forum?id=T5TtjbhlAZH

    :param kernel: the conv2d kernel, with shape (c_out, c_in, k, k)
    :param h: the image height
    :param w: the image width
    :param stride: the stride of convolution
    :return the top singular value for conv layer
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

    # compute singular value squares (faster than svd most of the time)
    eigvals = torch.linalg.eigvalsh(
        torch.einsum("...ij,...kj->...ik", torch.conj(P), P)
        if P.shape[3] > P.shape[2]
        else torch.einsum("...ji,...jk->...ik", torch.conj(P), P)
    )
    top_eig = eigvals.max()
    return top_eig


def get_output_size(input_h: int, input_w: int, conv: nn.Conv2d) -> Tuple[int]:
    """
    compute output size from input height and width

    follow exactly the way in documentation:
    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    pad_h, pad_w = conv.padding
    k_h, k_w = conv.kernel_size
    dilation_h, dilation_w = conv.dilation
    stride_h, stride_w = conv.stride

    output_h = int((input_h + 2 * pad_h - dilation_h * (k_h - 1) + 1) / stride_h + 1)
    output_w = int((input_w + 2 * pad_w - dilation_w * (k_w - 1) + 1) / stride_w + 1)

    return output_h, output_w
