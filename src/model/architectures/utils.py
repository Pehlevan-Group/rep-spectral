"""
architecture helper functions
"""

# load packages
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.optim.lr_scheduler import _LRScheduler

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


# ======================= conv wrapper =============================
class Conv2dWrap(nn.Module):
    """
    wrap default nn.Conv2d module to change to 'circular' mode and record input shapes
    """

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        conv.padding_mode = "circular"  # * change to circular
        self.wrap = conv
        self._input_shapes = None

    def forward(self, x: torch.Tensor):
        # register buffer
        if self._input_shapes is None:
            self._input_shapes = (x.shape[-2], x.shape[-1])  # (h, w)
        return self.wrap(x)

    def _get_conv_layer_eigvals(self) -> torch.Tensor:
        """compute eigvals"""
        kernel = self.wrap.weight
        stride = self.wrap.stride[
            0
        ]  # * assume square stride, which is True for pretrained resnet 50
        h, w = self._input_shapes
        eigval = get_multi_channel_top_eigval_with_stride(kernel, h, w, stride)
        return eigval


class WarmUpLR(_LRScheduler):
    """
    from https://github.com/weiaicunzai/pytorch-cifar100/blob/master/utils.py
    warmup_training learning rate scheduler
    :param optimizer: optimzier(e.g. SGD)
    :param total_iters: total_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]
