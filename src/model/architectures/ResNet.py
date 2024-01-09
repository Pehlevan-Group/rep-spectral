"""
PyTorch implementation of ResNet, adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

the following modifications are made for the project:
* note that to match with theoretical predictions, all padding are circular instead of zero padding
* each model is equipped with methods for extracting convolutional layers' eigen(singular) values for 
  later use
* each model, after calling forward, memorize the image shape, since the singular value depends on 
  the shape of the input
"""

# load packages
from typing import List
import torch
import torch.nn as nn

# load file
from .utils import get_multi_channel_top_eigval_with_stride, get_output_size


class BasicBlock(nn.Module):
    _expansion = 1

    def __init__(self, in_planes, planes, stride=1, nl=nn.GELU()):
        super(BasicBlock, self).__init__()
        self.nl = nl
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self._expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self._expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm2d(self._expansion * planes),
            )

        # for forward to ducktype
        self.input_shapes = None

    def forward(self, x):
        if self.input_shapes is None:
            self._get_conv_input_shapes(x.shape[-2], x.shape[-1])

        out = self.nl(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.nl(out)
        return out

    # ----- for computing eigenvalues of convolution layers --------
    def _get_conv_input_shapes(self, h: int, w: int) -> None:
        """
        infer the shapes of the input to each convolution layer
        :param h, w: h, w are the input shape of the forward method (i.e. the input of the first conv)
        """
        result_list = [(h, w)]

        # after conv1
        h, w = get_output_size(h, w, self.conv1)
        result_list.append((h, w))

        # after conv2
        for layer in self.shortcut:
            if isinstance(layer, nn.Conv2d):
                result_list.append(result_list[0])  # replicate original input shapes

        # ducktype
        self.input_shapes = result_list

    def get_conv_layer_eigvals(self) -> List[torch.Tensor]:
        """return a list of eigenvalues of each convolution layer"""
        conv_layers = [self.conv1, self.conv2]
        # add shortcut layer
        for layer in self.shortcut:
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)

        # get eigenvalues
        eigs = []
        for (h, w), conv_layer in zip(self.input_shapes, conv_layers):
            kernel = conv_layer.weight
            stride = conv_layer.stride[0]  # * assume square stride
            eigval = get_multi_channel_top_eigval_with_stride(kernel, h, w, stride)
            eigs.append(eigval)

        return eigs


class Bottleneck(nn.Module):
    _expansion = 4

    def __init__(self, in_planes, planes, stride=1, nl=nn.GELU()):
        super(Bottleneck, self).__init__()
        self.nl = nl
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=1, bias=False, padding_mode="circular"
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes,
            self._expansion * planes,
            kernel_size=1,
            bias=False,
            padding_mode="circular",
        )
        self.bn3 = nn.BatchNorm2d(self._expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self._expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self._expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    padding_mode="circular",
                ),
                nn.BatchNorm2d(self._expansion * planes),
            )

        # to be ducktyped
        self.input_shapes = None

    def forward(self, x):
        if self.input_shapes is None:
            self._get_conv_input_shapes()

        out = self.nl(self.bn1(self.conv1(x)))
        out = self.nl(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.nl(out)
        return out

    # ----- for computing eigenvalues of convolution layers --------
    def _get_conv_input_shapes(self, h: int, w: int) -> None:
        """
        infer the shapes of the input to each convolution layer
        :param h, w: h, w are the input shape of the forward method (i.e. the input of the first conv)
        """
        result_list = [(h, w)]

        # after conv1
        h, w = get_output_size(h, w, self.conv1)
        result_list.append((h, w))

        # after conv2
        h, w = get_output_size(h, w, self.conv2)
        result_list.append((h, w))

        # after conv3
        for layer in self.shortcut:
            if isinstance(layer, nn.Conv2d):
                result_list.append(result_list[0])  # replicate original input shapes

        # ducktype
        self.input_shapes = result_list

    def get_conv_layer_eigvals(self) -> List[torch.Tensor]:
        """return a list of eigenvalues of each convolution layer"""
        conv_layers = [self.conv1, self.conv2, self.conv3]
        # add shortcut layer
        for layer in self.shortcut:
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)

        # get eigenvalues
        eigs = []
        for (h, w), conv_layer in zip(self.input_shapes, conv_layers):
            kernel = conv_layer.weight
            stride = conv_layer.stride[0]  # * assume square stride
            eigval = get_multi_channel_top_eigval_with_stride(kernel, h, w, stride)
            eigs.append(eigval)

        return eigs


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, nl=nn.GELU()):
        super(ResNet, self).__init__()
        self.nl = nl
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            padding_mode="circular",
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.get_eigvals_funcs = [
            self.conv1.get_conv_layer_eigvals
        ]  # * store get eigenvalue functions binding to the instances
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block._expansion, num_classes)

        self.pooling = nn.AvgPool2d(4)

        # for geometric quantity computation purpose
        self.feature_map = nn.Sequential(
            self.conv1,
            self.bn1,
            self.nl,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.pooling,
            nn.Flatten(start_dim=1),  # take the place of view
        )

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            cur_layer = block(self.in_planes, planes, stride, nl=self.nl)
            layers.append(cur_layer)
            self.in_planes = planes * block._expansion

            # add eigen binding function
            self.get_eigvals_funcs.append(cur_layer.get_conv_layer_eigvals)

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_map(x)
        out = self.linear(out)
        return out

    # --------- for conv layer eigenvalues ----------
    def get_conv_layer_eigvals(self) -> List[torch.Tensor]:
        result_list = []
        for func in self.get_eigvals_funcs:
            result_list += func()
        return result_list


def ResNet18(nl=nn.GELU()):
    return ResNet(BasicBlock, [2, 2, 2, 2], nl=nl)


def ResNet34(nl=nn.GELU()):
    return ResNet(BasicBlock, [3, 4, 6, 3], nl=nl)


def ResNet50(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 4, 6, 3], nl=nl)


def ResNet101(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 4, 23, 3], nl=nl)


def ResNet152(nl=nn.GELU()):
    return ResNet(Bottleneck, [3, 8, 36, 3], nl=nl)
