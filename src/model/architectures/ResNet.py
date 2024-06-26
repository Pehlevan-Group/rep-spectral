"""
modified directly from the source code 

the following modifications are made for the project:
* note that to match with theoretical predictions, all padding are circular instead of zero padding
* each model is equipped with methods for extracting convolutional layers' eigen(singular) values for 
  later use
* each model, after calling forward, memorize the image shape, since the singular value depends on 
  the shape of the input
"""

# load packages
from typing import List, Optional, Callable, Union, Type
import torch
from torch import Tensor
import torch.nn as nn

# load file
from .utils import Conv2dWrap


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2dWrap(conv3x3(inplanes, planes, stride))
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2dWrap(conv3x3(planes, planes))
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = Conv2dWrap(conv1x1(inplanes, width))
        self.bn1 = norm_layer(width)
        self.conv2 = Conv2dWrap(conv3x3(width, width, stride, groups, dilation))
        self.bn2 = norm_layer(width)
        self.conv3 = Conv2dWrap(conv1x1(width, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        small_input=True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.small_input = small_input # small for (32 * 32), otherwise for (224 * 224)
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        if self.small_input:
            self.conv1 = Conv2dWrap(
                nn.Conv2d(
                    3, self.inplanes, kernel_size=3, padding=1, bias=False
                )  # for 32 * 32 inputs
            )
        else:
            self.conv1 = Conv2dWrap(
                nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False) # for 224 * 224 inputs
            )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # * testing without kaiming_normal initialization
        if not self.small_input:
            for m in self.modules():
                if isinstance(m, Conv2dWrap):
                    nn.init.kaiming_normal_(m.wrap.weight, mode="fan_out", nonlinearity="relu")
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        # # Zero-initialize the last BN in each residual branch,
        # # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             # type: ignore[arg-type]
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             # type: ignore[arg-type]
        #             nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2dWrap(conv1x1(self.inplanes, planes * block.expansion, stride)),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def feature_map(self, x: torch.Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.small_input:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.feature_map(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor, return_features=False) -> Tensor:
        features = self.feature_map(x)
        out = self.fc(features)
        if return_features:
            return features, out
        else:
            return out

    # ------- for regularization ---------
    def _chain_generators(self, *generators):
        """chain multiple named_parameter generators"""
        for gen in generators:
            yield from gen

    def get_conv_layers(self, max_layer: int = 4) -> List[Conv2dWrap]:
        assert max_layer in [1, 2, 3, 4], f"max_layer {max_layer} not in [1, 2, 3, 4]"

        # pre block conv layer
        conv_layers = [self.conv1]

        # each layer
        generator_list = []
        if max_layer >= 1:
            generator_list.append(self.layer1.named_modules())
        if max_layer >= 2:
            generator_list.append(self.layer2.named_modules())
        if max_layer >= 3:
            generator_list.append(self.layer3.named_modules())
        if max_layer >= 4:
            generator_list.append(self.layer4.named_modules())
        aggregated_generator = self._chain_generators(*generator_list)

        # retrieve conv2dtype
        for name, m in aggregated_generator:
            if isinstance(m, Conv2dWrap):
                conv_layers.append(m)

        return conv_layers

    def get_conv_layer_eigvals(self, max_layer: int = 4) -> List[torch.Tensor]:
        """get list of eigenvalues"""
        result_list = []
        conv_layers = self.get_conv_layers(max_layer=max_layer)
        for conv_layer in conv_layers:
            result_list.append(conv_layer._get_conv_layer_eigvals())

        return result_list

    def get_conv_layers_names(self, max_layer: int = 4) -> List[str]:
        """use string to store convolution layer right singular vector"""
        assert max_layer in [1, 2, 3, 4], f"max_layer {max_layer} not in [1, 2, 3, 4]"
        names = [
            name
            for (name, _) in list(self.named_parameters())
            if "conv" in name or "downsample.0" in name
        ]

        # filter out
        if max_layer < 4:
            names = [name for name in names if "layer4" not in name]
        if max_layer < 3:
            names = [name for name in names if "layer3" not in name]
        if max_layer < 2:
            names = [name for name in names if "layer2" not in name]

        # replace "." with "_"
        names = [name.replace(".", "_") for name in names]
        return names


# TODO: register l2sp buffer snapshots
# TODO: currently the nl api is fake


# ========== concrete initializations ========
def ResNet18(num_classes=10, nl=nn.ReLU(), small_input=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, small_input=small_input)


def ResNet34(num_classes=10, nl=nn.ReLU(), small_input=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, small_input=small_input)


def ResNet50(num_classes=10, nl=nn.ReLU(), small_input=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, small_input=small_input)


def ResNet101(num_classes=10, nl=nn.ReLU(), small_input=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, small_input=small_input)


def ResNet152(num_classes=10, nl=nn.ReLU(), small_input=True):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, small_input=small_input)
