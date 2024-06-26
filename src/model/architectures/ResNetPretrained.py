"""
for transfer learning only: load and wrap pretrained model for regularizations

1. change convolutions to periodic
2. register buffers such as the input shape to each convolution layer for later spectral regularization
"""

# load packages
from typing import Callable, List, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .utils import Conv2dWrap


class ResNet50Pretrained(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        weights=ResNet50_Weights.IMAGENET1K_V2,
        small_conv1: bool = False,
    ):
        """
        default to v2, the one with higher top-1 accuracy
        :param num_classes: the number of classes for the final linear layer
        :param weights: the version of weights to use for ResNet50
        :param small_conv1: change the input conv1 to kernel size of 3 (for small dimensional dataset such as cifar10)
        """
        super().__init__()

        # load pretrained weights on ImageNet
        model = resnet50(weights=weights)

        # adapt to smaller dimensional task (e.g. cifar10)
        self.small_conv1 = small_conv1
        if small_conv1:
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )

        # modify the linear head
        self.fc = nn.Linear(model.fc.in_features, num_classes, bias=True)
        # unit initialization
        nn.init.normal_(self.fc.weight, 0, 1)
        nn.init.normal_(self.fc.bias, 0, 1)
        model.fc = nn.Identity()  # for representation readouts

        # register model
        self.model = model

        # modify to periodic wrapper
        self._set_conv()

        # get a pretrained snapshot as buffer (for computing TL regularization loss)
        self._register_snapshot(weights=weights)

    def _get_pt_param_name(self, name: str) -> str:
        """encode pretrained parameter name"""
        return "pt_" + name.replace(".", "_")

    def _register_snapshot(self, weights=ResNet50_Weights.IMAGENET1K_V2):
        """
        move all trainable parameters in the pretrained model to buffer
        so that gradient would not flow to this static snapshot
        """
        # read a snapshot
        snapshot = resnet50(weights=weights)

        # register all parameters as buffers
        for name, parameter in snapshot.named_parameters():
            if (
                name != "fc.weight"
                and name != "fc.bias"
                and (name != "conv1.weight" or not self.small_conv1)
            ):
                pt_name = self._get_pt_param_name(name)
                self.register_buffer(pt_name, parameter, persistent=False)

    def _get_pre_layer_conv(self) -> List[nn.Conv2d]:
        return [self.model.conv1]

    def _set_pre_layer_conv(self):
        self.model.conv1 = Conv2dWrap(self.model.conv1)

    def _get_layer1_conv(self) -> List[nn.Conv2d]:
        # layer 1 (3 bottlenecks)
        layer1_conv = [
            self.model.layer1[0].conv1,
            self.model.layer1[0].conv2,
            self.model.layer1[0].conv3,
            self.model.layer1[0].downsample[0],
            self.model.layer1[1].conv1,
            self.model.layer1[1].conv2,
            self.model.layer1[1].conv3,
            self.model.layer1[2].conv1,
            self.model.layer1[2].conv2,
            self.model.layer1[2].conv3,
        ]
        return layer1_conv

    def _set_layer1_conv(self):
        """set layer1 conv layers to conv wrappers"""
        self.model.layer1[0].conv1 = Conv2dWrap(self.model.layer1[0].conv1)
        self.model.layer1[0].conv2 = Conv2dWrap(self.model.layer1[0].conv2)
        self.model.layer1[0].conv3 = Conv2dWrap(self.model.layer1[0].conv3)
        self.model.layer1[0].downsample[0] = Conv2dWrap(
            self.model.layer1[0].downsample[0]
        )
        self.model.layer1[1].conv1 = Conv2dWrap(self.model.layer1[1].conv1)
        self.model.layer1[1].conv2 = Conv2dWrap(self.model.layer1[1].conv2)
        self.model.layer1[1].conv3 = Conv2dWrap(self.model.layer1[1].conv3)
        self.model.layer1[2].conv1 = Conv2dWrap(self.model.layer1[2].conv1)
        self.model.layer1[2].conv2 = Conv2dWrap(self.model.layer1[2].conv2)
        self.model.layer1[2].conv3 = Conv2dWrap(self.model.layer1[2].conv3)

    def _get_layer2_conv(self) -> List[nn.Conv2d]:
        # layer 2, 4 bottle necks
        layer2_conv = [
            self.model.layer2[0].conv1,
            self.model.layer2[0].conv2,
            self.model.layer2[0].conv3,
            self.model.layer2[0].downsample[0],
            self.model.layer2[1].conv1,
            self.model.layer2[1].conv2,
            self.model.layer2[1].conv3,
            self.model.layer2[2].conv1,
            self.model.layer2[2].conv2,
            self.model.layer2[2].conv3,
            self.model.layer2[3].conv1,
            self.model.layer2[3].conv2,
            self.model.layer2[3].conv3,
        ]
        return layer2_conv

    def _set_layer2_conv(self):
        """set layer 2 conv to conv wrappers"""
        self.model.layer2[0].conv1 = Conv2dWrap(self.model.layer2[0].conv1)
        self.model.layer2[0].conv2 = Conv2dWrap(self.model.layer2[0].conv2)
        self.model.layer2[0].conv3 = Conv2dWrap(self.model.layer2[0].conv3)
        self.model.layer2[0].downsample[0] = Conv2dWrap(
            self.model.layer2[0].downsample[0]
        )
        self.model.layer2[1].conv1 = Conv2dWrap(self.model.layer2[1].conv1)
        self.model.layer2[1].conv2 = Conv2dWrap(self.model.layer2[1].conv2)
        self.model.layer2[1].conv3 = Conv2dWrap(self.model.layer2[1].conv3)
        self.model.layer2[2].conv1 = Conv2dWrap(self.model.layer2[2].conv1)
        self.model.layer2[2].conv2 = Conv2dWrap(self.model.layer2[2].conv2)
        self.model.layer2[2].conv3 = Conv2dWrap(self.model.layer2[2].conv3)
        self.model.layer2[3].conv1 = Conv2dWrap(self.model.layer2[3].conv1)
        self.model.layer2[3].conv2 = Conv2dWrap(self.model.layer2[3].conv2)
        self.model.layer2[3].conv3 = Conv2dWrap(self.model.layer2[3].conv3)

    def _get_layer3_conv(self) -> List[nn.Conv2d]:
        # layer 3, 6 bottle necks
        layer3_conv = [
            self.model.layer3[0].conv1,
            self.model.layer3[0].conv2,
            self.model.layer3[0].conv3,
            self.model.layer3[0].downsample[0],
            self.model.layer3[1].conv1,
            self.model.layer3[1].conv2,
            self.model.layer3[1].conv3,
            self.model.layer3[2].conv1,
            self.model.layer3[2].conv2,
            self.model.layer3[2].conv3,
            self.model.layer3[3].conv1,
            self.model.layer3[3].conv2,
            self.model.layer3[3].conv3,
            self.model.layer3[4].conv1,
            self.model.layer3[4].conv2,
            self.model.layer3[4].conv3,
            self.model.layer3[5].conv1,
            self.model.layer3[5].conv2,
            self.model.layer3[5].conv3,
        ]
        return layer3_conv

    def _set_layer3_conv(self):
        """set layer3 conv to conv wrappers"""
        self.model.layer3[0].conv1 = Conv2dWrap(self.model.layer3[0].conv1)
        self.model.layer3[0].conv2 = Conv2dWrap(self.model.layer3[0].conv2)
        self.model.layer3[0].conv3 = Conv2dWrap(self.model.layer3[0].conv3)
        self.model.layer3[0].downsample[0] = Conv2dWrap(
            self.model.layer3[0].downsample[0]
        )
        self.model.layer3[1].conv1 = Conv2dWrap(self.model.layer3[1].conv1)
        self.model.layer3[1].conv2 = Conv2dWrap(self.model.layer3[1].conv2)
        self.model.layer3[1].conv3 = Conv2dWrap(self.model.layer3[1].conv3)
        self.model.layer3[2].conv1 = Conv2dWrap(self.model.layer3[2].conv1)
        self.model.layer3[2].conv2 = Conv2dWrap(self.model.layer3[2].conv2)
        self.model.layer3[2].conv3 = Conv2dWrap(self.model.layer3[2].conv3)
        self.model.layer3[3].conv1 = Conv2dWrap(self.model.layer3[3].conv1)
        self.model.layer3[3].conv2 = Conv2dWrap(self.model.layer3[3].conv2)
        self.model.layer3[3].conv3 = Conv2dWrap(self.model.layer3[3].conv3)
        self.model.layer3[4].conv1 = Conv2dWrap(self.model.layer3[4].conv1)
        self.model.layer3[4].conv2 = Conv2dWrap(self.model.layer3[4].conv2)
        self.model.layer3[4].conv3 = Conv2dWrap(self.model.layer3[4].conv3)
        self.model.layer3[5].conv1 = Conv2dWrap(self.model.layer3[5].conv1)
        self.model.layer3[5].conv2 = Conv2dWrap(self.model.layer3[5].conv2)
        self.model.layer3[5].conv3 = Conv2dWrap(self.model.layer3[5].conv3)

    def _get_layer4_conv(self) -> List[nn.Conv2d]:
        # layer 4, 3 bottle necks
        layer4_conv = [
            self.model.layer4[0].conv1,
            self.model.layer4[0].conv2,
            self.model.layer4[0].conv3,
            self.model.layer4[0].downsample[0],
            self.model.layer4[1].conv1,
            self.model.layer4[1].conv2,
            self.model.layer4[1].conv3,
            self.model.layer4[2].conv1,
            self.model.layer4[2].conv2,
            self.model.layer4[2].conv3,
        ]
        return layer4_conv

    def _set_layer4_conv(self):
        """set layer4 conv to conv wrappers"""
        self.model.layer4[0].conv1 = Conv2dWrap(self.model.layer4[0].conv1)
        self.model.layer4[0].conv2 = Conv2dWrap(self.model.layer4[0].conv2)
        self.model.layer4[0].conv3 = Conv2dWrap(self.model.layer4[0].conv3)
        self.model.layer4[0].downsample[0] = Conv2dWrap(
            self.model.layer4[0].downsample[0]
        )
        self.model.layer4[1].conv1 = Conv2dWrap(self.model.layer4[1].conv1)
        self.model.layer4[1].conv2 = Conv2dWrap(self.model.layer4[1].conv2)
        self.model.layer4[1].conv3 = Conv2dWrap(self.model.layer4[1].conv3)
        self.model.layer4[2].conv1 = Conv2dWrap(self.model.layer4[2].conv1)
        self.model.layer4[2].conv2 = Conv2dWrap(self.model.layer4[2].conv2)
        self.model.layer4[2].conv3 = Conv2dWrap(self.model.layer4[2].conv3)

    def _set_conv(self):
        """wrap convolution layers"""
        # set layers to conv wrappers
        self._set_pre_layer_conv()
        self._set_layer1_conv()
        self._set_layer2_conv()
        self._set_layer3_conv()
        self._set_layer4_conv()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """output both feature representations and logits"""
        features = self.model(x)
        logits = self.fc(features)
        return features, logits

    # ----------------- for regularization only ---------------------
    def get_conv_layer_eigvals(self, max_layer: int = 4) -> List[torch.Tensor]:
        """get list of eigenvalues"""
        result_list = []
        for func in self.get_conv_layer_eigvals_funcs(max_layer=max_layer):
            result_list.append(func())
        return result_list

    def get_conv_layer_eigvals_funcs(self, max_layer: int = 4) -> List[Callable]:
        """get functions bind to conv2d wrappers"""
        conv_layers = self.get_conv_layers(max_layer=max_layer)
        result_list = []
        for conv_layer in conv_layers:
            result_list.append(conv_layer._get_conv_layer_eigvals)

        return result_list

    def get_conv_layers(self, max_layer: int = 4) -> List[Conv2dWrap]:
        """get references to all convolution layers"""
        assert max_layer in [1, 2, 3, 4], f"max_layer {max_layer} not in [1, 2, 3, 4]"

        conv_layers = self._get_pre_layer_conv() if not self.small_conv1 else []
        if max_layer >= 1:
            conv_layers += self._get_layer1_conv()
        if max_layer >= 2:
            conv_layers += self._get_layer2_conv()
        if max_layer >= 3:
            conv_layers += self._get_layer3_conv()
        if max_layer >= 4:
            conv_layers += self._get_layer4_conv()

        return conv_layers

    def get_conv_layers_names(self, max_layer: int = 4) -> List[str]:
        """use string to store convolution layer right singular vector"""
        assert max_layer in [1, 2, 3, 4], f"max_layer {max_layer} not in [1, 2, 3, 4]"
        names = [
            name
            for (name, _) in list(self.model.named_parameters())
            if "conv" in name or "downsample.0" in name
        ]

        # filter out
        if max_layer < 4:
            names = [name for name in names if "layer4" not in name]
        if max_layer < 3:
            names = [name for name in names if "layer3" not in name]
        if max_layer < 2:
            names = [name for name in names if "layer2" not in name]

        # remove initial layer if swapped
        if self.small_conv1:
            names = [name for name in names if "conv1.wrap.weight" != name]

        # replace "." with "_"
        names = [name.replace(".", "_") for name in names]
        return names

    def get_param_l2sp(self) -> torch.Tensor:
        """
        L2-SP measures the l2 deviations between the new model and pretrained model
        up to the feature space layer
        """
        loss = 0
        for name, parameter in self.model.named_parameters():
            # ignore aligning the first input conv1 layer if changed and randomly initialized
            if name == "conv1.wrap.weight" and self.small_conv1:
                continue

            pt_name = self._get_pt_param_name(name).replace("wrap_", "")
            parameter_pretrained = getattr(self, pt_name)
            loss += torch.sum((parameter - parameter_pretrained) ** 2)
        return loss
