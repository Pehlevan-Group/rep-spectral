"""
for transfer learning only: load and wrap pretrained model for regularizations

1. change convolutions to periodic
2. register buffers such as the input shape to each convolution layer for later spectral regularization
"""

# load packages
from typing import List, Tuple
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

from .utils import get_multi_channel_top_eigval_with_stride


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


class ResNet50Pretrained(nn.Module):
    def __init__(self, num_classes: int = 1000, weights=ResNet50_Weights.IMAGENET1K_V2):
        """
        default to v2, the one with higher top-1 accuracy
        :param num_classes: the number of classes for the final linear layer
        :param weights: the version of weights to use for ResNet50
        """
        super().__init__()

        # load pretrained weights on ImageNet
        model = resnet50(weights=weights)

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
            if name != "fc.weight" and name != "fc.bias":
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
        assert max_layer in [1, 2, 3, 4], f"max_layer {max_layer} not in [1, 2, 3, 4]"

        conv_layers = self._get_pre_layer_conv()
        if max_layer >= 1:
            conv_layers += self._get_layer1_conv()
        if max_layer >= 2:
            conv_layers += self._get_layer2_conv()
        if max_layer >= 3:
            conv_layers += self._get_layer3_conv()
        if max_layer >= 4:
            conv_layers += self._get_layer4_conv()

        result_list = []
        for conv_layer in conv_layers:
            result_list.append(conv_layer._get_conv_layer_eigvals())

        return result_list

    def get_param_l2sp(self) -> torch.Tensor:
        """
        L2-SP measures the l2 deviations between the new model and pretrained model
        up to the feature space layer
        """
        loss = 0
        for name, parameter in self.model.named_parameters():
            pt_name = self._get_pt_param_name(name).replace("wrap_", "")
            parameter_pretrained = getattr(self, pt_name)
            loss += torch.sum((parameter - parameter_pretrained) ** 2)
        return loss
