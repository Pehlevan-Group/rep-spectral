"""
model architecture
"""

# load packages
import os
import torch
import torch.nn as nn

# constants
INIT_MEAN = 0
INIT_STD = 1


# =========== utility layers
def weights_init(model: nn.Module):
    """initialize weights for linear layers in model to be normal"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(INIT_MEAN, INIT_STD)
            m.bias.data.normal_(INIT_MEAN, INIT_STD)


class ScaleLayer(nn.Module):
    """only perform scaling, normalized by width"""

    def __init__(self, width) -> None:
        super().__init__()
        self.width = width

    def forward(self, X):
        return (1 / self.width) ** (1 / 2) * X


# ============ basic architectures ==============
class SLP(nn.Module):
    """single hidden layer MLP"""

    def __init__(self, width=2, input_dim=2, output_dim=1, nl=nn.Sigmoid()):
        super(SLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, width)
        self.lin2 = nn.Linear(width, output_dim)
        self.width = width
        self.input_dim = input_dim 
        self.output_dim = output_dim
        self.nl = nl

        # define feature map for kernel computations
        self.feature_map = nn.Sequential(self.lin1, self.nl)

        self.model = nn.Sequential(self.feature_map, self.lin2, ScaleLayer(width))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.model(x)
        if self.output_dim == 1:
            return self.sigmoid(out)
        else:
            return out 


class MLP(nn.Module):
    """multiple hidden layer MLP"""

    def __init__(self, dimensions, nl=nn.Sigmoid()) -> None:
        """
        :param dimensions: an iterable storing the dimension at each layer.
            The first dim is the input dimension and last is the output dimension
        :param nl: the non linearity
        """
        super(MLP, self).__init__()

        # check type and input
        assert isinstance(
            dimensions, (list, tuple)
        ), "dimension argument not of list or tuple"
        assert len(dimensions) >= 2, "number of dimensions should be at least two"
        self.dimensions = dimensions

        # putup layers
        layers = []
        feature_maps = []  # store feature maps
        for i in range(1, len(self.dimensions)):
            cur_in_dim, cur_out_dim = self.dimensions[i - 1], self.dimensions[i]
            cur_layer = nn.Linear(cur_in_dim, cur_out_dim)
            layers.append(cur_layer)
            # scale
            if i > 1:
                layers.append(ScaleLayer(cur_in_dim))

            # non linearity except the last layer
            if i < len(self.dimensions) - 1:
                layers.append(nl)

                # defined feature map for layer i
                feature_maps.append(
                    nn.Sequential(*layers)
                )  # whatever is already in layer

        self.model = nn.Sequential(*layers)

        # append feature maps to self
        self.feature_maps = feature_maps

    def forward(self, X):
        return self.model(X)
