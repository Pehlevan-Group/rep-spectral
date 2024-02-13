"""
transfer learning wrapper: discard feature representations
"""

# load packages
import torch
import torch.nn as nn


class TransferWrap(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # discard features
        _, y_pred = self.model(x)
        return y_pred
