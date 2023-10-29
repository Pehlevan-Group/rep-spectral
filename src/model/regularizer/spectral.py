"""
regularizing the spectral norm
from https://arxiv.org/pdf/1705.10941.pdf
"""

# load packages
import torch
import torch.nn as nn

def spectral_ub_regularizer_autograd(model: nn.Module):
    """
    give spectral bound on also the last layer
    (served as benchmark of our noval method)
    """
    # sequential passing in
    eigs = []
    
    # multi-hidden-layer
    if hasattr(model, 'model'):
        layers = model.model 
    # single-hidden-layer
    else:
        layers = [model.lin1, model.lin2]
    
    # sequential passing
    for layer in layers:
        # a linear layer
        if isinstance(layer, nn.Linear):
            W = layer.weight
            eig = torch.linalg.eigvalsh(W.T @ W).max()
            eigs.append(eig)

    # compute regularization
    reg_term = sum(eigs)

    return reg_term