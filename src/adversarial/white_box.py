"""
white box attacks

- Projected Gradient Descent 
"""

# load packages
from typing import Callable, Union
import torch
import torch.nn as nn


def projection(
    x: torch.Tensor, x_adv: torch.Tensor, norm: Union[str, int], eps: float
) -> torch.Tensor:
    """projection onto the l-norm ball around x"""
    if norm == "inf":
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
    else:
        delta = x_adv - x

        # assume first dim is the batch dimension
        scaling_factor = delta.flatten(start_dim=1).norm(norm, dim=1).clamp(min=eps)
        delta *= eps / scaling_factor.view(
            x.shape[0], *[1 for _ in range(len(x.shape) - 1)]
        )
        x_adv = x + delta

    return x_adv


def pgd_perturbation(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    loss_fn: Callable,
    norm: Union[str, int] = 2,
    k: int = 10,
    step_size: float = 0.1,
    eps: float = 1,
) -> torch.Tensor:
    """
    obtain perturbed samples after a fixed number of iterations, projected
    onto the l-norm ball around the original sample

    :param model, X, y: torch model, features, and labels
    :param norm: l2 norm or infinity norm
    :param k: number of steps
    :param step_size: the multiplier for gradient
    :param eps: max perturbation level
    """
    # make copy
    X_copy = X.clone().detach().requires_grad_(True).to(X.device)
    X_adv = X_copy.clone().detach().requires_grad_(True).to(X_copy.device)

    for _ in range(k):
        # dump gradient information
        X_adv.grad = None

        # get new gradients
        y_pred = model(X_adv)
        loss = loss_fn(y_pred, y)
        loss.backward()

        with torch.no_grad():
            if norm == "inf":
                gradients = X_adv.grad.sign() * step_size
            else:
                gradients = X_adv.grad
                # normalize
                gradients /= (
                    gradients.flatten(start_dim=1)
                    .norm(norm, dim=-1)
                    .view(X.shape[0], *[1 for _ in range(len(X.shape) - 1)])
                )
                gradients *= step_size

            # update (gradient ascent)
            X_copy += gradients

        # project back to area
        X_copy = projection(X, X_copy, norm, eps)

    return X_copy
