"""
data io for synthetic dataset
"""

# load packages
import os
from typing import Tuple
import json

import numpy as np
import torch
from torch.utils.data import Dataset


def load_vol_inputs(file_path: str):
    """load vol inputs"""
    with open(file_path, "r") as f:
        data = json.load(f)

    # convert to numpy arrays
    data["X"] = np.array(data["X"])
    data["vol"] = np.array(data["vol"])
    return data


# ============== torch synthetic data samples ================
def train_test_index(
    n: int, test_size: float = 0.5, seed: int = 400
) -> Tuple[torch.Tensor]:
    """
    given the total number of samples, assign train test index
    """
    torch.manual_seed(seed)
    idx = torch.randperm(n)
    test_num = int(n * test_size)
    train_idx, test_idx = idx[:-test_num], idx[-test_num:]
    return train_idx, test_idx


def load_xor_symmetric() -> Tuple[torch.Tensor]:
    """load symmetric xor dataset"""
    X = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    y = torch.tensor([0, 1, 1, 0])
    X_test = torch.empty((0, 2))
    y_test = torch.empty(0)
    return X, X_test, y, y_test

def load_xor_noisy(step: int = 20, std: float = 0.2, seed: int=400) -> Tuple[torch.Tensor]:
    """load noisy xor dataset"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    # positive class
    X1 = np.random.normal([1, 1], std, (step, 2))
    X2 = np.random.normal([-1, -1], std, (step, 2))
    # negative class
    X3 = np.random.normal([-1, 1], std, (step, 2))
    X4 = np.random.normal([1, -1], std, (step, 2))

    X = torch.tensor(np.vstack([X1, X2, X3, X4])).float()
    y = torch.tensor([0] * step * 2 + [1] * step * 2)

    X_test = torch.empty((0, 2))
    y_test = torch.empty(0)
    return X, X_test, y, y_test
    


def load_linear_boundary(
    step: int = 20, test_size: float = 0.5, seed: int = 400
) -> Tuple[torch.Tensor]:
    """
    uniform test data from linear boundary in 2D unit square

    :param step: the sample frequency along each axis. In total step ** 2 samples
    :param test_size: the proportion of generated data for testing
    :param seed: the random seed

    :return train test splitted X and y
    """
    # set randomness
    torch.manual_seed(seed)

    x1, x2 = torch.linspace(-1, 1, step), torch.linspace(-1, 1, step)
    X = torch.cartesian_prod(x1, x2)
    y = (X[:, 0] + X[:, 1] > 0).to(torch.float32)

    # train test split
    train_idx, test_idx = train_test_index(len(y), test_size=test_size, seed=seed)
    X_train, X_test, y_train, y_test = (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )
    return X_train, X_test, y_train, y_test


def load_xor_boundary(
    step: int = 20, test_size: float = 0.5, seed: int = 400
) -> Tuple[torch.Tensor]:
    """
    uniform test data from XOR boundary in 2D unit square

    :param step: the sample frequency along each axis. In total step ** 2 samples
    :param test_size: the proportion of generated data for testing
    :param seed: the random seed

    :return train test splitted X and y
    """
    # set randomness
    torch.manual_seed(seed)

    x1, x2 = torch.linspace(-1, 1, step), torch.linspace(-1, 1, step)
    X = torch.cartesian_prod(x1, x2)
    y = (X[:, 0] * X[:, 1] >= 0).to(torch.float32)

    # train test split
    train_idx, test_idx = train_test_index(len(y), test_size=test_size, seed=seed)
    X_train, X_test, y_train, y_test = (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )
    return X_train, X_test, y_train, y_test


def load_sin_boundary(
    step: int = 20, test_size: float = 0.5, seed: int = 400
) -> Tuple[torch.Tensor]:
    """
    uniform test data from a sinusoidal boundary in 2D unit square

    :param step: the sample frequency along each axis. In total step ** 2 samples
    :param test_size: the proportion of generated data for testing
    :param seed: the random seed

    :return train test splitted X and y
    """
    # set randomness
    torch.manual_seed(seed)

    x1, x2 = torch.linspace(-1, 1, step), torch.linspace(-1, 1, step)
    X = torch.cartesian_prod(x1, x2)
    y = (X[:, 1] > 0.6 * np.sin(7 * X[:, 0] - 1)).to(torch.float32)

    # train test split
    train_idx, test_idx = train_test_index(len(y), test_size=test_size, seed=seed)
    X_train, X_test, y_train, y_test = (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )
    return X_train, X_test, y_train, y_test


def load_sin_random(
    step: int = 20, test_size: float = 0.5, seed: int = 400
) -> Tuple[torch.Tensor]:
    """
    a uniform sampling (i.e. not uniform stepsize sampling) from a sinusoidal boundary in 2D unit square
    """
    # set randomness
    torch.manual_seed(seed)

    X = torch.zeros((step**2, 2)).uniform_(-1, 1)
    y = (X[:, 1] > 0.6 * np.sin(7 * X[:, 0] - 1)).to(torch.float32)

    # train test split
    train_idx, test_idx = train_test_index(len(y), test_size=test_size, seed=seed)
    X_train, X_test, y_train, y_test = (
        X[train_idx],
        X[test_idx],
        y[train_idx],
        y[test_idx],
    )
    return X_train, X_test, y_train, y_test


# =========== torch dataset wrapper ===========
class CustomDataset(Dataset):
    """wrap x and y to a torch dataset"""

    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        super().__init__()
        self.x = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx) -> Tuple[torch.Tensor]:
        return self.x[idx], self.y[idx]
