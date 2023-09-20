"""
data io for synthetic dataset
"""

# load packages
import os 
from typing import Tuple
import json

import numpy as np
import torch

def load_vol_inputs(file_path: str):
    """load vol inputs"""
    with open(file_path, 'r') as f:
        data = json.load(f)

    # convert to numpy arrays
    data['X'] = np.array(data['X'])
    data['vol'] = np.array(data['vol'])
    return data

# ============== torch synthetic data samples ================
def train_test_index(n: int, test_size: float=0.5, seed: int=400) -> Tuple[torch.Tensor]:
    """
    given the total number of samples, assign train test index
    """
    torch.manual_seed(seed)
    idx = torch.randperm(n)
    test_num = int(n * test_size)
    train_idx, test_idx = idx[:-test_num], idx[-test_num:]
    return train_idx, test_idx


def load_linear_boundary(step: int=20, test_size: float=0.5, seed: int=400) -> Tuple[torch.Tensor]:
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
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def load_xor_boundary(step: int=20, test_size: float=0.5, seed: int=400) -> Tuple[torch.Tensor]:
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
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test

def load_sin_boundary(step: int=20, test_size: float=0.5, seed: int=400) -> Tuple[torch.Tensor]:
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
    X_train, X_test, y_train, y_test = X[train_idx], X[test_idx], y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test
