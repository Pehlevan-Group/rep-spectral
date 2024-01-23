"""
utils for contrastive downstream evaluations
"""
# load packages
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


def get_contrastive_acc(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    train_labels: torch.Tensor,
    test_loader: torch.utils.data.DataLoader,
    test_labels: torch.Tensor,
    device="cpu",
    random_state=None,
) -> Tuple[float]:
    """
    a helper function that evaluates downstream acc

    :param model: a contrastive model
    :param dataloader: train and test, for obtaining feature representations
    :param labels: the labels of train test samples
    :parma random_state: the random seed
    """
    # get train feature representations
    train_features = []
    with torch.no_grad():
        for X, labels in train_loader:
            X = X.to(device)
            train_features.append(model.feature_map(X).detach().cpu().numpy())

    train_features = np.vstack(train_features)
    train_labels = train_labels.detach().cpu().numpy()

    # get test feature representation
    test_features = []
    with torch.no_grad():
        for X, labels in test_loader:
            X = X.to(device)
            test_features.append(model.feature_map(X).detach().cpu().numpy())

    test_features = np.vstack(test_features)
    test_labels = test_labels.detach().cpu().numpy()

    lr = LogisticRegression(
        solver="lbfgs", n_jobs=-1, random_state=random_state
    ).fit(train_features, train_labels)
    downstream_train_acc = lr.score(train_features, train_labels)
    downstream_test_acc = lr.score(test_features, test_labels)

    return downstream_train_acc, downstream_test_acc
