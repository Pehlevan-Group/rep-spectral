"""
utils for contrastive downstream evaluations
"""
# load packages
from typing import Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


@torch.no_grad()
def _get_feature_representations(
    model: nn.Module, loader: torch.utils.data.DataLoader, device="cpu"
) -> np.ndarray:
    """retrieve the feature representation of model"""
    features = []
    for X, _ in loader:
        X = X.to(device)
        cur_features = model.feature_map(X).detach().cpu().numpy()
        features.append(cur_features)
    features = np.vstack(features)
    return features


@torch.no_grad()
def _get_feature_representations_and_label(
    model: nn.Module, loader: torch.utils.data.DataLoader, device="cpu"
) -> np.ndarray:
    """retrieve the feature representation of model"""
    features, labels = [], []
    for X, y in loader:
        X = X.to(device)
        cur_features = model.feature_map(X).detach().cpu().numpy()
        features.append(cur_features)
        labels.append(y.detach().cpu().numpy())
    features = np.vstack(features)
    labels = np.hstack(labels)
    return features, labels


def _train_downstream_logistic_regression(
    train_features: np.ndarray, train_labels: np.ndarray, random_state=None
) -> LogisticRegression:
    """
    unify training parameters for logistic regression training
    """
    # * use lbfgs for fast convergence
    lr = LogisticRegression(solver="lbfgs", n_jobs=-1, random_state=random_state).fit(
        train_features, train_labels
    )
    print(f"fit acc: {lr.score(train_features, train_labels):.4f}")
    return lr


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
    train_features = _get_feature_representations(model, train_loader, device=device)
    train_labels = train_labels.detach().cpu().numpy()

    # get test feature representation
    test_features = _get_feature_representations(model, test_loader, device=device)
    test_labels = test_labels.detach().cpu().numpy()

    lr = _train_downstream_logistic_regression(
        train_features, train_labels, random_state=random_state
    )
    downstream_train_acc = lr.score(train_features, train_labels)
    downstream_test_acc = lr.score(test_features, test_labels)

    return downstream_train_acc, downstream_test_acc


class LogisticRegressionTorch(nn.Module):
    """wrap trained logistic regression to a torch module"""

    def __init__(self, lr_model: LogisticRegression, device="cpu"):
        """
        :param lr_model: a model after being fitted
        """
        super().__init__()
        self.coefs = torch.tensor(lr_model.coef_).to(torch.float32).to(device)
        self.intercept = torch.tensor(lr_model.intercept_).to(torch.float32).to(device)
        self.softmax = nn.Softmax(dim=-1)

    @torch.no_grad()
    def _forward_multiclass(self, X: torch.Tensor) -> torch.Tensor:
        result = self.softmax(X @ self.coefs.T + self.intercept)
        return result

    @torch.no_grad()
    def _forward_binary(self, X: torch.Tensor) -> torch.Tensor:
        """manually create probabilities belonging to zero/one"""
        out = X @ self.coefs.T + self.intercept
        prob1 = 1 / (1 + torch.exp(-out))
        prob0 = 1 - prob1
        result = torch.hstack([prob0, prob1])
        return result

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # binary
        if len(self.intercept) == 1:
            return self._forward_binary(X)
        else:
            return self._forward_multiclass(X)


class ContrastiveWrap(nn.Module):
    """
    wrap contrastive model transfer learning evaluation into an nn module
    this architecture consists of [contrastive feature map, logistic regression]

    particularly, we extract the trained coefficients and wrap it to torch compatible
    operations to support cuda.
    """

    def __init__(
        self,
        contrastive_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        device="cpu",
        random_state=None,
    ):
        """
        :param contrastive_model: Barlow or SimCLR
        :param train_loader: the torch loader for fetching train samples and labels
        :param device: cpu vs cuda
        :param random_state: the random_state argument for training logistic regression
        """
        super().__init__()
        # get feature representations and labels
        train_features, train_labels = _get_feature_representations_and_label(
            contrastive_model, train_loader, device=device
        )

        # train lr
        lr = _train_downstream_logistic_regression(
            train_features, train_labels, random_state
        )

        # transfer end to end map: feature map + logistic
        self.feature_map = contrastive_model.feature_map
        self.logistic_out = LogisticRegressionTorch(lr, device=device)

    @torch.no_grad()
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        features = self.feature_map(X)
        out = self.logistic_out(features)
        return out
