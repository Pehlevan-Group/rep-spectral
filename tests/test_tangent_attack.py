"""
engineering purpose only: test tangent attack improved implementations
"""

# load packages
import torch
import torch.nn as nn

# load files
import sys

sys.path.append("../")
from src.adversarial import TangentAttack


# ====== test case 1: 2D input =======
class SimpleBoundary(nn.Module):
    """decision boundary given by y = 1/2 x"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_pred = torch.hstack(
            ((x[:, [0]] * 2 < x[:, [1]]).float(), (x[:, [0]] * 2 >= x[:, [1]]).float())
        )
        return y_pred


# make dataset
model = SimpleBoundary()
X = torch.tensor([[1.0, 1], [1, -1], [-1, 1], [-1, -1]])
y = torch.tensor([0, 0, 1, 1])
normal_vec_target = torch.tensor([2 / 5 ** (1 / 2), -1 / 5 ** (1 / 2)])
dists_target = torch.tensor(
    [1 / 5 ** (1 / 2), 3 / 5 ** (1 / 2), 3 / 5 ** (1 / 2), 1 / 5 ** (1 / 2)]
)


# ======== test case 2: multidimensional input ==========
class MultiDimInput(nn.Module):
    """binary decision boundary is a round circle at the origin with radius 1"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = x.flatten(start_dim=1).norm(dim=-1, keepdim=True)
        y_pred = torch.hstack((x_norm > 8, x_norm <= 8)).float()
        return y_pred

    def distance(self, x: torch.Tensor) -> torch.Tensor:
        """compute the distance from query to decision boundary"""
        dists = (x.flatten(start_dim=1).norm(dim=-1) - 8).abs()
        return dists


model2 = MultiDimInput()
X2 = torch.ones((4, 3, 5, 5))
X2[1] = -X2[1]
X2[2] = 0.5 * X2[2]
X2[3] = 0.05 * X2[3]
dists_target2 = model2.distance(X2)


class TestTangent:
    """test tangent attacker"""

    def test_boundary(self):
        """verify that the bondary is indeed the desired boundary, within tolerance"""
        # torch.manual_seed(seed)  # * for reproducibility
        attacker = TangentAttack(model, X, None, vmin=-5, vmax=5)
        cur = attacker._initialize()
        cur_dists = (
            (cur - X).view(attacker.batch_size, -1).norm(p=2, dim=-1, keepdim=True)
        )
        deltas = attacker._select_delta(1, cur_dists)
        normal_vecs = attacker._estimate_normal_direction(cur, 5000, deltas)

        assert (normal_vecs[0] - (-normal_vec_target)).norm() < 0.5
        assert (normal_vecs[1] - (-normal_vec_target)).norm() < 0.5
        assert (normal_vecs[2] - (normal_vec_target)).norm() < 0.5
        assert (normal_vecs[3] - (normal_vec_target)).norm() < 0.5

    def test_dist(self):
        # torch.manual_seed(seed)  # * for reproducibility
        attacker = TangentAttack(model, X, None, vmin=-2, vmax=2)
        X_adv = attacker.attack()
        dists = (X - X_adv).norm(p=2, dim=-1)

        assert (dists / dists_target - 1).abs().max() < 0.01

    def test_dist2(self):
        attacker = TangentAttack(model2, X2, None, vmin=-1.5, vmax=1.5, T=100)
        X_adv = attacker.attack()
        dists = (X2 - X_adv).flatten(start_dim=1).norm(p=2, dim=-1)

        assert (dists / dists_target2 - 1).abs().max() < 0.05
