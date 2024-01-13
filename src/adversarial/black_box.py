"""
Implementation of black-box attackers

- TangentAttack: https://openreview.net/forum?id=g0wang64Zjd. 
  Implementation adapted from https://github.com/machanic/TangentAttack. Here all operations are batched for efficiency
"""

# load packages
import warnings
import logging
from tqdm import tqdm
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# load file
from .utils import CustomAdversarialDataset


class Attacker:
    """parent class of collections of attackers"""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def check_attack_success(
        self,
        new_samples: torch.Tensor,
        true_labels: torch.LongTensor,
        target_labels: torch.LongTensor,
    ) -> torch.BoolTensor:
        """
        check if an attack is successful
        - targeted: success if new sample has target_label
        - untargeted: success if new sample does not have true label

        :param new_samples: the proposed perturbation. The first dim is batch dim
        :param true_label: the true labels
        :param target_label: None for untargeted attack
        :return True if the attack is successful
        """
        # check prediction
        new_sample_pred = self.get_decision(new_samples)

        # untargeted
        if target_labels is None:
            success_indicators = new_sample_pred != true_labels
        # targeted
        else:
            success_indicators = new_sample_pred == target_labels
        return success_indicators

    def binary_search(
        self,
        x: torch.Tensor,
        x_adv: torch.Tensor,
        y: torch.Tensor,
        y_target: torch.Tensor,
        tol: float = 1e-5,
    ) -> torch.Tensor:
        """
        binary search to find a sample at the decision boundary
        stop after a pre-specified tolerance is reached

        :param x: the original sample
        :param x_adv: the adversarial sample. The first dimension is batch dimension
        :param y: the original prediction
        :param y_target: the desired adversarial class
        :param tol: a tolerance level (for l2-norm measure)
        :return a sample at the adversarial boundary
        """
        num_dims = len(x_adv.shape) - 1
        lefts = torch.zeros((self.batch_size, *([1] * num_dims))).to(x_adv.device)
        rights = torch.ones((self.batch_size, *([1] * num_dims))).to(x_adv.device)
        # binary search
        while torch.max((rights - lefts)) > tol:
            mids = (lefts + rights) / 2
            x_mid = (1 - mids) * x + mids * x_adv
            success_indicators = self.check_attack_success(x_mid, y, y_target)
            success_indicators = success_indicators.float().reshape(
                *mids.shape
            )  # convert to float and match dimension

            # update boundaries
            lefts = (1 - success_indicators) * mids + success_indicators * lefts
            rights = success_indicators * mids + (1 - success_indicators) * rights

        result = (1 - rights) * x + rights * x_adv
        return result

    @torch.no_grad()
    def get_decision(self, x: torch.Tensor) -> torch.LongTensor:
        """
        compute y from model
        note that binary model only has one output dim (use 0 as a boundary to give decision)
        otherwise, use argmax

        :param x: model input
        :return decision for x
        """
        pred = self.model(x)
        # binary
        if pred.shape[1] == 1:
            decisions = (pred >= 0).long()
        else:
            decisions = pred.argmax(dim=-1)
        return decisions


class TangentAttack(Attacker):
    """
    Tangent Attacker from https://openreview.net/forum?id=g0wang64Zjd
    we use the hemisphere implementation with l2 distance measure
    """

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        x_targets: torch.Tensor,
        adv_batch_size: int = 16,
        tol: float = 1e-5,
        gamma: float = 1.0,
        vmin: float = 0.0,
        vmax: float = 1.0,
        T: int = 40,
        init_num_eval: int = 100,
        max_num_evals: int = 10000,
    ) -> None:
        """
        Initialize tangent attacker

        The number of random direction proposals to be made for finding an appropriate normal direction
        scales with square root of the number of iterations, and upper bounded by max_num_evals

        :param model: a trained pytorch model
        :param x: the sample to be attacked
        :param x_targets: specify a sample of another class, can be None for non-targeted attack
        :param adv_batch_size: the batch size for adversarial sample feeding
        :param tol: the threshold for stopping the binary search
        :param gamma: the scalar for the number of evaluations
        :param vmin, vmax: the feasible range of inputs, typically within 0 and 1 for images
        :param T: maximum iterations to find tangent point + projection
        :param init_num_eval: the starting number of evaluations, scales
        :param max_num_evals: give an bound number of proposals
        """
        super().__init__()
        self.model = model
        self.x_samples = x
        self.batch_size = adv_batch_size
        self.device = x.device
        self.dim = np.prod([*x.shape[1:]])
        self.shapes = x.shape[1:]

        # make dataset
        adversarial_dataset = CustomAdversarialDataset(x, x_targets)
        self.adversarial_dataloader = DataLoader(
            adversarial_dataset, batch_size=adv_batch_size, shuffle=False
        )

        # feasible range of input
        self.vmin = vmin
        self.vmax = vmax

        # parameters
        self.tol = tol
        self.T = T
        self.gamma = gamma
        self.init_num_eval = init_num_eval
        self.max_num_evals = max_num_evals

    def _initialize(
        self,
        x: torch.Tensor,
        x_targets: torch.Tensor,
        y: torch.Tensor,
        y_target: torch.Tensor,
        max_search_times: int = 5000,
    ) -> torch.Tensor:
        """
        initialize x0 to start tangent attack:
        - untargeted: uniformly choose images from within feasible range,
            and then binary search as a starting point
        - targeted: binary search between the current sample and adversarial sample

        :param max_search_times: put a maximum number of init
        """
        # non targeted attack
        if x_targets is None:
            # batched operation: if success, append to result, otherwise keep searching
            x_original = x
            idx_tensor = torch.arange(x.shape[0]).to(self.device)
            x0_inserted, insert_idx = (
                torch.empty(0, *x.shape[1:]).to(self.device),
                torch.empty(0).to(self.device).long(),
            )
            failure_count = 0
            while failure_count < max_search_times:
                random_noise = torch.zeros_like(x_original).uniform_(
                    self.vmin, self.vmax
                )
                random_noise_pred = self.get_decision(random_noise)
                failure_count += 1

                # append successful ones to queue, filter on unsuccessful ones for next round
                success_indicators = random_noise_pred != y[idx_tensor]
                x0_inserted = torch.concat(
                    (x0_inserted, random_noise[success_indicators]), dim=0
                )
                insert_idx = torch.hstack((insert_idx, idx_tensor[success_indicators]))

                x_original = x_original[~success_indicators]
                idx_tensor = idx_tensor[~success_indicators]

                if len(idx_tensor) == 0:
                    break

            # if still not able to find a random new sample, use old sample of a different class
            if len(idx_tensor) > 0:
                warnings.warn(
                    "random uniform init failed, take one correct sample as adversarial sample"
                )
                append_count = len(idx_tensor)
                for idx, cur_y in zip(idx_tensor, y[idx_tensor]):
                    rand_perm_idx = torch.randperm(self.x_samples.shape[0]).to(
                        self.device
                    )
                    for noise_idx in rand_perm_idx:
                        random_noise_candidate = self.x_samples[[noise_idx]]
                        random_noise_candidate_pred = self.get_decision(
                            random_noise_candidate
                        )
                        if cur_y != random_noise_candidate_pred.item():
                            x0_inserted = torch.concat(
                                (x0_inserted, random_noise_candidate), dim=0
                            )
                            insert_idx = torch.hstack((insert_idx, idx))
                            append_count -= 1
                            break
                if append_count > 0:
                    raise Exception(
                        "model predict the same label for all memorized input, program stopped!"
                    )

            # keep insertion order
            random_noises = torch.zeros_like(x)
            random_noises[insert_idx] = x0_inserted

            # project random noise to decision boundary using binary search
            x0 = self.binary_search(x, random_noises, y, y_target, self.tol)
        else:
            x0 = self.binary_search(x, x_targets, y, y_target, self.tol)

        return x0

    def _estimate_normal_direction(
        self,
        x_bounds: torch.Tensor,
        y: torch.LongTensor,
        num_evals: int,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """
        use random normal to estimate the local normal direction of adversarial region

        :param x_bounds: samples at the boundary
        :param y: the original predictions
        :param num_evals: we normalize the scale by iteration
        :param delta: the scale of perturbation
        :return normal directions pointing to the adversarial region
        """
        deltas = delta.reshape(self.batch_size, *([1] * (len(self.shapes) + 1)))
        # l2-norm perturbation
        rand_perturbation = (
            torch.randn(
                *[self.batch_size, num_evals, *self.shapes], device=x_bounds.device
            )
            * deltas
        )
        perturbed = x_bounds.unsqueeze(1) + rand_perturbation
        perturbed = torch.clamp(perturbed, self.vmin, self.vmax)
        rv = (perturbed - x_bounds.unsqueeze(1)) / deltas
        perturbed_prediction = self.get_decision(
            perturbed.reshape(
                -1, *self.shapes
            )  # bind batch and num_evals dimension together
        ).reshape(
            self.batch_size, num_evals, 1
        )  # convert back to shape prior to binding
        label_changed = (perturbed_prediction != y[:, None, None]).float()

        # label_changed is of shape (batch_size, num_eval, 1)
        pre_weights = label_changed.reshape(self.batch_size, num_evals) * 2 - 1
        weights = pre_weights - pre_weights.mean(dim=-1, keepdim=True)
        weights -= (
            (pre_weights == -1).all(dim=-1, keepdim=True).float()
        )  # if all labels unchanged, use the other side
        weights += (
            (pre_weights == 1).all(dim=-1, keepdim=True).float()
        )  # if all labels changed, just use its mean

        # get normal direction
        normal_directions = (
            weights.reshape(self.batch_size, num_evals, *([1] * len(self.shapes))) * rv
        ).mean(
            dim=1
        )  # collapse num_eval dimension

        # normalize (treat each image as a flattened vector)
        normal_vecs = normal_directions.flatten(start_dim=1)
        normal_vecs = normal_vecs / normal_vecs.norm(dim=-1, keepdim=True)
        # reshape back
        normal_vecs = normal_vecs.reshape(*x_bounds.shape)
        return normal_vecs

    def get_tangent_point(
        self,
        x: torch.Tensor,
        ball_centers: torch.Tensor,
        radius: torch.Tensor,
        normal_vecs: torch.Tensor,
    ) -> torch.Tensor:
        """
        a closed-form solution to the tangent point from ball_center to the query sample, measured in l2-norm

        :param x: original data sample
        :param ball_center: the current iterate at the adversarial boundary
        :param radius: the radius of the hemisphere
        :param normal_vec: the normal vector pointing into the adversarial direction
        :return the tangent point
        """
        # * all vectors below are batched row vectors
        x = x.flatten(start_dim=1)
        ball_centers = ball_centers.flatten(start_dim=1)
        normal_vecs = normal_vecs.flatten(start_dim=1)

        # prepare
        ox = x - ball_centers
        sin_alpha = -(ox * normal_vecs).sum(dim=-1, keepdim=True) / (
            torch.norm(ox, dim=-1, p=2, keepdim=True)
            * torch.norm(normal_vecs, dim=-1, p=2, keepdim=True)
        )
        cos_alpha = torch.sqrt(
            (1 - torch.square(sin_alpha)).clamp(min=0)
        )  # numerical fix
        cos_beta = radius / torch.norm(ox, dim=-1, p=2, keepdim=True)
        sin_beta = torch.sqrt(
            (1 - torch.square(cos_beta)).clamp(min=0)
        )  # numerical fix
        sin_gamma = sin_beta * cos_alpha - cos_beta * sin_alpha
        cos_gamma = sin_beta * cos_alpha + cos_beta * sin_alpha
        k_height = radius * sin_gamma

        # get tangent point
        numerator = (
            ox
            - (ox * normal_vecs).sum(dim=-1, keepdim=True)
            * normal_vecs
            / torch.norm(normal_vecs, dim=-1, keepdim=True) ** 2
        )
        ok_prime = (
            (numerator / torch.norm(numerator, dim=-1, keepdim=True, p=2))
            * radius
            * cos_gamma
        )
        ok = ok_prime + k_height * normal_vecs / torch.norm(
            normal_vecs, dim=-1, keepdim=True
        )

        # ? perturbed vectors are all dense ?
        result = ok + ball_centers

        # convert back to original shape
        result = result.reshape(x.shape[0], *self.shapes)
        return result

    def _geometric_progression_for_tangent_point(
        self,
        x: torch.Tensor,
        x_bound: torch.Tensor,
        y: torch.Tensor,
        y_target: torch.Tensor,
        normal_vec: torch.Tensor,
        cur_iter: int,
    ) -> torch.Tensor:
        """
        shrink search radius by half at each time to find a valid
        tangent point (guarantee that tangent point belongs to the adversarial class)

        :param x: current data sample
        :param x_bound: the adversarial sample
        :param y: the original predictions
        :param y_target: the desired adversarial target class
        :param normal_vec: the normal vector pointing to the adversarial region
        :param cur_iter: the current iteration, for normalizing the starting radius
        :return a valid tangent point
        """
        # normalize search radius in proportional to the distance between the current iterate
        # and the original data sample
        radius = (x - x_bound).reshape(self.batch_size, -1).norm(
            p=2, dim=-1, keepdim=True
        ) / cur_iter ** (1 / 2)

        # sequential passing in: keep shrinking radius if not yet success, otherwise record
        idx_tensor = torch.arange(self.batch_size, device=self.device)
        cur_x, cur_x_bound, cur_normal_vec, cur_radius = x, x_bound, normal_vec, radius
        idx_insert = torch.empty(0).to(self.device).long()
        tangent_point_insert = torch.empty(0, *self.shapes).to(self.device)

        while True:
            # get a tangent point candidate
            tangent_point = self.get_tangent_point(
                cur_x, cur_x_bound, cur_radius, cur_normal_vec
            )
            tangent_point = tangent_point.clamp(min=self.vmin, max=self.vmax)

            # verify if it belongs to an adversarial class
            success_indicators = self.check_attack_success(
                tangent_point,
                y[idx_tensor],
                y_target[idx_tensor] if y_target is not None else None,
            )

            # find unsuccessful ones and filter out successes
            idx_insert = torch.hstack((idx_insert, idx_tensor[success_indicators]))
            tangent_point_insert = torch.vstack(
                (tangent_point_insert, tangent_point[success_indicators])
            )

            # filter
            idx_tensor = idx_tensor[~success_indicators]
            cur_x = cur_x[~success_indicators]
            cur_x_bound = cur_x_bound[~success_indicators]
            cur_normal_vec = cur_normal_vec[~success_indicators]
            cur_radius = cur_radius[~success_indicators]

            if len(idx_tensor) == 0:
                break
            else:
                cur_radius /= 2

            # ! HARD STOP: not in original implementation
            # * if no tangent point found due to numerical issues, just bump
            # * the boundary a bit away from the sample towards the adversarial region
            if torch.all(cur_radius < 1e-8):
                adv_direction = cur_x_bound - cur_x
                adv_direction_flatten = adv_direction.flatten(start_dim=1)
                adv_direction_unit = (
                    adv_direction_flatten
                    / adv_direction_flatten.norm(dim=-1, keepdim=True)
                ).reshape(*adv_direction.shape)
                tangent_point = cur_x_bound + 1e-4 * adv_direction_unit

                # append final
                idx_insert = torch.hstack((idx_insert, idx_tensor))
                tangent_point_insert = torch.vstack(
                    (tangent_point_insert, tangent_point)
                )

                warnings.warn("tangent point failed, hard stop bumping triggered")
                break

            # ! HARD STOP: if all nan due to numerical issues
            if torch.isnan(cur_radius).all():
                # append final
                idx_insert = torch.hstack((idx_insert, idx_tensor))
                tangent_point_insert = torch.vstack((tangent_point_insert, cur_x_bound))

                warnings.warn(
                    "adversarial samples too close to the boundary, causing numerical issues; return the boundary point"
                )
                break

        # keep insertion order
        tangent_points = torch.zeros_like(x)
        tangent_points[idx_insert] = tangent_point_insert

        return tangent_points

    def _select_delta(self, i: int, dists: torch.Tensor) -> torch.Tensor:
        """
        pick an appropriate distance scalar for estimating the normal direction
        :param i: the current iteration to scale with
        :param dist: the current distance between the perturbed sample and original sample
        """
        if i == 1:
            return (
                0.1
                * (self.vmax - self.vmin)
                * torch.ones((self.batch_size, 1)).to(self.device)
            )
        else:
            return dists * self.gamma / self.dim

    def attack(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """launch adversarial attack against a specific input"""
        dists = torch.empty(0)  # on cpu
        adv_samples = torch.empty(0, *self.shapes)
        for x, x_target in tqdm(self.adversarial_dataloader):
            # keep first dimension the batch dimension
            x = x.to(self.device)
            if x_target.shape[-1] > 0:  # targetd:
                x_target = x_target.to(self.device)
            else:  # untargeted
                x_target = None

            # initialize
            y = self.get_decision(x)
            y_target = None if x_target is None else self.get_decision(x_target)

            cur = self._initialize(x, x_target, y, y_target)
            cur_dists = (
                (cur - x).view(self.batch_size, -1).norm(p=2, dim=-1, keepdim=True)
            )
            for i in range(1, self.T + 1):
                # get delta
                deltas = self._select_delta(i, cur_dists)

                # get number of iterations to evaluate
                num_evals = min(
                    int(i ** (1 / 2) * self.init_num_eval), self.max_num_evals
                )

                # estimate normal vector
                normal_vecs = self._estimate_normal_direction(cur, y, num_evals, deltas)

                # find valid tangent point
                tangent_point = self._geometric_progression_for_tangent_point(
                    x, cur, y, y_target, normal_vecs, cur_iter=i
                )

                # binary search back to the boundary
                cur = self.binary_search(x, tangent_point, y, y_target, self.tol)

                # evaluate current distance
                cur_dists = (
                    (cur - x).view(self.batch_size, -1).norm(p=2, dim=-1, keepdim=True)
                )
                logging.info(
                    f"Iteration {i}: average distance = {cur_dists.mean():.6f}"
                )

            # append to dists
            dists = torch.hstack((dists, cur_dists.detach().cpu().flatten()))
            adv_samples = torch.vstack((adv_samples, cur))
        
        return dists, adv_samples
