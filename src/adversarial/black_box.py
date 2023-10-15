"""
Implementation of black-box attackers

- TangentAttack: https://openreview.net/forum?id=g0wang64Zjd. 
  Implementation modified from https://github.com/machanic/TangentAttack
"""

# load packages
import logging
import numpy as np
import torch
import torch.nn as nn


class Attacker:
    """parent class of collections of attackers"""
    def __init__(self) -> None:
        pass
    
    def binary_search(self, x_adv: torch.Tensor, tol: float=1e-5) -> torch.Tensor:
        """
        binary search to find a sample at the decision boundary
        stop after a prespecified tolerance is reached
        
        :param x_adv: the adversarial sample
        :param tol: a tolerance level (for l2-norm measure)
        :return a sample at the adversarial boundary
        """
        y_adv = self.model(x_adv).argmax(dim=-1).item()
        l, r = 0, 1
        while r - l > tol:
            mid = (l + r) / 2
            x_mid = (1 - mid) * self.x + mid * x_adv
            y_mid = self.model(x_mid).argmax(dim=-1).item()
            if y_mid == y_adv:
                r = mid
            else:
                l = mid
        
        result = (1 - r) * self.x + r * x_adv
        return result


class TangentAttack(Attacker):
    """
    Tangent Attacker from https://openreview.net/forum?id=g0wang64Zjd
    we use the hemisphere implementation with l2 distance measure
    """

    def __init__(self,
            model: nn.Module, 
            x: torch.Tensor,
            x_target: torch.Tensor,
            tol: float=1e-5,
            gamma: float=1.,
            vmin: float=0.,
            vmax: float=1.,
            T: int=40,
            init_num_eval: int=100,
            max_num_evals: int=10000
        ) -> None:
        """
        # TODO
        """
        super().__init__()
        self.model = model
        self.x = x 
        self.dim = np.prod([*x.shape[1:]])
        self.y = self.model(self.x).argmax(dim=-1).item() # predicted label

        # adversarial labels
        self.x_target = x_target
        self.y_target = None if x_target is None else self.model(x_target).argmax(dim=-1).item()

        # feasible range of input
        self.vmin = vmin 
        self.vmax = vmax
        
        # parameters
        self.tol = tol
        self.T = T 
        self.gamma = gamma
        self.init_num_eval = init_num_eval
        self.max_num_evals = max_num_evals


    def initialize(self, x: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor: 
        """
        initialize x0 to start tangent attack: uniformly choose images from within feasible range, 
        and then binary search as a starting point
        """
        # non targeted attack
        if x_target is None: 
            while True:
                random_noise = torch.zeros_like(self.x).uniform_(self.vmin, self.vmax)
                random_noise_pred = self.model(random_noise).argmax(dim=-1).item()
                if random_noise_pred != self.y: 
                    # assign adversarial target label
                    self.y_target = random_noise_pred
                    break
            
            # project random noise to decisin boundary using binary search 
            x0 = self.binary_search(random_noise, self.tol)
        else:
            x0 = self.binary_search(x_target, self.tol)
        
        return x0

    def estimate_normal_direction(self, x_bound: torch.Tensor, num_evals: int, delta: float) -> torch.Tensor:
        """
        use random normal to estimate the local normal direction of adversarial region

        :param x_bound: samples at the boundary
        :param num_evals: we normalie the scale by iteration
        :param delta: the scale of perturbation
        :return normal directions pointing to the adversarial region
        """
        # l2-norm perturbation
        rand_perturbation = torch.randn(*[num_evals, *x_bound.shape]) * delta
        perturbed = x_bound + rand_perturbation
        perturbed_prediction = self.model(perturbed).argmax(dim=-1)
        label_changed = (perturbed_prediction == self.y).float()

        # get normal direction
        if label_changed.mean() == 0: # all labels changed
            normal_direction = perturbed.mean(dim=0)
        elif label_changed.mean() == 1: # all labels unchanged
            normal_direction = -perturbed.mean(dim=0)
        else:
            normal_direction = ((label_changed - label_changed.mean()) * perturbed).mean(dim=0)
        
        # normalize
        normal_vec = normal_direction / torch.linalg.norm(normal_direction.flatten(start_dim=1), dim=-1)
        return normal_vec

    def get_tangent_point(self, x: torch.Tensor, ball_center: torch.Tensor, radius: float, normal_vec: torch.Tensor) -> torch.Tensor:
        """
        a closed-form solution to the tangent point from ball_center to the query sample, measured in l2-norm
        """
        # prepare
        ox = x - ball_center
        sin_alpha = - torch.dot(ox, normal_vec) / (torch.norm(ox, p=2) * torch.norm(normal_vec, p=2))
        cos_alpha = torch.sqrt(1- torch.square(sin_alpha()))
        cos_beta = radius / torch.norm(ox, p=2)
        sin_beta = torch.sqrt(1 - torch.square(cos_beta))
        sin_gamma = sin_beta * cos_alpha - cos_beta * sin_alpha
        cos_gamma = sin_beta * cos_alpha + cos_beta * sin_alpha
        k_height = radius * sin_gamma

        # get tangent point
        numerator = ox - torch.dot(ox, normal_vec) * normal_vec / torch.norm(normal_vec) ** 2
        ok_prime = (numerator / torch.norm(numerator, p=2)) * radius * cos_gamma
        ok = ok_prime + k_height * normal_vec / torch.norm(normal_vec)
        
        result = ok + ball_center
        return result

    def geometric_progression_for_tangent_point(
            self, x: torch.Tensor, 
            x_bound: torch.Tensor, 
            normal_vec: torch.Tensor, 
            target_label: int,
            cur_iter: int
        ) -> torch.Tensor:
        """
        shrink search radius by half at each time to find a valid 
        tangent point (guarantee that tangent point belongs to the adversarial class)
        
        :param x: current data sample
        :param x_bound: the adversarial sample
        :param normal_vec: the normal vector pointing to the adversarial region
        :param target_label: the prediction of the adversarial region
        :param cur_iter: the current iteration, for normalizing the starting radius
        :return a valid tagent point
        """
        # normalize search radius in proportional to the distance between the current iterate
        # and the original data sample
        radius = (x - x_bound).norm(p=2) / cur_iter ** (1/2)
        while True:
            # get a tangent point candidate
            tangent_point = self.get_tangent_point(x, x_bound, radius, normal_vec)

            # verify if it belongs to an adversarial class
            tagent_point_pred = self.model(tangent_point).argmax(dim=-1).item()
            if tagent_point_pred == target_label:
                break
            else:
                radius /= 2
        
        # ensure valid range
        tangent_point = tangent_point.clamp(min=self.vmin, max=self.vmax)
        return tangent_point

    def _select_delta(self, i: int, dist: float) -> float: 
        """
        pick an appropriate distance scalar for estimating the normal direction
        :param i: the current iteration to scale with
        :param dist: the current distance between the perturbed sample and original sample
        """
        if i == 1: return 0.1 * (self.vmax - self.vmin)
        else: return dist * self.gamma / self.dim


    def attack(self, x_target: torch.Tensor) -> torch.Tensor:
        """
        launch adversarial attack against a specific input

        :param x: the origin to perturbe
        """
        # initialize
        cur = self.initialize(self.x, x_target)
        cur_dist = (cur - self.x).norm(p=2)
        for i in range(1, self.T + 1):
            # get delta 
            delta = self._select_delta(i, cur_dist)
            
            # get number of iterations to evaluate
            num_evals = min(int(i ** (1/2) * self.init_num_eval), self.max_num_evals)

            # estimate normal vector
            normal_vec = self.estimate_normal_direction(cur, num_evals, delta)

            # find valid tangent point
            tangent_point = self.geometric_progression_for_tangent_point(self.x, cur, normal_vec, target_label=self.y_target, cur_iter=i)

            # binary search back to the boundary
            cur = self.binary_search(tangent_point, self.tol)
        
            # evaluate current distance
            cur_dist = (cur - self.x).norm(p=2).item()
            logging.info(f"Iteration {i}: distance = {cur_dist:.6f}")

        return cur

# TODO: 1. verify dimension along the way; 2. write documentations; 3. GPU memory profiling
