# load packages
from typing import Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset


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


class CustomScan(Dataset):
    """wrap x to a torch dataset"""

    def __init__(self, X: torch.Tensor):
        super().__init__()
        self.x = X

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index) -> torch.Tensor:
        return self.x[index]


def random_samples_by_targets(
    dataset: Dataset, targets: List[int] = [7, 6], seed: int = 0
) -> List[torch.Tensor]:
    """
    randomly sample len(target) many instances with corresponding targets

    :param dataset: the PyTorch dataset
    :param target: the y value of samples
    :param seed: for reproducibility
    """
    torch.manual_seed(seed)
    total_num_samples = len(dataset)
    samples = []
    for cur_target in targets:
        target = None
        while cur_target != target:
            random_index = torch.randint(total_num_samples, (1,)).item()
            data, target = dataset[random_index]
        samples.append(data)
    return samples


def rand_bbox(size: torch.Size, lam: torch.Tensor) -> Tuple[int]:
    """
    get a random bounding box regions
    modified from https://github.com/clovaai/CutMix-PyTorch
    """
    W = size[2]
    H = size[3]
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = int((W * cut_rat).item())
    cut_h = int((H * cut_rat).item())

    # uniform
    cx = torch.randint(0, W, (1,)).item()
    cy = torch.randint(0, H, (1,)).item()

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
