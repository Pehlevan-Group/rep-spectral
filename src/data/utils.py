# load packages
from typing import Tuple, List
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
            random_index = torch.randint(total_num_samples, (1,))
            data, target = dataset[random_index]
        samples.append(data)
    return samples
