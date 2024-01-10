"""
for adversarial data IO
"""

# load packages
import torch
from torch.utils.data import Dataset

class CustomAdversarialDataset(Dataset):
    """wrap adversarial data IO to a dataset"""
    def __init__(self, samples: torch.Tensor, target_samples: torch.Tensor) -> None:
        super().__init__()
        self.samples = samples
        self.target_samples = target_samples
    
    def __getitem__(self, index):
        # untargeted, return empty
        if self.target_samples is None:
            return self.samples[index], torch.empty(0)
        
        # targeted, draw a random sample
        else:
            target_sample = self.target_samples[torch.randperm(len(self.target_samples))[[1]]]
            return self.samples[index], target_sample

    def __len__(self):
        return len(self.samples)
