"""
collection of real dataset
"""

# load packages
import torch
import torchvision.transforms as transforms

# ======= set of transformations =======


# ====== datasets ========
def mnist(path: str=None, flatten=True):
    """load mnist dataset"""
    from torchvision.datasets import MNIST
    if flatten:
        train_set = MNIST(path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda t: t.flatten()]))
        test_set = MNIST(path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda t: t.flatten()]))
    else:
        train_set = MNIST(path, train=True, download=True)
        test_set = MNIST(path, train=False, download=True)
    return train_set, test_set

def fashion_mnist(path: str=None, flatten=True):
    """load fashion mnist data"""
    from torchvision.datasets import FashionMNIST
    if flatten:
        train_set = FashionMNIST(path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda t: t.flatten()]))
        test_set = FashionMNIST(path, train=False, download=True, transform=transforms.Compose([transforms.ToTensor(), lambda t: t.flatten()]))
    else:
        train_set = FashionMNIST(path, train=True, download=True)
        test_set = FashionMNIST(path, train=False, download=True)
    return train_set, test_set
