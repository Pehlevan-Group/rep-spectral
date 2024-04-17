"""
collection of real dataset
- MNIST
- FashionMNIST
- CIFAR10
- CIFAR100
- Imangenette
"""

# load packages
import torch
import torchvision
import torchvision.transforms as transforms


# ====== datasets ========
def mnist(path: str = None, flatten=True):
    """load mnist dataset"""
    from torchvision.datasets import MNIST

    if flatten:
        train_set = MNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), lambda t: t.flatten()]
            ),
        )
        test_set = MNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), lambda t: t.flatten()]
            ),
        )
    else:
        train_set = MNIST(path, train=True, download=True)
        test_set = MNIST(path, train=False, download=True)
    return train_set, test_set


def fashion_mnist(path: str = None, flatten=True):
    """load fashion mnist data"""
    from torchvision.datasets import FashionMNIST

    if flatten:
        train_set = FashionMNIST(
            path,
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), lambda t: t.flatten()]
            ),
        )
        test_set = FashionMNIST(
            path,
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), lambda t: t.flatten()]
            ),
        )
    else:
        train_set = FashionMNIST(path, train=True, download=True)
        test_set = FashionMNIST(path, train=False, download=True)
    return train_set, test_set


# =================== CIFAR 10 ===================
def cifar10(data_path):
    """load (download if necessary) cifar10"""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    return trainset, testset


def cifar10_resized(data_path):
    """load (download if necessary) cifar10"""
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )

    return trainset, testset


def cifar10_clean(data_path):
    """load (download if necessary) unpreprocessed cifar10 data"""
    img_transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=data_path, train=True, download=True, transform=img_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=img_transform
    )

    return trainset, testset


def get_cifar_class_names():
    """return the name of the cifar 10 classes"""
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )
    return classes


def random_samples_by_targets(dataset, targets=[7, 6], seed=42):
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


# =================== CIFAR 100 ===================
def cifar100(data_path):
    """load (download if necessary) cifar100"""
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR100(
        root=data_path, train=True, download=True, transform=transform_train
    )

    testset = torchvision.datasets.CIFAR100(
        root=data_path, train=False, download=True, transform=transform_test
    )

    return trainset, testset


# =================== imagenette ===================
def imagenette(data_path):
    """load (download if necessary) imagenette"""
    # specify transformations
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    transform_test = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalization]
    )

    # get dataset
    trainset = torchvision.datasets.Imagenette(
        root=data_path, split="train", download=True, transform=transform_train
    )

    testset = torchvision.datasets.Imagenette(
        root=data_path, split="val", download=True, transform=transform_test
    )

    return trainset, testset
