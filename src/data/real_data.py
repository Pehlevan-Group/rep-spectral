"""
collection of real dataset
- MNIST
- FashionMNIST
- CIFAR10
- CIFAR100
- Imangenette
- ImageNet1K
"""

# load packages
import os
import math
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


# ================= ImageNet ==================
def imagenet1k(data_path):
    """
    load imagenet1k

    transformation adapted from FixRes: https://github.com/facebookresearch/FixRes
    """
    normalization = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    jittering = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4)
    lighting = Lighting(
        alphastd=0.1,
        eigval=[0.2175, 0.0188, 0.0045],
        eigvec=[
            [-0.5675, 0.7192, 0.4009],
            [-0.5808, -0.0045, -0.8140],
            [-0.5836, -0.6948, 0.4203],
        ],
    )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            jittering,
            lighting,
            normalization,
        ]
    )

    transform_val = transforms.Compose(
        [
            transforms.Resize(
                int((256 / 224) * 224)
            ),  # to maintain same ratio w.r.t. 224 images
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization,
        ]
    )

    trainset = torchvision.datasets.ImageNet(
        data_path, "train", transform=transform_train
    )
    valset = torchvision.datasets.ImageNet(data_path, "val", transform=transform_val)
    return trainset, valset

class Lighting:
    """Lighting noise(AlexNet - style PCA - based noise)"""

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))
class RASampler(torch.utils.data.Sampler):
    """
    Batch Sampler with Repeated Augmentations (RA), adapted from FixRes
    - dataset_len: original length of the dataset
    - batch_size
    - repetitions: instances per image
    - len_factor: multiplicative factor for epoch size
    # TODO: document clearly
    """

    def __init__(
        self,
        dataset,
        num_replicas,
        rank,
        dataset_len,
        batch_size,
        repetitions=1,
        len_factor=1.0,
        shuffle=False,
        drop_last=False,
    ):
        self.dataset = dataset
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * self.repetitions * 1.0 / self.num_replicas)
        )
        self.total_size = self.num_samples * self.num_replicas

    def shuffler(self):
        if self.shuffle:
            new_perm = lambda: iter(torch.randperm(self.dataset_len))
        else:
            new_perm = lambda: iter(torch.arange(self.dataset_len))
        shuffle = new_perm()
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            for repetition in range(self.repetitions):
                yield index

    def __iter__(self):
        shuffle = iter(self.shuffler())
        seen = 0
        indices = []
        for _ in range(self.len_images):
            index = next(shuffle)
            indices.append(index)
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
