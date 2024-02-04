"""
collection of transfer data
- Stanford Dog
- Oxford Flowers
- MIT Indoor Scene Recognition

Many are modified from https://github.com/zrsmithson/Stanford-dogs/blob/master/data/stanford_dogs_data.py
"""

# load packages
from typing import List
from PIL import Image, ImageFile
from os.path import join
import os
import scipy.io

import torch.utils.data as data
from torchvision.datasets.utils import download_url, list_dir
from torchvision.transforms import transforms


# =================== Stanford Dog ===================
class dogs(data.Dataset):
    """
    `Stanford Dogs <http://vision.stanford.edu/aditya86/ImageNetDogs/>`_ Dataset.

    root (string): Root directory of dataset where directory
        ``omniglot-py`` exists.
    cropped (bool, optional): If true, the images will be cropped into the bounding box specified
        in the annotations
    transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    download (bool, optional): If true, downloads the dataset tar files from the internet and
        puts it in root directory. If the tar files are already downloaded, they are not
        downloaded again.
    """

    folder = "StanfordDogs"
    download_url_prefix = "http://vision.stanford.edu/aditya86/ImageNetDogs"

    def __init__(
        self,
        root,
        train=True,
        cropped=False,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.cropped = cropped
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        split = self.load_split()

        self.images_folder = join(self.root, "Images")
        self.annotations_folder = join(self.root, "Annotation")
        self._breeds = list_dir(self.images_folder)

        if self.cropped:
            self._breed_annotations = [
                [
                    (annotation, box, idx)
                    for box in self.get_boxes(join(self.annotations_folder, annotation))
                ]
                for annotation, idx in split
            ]
            self._flat_breed_annotations = sum(self._breed_annotations, [])

            self._flat_breed_images = [
                (annotation + ".jpg", idx)
                for annotation, box, idx in self._flat_breed_annotations
            ]
        else:
            self._breed_images = [
                (annotation + ".jpg", idx) for annotation, idx in split
            ]

            self._flat_breed_images = self._breed_images

        self.classes = [
            "Chihuaha",
            "Japanese Spaniel",
            "Maltese Dog",
            "Pekinese",
            "Shih-Tzu",
            "Blenheim Spaniel",
            "Papillon",
            "Toy Terrier",
            "Rhodesian Ridgeback",
            "Afghan Hound",
            "Basset Hound",
            "Beagle",
            "Bloodhound",
            "Bluetick",
            "Black-and-tan Coonhound",
            "Walker Hound",
            "English Foxhound",
            "Redbone",
            "Borzoi",
            "Irish Wolfhound",
            "Italian Greyhound",
            "Whippet",
            "Ibizian Hound",
            "Norwegian Elkhound",
            "Otterhound",
            "Saluki",
            "Scottish Deerhound",
            "Weimaraner",
            "Staffordshire Bullterrier",
            "American Staffordshire Terrier",
            "Bedlington Terrier",
            "Border Terrier",
            "Kerry Blue Terrier",
            "Irish Terrier",
            "Norfolk Terrier",
            "Norwich Terrier",
            "Yorkshire Terrier",
            "Wirehaired Fox Terrier",
            "Lakeland Terrier",
            "Sealyham Terrier",
            "Airedale",
            "Cairn",
            "Australian Terrier",
            "Dandi Dinmont",
            "Boston Bull",
            "Miniature Schnauzer",
            "Giant Schnauzer",
            "Standard Schnauzer",
            "Scotch Terrier",
            "Tibetan Terrier",
            "Silky Terrier",
            "Soft-coated Wheaten Terrier",
            "West Highland White Terrier",
            "Lhasa",
            "Flat-coated Retriever",
            "Curly-coater Retriever",
            "Golden Retriever",
            "Labrador Retriever",
            "Chesapeake Bay Retriever",
            "German Short-haired Pointer",
            "Vizsla",
            "English Setter",
            "Irish Setter",
            "Gordon Setter",
            "Brittany",
            "Clumber",
            "English Springer Spaniel",
            "Welsh Springer Spaniel",
            "Cocker Spaniel",
            "Sussex Spaniel",
            "Irish Water Spaniel",
            "Kuvasz",
            "Schipperke",
            "Groenendael",
            "Malinois",
            "Briard",
            "Kelpie",
            "Komondor",
            "Old English Sheepdog",
            "Shetland Sheepdog",
            "Collie",
            "Border Collie",
            "Bouvier des Flandres",
            "Rottweiler",
            "German Shepard",
            "Doberman",
            "Miniature Pinscher",
            "Greater Swiss Mountain Dog",
            "Bernese Mountain Dog",
            "Appenzeller",
            "EntleBucher",
            "Boxer",
            "Bull Mastiff",
            "Tibetan Mastiff",
            "French Bulldog",
            "Great Dane",
            "Saint Bernard",
            "Eskimo Dog",
            "Malamute",
            "Siberian Husky",
            "Affenpinscher",
            "Basenji",
            "Pug",
            "Leonberg",
            "Newfoundland",
            "Great Pyrenees",
            "Samoyed",
            "Pomeranian",
            "Chow",
            "Keeshond",
            "Brabancon Griffon",
            "Pembroke",
            "Cardigan",
            "Toy Poodle",
            "Miniature Poodle",
            "Standard Poodle",
            "Mexican Hairless",
            "Dingo",
            "Dhole",
            "African Hunting Dog",
        ]

    def __len__(self):
        return len(self._flat_breed_images)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        image_name, target_class = self._flat_breed_images[index]
        image_path = join(self.images_folder, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.cropped:
            image = image.crop(self._flat_breed_annotations[index][1])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if os.path.exists(join(self.root, "Images")) and os.path.exists(
            join(self.root, "Annotation")
        ):
            if (
                len(os.listdir(join(self.root, "Images")))
                == len(os.listdir(join(self.root, "Annotation")))
                == 120
            ):
                print("Files already downloaded and verified")
                return

        for filename in ["images", "annotation", "lists"]:
            tar_filename = filename + ".tar"
            url = self.download_url_prefix + "/" + tar_filename
            download_url(url, self.root, tar_filename, None)
            print("Extracting downloaded file: " + join(self.root, tar_filename))
            with tarfile.open(join(self.root, tar_filename), "r") as tar_file:
                tar_file.extractall(self.root)
            os.remove(join(self.root, tar_filename))

    def load_split(self):
        if self.train:
            split = scipy.io.loadmat(join(self.root, "train_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "train_list.mat"))["labels"]
        else:
            split = scipy.io.loadmat(join(self.root, "test_list.mat"))[
                "annotation_list"
            ]
            labels = scipy.io.loadmat(join(self.root, "test_list.mat"))["labels"]

        split = [item[0][0] for item in split]
        labels = [item[0] - 1 for item in labels]
        return list(zip(split, labels))


# ====================== Oxford Flowers =======================
class flowers(data.Dataset):
    """
    Oxford flower
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        cropped (bool, optional): If true, the images will be cropped into the bounding box specified
            in the annotations
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset tar files from the internet and
            puts it in root directory. If the tar files are already downloaded, they are not
            downloaded again.
    """

    folder = "OxfordFlowers"
    download_url_prefix = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102"

    def __init__(
        self,
        root,
        train=True,
        val=False,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.val = val
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()

        self.split = self.load_split()

        self.images_folder = join(self.root, "jpg")

    def __len__(self):
        return len(self.split)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target character class.
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name, target_class = self.split[index]
        image_path = join(self.images_folder, "image_%05d.jpg" % (image_name + 1))
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        import tarfile

        if (
            os.path.exists(join(self.root, "jpg"))
            and os.path.exists(join(self.root, "imagelabels.mat"))
            and os.path.exists(join(self.root, "setid.mat"))
        ):
            if len(os.listdir(join(self.root, "jpg"))) == 8189:
                print("Files already downloaded and verified")
                return

        filename = "102flowers"
        tar_filename = filename + ".tgz"
        url = self.download_url_prefix + "/" + tar_filename
        download_url(url, self.root, tar_filename, None)
        with tarfile.open(join(self.root, tar_filename), "r") as tar_file:
            tar_file.extractall(self.root)
        os.remove(join(self.root, tar_filename))

        filename = "imagelabels.mat"
        url = self.download_url_prefix + "/" + filename
        download_url(url, self.root, filename, None)

        filename = "setid.mat"
        url = self.download_url_prefix + "/" + filename
        download_url(url, self.root, filename, None)

    def load_split(self):
        split = scipy.io.loadmat(join(self.root, "setid.mat"))
        labels = scipy.io.loadmat(join(self.root, "imagelabels.mat"))["labels"]
        if self.train:
            split = split["trnid"]
        elif self.val:
            split = split["valid"]
        else:
            split = split["tstid"]

        # set it all back 1 as img indexs start at 1
        split = list(split[0] - 1)
        labels = list(labels[0][split] - 1)
        return list(zip(split, labels))


# =================== MIT Indoor ======================
class indoor(data.Dataset):
    """
    MIT indoor scene classification dataset
    """

    folder = "MITIndoor"
    download_url_prefix = "http://groups.csail.mit.edu/vision/LabelMe/NewImages"

    def __init__(
        self,
        root,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = join(os.path.expanduser(root), self.folder)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.class_name_map = {}  # to be populated

        if download:
            self.download()

        self.split = self.load_split()

    def __len__(self) -> int:
        return len(self.split)

    def __getitem__(self, idx):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        image_name = self.split[idx]
        image_path = join(self.root, "images", image_name)
        image = Image.open(image_path).convert("RGB")
        target_class = self.class_name_map[image_name.split("/")[0]]

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target_class = self.target_transform(target_class)

        return image, target_class

    def download(self):
        """download dataset"""
        import tarfile

        if (
            os.path.exists(join(self.root, "images"))
            and os.path.exists(join(self.root, "TrainImages.txt"))
            and os.path.exists(join(self.root, "TestImages.txt"))
        ):
            if len(os.listdir(join(self.root, "images"))) == 67:
                print("Files already downloaded and verified")
                # populate class
                class_names = sorted(os.listdir(join(self.root, "images")))
                self.class_name_map = dict(
                    zip(class_names, list(range(len(class_names))))
                )
                return

        # download raw file
        tar_filename = "indoorCVPR_09.tar"
        url = self.download_url_prefix + "/" + tar_filename
        download_url(url, self.root, tar_filename, None)
        with tarfile.open(join(self.root, tar_filename), "r") as tar_file:
            tar_file.extractall(self.root)
        os.remove(join(self.root, tar_filename))

        # populate class
        class_names = sorted(os.listdir(join(self.root, "images")))
        self.class_name_map = dict(zip(class_names, list(range(len(class_names)))))

        # download splits
        train_img_list_url = "https://web.mit.edu/torralba/www/TrainImages.txt"
        test_img_list_url = "https://web.mit.edu/torralba/www/TestImages.txt"
        download_url(train_img_list_url, self.root, "TrainImages.txt")
        download_url(test_img_list_url, self.root, "TestImages.txt")

    def load_split(self) -> List[str]:
        """get train test split"""
        if self.train:
            txt_file_path = join(self.root, "TrainImages.txt")
        else:
            txt_file_path = join(self.root, "TestImages.txt")

        # load image paths
        with open(txt_file_path, "r") as f:
            img_names = f.read().split("\n")
        return img_names


# ----------------------------------------
# ========== load dataset ================
# ----------------------------------------

NORMALIZATION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
TRAIN_TRANSFORMATION = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, ratio=(1, 1.3)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        NORMALIZATION,
    ]
)
TEST_TRANSFORMATION = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), NORMALIZATION]
)


def load_dog(path: str):
    """load stanford dog dataset"""
    train_dataset = dogs(
        root=path,
        train=True,
        cropped=False,
        transform=TRAIN_TRANSFORMATION,
        download=True,
    )
    test_dataset = dogs(
        root=path,
        train=False,
        cropped=False,
        transform=TEST_TRANSFORMATION,
        download=True,
    )

    return train_dataset, test_dataset


def load_flower(path: str):
    """load flower dataset"""
    train_dataset = flowers(
        root=path, train=True, val=False, transform=TRAIN_TRANSFORMATION, download=True
    )
    test_dataset = flowers(
        root=path, train=False, val=True, transform=TEST_TRANSFORMATION, download=True
    )
    return train_dataset, test_dataset


def load_indoor(path: str):
    """load MIT Indoor dataset"""
    train_dataset = indoor(
        path, train=True, transform=TRAIN_TRANSFORMATION, download=True
    )
    test_dataset = indoor(
        path, train=False, transform=TEST_TRANSFORMATION, download=True
    )
    return train_dataset, test_dataset
