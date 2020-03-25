import numpy as np
from PIL import Image

import torchvision
import torch

import torchvision.transforms as transforms
from torchvision.datasets import STL10

# dict containing supported datasets with their image resolutions
imsize_dict = {'C10': 32, 'STL10': 96}

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2023, 0.1994, 0.2010)

stl10_mean = (0.4914, 0.4822, 0.4465)
stl10_std = (0.2471, 0.2435, 0.2616)

dataset_stats = {
    'C10' : {
        'mean': cifar10_mean,
        'std': cifar10_std
    },
    'STL10' : {
        'mean': stl10_mean,
        'std': stl10_std
    },
}

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar10(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True):

    base_dataset = torchvision.datasets.CIFAR10(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/10))

    train_labeled_dataset = CIFAR10_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR10_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train))
    val_dataset = CIFAR10_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR10_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset

def get_stl10(root,
                 transform_train=None, transform_val=None,
                 download=True):

    training_set = STL10(root, split='train', download=True, transform=transform_train)
    dev_set = STL10(root, split='test', download=True, transform=transform_val)
    unl_set = STL10(root, split='unlabeled', download=True, transform=transform_train)

    print (f"#Labeled: {len(training_set)} #Unlabeled: {len(unl_set)} #Val: {len(dev_set)} #Test: None")
    return training_set, unl_set, dev_set, None

def validate_dataset(dataset):
    if dataset not in imsize_dict:
        raise ValueError("Dataset %s not supported." % dataset)

def get_transforms(dataset, resolution):
    dataset_resolution = imsize_dict[dataset]

    if dataset == 'STL10':
        if resolution == 96:
            transform_train = transforms.Compose([
                transforms.RandomCrop(96, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stl10_mean, stl10_std),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(86, padding=0),
                transforms.Resize(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(stl10_mean, stl10_std),
            ])
        if dataset_resolution == resolution:
            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
            ])
        else:
            transform_val = transforms.Compose([
                transforms.Resize(resolution),
                transforms.ToTensor(),
                transforms.Normalize(dataset_stats[dataset]['mean'], dataset_stats[dataset]['std']),
            ])
    if dataset == 'C10':
        # already normalized in the CIFAR10_labeled/CIFAR10_unlabeled class
        transform_train = transforms.Compose([
            RandomPadandCrop(resolution),
            RandomFlip(),
            ToTensor(),
        ])
        transform_val = transforms.Compose([
            ToTensor(),
        ])


    return transform_train, transform_val



def train_val_split(labels, n_labeled_per_class):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(10):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-500])
        val_idxs.extend(idxs[-500:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

def normalise(x, mean=cifar10_mean, std=cifar10_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR10_labeled(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
    

class CIFAR10_unlabeled(CIFAR10_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        self.targets = np.array([-1 for i in range(len(self.targets))])
        