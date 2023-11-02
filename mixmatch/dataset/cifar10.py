from __future__ import annotations
from pathlib import Path
from typing import Callable, Sequence, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomCrop,
)

tf_preproc = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ]
)

tf_aug = transforms.Compose(
    [
        lambda x: torch.nn.functional.pad(
            x,
            (
                4,
                4,
                4,
                4,
            ),
            mode="reflect",
        ),
        RandomCrop(32),
        RandomHorizontalFlip(),
    ]
)


class CIFAR10Subset(CIFAR10):
    def __init__(
        self,
        root: str,
        idxs: Sequence[int] | None = None,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if idxs is not None:
            self.data = self.data[idxs]
            self.targets = np.array(self.targets)[idxs].tolist()


class CIFAR10SubsetKAug(CIFAR10Subset):
    def __init__(
        self,
        root: str,
        k_augs: int,
        aug: Callable,
        idxs: Sequence[int] | None = None,
        train: bool = True,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        download: bool = False,
    ):
        super().__init__(
            root=root,
            idxs=idxs,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.k_augs = k_augs
        self.aug = aug

    def __getitem__(self, item):
        img, target = super().__getitem__(item)
        return tuple(self.aug(img) for _ in range(self.k_augs)), target


def get_dataloaders(
    dataset_dir: Path | str,
    train_lbl_size: float = 0.005,
    train_unl_size: float = 0.980,
    batch_size: int = 48,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """Get the dataloaders for the CIFAR10 dataset.

    Notes:
        The train_lbl_size and train_unl_size must sum to less than 1.
        The leftover data is used for the validation set.

    Args:
        dataset_dir: The directory where the dataset is stored.
        train_lbl_size: The size of the labelled training set.
        train_unl_size: The size of the unlabelled training set.
        batch_size: The batch size.
        num_workers: The number of workers for the dataloaders.
        seed: The seed for the random number generators.

    Returns:
        4 DataLoaders: train_lbl_dl, train_unl_dl, val_unl_dl, test_dl
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    src_train_ds = CIFAR10(
        dataset_dir,
        train=True,
        download=True,
        transform=tf_preproc,
    )
    src_test_ds = CIFAR10(
        dataset_dir,
        train=False,
        download=True,
        transform=tf_preproc,
    )

    train_size = len(src_train_ds)
    train_unl_size = int(train_size * train_unl_size)
    train_lbl_size = int(train_size * train_lbl_size)
    val_size = int(train_size - train_unl_size - train_lbl_size)

    targets = np.array(src_train_ds.targets)
    ixs = np.arange(len(targets))
    train_unl_ixs, lbl_ixs = train_test_split(
        ixs,
        train_size=train_unl_size,
        stratify=targets,
    )
    lbl_targets = targets[lbl_ixs]

    val_ixs, train_lbl_ixs = train_test_split(
        lbl_ixs,
        train_size=val_size,
        stratify=lbl_targets,
    )

    train_lbl_ds = CIFAR10SubsetKAug(
        dataset_dir,
        idxs=train_lbl_ixs,
        train=True,
        transform=tf_preproc,
        download=True,
        k_augs=1,
        aug=tf_aug,
    )
    train_unl_ds = CIFAR10SubsetKAug(
        dataset_dir,
        idxs=train_unl_ixs,
        train=True,
        transform=tf_preproc,
        download=True,
        k_augs=2,
        aug=tf_aug,
    )
    val_ds = CIFAR10Subset(
        dataset_dir,
        idxs=val_ixs,
        train=True,
        transform=tf_preproc,
        download=True,
    )

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
    )

    train_lbl_dl = DataLoader(
        train_lbl_ds, shuffle=True, drop_last=True, **dl_args
    )
    train_unl_dl = DataLoader(
        train_unl_ds, shuffle=True, drop_last=True, **dl_args
    )
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)
    test_dl = DataLoader(src_test_ds, shuffle=False, **dl_args)

    return (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        src_train_ds.classes,
    )
