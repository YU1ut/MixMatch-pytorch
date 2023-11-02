from pathlib import Path
from typing import Callable, Sequence, List

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data.dataset import T_co
from torchvision.datasets import CIFAR10

import torch
from torchvision import transforms
from torchvision.transforms.v2 import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomCrop,
)

# from torchvision.transforms.v2 import AutoAugmentPolicy, AutoAugment

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


class SubsetKAugments(Subset):
    def __init__(
        self,
        dataset: Dataset,
        indices: Sequence[int],
        augment: Callable,
        k: int,
    ):
        super().__init__(dataset, indices)
        self.augment = augment
        self.k = k

    def __getitems__(self, indices: List[int]) -> List:
        xs: list[tuple[torch.Tensor, int]] = super().__getitems__(indices)
        # [(x1, y1), (x2, y2), ...]
        xs_aug = [(tuple(tf_aug(x) for _ in range(self.k)), y) for x, y in xs]
        return xs_aug


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

    train_lbl_ds = Subset(src_train_ds, train_lbl_ixs)
    train_unl_ds = SubsetKAugments(
        src_train_ds, train_unl_ixs, augment=tf_aug, k=2
    )
    val_ds = Subset(src_train_ds, val_ixs)

    dl_args = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # We use drop_last=True to ensure that the batch size is always the same
    # This is crucial as we need to average the predictions across the batch
    # size axis.

    train_lbl_dl = DataLoader(train_lbl_ds, shuffle=True, **dl_args)
    train_unl_dl = DataLoader(train_unl_ds, shuffle=True, **dl_args)
    val_dl = DataLoader(val_ds, shuffle=False, **dl_args)
    test_dl = DataLoader(src_test_ds, shuffle=False, **dl_args)

    return (
        train_lbl_dl,
        train_unl_dl,
        val_dl,
        test_dl,
        src_train_ds.classes,
    )
