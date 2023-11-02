import numpy as np

from mixmatch.dataset.cifar10 import get_dataloaders


def test_load_seeded():
    """Test that the dataset is always loaded the same way when seeded."""
    n_labeled = 250
    batch_size = 64
    seed = 42

    (
        train_lbl_dl,
        train_unl_dl,
        val_loader,
        test_loader,
        classes,
    ) = get_dataloaders(
        dataset_dir="./data",
        train_lbl_size=0.005,
        train_unl_size=0.980,
        batch_size=batch_size,
        seed=seed,
    )
    labeled_indices_1 = train_lbl_dl.dataset.indices
    unlabeled_indices_1 = train_unl_dl.dataset.indices
    val_indices_1 = val_loader.dataset.indices
    test_targets_1 = test_loader.dataset.targets

    (
        train_lbl_dl,
        train_unl_dl,
        val_loader,
        test_loader,
        classes,
    ) = get_dataloaders(
        dataset_dir="./data",
        train_lbl_size=0.005,
        train_unl_size=0.980,
        batch_size=batch_size,
        seed=seed,
    )
    labeled_indices_2 = train_lbl_dl.dataset.indices
    unlabeled_indices_2 = train_unl_dl.dataset.indices
    val_indices_2 = val_loader.dataset.indices
    test_targets_2 = test_loader.dataset.targets

    assert np.all(labeled_indices_1 == labeled_indices_2)
    assert np.all(unlabeled_indices_1 == unlabeled_indices_2)
    assert np.all(val_indices_1 == val_indices_2)
    assert np.all(test_targets_1 == test_targets_2)
