import numpy as np

from dataset.cifar10 import get_cifar10


def test_load_seeded():
    """Test that the dataset is always loaded the same way when seeded."""
    n_labeled = 250
    batch_size = 64
    seed = 42

    (
        labeled_trainloader,
        unlabeled_trainloader,
        val_loader,
        test_loader,
    ) = get_cifar10("./data", n_labeled, batch_size=batch_size, seed=seed)

    labeled_targets_1 = labeled_trainloader.dataset.targets
    unlabeled_targets_1 = unlabeled_trainloader.dataset.targets
    val_targets_1 = val_loader.dataset.targets
    test_targets_1 = test_loader.dataset.targets

    (
        labeled_trainloader,
        unlabeled_trainloader,
        val_loader,
        test_loader,
    ) = get_cifar10("./data", n_labeled, batch_size=batch_size, seed=seed)

    labeled_targets_2 = labeled_trainloader.dataset.targets
    unlabeled_targets_2 = unlabeled_trainloader.dataset.targets
    val_targets_2 = val_loader.dataset.targets
    test_targets_2 = test_loader.dataset.targets

    assert np.all(labeled_targets_1 == labeled_targets_2)
    assert np.all(unlabeled_targets_1 == unlabeled_targets_2)
    assert np.all(val_targets_1 == val_targets_2)
    assert np.all(test_targets_1 == test_targets_2)

    print(labeled_targets_1)
    pass
