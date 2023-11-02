from mixmatch.main import main


def test_main_seed_fast():
    """The fast variant to ensure that the model doesn't change."""
    epochs = 1
    train_iteration = 8
    best_acc_1, mean_acc_1 = main(
        epochs=epochs, train_iteration=train_iteration
    )

    assert best_acc_1 == 8.53333332570394
    assert mean_acc_1 == 8.79


def test_main_seed_epoch():
    """Ensure that the model doesn't change when refactoring"""
    epochs = 1
    best_acc_1, mean_acc_1 = main(epochs=epochs)
    assert best_acc_1 == 24.266666646321614
    assert mean_acc_1 == 23.01
