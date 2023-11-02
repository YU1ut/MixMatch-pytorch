from mixmatch.main import main


def test_main_seed():
    """Test that the model is always initialized the same way when seeded.

    Ensure that training is always the same when seeded."""
    epochs = 1
    train_iteration = 8
    best_acc_1, mean_acc_1 = main(
        epochs=epochs, train_iteration=train_iteration
    )
    best_acc_2, mean_acc_2 = main(
        epochs=epochs, train_iteration=train_iteration
    )

    assert best_acc_1 == best_acc_2
    assert mean_acc_1 == mean_acc_2


def test_main_seed_epoch():
    """Test that the model is always initialized the same way when seeded.

    Ensure that training is always the same when seeded."""
    epochs = 1
    best_acc_1, mean_acc_1 = main(epochs=epochs)
    # best_acc_2, mean_acc_2 = main(epochs=epochs)

    # assert best_acc_1 == best_acc_2
    # assert mean_acc_1 == mean_acc_2
    assert best_acc_1 == 20.28
    assert mean_acc_1 == 19.54
