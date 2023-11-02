import os
import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader

import mixmatch.dataset.cifar10 as dataset
import mixmatch.models.wideresnet as models
from train import SemiLoss, WeightEMA, validate, train
from utils import mkdir_p

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()

SEED = 42

best_acc = 0  # best test accuracy


def main(
    epochs: int = 1024,
    batch_size: int = 64,
    lr: float = 0.002,
    n_labeled: int = 250,
    train_iteration: int = 1024,
    out: str = "result",
    ema_decay: float = 0.999,
    lambda_u: float = 75,
    alpha: float = 0.75,
    t: float = 0.5,
    device: str = "cuda",
):
    random.seed(42)
    np.random.seed(42)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global best_acc

    if not os.path.isdir(out):
        mkdir_p(out)

    # Data
    print(f"==> Preparing cifar10")

    (
        labeled_trainloader,
        unlabeled_trainloader,
        val_loader,
        test_loader,
    ) = dataset.get_cifar10(
        "./data", n_labeled, batch_size=batch_size, seed=SEED
    )

    # Model
    print("==> creating WRN-28-2")

    model = models.WideResNet(num_classes=10).to(device)
    ema_model = deepcopy(model).to(device)
    for param in ema_model.parameters():
        param.detach_()

    # cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay, lr=lr)

    test_accs = []
    # Train and val
    for epoch in range(epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, epochs, lr))

        train_loss, train_loss_x, train_loss_u = train(
            labeled_trainloader=labeled_trainloader,
            unlabeled_trainloader=unlabeled_trainloader,
            model=model,
            optimizer=optimizer,
            ema_optimizer=ema_optimizer,
            criterion=train_criterion,
            epoch=epoch,
            device="cuda",
            train_iteration=train_iteration,
            lambda_u=lambda_u,
            alpha=alpha,
            epochs=epochs,
            t=t,
        )

        def val_ema(dl: DataLoader):
            return validate(
                valloader=dl,
                model=ema_model,
                criterion=criterion,
                device=device,
            )

        _, train_acc = val_ema(labeled_trainloader)
        val_loss, val_acc = val_ema(val_loader)
        test_loss, test_acc = val_ema(test_loader)

        best_acc = max(val_acc, best_acc)
        test_accs.append(test_acc)

        print(
            f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f} | "
            f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc:.3f} | "
            f"Best Acc: {best_acc:.3f} | "
            f"Mean Acc: {np.mean(test_accs[-20:]):.3f} | "
            f"LR: {lr:.5f} | "
            f"Train Loss X: {train_loss_x:.3f} | "
            f"Train Loss U: {train_loss_u:.3f} "
        )

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])
