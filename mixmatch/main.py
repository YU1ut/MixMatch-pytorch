import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from tensorboardX import SummaryWriter

import mixmatch.dataset.cifar10 as dataset
import mixmatch.models.wideresnet as models
from train import SemiLoss, WeightEMA, validate, save_checkpoint, train
from utils import mkdir_p, Logger

EPOCHS: int = 1024
START_EPOCH: int = 0
MANUAL_SEED: int = 0
RESUME: str = ""
GPU: str = "0"
OUT: str = "result"
BATCH_SIZE: int = 64
LR: float = 0.002
N_LABELED: int = 250
TRAIN_ITERATION: int = 1024
EMA_DECAY: float = 0.999
ALPHA: float = 0.75
LAMBDA_U: float = 75
T: float = 0.5

state = {
    "epochs": EPOCHS,
    "start_epoch": START_EPOCH,
    "manual_seed": MANUAL_SEED,
    "resume": RESUME,
    "gpu": GPU,
    "out": OUT,
    "batch_size": BATCH_SIZE,
    "lr": LR,
    "n_labeled": N_LABELED,
    "train_iteration": TRAIN_ITERATION,
    "ema_decay": EMA_DECAY,
    "alpha": ALPHA,
    "lambda_u": LAMBDA_U,
    "T": T,
}

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
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

    def create_model(ema=False):
        model_ = models.WideResNet(num_classes=10)
        model_ = model_.cuda()

        if ema:
            for param in model_.parameters():
                param.detach_()

        return model_

    model = create_model()
    ema_model = create_model(ema=True)

    # cudnn.benchmark = True
    print(
        "    Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters()) / 1000000.0)
    )

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)
    start_epoch = 0

    # Resume
    title = "noisy-cifar-10"

    logger = Logger(os.path.join(out, "log.txt"), title=title)
    logger.set_names(
        [
            "Train Loss",
            "Train Loss X",
            "Train Loss U",
            "Valid Loss",
            "Valid Acc.",
            "Test Loss",
            "Test Acc.",
        ]
    )

    writer = SummaryWriter(out)
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, epochs):
        print("\nEpoch: [%d | %d] LR: %f" % (epoch + 1, epochs, state["lr"]))

        train_loss, train_loss_x, train_loss_u = train(
            labeled_trainloader=labeled_trainloader,
            unlabeled_trainloader=unlabeled_trainloader,
            model=model,
            optimizer=optimizer,
            ema_optimizer=ema_optimizer,
            criterion=train_criterion,
            epoch=epoch,
            use_cuda=use_cuda,
            train_iteration=train_iteration,
            lambda_u=lambda_u,
            alpha=alpha,
            epochs=epochs,
        )
        _, train_acc = validate(
            valloader=labeled_trainloader,
            model=ema_model,
            criterion=criterion,
            use_cuda=use_cuda,
            mode="Train Stats",
        )
        val_loss, val_acc = validate(
            valloader=val_loader,
            model=ema_model,
            criterion=criterion,
            use_cuda=use_cuda,
            mode="Valid Stats",
        )
        test_loss, test_acc = validate(
            valloader=test_loader,
            model=ema_model,
            criterion=criterion,
            use_cuda=use_cuda,
            mode="Test Stats ",
        )

        step = train_iteration * (epoch + 1)

        writer.add_scalar("losses/train_loss", train_loss, step)
        writer.add_scalar("losses/valid_loss", val_loss, step)
        writer.add_scalar("losses/test_loss", test_loss, step)

        writer.add_scalar("accuracy/train_acc", train_acc, step)
        writer.add_scalar("accuracy/val_acc", val_acc, step)
        writer.add_scalar("accuracy/test_acc", test_acc, step)

        # append logger file
        logger.append(
            [
                train_loss,
                train_loss_x,
                train_loss_u,
                val_loss,
                val_acc,
                test_loss,
                test_acc,
            ]
        )

        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "acc": val_acc,
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint=out,
        )
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])
