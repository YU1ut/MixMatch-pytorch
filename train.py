from __future__ import print_function

import argparse
import os
import random
import shutil
import time
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from progress.bar import Bar
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

import dataset.cifar10 as dataset
import models.wideresnet as models
from utils import Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description="PyTorch MixMatch Training")
# Optimization options
parser.add_argument(
    "--epochs",
    default=1024,
    type=int,
    metavar="N",
    help="number of total epochs to run",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "--batch-size", default=64, type=int, metavar="N", help="train batchsize"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.002,
    type=float,
    metavar="LR",
    help="initial learning rate",
)
# Checkpoints
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
# Miscs
parser.add_argument("--manualSeed", type=int, default=0, help="manual seed")
# Device options
parser.add_argument(
    "--gpu", default="0", type=str, help="id(s) for CUDA_VISIBLE_DEVICES"
)
# Method options
parser.add_argument(
    "--n-labeled", type=int, default=250, help="Number of labeled data"
)
parser.add_argument(
    "--train-iteration",
    type=int,
    default=1024,
    help="Number of iteration per epoch",
)
parser.add_argument(
    "--out", default="result", help="Directory to output the result"
)
parser.add_argument("--alpha", default=0.75, type=float)
parser.add_argument("--lambda-u", default=75, type=float)
parser.add_argument("--T", default=0.5, type=float)
parser.add_argument("--ema-decay", default=0.999, type=float)

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
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
            labeled_trainloader,
            unlabeled_trainloader,
            model,
            optimizer,
            ema_optimizer,
            train_criterion,
            epoch,
            use_cuda,
        )
        _, train_acc = validate(
            labeled_trainloader,
            ema_model,
            criterion,
            epoch,
            use_cuda,
            mode="Train Stats",
        )
        val_loss, val_acc = validate(
            val_loader,
            ema_model,
            criterion,
            epoch,
            use_cuda,
            mode="Valid Stats",
        )
        test_loss, test_acc = validate(
            test_loader,
            ema_model,
            criterion,
            epoch,
            use_cuda,
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
        )
        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print("Best acc:")
    print(best_acc)

    print("Mean acc:")
    print(np.mean(test_accs[-20:]))

    return best_acc, np.mean(test_accs[-20:])


def train(
    labeled_trainloader: DataLoader,
    unlabeled_trainloader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema_optimizer,
    criterion: Callable,
    epoch: int,
    use_cuda: bool,
) -> tuple[float, float, float]:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar("Training", max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except StopIteration:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)
        except StopIteration:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(
            1, targets_x.view(-1, 1).long(), 1
        )

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(
                non_blocking=True
            )
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (
                torch.softmax(outputs_u, dim=1)
                + torch.softmax(outputs_u2, dim=1)
            ) / 2
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        ratio = np.random.beta(args.alpha, args.alpha)

        ratio = max(ratio, 1 - ratio)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = ratio * input_a + (1 - ratio) * input_b
        mixed_target = ratio * target_a + (1 - ratio) * target_b

        # interleave labeled and unlabed samples between batches to
        # get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for x in mixed_input[1:]:
            logits.append(model(x))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        l_x, l_u, w = criterion(
            logits_x,
            mixed_target[:batch_size],
            logits_u,
            mixed_target[batch_size:],
            epoch + batch_idx / args.train_iteration,
        )

        loss = l_x + w * l_u

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(l_x.item(), inputs_x.size(0))
        losses_u.update(l_u.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = (
            "({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s |"
            " Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | "
            "Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | "
            "W: {w:.4f}"
        ).format(
            batch=batch_idx + 1,
            size=args.train_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            w=ws.avg,
        )
        bar.next()
    bar.finish()

    return (
        losses.avg,
        losses_x.avg,
        losses_u.avg,
    )


def validate(
    valloader: DataLoader,
    model: nn.Module,
    criterion: Callable,
    epoch: int,
    use_cuda: bool,
    mode: str,
):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f"{mode}", max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(
                    non_blocking=True
                )
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = (
                "({batch}/{size}) Data: {data:.3f}s | "
                "Batch: {bt:.3f}s | "
                "Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | "
                "top1: {top1: .4f} | top5: {top5: .4f}"
            ).format(
                batch=batch_idx + 1,
                size=len(valloader),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()
    return losses.avg, top1.avg


def save_checkpoint(
    state,
    is_best: bool,
    checkpoint: str = args.out,
    filename: str = "checkpoint.pth.tar",
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath, os.path.join(checkpoint, "model_best.pth.tar")
        )


def linear_rampup(current: int, rampup_length: int = args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(
        self,
        outputs_x: torch.Tensor,
        targets_x: torch.Tensor,
        outputs_u: torch.Tensor,
        targets_u: torch.Tensor,
        epoch: int,
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        l_x = -torch.mean(
            torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
        )
        l_u = torch.mean((probs_u - targets_u) ** 2)

        return l_x, l_u, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        alpha: float = 0.999,
    ):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype == torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    main()
