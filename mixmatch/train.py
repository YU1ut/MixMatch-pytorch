import os
import shutil
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import AverageMeter, accuracy


def validate(
    *,
    valloader: DataLoader,
    model: nn.Module,
    criterion: Callable,
    use_cuda: bool,
    mode: str,
):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(valloader)):
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

    return losses.avg, top1.avg


def save_checkpoint(
    state,
    is_best: bool,
    checkpoint: str,
    filename: str = "checkpoint.pth.tar",
):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(
            filepath, os.path.join(checkpoint, "model_best.pth.tar")
        )


def linear_rampup(current: float, rampup_length: int):
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
        epoch: float,
        lambda_u: float,
        epochs: int,
    ):
        probs_u = torch.softmax(outputs_u, dim=1)

        l_x = -torch.mean(
            torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1)
        )
        l_u = torch.mean((probs_u - targets_u) ** 2)

        return (
            l_x,
            l_u,
            lambda_u * linear_rampup(epoch, epochs),
        )


class WeightEMA(object):
    def __init__(
        self,
        model: nn.Module,
        ema_model: nn.Module,
        alpha: float = 0.999,
        lr: float = 0.002,
    ):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * lr

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


def train(
    *,
    labeled_trainloader: DataLoader,
    unlabeled_trainloader: DataLoader,
    model: nn.Module,
    optimizer: optim.Optimizer,
    ema_optimizer,
    criterion: SemiLoss,
    epoch: int,
    device: str,
    train_iteration: int,
    lambda_u: float,
    alpha: float,
    epochs: int,
    t: float,
) -> tuple[float, float, float]:
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in tqdm(range(train_iteration)):
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

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, 10).scatter_(
            1, targets_x.view(-1, 1).long(), 1
        )

        inputs_x = inputs_x.to(device)
        targets_x = targets_x.to(device)
        inputs_u = inputs_u.to(device)
        inputs_u2 = inputs_u2.to(device)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            outputs_u2 = model(inputs_u2)
            p = (
                torch.softmax(outputs_u, dim=1)
                + torch.softmax(outputs_u2, dim=1)
            ) / 2
            pt = p ** (1 / t)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        ratio = np.random.beta(alpha, alpha)

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
            outputs_x=logits_x,
            targets_x=mixed_target[:batch_size],
            outputs_u=logits_u,
            targets_u=mixed_target[batch_size:],
            epoch=epoch + batch_idx / train_iteration,
            lambda_u=lambda_u,
            epochs=epochs,
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

    return (
        losses.avg,
        losses_x.avg,
        losses_u.avg,
    )
