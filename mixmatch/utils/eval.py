from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from tqdm import tqdm

from utils.ema import WeightEMA
from utils.interleave import interleave
from utils.loss import SemiLoss


def train(
    *,
    train_lbl_dl: DataLoader,
    train_unl_dl: DataLoader,
    model: nn.Module,
    optim: optim.Optimizer,
    ema_optim: WeightEMA,
    loss_fn: SemiLoss,
    epoch: int,
    epochs: int,
    device: str,
    train_iters: int,
    lambda_u: float,
    mix_beta_alpha: float,
    sharpen_temp: float,
) -> tuple[float, float, float]:
    losses = []
    losses_x = []
    losses_u = []
    n = []

    lbl_iter = iter(train_lbl_dl)
    unl_iter = iter(train_unl_dl)

    model.train()
    for batch_idx in tqdm(range(train_iters)):
        try:
            x_lbl, y_lbl = next(lbl_iter)
        except StopIteration:
            lbl_iter = iter(train_lbl_dl)
            x_lbl, y_lbl = next(lbl_iter)

        try:
            x_unls, _ = next(unl_iter)
        except StopIteration:
            unl_iter = iter(train_unl_dl)
            x_unls, _ = next(unl_iter)

        batch_size = x_lbl.size(0)

        # Transform label to one-hot
        y_lbl = one_hot(y_lbl.long(), num_classes=10)

        x_lbl = x_lbl.to(device)
        y_lbl = y_lbl.to(device)
        x_unls = [u.to(device) for u in x_unls]

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            y_unls = [torch.softmax(model(u), dim=1) for u in x_unls]
            p = sum(y_unls) / 2
            pt = p ** (1 / sharpen_temp)
            y_unl = pt / pt.sum(dim=1, keepdim=True).detach()

        # mixup
        x = torch.cat([x_lbl, *x_unls], dim=0)
        y = torch.cat([y_lbl, y_unl, y_unl], dim=0)

        ratio = np.random.beta(mix_beta_alpha, mix_beta_alpha)
        ratio = max(ratio, 1 - ratio)

        shuf_idx = torch.randperm(x.size(0))

        x_mix = ratio * x + (1 - ratio) * x[shuf_idx]
        y_mix = ratio * y + (1 - ratio) * y[shuf_idx]

        # interleave labeled and unlabed samples between batches to
        # get correct batchnorm calculation
        x_mix = list(torch.split(x_mix, batch_size))
        x_mix = interleave(x_mix, batch_size)

        y_mix_pred = [model(x) for x in x_mix]

        # put interleaved samples back
        y_mix_pred = interleave(y_mix_pred, batch_size)
        y_mix_lbl_pred = y_mix_pred[0]
        y_mix_lbl = y_mix[:batch_size]
        y_mix_unl_pred = torch.cat(y_mix_pred[1:], dim=0)
        y_mix_unl = y_mix[batch_size:]

        loss_lbl, loss_unl, loss_unl_scale = loss_fn(
            outputs_x=y_mix_lbl_pred,
            targets_x=y_mix_lbl,
            outputs_u=y_mix_unl_pred,
            targets_u=y_mix_unl,
            epoch=epoch + batch_idx / train_iters,
            lambda_u=lambda_u,
            epochs=epochs,
        )

        loss = loss_lbl + loss_unl_scale * loss_unl

        # record loss
        losses.append(loss)
        losses_x.append(loss_lbl)
        losses_u.append(loss_unl)
        n.append(x_lbl.size(0))

        # compute gradient and do SGD step
        optim.zero_grad()
        loss.backward()
        optim.step()
        ema_optim.step()

    return (
        sum([loss * n for loss, n in zip(losses, n)]) / sum(n),
        sum([loss * n for loss, n in zip(losses_x, n)]) / sum(n),
        sum([loss * n for loss, n in zip(losses_u, n)]) / sum(n),
    )


def validate(
    *,
    valloader: DataLoader,
    model: nn.Module,
    loss_fn: Callable,
    device: str,
):
    n = []
    losses = []
    accs = []

    model.eval()
    with torch.no_grad():
        for x, y in tqdm(valloader):
            x = x.to(device)
            y = y.to(device)
            # compute output
            y_pred = model(x)
            loss = loss_fn(y_pred, y.long())

            # measure accuracy and record loss
            # TODO: Technically, we shouldn't * 100, but it's fine for now as
            #  it doesn't impact training
            acc = (
                accuracy(
                    y_pred,
                    y,
                    task="multiclass",
                    num_classes=y_pred.shape[1],
                )
                * 100
            )
            losses.append(loss.item())
            accs.append(acc.item())
            n.append(x.size(0))

    # return weighted mean
    return (
        sum([loss * n for loss, n in zip(losses, n)]) / sum(n),
        sum([top * n for top, n in zip(accs, n)]) / sum(n),
    )
