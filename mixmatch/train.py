from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.nn.functional import cross_entropy, one_hot
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchmetrics.functional import accuracy


def validate(
    *,
    valloader: DataLoader,
    model: nn.Module,
    criterion: Callable,
    device: str,
):
    n = []
    losses = []
    accs = []

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(valloader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, targets.long())

            # measure accuracy and record loss
            # TODO: Technically, we shouldn't * 100, but it's fine for now as
            #  it doesn't impact training
            acc = (
                accuracy(
                    outputs,
                    targets,
                    task="multiclass",
                    num_classes=outputs.shape[1],
                )
                * 100
            )
            losses.append(loss.item())
            accs.append(acc.item())
            n.append(inputs.size(0))

    # return weighted mean
    return (
        sum([loss * n for loss, n in zip(losses, n)]) / sum(n),
        sum([top * n for top, n in zip(accs, n)]) / sum(n),
    )


class SemiLoss(object):
    @staticmethod
    def linear_rampup(current: float, rampup_length: int):
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)

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

        l_x = cross_entropy(outputs_x, targets_x)
        # TODO: Not sure why this is different from MSELoss
        #  It's likely not a big deal, but it's worth investigating if we have
        #  too much time on our hands
        l_u = torch.mean((probs_u - targets_u) ** 2)
        return (
            l_x,
            l_u,
            lambda_u * self.linear_rampup(epoch, epochs),
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
