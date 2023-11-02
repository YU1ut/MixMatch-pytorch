import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.nn.functional import cross_entropy


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
