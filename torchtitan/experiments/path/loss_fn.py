# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .model_constants import DESIRE_LEN, PLAN_WIDTH, T_IDXS

# tensor reduction helpers
def reduce(tensor: torch.Tensor, reduction: str | None = None, metric: bool = False) -> torch.Tensor:
    def mean(*args, **kwargs) -> torch.Tensor:
        return torch.nanmean(*args, **kwargs) if metric else torch.mean(*args, **kwargs)

    if reduction == "mean":
        return mean(tensor)
    elif reduction is None and len(tensor.shape) > 1:
        # Reduce all but the first dimension
        return mean(tensor, dim=tuple(range(1, len(tensor.shape))))
    else:
        return tensor


def flatten_and_reduce(tdict, metric=False, lkey=""):
    reduced_dict = {}
    for rkey, val in tdict.items():
        key = lkey + rkey
        if isinstance(val, dict):
            reduced_dict.update(flatten_and_reduce(val, metric=metric, lkey=key + "_"))
        else:
            reduced_dict[key] = reduce(val, metric=metric)
    return reduced_dict


def num_from_nan(tensor, constant: float = 0):
    mask_tensor = torch.isnan(tensor)
    safe_tensor = tensor.masked_fill(mask_tensor, constant)
    return safe_tensor.detach(), ~mask_tensor  # detach unexpected gradient


def get_gaussian_log_likelihood(x, mu, log_sigma_raw, std_clamp: float = 1e-3, loss_clamp: float = 1000.0):
    err = torch.abs(x - mu)
    log_sigma_min = torch.clamp(log_sigma_raw, min=math.log(std_clamp))
    # an error 10 times the log_sigma corresponds to a probability on the order of 1e-10
    log_sigma = torch.max(log_sigma_raw, torch.log(1e-6 + err / math.sqrt(loss_clamp)))
    # log_lik = (err**2) * torch.exp(-2*log_sigma) / 2 + log_sigma_min + math.log(math.sqrt(2 * math.pi))
    log_lik = (err**2) * torch.exp(-2 * log_sigma) / 2 + log_sigma_min
    return err**2, log_lik


def get_laplacian_log_likelihood(x, mu, log_sigma_raw, std_clamp: float = 1e-3, loss_clamp: float = 1000.0):
    err = torch.abs(x - mu)
    log_sigma_min = torch.clamp(log_sigma_raw, min=math.log(std_clamp))
    log_sigma = torch.max(log_sigma_raw, torch.log(1e-6 + err / loss_clamp))
    # an error 20 times the b corresponds to a probability on the order of 1e-10
    # log_lik = err * torch.exp(-log_sigma) + log_sigma_min + math.log(2)
    log_lik = err * torch.exp(-log_sigma) + log_sigma_min
    return err, log_lik


def get_laplacian_metric(y_true, y_pred, n_values: int):
    return torch.abs(y_pred[..., :n_values] - y_true[..., :n_values])


def get_gaussian_metric(y_true, y_pred, n_values):
    return (y_pred[..., :n_values] - y_true[..., :n_values]) ** 2


def get_miou_metric(y_true, y_pred):
    nc = y_pred.shape[1]
    cm = (
        torch.bincount(y_true.long().flatten() * nc + y_pred.argmax(1).flatten(), minlength=nc * nc)
        .reshape(nc, nc)
        .float()
    )
    inter, union = cm.diag(), cm.sum(0) + cm.sum(1) - cm.diag()
    return torch.nanmean(torch.where(union > 0, inter / union.clamp(min=1), torch.full_like(inter, float("nan"))))


def get_soft_dice(logits, targets, eps=1.0):
    p = logits.softmax(1)
    t = F.one_hot(targets.long(), logits.shape[1]).permute(0, 3, 1, 2).to(p.dtype)
    dims = (0, 2, 3)
    dice = (2 * (p * t).sum(dims) + eps) / (p.sum(dims) + t.sum(dims) + eps)
    return 1 - dice.mean()


METRIC_FNS = {"laplacian": get_laplacian_metric, "gaussian": get_gaussian_metric}
LOG_LIKELIHOOD_FNS = {"laplacian": get_laplacian_log_likelihood, "gaussian": get_gaussian_log_likelihood}


class DensityMetric(nn.Module):
    def __init__(self, distribution="laplacian"):
        super().__init__()
        self.get_metric = METRIC_FNS[distribution]

    def forward(self, y_true, y_pred):
        n_values = int(y_pred.size(-1)) // 2
        return self.get_metric(y_true, y_pred, n_values)


class DensityLoss(nn.Module):
    def __init__(self, distribution="laplacian", std_clamp=1e-3, loss_clamp=1000.0, log_sigma_from_true=False):
        super().__init__()
        self.get_log_likelihood = LOG_LIKELIHOOD_FNS[distribution]
        self.std_clamp = std_clamp
        self.loss_clamp = loss_clamp
        self.log_sigma_from_true = log_sigma_from_true

    def forward(self, y_true, y_pred):
        n_values = int(y_pred.size(-1)) // 2
        mu_true = y_true[..., :n_values]
        mu_pred = y_pred[..., :n_values]
        mu_true, y_mask = num_from_nan(mu_true)
        log_sigma = y_true[..., n_values:] if self.log_sigma_from_true else y_pred[..., n_values:]
        log_likelihood = self.get_log_likelihood(mu_true, mu_pred, log_sigma, self.std_clamp, self.loss_clamp)[1]
        return y_mask * log_likelihood


class BinaryCrossentropy(nn.Module):
    def __init__(self, from_logits=False, label_smoothing=0.0):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.base_loss: nn.BCEWithLogitsLoss | nn.BCELoss
        if from_logits:
            self.base_loss = nn.BCEWithLogitsLoss(reduction="none")
        else:
            self.base_loss = nn.BCELoss(reduction="none")

    def forward(self, y_true, y_pred):
        y_true, y_mask = num_from_nan(y_true)
        if self.label_smoothing > 0.0:
            y_true = y_true.clamp(self.label_smoothing, 1 - self.label_smoothing)
        return y_mask * self.base_loss(y_pred, y_true)


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha: float = -1.0):
        super().__init__()
        import torchvision

        self.alpha = alpha
        self._sigmoid_focal_loss = torchvision.ops.sigmoid_focal_loss

    def forward(self, y_true, y_pred):
        y_true, y_mask = num_from_nan(y_true)
        error = self._sigmoid_focal_loss(y_pred, y_true, alpha=self.alpha, reduction="none")
        return y_mask * error


class CategoricalFocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, y_true, y_pred):
        input_log_softmax = torch.log_softmax(y_pred, dim=1)  # B C ...
        input_log_softmax = input_log_softmax.gather(1, y_true.unsqueeze(1)).squeeze(1)
        weight = torch.pow(1.0 - input_log_softmax.exp(), self.gamma)
        return -self.alpha * weight * input_log_softmax


class PlanMetric(nn.Module):
    def __init__(self, distribution="laplacian"):
        from xx.stages.lib.vt2_helpers import Plan

        super().__init__()
        T_IDXS_5S = int(np.argmax(T_IDXS > 5))
        self.distribution = distribution
        self.density_metric = DensityMetric(distribution)
        self.pos_slice = slice(
            PLAN_WIDTH * T_IDXS_5S + Plan.POSITION.start, PLAN_WIDTH * T_IDXS_5S + Plan.POSITION.stop
        )
        self.vel_slice = slice(
            PLAN_WIDTH * T_IDXS_5S + Plan.VELOCITY.start, PLAN_WIDTH * T_IDXS_5S + Plan.VELOCITY.stop
        )

    def forward(self, y_true, y_pred):
        return {
            self.distribution: self.density_metric(y_true, y_pred),
            "5s_position_mae": torch.abs(y_pred[..., self.pos_slice] - y_true[..., self.pos_slice]),
            "5s_velocity_mae": torch.abs(y_pred[..., self.vel_slice] - y_true[..., self.vel_slice]),
        }


class FutureDesireLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_loss = nn.CrossEntropyLoss(reduction="none")
        self.desire_len = DESIRE_LEN

    def forward(self, y_true, y_pred):
        leading_shape = y_true.shape[:-1]
        y_true = y_true.reshape(-1, 4, self.desire_len)
        y_pred = y_pred.reshape(-1, 4, self.desire_len)
        y_true, y_mask = num_from_nan(y_true)
        total = []
        for i in range(4):
            total.append(y_mask[:, i].all(-1) * self.cat_loss(y_pred[:, i], torch.argmax(y_true[:, i], 1)))
        return torch.stack(total).sum(0).reshape(leading_shape)


class DesireStateLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cat_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(self, y_true, y_pred):
        leading_shape = y_true.shape[:-1]
        y_true, y_mask = num_from_nan(y_true)
        loss = self.cat_loss(y_pred.reshape(-1, y_pred.size(-1)), y_true.argmax(-1).reshape(-1))
        return (y_mask.all(-1).reshape(-1) * loss).reshape(leading_shape)


class DrivingLoss(nn.Module):
    def __init__(self):
        # TODO: move in
        from xx.training.lib.lpips import ImageScale, ReconstructionLoss

        super().__init__()
        self.losses = nn.ModuleDict(
            {
                "plan": DensityLoss("laplacian"),
                "road_edges": DensityLoss("laplacian"),
                "lane_lines": DensityLoss("laplacian"),
                "lane_lines_prob": BinaryCrossentropy(from_logits=True),
                "pose": DensityLoss("laplacian"),
                "sim_pose": DensityLoss("laplacian"),
                "height": DensityLoss("laplacian", std_clamp=3e-2),
                "road_from_device": DensityLoss("laplacian"),
                "lead": DensityLoss("laplacian"),
                "lead_prob": BinaryCrossentropy(from_logits=True),
                "meta": BinaryCrossentropy(from_logits=True),
                "road_transform": DensityLoss("laplacian"),
                "wide_from_device_euler": DensityLoss("laplacian"),
                "desire_pred": FutureDesireLoss(),
                "desire_state": DesireStateLoss(),
                "action": DensityLoss("laplacian"),
                "imgs": ReconstructionLoss(
                    y_pred_format="NCHW",
                    n_images=2,
                    cat_axis=1,
                    y_pred_scale=ImageScale.UINT8,
                    y_true_scale=ImageScale.UINT8,
                ),
            }
        )
        self.lossw = defaultdict(lambda: 0.1)
        self.lossw.update(
            {
                "desire_pred": 0.01,
                "lead": 1.0,
                "lead_prob": 1.0,
                "plan": 5.0,
                "pose": 1.0,
                "sim_pose": 1.0,
                "lane_lines": 1.0,
                "action": 1.0,
                "imgs_lpips": 10.0,
                "imgs_L2": 25.0,
                "desire_state": 0.1,
            }
        )

    def forward(self, ret, targets):
        tlosses = {}
        for k in targets:
            if k not in self.losses:
                continue
            if k not in ret:
                continue
            tlosses[k] = self.losses[k](targets[k], ret[k])

        tlosses = flatten_and_reduce(tlosses)
        loss = torch.stack([tlosses[k] * self.lossw[k] for k in tlosses]).sum(dim=0)

        rlosses = {"loss": loss, **tlosses}
        rlosses = {k: v.detach() for k, v in rlosses.items()}
        del targets, tlosses
        return loss, rlosses


class DrivingMetric(nn.Module):
    def __init__(self):
        super().__init__()
        self.metrics = nn.ModuleDict(
            {
                "plan": PlanMetric("laplacian"),
                "height": DensityMetric("laplacian"),
            }
        )

    @torch.no_grad()
    def forward(self, ret, targets):
        tmetrics = {}
        for k in targets:
            if k not in self.metrics:
                continue
            if k not in ret:
                continue
            tmetrics[k] = self.metrics[k](targets[k], ret[k])

        tmetrics = flatten_and_reduce(tmetrics, metric=True)
        rmetrics = {k + "_metric": v.detach() for k, v in tmetrics.items()}
        del targets, tmetrics
        return rmetrics
