# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file.
#
# For inquiries contact george.drettakis@inria.fr
#
"""The original image loss and the ScanNet++ valid-window variant."""

import math

import torch
import torch.nn.functional as F


def ssim(prediction, target, mask=None):
    if prediction.ndim == 3:
        prediction = prediction[None]
        target = target[None]
    channels = prediction.shape[1]
    if mask is None:
        gaussian = torch.tensor(
            [math.exp(-((x - 5) ** 2) / (2 * 1.5**2)) for x in range(11)],
            dtype=torch.float32,
        )
        gaussian = gaussian / gaussian.sum()
        window = (
            (gaussian[:, None] @ gaussian[None, :])
            .float()[None, None]
            .expand(channels, 1, 11, 11)
            .contiguous()
            .to(prediction)
        )
    else:
        x = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
        gaussian = torch.exp(-x * x / (2 * 1.5**2))
        gaussian /= gaussian.sum()
        window = (gaussian[:, None] * gaussian[None, :])[None, None].expand(
            channels, 1, 11, 11
        )

    def filt(tensor):
        return F.conv2d(tensor, window, padding=5, groups=channels)

    a, b = filt(prediction), filt(target)
    aa, bb, ab = a * a, b * b, a * b
    va = filt(prediction * prediction) - aa
    vb = filt(target * target) - bb
    covariance = filt(prediction * target) - ab
    value = ((2 * ab + 0.01**2) * (2 * covariance + 0.03**2)) / (
        (aa + bb + 0.01**2) * (va + vb + 0.03**2)
    )
    if mask is None:
        return value.mean()
    if mask.ndim == 3:
        mask = mask[None]
    mask = mask.to(prediction)
    valid = F.conv2d(mask, mask.new_ones(1, 1, 11, 11), padding=5) >= 120.999
    if not valid.any():
        raise ValueError("No fully valid SSIM windows")
    return (value * valid).sum() / (valid.sum() * channels)


def image_loss(prediction, target, lambda_dssim=0.2, mask=None):
    if mask is None:
        l1 = (prediction - target).abs().mean()
    else:
        mask = mask.to(prediction)
        l1 = ((prediction - target).abs() * mask).sum() / (3 * mask.sum()).clamp_min(1)
    return (1 - lambda_dssim) * l1 + lambda_dssim * (1 - ssim(prediction, target, mask))
