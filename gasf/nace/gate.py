"""Shared per-Gaussian/view gate with detached geometric features."""

import torch
from torch import nn


class NACEGate(nn.Module):
    def __init__(self):
        super().__init__()
        # Preserve the CPU random-draw sequence of the evaluated implementation.
        # Its discarded auxiliary gate preceded NACE initialization; no auxiliary
        # network or parameters are retained in this release.
        for shape in [(16, 5), (16,), (16, 16), (16,), (1, 16), (1,)]:
            torch.empty(shape).uniform_(-1.0, 1.0)
        torch.empty((1, 16)).normal_(mean=0.0, std=0.01)
        self.network = nn.Sequential(
            nn.Linear(7, 24),
            nn.SiLU(),
            nn.Linear(24, 16),
            nn.SiLU(),
            nn.Linear(16, 1),
        )
        nn.init.normal_(self.network[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.network[-1].bias, -2.0)

    def forward(self, features):
        return self.network(features).squeeze(-1)


def gate_probability(model, camera, single_covariance, curvature, temperature):
    if temperature <= 0:
        raise ValueError("NACE temperature must be positive")
    xyz = model.get_xyz
    camera_xyz = (
        torch.cat((xyz, torch.ones_like(xyz[:, :1])), dim=-1)
        @ camera.world_view_transform
    )[:, :3]
    depth = camera_xyz[:, 2].clamp_min(1e-6)
    rho = torch.linalg.vector_norm(camera_xyz[:, :2], dim=-1)
    angle = (torch.atan2(rho, depth) / (torch.pi / 2)).clamp(0.0, 2.0)
    scales = model.get_scaling
    maximum = scales.max(dim=1).values
    minimum = scales.min(dim=1).values.clamp_min(1e-8)
    a, b, c = single_covariance.unbind(dim=-1)
    determinant = (a * c - b.square()).clamp_min(1e-8)
    middle = 0.5 * (a + c)
    eigenvalue = middle + torch.sqrt((middle.square() - determinant).clamp_min(0.0))
    radius = 3 * torch.sqrt(eigenvalue.clamp_min(1e-8))
    features = torch.stack(
        (
            torch.log((maximum / depth).clamp_min(1e-8) / 0.05).clamp(-6.0, 6.0) / 6.0,
            angle,
            angle.square(),
            torch.log((maximum / minimum).clamp_min(1.0)).clamp(0.0, 6.0) / 6.0,
            model.get_opacity.squeeze(-1),
            (curvature / radius.clamp_min(1.0)).clamp(0.0, 2.0),
            (1.0 - angle).clamp(-0.5, 1.0),
        ),
        dim=-1,
    ).detach()
    return torch.sigmoid(model.nace_gate(features) / temperature)
