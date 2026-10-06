"""Unscented projection and moment-preserving virtual candidates."""

import math

import torch

from gasf.utils.math import build_scaling_rotation


def project_fisheye(points, viewmatrix, fx, fy, width, height, fisheye_params=None):
    # Points are world coordinates; viewmatrix uses row-vector convention.
    ones = torch.ones_like(points[..., :1])
    points_h = torch.cat([points, ones], dim=-1)

    # The stored world-to-camera matrix is transposed for row-vector products.
    camera = points_h @ viewmatrix
    x, y, z = camera[..., 0], camera[..., 1], camera[..., 2]

    radius = torch.sqrt(x * x + y * y).clamp_min(1e-7)
    theta = torch.atan2(radius, z)

    if fisheye_params is not None and fisheye_params.numel() == 8:
        fx, fy, cx, cy, k1, k2, k3, k4 = fisheye_params.unbind()
        theta2 = theta.square()
        theta4 = theta2.square()
        theta6 = theta4 * theta2
        theta8 = theta4.square()
        theta_d = theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        u = fx * theta_d * x / radius + cx
        v = fy * theta_d * y / radius + cy
    else:
        u = fx * theta * x / radius + 0.5 * (width - 1)
        v = fy * theta * y / radius + 0.5 * (height - 1)

    return torch.stack([u, v], dim=-1)


def split_ut_project_once(
    means3D,
    scales,
    rotations,
    viewmatrix,
    fx,
    fy,
    width,
    height,
    scaling_modifier=1.0,
    alpha=1.0,
    beta=2.0,
    kappa=0.0,
    fisheye_params=None,
    split_offset=0.55,
):
    """Moment-preserving binary split followed by one UT per child.

    The split axis is the Gaussian axis with the largest projected midpoint
    asymmetry.  The returned children are candidate-footprint components only;
    rendering still evaluates the original 3D Gaussian exactly once.
    """
    if not 0.0 < split_offset < 1.0:
        raise ValueError("split_offset must be in (0, 1)")
    factor = build_scaling_rotation(scaling_modifier * scales, rotations)
    axes = factor.transpose(1, 2)
    center = means3D[:, None, :]
    probe = torch.cat((center, center + axes, center - axes), dim=1)
    projected_probe = project_fisheye(
        probe,
        viewmatrix,
        fx,
        fy,
        width,
        height,
        fisheye_params=fisheye_params,
    )
    midpoint_error = torch.linalg.norm(
        projected_probe[:, 1:4]
        + projected_probe[:, 4:7]
        - 2.0 * projected_probe[:, :1],
        dim=-1,
    )
    split_axis_id = midpoint_error.argmax(dim=1)
    batch = torch.arange(means3D.shape[0], device=means3D.device)
    split_axis = axes[batch, split_axis_id]
    child_offset = split_offset * split_axis
    child_means = torch.stack((means3D - child_offset, means3D + child_offset), dim=1)

    residual = math.sqrt(1.0 - split_offset**2)
    child_scales = scales[:, None, :].expand(-1, 2, -1).clone()
    child_scales[batch, :, split_axis_id] *= residual
    child_rotations = rotations[:, None, :].expand(-1, 2, -1)
    child_mean2, child_cov2 = unscented_project(
        means3D=child_means.reshape(-1, 3),
        scales=child_scales.reshape(-1, 3),
        rotations=child_rotations.reshape(-1, 4),
        viewmatrix=viewmatrix,
        fx=fx,
        fy=fy,
        width=width,
        height=height,
        scaling_modifier=scaling_modifier,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        fisheye_params=fisheye_params,
    )
    return (
        child_mean2.reshape(-1, 2, 2),
        child_cov2.reshape(-1, 2, 3),
        midpoint_error.max(dim=1).values,
    )


def unscented_project(
    means3D,
    scales,
    rotations,
    viewmatrix,
    fx,
    fy,
    width,
    height,
    scaling_modifier=1.0,
    alpha=1.0,
    beta=2.0,
    kappa=0.0,
    fisheye_params=None,
):
    dim = 3
    lam = alpha**2 * (dim + kappa) - dim
    delta = (dim + lam) ** 0.5

    # L L^T = 3D covariance, 3D covariance = RSS^TR^T
    L = build_scaling_rotation(
        scaling_modifier * scales,
        rotations,
    )

    # Each column of RS supplies a sigma-point displacement direction.
    axes = L.transpose(1, 2)

    center = means3D[:, None, :]
    sigma_points = torch.cat(
        [
            center,
            center + delta * axes,
            center - delta * axes,
        ],
        dim=1,
    )

    projected = project_fisheye(
        sigma_points,
        viewmatrix,
        fx,
        fy,
        width,
        height,
        fisheye_params=fisheye_params,
    )

    wm = means3D.new_full((7,), 1.0 / (2.0 * (dim + lam)))
    wc = wm.clone()
    wm[0] = lam / (dim + lam)
    wc[0] = wm[0] + 1.0 - alpha**2 + beta

    mean2D = torch.sum(wm[None, :, None] * projected, dim=1)

    centered = projected - mean2D[:, None, :]
    cov2D_matrix = torch.einsum(
        "k,nki,nkj->nij",
        wc,
        centered,
        centered,
    )

    # Preserve the native rasterizer's pixel-space low-pass dilation.

    cov2D = torch.stack(
        [
            cov2D_matrix[:, 0, 0] + 0.3,
            cov2D_matrix[:, 0, 1],
            cov2D_matrix[:, 1, 1] + 0.3,
        ],
        dim=-1,
    )
    return mean2D, cov2D
