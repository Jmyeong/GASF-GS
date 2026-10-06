"""Sampled, response-based supervision for the learned Split-UT gate.

The oracle compares the actual 3D Gaussian response captured by the Single-UT
tile rectangle and by the union of the two virtual-child tile rectangles.  It
does not use a curvature threshold, opacity penalty, or projected-area
threshold.  To keep online training practical, it evaluates a stratified
subset of Gaussians and uniformly sampled pixels in their joint bounding box.
"""

import torch

from gasf.utils.math import build_rotation

TILE_SIZE = 16
POWER_CUTOFF = -4.482
ALPHA_CUTOFF = 1.0 / 255.0


def _covariance_radius(covariance):
    """Match the CUDA rasterizer's packed-covariance 3-sigma radius."""
    cov_xx, cov_xy, cov_yy = covariance.unbind(dim=-1)
    determinant = cov_xx * cov_yy - cov_xy.square()
    middle = 0.5 * (cov_xx + cov_yy)
    lambda_max = middle + torch.sqrt(
        torch.clamp(middle.square() - determinant, min=0.1)
    )
    return torch.ceil(3.0 * torch.sqrt(torch.clamp(lambda_max, min=0.1)))


def _tile_rectangles(means2d, radii, width, height):
    grid_x = (width + TILE_SIZE - 1) // TILE_SIZE
    grid_y = (height + TILE_SIZE - 1) // TILE_SIZE
    min_x = torch.trunc((means2d[..., 0] - radii) / TILE_SIZE).long().clamp(0, grid_x)
    min_y = torch.trunc((means2d[..., 1] - radii) / TILE_SIZE).long().clamp(0, grid_y)
    max_x = (
        torch.trunc((means2d[..., 0] + radii + TILE_SIZE - 1) / TILE_SIZE)
        .long()
        .clamp(0, grid_x)
    )
    max_y = (
        torch.trunc((means2d[..., 1] + radii + TILE_SIZE - 1) / TILE_SIZE)
        .long()
        .clamp(0, grid_y)
    )
    return (
        min_x * TILE_SIZE,
        min_y * TILE_SIZE,
        torch.minimum(max_x * TILE_SIZE, max_x.new_full(max_x.shape, width)),
        torch.minimum(max_y * TILE_SIZE, max_y.new_full(max_y.shape, height)),
    )


def _select_stratified(radii, valid, per_bin):
    selected = []
    for mask in (
        valid & (radii < 8),
        valid & (radii >= 8) & (radii < 64),
        valid & (radii >= 64),
    ):
        ids = mask.nonzero(as_tuple=False).squeeze(1)
        if ids.numel() == 0:
            continue
        count = min(per_bin, ids.numel())
        order = torch.randperm(ids.numel(), device=ids.device)[:count]
        selected.append(ids.index_select(0, order))
    if not selected:
        return radii.new_empty((0,), dtype=torch.long)
    return torch.cat(selected)


def _fisheye_world_rays(pixel_x, pixel_y, params, world_view_transform):
    fx, fy, cx, cy, k1, k2, k3, k4 = params.unbind()
    px = (pixel_x - cx) / fx
    py = (pixel_y - cy) / fy
    theta_d = torch.sqrt(px.square() + py.square())
    theta = theta_d.clone()
    for _ in range(8):
        theta2 = theta.square()
        theta4 = theta2.square()
        theta6 = theta4 * theta2
        theta8 = theta4.square()
        value = (
            theta * (1.0 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
            - theta_d
        )
        derivative = (
            1.0
            + 3.0 * k1 * theta2
            + 5.0 * k2 * theta4
            + 7.0 * k3 * theta6
            + 9.0 * k4 * theta8
        )
        theta = theta - value / derivative.abs().clamp_min(1e-8)
    radial = torch.where(
        theta_d > 1e-7,
        torch.sin(theta) / theta_d.clamp_min(1e-7),
        torch.ones_like(theta_d),
    )
    camera_rays = torch.stack(
        (radial * px, radial * py, torch.cos(theta)),
        dim=-1,
    )
    rotation = world_view_transform[:3, :3]
    world_rays = camera_rays @ rotation.transpose(0, 1)
    return torch.nn.functional.normalize(world_rays, dim=-1)


@torch.no_grad()
def sampled_coverage_oracle_teacher(
    means3d,
    scales,
    rotations,
    opacity,
    world_view_transform,
    camera_center,
    fisheye_params,
    single_mean2,
    single_cov2,
    child_mean2,
    child_cov2,
    width,
    height,
    gaussians_per_bin=16,
    pixels_per_gaussian=512,
    scaling_modifier=1.0,
    diagnostics=None,
):
    """Label Split only when sampled valid-response coverage strictly improves."""
    if gaussians_per_bin <= 0:
        raise ValueError("split_oracle_gaussians_per_bin must be positive")
    if pixels_per_gaussian <= 0:
        raise ValueError("split_oracle_pixels_per_gaussian must be positive")
    if fisheye_params.numel() != 8:
        raise ValueError("coverage oracle requires 8 OPENCV_FISHEYE parameters")
    means3d = means3d.detach()
    effective_scales = (scales.detach() * scaling_modifier).clamp_min(1e-12)
    rotations = rotations.detach()
    opacity = opacity.detach().reshape(-1)
    single_mean2 = single_mean2.detach()
    single_cov2 = single_cov2.detach()
    child_mean2 = child_mean2.detach()
    child_cov2 = child_cov2.detach()

    single_radii = _covariance_radius(single_cov2)
    child_radii = _covariance_radius(child_cov2)
    single_bounds = _tile_rectangles(single_mean2, single_radii, width, height)
    child_bounds = _tile_rectangles(child_mean2, child_radii, width, height)

    camera_points = (
        torch.cat((means3d, torch.ones_like(means3d[:, :1])), dim=-1)
        @ world_view_transform
    )[:, :3]
    single_nonempty = (single_bounds[2] > single_bounds[0]) & (
        single_bounds[3] > single_bounds[1]
    )
    child_nonempty = (
        (child_bounds[2] > child_bounds[0]) & (child_bounds[3] > child_bounds[1])
    ).any(dim=1)
    valid = (
        torch.isfinite(camera_points).all(dim=1)
        & (camera_points[:, 2] > 1e-6)
        & (single_nonempty | child_nonempty)
    )
    selected = _select_stratified(single_radii, valid, gaussians_per_bin)

    target = means3d.new_zeros((means3d.shape[0],))
    supervision_mask = torch.zeros(
        means3d.shape[0],
        dtype=torch.bool,
        device=means3d.device,
    )
    sampled_gain = means3d.new_zeros((means3d.shape[0],))
    if selected.numel() == 0:
        return target, supervision_mask, sampled_gain

    s_bounds = tuple(bound.index_select(0, selected) for bound in single_bounds)
    c_bounds = tuple(bound.index_select(0, selected) for bound in child_bounds)
    joint_min_x = torch.minimum(s_bounds[0], c_bounds[0].min(dim=1).values)
    joint_min_y = torch.minimum(s_bounds[1], c_bounds[1].min(dim=1).values)
    joint_max_x = torch.maximum(s_bounds[2], c_bounds[2].max(dim=1).values)
    joint_max_y = torch.maximum(s_bounds[3], c_bounds[3].max(dim=1).values)
    span_x = (joint_max_x - joint_min_x).clamp_min(1)
    span_y = (joint_max_y - joint_min_y).clamp_min(1)

    sample_shape = (selected.numel(), pixels_per_gaussian)
    pixel_x = (
        joint_min_x[:, None]
        + torch.floor(
            torch.rand(sample_shape, device=means3d.device) * span_x[:, None]
        ).long()
    )
    pixel_y = (
        joint_min_y[:, None]
        + torch.floor(
            torch.rand(sample_shape, device=means3d.device) * span_y[:, None]
        ).long()
    )
    pixel_x_float = pixel_x.to(means3d.dtype)
    pixel_y_float = pixel_y.to(means3d.dtype)

    single_candidate = (
        (pixel_x >= s_bounds[0][:, None])
        & (pixel_x < s_bounds[2][:, None])
        & (pixel_y >= s_bounds[1][:, None])
        & (pixel_y < s_bounds[3][:, None])
    )
    child_candidate = (
        (pixel_x[:, None, :] >= c_bounds[0][..., None])
        & (pixel_x[:, None, :] < c_bounds[2][..., None])
        & (pixel_y[:, None, :] >= c_bounds[1][..., None])
        & (pixel_y[:, None, :] < c_bounds[3][..., None])
    ).any(dim=1)

    rays = _fisheye_world_rays(
        pixel_x_float,
        pixel_y_float,
        fisheye_params.to(device=means3d.device, dtype=means3d.dtype),
        world_view_transform,
    )
    selected_means = means3d.index_select(0, selected)
    selected_scales = effective_scales.index_select(0, selected)
    selected_rotations = build_rotation(rotations.index_select(0, selected))
    selected_opacity = opacity.index_select(0, selected)
    ray_origin = camera_center.to(means3d)
    offset = ray_origin[None, :] - selected_means
    offset_local = (
        torch.einsum("gi,gij->gj", offset, selected_rotations) / selected_scales
    )
    direction_local = (
        torch.einsum("gsj,gji->gsi", rays, selected_rotations)
        / selected_scales[:, None, :]
    )
    denominator = (direction_local * direction_local).sum(dim=-1)
    numerator = (offset_local[:, None, :] * direction_local).sum(dim=-1)
    tau = -numerator / denominator.clamp_min(1e-20)
    hit_local = offset_local[:, None, :] + tau[..., None] * direction_local
    power = -0.5 * (hit_local * hit_local).sum(dim=-1)
    covariance_valid = torch.isfinite(selected_scales).all(dim=1) & (
        selected_scales.square().prod(dim=1) > 1e-18
    )
    true_support = (
        covariance_valid[:, None]
        & torch.isfinite(power)
        & (denominator > 1e-12)
        & (tau > 0.0)
        & (power > POWER_CUTOFF)
        & (selected_opacity[:, None] * torch.exp(power) >= ALPHA_CUTOFF)
    )

    single_hits = (true_support & single_candidate).sum(dim=1)
    split_hits = (true_support & child_candidate).sum(dim=1)
    gain = split_hits - single_hits
    if diagnostics is not None:
        diagnostics.update(
            selected=selected,
            support=true_support.sum(1),
            single_hits=single_hits,
            split_hits=split_hits,
        )
    selected_target = (gain > 0).to(target.dtype)
    supervised_selected = torch.ones_like(selected_target, dtype=torch.bool)
    target.index_copy_(0, selected, selected_target)
    sampled_gain.index_copy_(0, selected, gain.to(target.dtype))
    supervision_mask[selected] = supervised_selected
    return target, supervision_mask, sampled_gain
