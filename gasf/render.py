"""UT candidates + NACE selection; render each original 3D Gaussian once."""

import math

import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

from gasf.nace.gate import gate_probability
from gasf.nace.projection import split_ut_project_once, unscented_project
from gasf.nace.teacher import sampled_coverage_oracle_teacher


def render(camera, model, config, background, temperature=1.0, train_gate=False):
    params = camera.fisheye_params.to(model.get_xyz).contiguous()
    tangent_x = math.tan(camera.FoVx * 0.5)
    tangent_y = math.tan(camera.FoVy * 0.5)
    projection = dict(
        means3D=model.get_xyz,
        scales=model.get_scaling,
        rotations=model.get_rotation,
        viewmatrix=camera.world_view_transform,
        fx=camera.image_width / (2 * tangent_x),
        fy=camera.image_height / (2 * tangent_y),
        width=camera.image_width,
        height=camera.image_height,
        alpha=config.ut_alpha,
        beta=config.ut_beta,
        kappa=config.ut_kappa,
        fisheye_params=params,
    )
    single_mean, single_cov = unscented_project(**projection)
    child_mean, child_cov, curvature = split_ut_project_once(
        **projection, split_offset=config.split_offset
    )
    probability = gate_probability(model, camera, single_cov, curvature, temperature)
    split = probability.detach() >= 0.5
    means = (
        torch.where(
            split[:, None, None], child_mean, single_mean[:, None, :].expand(-1, 2, -1)
        )
        .detach()
        .contiguous()
    )
    covariances = (
        torch.where(
            split[:, None, None], child_cov, single_cov[:, None, :].expand(-1, 2, -1)
        )
        .detach()
        .contiguous()
    )
    counts = (1 + split.to(torch.int32)).contiguous()
    normalized_mean = torch.stack(
        (
            2 * (single_mean[:, 0] + 0.5) / camera.image_width - 1,
            2 * (single_mean[:, 1] + 0.5) / camera.image_height - 1,
        ),
        dim=-1,
    )
    means2d = torch.cat(
        (normalized_mean, torch.zeros_like(normalized_mean[:, :1])), dim=-1
    )
    if means2d.requires_grad:
        means2d.retain_grad()
    settings = GaussianRasterizationSettings(
        image_height=camera.image_height,
        image_width=camera.image_width,
        tanfovx=tangent_x,
        tanfovy=tangent_y,
        bg=background,
        is_fisheye=True,
        scale_modifier=1.0,
        viewmatrix=camera.world_view_transform,
        projmatrix=camera.full_proj_transform,
        sh_degree=model.active_sh_degree,
        campos=camera.camera_center,
        prefiltered=False,
        debug=False,
        use_ut=True,
        use_3d_response=True,
        sort_by_camera_distance=False,
        sort_by_tile_tau=False,
        tau_k_buffer_size=16,
        fisheye_params=params,
        fisheye_no_center_cull=False,
    )
    image, radii = GaussianRasterizer(raster_settings=settings)(
        means3D=model.get_xyz,
        means2D=means2d,
        cov2D_precomp=single_cov.contiguous(),
        candidate_means2D=means,
        candidate_cov2D=covariances,
        candidate_counts=counts,
        ut_gate=model.get_xyz.new_empty((0,)),
        shs=model.get_features,
        colors_precomp=None,
        opacities=model.get_opacity,
        scales=model.get_scaling,
        rotations=model.get_rotation,
        cov3D_precomp=None,
    )
    target = torch.zeros_like(probability)
    mask = torch.zeros_like(split)
    gain = torch.zeros_like(probability)
    if train_gate:
        target, mask, gain = sampled_coverage_oracle_teacher(
            means3d=model.get_xyz,
            scales=model.get_scaling,
            rotations=model.get_rotation,
            opacity=model.get_opacity,
            world_view_transform=camera.world_view_transform,
            camera_center=camera.camera_center,
            fisheye_params=params,
            single_mean2=single_mean,
            single_cov2=single_cov,
            child_mean2=child_mean,
            child_cov2=child_cov,
            width=camera.image_width,
            height=camera.image_height,
            gaussians_per_bin=config.gaussians_per_bin,
            pixels_per_gaussian=config.pixels_per_gaussian,
        )
    return dict(
        render=image,
        radii=radii,
        visibility_filter=radii > 0,
        viewspace_points=means2d,
        gate_probability=probability,
        gate_target=target,
        gate_mask=mask,
        gate_gain=gain,
    )
