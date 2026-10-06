# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE file.
#
# For inquiries contact george.drettakis@inria.fr
#
# Derived from Inria / GRAPHDECO 3DGS; see LICENSE.
"""Physical Gaussians, SH appearance and the original clone/split optimizer."""

from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gasf.nace.gate import NACEGate
from gasf.utils.graphics import BasicPointCloud
from gasf.utils.math import (
    build_rotation,
    build_scaling_rotation,
    get_expon_lr_func,
    inverse_sigmoid,
    strip_symmetric,
)
from gasf.utils.sh import RGB2SH


def mkdir_p(path):
    Path(path).mkdir(parents=True, exist_ok=True)


class GaussianModel:
    def __init__(self, sh_degree=3):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        for name in [
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
        ]:
            setattr(self, name, torch.empty(0))
        self.nace_gate = NACEGate().cuda()
        self.gate_optimizer = None
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0.0
        self.setup_functions()

    def training_setup(self, config):
        self.percent_dense = config.percent_dense
        self.xyz_gradient_accum = torch.zeros((len(self._xyz), 1), device="cuda")
        self.denom = torch.zeros_like(self.xyz_gradient_accum)
        groups = [
            ("xyz", self._xyz, config.position_lr_init * self.spatial_lr_scale),
            ("f_dc", self._features_dc, config.feature_lr),
            ("f_rest", self._features_rest, config.feature_lr / 20),
            ("opacity", self._opacity, config.opacity_lr),
            ("scaling", self._scaling, config.scaling_lr),
            ("rotation", self._rotation, config.rotation_lr),
        ]
        self.optimizer = torch.optim.Adam(
            [{"name": n, "params": [p], "lr": lr} for (n, p, lr) in groups],
            lr=0.0,
            eps=1e-15,
        )
        self.gate_optimizer = torch.optim.Adam(
            self.nace_gate.parameters(), lr=config.gate_lr, eps=1e-15
        )
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=config.position_lr_init * self.spatial_lr_scale,
            lr_final=config.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=config.position_lr_delay_mult,
            max_steps=config.position_lr_max_steps,
        )

    def accumulate_densification(self, visible, camera_center, photometric_gradient):
        valid = visible & torch.any(photometric_gradient != 0, dim=-1)
        distance = (self._xyz[valid] - camera_center).norm(dim=-1, keepdim=True)
        self.xyz_gradient_accum[valid] += (
            photometric_gradient[valid] * distance / 2
        ).norm(dim=-1, keepdim=True)
        self.denom[valid] += 1

    def reset_opacity(self):
        opacity = inverse_sigmoid(
            torch.minimum(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01)
        )
        self._opacity = self.replace_tensor_to_optimizer(opacity, "opacity")["opacity"]

    def load_ply(self, path, load_gate=True):
        """Load exported Gaussians and their companion learned NACE weights."""
        path = Path(path)
        if load_gate:
            gate_path = path.with_name("nace_gate.pth")
            self.nace_gate.load_state_dict(
                torch.load(gate_path, map_location="cpu", weights_only=True)
            )
        vertices = PlyData.read(path)["vertex"]
        xyz = np.column_stack([vertices[a] for a in ["x", "y", "z"]])
        dc = np.stack([vertices[f"f_dc_{i}"] for i in range(3)], axis=1)[:, :, None]
        rest_names = sorted(
            (p.name for p in vertices.properties if p.name.startswith("f_rest_")),
            key=lambda s: int(s.split("_")[-1]),
        )
        expected = 3 * ((self.max_sh_degree + 1) ** 2 - 1)
        if len(rest_names) != expected:
            raise ValueError(
                f"Expected {expected} SH coefficients, found {len(rest_names)}"
            )
        rest = np.stack([vertices[n] for n in rest_names], axis=1).reshape(
            -1, 3, expected // 3
        )
        for name, value in {
            "_xyz": xyz,
            "_features_dc": dc.transpose(0, 2, 1),
            "_features_rest": rest.transpose(0, 2, 1),
            "_opacity": np.asarray(vertices["opacity"])[:, None],
            "_scaling": np.column_stack([vertices[f"scale_{i}"] for i in range(3)]),
            "_rotation": np.column_stack([vertices[f"rot_{i}"] for i in range(4)]),
        }.items():
            setattr(
                self,
                name,
                nn.Parameter(
                    torch.as_tensor(value.copy(), dtype=torch.float32, device="cuda")
                ),
            )
        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros(len(self._xyz), device="cuda")

    def capture(self):
        return {
            "parameters": {
                n: getattr(self, n).detach()
                for n in [
                    "_xyz",
                    "_features_dc",
                    "_features_rest",
                    "_opacity",
                    "_scaling",
                    "_rotation",
                ]
            },
            "active_sh_degree": self.active_sh_degree,
            "spatial_lr_scale": self.spatial_lr_scale,
            "nace_gate": self.nace_gate.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "gate_optimizer": self.gate_optimizer.state_dict(),
            "max_radii2D": self.max_radii2D,
            "xyz_gradient_accum": self.xyz_gradient_accum,
            "denom": self.denom,
        }

    def restore(self, state, config):
        for name, value in state["parameters"].items():
            setattr(self, name, nn.Parameter(value.to("cuda")))
        self.active_sh_degree = state["active_sh_degree"]
        self.spatial_lr_scale = state["spatial_lr_scale"]
        self.training_setup(config)
        self.nace_gate.load_state_dict(state["nace_gate"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.gate_optimizer.load_state_dict(state["gate_optimizer"])
        for name in ["max_radii2D", "xyz_gradient_accum", "denom"]:
            setattr(self, name, state[name].to("cuda"))

    def setup_functions(self):

        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])
        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 1e-07
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.1
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                # Keep optimizer checkpoints compatible with weights_only loading.
                lr = float(self.xyz_scheduler_args(iteration))
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        names = ["x", "y", "z", "nx", "ny", "nz"]
        names += [
            f"f_dc_{i}"
            for i in range(self._features_dc.shape[1] * self._features_dc.shape[2])
        ]
        names += [
            f"f_rest_{i}"
            for i in range(self._features_rest.shape[1] * self._features_rest.shape[2])
        ]
        names += ["opacity"]
        names += [f"scale_{i}" for i in range(3)]
        names += [f"rot_{i}" for i in range(4)]
        return names

    def save_ply(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        xyz = self._xyz.detach().cpu().numpy()
        arrays = [
            xyz,
            np.zeros_like(xyz),
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .cpu()
            .numpy(),
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .cpu()
            .numpy(),
            self._opacity.detach().cpu().numpy(),
            self._scaling.detach().cpu().numpy(),
            self._rotation.detach().cpu().numpy(),
        ]
        vertices = np.empty(
            len(xyz), dtype=[(n, "f4") for n in self.construct_list_of_attributes()]
        )
        vertices[:] = list(map(tuple, np.concatenate(arrays, axis=1)))
        PlyData([PlyElement.describe(vertices, "vertex")]).write(path)
        torch.save(
            {k: v.detach().cpu() for (k, v) in self.nace_gate.state_dict().items()},
            path.with_name("nace_gate.pth"),
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )
                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
    ):
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros(self.get_xyz.shape[0], device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        padded_grad = torch.zeros(n_init_points, device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
        )
        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
