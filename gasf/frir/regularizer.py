"""Reliability-weighted supervision of native Gaussian response peaks."""

from pathlib import Path

import numpy as np
import torch

from gasf.frir.loss import guided_loss


class FRIR:
    def __init__(self, scene, anchors, config, seed):
        self.config = config
        self.rng = torch.Generator().manual_seed(seed + 190943)
        self.updates = 0
        self.gradient_checked = False
        self.observations = {}
        names = {Path(n).stem: i for i, n in enumerate(anchors["training_images"])}
        for camera in scene.train_cameras:
            index = names[camera.image_name]
            rows = []
            if [camera.image_height, camera.image_width] != anchors["image_shapes"][
                index
            ]:
                raise ValueError("Anchor image dimensions do not match training")
            for anchor in anchors["anchors"]:
                if index not in anchor["cameras"]:
                    continue
                k = anchor["cameras"].index(index)
                uv = np.rint(anchor["uv"][k]).astype(int)
                if not (
                    0 <= uv[0] < camera.image_width and 0 <= uv[1] < camera.image_height
                ):
                    continue
                distance = anchor["range"][k]
                epsilon = max(config.epsilon_min, 2 * anchor["sigma"][k] / distance)
                rows.append((uv, distance, epsilon))
            uv = (
                torch.as_tensor(np.stack([r[0] for r in rows]))
                if rows
                else torch.empty((0, 2), dtype=torch.long)
            )
            target = torch.tensor([r[1] for r in rows], device="cuda")
            tolerance = torch.tensor([r[2] for r in rows], device="cuda")
            self.observations[camera.image_name] = (uv, target, tolerance)
        self.last = {}

    def loss(self, iteration, camera, image, model):
        c = self.config
        uv, target, tolerance = self.observations[camera.image_name]
        if iteration < c.start_iteration or iteration % c.interval != 0 or not len(uv):
            return image.new_zeros(())
        selected = torch.randperm(len(uv), generator=self.rng)[: c.rays_per_step]
        raw, info = guided_loss(
            model,
            camera,
            image,
            uv[selected],
            target[selected.to(target.device)],
            tolerance[selected.to(tolerance.device)],
            c.lambda_mass,
            c.pseudo_huber_delta,
        )
        ramp = min(
            1.0,
            (iteration - (c.start_iteration - 1))
            / (c.warmup_end - c.start_iteration + 1),
        )
        loss = raw * (c.lambda_ri * ramp)
        if info["trusted"]:
            self.updates += 1
            if not self.gradient_checked and iteration >= 400:
                gradients = torch.autograd.grad(
                    loss,
                    (
                        model._xyz,
                        model._scaling,
                        model._rotation,
                        model._opacity,
                        model._features_dc,
                        model._features_rest,
                    ),
                    retain_graph=True,
                    allow_unused=True,
                )
                if not all(
                    g is not None and torch.isfinite(g).all() for g in gradients[:4]
                ):
                    raise RuntimeError("FRIR geometry gradient check failed")
                if any(g is not None for g in gradients[4:]):
                    raise RuntimeError("FRIR must not update SH appearance directly")
                self.gradient_checked = True
        self.last = info
        return loss
