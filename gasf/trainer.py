"""Single-image optimization with independent NACE and FRIR gradients."""

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from gasf.data import Scene
from gasf.frir.regularizer import FRIR
from gasf.io import write_json
from gasf.losses import image_loss
from gasf.model import GaussianModel
from gasf.render import render


def rng_state():
    numpy = np.random.get_state()
    return dict(
        python=random.getstate(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state(),
        numpy=(
            numpy[0],
            torch.from_numpy(numpy[1].astype(np.int64)),
            numpy[2],
            numpy[3],
            numpy[4],
        ),
    )


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
    n = state["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), n[2], n[3], n[4]))


def train(
    source,
    output,
    dataset,
    views,
    seed,
    optimization,
    nace,
    frir_config,
    anchors,
    resume=None,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = GaussianModel(sh_degree=3)
    scene = Scene.load(source, views)
    model.create_from_pcd(scene.point_cloud, scene.cameras_extent)
    model.training_setup(optimization)
    regularizer = FRIR(scene, anchors, frir_config, seed)
    background = torch.zeros(3, device="cuda")
    first = 0
    stack = []
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            checkpoint["dataset"] != dataset
            or checkpoint["views"] != views
            or checkpoint["seed"] != seed
        ):
            raise ValueError("Checkpoint dataset/views/seed differ")
        model.restore(checkpoint["model"], optimization)
        first = checkpoint["iteration"]
        regularizer.rng.set_state(checkpoint["frir_rng"].cpu())
        regularizer.updates = checkpoint["frir_updates"]
        regularizer.gradient_checked = checkpoint["frir_gradient_checked"]
        by_name = {c.image_name: c for c in scene.train_cameras}
        stack = [by_name[n] for n in checkpoint["camera_stack"]]
        restore_rng(checkpoint["rng"])
    started = time.perf_counter()
    progress = tqdm(range(first + 1, optimization.iterations + 1), desc="GASF-GS")
    for iteration in progress:
        fraction = min(iteration / max(1, nace.anneal_until), 1.0)
        temperature = math.exp(
            (1 - fraction) * math.log(nace.temperature)
            + fraction * math.log(nace.temperature_final)
        )
        model.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            model.oneupSHdegree()
        if not stack:
            stack = scene.train_cameras.copy()
        camera = stack.pop(random.randint(0, len(stack) - 1))
        package = render(camera, model, nace, background, temperature, train_gate=True)
        image = package["render"]
        target = camera.original_image.cuda()
        mask = camera.valid_mask.cuda() if dataset == "ScanNetpp" else None
        photometric = image_loss(image, target, optimization.lambda_dssim, mask)
        supervised = package["visibility_filter"] & package["gate_mask"]
        gate = image.new_zeros(())
        if supervised.any():
            probability = package["gate_probability"][supervised].clamp(1e-6, 1 - 1e-6)
            gate = F.binary_cross_entropy(
                probability, package["gate_target"][supervised]
            )
        ray_loss = regularizer.loss(iteration, camera, image, model)
        total = photometric + ray_loss + optimization.lambda_gate * gate
        if not torch.isfinite(total):
            raise RuntimeError(f"Nonfinite loss at iteration {iteration}")
        photo_gradient = None
        if iteration < optimization.densify_until_iter:
            photo_gradient = torch.autograd.grad(
                photometric, model._xyz, retain_graph=True
            )[0].detach()
        total.backward()
        if model._xyz.grad is None or not torch.isfinite(model._xyz.grad).all():
            raise RuntimeError(
                f"Invalid Gaussian position gradient at iteration {iteration}"
            )
        with torch.no_grad():
            if iteration < optimization.densify_until_iter:
                visible = package["visibility_filter"]
                radii = package["radii"]
                model.max_radii2D[visible] = torch.maximum(
                    model.max_radii2D[visible], radii[visible].float()
                )
                model.accumulate_densification(
                    visible, camera.camera_center, photo_gradient
                )
                if (
                    iteration > optimization.densify_from_iter
                    and iteration % optimization.densification_interval == 0
                ):
                    model.densify_and_prune(
                        optimization.densify_grad_threshold,
                        0.005,
                        scene.cameras_extent,
                        None,
                    )
                if iteration % optimization.opacity_reset_interval == 0:
                    model.reset_opacity()
            # Match the original final-iteration update policy.
            if iteration < optimization.iterations:
                model.optimizer.step()
                model.optimizer.zero_grad(set_to_none=True)
                model.gate_optimizer.step()
                model.gate_optimizer.zero_grad(set_to_none=True)
            if iteration % 100 == 0 or iteration == optimization.iterations:
                details = dict(
                    iteration=iteration,
                    gaussians=len(model.get_xyz),
                    loss=float(total),
                    image_loss=float(photometric),
                    gate_loss=float(gate),
                    frir_loss=float(ray_loss),
                    frir_updates=regularizer.updates,
                    elapsed_seconds=time.perf_counter() - started,
                )
                write_json(output / "progress.json", details)
                progress.set_postfix(
                    loss=f"{float(total):.5f}", gaussians=len(model.get_xyz)
                )
            if iteration % 5000 == 0 or iteration == optimization.iterations:
                checkpoint = dict(
                    model=model.capture(),
                    iteration=iteration,
                    dataset=dataset,
                    views=views,
                    seed=seed,
                    camera_stack=[c.image_name for c in stack],
                    rng=rng_state(),
                    frir_rng=regularizer.rng.get_state(),
                    frir_updates=regularizer.updates,
                    frir_gradient_checked=regularizer.gradient_checked,
                )
                torch.save(checkpoint, output / "checkpoint.pth")
    torch.cuda.synchronize()
    point_cloud = (
        output
        / "point_cloud"
        / f"iteration_{optimization.iterations}"
        / "point_cloud.ply"
    )
    model.save_ply(point_cloud)
    info = dict(
        train_seconds=time.perf_counter() - started,
        iterations=optimization.iterations,
        seed=seed,
        train_views=len(scene.train_cameras),
        test_views=len(scene.test_cameras),
        frir_updates=regularizer.updates,
        frir_gradient_checked=regularizer.gradient_checked,
        anchor_observations={
            n: len(obs[0]) for n, obs in regularizer.observations.items()
        },
    )
    write_json(output / "training.json", info)
    return scene, model, info
