"""Held-out renders and the metric conventions of the two datasets."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from gasf.io import load_json, write_json
from gasf.losses import ssim
from gasf.render import render


def save_rgb(path, tensor, torchvision_rounding=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu().clamp(0, 1)
    if torchvision_rounding:
        value = (value * 255 + 0.5).to(torch.uint8)
    else:
        value = torch.from_numpy(np.rint(value.numpy() * 255).astype(np.uint8))
    Image.fromarray(value.permute(1, 2, 0).numpy()).save(path)


def common_mask(scene, camera):
    # Prepared benchmark inputs include the exact mask used for published results.
    meta = scene.source / "benchmark.json"
    if meta.exists():
        shared = Path(load_json(meta)["shared"])
        path = shared / "common_masks" / (camera.image_name + ".png")
        if path.exists():
            with Image.open(path) as image:
                mask = np.asarray(image.convert("L")).copy() > 0
            return torch.from_numpy(mask)[None].cuda()
    # Same geometric intersection without a private benchmark adapter.
    h, w = camera.image_height, camera.image_width
    yy, xx = np.mgrid[:h, :w]
    params = camera.calibration_params
    xy = (np.stack((xx, yy), -1) - params[2:4]) / params[:2]
    radius = np.linalg.norm(xy, axis=-1)
    theta = radius.copy()
    for _ in range(20):
        rho = theta * (1 + sum(params[4 + k] * theta ** (2 * k + 2) for k in range(4)))
        slope = 1 + sum(
            (2 * k + 3) * params[4 + k] * theta ** (2 * k + 2) for k in range(4)
        )
        theta -= (rho - radius) / slope
    center = np.array([w / 2, h / 2])
    focal = params[:2]
    projected = [
        xy * (theta / np.maximum(radius, 1e-12))[..., None] * focal + center,
        xy * (np.tan(theta) / np.maximum(radius, 1e-12))[..., None] * focal + center,
        np.stack((xx, yy), -1) - params[2:4] + center,
    ]
    valid = np.ones((h, w), dtype=bool)
    for points in projected:
        points = points.astype(np.float32)
        valid &= (
            (points[..., 0] >= 1)
            & (points[..., 0] < w - 2)
            & (points[..., 1] >= 1)
            & (points[..., 1] < h - 2)
        )
    return torch.from_numpy(valid)[None].cuda() & camera.valid_mask.cuda()


@torch.no_grad()
def evaluate(scene, model, nace, output, dataset, iteration, skip_lpips=False):
    root = Path(output) / "test" / f"iteration_{iteration}"
    background = torch.zeros(3, device="cuda")
    criterion = None
    if not skip_lpips:
        import lpips

        criterion = (
            lpips.LPIPS(
                net="vgg",
                version="0.1",
                spatial=dataset == "ScanNetpp",
                eval_mode=True,
                verbose=False,
            )
            .cuda()
            .eval()
        )
    rows = []
    for camera in sorted(scene.test_cameras, key=lambda c: c.image_name):
        image = render(
            camera, model, nace, background, temperature=nace.temperature_final
        )["render"].clamp(0, 1)
        target = camera.original_image.cuda()
        mask = common_mask(scene, camera) if dataset == "ScanNetpp" else None
        error = (image - target).square()
        mse = (
            error.mean()
            if mask is None
            else (error * mask).sum() / (3 * mask.sum()).clamp_min(1)
        )
        row = dict(
            view=camera.image_name,
            psnr=float(-10 * mse.clamp_min(1e-12).log10()),
            ssim=float(ssim(image, target, mask)),
        )
        row["valid_pixels"] = (
            int(mask.sum())
            if mask is not None
            else camera.image_width * camera.image_height
        )
        for kind, tensor in [("renders", image), ("gt", target)]:
            save_rgb(
                root / kind / (camera.image_name + ".png"),
                tensor,
                dataset == "FisheyeNeRF",
            )
        if mask is not None:
            save_rgb(
                root / "mask" / (camera.image_name + ".png"),
                mask.expand(3, -1, -1).float(),
            )
        if criterion is not None:
            if mask is None:
                value = criterion(2 * image[None] - 1, 2 * target[None] - 1).mean()
            else:
                # Published masked LPIPS uses saved uint8 renders and canonical GT.
                pred = (
                    torch.from_numpy(np.rint(image.cpu().numpy() * 255)).to(image) / 255
                )[None]
                gt = target[None]
                valid = mask.bool()[None]
                a = torch.where(valid, 2 * pred - 1, 0.0)
                b = torch.where(valid, 2 * gt - 1, 0.0)
                feature_map = criterion(a, b, normalize=False)
                value = (feature_map * valid).sum() / valid.sum()
            row["lpips" if mask is None else "lpips_masked_vgg"] = float(value)
        rows.append(row)
    keys = [k for k in rows[0] if k not in ["view", "valid_pixels"]]
    mean = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    result = dict(
        mean=mean,
        views=rows,
        gaussians=len(model.get_xyz),
        evaluation="Full native image"
        if dataset == "FisheyeNeRF"
        else "Common calibrated FOV intersection; invalid pixels excluded",
        lpips_protocol="VGG v0.1 full-image"
        if dataset == "FisheyeNeRF"
        else "Masked spatial VGG v0.1, RGB0.5 outside mask; saved uint8 render",
    )
    write_json(Path(output) / "metrics.json", result)
    return result
