"""Individual response-peak interval loss, using the native hit order."""

import torch

from gasf.frir.trace import native_rays, tracer, unpack
from gasf.utils.math import build_rotation


def response(model, ids, origin, directions):
    idx = ids.clamp_min(0).long()
    R = build_rotation(model._rotation)
    # Stable whitening avoids explicit covariance inversion/cancellation.
    scales = model.get_scaling.clamp_min(1e-10)
    v = model.get_xyz[idx] - origin
    rotation = R[idx]
    a = torch.einsum("qmi,qmij->qmj", v, rotation) / scales[idx]
    b = torch.einsum("qi,qmij->qmj", directions, rotation) / scales[idx]
    denom = b.square().sum(-1).clamp_min(1e-12)
    tau = (a * b).sum(-1) / denom
    perpendicular = a - tau[..., None] * b
    alpha = (
        model.get_opacity[idx, 0] * torch.exp(-0.5 * perpendicular.square().sum(-1))
    ).clamp(max=0.99)
    return tau, torch.where(ids >= 0, alpha, torch.zeros_like(alpha))


def ray_objective(tau, alpha, target, tolerance, mass_weight=0.01, delta=0.1):
    T = torch.cumprod(
        torch.cat((torch.ones_like(alpha[:, :1]), 1 - alpha), dim=1), dim=1
    )
    # Native renderer does not blend the hit crossing early-termination 1e-4.
    keep = T[:, 1:].detach() >= 1e-4
    w = T[:, :-1] * alpha * keep
    mass = w.sum(-1)
    ratio = tau / target[:, None].clamp_min(1e-6)
    excess = (abs(ratio - 1) - tolerance[:, None]).clamp_min(0)
    penalty = delta**2 * (torch.sqrt(1 + (excess / delta).square()) - 1)
    geometric = (w * penalty).sum(-1) / mass.clamp_min(0.05)
    return geometric + mass_weight * (1 - mass).square(), mass, geometric


def guided_loss(
    model, camera, image, uv, target, tolerance, mass_weight=0.01, delta=0.1
):
    state = unpack(image)
    h, w = image.shape[-2:]
    uv_gpu = uv.to(device=image.device, dtype=torch.int32).contiguous()
    tiles = (uv_gpu[:, 1] // 16) * ((w + 15) // 16) + uv_gpu[:, 0] // 16
    selected = state["ranges"][tiles.long()]
    capacity = max(1, int((selected[:, 1] - selected[:, 0]).max()))
    with torch.no_grad():
        ids, alpha_native, counts, replayed = tracer().trace(
            *[
                state[key].contiguous()
                for key in ["xyz", "cov", "opacity", "rgb", "ids", "ranges"]
            ],
            uv_gpu,
            camera.world_view_transform.contiguous(),
            camera.fisheye_params.to(image).contiguous(),
            camera.camera_center.contiguous(),
            state["settings"].bg.contiguous(),
            w,
            h,
            state["settings"].tau_k_buffer_size,
            capacity,
        )
        assert not (counts > capacity).any()
        # Drop padded columns early; all contributing native IDs remain.
        m = max(1, int(counts.max()))
        ids = ids[:, :m].long()
        alpha_native = alpha_native[:, :m]
        native_rgb = image.detach()[:, uv_gpu[:, 1].long(), uv_gpu[:, 0].long()].T
        rgb_error = (replayed - native_rgb).abs()
        trusted = (rgb_error.max(-1).values <= 0.002) & (rgb_error.mean(-1) <= 0.0002)
    directions = torch.as_tensor(
        native_rays(
            uv.cpu().numpy(),
            camera.fisheye_params.detach().cpu().numpy(),
            camera.world_view_transform.detach().cpu().numpy(),
        ),
        device=image.device,
    )
    tau, alpha = response(model, ids, camera.camera_center, directions)
    with torch.no_grad():
        alpha_error = (alpha - alpha_native).abs().max(-1).values
        trusted &= (alpha_error < 0.002) & torch.isfinite(tau).all(-1) & (counts > 0)
    # Forward uses authoritative native alpha; gradients use the same analytic
    # peak response. Rays with numerical disagreement are rejected, not trusted.
    alpha = alpha + (alpha_native - alpha).detach()
    losses, mass, geo = ray_objective(tau, alpha, target, tolerance, mass_weight, delta)
    if not trusted.any():
        return model._xyz.sum() * 0, dict(
            trusted=0,
            rays=len(uv),
            rgb_max=float(rgb_error.max()),
            alpha_max=float(alpha_error.max()),
        )
    confidence = (0.01 / tolerance.clamp_min(0.01)).detach()
    weights = confidence * trusted
    loss = (losses * weights).sum() / weights.sum().clamp_min(1e-8)
    return loss, dict(
        trusted=int(trusted.sum()),
        rays=len(uv),
        rgb_max=float(rgb_error.max()),
        alpha_max=float(alpha_error.max()),
        trusted_rgb_max=float(rgb_error[trusted].max()),
        trusted_alpha_max=float(alpha_error[trusted].max()),
        mean_mass=float(mass[trusted].detach().mean()),
        mean_geometric=float(geo[trusted].detach().mean()),
        loss=float(loss.detach()),
    )
