"""Fixed-calibration fisheye triangulation and conditional uncertainty."""

import numpy as np


def project(X, c):
    a = np.asarray(X) @ c["R"].T + c["t"]
    xy = a[..., :2]
    r = np.linalg.norm(xy, axis=-1)
    theta = np.arctan2(r, a[..., 2])
    p = c["params"]
    rho = theta * (1 + sum(p[4 + j] * theta ** (2 * j + 2) for j in range(4)))
    return xy * (rho / np.maximum(r, 1e-12))[..., None] * p[:2] + p[2:4]


def rays(uv, c):
    p = c["params"]
    xy = (np.asarray(uv) - p[2:4]) / p[:2]
    rd = np.linalg.norm(xy, axis=-1)
    theta = rd.copy()
    for _ in range(20):
        rho = theta * (1 + sum(p[4 + j] * theta ** (2 * j + 2) for j in range(4)))
        slope = 1 + sum((2 * j + 3) * p[4 + j] * theta ** (2 * j + 2) for j in range(4))
        theta -= (rho - rd) / np.maximum(slope, 1e-8)
    d = np.concatenate(
        (
            xy * (np.sin(theta) / np.maximum(rd, 1e-12))[..., None],
            np.cos(theta)[..., None],
        ),
        axis=-1,
    )
    return d @ c["R"]


def triangulate(c0, c1, u0, u1):
    d0 = rays(u0[None], c0)[0]
    d1 = rays(u1[None], c1)[0]
    angle = np.degrees(np.arccos(np.clip(d0 @ d1, -1, 1)))
    if not 2 <= angle <= 175:
        return None
    A = np.stack((d0, -d1), axis=1)
    rr = np.linalg.lstsq(A, c1["C"] - c0["C"], rcond=None)[0]
    if (rr <= 0).any():
        return None
    x = (c0["C"] + rr[0] * d0 + c1["C"] + rr[1] * d1) / 2

    def residual(y):
        return np.r_[project(y, c0) - u0, project(y, c1) - u1]

    if np.max(abs(residual(x))) > 4:
        return None
    eps = max(np.linalg.norm(c0["C"] - c1["C"]) * 1e-5, 1e-7)

    def jacobian(y):
        return np.stack(
            [
                (residual(y + np.eye(3)[k] * eps) - residual(y - np.eye(3)[k] * eps))
                / (2 * eps)
                for k in range(3)
            ],
            axis=1,
        )

    for _ in range(20):
        err = residual(x)
        J = jacobian(x)
        step = np.linalg.lstsq(J, -err, rcond=None)[0]
        if np.linalg.norm(step) < eps * 0.01:
            break
        accepted = False
        for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            y = x + scale * step
            if np.square(residual(y)).sum() < np.square(err).sum():
                x = y
                accepted = True
                break
        if not accepted:
            break
    error = residual(x)
    if (
        not np.isfinite(x).all()
        or np.linalg.norm(error.reshape(2, 2), axis=1).max() > 1
    ):
        return None
    J = jacobian(x)
    H = J.T @ J
    ev = np.linalg.eigvalsh(H)
    if ev[0] <= 0 or ev[-1] / ev[0] > 1e6:
        return None
    # One-pixel localization uncertainty including resampling/model mismatch.
    cov = np.linalg.inv(H)
    r = np.array([np.linalg.norm(x - c["C"]) for c in (c0, c1)])
    sig = np.sqrt([d @ cov @ d for d in (d0, d1)])
    if np.max(sig / r) > 0.1:
        return None
    if any((x - c["C"]) @ d <= 0 for c, d in [(c0, d0), (c1, d1)]):
        return None
    return dict(
        xyz=x.tolist(),
        sigma=sig.tolist(),
        range=r.tolist(),
        angle=float(angle),
        error=float(np.linalg.norm(error.reshape(2, 2), axis=1).max()),
        uv=[u0.tolist(), u1.tolist()],
    )
