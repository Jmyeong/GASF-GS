"""Read the renderer buffers and replay only supervised rays."""

import numpy as np
import torch


def tracer():
    from gasf_ray_trace import _C

    return _C


def take(buffer, offset, count, dtype):
    offset = (offset + 127) // 128 * 128
    size = torch.empty((), dtype=dtype).element_size() * count
    if offset + size > buffer.numel():
        raise RuntimeError("Raster buffer ABI mismatch")
    return buffer[offset : offset + size].view(dtype), offset + size


def unpack(image):
    node = image.grad_fn
    if node is None or not hasattr(node, "raster_settings"):
        raise RuntimeError("Expected the native raster autograd output")
    saved = node.saved_tensors
    colors, xyz = saved[0], saved[1]
    geom, binning, img = saved[-3:]
    n = len(xyz)
    offset = 0
    fields = []
    for count, dtype in [
        (n, torch.float32),
        (3 * n, torch.uint8),
        (n, torch.int32),
        (2 * n, torch.float32),
        (6 * n, torch.float32),
        (4 * n, torch.float32),
        (3 * n, torch.float32),
    ]:
        field, offset = take(geom, offset, count, dtype)
        fields.append(field)
    cov = fields[4].reshape(n, 6)
    opacity = fields[5].reshape(n, 4)[:, 3]
    rgb = colors if colors.numel() else fields[6].reshape(n, 3)
    point_list, _ = take(binning, 0, node.num_rendered, torch.int32)
    h, w = image.shape[-2:]
    offset = 0
    _, offset = take(img, offset, h * w, torch.float32)
    _, offset = take(img, offset, h * w, torch.int32)
    ranges, _ = take(img, offset, 2 * h * w, torch.int32)
    tiles = ((h + 15) // 16) * ((w + 15) // 16)
    return dict(
        xyz=xyz.detach(),
        cov=cov.detach(),
        opacity=opacity.detach(),
        rgb=rgb.detach(),
        ids=point_list,
        ranges=ranges.reshape(-1, 2)[:tiles],
        settings=node.raster_settings,
    )


def native_rays(uv, params, view):
    p = np.asarray(params, dtype=np.float32)
    xy = (np.asarray(uv, dtype=np.float32) - p[2:4]) / p[:2]
    td = np.sqrt(np.sum(xy * xy, axis=-1))
    t = td.copy()
    for _ in range(8):
        t2 = t * t
        t4 = t2 * t2
        t6 = t4 * t2
        t8 = t4 * t4
        value = t * (1 + p[4] * t2 + p[5] * t4 + p[6] * t6 + p[7] * t8) - td
        derivative = 1 + 3 * p[4] * t2 + 5 * p[5] * t4 + 7 * p[6] * t6 + 9 * p[7] * t8
        t -= value / np.maximum(np.abs(derivative), 1e-8)
    d = np.concatenate(
        (xy * (np.sin(t) / np.maximum(td, 1e-12))[:, None], np.cos(t)[:, None]), axis=1
    )
    d[td <= 1e-7] = [0, 0, 1]
    world = d @ view[:3, :3].T
    return world / np.sqrt(np.sum(world * world, axis=-1, keepdims=True) + 1e-12)
