"""Prepared native-fisheye COLMAP scenes with explicit train/test lists."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from gasf.utils.colmap import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
    read_points3D_binary,
    read_points3D_text,
)
from gasf.utils.graphics import (
    BasicPointCloud,
    focal2fov,
    getProjectionMatrix,
    getWorld2View2,
)


def read_model(source):
    model = Path(source) / "sparse/0"
    if (model / "cameras.bin").exists():
        return read_intrinsics_binary(
            str(model / "cameras.bin")
        ), read_extrinsics_binary(str(model / "images.bin"))
    return read_intrinsics_text(str(model / "cameras.txt")), read_extrinsics_text(
        str(model / "images.txt")
    )


def split_names(source, split):
    for path in [
        Path(source) / (split + ".txt"),
        Path(source) / "sparse/0" / (split + ".txt"),
    ]:
        if path.exists():
            names = [n.strip() for n in path.read_text().splitlines() if n.strip()]
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate image in {path}")
            return names
    raise FileNotFoundError(f"{source}: explicit {split}.txt required")


def initial_points(source):
    model = Path(source) / "sparse/0"
    if (model / "points3D.ply").exists():
        vertices = PlyData.read(model / "points3D.ply")["vertex"]
        xyz = np.column_stack([vertices[n] for n in ["x", "y", "z"]])
        rgb = np.column_stack([vertices[n] for n in ["red", "green", "blue"]]) / 255.0
    elif (model / "points3D.bin").exists():
        xyz, rgb, _ = read_points3D_binary(str(model / "points3D.bin"))
        rgb = rgb / 255.0
    else:
        xyz, rgb, _ = read_points3D_text(str(model / "points3D.txt"))
        rgb = rgb / 255.0
    if not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Initialization must contain finite COLMAP points")
    return BasicPointCloud(xyz, rgb, np.zeros_like(xyz))


class Camera:
    def __init__(self, image, intrinsic, source):
        self.image_name = Path(image.name).stem
        self.source_name = image.name
        self.uid = image.id
        self.R = qvec2rotmat(image.qvec).T
        self.T = np.array(image.tvec)
        path = Path(source) / "images" / image.name
        with Image.open(path) as raw:
            rgb = np.asarray(raw.convert("RGB")).copy()
        self.image_height, self.image_width = rgb.shape[:2]
        params = np.asarray(intrinsic.params, dtype=np.float64).copy()
        if intrinsic.model != "OPENCV_FISHEYE" or len(params) != 8:
            raise ValueError(f"{image.name}: expected OPENCV_FISHEYE calibration")
        params[[0, 2]] *= self.image_width / intrinsic.width
        params[[1, 3]] *= self.image_height / intrinsic.height
        self.calibration_params = params.copy()
        self.fisheye_params = torch.as_tensor(params, dtype=torch.float32)
        self.FoVx = focal2fov(intrinsic.params[0], intrinsic.width)
        self.FoVy = focal2fov(intrinsic.params[1], intrinsic.height)
        self.original_image = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mask_path = Path(source) / "masks" / (self.image_name + ".png")
        if mask_path.exists():
            with Image.open(mask_path) as raw_mask:
                mask = np.asarray(raw_mask.convert("L")).copy() > 0
            if mask.shape != (self.image_height, self.image_width):
                raise ValueError(f"{mask_path}: mask/image dimensions differ")
            self.valid_mask = torch.from_numpy(mask)[None]
        else:
            self.valid_mask = torch.ones(
                (1, self.image_height, self.image_width), dtype=torch.bool
            )
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T)
        ).T.cuda()
        self.projection_matrix = getProjectionMatrix(
            0.01, 100.0, self.FoVx, self.FoVy
        ).T.cuda()
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


@dataclass
class Scene:
    source: Path
    train_cameras: list
    test_cameras: list
    point_cloud: BasicPointCloud
    cameras_extent: float

    @classmethod
    def load(cls, source, expected_views=None):
        source = Path(source).resolve()
        intrinsics, images = read_model(source)
        train_names = split_names(source, "train")
        test_names = split_names(source, "test")
        if not train_names or not test_names or set(train_names) & set(test_names):
            raise ValueError("Nonempty, disjoint train/test lists required")
        if expected_views is not None and len(train_names) != expected_views:
            raise ValueError(
                f"Expected {expected_views} train views; found {len(train_names)}"
            )
        by_name = {im.name: im for im in images.values()}
        missing = set(train_names + test_names) - set(by_name)
        if missing:
            raise ValueError(f"Missing COLMAP poses: {sorted(missing)}")

        # Same lexical camera ordering and shuffle as the original Scene loader.
        def cameras(names):
            return [
                Camera(by_name[n], intrinsics[by_name[n].camera_id], source)
                for n in sorted(names, key=lambda n: Path(n).stem)
            ]

        train = cameras(train_names)
        test = cameras(test_names)
        centers = np.hstack(
            [np.linalg.inv(getWorld2View2(c.R, c.T))[:3, 3:4] for c in train]
        )
        extent = float(
            np.linalg.norm(centers - centers.mean(axis=1, keepdims=True), axis=0).max()
            * 1.1
        )
        if extent <= 0:
            raise ValueError("Distinct training camera centers required")
        random.shuffle(train)
        random.shuffle(test)
        return cls(source, train, test, initial_points(source), extent)
