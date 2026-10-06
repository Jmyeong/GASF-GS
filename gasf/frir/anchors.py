"""Training-only RootSIFT + COLMAP track anchors, with fixed calibration."""

import fcntl
import hashlib
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from gasf.data import read_model, split_names
from gasf.frir.geometry import rays, triangulate
from gasf.io import digest, load_json, write_json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def camera_data(source):
    source = Path(source)
    intrinsics, images = read_model(source)
    names = set(split_names(source, "train"))
    cameras = []
    for image in images.values():
        if image.name not in names:
            continue
        intrinsic = intrinsics[image.camera_id]
        if intrinsic.model != "OPENCV_FISHEYE":
            raise ValueError("FRIR requires OPENCV_FISHEYE calibration")
        with Image.open(source / "images" / image.name) as raw:
            rgb = np.asarray(raw.convert("RGB")).copy()
        h, w = rgb.shape[:2]
        scale = np.array([w / intrinsic.width, h / intrinsic.height])
        params = np.array(intrinsic.params).copy()
        params[:4] *= [scale[0], scale[1], scale[0], scale[1]]
        rotation = image.qvec2rotmat()
        mask_path = source / "masks" / (Path(image.name).stem + ".png")
        mask = None
        if mask_path.exists():
            with Image.open(mask_path) as raw:
                mask = np.asarray(raw.convert("L")).copy()
            if mask.shape != (h, w):
                raise ValueError("Mask/image shape mismatch")
            uv = np.rint(image.xys * scale).astype(int)
            valid = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            valid[valid] &= mask[uv[valid, 1], uv[valid, 0]] > 0
            ids = image.point3D_ids.copy()
            ids[~valid] = -1
            image = image._replace(point3D_ids=ids)
        cameras.append(
            dict(
                name=image.name,
                R=rotation,
                t=image.tvec,
                C=-rotation.T @ image.tvec,
                params=params,
                rgb=rgb,
                im=image,
                scale=scale,
                mask=mask,
                image_hash=digest(source / "images" / image.name),
                mask_hash=digest(mask_path) if mask_path.exists() else None,
            )
        )
    if len(cameras) != len(names):
        raise ValueError("Training images missing COLMAP poses")
    return sorted(cameras, key=lambda c: c["name"])


def features(camera, cache):
    tag = fingerprint(
        dict(
            image=camera["image_hash"],
            mask=camera["mask_hash"],
            recipe="RootSIFT12000 contrast.01 edge12",
            opencv=cv2.__version__,
        )
    )
    path = cache / "features" / (tag + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not path.exists():
            detector = cv2.SIFT_create(
                nfeatures=12000, contrastThreshold=0.01, edgeThreshold=12
            )
            keypoints, desc = detector.detectAndCompute(
                cv2.cvtColor(camera["rgb"], cv2.COLOR_RGB2GRAY), camera["mask"]
            )
            uv = np.array([k.pt for k in keypoints]).reshape(-1, 2)
            desc = (
                np.empty((0, 128), np.float32)
                if desc is None
                else np.sqrt(
                    desc / np.maximum(desc.sum(1, keepdims=True), 1e-12)
                ).astype(np.float32)
            )
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, uv=uv, desc=desc)
            temporary.replace(path)
        with np.load(path) as data:
            return data["uv"], data["desc"], tag


def matched_pair(c0, c1, f0, f1, cache):
    calibration = [{k: c[k].tolist() for k in ["R", "t", "params"]} for c in [c0, c1]]
    tag = fingerprint(
        dict(
            features=[f0[2], f1[2]],
            calibration=calibration,
            triangulator=digest(Path(__file__).with_name("geometry.py")),
            recipe="mutual_ratio.8",
        )
    )
    path = cache / "pairs" / (tag + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            return load_json(path)
        u, d, _ = f0
        v, e, _ = f1
        accepted = []
        matches = []
        if len(d) >= 2 and len(e) >= 2:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            forward = matcher.knnMatch(d, e, k=2)
            reverse = matcher.knnMatch(e, d, k=2)
            mutual = {
                a.queryIdx: a.trainIdx
                for a, b in reverse
                if a.distance < 0.8 * b.distance
            }
            matches = [
                a
                for a, b in forward
                if a.distance < 0.8 * b.distance
                and mutual.get(a.trainIdx) == a.queryIdx
            ]
            for match in matches:
                anchor = triangulate(c0, c1, u[match.queryIdx], v[match.trainIdx])
                if anchor is not None:
                    accepted.append(anchor)
        result = dict(mutual_matches=len(matches), verified=accepted)
        write_json(path, result)
        return result


def deduplicate(existing, candidates):
    grid = {}
    combined = list(existing)
    added = []

    def insert(anchor):
        for camera, uv in zip(anchor["cameras"], anchor["uv"]):
            grid.setdefault(
                (camera, math.floor(uv[0] / 3), math.floor(uv[1] / 3)), []
            ).append(uv)

    for anchor in existing:
        insert(anchor)
    for anchor in sorted(candidates, key=lambda a: (a["error"], -a["angle"])):
        duplicate = False
        for camera, uv in zip(anchor["cameras"], anchor["uv"]):
            x, y = math.floor(uv[0] / 3), math.floor(uv[1] / 3)
            if any(
                np.linalg.norm(np.array(uv) - p) < 3
                for dx in [-1, 0, 1]
                for dy in [-1, 0, 1]
                for p in grid.get((camera, x + dx, y + dy), [])
            ):
                duplicate = True
                break
        if not duplicate:
            combined.append(anchor)
            added.append(anchor)
            insert(anchor)
    return combined, added


def validate_anchors(data, source):
    source = Path(source)
    if sorted(data["training_images"]) != sorted(split_names(source, "train")):
        raise ValueError("Anchor/train split mismatch")
    for name, expected in data.get("input_hashes", {}).items():
        if digest(source / name) != expected:
            raise ValueError(f"Anchor input changed: {name}")
    for name, expected in data.get("mask_hashes", {}).items():
        if digest(source / "masks" / (Path(name).stem + ".png")) != expected:
            raise ValueError(f"Anchor mask changed: {name}")


def prepare_anchors(source, output, cache):
    source = Path(source)
    output = Path(output)
    cache = Path(cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if output.exists():
            data = load_json(output)
            validate_anchors(data, source)
            return data
        cv2.setNumThreads(4)
        cv2.setRNGSeed(49)
        cameras = camera_data(source)
        descriptors = [features(c, cache) for c in cameras]
        candidates = []
        pairs = []
        for i, j in itertools.combinations(range(len(cameras)), 2):
            match = matched_pair(
                cameras[i], cameras[j], descriptors[i], descriptors[j], cache
            )
            candidates.extend(
                dict(a, cameras=[i, j], origin="rootsift") for a in match["verified"]
            )
            pairs.append(
                dict(
                    cameras=[cameras[i]["name"], cameras[j]["name"]],
                    mutual_matches=match["mutual_matches"],
                    verified=len(match["verified"]),
                )
            )
        tracks = {}
        for i, c in enumerate(cameras):
            for uv, pid in zip(c["im"].xys, c["im"].point3D_ids):
                if pid >= 0:
                    tracks.setdefault(int(pid), []).append((i, uv * c["scale"]))
        existing = []
        for pid, observations in tracks.items():
            directions = [rays(uv[None], cameras[i])[0] for i, uv in observations]
            ranked = []
            for serial, ((a, (i, u)), (b, (j, v))) in enumerate(
                itertools.combinations(enumerate(observations), 2)
            ):
                angle = float(
                    np.degrees(np.arccos(np.clip(directions[a] @ directions[b], -1, 1)))
                )
                if 2 <= angle <= 175:
                    ranked.append((-angle, serial, i, u, j, v))
            for _, _, i, u, j, v in sorted(ranked, key=lambda x: (x[0], x[1])):
                anchor = triangulate(cameras[i], cameras[j], u, v)
                if anchor is not None:
                    existing.append(
                        dict(anchor, cameras=[i, j], origin="colmap", point_id=pid)
                    )
                    break
        anchors, added = deduplicate(existing, candidates)
        model_files = [
            str(p.relative_to(source))
            for p in [source / "sparse/0/cameras.bin", source / "sparse/0/images.bin"]
            if p.exists()
        ]
        if not model_files:
            model_files = ["sparse/0/cameras.txt", "sparse/0/images.txt"]
        hashes = {
            name: digest(source / name)
            for name in ["train.txt"]
            + model_files
            + [f"images/{c['name']}" for c in cameras]
        }
        data = dict(
            training_images=[c["name"] for c in cameras],
            image_shapes=[list(c["rgb"].shape[:2]) for c in cameras],
            anchors=anchors,
            pairs=pairs,
            colmap_anchors=len(existing),
            new_anchors=len(added),
            input_hashes=hashes,
            mask_hashes={
                c["name"]: c["mask_hash"] for c in cameras if c["mask_hash"] is not None
            },
            localization_sigma_px=1.0,
            max_reprojection_px=1.0,
            min_parallax_degrees=2.0,
            conditional_uncertainty="Fixed calibration and correct correspondence; not camera/matching uncertainty",
        )
        write_json(output, data)
        return data
