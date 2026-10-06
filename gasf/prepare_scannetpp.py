"""Nested 3/12/24-view ScanNet++ inputs, using training-only COLMAP tracks."""

import itertools
import json
import os
import sqlite3
import struct
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from gasf.io import write_json as dump
from gasf.utils import colmap as cm


def poses_only(path):
    result = {}
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            result[p[9]] = cm.Image(
                id=int(p[0]),
                qvec=np.array(p[1:5], float),
                tvec=np.array(p[5:8], float),
                camera_id=int(p[8]),
                name=p[9],
                xys=np.empty((0, 2)),
                point3D_ids=np.empty(0, np.int64),
            )
            next(f)  # Discard all full-scene feature observations/point IDs.
    return result


def select_nested(names, poses):
    """Pose-only farthest point sampling; nested 3/12/24, no test information."""
    names = sorted(names)
    centers = np.array([-poses[n].qvec2rotmat().T @ poses[n].tvec for n in names])
    axes = np.array([poses[n].qvec2rotmat()[2] for n in names])
    center = np.median(centers, axis=0)
    scale = max(float(np.median(np.linalg.norm(centers - center, axis=1))), 1e-6)
    features = np.column_stack(((centers - center) / scale, axes))
    selected = [int(np.argmin(np.linalg.norm(centers - center, axis=1)))]
    distance = np.full(len(names), np.inf)
    for _ in range(23):
        distance = np.minimum(
            distance, np.sum((features - features[selected[-1]]) ** 2, axis=1)
        )
        distance[selected] = -np.inf
        selected.append(int(np.argmax(distance)))
    return [names[i] for i in selected]


def link(source, dest):
    if dest.is_symlink():
        assert dest.resolve() == source.resolve()
    elif dest.exists():
        raise RuntimeError(f"Refusing to replace {dest}")
    else:
        dest.symlink_to(source, target_is_directory=source.is_dir())


def command(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    print("COLMAP", args[0], log.stem, flush=True)
    with log.open("w") as f:
        result = subprocess.run(
            ["colmap", *map(str, args)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env={
                **os.environ,
                "QT_QPA_PLATFORM": "offscreen",
                "CUDA_VISIBLE_DEVICES": "",
            },
        )
    if result.returncode:
        raise RuntimeError(f"COLMAP exit {result.returncode}: {log}")


def camera_text(camera):
    return (
        f"1 OPENCV_FISHEYE {camera.width} {camera.height} "
        + " ".join(map(str, camera.params))
        + "\n"
    )


def pose_text(im, image_id):
    return " ".join(map(str, [image_id, *im.qvec, *im.tvec, 1, im.name])) + "\n\n"


def write_images_binary(path, images):
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images:
            f.write(struct.pack("<i7di", im.id, *im.qvec, *im.tvec, im.camera_id))
            f.write(im.name.encode() + b"\0")
            f.write(struct.pack("<Q", len(im.xys)))
            for uv, pid in zip(im.xys, im.point3D_ids):
                f.write(struct.pack("<ddq", *uv, int(pid)))


def montage(names, images, dest, selected=None):
    selected = selected or []
    canvas = Image.new("RGB", (4 * 320, ((len(names) + 3) // 4) * 240), "white")
    draw = ImageDraw.Draw(canvas)
    for i, name in enumerate(names):
        with Image.open(images / name) as im:
            thumb = im.convert("RGB")
            thumb.thumbnail((320, 210))
        x, y = (i % 4) * 320, (i // 4) * 240
        canvas.paste(thumb, (x, y))
        tag = (
            "3/12/24"
            if name in selected[:3]
            else ("12/24" if name in selected[:12] else "24")
        )
        draw.text(
            (x + 3, y + 212),
            name + ("  views: " + tag if selected else "  TEST"),
            fill="black",
        )
    canvas.save(dest)


def prepare(scene, raw_root, prepared_root, work_root):
    started = time.monotonic()
    raw = Path(raw_root) / scene / "dslr"
    work = Path(work_root) / scene
    work.mkdir(parents=True, exist_ok=True)
    split = json.loads((raw / "train_test_lists.json").read_text())
    transforms = json.loads((raw / "nerfstudio/transforms.json").read_text())
    poses = poses_only(raw / "colmap/images.txt")
    cameras = cm.read_intrinsics_text(str(raw / "colmap/cameras.txt"))
    assert len(cameras) == 1
    camera = next(iter(cameras.values()))
    assert camera.model == "OPENCV_FISHEYE" and len(camera.params) == 8
    bad = {
        Path(f["file_path"]).name
        for f in transforms["frames"]
        if f.get("is_bad", False)
    }
    train = [n for n in split["train"] if n not in bad and n in poses]
    test = sorted(split["test"])
    assert len(train) >= 24 and test and not set(train) & set(test)
    selected = select_nested(train, poses)
    for name in selected + test:
        assert name in poses
        with Image.open(raw / "resized_images" / name) as im:
            assert im.size == (camera.width, camera.height), (scene, name, im.size)
            im.verify()
    mask_dir = work / "feature_masks"
    mask_dir.mkdir(exist_ok=True)
    valid_fraction = {}
    for name in selected + test:
        mask = raw / "resized_anon_masks" / (Path(name).stem + ".png")
        if split["has_masks"]:
            assert mask.exists(), mask
            with Image.open(mask) as im:
                assert im.size == (camera.width, camera.height)
                valid_fraction[name] = float((np.asarray(im.convert("L")) > 0).mean())
            if name in selected:
                link(mask, mask_dir / (name + ".png"))
        else:
            valid_fraction[name] = 1.0
    selection = dict(
        scene=scene,
        candidate_train_order=selected,
        test=test,
        excluded_bad_train=len(bad),
        selection="24 pose-only FPS candidates; strongest minimum-edge verified-match triangle for 3 views; remaining FPS order for 12/24",
        image_size=[camera.width, camera.height],
        camera_params=camera.params.tolist(),
        masks_required=bool(split["has_masks"]),
        valid_pixel_fraction=valid_fraction,
    )
    (work / "train24.txt").write_text("\n".join(selected) + "\n")
    database = work / "features.db"
    if not (work / "MATCHING_COMPLETE").exists():
        args = [
            "feature_extractor",
            "--database_path",
            database,
            "--image_path",
            raw / "resized_images",
            "--image_list_path",
            work / "train24.txt",
            "--ImageReader.camera_model",
            "OPENCV_FISHEYE",
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            ",".join(map(str, camera.params)),
            "--SiftExtraction.use_gpu",
            "0",
            "--SiftExtraction.num_threads",
            "4",
        ]
        if split["has_masks"]:
            args += ["--ImageReader.mask_path", mask_dir]
        command(args, work / "feature_extractor.log")
        command(
            [
                "exhaustive_matcher",
                "--database_path",
                database,
                "--SiftMatching.use_gpu",
                "0",
                "--SiftMatching.num_threads",
                "4",
            ],
            work / "exhaustive_matcher.log",
        )
        (work / "MATCHING_COMPLETE").write_text("completed\n")
    with sqlite3.connect(database) as db:
        ids = {
            name: ident
            for ident, name in db.execute("SELECT image_id,name FROM images")
        }
        assert set(ids) == set(selected), (
            "Feature DB must contain only selected 24 training images."
        )
        edges = {
            tuple(divmod(k, 2147483647)): count
            for k, count in db.execute("SELECT pair_id,rows FROM two_view_geometries")
        }
    candidates = []
    for triple in itertools.combinations(sorted(selected), 3):
        counts = [
            edges.get(tuple(sorted((ids[a], ids[b]))), 0)
            for a, b in itertools.combinations(triple, 2)
        ]
        if min(counts) >= 15:
            candidates.append((min(counts), sum(counts), triple))
    if not candidates:
        raise RuntimeError(
            f"{scene}: no connected 3-view triangle in 24 training candidates."
        )
    minimum, total, triple = max(candidates)
    selected = list(triple) + [n for n in selected if n not in triple]
    selection.update(
        train_order=selected,
        seed_triplet_min_pair_matches=minimum,
        seed_triplet_total_pair_matches=total,
        selection_scope="RGB matching of 24 candidate TRAIN views used for view selection; no test RGB or full-scene points used. Each final cloud triangulated from its selected views only.",
    )
    dump(work / "selection.json", selection)
    montage(selected, raw / "resized_images", work / "train_3_12_24.jpg", selected)
    montage(test, raw / "resized_images", work / "test.jpg")
    reports = []
    for views in [3, 12, 24]:
        names = selected[:views]
        dest = Path(prepared_root) / f"{scene}_{views}"
        model = dest / "sparse/0"
        model.mkdir(parents=True, exist_ok=True)
        input_model = work / f"input_{views}"
        input_model.mkdir(exist_ok=True)
        (input_model / "cameras.txt").write_text(camera_text(camera))
        (input_model / "images.txt").write_text(
            "".join(pose_text(poses[n], ids[n]) for n in names)
        )
        (input_model / "points3D.txt").write_text("")
        if not (dest / "prepared.json").exists():
            command(
                [
                    "point_triangulator",
                    "--database_path",
                    database,
                    "--image_path",
                    raw / "resized_images",
                    "--input_path",
                    input_model,
                    "--output_path",
                    model,
                    "--clear_points",
                    "1",
                    "--Mapper.fix_existing_images",
                    "1",
                    "--Mapper.num_threads",
                    "4",
                    "--Mapper.ba_refine_focal_length",
                    "0",
                    "--Mapper.ba_refine_principal_point",
                    "0",
                    "--Mapper.ba_refine_extra_params",
                    "0",
                    "--Mapper.tri_ignore_two_view_tracks",
                    "0",
                ],
                work / f"triangulate_{views}.log",
            )
            output_images = cm.read_extrinsics_binary(str(model / "images.bin"))
            assert {x.name for x in output_images.values()} == set(names)
            for im in output_images.values():
                np.testing.assert_allclose(
                    im.qvec2rotmat(), poses[im.name].qvec2rotmat(), atol=1e-8
                )
                np.testing.assert_allclose(im.tvec, poses[im.name].tvec, atol=1e-8)
            for cam in cm.read_intrinsics_binary(str(model / "cameras.bin")).values():
                np.testing.assert_allclose(cam.params, camera.params, atol=1e-8)
            xyz, rgb, errors = cm.read_points3D_binary(str(model / "points3D.bin"))
            if not len(xyz):
                raise RuntimeError(
                    f"{scene}_{views}: zero triangulated training points; not trainable."
                )
            assert np.isfinite(xyz).all()
            # Append held-out camera poses with zero 2D observations after triangulation.
            extra = [
                poses[n]._replace(id=max(ids.values()) + 1 + i, camera_id=1)
                for i, n in enumerate(test)
            ]
            write_images_binary(
                model / "images.bin", list(output_images.values()) + extra
            )
            link(raw / "resized_images", dest / "images")
            if split["has_masks"]:
                link(raw / "resized_anon_masks", dest / "masks")
            (dest / "train.txt").write_text("\n".join(sorted(names)) + "\n")
            (dest / "test.txt").write_text("\n".join(test) + "\n")
            report = dict(
                scene=scene,
                views=views,
                source=str(dest),
                test_views=len(test),
                points=len(xyz),
                median_reprojection_error=float(np.median(errors)),
                masks_required=bool(split["has_masks"]),
                camera_model="OPENCV_FISHEYE",
                selection_protocol=selection["selection_scope"],
                calibration="provided full-scene fixed calibration; not sparse pose estimation",
                initialization="COLMAP triangulated from selected training views only; no full-scene point reuse",
                masked_training_required=True,
                training_started=False,
            )
            dump(dest / "prepared.json", report)
        reports.append(json.loads((dest / "prepared.json").read_text()))
        print(
            "PREPARED",
            scene,
            views,
            "views;",
            reports[-1]["points"],
            "points",
            flush=True,
        )
    dump(
        work / "completed.json", dict(seconds=time.monotonic() - started, cases=reports)
    )
    return reports
