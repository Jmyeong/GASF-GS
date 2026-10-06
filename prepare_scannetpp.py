#!/usr/bin/env python3
"""Prepare 3/12/24-view COLMAP inputs from downloaded native DSLR scenes."""

import argparse
from pathlib import Path

from gasf.io import write_json
from gasf.prepare_scannetpp import prepare

SCANNET_ROOT = Path(__file__).resolve().parent.parent / "dataset" / "ScanNet++"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-root",
        type=Path,
        default=SCANNET_ROOT / "raw" / "data",
        help="Downloaded data/ parent containing <scene>/dslr/",
    )
    p.add_argument("--output-root", type=Path, default=SCANNET_ROOT / "sparse")
    p.add_argument("--scenes", nargs="+", required=True)
    args = p.parse_args()
    if len(args.scenes) != len(set(args.scenes)):
        p.error("Duplicate scene IDs")
    raw = args.raw_root.resolve()
    output = args.output_root.resolve()
    reports = []
    for scene in args.scenes:
        if Path(scene).name != scene:
            p.error("Invalid scene name")
        if not (raw / scene / "dslr").is_dir():
            p.error(f"Missing {raw / scene}/dslr")
        reports.extend(prepare(scene, raw, output, output / ".preparation"))
    write_json(output / "prepared_scenes.json", reports)


if __name__ == "__main__":
    main()
