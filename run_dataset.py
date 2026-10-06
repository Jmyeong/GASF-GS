#!/usr/bin/env python3
"""Distribute prepared scene jobs over the explicitly selected GPUs."""

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOTS = {
    "FisheyeNeRF": ROOT.parent / "dataset" / "FisheyeNeRF",
    "ScanNetpp": ROOT.parent / "dataset" / "ScanNet++" / "benchmark" / "ours",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["FisheyeNeRF", "ScanNetpp"], required=True
    )
    parser.add_argument("--views", type=int, choices=[3, 12, 24], required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        help="Parent of <scene>_<views> prepared folders; default: the sibling dataset/ directory",
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "output")
    parser.add_argument(
        "--scenes",
        nargs="+",
        help="Scene names without the _views suffix; default: discover all",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["0"],
        help="Physical CUDA indices; one scene per GPU",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument(
        "--anchors-root",
        type=Path,
        help="Optional parent of <scene>_<views>/anchors.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)) or any(not g.isdigit() for g in args.gpus):
        parser.error("Supply distinct nonnegative GPU indices")
    data = (args.data_root or DEFAULT_DATA_ROOTS[args.dataset]).resolve()
    scenes = args.scenes
    if not scenes:
        scenes = sorted(
            p.name.rsplit("_", 1)[0]
            for p in data.glob("*_" + str(args.views))
            if (p / "train.txt").is_file()
        )
        if args.dataset == "FisheyeNeRF":
            scenes = [
                s
                for s in scenes
                if s in ["chairs", "cube", "flowers", "globe", "heart", "rock"]
            ]
    if not scenes:
        parser.error("No prepared scenes found; provide --data-root and/or --scenes")
    if len(scenes) != len(set(scenes)):
        parser.error("Duplicate scene names")
    commands = []
    for name in scenes:
        if Path(name).name != name:
            parser.error("Scene names must not contain path components")
        source = data / (name + "_" + str(args.views))
        if not source.is_dir():
            parser.error(f"Missing input: {source}")
        output = (
            args.output_root.resolve()
            / args.dataset
            / (name + "_" + str(args.views))
            / f"seed_{args.seed}"
        )
        command = [
            sys.executable,
            str(ROOT / "train.py"),
            "--dataset",
            args.dataset,
            "--views",
            str(args.views),
            "--source",
            str(source),
            "--output",
            str(output),
            "--iterations",
            str(args.iterations),
            "--seed",
            str(args.seed),
            "--cache",
            str(ROOT / ".cache/anchors"),
        ]
        if args.anchors_root:
            command += [
                "--anchors",
                str(
                    args.anchors_root.resolve()
                    / (name + "_" + str(args.views))
                    / "anchors.json"
                ),
            ]
        if args.skip_lpips:
            command += ["--skip-lpips"]
        commands.append((name, output, command))
    if args.dry_run:
        import shlex

        for i, (name, output, command) in enumerate(commands):
            print(f"GPU {args.gpus[i % len(args.gpus)]} | {name} | {output}")
            print(shlex.join(command))
        return
    stopped = threading.Event()
    pending = queue.Queue()
    errors = []
    mutex = threading.Lock()
    for job in commands:
        pending.put(job)
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())

    def worker(gpu):
        while not stopped.is_set():
            try:
                name, output, command = pending.get_nowait()
            except queue.Empty:
                return
            # Keep logs outside a run directory so partial-run detection is reliable.
            log = output.parent / (output.name + ".log")
            log.parent.mkdir(parents=True, exist_ok=True)
            print(f"START {name}, GPU {gpu}, {args.views} views", flush=True)
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                PYTHONDONTWRITEBYTECODE="1",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
                OPENBLAS_NUM_THREADS="4",
            )
            with log.open("a") as stream:
                proc = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                while proc.poll() is None:
                    if stopped.wait(0.5):
                        os.killpg(proc.pid, signal.SIGTERM)
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            proc.wait()
                        break
                rc = proc.wait()
            with mutex:
                if rc:
                    errors.append((name, rc, str(log)))
            print(f"{'DONE' if rc == 0 else 'FAILED'} {name}: {log}", flush=True)

    with ThreadPoolExecutor(len(args.gpus)) as pool:
        futures = [pool.submit(worker, g) for g in args.gpus]
        for f in futures:
            f.result()
    if stopped.is_set():
        raise SystemExit(130)
    if errors:
        for name, rc, log in errors:
            print(f"{name}: exit {rc}, {log}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
