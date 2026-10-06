#!/usr/bin/env python3
"""Train GASF-GS on one prepared sparse fisheye scene."""

import argparse
import fcntl
import time
from dataclasses import asdict
from pathlib import Path

from gasf.config import FRIRConfig, NACEConfig, OptimizationConfig
from gasf.io import load_json, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        type=Path,
        required=True,
        help="images/, sparse/0/, train.txt and test.txt",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset", choices=["FisheyeNeRF", "ScanNetpp"], required=True)
    p.add_argument("--views", type=int, choices=[3, 12, 24], required=True)
    p.add_argument("--iterations", type=int, default=30_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--anchors", type=Path, help="Optional validated precomputed training anchors"
    )
    p.add_argument("--cache", type=Path, default=Path(".cache/anchors"))
    p.add_argument("--resume", type=Path, help="Checkpoint from this GASF-GS release")
    p.add_argument("--lambda-ri", type=float, default=0.5)
    p.add_argument("--lambda-mass", type=float, default=0.01)
    p.add_argument(
        "--skip-lpips", action="store_true", help="Render and compute PSNR/SSIM only"
    )
    return p


def main():
    args = parser().parse_args()
    if not 1 <= args.iterations <= 30000:
        raise ValueError("iterations must be in [1,30000]")
    if args.seed < 0 or args.lambda_ri <= 0 or args.lambda_mass <= 0:
        raise ValueError("Invalid seed/loss weights")
    args.source = args.source.resolve()
    args.output = args.output.resolve()
    args.cache = args.cache.resolve()
    # Dataset/checkpoint/result directories are supplied by the caller.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = dict(
            dataset=args.dataset,
            source=str(args.source),
            views=args.views,
            seed=args.seed,
            optimization=asdict(OptimizationConfig(iterations=args.iterations)),
            nace=asdict(NACEConfig()),
            frir=asdict(
                FRIRConfig(lambda_ri=args.lambda_ri, lambda_mass=args.lambda_mass)
            ),
            skip_lpips=args.skip_lpips,
        )
        if (args.output / "completed.json").exists():
            if load_json(args.output / "config.json") != config:
                raise ValueError("Existing completed run uses different settings")
            print("Already complete:", args.output)
            return
        if (args.output / "config.json").exists():
            if not args.resume:
                raise ValueError(
                    "Partial run exists; supply --resume <checkpoint.pth> or a fresh --output"
                )
            if load_json(args.output / "config.json") != config:
                raise ValueError("Resume settings differ from original run")
        from gasf.evaluation import evaluate
        from gasf.frir.anchors import prepare_anchors, validate_anchors
        from gasf.trainer import train

        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "config.json", config)
        if args.anchors:
            anchors = load_json(args.anchors)
            validate_anchors(anchors, args.source)
            write_json(args.output / "anchors.json", anchors)
        else:
            anchors = prepare_anchors(
                args.source, args.output / "anchors.json", args.cache
            )
        optimization = OptimizationConfig(**config["optimization"])
        nace = NACEConfig(**config["nace"])
        frir = FRIRConfig(**config["frir"])
        started = time.perf_counter()
        scene, model, training = train(
            args.source,
            args.output,
            args.dataset,
            args.views,
            args.seed,
            optimization,
            nace,
            frir,
            anchors,
            args.resume,
        )
        metrics = evaluate(
            scene,
            model,
            nace,
            args.output,
            args.dataset,
            args.iterations,
            args.skip_lpips,
        )
        write_json(
            args.output / "completed.json",
            dict(
                method="GASF-GS (NACE + FRIR)",
                iterations=args.iterations,
                seed=args.seed,
                dataset=args.dataset,
                views=args.views,
                mean=metrics["mean"],
                wall_seconds=time.perf_counter() - started,
                frir_updates=training["frir_updates"],
                frir_gradient_checked=training["frir_gradient_checked"],
            ),
        )
        print("Complete:", args.output, metrics["mean"])


if __name__ == "__main__":
    main()
