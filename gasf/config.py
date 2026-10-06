"""Settings of the evaluated NACE + FRIR model."""

from dataclasses import dataclass


@dataclass
class OptimizationConfig:
    iterations: int = 30_000
    position_lr_init: float = 0.00016
    position_lr_final: float = 0.0000016
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densification_interval: int = 300
    opacity_reset_interval: int = 3000
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densify_grad_threshold: float = 0.001
    gate_lr: float = 0.001
    lambda_gate: float = 0.01


@dataclass
class NACEConfig:
    split_offset: float = 0.55
    temperature: float = 1.0
    temperature_final: float = 0.1
    anneal_until: int = 15_000
    gaussians_per_bin: int = 64
    pixels_per_gaussian: int = 128
    ut_alpha: float = 1.0
    ut_beta: float = 2.0
    ut_kappa: float = 0.0


@dataclass
class FRIRConfig:
    lambda_ri: float = 0.5
    lambda_mass: float = 0.01
    pseudo_huber_delta: float = 0.1
    epsilon_min: float = 0.01
    rays_per_step: int = 64
    start_iteration: int = 200
    warmup_end: int = 1000
    interval: int = 8
