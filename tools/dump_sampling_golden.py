#!/usr/bin/env python3
"""Dump S4 goldens: r2v (chained APG) + rv2v (CFG chain) at tiny scale.

Runs the ORACLE's sampling.py verbatim (4 steps — both experts engaged:
999/899 high, 749/499 low) on seeded inputs, CPU stream. Also dumps an RNG
cross-binding fixture (seed 42 normal of the r2v target shape) so the Swift
side can verify Python-MLX <-> Swift-MLX seed-stream equality.

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/dump_sampling_golden.py
"""

from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
OUT = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"

from bernini_r_mlx.config import BerniniRendererConfig
from bernini_r_mlx.sampling import cfg_edit_sample, r2v_sample
from mlx_video.models.wan_2.utils import load_wan_model

STEPS = 4
TARGET_SHAPE = (16, 1, 16, 16)  # grid (1,8,8) = 64 tokens


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def main():
    cfg = BerniniRendererConfig().wan_config()
    boundary = cfg.boundary * cfg.num_train_timesteps

    # RNG cross-binding fixture: the exact stream r2v_sample(seed=42) consumes
    mx.random.seed(42)
    save("rng_seed42_target", mx.random.normal(TARGET_SHAPE))

    rng = np.random.default_rng(31)
    ref = mx.array(rng.standard_normal((16, 1, 16, 16)).astype(np.float32) * 0.5)
    video = mx.array(rng.standard_normal((16, 2, 16, 16)).astype(np.float32) * 0.5)
    ctx_cond_raw = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    ctx_null_raw = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    save("sampling_ref", ref)
    save("sampling_video", video)
    save("sampling_ctx_cond_raw", ctx_cond_raw)
    save("sampling_ctx_null_raw", ctx_null_raw)

    print("loading experts (2 x 28.6 GB)…")
    high = load_wan_model(WEIGHTS / "high_noise_model.safetensors", cfg)
    low = load_wan_model(WEIGHTS / "low_noise_model.safetensors", cfg)

    cond_high = high.embed_text([ctx_cond_raw])
    cond_low = low.embed_text([ctx_cond_raw])
    uncond_high = high.embed_text([ctx_null_raw])
    uncond_low = low.embed_text([ctx_null_raw])
    mx.eval(cond_high, cond_low, uncond_high, uncond_low)

    print("r2v (4 steps, 3 fwd/step)…")
    out_r2v = r2v_sample(
        high=high, low=low, ref_latents=[ref],
        cond_ctx_high=cond_high, cond_ctx_low=cond_low,
        uncond_ctx_high=uncond_high, uncond_ctx_low=uncond_low,
        target_shape=TARGET_SHAPE, head_dim=128,
        boundary_timestep=boundary, steps=STEPS, seed=42,
    )
    save("sampling_r2v_final", out_r2v)

    print("rv2v (4 steps, 4 fwd/step)…")
    out_rv2v = cfg_edit_sample(
        high=high, low=low, guidance_mode="rv2v",
        video_latents=[video], ref_latents=[ref],
        cond_ctx_high=cond_high, cond_ctx_low=cond_low,
        uncond_ctx_high=uncond_high, uncond_ctx_low=uncond_low,
        target_shape=TARGET_SHAPE, head_dim=128,
        boundary_timestep=boundary, steps=STEPS, seed=42,
    )
    save("sampling_rv2v_final", out_rv2v)

    print("v2v (4 steps, 2 fwd/step)…")
    out_v2v = cfg_edit_sample(
        high=high, low=low, guidance_mode="v2v",
        video_latents=[video], ref_latents=[],
        cond_ctx_high=cond_high, cond_ctx_low=cond_low,
        uncond_ctx_high=uncond_high, uncond_ctx_low=uncond_low,
        target_shape=TARGET_SHAPE, head_dim=128,
        boundary_timestep=boundary, steps=STEPS, seed=42,
    )
    save("sampling_v2v_final", out_v2v)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
