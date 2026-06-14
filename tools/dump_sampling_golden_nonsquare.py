#!/usr/bin/env python3
"""Dump NON-SQUARE rv2v/v2v goldens to expose a height/width-axis bug.

The square [16,1,16,16] S4 fixture cannot catch an H<->W transpose or an H-axis
positional error: when H==W the two axes are indistinguishable. The in-app rv2v
run (60x104, non-square) showed a vertical-mirror artifact that v2v (same code,
no ref segment) and r2v (different sampler) did not. This dumps the SAME oracle
calls at NON-SQUARE multi-frame geometry [16,2,16,24] (grid 2,8,12) so the Swift
parity gate exercises the regime where the artifact appears.

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/dump_sampling_golden_nonsquare.py
"""

from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
OUT = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"

from bernini_r_mlx.config import BerniniRendererConfig
from bernini_r_mlx.sampling import cfg_edit_sample
from mlx_video.models.wan_2.utils import load_wan_model

STEPS = 4
TARGET_SHAPE = (16, 2, 16, 24)  # NON-SQUARE H=16,W=24 ; multi-frame T=2 ; grid (2,8,12)


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def main():
    cfg = BerniniRendererConfig().wan_config()
    boundary = cfg.boundary * cfg.num_train_timesteps

    # Oracle generates its start latent as normal(target_shape) after seed(42).
    # Save that exact array so the Swift side injects identical noise (no RNG
    # cross-binding dependence at this new shape).
    mx.random.seed(42)
    save("sampling_ns_noise", mx.random.normal(TARGET_SHAPE))

    rng = np.random.default_rng(31)
    ref = mx.array(rng.standard_normal((16, 1, 16, 24)).astype(np.float32) * 0.5)
    video = mx.array(rng.standard_normal((16, 2, 16, 24)).astype(np.float32) * 0.5)
    ctx_cond_raw = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    ctx_null_raw = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    save("sampling_ns_ref", ref)
    save("sampling_ns_video", video)
    save("sampling_ns_ctx_cond_raw", ctx_cond_raw)
    save("sampling_ns_ctx_null_raw", ctx_null_raw)

    print("loading experts (2 x 28.6 GB)…")
    high = load_wan_model(WEIGHTS / "high_noise_model.safetensors", cfg)
    low = load_wan_model(WEIGHTS / "low_noise_model.safetensors", cfg)

    cond_high = high.embed_text([ctx_cond_raw])
    cond_low = low.embed_text([ctx_cond_raw])
    uncond_high = high.embed_text([ctx_null_raw])
    uncond_low = low.embed_text([ctx_null_raw])
    mx.eval(cond_high, cond_low, uncond_high, uncond_low)

    print("rv2v NS (4 steps, 4 fwd/step)…")
    out_rv2v = cfg_edit_sample(
        high=high, low=low, guidance_mode="rv2v",
        video_latents=[video], ref_latents=[ref],
        cond_ctx_high=cond_high, cond_ctx_low=cond_low,
        uncond_ctx_high=uncond_high, uncond_ctx_low=uncond_low,
        target_shape=TARGET_SHAPE, head_dim=128,
        boundary_timestep=boundary, steps=STEPS, seed=42,
    )
    save("sampling_ns_rv2v_final", out_rv2v)

    print("v2v NS control (4 steps, 2 fwd/step)…")
    out_v2v = cfg_edit_sample(
        high=high, low=low, guidance_mode="v2v",
        video_latents=[video], ref_latents=[],
        cond_ctx_high=cond_high, cond_ctx_low=cond_low,
        uncond_ctx_high=uncond_high, uncond_ctx_low=uncond_low,
        target_shape=TARGET_SHAPE, head_dim=128,
        boundary_timestep=boundary, steps=STEPS, seed=42,
    )
    save("sampling_ns_v2v_final", out_v2v)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
