#!/usr/bin/env python3
"""Dump the S2 e2e golden: dual-expert CFG t2v denoise at tiny scale.

Faithful excerpt of mlx_video generate.py::generate_video's dual-model CFG
path (lines ~490-673 @ 87db56a) with injected noise and injected raw text
features (no tokenizer / T5 — those stages are parity-locked separately).
CPU stream. 4 steps exercises both experts (timesteps 999, 899 high; 749,
499 low at shift 3.0) and the UniPC corrector.

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/dump_e2e_golden.py
"""

from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
OUT = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"

from bernini_r_mlx.config import BerniniRendererConfig
from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler
from mlx_video.models.wan_2.utils import load_vae_encoder, load_wan_model

# E11: the stock mlx-video upsample3d ALWAYS doubles every frame (T_lat*4);
# official Wan2.2 bypasses time_conv for frame 0 of the first chunk ->
# (T_lat-1)*4+1. Apply the corrected first-chunk rule to the whole-seq decode so
# the e2e frames golden matches the fixed Swift WanVAE. Same patch as
# tools/dump_vae_firstchunk_golden.py (ported from helios-branch wan/vae22.py).
from mlx_video.models.wan_2.vae import Resample

_orig_resample_call = Resample.__call__


def _patched_resample_call(self, x, feat_cache=None, feat_idx=None):
    if self.mode == "upsample3d" and feat_cache is None:
        b, c, t, h, w = x.shape
        if t > 1:
            first = x[:, :, 0:1]  # frame 0 BYPASSES time_conv
            rest = x[:, :, 1:]  # [B,C,T-1,H,W]
            x_t = self.time_conv(rest).reshape(b, 2, c, t - 1, h, w)
            rest_up = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(
                b, c, (t - 1) * 2, h, w
            )
            x = mx.concatenate([first, rest_up], axis=2)  # 1 + (T-1)*2 = 2T-1
        else:
            x_t = self.time_conv(x).reshape(b, 2, c, t, h, w)
            x = mx.stack([x_t[:, 0], x_t[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
        t2 = x.shape[2]
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t2, h, w, c)
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.resample[1](x)
        c_out = x.shape[-1]
        return x.reshape(b, t2, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)
    return _orig_resample_call(self, x, feat_cache, feat_idx)


Resample.__call__ = _patched_resample_call

STEPS = 4
SHIFT = 3.0
GUIDE_SCALE = (3.0, 4.0)  # (low, high)
TARGET_SHAPE = (16, 1, 16, 16)  # [C, T_lat, H_lat, W_lat] -> grid (1,8,8)


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def main():
    cfg = BerniniRendererConfig().wan_config()

    rng = np.random.default_rng(23)
    noise = mx.array(rng.standard_normal(TARGET_SHAPE).astype(np.float32))
    context = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    context_null = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    save("e2e_noise", noise)
    save("e2e_ctx_cond", context)
    save("e2e_ctx_null", context_null)

    print("loading experts (2 x 28.6 GB)…")
    high_noise_model = load_wan_model(WEIGHTS / "high_noise_model.safetensors", cfg)
    low_noise_model = load_wan_model(WEIGHTS / "low_noise_model.safetensors", cfg)

    # — generate.py excerpt (dual + CFG path), verbatim modulo fixtures —
    context_emb_low = low_noise_model.embed_text([context, context_null])
    context_emb_high = high_noise_model.embed_text([context, context_null])
    mx.eval(context_emb_low, context_emb_high)
    context_cfg_low = mx.concatenate(
        [context_emb_low[0:1], context_emb_low[1:2]], axis=0
    )
    context_cfg_high = mx.concatenate(
        [context_emb_high[0:1], context_emb_high[1:2]], axis=0
    )

    cross_kv_low = low_noise_model.prepare_cross_kv(context_cfg_low)
    cross_kv_high = high_noise_model.prepare_cross_kv(context_cfg_high)
    mx.eval(cross_kv_low, cross_kv_high)

    c, t_latent, h_latent, w_latent = TARGET_SHAPE
    f_grid = t_latent // cfg.patch_size[0]
    h_grid = h_latent // cfg.patch_size[1]
    w_grid = w_latent // cfg.patch_size[2]
    rope_grid_sizes = [(f_grid, h_grid, w_grid), (f_grid, h_grid, w_grid)]
    rope_cos_sin_low = low_noise_model.prepare_rope(rope_grid_sizes)
    rope_cos_sin_high = high_noise_model.prepare_rope(rope_grid_sizes)
    mx.eval(rope_cos_sin_low, rope_cos_sin_high)
    seq_len = f_grid * h_grid * w_grid

    sched = FlowUniPCScheduler(num_train_timesteps=cfg.num_train_timesteps)
    sched.set_timesteps(STEPS, shift=SHIFT)

    latents = noise
    boundary = cfg.boundary * cfg.num_train_timesteps
    timestep_list = sched.timesteps.tolist()
    print("timesteps:", timestep_list, "boundary:", boundary)

    for i in range(STEPS):
        timestep_val = timestep_list[i]
        if timestep_val >= boundary:
            model, kv, rcs, ctx = (
                high_noise_model, cross_kv_high, rope_cos_sin_high, context_cfg_high)
            gs = GUIDE_SCALE[1]
        else:
            model, kv, rcs, ctx = (
                low_noise_model, cross_kv_low, rope_cos_sin_low, context_cfg_low)
            gs = GUIDE_SCALE[0]

        t_batch = mx.array([timestep_val, timestep_val])
        preds = model(
            [latents, latents],
            t=t_batch,
            context=ctx,
            seq_len=seq_len,
            cross_kv_caches=kv,
            rope_cos_sin=rcs,
        )
        noise_pred_cond, noise_pred_uncond = preds[0], preds[1]
        noise_pred = noise_pred_uncond + gs * (noise_pred_cond - noise_pred_uncond)

        latents = sched.step(noise_pred[None], timestep_val, latents[None]).squeeze(0)
        mx.eval(latents)
        save(f"e2e_latent_after_{i}", latents)
        print(f"  step {i} (t={timestep_val}, {'high' if timestep_val >= boundary else 'low'}) done")

    save("e2e_latent_final", latents)

    del high_noise_model, low_noise_model
    print("decoding…")
    vae = load_vae_encoder(WEIGHTS / "vae.safetensors", cfg)
    frames = vae.decode(latents[None])
    mx.eval(frames)
    save("e2e_frames", frames)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
