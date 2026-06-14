#!/usr/bin/env python3
"""E11 — corrected VAE decode golden (first-chunk frame-0 bypass).

The stock mlx-video wan_2 VAE upsample3d ALWAYS doubles every frame (T_lat*4
output), diverging from official Wan2.2 which BYPASSES time_conv for frame 0 of
the first chunk -> (T_lat-1)*4+1. Both bernini's Swift AND its Python oracle
inherited the bug, so there is no stock golden to validate the fix against.

This monkeypatches the whole-seq upsample3d with the corrected first-chunk rule
(ported verbatim from the test-verified helios-branch `wan/vae22.py`
Resample.first_chunk path) on the REAL 16-ch VAE weights, and dumps a golden the
Swift fix bit-checks against. Frame-count assert encodes the official formula.

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/dump_vae_firstchunk_golden.py
"""

from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
OUT = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"

from bernini_r_mlx.config import BerniniRendererConfig
from mlx_video.models.wan_2.utils import load_vae_encoder
from mlx_video.models.wan_2.vae import Resample

# --- the corrected first-chunk upsample3d (whole-seq path only) ---
_orig_call = Resample.__call__


def _patched_call(self, x, feat_cache=None, feat_idx=None):
    # Only override the whole-seq (feat_cache is None) upsample3d temporal step.
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
        # spatial upsample — identical to the stock upsample path
        x = x.transpose(0, 2, 3, 4, 1).reshape(b * t2, h, w, c)
        x = mx.repeat(x, 2, axis=1)
        x = mx.repeat(x, 2, axis=2)
        x = self.resample[1](x)
        c_out = x.shape[-1]
        return x.reshape(b, t2, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)
    return _orig_call(self, x, feat_cache, feat_idx)


Resample.__call__ = _patched_call


def save(name, arr):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def main():
    cfg = BerniniRendererConfig().wan_config()
    print("loading VAE…")
    vae = load_vae_encoder(WEIGHTS / "vae.safetensors", cfg)

    T_LAT = 3  # -> corrected (3-1)*4+1 = 9 output frames (stock bug would give 12)
    rng = np.random.default_rng(7)
    z = mx.array(rng.standard_normal((16, T_LAT, 8, 8)).astype(np.float32) * 0.5)
    save("vae_fc_z", z)

    frames = vae.decode(z[None])  # [1, 3, T_out, H, W]
    mx.eval(frames)
    t_out = frames.shape[2]
    expected = (T_LAT - 1) * 4 + 1
    print(f"  T_lat={T_LAT} -> T_out={t_out} (official formula (T_lat-1)*4+1 = {expected})")
    assert t_out == expected, f"FRAME COUNT WRONG: {t_out} != {expected}"
    save("vae_fc_frames", frames)

    # Regenerate the existing vae decode golden (was the flawed 12-frame always-double;
    # corrected is (3-1)*4+1 = 9). Encode side is unchanged, so only vae_decoded moves.
    vae_latent = mx.array(np.load(OUT / "vae_latent.npy"))  # [1,16,3,8,8]
    vae_decoded = vae.decode(vae_latent)
    mx.eval(vae_decoded)
    assert vae_decoded.shape[2] == 9, f"vae_decoded T_out={vae_decoded.shape[2]} != 9"
    save("vae_decoded", vae_decoded)
    print("OK -> corrected goldens written (vae_fc_frames + regenerated vae_decoded)")


if __name__ == "__main__":
    main()
