#!/usr/bin/env python3
"""Apples-to-apples int4-vs-bf16 per-pass cosine in the ORACLE (Python MLX),
on the exact S6 fixture inputs, so the Swift gate number can be compared
1:1. (The published 0.9992 was measured under the conversion's own input
conditions; the oracle's actual gate was >= 0.99.)

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/int4_cosine_reference.py
"""

from pathlib import Path

import numpy as np

import mlx.core as mx

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights")
FIX = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"

from bernini_r_mlx.config import BerniniRendererConfig
from mlx_video.models.wan_2.utils import load_wan_model

cfg = BerniniRendererConfig().wan_config()
x = mx.array(np.load(FIX / "dit_x.npy"))
ctx = mx.array(np.load(FIX / "dit_ctx_raw.npy"))

print("bf16 (CPU stream)…")
mx.set_default_device(mx.cpu)
bf16 = load_wan_model(WEIGHTS / "ckpt-bf16/high_noise_model.safetensors", cfg)
emb = bf16.embed_text([ctx])
out_bf16 = bf16([x], mx.array([999.0]), emb, seq_len=16)[0]
mx.eval(out_bf16)
del bf16

print("int4 (GPU stream — quantized matmul routes to Metal)…")
mx.set_default_device(mx.gpu)
int4 = load_wan_model(
    WEIGHTS / "ckpt-int4/high_noise_model.safetensors", cfg,
    quantization={"group_size": 64, "bits": 4})
emb_q = int4.embed_text([ctx])
out_int4 = int4([x], mx.array([999.0]), emb_q, seq_len=16)[0]
mx.eval(out_int4)

a = np.array(out_bf16).astype(np.float64).ravel()
b = np.array(out_int4).astype(np.float64).ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"python per-pass cosine (same fixture, t=999) = {cos:.7f}")
