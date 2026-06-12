#!/usr/bin/env python3
"""Dump S1 parity fixtures from the Python oracle (mlx-video backbone).

Run with the oracle venv, CPU stream (GPU fp32 matmul noise would mask bugs):

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/dump_parity_fixtures.py [component]

component in {rope, scheduler, vae, t5, dit, all} (default all). Outputs .npy
files into Tests/BerniniRTests/Fixtures/parity/ (bf16 tensors saved as fp32).
Inputs are seeded numpy so both sides consume identical bytes.
"""

import gc
import json
import sys
from pathlib import Path

import numpy as np

import mlx.core as mx

mx.set_default_device(mx.cpu)

WEIGHTS = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
OUT = Path(__file__).resolve().parents[1] / "Tests/BerniniRTests/Fixtures/parity"
OUT.mkdir(parents=True, exist_ok=True)


def save(name: str, arr: mx.array):
    a = arr.astype(mx.float32) if arr.dtype == mx.bfloat16 else arr
    np.save(OUT / f"{name}.npy", np.array(a))
    print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")


def wan_config():
    from bernini_r_mlx.config import BerniniRendererConfig

    return BerniniRendererConfig().wan_config()


def dump_rope():
    print("[rope]")
    from mlx_video.models.wan_2.rope import rope_params, rope_precompute_cos_sin

    d = 128  # A14B head_dim
    freqs = mx.concatenate(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        axis=1,
    )
    save("rope_freqs_d128", freqs)
    cos_f, sin_f = rope_precompute_cos_sin([(3, 4, 4)], freqs)
    save("rope_cos_grid344", cos_f)
    save("rope_sin_grid344", sin_f)


def dump_scheduler():
    print("[scheduler]")
    from mlx_video.models.wan_2.scheduler import FlowUniPCScheduler

    rng = np.random.default_rng(7)
    sample = mx.array(rng.standard_normal((4, 8)).astype(np.float32))
    vs = mx.array(rng.standard_normal((40, 4, 8)).astype(np.float32))
    save("unipc_sample0", sample)
    save("unipc_vs", vs)

    sched = FlowUniPCScheduler()
    sched.set_timesteps(40, shift=3.0)
    checkpoints = {0, 1, 2, 12}
    ts = np.array(sched.timesteps)
    for i in range(40):
        sample = sched.step(vs[i], float(ts[i]), sample)
        if i in checkpoints:
            mx.eval(sample)
            save(f"unipc_sample_after_{i}", sample)
    mx.eval(sample)
    save("unipc_sample_final", sample)


def dump_vae():
    print("[vae] loading fp32 VAE…")
    from mlx_video.models.wan_2.utils import load_vae_encoder

    cfg = wan_config()
    vae = load_vae_encoder(WEIGHTS / "vae.safetensors", cfg)

    rng = np.random.default_rng(11)
    # [B, 3, T, H, W], T = 9 = 1 + 4k exercises the chunked encode path
    x = mx.array((rng.random((1, 3, 9, 64, 64), dtype=np.float64) * 2 - 1).astype(np.float32))
    save("vae_input", x)
    z = vae.encode(x)
    mx.eval(z)
    save("vae_latent", z)
    y = vae.decode(z)
    mx.eval(y)
    save("vae_decoded", y)
    del vae
    gc.collect()


def dump_t5():
    print("[t5] loading fp32 umT5 (11 GB)…")
    from mlx_video.models.wan_2.utils import load_t5_encoder

    cfg = wan_config()
    t5 = load_t5_encoder(WEIGHTS / "t5_encoder.safetensors", cfg)

    rng = np.random.default_rng(13)
    ids = mx.array(rng.integers(2, 256000, size=(1, 16)).astype(np.int32))
    mask = mx.ones((1, 16), dtype=mx.int32)
    save("t5_ids", ids)
    out = t5(ids, mask)
    mx.eval(out)
    save("t5_features", out)
    del t5
    gc.collect()


def dump_dit():
    print("[dit] loading bf16 high-noise expert (28.6 GB)…")
    from mlx_video.models.wan_2.utils import load_wan_model

    cfg = wan_config()
    model = load_wan_model(WEIGHTS / "high_noise_model.safetensors", cfg)

    rng = np.random.default_rng(17)
    # Tiny latent: [C=16, F=1, H=8, W=8] -> grid (1,4,4), 16 tokens
    x = mx.array(rng.standard_normal((16, 1, 8, 8)).astype(np.float32))
    ctx = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    t = mx.array([999.0])
    save("dit_x", x)
    save("dit_ctx_raw", ctx)

    embedded = model.embed_text([ctx])
    mx.eval(embedded)
    save("dit_ctx_embedded", embedded)

    out = model([x], t, embedded, seq_len=16)
    mx.eval(out[0])
    save("dit_out_t999", out[0])

    # Low-noise regime timestep too (same expert weights — exercises the
    # time-embedding path at a second point)
    out2 = model([x], mx.array([400.0]), embedded, seq_len=16)
    mx.eval(out2[0])
    save("dit_out_t400", out2[0])
    del model
    gc.collect()


def dump_sa3d():
    print("[sa3d]")
    from bernini_r_mlx.model.rope_sa3d import prepare_sa3d_rope_cos_sin, visual_id_phase
    from mlx_video.models.wan_2.rope import rope_params

    d = 128
    freqs = mx.concatenate(
        [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ],
        axis=1,
    )
    for sid in (0, 1, 3):
        c, s = visual_id_phase(sid, d)
        save(f"sa3d_phase_cos_sid{sid}", c)
        save(f"sa3d_phase_sin_sid{sid}", s)
    # Two-segment sequence: ref grid (1,4,4) sid 1 + target grid (3,4,4) sid 0
    cos, sin = prepare_sa3d_rope_cos_sin([((1, 4, 4), 1), ((3, 4, 4), 0)], freqs, d)
    save("sa3d_cos_ref144s1_tgt344s0", cos)
    save("sa3d_sin_ref144s1_tgt344s0", sin)


def dump_multiseg():
    print("[multiseg] loading bf16 high-noise expert (28.6 GB)…")
    from bernini_r_mlx.model.multiseg import forward_multiseg
    from mlx_video.models.wan_2.utils import load_wan_model

    cfg = wan_config()
    model = load_wan_model(WEIGHTS / "high_noise_model.safetensors", cfg)

    rng = np.random.default_rng(29)
    target = mx.array(rng.standard_normal((16, 1, 8, 8)).astype(np.float32))
    ref = mx.array(rng.standard_normal((16, 1, 8, 8)).astype(np.float32) * 0.5)
    ctx = mx.array(rng.standard_normal((16, 4096)).astype(np.float32) * 0.5)
    save("multiseg_target", target)
    save("multiseg_ref", ref)
    save("multiseg_ctx_raw", ctx)

    embedded = model.embed_text([ctx])
    mx.eval(embedded)

    t = mx.array(999.0)
    out_t2v = forward_multiseg(model, [], target, t, embedded, head_dim=128)
    mx.eval(out_t2v)
    save("multiseg_out_targetonly", out_t2v)

    out_ref = forward_multiseg(model, [(ref, 1)], target, t, embedded, head_dim=128)
    mx.eval(out_ref)
    save("multiseg_out_1ref", out_ref)


if __name__ == "__main__":
    component = sys.argv[1] if len(sys.argv) > 1 else "all"
    dumps = {
        "rope": dump_rope,
        "scheduler": dump_scheduler,
        "vae": dump_vae,
        "t5": dump_t5,
        "dit": dump_dit,
        "sa3d": dump_sa3d,
        "multiseg": dump_multiseg,
    }
    if component == "all":
        for fn in dumps.values():
            fn()
    else:
        dumps[component]()
    print("done ->", OUT)
