#!/usr/bin/env python3
"""Merge a lightx2v Wan2.2-Lightning 4-step LoRA into the Bernini/Wan2.2-A14B
experts → a standalone `Wan2.2-T2V-A14B-Lightning` checkpoint our existing Swift
loader reads unchanged (same 1095-key contract; the LoRA only modifies the 10
per-block linears, adds no keys).

kohya merge: W_merged = W + strength * (alpha/rank) * (lora_up @ lora_down).
LoRA key `diffusion_model.blocks.N.X.{lora_down,lora_up,alpha}` maps to base
`blocks.N.X.weight` (strip prefix; `.ffn.0/2.` → `.ffn.fc1/fc2.`).

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/merge_lightning.py

Key-match is the gate: every LoRA module MUST map to an existing base weight, and
the merged key set MUST equal the base key set (0 added / 0 dropped).
"""

import shutil
from pathlib import Path

import mlx.core as mx

BASE = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
LORA = Path("/Volumes/DEV_ARCHIVE/weights/lightning-loras/Seko-V1.1")
OUT = Path("/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-lightning")
STRENGTH = 1.0  # the workflow's LoRA strength


def lora_key_to_base(module: str) -> str:
    """`diffusion_model.blocks.N.X` -> `blocks.N.X.weight` (base layout)."""
    assert module.startswith("diffusion_model."), module
    k = module[len("diffusion_model."):] + ".weight"
    # ffn is Sequential(Linear, GELU, Linear): .ffn.0/.2 -> .ffn.fc1/.fc2
    # (do this AFTER appending .weight so the `.N.weight` boundary matches).
    k = k.replace(".ffn.0.weight", ".ffn.fc1.weight").replace(".ffn.2.weight", ".ffn.fc2.weight")
    return k


def merge_expert(expert: str):
    print(f"\n=== {expert} ===")
    base = mx.load(str(BASE / f"{expert}.safetensors"))
    lora = mx.load(str(LORA / f"{expert}.safetensors"))
    base_keys = set(base.keys())

    # Group LoRA tensors by module prefix (strip .lora_down/.lora_up/.alpha)
    modules = {}
    for k in lora:
        if k.endswith(".lora_down.weight"):
            modules.setdefault(k[: -len(".lora_down.weight")], {})["down"] = lora[k]
        elif k.endswith(".lora_up.weight"):
            modules.setdefault(k[: -len(".lora_up.weight")], {})["up"] = lora[k]
        elif k.endswith(".alpha"):
            modules.setdefault(k[: -len(".alpha")], {})["alpha"] = lora[k]

    merged = dict(base)
    touched = 0
    for module, parts in modules.items():
        base_key = lora_key_to_base(module)
        assert base_key in base_keys, f"LoRA module {module} -> {base_key} NOT in base"
        down = parts["down"].astype(mx.float32)          # [rank, in]
        up = parts["up"].astype(mx.float32)              # [out, rank]
        rank = down.shape[0]
        alpha = float(parts["alpha"].astype(mx.float32).item()) if "alpha" in parts else rank
        scale = STRENGTH * (alpha / rank)
        delta = scale * (up @ down)                      # [out, in]
        w = base[base_key].astype(mx.float32) + delta
        merged[base_key] = w.astype(base[base_key].dtype)
        touched += 1
    print(f"  merged {touched} linears (rank {rank}, alpha {alpha}, scale {scale:.4f})")

    # Key contract: merged == base, nothing added/dropped.
    assert set(merged.keys()) == base_keys, "merge changed the key set!"
    # Materialize before save (lazy-zeros killer).
    mx.eval(list(merged.values()))
    OUT.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(OUT / f"{expert}.safetensors"), merged)
    print(f"  wrote {OUT / f'{expert}.safetensors'} ({len(merged)} tensors)")


def main():
    for expert in ("high_noise_model", "low_noise_model"):
        merge_expert(expert)
    # Complete the checkpoint: vae + t5 + config travel unchanged.
    for f in ("vae.safetensors", "t5_encoder.safetensors", "config.json"):
        shutil.copy2(BASE / f, OUT / f)
        print(f"copied {f}")
    print(f"\nLightning checkpoint → {OUT}")


if __name__ == "__main__":
    main()
