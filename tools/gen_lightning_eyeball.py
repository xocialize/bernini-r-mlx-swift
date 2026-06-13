#!/usr/bin/env python3
"""Python eyeball: generate 4-step Lightning video from the MERGED checkpoint,
to confirm the merge + the 4-step/euler/shift-5/CFG-off recipe produce good
output BEFORE building the Swift sampler. Uses the merged ckpt directly (no
loras arg → sidesteps the pinned mlx-video's missing lora.py).

    /Volumes/DEV_ARCHIVE/bernini-r-mlx/.venv/bin/python tools/gen_lightning_eyeball.py
"""
from mlx_video.models.wan_2.generate import generate_video

generate_video(
    model_dir="/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-lightning",
    prompt="A red fox walking through fresh snow, golden hour",
    width=832,
    height=480,
    num_frames=17,
    steps=4,
    guide_scale=1.0,   # <= 1.0 -> CFG disabled (B=1), the Lightning recipe
    shift=5.0,         # the ComfyUI ModelSamplingSD3 shift
    seed=42,
    scheduler="euler",
    tiling="none",
    output_path="/tmp/lightning_4step.mp4",
)
print("wrote /tmp/lightning_4step.mp4")
