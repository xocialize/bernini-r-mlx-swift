# bernini-r-mlx-swift — Porting Spec

**Goal:** Swift/MLX core package serving **Bernini-R** — provenance-audited **byte-stock
Wan2.2-T2V-A14B** (ByteDance; zero extra tensors vs the base) — for MLXEngine's
`textToVideo` + `textToImage`, with the renderer's editing surfaces (r2v / v2v / rv2v)
following as the motivating packages for `imageEdit`/`videoEdit` at contract 1.2.0.
Replaces the dropped Lance generation phases (L2–L4).

**Method: Python-MLX → Swift-MLX port with a Swift component donor.** The Python port
(`/Volumes/DEV_ARCHIVE/bernini-r-mlx`) is **done and e2e-validated** (t2v/t2i/r2v/v2v/rv2v
coherent; int4 per-pass cosine 0.9992) — it is the parity oracle. The Wan substrate already
exists in Swift in `/Volumes/DEV_ARCHIVE/longcat-avatar-mlx-swift` (feature-complete,
parity-locked) — it is the component donor. Same array semantics on both sides (MLX core);
this is a transpose, not a redesign. Preserve isomorphic structure with the Python oracle
and the mlx-video backbone — same file/class/method decomposition, PyTorch↔/Python↔Swift op
substitutions only.

## Assets & references (all local, pinned)

| Asset | Path | Role |
|---|---|---|
| Python oracle | `/Volumes/DEV_ARCHIVE/bernini-r-mlx` (~1.7k LOC) | Numerics + pipeline reference; fixture generator |
| Wan2.2 backbone source | `/Volumes/DEV_ARCHIVE/longcat-avatar-mlx/refs/mlx-video` @ `87db56a`, `mlx_video/models/wan_2/` (5.9k LOC: `wan_2.py` DiT, `vae.py`, `text_encoder.py`, `scheduler.py`, `rope.py`, `attention.py`) | The "upstream" the Swift DiT/VAE/T5/UniPC must be isomorphic to |
| Swift donor | `/Volumes/DEV_ARCHIVE/longcat-avatar-mlx-swift` | Lift/adapt table below |
| Converted ckpts | **Published:** `mlx-community/Bernini-R-bf16` · `mlx-community/Bernini-R-int4` (files: `high_noise_model` / `low_noise_model` / `t5_encoder` / `vae` `.safetensors` + `config.json`). Local mirrors: `/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16` (64 GB) · `ckpt-int4` (27 GB) | Runtime weights — Swift consumes the published repos verbatim (donor precedent: no Swift-side re-conversion); `BERNINI_R_WEIGHTS_DIR` overrides to the local mirror for dev |
| Diffusers source | `…/bernini-r-mlx-weights/Bernini-R-Diffusers` (106 GB) | Conversion provenance only — Swift never reads it |
| Must-read docs | oracle `_research/{PROVENANCE,SA3D_ROPE,PHASE3_CONDITIONING}.md` | Config truths, SA-3D math, conditioning mechanics |

> ⚠️ The oracle's venv has mlx-video as an **editable install pointing at a stale path**
> (`~/DEV_INT/longcat-avatar-mlx/refs/mlx-video` — moved to DEV_ARCHIVE). Before re-running
> the Python oracle to dump fixtures: `pip install -e /Volumes/DEV_ARCHIVE/longcat-avatar-mlx/refs/mlx-video`
> (or re-point the `.pth`), and verify `import mlx_video` resolves.

## Donor salvage map (from longcat-avatar-mlx-swift)

| Donor file | Action | Notes |
|---|---|---|
| `Models/AutoencoderKLWan.swift` (1073) | **KEEP as-is** | Same 16-ch Wan VAE (dim 96, mult [1,2,4,4], temporal down [f,t,t]); parity already proven (encode 3.1e-6 / decode 1.76e-3). Keep the debugged-the-hard-way choices: L10 attention on `.cpu` stream, the upsample3d "Rep" feat-cache sentinel, chunked encode/decode refs. Verify config equality vs `wan_2/vae.py` + oracle `vae.safetensors` keys before trusting. |
| `Models/UMT5EncoderModel.swift` (367) | **KEEP, verify** | umT5-XXL (24L, d4096, ffn 10240, 64H, vocab 256384, per-block relative bias). Cross-check every config flag vs `wan_2/text_encoder.py` (bias-free linears, no QK^T scale, fp32 softmax, `sharedPos`). Tokenizer via swift-transformers sentencepiece; text_len 512. |
| `Models/RoPE3D.swift` (185) | **ADAPT** | Same head_dim split (128 → t44/h42/w42) but Wan's rope comes from `wan_2/rope.py` (`rope_params`/`rope_precompute_cos_sin`) — parity-gate the convention, don't assume. Then extend with the SA-3D segment phase (net-new below). |
| `Models/Attention.swift` | **TEMPLATE** | Wan self/cross-attn have RMS `norm_q`/`norm_k` and biased q/k/v/o (per ckpt keys) — donor block is the SDPA/packing pattern, not the literal module. Keep fused-SDPA-everywhere (L22: ~10× tighter parity than manual matmul+softmax). |
| `Models/Blocks.swift` | **TEMPLATE** | Wan modulation is a per-block `modulation` parameter (6,dim) + global `time_projection`, NOT LongCat's AdaLN MLP. Keep the `modulateFP32` lesson (L14): all modulation math in fp32. |
| `Utilities/WeightLoader.swift` (300+) | **KEEP** | HF snapshot download, sharded safetensors, `detectQuantization` + **apply-quant-BEFORE-load** (QuantizedLinear slots must exist when bit-packed tensors load), env override → `BERNINI_R_WEIGHTS_DIR`. |
| `Pipeline/FlowMatchEulerDiscreteScheduler.swift` | **SKIP** | Bernini uses **FlowUniPC** (bh2) — net-new port from `wan_2/scheduler.py`. |
| Whisper / Audio / Avatar files (`AvatarAttention`, `AudioProjModel`, `SingleStreamAttention`, audio pipeline) | **SKIP** | Avatar-specific; no audio path in Bernini-R. |

## Net-new Swift (1:1 ports, isomorphic file/class/method names)

| Source (Python) | LOC | Swift target | Notes |
|---|---|---|---|
| `wan_2/wan_2.py` `WanModel` | 388 | `Models/WanModel.swift` | The main translation: 40L · dim 5120 · 40H (head_dim 128) · ffn 13824 · patch (1,2,2); blocks = {self_attn(+norm_q/k), cross_attn(+norm_q/k), ffn.0/2, norm3, modulation}; globals = patch_embedding, text_embedding.{0,2}, time_embedding.{0,2}, time_projection, head(+modulation). |
| `wan_2/scheduler.py` `FlowUniPCScheduler` | ~200 | `Pipeline/FlowUniPCScheduler.swift` | bh2 solver, shift 3.0, 1000 train timesteps. |
| oracle `model/renderer.py` | 69 | `Models/BerniniRendererModel.swift` | Dual experts (high/low), `selectExpert(t)`, boundary 875.0, `fromPretrained` over the two expert files. Both experts resident. |
| oracle `model/rope_sa3d.py` | 82 | `Models/RoPESA3D.swift` | Zero-parameter segment phase: `visual_id_phase` (θ^(−2k/head_dim) · source_id), complex multiply onto base 3D rope. source_id=0 ⇒ identity ⇒ t2v numerically unchanged. |
| oracle `model/multiseg.py` | 120 | `Models/Multiseg.swift` | `patchSegment` / `forwardMultiseg` / `timeEmbed`: concat conditioning segments' tokens+ropes, joint attention, slice target tokens from the end. source_id order matters (target=0, conditioners 1..K). |
| oracle `sampling.py` | 208 | `Pipeline/Sampling.swift` | `MomentumBuffer`, `r2vSample` (3 fwd/step, APG `normalizedGuidanceChain`), `cfgEditSample` (v2v/v2v_chain/rv2v, 2–4 fwd/step). |
| oracle `streaming_decode.py` | 114 | `Models/StreamingDecode.swift` | Lossless temporal-chunked VAE decode (cross-chunk causal-conv cache). Reconcile with the donor VAE's built-in per-frame chunked decode — one mechanism, bit-identity-gated, not two. |
| oracle `pipeline_mlx.py` | 301 | `Pipeline/BerniniPipeline.swift` | `t2v / t2i / r2v / v2v / rv2v` entry points + ref/video preprocessing + decode/save. t2i = t2v with numFrames 1. |
| oracle `config.py` | 55 | `Models/BerniniConfig.swift` | Wrapper config + derived Wan config; defaults table below. |

`utils/weights.py` (diffusers→Wan keymap) is **conversion-time only** — the converted ckpts
are already in original-Wan naming; Swift loads them directly. Pin the key contract like
lance-mlx-swift's `LanceWeightKeys` (verified: **1095 tensors/expert**, e.g.
`blocks.0.cross_attn.norm_k.weight`) and refuse partial loads (0 missing / 0 unused).

## Pinned config (oracle truths — deviations are bugs)

| Knob | Value | Source |
|---|---|---|
| Expert switch boundary | 0.875 → t ≥ 875.0 high-noise, else low | `config.py` / `renderer.py` |
| Flow shift | **3.0** (NOT Wan2.2's 12.0) | `config.py` |
| Scheduler | FlowUniPC `bh2`, 40 steps | `sampling.py` |
| Guidance | ω_I 3.0 · ω_TI 4.0 · ω_V 3.0 · η 0.5 · norm_threshold (50,50) · momentum −0.5 | `r2v`/`v2v` signatures |
| Boundary-crossing rule | ω × 0.75 once, on first high→low transition | `sampling.py` step loop |
| Text | UMT5 features [1, 512, 4096]; contexts pre-embedded per expert, never re-embedded per step | `pipeline_mlx.py` |
| RoPE | head_dim 128 → t44/h42/w42 (complex 22/21/21), θ 10000 | `SA3D_ROPE.md` |
| VAE | 16-ch (z_dim 16), stride 4/8/8 | `PROVENANCE.md` |
| Defaults | 832×480 · 49 frames · seed 42 (edit paths) · `wan_neg` negative prompt | `pipeline_mlx.py` |
| Quant | int4 g64, Linear-only, skip embeddings/norms/head | `scripts/quantize.py` |

## Phase plan (each phase: parity gate → then move on; mirror the oracle's own phases)

| Phase | Scope | Parity gate |
|---|---|---|
| **S0** ✅ | Scaffold; config decode; **key contract** test against local ckpt headers (1095/expert, bf16 + int4 quantized-key variants) | Offline; 0 missing / 0 unused on both variants — **PASSED 2026-06-12** |
| **S1** ✅ | Wan substrate: VAE 1:1 translation (donor naming diverged — diffusers vs mlx-video keys), umT5 lift (fp32 at load, per mlx-video), `WanModel` 1:1, Euler/DPM++/UniPC | **PASSED 2026-06-12**: VAE enc/dec, umT5 full-forward, DiT full 40-block forward vs the real 28.6 GB expert (`BERNINI_R_PARITY_DIT=1`), UniPC 40-step trajectory ≤1e-5 (corrector needs float64 Gaussian solve even at order 2) |
| **S2** ✅ | t2v denoise core (`BerniniRendererModel` + `denoiseT2V`, dual-expert CFG; t2i = 1-frame) | **PASSED 2026-06-12**: 4-step golden (999/899 high, 749/499 low — both experts + corrector) + VAE decode vs oracle, per-step probes ≤0.05 (`BERNINI_R_PARITY_E2E=1`). Remaining S2b: prompt-level entry (umT5 tokenizer from `google/umt5-xxl` — published weight repos ship no tokenizer) + first GPU generation (xcodebuild; metallib boundary) |
| **S3** ✅ | SA-3D RoPE + multiseg | **PASSED 2026-06-12**: phase + 2-segment rope ≤1e-6 vs oracle; `forwardMultiseg([], target)` ≡ plain forward ≤1e-3 (random-init tiny config); real-expert multiseg (target-only + 1-ref) ≤1e-2 vs oracle (`BERNINI_R_PARITY_DIT=1`) |
| **S4** ✅ | r2v (APG) → v2v → rv2v | **PASSED 2026-06-12, BIT-EXACT (max_abs 0.0 on all three samplers)** — 4-step goldens vs oracle, injected noise. Canonical gate: `swift run RunBernini --s4-gate` (the identical workload crashes with a spurious Metal GPU-timeout under swiftpm-testing-helper — `SamplingTests.samplersMatchOracleGoldens` stays env-gated as a repro; known issue). Bonus: Python-MLX ↔ Swift-MLX RNG seed streams verified bit-identical |
| **S5** ✅ | Streaming decode | **PASSED 2026-06-12, BIT-IDENTICAL (max_abs 0.0)** at 1/2/3/5 latent frames vs whole-seq decode, real fp32 weights. Canonical gate: `swift run RunBernini --s5-gate`. Wired as BerniniPipeline's decode path. NOTE: SPM test-product Cmlx bundle assembly is broken in this env (metallib not found from xctest; clean rebuild did not fix) — Metal-context gates run as CLI; StreamingDecodeTests kept as repro |
| **S6** ✅ | int4 path | **PASSED 2026-06-12**: per-pass cosine 0.9977231 vs bf16 (gate ≥0.99 — the oracle's own gate; Python on the SAME fixture measures 0.9976654, 4th-decimal agreement — `tools/int4_cosine_reference.py`; the published 0.9992 was a different input distribution). GPU t2i smoke valid + on-prompt (`assets/smoke_t2i_fox_int4.png`): load 77.8 s, 2.8 s/step, **peak 52.5 GB** (fp32 T5 residency ≈22 GB — wrap-level eviction candidate). Gate: `swift run RunBernini --s6-gate`. LESSON: quantized matmuls route to Metal even under a CPU pin — quantized forwards must run wholly on the GPU stream |
| **S7** 🟡 | Engine wrap (`MLXBerniniR`) | **Wrap BUILT + offline-conformance green 2026-06-12** (`BerniniRPackage`: textToVideo + textToImage from one renderer; measured footprints bf16 91 GB / int4 53 GB; per-step C13 cancellation; PNG/H.264-mp4 canonical artifacts; 6 conformance tests). `APP-VALIDATION.md` entry written. REMAINING: hand-link `MLXBerniniR` into the Testing app target (manual Xcode step), then the live engine-seam run |

## Validation doctrine (per the mlx-porting skill)

- **CPU stream for all numeric gates** (`mx.set_default_device(cpu)` equivalent) — GPU fp32
  matmul noise ≈8e-4/op compounds over 40 blocks and masks real bugs.
- Fixtures: `.npy` dumps from the Python oracle (donor's `NumpyArray.swift` + fixture-dump
  script pattern; oracle has `tests/fixtures/natural_256.png` + `_research/configs/transformer_keyshapes.json`).
- **`xcodebuild test` for anything touching Metal** — `swift test` can't load the metallib
  (workspace-wide rule); offline structural tests stay on `xcrun swift test`.
- Check the GPU is free before parity/e2e gates (another port may be mid-run — the
  qwen25vl spec's SCAIL-2 constraint pattern).
- Long generations (40 steps × 3–4 forwards on a 2×14B): per-step `mx.eval` + cache
  clearing (`clear_cache()`; watch `get_active_memory()` for ratchet), and any >10-min run
  goes detached (`nohup … & disown`), never as a harness-tracked task.

## Memory & runtime envelope (from the oracle, M-series Max 96 GB)

- bf16: both experts resident ~57 GB + UMT5 ~11 GB + VAE → **~69 GB working set**. Fits the
  dev machine; manifest should declare it honestly (chipFloor high).
- int4: experts 2×8.4 GB; observed **peak ~32 GB** e2e. This is the realistic consumer config.
- VAE decode is the OOM lever past ~49 frames → streaming decode (S5) is not optional.

## MLXEngine integration (S7)

- Same-repo wrapper target **`MLXBerniniR`** (qwen25vl pattern; core stays engine-agnostic):
  `BerniniRPackage: ModelPackage` on `@InferenceActor`; manifest Apache-2.0 (weights:
  Bernini/Wan2.2; port: this repo) · footprints {bf16 ≈69 GB, int4 ≈32 GB} (`Quant.int4`) ·
  surfaces `textToVideo` + `textToImage` (one checkpoint, both surfaces — the "one model,
  N surfaces" premise actually holds here, unlike Lance).
- `run()` must `Task.checkCancellation()` at denoise-step boundaries (eviction lever, C13).
- Editing surfaces: v2v/rv2v motivate **`videoEdit`**, r2v is reference-conditioned
  generation — candidate canonical field on `T2VRequest` (promotion trigger per CLAUDE.md:
  second package wanting the lever; until then `metaData`). `imageEdit`/`videoEdit` land at
  **contract 1.2.0** carried over from the Lance plan.
- Weights are already published: `mlx-community/Bernini-R-bf16` + `Bernini-R-int4`. The
  manifest's provenance + repo ids point there; first `load()` materializes into the engine
  model store (`mlx-package.json` marker convention).

## Non-goals

- The Bernini MLLM **planner** (not open-sourced; conditioning channel stubbed upstream — scope is renderer-only, same as the Python port).
- Training/LoRA. Audio. LongCat reuse beyond the donor files named above.
- No diffusers-format loading — converted ckpts only.
- No quant variants beyond int4/bf16 until S6 data justifies them.
