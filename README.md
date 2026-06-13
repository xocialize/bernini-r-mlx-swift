# bernini-r-mlx-swift

Swift port of [bernini-r-mlx](https://github.com/xocialize/bernini-r-mlx) — the Apple-MLX
port of **ByteDance Bernini-R**, whose renderer is provenance-audited **byte-stock
Wan2.2-T2V-A14B** (zero extra tensors; Segment-Aware 3D RoPE, source-VAE feature injection,
and APG guidance are runtime-only deltas). Provides `t2v` / `t2i` / `r2v` / `v2v` / `rv2v`
on Apple Silicon via [mlx-swift](https://github.com/ml-explore/mlx-swift).

> **Status: S2 (t2v denoise core) parity-locked.** Read [`PORTING-SPEC.md`](PORTING-SPEC.md)
> FIRST — it pins the oracle, the component donor, the key contract, the config truths, and
> the phase gates. Landed: S0 key contract (all 4 components, bf16+int4 headers) · WanVAE
> 1:1 translation (encode/decode parity on real fp32 weights) · umT5 lift (full-forward
> parity, fp32) · WanModel DiT 1:1 (1095-key match; **full 40-block forward parity vs the
> real 28.6 GB expert**) · Euler / DPM++(2M) / UniPC schedulers (trajectory bit-parity
> incl. corrector solves) · `BerniniRendererModel` + `denoiseT2V` (**4-step dual-expert
> CFG e2e + VAE decode matches the oracle golden**; gates: `BERNINI_R_PARITY_DIT=1`,
> `BERNINI_R_PARITY_E2E=1`). Next: prompt-level t2v/t2i entry (umT5 tokenizer) + GPU
> smoke (S2b), then SA-3D RoPE + multiseg (S3). **All since landed: S3 (SA-3D + multiseg,
> real-expert parity) · S4 (r2v/v2v/rv2v samplers, BIT-EXACT) · S5 (streaming decode,
> BIT-IDENTICAL, the pipeline decode path) · S6 (int4, cosine 0.9977 cross-validated) ·
> S7 wrap (`MLXBerniniR`: `BerniniRPackage` — textToVideo + textToImage, offline-conformance
> green; live engine-seam validation pending the manual app-target link).** Heavy/Metal gates
> run as CLI modes: `swift run RunBernini --s4-gate | --s5-gate | --s6-gate`.

> **Speed:** `.fast` request mode = DPM++(2M) at 16 steps — **2.53× faster** than the 40-step
> UniPC default at near-identical quality (int4, 17-frame t2v: 415.6 s vs 1049.7 s, same seed
> + 58 GB peak, 2026-06-12). CLI: `--solver dpm++ --steps 16`. The quality path stays the
> default (`.quality` / no mode).

> **Lightning (4-step, real-time):** the `lightx2v/Wan2.2-Lightning` 4-step distillation
> (Apache-2.0) merged into the experts → **CFG-free, ~35× faster denoise** (17-frame 832×480:
> 67.5 s vs ~1050 s at 40-step UniPC; visual quality holds). `BerniniRConfiguration.lightning`
> · `RunBernini --lightning`. Merge tool: `tools/merge_lightning.py` (key-matched to our
> linears; merged ckpt keeps the 1095-key contract → loads unchanged).
>
> ![lightning 4-step](assets/smoke_lightning_fox.png)

**S2b GPU smoke (2026-06-12):** real-prompt t2i on GPU via plain `swift run RunBernini`
(no metallib issue under the SPM CLI; weight loads must ride the CPU stream — see
`WeightLoader.loadVerifiedSafetensors`). 832x480, 40 steps: **load 142.5 s** (archive
disk) · **2.7 s/step, 107.7 s generate** · **peak 90.8 GB** (bf16, both experts +
fp32 umT5 resident; memory flat across steps).

![t2i smoke](assets/smoke_t2i_fox.png)

| | |
|---|---|
| Backbone | Wan2.2-T2V-A14B dual expert (40L · dim 5120 · 40H · ffn 13824) |
| VAE | 16-ch `AutoencoderKLWan` — Swift port lifted from `longcat-avatar-mlx-swift` (parity-locked) |
| Text encoder | UMT5-xxl |
| Scheduler | FlowUniPC (`bh2`, shift 3.0), expert boundary 0.875 |
| Parity oracle | `/Volumes/DEV_ARCHIVE/bernini-r-mlx` (Python MLX, e2e-validated, int4 cosine 0.9992) |
| Weights | [mlx-community/Bernini-R-bf16](https://huggingface.co/mlx-community/Bernini-R-bf16) · [mlx-community/Bernini-R-int4](https://huggingface.co/mlx-community/Bernini-R-int4) — consumed verbatim |

Targets: `BerniniR` (engine-agnostic core) · `MLXBerniniR` (MLXEngine `ModelPackage` wrapper,
`textToVideo` + `textToImage`).

License: Apache-2.0 (Bernini, Wan2.2 acknowledged upstream — see NOTICE).
