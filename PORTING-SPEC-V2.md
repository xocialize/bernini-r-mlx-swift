# bernini-r-mlx-swift — Bernini-v2 planner plane porting spec

**Goal:** extend `bernini-r-mlx-swift` with the full unified Bernini
(`ByteDance/Bernini-Diffusers-v2`, Apache-2.0) — the MLLM semantic-planner plane the June port
stubbed, plus the retrained A14B experts. NOT a new package: a v2 checkpoint family + planner
modes on the existing `BerniniRPackage`. Program: ENHANCEMENTS **E7** / bridge **AB-T-0065**;
scoping receipts **AB-R-0097** + `WAN_TESTING/companion/ENH-wan-bernini-v2-planner.md`.

Oracle = `bytedance/Bernini` @ main (mirrored `/Volumes/DEV_ARCHIVE/bernini-v2/Bernini-upstream`);
masters `/Volumes/DEV_ARCHIVE/bernini-v2/Bernini-Diffusers-v2` (pure-T5 shards 26–31 skipped).
Conversion: `bernini-r-mlx/scripts/convert_bernini_v2.py` (name-level bijective gate vs the
published expert layout PASSED pre-download: 1083 block keys exact + 12 globals covered by
`sanitize_wan_transformer_weights`).

## Component inventory (what Swift must load)

| File (ckpt-bf16) | Module | Keys | Notes |
|---|---|---|---|
| `high_noise_model.safetensors` | existing `WanModel` (high expert) | 1095 | RETRAINED vs Bernini-R — same arch/config (40L, 40×128, ffn 13824) |
| `low_noise_model.safetensors` | existing `WanModel` (low expert) | 1095 | ditto |
| `mllm/model.safetensors` (+ configs) | Qwen2.5-VL-7.7B planner | 729 | HF layout (`model.*`, `visual.*`, `lm_head.weight`); Bernini-trained (`scratch_mllm:true`), NOT stock Qwen |
| `vit_decoder.safetensors` | `DiffLossFM` | 140 | `net.*` = SimpleMLPAdaLN width 4096, depth 16, in/out 3584, z 3584 (1.34B) |
| `planner_glue.safetensors` | connector + mask tokens | 13 | keys verbatim upstream: `connector.proj_gen.{0,2,3}.*`, `connector.pred_vit.{0,2,4}.*`, `mask_tokens` [1,4096,3584] |
| `t5_encoder.safetensors`, `vae.safetensors` | existing umT5 / WanVAE | — | copied from Bernini-R (bit-identical, AB-R-0097) |

v2 config deltas vs our `BerniniConfig`: `boundary_ratio 0.417` (train-only), `switch_dit_boundary
0.875` (same), `shift 3.0` (same), `use_src_id_rotary_emb true` (have), **new:**
`interpolate_src_id true`, `max_trained_src_id 5` (src-id assignment when segments > trained
range — evenly spread into [1,5] via linspace instead of extrapolating; see `_make_sids` in
`wan_diffusion.py sample`), `num_mask_token 4096`, `t5_combine_type concat_with_zero_init`,
`feature_type_from_stage_one masked_tgt_embed_with_qwen_txt_vit_tokens`, `target_fps 16`,
`t5_max_sequence_length 512`.

## The planner inference graph (oracle: `bernini/pipeline.py sample_vit_embed` + `models/bernini.py`)

MAR/MaskGIT — **forward-only, no AR generation, no KV cache**:

1. **Assemble inputs_embeds** (`format_mllm_inputs_embeds`): token embeds from `input_ids`;
   visual features scattered into `visual_input_mask | visual_output_mask` positions. Then
   `post_process_input_embeds(inference=True)`: ALL 4096 target-ViT positions overwritten with
   `mask_tokens[:, :1]` (the FIRST mask token broadcast — not the full table).

   **The 4D mask is simple and Swift-buildable** (`data/utils/attention_utils.py
   build_custom_attention_mask(token_type, token_segment_ids)` → additive `(B,L,L)` 0/−inf):
   token types 0=text · 1=planning-p · 2=image-input · 3=output-o; every query sees prior
   t/i **causally**; p queries additionally see p of the SAME segment id **bidirectionally**;
   o queries additionally see o of the same id bidirectionally. Port the builder, verify
   against the `00_*_attention_mask_4d` fixtures.
2. **Three streams**: cond / uncond / imgcond — each its own `input_embeds`, 3D M-RoPE
   `position_ids` `(dim,bs,L)` after transpose, and a **4D attention mask**. (Mask/position
   construction lives in `bernini/data/bernini_process.py` + `data_utils.py` — read at
   implementation time; it is the prompt-template/packing layer.)
3. **MaskGIT loop** (`planning_step=25` default at `__call__`, cosine schedule
   `cos(π/2·(s+1)/S)`, shuffled reveal order): per step,
   - 3 × Qwen2.5-VL forwards (one per stream), tap **`hidden_states[-2]`** (penultimate layer —
     the classic N-1 trap, mind pitfall #7),
   - `connector.for_vit` (3584→3584 MLP: Linear-GELU-Linear-RMSNorm-Linear) on the ViT
     positions of each stream,
   - reveal-set selection (`mask_len = max(1, min(remaining-1, floor(N·ratio)))`; last step
     predicts ALL remaining),
   - `DiffLossFM.sample` on the revealed positions: euler flow-match, **`vit_denoising_step=3`
     at the production `__call__` level** (the inner `sample_vit_embed` signature default of 1 is
     NOT the shipped value), **shared noise across CFG streams** (`randn(N//3)` then `cat` ×3),
     3-stream guidance `forward_with_txt_img_cfg` (txt 1.4 / img 1.2),
   - write sampled embeds back into ALL THREE streams' input_embeds at the revealed positions.
4. **2 final forwards** (cond/uncond) → `feat_from_planner_to_renderer(inference=True)`:
   context mask = text/non-vit tokens ∪ predicted-vit tokens; `connector.for_gen`
   (3584→4096: Linear-GELU-RMSNorm-Linear) → `diff_mllm_contexts` + txt/vit submasks.
5. **Context variants** (`feature_type = masked_tgt_embed_with_qwen_txt_vit_tokens` → the ELSE
   branch in `sample_vit_embed`'s tail): `wtxt_wvit` = cond contexts; `wtxt_wovit` = cond[txt
   mask]; `wotxt_wvit` = cond[vit mask]; `wotxt_wovit` = uncond[txt mask].
6. **T5 concat** (`pipeline.__call__`): `wtxt_*` variants get `cat([t5_embeds(prompt), ctx], 1)`;
   `wotxt_*` get `cat([neg_t5_embeds, ctx], 1)` — neg prompt through umT5
   (`get_t5_text_embeddings_sample`, truncation at 512, NFKC discipline applies as everywhere).

Planner cost: 25×3 + 2 = **77 MLLM forwards** at seqlen ≈ text + 4096 + visual-input tokens.
Port the defaults verbatim; `planning_step` is a knob but NOT at parity time.

## DiffLossFM (oracle: `models/diffloss_fm.py`, 460 LOC — clean port)

- `SimpleMLPAdaLN`: input proj 3584→4096, `TimestepEmbedder` (sinusoidal 256 → MLP 4096),
  cond proj z 3584→4096, `depth=16` ResBlocks (LayerNorm-affine-free → modulate(shift,scale) →
  Linear-SiLU-Linear → gate; adaLN produces 3×4096 per block from SiLU(y)), `FinalLayer`
  (norm → modulate → zero-init Linear 4096→3584). Qwen2RMSNorm eps 1e-6 in connector only.
- `FlowMatchScheduler` (`models/scheduler.py`): shift 2.0 (v2 `clip_diff_cfg.shift`), sigma_max
  1.0, sigma_min 0.003/1.002, `extra_one_step=True`, `num_inference_steps` from cfg (100) but
  **`sample()` re-sets timesteps to the CALLER's `num_inference_steps` = `vit_denoising_step`
  (1)** — one euler step in production.
- CFG batching: noise generated once for N//3 rows then tiled ×3 (or ×2) — **RNG parity point:
  inject the SAME noise rows in Swift** (numpy-generated fixture; `mx.random` ≠ torch).
- `forward_with_txt_img_cfg` (3-way): eps = uncond + img_cfg·(imgcond−uncond) +
  txt_cfg·(cond−imgcond) on the channel-concat convention — read the exact split at
  implementation time (uses torch.split thirds).

## Renderer additions (oracle: `models/wan_diffusion.py`)

`GEN_Wanx22.sample_bernini_wvitcfg` — the v2 denoise entry (UniPC when `use_unipc`, shift 3.0;
noise via seeded CPU `torch.Generator` + `randn_tensor` in fp32, patchified `b (t h w) (ph pw c)`
layout with ph=pw=2 — matches our existing packing):

- Latent combos per step: `wovae` (noisy only) / `wimgvae` / `wvidvae` / `wvae` (img+vid+noisy),
  each with src-id RoPE (`patch_vae_latent`, source ids via `_make_sids`, noisy target id 0) —
  the machinery our editing surfaces already implement (`forwardMultiseg`).
- `sample_one_step` guidance (mode `default`/`rv2v_wapg` family): cascaded deltas with
  zero-omega short-circuits —
  `base = ε(wovae, wotxt_wovit)`; `ε_V = ε(wvidvae, wotxt_wovit)` if ω_vid>0 else base;
  `ε_VI = ε(wvae, wotxt_wovit)` if ω_img>0 else ε_V; `ε_VTI = ε(wvae, wtxt_wovit)` if ω_txt>0
  else ε_VI; `ε_VTIC = ε(wvae, wtxt_wvit)` if ω_tgt>0 else ε_VTI;
  `ε̂ = base + ω_vid·Δ_V + ω_img·Δ_I + ω_txt·Δ_T + ω_tgt·Δ_C` (each Δ optionally
  APG-projected in `*_wapg` modes — `apg_delta` with ref = previous rung, norm_threshold 50).
  Up to **5 DiT forwards/step** at full omegas. Expert switch at t≥0.875·1000 as today, with
  post-switch omega scaling by `omega_scale`.
- The t2i/t2v-no-source case degenerates to combos {wovae, wvae=wovae} — verify the code path
  at implementation (source lists empty).

**This is an EXTENSION of the S4-gated sampler family** — same shared_step/multiseg skeleton,
one new delta rung (ω_tgt, the planner-vit conditioning) + the 4-context text axis.

**v1→v2 `transformer_wan.py` diff (verified 2026-08-18, refs/Bernini vs Bernini-upstream): block
math UNCHANGED.** Inference-relevant deltas are exactly two: (a) **fractional src-id RoPE** — v2
computes the per-source rotary phase on the fly (`get_1d_rotary_pos_embed` at a float position,
fp64) so interpolated ids from `_make_sids` land inside the trained manifold; integer ids
reproduce the v1 precomputed-table behaviour exactly (⇒ Swift: generalize the SA3D src-id phase
from table-lookup to computed-at-float; parity vs v1 unchanged for integer ids). (b) trivial
`patch_vae_embedding` helper for pre-packed source patches. Everything else is training-only
(gradient checkpointing, sequence-parallel padding). Our parity-locked WanModel/forwardMultiseg
carries over as-is.

## Swift port order (tasks #5–#6)

1. **Rung decision** (at #5 start): mlx-vlm Qwen2.5-VL exists as the Python-MLX backbone donor;
   build the throwaway planner-harness scratchpad in `/Volumes/DEV_ARCHIVE/bernini-v2/measure/`
   ONLY if the torch-oracle goldens prove insufficient to localize Swift breaks (default: skip
   the rung — granular per-stage goldens first, per mlx-porting doctrine).
2. **Golden fixtures** (task #3, torch oracle, planner-only fits the 128 GB box in bf16):
   seed-injected run at `planning_step=2`, `num_mask_token` full, tiny text; dump per stage —
   inputs_embeds post-mask, per-step hidden_states[-2] (3 streams), for_vit outs, revealed
   index sets + noise rows, DiffLossFM samples, final for_gen contexts + submasks, T5-concat
   contexts, and (renderer) per-step ε per rung at 4-step denoise on a small grid. `.npy`,
   fp32, CPU-comparable.
3. **Qwen2.5-VL driver** — recon done 2026-08-18 against
   `mlx-swift-lm/Libraries/MLXVLM/Models/Qwen25VL.swift`:
   - `inputEmbedding:` injection — **native** (`Qwen25Model.callAsFunction`).
   - `positionIds:` — **native**, exactly the `[3, batch, seq]` M-RoPE layout Bernini passes;
     the attention's `mropeCosSin` covers the 3-axis path.
   - custom 4D mask — **NOT exposed** (mask built internally via `createAttentionMask`,
     causal-only). Bernini needs the processor's `attention_mask_4d` injected.
   - `hidden_states[-2]` tap — **NOT exposed** (returns after final norm + lm_head).
   - ALL relevant classes are `fileprivate` inside `private enum Language` → **donor-LIFT the
     file** into the bernini package (per `swift-port-parity.md` lift rules), don't subclass.
     The lifted planner forward: run `layers[0..<(N-1)]` with the injected additive mask +
     positions, return unnormed h (that IS `hidden_states[-2]`: embeddings + per-layer outputs
     list, index -2 = layer N-1 output, pre-final-norm) — skip final norm, last layer, lm_head.
   - Vision tower (`Vision` enum, also fileprivate): needed only for real image/video INPUTS
     (r2v/v2v/i2i). Phase-1 t2v/t2i skips the ViT — output-ViT positions are all overwritten by
     `mask_tokens` (fixture 01 verifies); only token COUNTS from `grid_thw` matter. Lift the
     vision tower in phase 2.
4. **DiffLossFM + MaskGIT loop**: direct port, RNG injected (shuffle order + FM noise as
   fixtures/seed streams — same discipline as S4's bit-identical RNG verification).
5. **wvitcfg sampler**: extend the S4 sampler set with the ω_tgt rung + 4-context axis on
   `forwardMultiseg`; UniPC shift 3.0 path exists.
6. **Config/package**: `BerniniRConfiguration` v2 variants (`.v2bf16`, `.v2int4` — int4 scope =
   experts only per S6; planner mllm int8 candidate LATER, gate first); WeightSourcing globs
   per component; planner phase evict → render phase (residentBytes = max(phase), re-measure).

## Gates (S-numbering continues)

- **V0** conversion: bijective key gate per component vs upstream index (name-level PASSED
  2026-08-18 pre-download for experts; re-run on real files post-convert + fingerprint spot-check
  vs masters).
- **V1** planner parity: per-stage max-abs vs goldens (CPU stream, fp32) — embeds assembly,
  one MLLM forward per stream, for_vit/for_gen, DiffLossFM 1-step (injected noise), one full
  MaskGIT step, full 2-step plan.
- **V2** renderer parity: ε per guidance rung (4-step, injected noise) — extends the S4 gate.
- **V3** e2e: planner-conditioned t2i/t2v vs oracle render at matched seed (decoded-output
  eyeball + PSNR band), largest production grid per tier.
- **V4** quant: int4 experts per-pass cosine ≥0.99 (S6 discipline); planner stays bf16 v1.
- **V5** engine: C0–C13 + MAT + CAN unchanged-green; app-seam footprint re-measure
  (`APP-VALIDATION-WAN.md` entry).

## Standing constraints

- fp32 DiT at video seqLen (`ditDType .float32` default) unchanged; planner runs bf16 (its own
  gate decides if any sub-block needs fp32 — watch the FinalLayer zero-init + 1-step euler).
- NFKC the negative prompt (wan-core `cleanText`) — v2 uses the same Chinese default negative.
- BlockStreamer: planner NEVER binds a streamer (hand-driven forwards); editing multiseg stays
  refuse-while-streamed; the wvitcfg t2v path routes `runBlocks` only if it drops multiseg —
  it does NOT (4 latent combos) → **v2 planner modes REFUSE while streamed** until the
  group-window API lands.
- Deviation doctrine: reference defaults verbatim — the PRODUCTION defaults are the
  `pipeline.__call__` signature values, not inner-function defaults: `planning_step 25`,
  `vit_denoising_step 3`, `vit_txt_cfg 1.4` / `vit_img_cfg 1.2`, `omega_vid 3.0` /
  `omega_img 3.0` / `omega_txt 4.0` / `omega_tgt 4.0` / `omega_scale 0.75`,
  `num_inference_steps 40`, `flow_shift 5.0` (passed INTO `sample_bernini_wvitcfg`; the config
  `shift 3.0` is a different plane), `eta 0.5`, `momentum -0.5`, `norm_threshold (50,50)`,
  `guidance_mode "rv2v"`, `fps 16` (`vit_fps = fps//8`), `seed` seeds random+np+torch together
  (np drives the MaskGIT shuffle). Per-task overrides come from `bernini/data/bernini_template.py`
  / `cli.py` — check per task at implementation. Knobs only after V3.
