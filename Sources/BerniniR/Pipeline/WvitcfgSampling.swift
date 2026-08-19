//
//  WvitcfgSampling.swift
//  BerniniR — Bernini-v2 planner-conditioned renderer sampling (PORTING-SPEC-V2.md)
//
//  Port of upstream `GEN_Wanx22.sample_bernini_wvitcfg` + `sample_one_step` for the
//  production full-Bernini guidance mode `vae_txt_vit_wapg` (docs/bernini.md), on the
//  t2i/t2v NO-SOURCE path (source image/video combos land with the editing phase):
//
//    base  = ε(noisy, ctx_wotxt_wovit)
//    Δ_img = ε(wvae, ctx_wotxt_wovit) − base                (≡ 0 with no sources —
//                                                            the wvae combo IS the
//                                                            noisy-only combo)
//    Δ_txt = ε(wvae, ctx_wtxt_wovit) − ε(wvae, ctx_wotxt_wovit)
//    Δ_vit = ε(wvae, ctx_wtxt_wvit)  − ε(wvae, ctx_wtxt_wovit)
//    ε̂ = base + ω_img·apg(Δ_img) + ω_txt·apg(Δ_txt) + ω_tgt·apg(Δ_vit)
//
//  with `apg_delta` (arXiv 2410.02416; parallel 0.2 / orthogonal 1.0) and the
//  renderer's own FlowMatch grid (shift 5.0, sigma_min 0, no extra step; torch holds
//  it in bf16 — same grid emulation as the planner's scheduler). Contexts arrive RAW
//  (T5+planner concat, 4096-d) and are embedded per expert via `embedText`, exactly
//  as `shared_step` feeds the transformer.
//
//  Expert switching at `t < boundary` with post-switch omega scaling mirrors
//  `sample_bernini_wvitcfg`; short parity runs stay in the high expert.
//

import Foundation
import MLX
import WanCore

/// Upstream `apg_delta`: project `delta` onto/against `ref` per batch row and
/// recombine with separate parallel/orthogonal scales.
public func apgDelta(
    _ delta: MLXArray, ref: MLXArray,
    parallelScale: Float = 0.2, orthogonalScale: Float = 1.0, eps: Float = 1e-8
) -> MLXArray {
    let b = delta.dim(0)
    let deltaF = delta.reshaped(b, -1)
    let refF = ref.reshaped(b, -1)
    let refNormSq = maximum((refF * refF).sum(axis: 1, keepDims: true), MLXArray(eps))
    let projCoeff = (deltaF * refF).sum(axis: 1, keepDims: true) / refNormSq
    let parallel = projCoeff * refF
    let orthogonal = deltaF - parallel
    return (parallelScale * parallel + orthogonalScale * orthogonal)
        .reshaped(delta.shape)
}

/// The four T5+planner context variants, RAW 4096-d (pre-`embedText`).
public struct WvitcfgContexts {
    public var wtxtWvit: MLXArray
    public var wtxtWovit: MLXArray
    public var wotxtWvit: MLXArray
    public var wotxtWovit: MLXArray

    public init(wtxtWvit: MLXArray, wtxtWovit: MLXArray,
                wotxtWvit: MLXArray, wotxtWovit: MLXArray) {
        self.wtxtWvit = wtxtWvit
        self.wtxtWovit = wtxtWovit
        self.wotxtWvit = wotxtWvit
        self.wotxtWovit = wotxtWovit
    }
}

/// Per-step instrumentation for the V2 gate (ε per rung + combined prediction).
public struct WvitcfgStepTrace {
    public let epsWtxtWvit: MLXArray
    public let base: MLXArray
    public let epsWtxtWovit: MLXArray
    public let noisePred: MLXArray
}

/// `sample_bernini_wvitcfg` (no-source path). Returns the final latent `[C, T, H, W]`.
public func wvitcfgSample(
    high: WanModel,
    low: WanModel?,
    contexts: WvitcfgContexts,
    targetShape: [Int],          // (C, T_lat, H_lat, W_lat)
    headDim: Int,
    boundaryTimestep: Double,
    steps: Int,
    flowShift: Float = 5.0,
    omegaTxt: Float = 4.0,
    omegaImg: Float = 3.0,
    omegaTgt: Float = 4.0,
    omegaScale: Float = 0.75,
    injectedNoise: MLXArray? = nil,       // unpacked [C, T, H, W]
    injectedSigmas: MLXArray? = nil,      // renderer bf16 grid (fixtures)
    injectedTimesteps: MLXArray? = nil,
    seed: UInt64 = 42,
    onStep: ((Int, WvitcfgStepTrace) throws -> Void)? = nil
) rethrows -> MLXArray {
    let scheduler = injectedSigmas.map {
        FlowMatchScheduler(sigmas: $0, timesteps: injectedTimesteps)
    } ?? FlowMatchScheduler(
        numInferenceSteps: steps, shift: flowShift, sigmaMin: 0, extraOneStep: false)

    var latent: MLXArray
    if let injectedNoise {
        latent = injectedNoise
    } else {
        MLXRandom.seed(seed)
        latent = MLXRandom.normal(targetShape)
    }

    // Pre-embed the four contexts per expert (shared_step: transformer applies
    // text_embedding internally; our WanModel exposes it as embedText).
    // v2 feeds EXACT-length contexts (upstream varlen cross-attn); the classic
    // pad-to-512 would add embedded-zero keys (wan-core >= 0.2.1).
    func embedAll(_ model: WanModel) -> [MLXArray] {
        [contexts.wtxtWvit, contexts.wtxtWovit, contexts.wotxtWvit, contexts.wotxtWovit]
            .map { model.embedTextUnpadded($0) }
    }
    let ctxHigh = embedAll(high)
    let ctxLow = low.map(embedAll)

    var wTxt = omegaTxt
    var wImg = omegaImg
    var wTgt = omegaTgt
    var switched = false

    for i in 0 ..< steps {
        let t = Double(scheduler.timesteps[i].item(Float.self))
        let model: WanModel
        let ctx: [MLXArray]
        if t >= boundaryTimestep || low == nil {
            (model, ctx) = (high, ctxHigh)
        } else {
            if !switched {
                wTxt *= omegaScale
                wImg *= omegaScale
                wTgt *= omegaScale
                switched = true
            }
            (model, ctx) = (low!, ctxLow!)
        }
        let tt = MLXArray([Float(t)])

        // No sources: the wvae combo IS the noisy-only combo — 3 unique forwards.
        func eps(_ context: MLXArray) -> MLXArray {
            forwardMultiseg(
                model: model, condSegments: [], targetLatent: latent, t: tt,
                context: context, headDim: headDim)
        }
        let epsWtxtWvit = eps(ctx[0])
        let base = eps(ctx[3])          // wotxt_wovit on the wovae combo
        let epsWtxtWovit = eps(ctx[1])
        let epsImgRung = base           // ε(wvae, wotxt_wovit) ≡ base without sources

        let deltaImg = epsImgRung - base                 // exactly zero, kept for shape
        let deltaTxt = epsWtxtWovit - epsImgRung
        let deltaVit = epsWtxtWvit - epsWtxtWovit

        // Upstream projects over the whole (batch-1) latent — flatten to [1, N]
        // regardless of our unpacked [C,T,H,W] layout.
        func apg1(_ delta: MLXArray, ref: MLXArray) -> MLXArray {
            apgDelta(delta.reshaped(1, -1), ref: ref.reshaped(1, -1))
                .reshaped(delta.shape)
        }
        let noisePred = base
            + wImg * apg1(deltaImg, ref: epsImgRung)
            + wTxt * apg1(deltaTxt, ref: epsWtxtWovit)
            + wTgt * apg1(deltaVit, ref: epsWtxtWvit)

        try onStep?(i, WvitcfgStepTrace(
            epsWtxtWvit: epsWtxtWvit, base: base,
            epsWtxtWovit: epsWtxtWovit, noisePred: noisePred))

        latent = scheduler.step(modelOutput: noisePred, stepIndex: i, sample: latent)
        eval(latent)
    }
    return latent
}
