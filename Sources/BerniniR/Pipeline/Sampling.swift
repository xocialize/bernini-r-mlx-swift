// 1:1 translation of bernini_r_mlx/sampling.py — r2v (chained Adaptive
// Projected Guidance over UMT5-only conditioning) and the plain-CFG
// video-editing sampler (v2v / v2v_chain / rv2v). Per step: 3 (r2v) or 2–4
// (edit) multiseg expert forwards; SA-3D RoPE keeps reference/source segments
// separated from the target, which is what preserves subject identity.

import Foundation
import MLX
import MLXRandom

// APG reduces over (channel, height, width) per (batch, frame): upstream
// norm dims [-1,-2,-4] on [b,c,t,h,w] -> axes (1,3,4), keepdims.
private let apgAxes = [1, 3, 4]

final class MomentumBuffer {
    let momentum: Float
    var runningAverage: MLXArray = MLXArray(Float(0))

    init(_ momentum: Float) {
        self.momentum = momentum
    }

    func update(_ v: MLXArray) {
        runningAverage = v + momentum * runningAverage
    }
}

private func l2(_ x: MLXArray) -> MLXArray {
    sqrt(sum(x * x, axes: apgAxes, keepDims: true))
}

func normalizeDiff(
    _ diff: MLXArray, basePred: MLXArray, momentumBuffer: MomentumBuffer?,
    eta: Float, normThreshold: Float
) -> MLXArray {
    var diff = diff
    if let momentumBuffer {
        momentumBuffer.update(diff)
        diff = momentumBuffer.runningAverage
    }
    if normThreshold > 0 {
        let diffNorm = l2(diff)
        diff = diff * minimum(MLXArray.ones(like: diffNorm), normThreshold / diffNorm)
    }
    let v1 = basePred / (l2(basePred) + 1e-12)
    let v0Parallel = sum(diff * v1, axes: apgAxes, keepDims: true) * v1
    let v0Orthogonal = diff - v0Parallel
    return v0Orthogonal + eta * v0Parallel
}

func normalizedGuidanceChain(
    predUncond: MLXArray, preds: [MLXArray], scales: [Float],
    momentumBuffers: [MomentumBuffer], eta: Float, normThresholds: [Float]
) -> MLXArray {
    let bases = [predUncond] + preds
    var result = predUncond
    for (i, cond) in preds.enumerated() {
        let nd = normalizeDiff(
            cond - bases[i], basePred: cond, momentumBuffer: momentumBuffers[i],
            eta: eta, normThreshold: normThresholds[i])
        result = result + scales[i] * nd
    }
    return result
}

/// Run r2v_apg sampling: no source video, K reference images; per step 3
/// expert forwards — ∅ (target only), I (refs + target), TI (refs + target,
/// with text) — combined by chained APG in x0 space, then a UniPC velocity
/// step. Returns the denoised target latent [C, T_lat, H_lat, W_lat].
/// - injectedNoise: parity-test override for the seeded initial latent.
public func r2vSample(
    high: WanModel,
    low: WanModel,
    refLatents: [MLXArray],  // reference VAE latents, each [C, 1, H, W]
    condCtxHigh: MLXArray,  // pre-embedded UMT5 text per expert [1, textLen, dim]
    condCtxLow: MLXArray,
    uncondCtxHigh: MLXArray,
    uncondCtxLow: MLXArray,
    targetShape: [Int],  // (C, T_lat, H_lat, W_lat)
    headDim: Int,
    boundaryTimestep: Double,
    steps: Int = 40,
    shift: Double = 3.0,
    omegaI: Float = 3.0,
    omegaTI: Float = 4.0,
    omegaScale: Float = 0.75,
    eta: Float = 0.5,
    normThreshold: (Float, Float) = (50.0, 50.0),
    momentum: Float = -0.5,
    seed: UInt64 = 42,
    injectedNoise: MLXArray? = nil,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let sched = FlowUniPCScheduler()
    sched.setTimesteps(steps, shift: shift)
    let sigmas = sched.sigmas

    var latent: MLXArray
    if let injectedNoise {
        latent = injectedNoise
    } else {
        MLXRandom.seed(seed)
        latent = MLXRandom.normal(targetShape)  // sigma=1 start
    }

    let refSegments: [MultisegSegment] =
        refLatents.enumerated().map { (i, rl) in (rl, i + 1) }  // source_id 1..K
    let mb1 = MomentumBuffer(momentum)
    let mb2 = MomentumBuffer(momentum)
    var wI = omegaI
    var wTI = omegaTI
    var switched = false

    for i in 0..<steps {
        let t = Double(sched.timesteps[i])
        let model: WanModel
        let cCtx: MLXArray
        let uCtx: MLXArray
        if t >= boundaryTimestep {
            (model, cCtx, uCtx) = (high, condCtxHigh, uncondCtxHigh)
        } else {
            if !switched {
                wI *= omegaScale
                wTI *= omegaScale
                switched = true
            }
            (model, cCtx, uCtx) = (low, condCtxLow, uncondCtxLow)
        }

        let tt = MLXArray([Float(t)])
        let vUncond = forwardMultiseg(
            model: model, condSegments: [], targetLatent: latent, t: tt,
            context: uCtx, headDim: headDim)
        let vI = forwardMultiseg(
            model: model, condSegments: refSegments, targetLatent: latent, t: tt,
            context: uCtx, headDim: headDim)
        let vTI = forwardMultiseg(
            model: model, condSegments: refSegments, targetLatent: latent, t: tt,
            context: cCtx, headDim: headDim)

        let sigma = sigmas[i]
        let b = latent.expandedDimensions(axis: 0)  // [1,C,T,H,W] for APG norms
        let xUncond = b - sigma * vUncond.expandedDimensions(axis: 0)
        let xI = b - sigma * vI.expandedDimensions(axis: 0)
        let xTI = b - sigma * vTI.expandedDimensions(axis: 0)
        let xGuided = normalizedGuidanceChain(
            predUncond: xUncond, preds: [xI, xTI], scales: [wI, wTI],
            momentumBuffers: [mb1, mb2], eta: eta,
            normThresholds: [normThreshold.0, normThreshold.1])
        let vGuided = ((b - xGuided) / sigma).squeezed(axis: 0)  // back to velocity

        latent = sched.step(modelOutput: vGuided, timestep: Float(t), sample: latent)
        eval(latent)
        MLX.GPU.clearCache()
        try onStep?(i, steps, latent)
    }
    return latent
}

public enum EditGuidanceMode: String, Sendable {
    case v2v
    case v2vChain = "v2v_chain"
    case rv2v
}

/// Plain-CFG video-editing sampler (v2v / v2v_chain / rv2v).
///
/// Conditioning combos (segment source_ids match upstream exactly):
///     VI : video segs (1..nv) + ref segs (nv+1..nv+ni)   I : ref segs (1..ni)
///     V  : first video seg (1)                           none : target only
/// Guidance (velocity space; UniPC handles velocity->x0):
///     v2v       : ε_VI + ω_TI(ε_VTI − ε_VI)
///     v2v_chain : ε∅ + ω_V(ε_V−ε∅) + ω_TI(ε_VTI−ε_V)
///     rv2v      : ε∅ + ω_V(ε_V−ε∅) + ω_I(ε_VI−ε_V) + ω_TI(ε_VTI−ε_VI)
public func cfgEditSample(
    high: WanModel,
    low: WanModel,
    guidanceMode: EditGuidanceMode,
    videoLatents: [MLXArray],  // source video VAE latents, each [C, T, H, W]
    refLatents: [MLXArray],  // reference image latents, each [C, 1, H, W]
    condCtxHigh: MLXArray,
    condCtxLow: MLXArray,
    uncondCtxHigh: MLXArray,
    uncondCtxLow: MLXArray,
    targetShape: [Int],
    headDim: Int,
    boundaryTimestep: Double,
    steps: Int = 40,
    shift: Double = 3.0,
    omegaV: Float = 3.0,
    omegaI: Float = 3.0,
    omegaTI: Float = 4.0,
    omegaScale: Float = 0.75,
    seed: UInt64 = 42,
    injectedNoise: MLXArray? = nil,
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let sched = FlowUniPCScheduler()
    sched.setTimesteps(steps, shift: shift)

    var latent: MLXArray
    if let injectedNoise {
        latent = injectedNoise
    } else {
        MLXRandom.seed(seed)
        latent = MLXRandom.normal(targetShape)
    }

    let nv = videoLatents.count
    let viSegments: [MultisegSegment] =
        videoLatents.enumerated().map { (i, v) in (v, i + 1) }
        + refLatents.enumerated().map { (j, r) in (r, nv + 1 + j) }
    let vSegments: [MultisegSegment] = nv > 0 ? [(videoLatents[0], 1)] : []

    var wV = omegaV
    var wI = omegaI
    var wTI = omegaTI
    var switched = false

    for i in 0..<steps {
        let t = Double(sched.timesteps[i])
        let model: WanModel
        let cCtx: MLXArray
        let uCtx: MLXArray
        if t >= boundaryTimestep {
            (model, cCtx, uCtx) = (high, condCtxHigh, uncondCtxHigh)
        } else {
            if !switched {
                wV *= omegaScale
                wI *= omegaScale
                wTI *= omegaScale
                switched = true
            }
            (model, cCtx, uCtx) = (low, condCtxLow, uncondCtxLow)
        }

        let tt = MLXArray([Float(t)])
        func fwd(_ segs: [MultisegSegment], _ ctx: MLXArray) -> MLXArray {
            forwardMultiseg(
                model: model, condSegments: segs, targetLatent: latent, t: tt,
                context: ctx, headDim: headDim)
        }

        let noisePred: MLXArray
        switch guidanceMode {
        case .v2v:
            let eVI = fwd(viSegments, uCtx)
            let eVTI = fwd(viSegments, cCtx)
            noisePred = eVI + wTI * (eVTI - eVI)
        case .v2vChain:
            let e0 = fwd([], uCtx)
            let eV = fwd(vSegments, uCtx)
            let eVTI = fwd(viSegments, cCtx)
            noisePred = e0 + wV * (eV - e0) + wTI * (eVTI - eV)
        case .rv2v:
            let e0 = fwd([], uCtx)
            let eV = fwd(vSegments, uCtx)
            let eVI = fwd(viSegments, uCtx)
            let eVTI = fwd(viSegments, cCtx)
            noisePred = e0 + wV * (eV - e0) + wI * (eVI - eV) + wTI * (eVTI - eVI)
        }

        latent = sched.step(modelOutput: noisePred, timestep: Float(t), sample: latent)
        eval(latent)
        MLX.GPU.clearCache()
        try onStep?(i, steps, latent)
    }
    return latent
}
