// t2v denoising core — isomorphic to the dual-expert CFG path of
// mlx_video/models/wan_2/generate.py::generate_video (mlx-video @ 87db56a),
// which the Bernini oracle's t2v/t2i reuse verbatim. The I2V branches
// (mask-blend, channel-concat) and mx.compile are not ported: Bernini-R never
// reaches them, and Swift MLX traces lazily without an explicit compile step.
//
// Conditioning enters as raw UMT5 features so the text encoder stays a
// separate, separately-parity-locked stage (and the e2e gate can inject
// fixtures). t2i is t2v with one frame.

import Foundation
import MLX

public enum SchedulerKind: String, Sendable {
    case euler
    case dpmpp = "dpm++"
    case unipc
}

public struct T2VOptions: Sendable {
    public var steps: Int
    public var shift: Double
    /// (low, high) guidance scales — config.sample_guide_scale order:
    /// index 0 applies below the boundary, index 1 at/above it.
    public var guideScale: (Double, Double)
    public var scheduler: SchedulerKind
    /// CFG-free generation (B=1, no uncond pass). The Lightning 4-step
    /// distillation runs without the CFG trick; `guideScale` is then unused.
    public var noCFG: Bool

    public init(
        steps: Int = 40, shift: Double = 3.0,
        guideScale: (Double, Double) = (3.0, 4.0),
        scheduler: SchedulerKind = .unipc,
        noCFG: Bool = false
    ) {
        self.steps = steps
        self.shift = shift
        self.guideScale = guideScale
        self.scheduler = scheduler
        self.noCFG = noCFG
    }

    public static func fromConfig(_ config: WanConfig) -> T2VOptions {
        T2VOptions(
            steps: config.sampleSteps,
            shift: config.sampleShift,
            guideScale: (config.sampleGuideScale[0], config.sampleGuideScale[1]))
    }

    /// lightx2v Wan2.2-Lightning 4-step recipe: euler · shift 5.0 · CFG-free.
    /// (Expert switch stays the timestep boundary 875 — at shift 5 the 4
    /// timesteps split 2 high / 2 low, matching the reference workflow.)
    public static var lightning: T2VOptions {
        T2VOptions(steps: 4, shift: 5.0, scheduler: .euler, noCFG: true)
    }
}

/// Run the dual-expert CFG denoising loop and return the final latent
/// [C, T_lat, H_lat, W_lat].
///
/// - contextCond / contextNull: raw UMT5 features [L, text_dim] for the
///   prompt and the negative prompt.
/// - noise: initial latent noise [C, T_lat, H_lat, W_lat] (caller-seeded so
///   parity tests can inject the oracle's bytes).
/// - onStep: called after each step's eval with (step, totalSteps, latents) —
///   the engine wrap's cancellation checkpoint (C13), progress hook, and the
///   parity tests' per-step probe.
public func denoiseT2V(
    renderer: BerniniRendererModel,
    contextCond: MLXArray,
    contextNull: MLXArray,
    noise: MLXArray,
    options: T2VOptions = T2VOptions(),
    onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
) rethrows -> MLXArray {
    let config = renderer.config
    let high = renderer.highNoiseExpert
    let low = renderer.lowNoiseExpert

    // Pre-embed conditioning per expert (each has its own text MLP). CFG →
    // [cond, uncond] (B=2); CFG-free (Lightning) → [cond] only (B=1, no uncond).
    let textIn = options.noCFG ? [contextCond] : [contextCond, contextNull]
    let contextCfgHigh = high.embedText(textIn)
    let contextCfgLow = low.embedText(textIn)
    eval(contextCfgHigh, contextCfgLow)

    // Precompute cross-attention K/V caches (constant across all steps)
    let crossKVHigh = high.prepareCrossKV(contextCfgHigh)
    let crossKVLow = low.prepareCrossKV(contextCfgLow)

    // Precompute RoPE frequencies (grid sizes are constant across all steps)
    let (c, tLat, hLat, wLat) = (noise.dim(0), noise.dim(1), noise.dim(2), noise.dim(3))
    precondition(c == config.inDim)
    let fGrid = tLat / config.patchSize[0]
    let hGrid = hLat / config.patchSize[1]
    let wGrid = wLat / config.patchSize[2]
    let grid = (fGrid, hGrid, wGrid)
    let ropeGridSizes = options.noCFG ? [grid] : [grid, grid]
    let ropeCosSinHigh = high.prepareRope(ropeGridSizes)
    let ropeCosSinLow = low.prepareRope(ropeGridSizes)
    let seqLen = fGrid * hGrid * wGrid

    // Setup scheduler
    let unipc: FlowUniPCScheduler? =
        options.scheduler == .unipc
        ? FlowUniPCScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    let euler: FlowMatchEulerScheduler? =
        options.scheduler == .euler
        ? FlowMatchEulerScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    let dpmpp: FlowDPMPP2MScheduler? =
        options.scheduler == .dpmpp
        ? FlowDPMPP2MScheduler(numTrainTimesteps: config.numTrainTimesteps) : nil
    unipc?.setTimesteps(options.steps, shift: options.shift)
    euler?.setTimesteps(options.steps, shift: options.shift)
    dpmpp?.setTimesteps(options.steps, shift: options.shift)
    let timesteps = unipc?.timesteps ?? euler?.timesteps ?? dpmpp!.timesteps

    let boundary = config.boundaryTimestep

    var latents = noise

    // Diffusion loop — CFG batches cond + uncond into a single B=2 forward
    for i in 0..<options.steps {
        let timestepVal = Double(timesteps[i])
        let isHigh = timestepVal >= boundary

        let model = isHigh ? high : low
        let kv = isHigh ? crossKVHigh : crossKVLow
        let rcs = isHigh ? ropeCosSinHigh : ropeCosSinLow
        let ctx = isHigh ? contextCfgHigh : contextCfgLow

        let noisePred: MLXArray
        if options.noCFG {
            // CFG-free (Lightning): single forward, the prediction IS the output.
            let preds = model(
                [latents], t: MLXArray([Float(timestepVal)]), context: .embedded(ctx),
                seqLen: seqLen, crossKVCaches: kv, ropeCosSin: rcs)
            noisePred = preds[0]
        } else {
            // CFG: cond + uncond batched into one B=2 forward, then combined.
            let gs = isHigh ? options.guideScale.1 : options.guideScale.0
            let tBatch = MLXArray([Float(timestepVal), Float(timestepVal)])
            let preds = model(
                [latents, latents], t: tBatch, context: .embedded(ctx), seqLen: seqLen,
                crossKVCaches: kv, ropeCosSin: rcs)
            noisePred = preds[1] + Float(gs) * (preds[0] - preds[1])
        }

        let stepped: MLXArray
        let predB = noisePred.expandedDimensions(axis: 0)
        let sampleB = latents.expandedDimensions(axis: 0)
        if let unipc {
            stepped = unipc.step(
                modelOutput: predB, timestep: Float(timestepVal), sample: sampleB)
        } else if let euler {
            stepped = euler.step(
                modelOutput: predB, timestep: Float(timestepVal), sample: sampleB)
        } else {
            stepped = dpmpp!.step(
                modelOutput: predB, timestep: Float(timestepVal), sample: sampleB)
        }
        latents = stepped.squeezed(axis: 0)

        eval(latents)
        // Metal buffer-cache discipline for the multi-hour production configs:
        // freed per-step workspace otherwise ratchets RSS until SIGKILL.
        MLX.GPU.clearCache()
        try onStep?(i, options.steps, latents)
    }

    return latents
}
