// r2v / v2v / rv2v editing entry points — the Swift mirror of the oracle's
// pipeline_mlx.py editing section. Pixel tensors in (the wrapper decodes the
// canonical Image/Video artifacts → pixels; the core stays engine-agnostic),
// frames out. The samplers (`r2vSample` / `cfgEditSample`) are parity-locked
// bit-exact (S4); these methods are the preprocess → VAE-encode → sample →
// streaming-decode chain around them.

import Foundation
import MLX
import MLXRandom

extension BerniniPipeline {

    /// Pre-embedded cond/uncond UMT5 contexts per expert (the oracle's `_edit_setup` text half).
    private func editContexts(prompt: String, negative: String)
        -> (condHigh: MLXArray, condLow: MLXArray, uncondHigh: MLXArray, uncondLow: MLXArray)
    {
        let rawC = encodeText(
            encoder: textEncoder, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
        let rawU = encodeText(
            encoder: textEncoder, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
        let high = renderer.highNoiseExpert
        let low = renderer.lowNoiseExpert
        let ctx = (high.embedText([rawC]), low.embedText([rawC]),
                   high.embedText([rawU]), low.embedText([rawU]))
        eval(ctx.0, ctx.1, ctx.2, ctx.3)
        return ctx
    }

    /// Target latent shape from the output geometry (VAE strides).
    private func targetShape(width: Int, height: Int, numFrames: Int) -> [Int] {
        [config.vaeZDim,
         (numFrames - 1) / config.vaeStride[0] + 1,
         height / config.vaeStride[1],
         width / config.vaeStride[2]]
    }

    /// VAE-encode reference image pixels ([1,3,1,H,W] in [-1,1]) → ref latents
    /// ([C,1,H_lat,W_lat], source_id 1..K).
    private func encodeRefs(_ referencePixels: [MLXArray]) -> [MLXArray] {
        referencePixels.map { vae.encode($0)[0] }
    }

    /// Reference-to-video (r2v): generate a video of the reference subject(s) following
    /// `prompt`. Chained APG over ∅ / I / TI with SA-3D RoPE separating the reference
    /// segments from the target (subject identity). Returns frames [1, 3, T, H, W] in [-1, 1].
    public func r2v(
        prompt: String,
        referencePixels: [MLXArray],
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        numFrames: Int = 49,
        steps: Int = 40,
        seed: UInt64 = 42,
        omegaI: Float = 3.0,
        omegaTI: Float = 4.0,
        eta: Float = 0.5,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt
        let ctx = editContexts(prompt: prompt, negative: negative)
        let refLatents = encodeRefs(referencePixels)

        let latent = try r2vSample(
            high: renderer.highNoiseExpert, low: renderer.lowNoiseExpert,
            refLatents: refLatents,
            condCtxHigh: ctx.condHigh, condCtxLow: ctx.condLow,
            uncondCtxHigh: ctx.uncondHigh, uncondCtxLow: ctx.uncondLow,
            targetShape: targetShape(width: width, height: height, numFrames: numFrames),
            headDim: config.headDim, boundaryTimestep: renderer.boundaryTimestep,
            steps: steps, shift: config.sampleShift,
            omegaI: omegaI, omegaTI: omegaTI, eta: eta, seed: seed, onStep: onStep)

        let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
        eval(frames)
        return frames
    }

    /// Prompt-based video editing (v2v / v2v_chain / rv2v): the source video (+ optional
    /// reference images) is injected as conditioning segment(s) and re-rendered toward
    /// `prompt`. Returns frames [1, 3, T, H, W] in [-1, 1].
    public func videoEdit(
        prompt: String,
        sourceVideoPixels: MLXArray,
        referencePixels: [MLXArray] = [],
        mode: EditGuidanceMode = .v2v,
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        numFrames: Int = 49,
        steps: Int = 40,
        seed: UInt64 = 42,
        omegaV: Float = 3.0,
        omegaI: Float = 3.0,
        omegaTI: Float = 4.0,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt
        let ctx = editContexts(prompt: prompt, negative: negative)
        let videoLatents = [vae.encode(sourceVideoPixels)[0]]
        let refLatents = encodeRefs(referencePixels)

        let latent = try cfgEditSample(
            high: renderer.highNoiseExpert, low: renderer.lowNoiseExpert,
            guidanceMode: mode, videoLatents: videoLatents, refLatents: refLatents,
            condCtxHigh: ctx.condHigh, condCtxLow: ctx.condLow,
            uncondCtxHigh: ctx.uncondHigh, uncondCtxLow: ctx.uncondLow,
            targetShape: targetShape(width: width, height: height, numFrames: numFrames),
            headDim: config.headDim, boundaryTimestep: renderer.boundaryTimestep,
            steps: steps, shift: config.sampleShift,
            omegaV: omegaV, omegaI: omegaI, omegaTI: omegaTI, seed: seed, onStep: onStep)

        let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
        eval(frames)
        return frames
    }
}
