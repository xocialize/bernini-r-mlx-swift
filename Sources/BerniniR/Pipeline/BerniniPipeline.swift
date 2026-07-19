// High-level t2v / t2i entry — the Swift mirror of the Bernini oracle's
// pipeline_mlx.py t2v/t2i (which wrap mlx-video's generate_video). Owns the
// component loads (renderer + VAE + UMT5 + tokenizer) and the prompt →
// frames path; saving/encoding artifacts is the caller's concern (the
// MLXEngine wrap returns canonical Video/Image; the smoke CLI writes PNGs).

import Foundation
import WanCore
import MLX
import MLXNN
import MLXRandom
import Tokenizers

public final class BerniniPipeline: @unchecked Sendable {
    public let config: WanConfig
    public let renderer: BerniniRendererModel
    public let vae: WanVAE
    /// Checkpoint dir — kept so umT5 can be (re)loaded on demand per request and
    /// evicted before denoise (the §2.4 T5-eviction lever), rather than held resident.
    public let modelDir: URL
    public let tokenizer: any Tokenizer

    public init(
        config: WanConfig,
        renderer: BerniniRendererModel,
        vae: WanVAE,
        modelDir: URL,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.renderer = renderer
        self.vae = vae
        self.modelDir = modelDir
        self.tokenizer = tokenizer
    }

    /// Load all components from a converted checkpoint directory (flat
    /// layout: {high_noise_model,low_noise_model,vae,t5_encoder}.safetensors
    /// + config.json). The tokenizer comes from google/umt5-xxl (HF), exactly
    /// like mlx-video.
    public static func fromPretrained(
        modelDir: URL, quantization: WanQuantization? = nil
    ) async throws -> BerniniPipeline {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))

        let renderer = try BerniniRendererModel.fromPretrained(
            modelDir: modelDir, quantization: quantization)

        let vae = WanVAE(zDim: config.vaeZDim, encoder: true)
        let vaeWeights = try Device.withDefaultDevice(.cpu) {
            let loaded = try MLX.loadArrays(
                url: modelDir.appendingPathComponent("vae.safetensors"))
            WeightLoader.materialize(loaded)
            return loaded
        }
        try vae.update(
            parameters: ModuleParameters.unflattened(vaeWeights),
            verify: [.noUnusedKeys])

        // umT5 is NOT loaded here — it's paged in per request and evicted before
        // denoise (see `withTextEncoder`), so it never co-resides with the heavy
        // denoise activations. Only the renderer + VAE stay resident.
        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return BerniniPipeline(
            config: config, renderer: renderer, vae: vae,
            modelDir: modelDir, tokenizer: tokenizer)
    }

    /// Load the fp32 umT5 encoder from the checkpoint (fp32 like mlx-video's
    /// `load_t5_encoder`). Loaded on demand, not held resident.
    private func loadTextEncoder() throws -> UMT5EncoderModel {
        let textEncoder = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(t5Weights)
        try textEncoder.update(
            parameters: ModuleParameters.unflattened(t5Weights),
            verify: [.noUnusedKeys])
        return textEncoder
    }

    /// §2.4 T5 eviction: load umT5, run `body` to produce its text contexts, then
    /// drop the encoder and reclaim its ~22 GB fp32 working set before returning —
    /// so the denoise loop never co-resides with the encoder.
    /// ⚠️ `body` MUST `eval` everything it returns; an un-eval'd lazy graph would
    /// keep the encoder weights alive past the `clearCache`, defeating the eviction.
    func withTextEncoder<R>(_ body: (UMT5EncoderModel) throws -> R) throws -> R {
        var encoder: UMT5EncoderModel? = try loadTextEncoder()
        let result = try body(encoder!)
        encoder = nil                 // drop the only strong ref → weights deallocate
        MLX.Memory.clearCache()          // return the freed buffers to the OS
        return result
    }

    /// Text-to-video. Returns decoded frames [1, 3, T, H, W] in [-1, 1].
    /// Defaults mirror the oracle: 832x480, 49 frames, config steps/shift/
    /// guide scales, config negative prompt.
    public func t2v(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        numFrames: Int = 49,
        steps: Int? = nil,
        shift: Double? = nil,
        guideScale: (Double, Double)? = nil,
        scheduler: SchedulerKind? = nil,
        lightning: Bool = false,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt

        // §2.4: page umT5 in, encode cond/uncond, evict it before denoise.
        let (contextCond, contextNull) = try withTextEncoder { enc -> (MLXArray, MLXArray) in
            let c = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
            let n = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
            eval(c, n)
            return (c, n)
        }

        // Latent geometry from the VAE strides (temporal 1 + (F-1)/4, spatial /8)
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]

        if let seed {
            MLXRandom.seed(seed)
        }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        // Lightning preset (euler/shift5/4-step/CFG-free) requires the merged
        // Lightning checkpoint; otherwise the config-default CFG path.
        var options = lightning ? T2VOptions.lightning : T2VOptions.fromConfig(config)
        if let steps { options.steps = steps }
        if let shift { options.shift = shift }
        if let guideScale { options.guideScale = guideScale }
        if let scheduler { options.scheduler = scheduler }

        let latent = try denoiseT2V(
            renderer: renderer,
            contextCond: contextCond,
            contextNull: contextNull,
            noise: noise,
            options: options,
            onStep: onStep)

        // Streaming decode: bit-identical to whole-sequence decode with flat
        // peak memory (whole-sequence OOMs past ~49 frames) — the oracle's
        // _vae_decode default.
        let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
        eval(frames)
        return frames
    }

    /// Text-to-image = single-frame t2v. Returns [1, 3, 1, H, W] in [-1, 1].
    public func t2i(
        prompt: String,
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        steps: Int? = nil,
        shift: Double? = nil,
        guideScale: (Double, Double)? = nil,
        scheduler: SchedulerKind? = nil,
        lightning: Bool = false,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        try t2v(
            prompt: prompt, negativePrompt: negativePrompt, width: width,
            height: height, numFrames: 1, steps: steps, shift: shift, guideScale: guideScale,
            scheduler: scheduler, lightning: lightning, seed: seed, onStep: onStep)
    }
}
