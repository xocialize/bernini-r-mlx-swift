// High-level t2v / t2i entry — the Swift mirror of the Bernini oracle's
// pipeline_mlx.py t2v/t2i (which wrap mlx-video's generate_video). Owns the
// component loads (renderer + VAE + UMT5 + tokenizer) and the prompt →
// frames path; saving/encoding artifacts is the caller's concern (the
// MLXEngine wrap returns canonical Video/Image; the smoke CLI writes PNGs).

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

public final class BerniniPipeline: @unchecked Sendable {
    public let config: WanConfig
    public let renderer: BerniniRendererModel
    public let vae: WanVAE
    public let textEncoder: UMT5EncoderModel
    public let tokenizer: any Tokenizer

    public init(
        config: WanConfig,
        renderer: BerniniRendererModel,
        vae: WanVAE,
        textEncoder: UMT5EncoderModel,
        tokenizer: any Tokenizer
    ) {
        self.config = config
        self.renderer = renderer
        self.vae = vae
        self.textEncoder = textEncoder
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

        let textEncoder = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }  // fp32 like mlx-video's load_t5_encoder
        WeightLoader.materialize(t5Weights)
        try textEncoder.update(
            parameters: ModuleParameters.unflattened(t5Weights),
            verify: [.noUnusedKeys])

        let tokenizer = try await AutoTokenizer.from(pretrained: umt5TokenizerRepo)
        return BerniniPipeline(
            config: config, renderer: renderer, vae: vae,
            textEncoder: textEncoder, tokenizer: tokenizer)
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
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt

        let contextCond = encodeText(
            encoder: textEncoder, tokenizer: tokenizer, prompt: prompt,
            textLen: config.textLen)
        let contextNull = encodeText(
            encoder: textEncoder, tokenizer: tokenizer, prompt: negative,
            textLen: config.textLen)
        eval(contextCond, contextNull)

        // Latent geometry from the VAE strides (temporal 1 + (F-1)/4, spatial /8)
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]

        if let seed {
            MLXRandom.seed(seed)
        }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        var options = T2VOptions.fromConfig(config)
        if let steps { options.steps = steps }

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
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        try t2v(
            prompt: prompt, negativePrompt: negativePrompt, width: width,
            height: height, numFrames: 1, steps: steps, seed: seed, onStep: onStep)
    }
}
