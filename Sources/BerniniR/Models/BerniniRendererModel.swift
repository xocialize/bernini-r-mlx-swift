// 1:1 translation of bernini_r_mlx/model/renderer.py — the Wan2.2-A14B
// dual-expert renderer. Holds the two Wan2.2 DiT experts (`transformer` =
// high-noise, `transformer_2` = low-noise) and selects between them at the
// switch_dit_boundary timestep. Each expert is the stock WanModel — Phase 0
// of the Python port proved the Bernini renderer weights ARE stock Wan2.2
// with no extra tensors. The Bernini deltas (SA-3D RoPE, source-VAE feature
// injection) attach at the RoPE / latent-prep seams in later phases.

import Foundation
import MLX
import MLXNN

public final class BerniniRendererModel: Module, @unchecked Sendable {
    public let config: WanConfig

    @ModuleInfo(key: "transformer") var transformer: WanModel  // high-noise expert
    @ModuleInfo(key: "transformer_2") var transformer2: WanModel  // low-noise expert

    public init(_ config: WanConfig) {
        self.config = config
        self._transformer.wrappedValue = WanModel(config)
        self._transformer2.wrappedValue = WanModel(config)
    }

    /// Timestep at/above which the high-noise expert is used (boundary * T).
    public var boundaryTimestep: Double { config.boundaryTimestep }

    /// Return the active expert for a given diffusion timestep (Wan2.2 routing).
    public func selectExpert(_ timestep: Double) -> WanModel {
        timestep >= boundaryTimestep ? transformer : transformer2
    }

    public var highNoiseExpert: WanModel { transformer }
    public var lowNoiseExpert: WanModel { transformer2 }

    /// Load a converted (mlx-video-layout) Bernini-R checkpoint directory:
    /// `high_noise_model.safetensors` + `low_noise_model.safetensors` (+
    /// config.json). The VAE and UMT5 encoder are loaded separately by the
    /// pipeline, matching mlx-video. For quantized checkpoints the
    /// QuantizedLinear slots are created BEFORE the bit-packed weights load.
    public static func fromPretrained(
        modelDir: URL,
        quantization explicitQuantization: WanQuantization? = nil
    ) throws -> BerniniRendererModel {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        let model = BerniniRendererModel(config)
        try loadExpert(
            model.transformer,
            url: modelDir.appendingPathComponent("high_noise_model.safetensors"),
            quantization: quantization)
        try loadExpert(
            model.transformer2,
            url: modelDir.appendingPathComponent("low_noise_model.safetensors"),
            quantization: quantization)
        eval(model.parameters())
        return model
    }

    private static func loadExpert(
        _ expert: WanModel, url: URL, quantization: WanQuantization?
    ) throws {
        if let quantization {
            WeightLoader.applyQuantization(to: expert, quantization: quantization)
        }
        // The int4 files carry a stray serialized `freqs` rope table the model
        // never loads — tolerated (dropped), so the expected set excludes it.
        let weights = try WeightLoader.loadVerifiedSafetensors(
            url: url,
            expectedKeys: BerniniWeightKeys.ditKeys(quantized: quantization != nil)
                .subtracting(["freqs"]),
            toleratedExtras: quantization != nil ? ["freqs"] : []
        )
        try expert.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys])
    }
}
