// 1:1 translation of bernini_r_mlx/model/renderer.py — the Bernini-R renderer.
// The A14B (Wan2.2) tier is DUAL-expert (`transformer` = high-noise,
// `transformer_2` = low-noise, selected at the switch_dit_boundary timestep);
// the 1.3B (Wan2.1) tier is DENSE — `dual_model: false`, ONE expert, no switch.
// Each expert is the stock `WanModel` (Phase 0 proved the Bernini renderer
// weights ARE stock Wan with no extra tensors). The Bernini deltas (SA-3D RoPE,
// source-VAE feature injection) are backbone-agnostic and attach at the RoPE /
// latent-prep seams, so they carry over to 1.3B wholesale.

import Foundation
import WanCore
import MLX
import MLXNN

public final class BerniniRendererModel: Module, @unchecked Sendable {
    public let config: WanConfig

    @ModuleInfo(key: "transformer") var transformer: WanModel  // high-noise / the dense expert
    /// Low-noise expert — present only for the dual-expert (A14B) tier; nil for
    /// the dense 1.3B tier (`dual_model: false`).
    @ModuleInfo(key: "transformer_2") var transformer2: WanModel?

    public init(_ config: WanConfig) {
        self.config = config
        self._transformer.wrappedValue = WanModel(config)
        self._transformer2.wrappedValue = config.dualModel ? WanModel(config) : nil
    }

    /// Timestep at/above which the high-noise expert is used (boundary * T).
    public var boundaryTimestep: Double { config.boundaryTimestep }

    /// Return the active expert for a given diffusion timestep. Dense (1.3B) →
    /// always the single expert; dual (A14B) → Wan2.2 high/low routing.
    public func selectExpert(_ timestep: Double) -> WanModel {
        guard let transformer2 else { return transformer }
        return timestep >= boundaryTimestep ? transformer : transformer2
    }

    public var highNoiseExpert: WanModel { transformer }
    public var lowNoiseExpert: WanModel { transformer2 ?? transformer }

    /// Load a converted (mlx-video-layout) Bernini-R checkpoint directory (+
    /// config.json). Dual-expert layout: `high_noise_model.safetensors` +
    /// `low_noise_model.safetensors`; dense layout: a single `model.safetensors`.
    /// The VAE and UMT5 encoder are loaded separately by the pipeline. For
    /// quantized checkpoints the QuantizedLinear slots are created BEFORE load.
    public static func fromPretrained(
        modelDir: URL,
        quantization explicitQuantization: WanQuantization? = nil
    ) throws -> BerniniRendererModel {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        let model = BerniniRendererModel(config)
        if config.dualModel {
            try loadExpert(
                model.transformer,
                url: modelDir.appendingPathComponent("high_noise_model.safetensors"),
                layers: config.numLayers, quantization: quantization)
            try loadExpert(
                model.transformer2!,
                url: modelDir.appendingPathComponent("low_noise_model.safetensors"),
                layers: config.numLayers, quantization: quantization)
        } else {
            // Dense (1.3B): one model.safetensors, one expert.
            try loadExpert(
                model.transformer,
                url: modelDir.appendingPathComponent("model.safetensors"),
                layers: config.numLayers, quantization: quantization)
        }
        eval(model.parameters())
        return model
    }

    private static func loadExpert(
        _ expert: WanModel, url: URL, layers: Int, quantization: WanQuantization?
    ) throws {
        if let quantization {
            WeightLoader.applyQuantization(to: expert, quantization: quantization)
        }
        // The int4 files carry a stray serialized `freqs` rope table the model
        // never loads — tolerated (dropped), so the expected set excludes it.
        let weights = try WeightLoader.loadVerifiedSafetensors(
            url: url,
            expectedKeys: BerniniWeightKeys.ditKeys(layers: layers, quantized: quantization != nil)
                .subtracting(["freqs"]),
            toleratedExtras: quantization != nil ? ["freqs"] : []
        )
        WeightLoader.materialize(weights)
        try expert.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys])
    }
}
