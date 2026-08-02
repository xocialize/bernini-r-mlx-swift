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

public enum BerniniStreamingError: LocalizedError {
    case granuleMismatch(String)
    case unsupportedSurface(String)

    public var errorDescription: String? {
        switch self {
        case .granuleMismatch(let m): return "Bernini block streaming: \(m)"
        case .unsupportedSurface(let m): return "Bernini block streaming: \(m)"
        }
    }
}

/// Opt-in HV2 block streaming for the renderer's experts (NEUROSTREAM-ACTIONS HV2, proven
/// bit-exact on real A14B by wan-core's `RunWanStream` — receipt
/// `mlxengine-todo/probes/hv2_wan_blockstreamer.out`). The transformer blocks are read from
/// per-block granule files through two resident slots instead of being loaded resident;
/// `WanModel.prepareCrossKV` / `runBlocks` route through the streamed group loop
/// automatically once a streamer is bound, so the denoise path needs no changes and the
/// A14B t=875 expert switch just activates the other granule set.
public struct BerniniStreamingConfiguration: Sendable {
    /// Root of a `wan-granule-layout` tree. Expert granule dirs are resolved beneath it by
    /// checkpoint layout: dual-expert → `<root>/high` + `<root>/low`; dense → `<root>/model`.
    public var granuleRoot: URL
    /// Slot geometry and gate policy (wan-core defaults: group 2, `.auto` with automatic
    /// fully-resident fallback when `N ≥ B·F/(2·S)` doesn't clear).
    public var options: BlockStreamingOptions

    public init(granuleRoot: URL, options: BlockStreamingOptions = .init()) {
        self.granuleRoot = granuleRoot
        self.options = options
    }
}

public final class BerniniRendererModel: Module, @unchecked Sendable {
    public let config: WanConfig

    @ModuleInfo(key: "transformer") var transformer: WanModel  // high-noise / the dense expert
    /// Low-noise expert — present only for the dual-expert (A14B) tier; nil for
    /// the dense 1.3B tier (`dual_model: false`).
    @ModuleInfo(key: "transformer_2") var transformer2: WanModel?

    /// The bound `BlockStreamer` when this renderer was loaded streamed — nil for the
    /// (default) fully-resident load. Retained here because it owns the slot arrays every
    /// expert's block parameters alias: releasing it while the experts live would leave
    /// those parameters pointing at freed memory.
    public private(set) var blockStreamer: BlockStreamer?

    /// True while blocks are being streamed. Goes false when the runtime gate falls back to
    /// fully-resident (`.auto` policy) — the streamer detaches itself from the experts and
    /// the normal resident paths resume, output-invisibly.
    public var isStreaming: Bool { transformer.blockStreamer != nil }

    /// Release the streamer once the runtime gate has fallen back to fully-resident. After
    /// `fallBackResident` the blocks own fresh resident arrays and nothing aliases the slots
    /// any more, but this renderer still holds the streamer — and with it 2×G blocks of slot
    /// memory (2.62 GiB bf16 / 0.74 GiB int4) that can never be read again. Dropping the
    /// reference frees them. No-op while streaming is live: releasing then would leave every
    /// block parameter aliasing freed memory.
    func releaseStreamerIfFellBack() {
        guard let streamer = blockStreamer, !isStreaming else { return }
        lastStreamingVerdict = streamer.verdict
        blockStreamer = nil
    }

    /// The gate verdict of the streamer this renderer was loaded with, surviving the
    /// release above. Nil for a fully-resident load; prefer the live streamer's verdict
    /// while one is attached.
    public private(set) var lastStreamingVerdict: BlockStreamer.Verdict?

    /// What the HV2 runtime gate decided, or nil if this renderer never streamed.
    public var streamingVerdict: BlockStreamer.Verdict? {
        blockStreamer?.verdict ?? lastStreamingVerdict
    }

    /// Refuse a surface whose forward does not route through `WanModel.runBlocks` while
    /// blocks are streamed. The hand-written multiseg loop the editing samplers use bypasses
    /// the streamer's group window entirely (see `forwardMultiseg`), so it would read
    /// unrefilled slots and return a plausible-looking but wrong result. Failing loudly is
    /// the only safe behavior until that loop is routed and receipted.
    public func requireResidentBlocks(_ surface: String) throws {
        guard isStreaming else { return }
        throw BerniniStreamingError.unsupportedSurface(
            "\(surface) is not supported with streamed blocks — its multiseg forward bypasses "
                + "the streamer's group loop and would read unrefilled slots. Load this "
                + "configuration without `streamedBlocks` for editing, or use t2v/t2i.")
    }

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
    ///
    /// Pass `streaming` to take the HV2 block-streaming path instead: blocks are never
    /// loaded resident — only the non-block globals are — and the experts' block parameters
    /// alias the streamer's two slots. Everything downstream is unchanged.
    public static func fromPretrained(
        modelDir: URL,
        quantization explicitQuantization: WanQuantization? = nil,
        streaming: BerniniStreamingConfiguration? = nil
    ) throws -> BerniniRendererModel {
        let config = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let quantization = explicitQuantization ?? config.quantization

        let model = BerniniRendererModel(config)
        // (expert, checkpoint file, granule subdirectory) — dual-expert A14B or dense 1.3B.
        let experts: [(model: WanModel, file: String, granuleDir: String)] =
            config.dualModel
            ? [(model.transformer, "high_noise_model.safetensors", "high"),
               (model.transformer2!, "low_noise_model.safetensors", "low")]
            : [(model.transformer, "model.safetensors", "model")]

        guard let streaming else {
            for expert in experts {
                try loadExpert(
                    expert.model, url: modelDir.appendingPathComponent(expert.file),
                    layers: config.numLayers, quantization: quantization)
            }
            eval(model.parameters())
            return model
        }

        // Streamed load — the wan-core consumer recipe, in order:
        //   construct → applyQuantization (int4 slots must exist before bind's contract
        //   check) → bind (allocate slots, inject slot-backed parameters once) →
        //   loadStreamingGlobals per expert. The resident block load is skipped entirely.
        if let quantization {
            for expert in experts {
                WeightLoader.applyQuantization(to: expert.model, quantization: quantization)
            }
        }
        let granuleDirs = experts.map {
            streaming.granuleRoot.appendingPathComponent($0.granuleDir)
        }
        for (expert, dir) in zip(experts, granuleDirs) {
            try verifyGranules(
                dir: dir, source: modelDir.appendingPathComponent(expert.file))
        }
        let streamer = try BlockStreamer(
            granuleDirs: granuleDirs, options: streaming.options)
        try streamer.bind(experts: experts.map(\.model))
        for expert in experts {
            try streamer.loadStreamingGlobals(
                expert: expert.model, from: modelDir.appendingPathComponent(expert.file))
        }
        // `loadStreamingGlobals` already eval'd the resident globals; a blanket
        // `eval(model.parameters())` here would only re-touch the zeroed slots.
        model.blockStreamer = streamer
        return model
    }

    /// Granules are a DERIVED artifact — re-converting or re-quantizing a checkpoint leaves
    /// a stale tree behind that still loads. The manifest records which safetensors it was
    /// cut from and how big that file was, so check both before binding. Without this the
    /// failure mode is either a confusing shape/dtype mismatch deep inside `bind`, or —
    /// when the layouts happen to agree, e.g. a re-quantized checkpoint at the same
    /// geometry — silently streaming the wrong weights.
    private static func verifyGranules(dir: URL, source: URL) throws {
        let manifest = try GranuleManifest.load(from: dir)
        let expected = source.lastPathComponent
        guard manifest.sourceFile == expected else {
            throw BerniniStreamingError.granuleMismatch(
                "granules at \(dir.path) were laid out from \(manifest.sourceFile), "
                    + "not \(expected) — wrong granule root for this checkpoint?")
        }
        let size = (try? FileManager.default.attributesOfItem(atPath: source.path)[.size])
            .flatMap { $0 as? Int }
        guard let size else {
            throw BerniniStreamingError.granuleMismatch(
                "cannot stat \(source.path) to validate granules")
        }
        guard manifest.sourceSize == size else {
            throw BerniniStreamingError.granuleMismatch(
                "granules at \(dir.path) are stale: laid out from a \(manifest.sourceSize)-byte "
                    + "\(expected), but that file is now \(size) bytes — re-run wan-granule-layout")
        }
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
