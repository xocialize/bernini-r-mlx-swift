import Foundation
import MLXToolKit

/// Init-time configuration for `BerniniRPackage` (C9): which published variant and where the
/// flat checkpoint lives. Per-request prompt/size/steps ride the canonical `T2VRequest` /
/// `T2IRequest`, not here.
///
/// Checkpoint resolution order at `load()`:
///   1. `modelDirectory` (a resolved flat checkpoint dir:
///      `{high_noise_model,low_noise_model,vae,t5_encoder}.safetensors` + `config.json`)
///   2. `BERNINI_R_WEIGHTS_DIR` env override (honored by `WeightLoader`)
///   3. HF download of `repo` into the local cache (`WeightLoader.snapshotDownload`)
/// `modelsRootDirectory` is the engine-store seam (`ModelStorable`); auto-materializing into it
/// is the next additive step, mirroring the other wrappers' V1 posture.
public struct BerniniRConfiguration: PackageConfiguration, ModelStorable {
    /// Published variant repo id (also the provenance source).
    public var repo: String
    public var revision: String?
    /// Backbone quant of the chosen variant (bf16 or int4) — selection metadata; the loader
    /// auto-detects the actual quantization from the checkpoint's config.json.
    public var quant: Quant
    /// The checkpoint is the lightx2v 4-step Lightning merge → the package always uses the
    /// CFG-free 4-step euler/shift-5 sampler (the merged weights only work few-step). A
    /// *different checkpoint*, so it's a config (which package loads), not a request mode.
    public var lightning: Bool
    /// Resolved local checkpoint folder. Environment-specific → excluded from `Codable`.
    public var modelDirectory: URL?
    /// Engine-chosen models root (future auto-materialization target). Environment-specific →
    /// excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Bernini-R-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        lightning: Bool = false,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.lightning = lightning
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    /// The published int4 variant (the consumer config: ~27 GB on disk, ~53 GB peak).
    public static var int4: BerniniRConfiguration {
        BerniniRConfiguration(repo: "mlx-community/Bernini-R-int4", quant: .int4)
    }

    /// The lightx2v 4-step Lightning merge — CFG-free, ~35× faster denoise. Point
    /// `modelDirectory` at the merged checkpoint (publish to HF pending).
    public static var lightning: BerniniRConfiguration {
        BerniniRConfiguration(repo: "mlx-community/Wan2.2-T2V-A14B-Lightning", lightning: true)
    }

    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, lightning
    }
}
