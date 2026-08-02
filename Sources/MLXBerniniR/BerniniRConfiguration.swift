import Foundation
import WanCore
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
public struct BerniniRConfiguration: PackageConfiguration, ModelStorable, QuantConfigured {
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
    /// Opt in to HV2 block streaming: the DiT's transformer blocks are read from per-block
    /// granule files on demand through two resident slots instead of being loaded resident.
    /// Requires `granuleRootDirectory`; inert without it. Excluded from `Codable` — see
    /// `CodingKeys`. The `= false` is load-bearing: a stored property outside `CodingKeys`
    /// must be default-initializable for the synthesized decoder (the `URL?` members get
    /// their implicit nil).
    public var streamedBlocks: Bool = false
    /// Root of the granule tree laid out by wan-core's `wan-granule-layout`, holding one
    /// subtree per quant: `<root>/<quant>/{high,low}` for the dual-expert A14B tiers,
    /// `<root>/<quant>/model` for the dense 1.3B ones. One setting therefore serves every
    /// variant — `quant` selects the subtree. Environment-specific → excluded from `Codable`.
    public var granuleRootDirectory: URL?

    public init(
        repo: String = "mlx-community/Bernini-R-bf16",
        revision: String? = nil,
        quant: Quant = .bf16,
        lightning: Bool = false,
        modelDirectory: URL? = nil,
        modelsRootDirectory: URL? = nil,
        streamedBlocks: Bool = false,
        granuleRootDirectory: URL? = nil
    ) {
        self.repo = repo
        self.revision = revision
        self.quant = quant
        self.lightning = lightning
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
        self.streamedBlocks = streamedBlocks
        self.granuleRootDirectory = granuleRootDirectory
    }

    /// The granule subtree for THIS variant, or nil when streaming is off or unconfigured.
    /// A `quant` that disagrees with the checkpoint's own `config.json` resolves to the wrong
    /// subtree; the loader's manifest-provenance check catches that at load with a message
    /// naming both files, rather than streaming mismatched weights.
    public var resolvedGranuleRoot: URL? {
        guard streamedBlocks, let root = granuleRootDirectory else { return nil }
        return root.appendingPathComponent(quant.rawValue)
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

    /// The LOWEST tier — Bernini-R-1.3B (Wan2.1-1.3B dense backbone, `dual_model:false`).
    /// One resident `WanModel` (~2.8 GB bf16) + the 16-ch WanVAE → ~3.6 GB active working
    /// set (measured via `RunBernini` t2i). Same editing surfaces (SA-3D RoPE / APG),
    /// smaller backbone. ⚠️ The package manifest still declares the A14B footprints — a
    /// distinct 1.3B PackageID with per-variant requirements is the follow-up before this
    /// can be admitted on the small machines this tier is FOR.
    public static var oneThreeB: BerniniRConfiguration {
        BerniniRConfiguration(repo: "mlx-community/Bernini-R-1.3B-bf16")
    }

    public static var oneThreeBInt4: BerniniRConfiguration {
        BerniniRConfiguration(repo: "mlx-community/Bernini-R-1.3B-int4", quant: .int4)
    }

    /// int4 with HV2 block streaming — the A14B DiT's 40 blocks per expert are read from
    /// granules through two ~377 MiB slots instead of ~17 GB of resident expert weights.
    /// Point `granuleRootDirectory` at a `wan-granule-layout` tree (`<root>/int4/{high,low}`).
    ///
    /// ⚠️ The footprint declaration is deliberately UNCHANGED (still the resident int4
    /// figure). The HV2 receipt measured the **DiT denoise working set only** — MLX peak
    /// 7.11 GB at 480p/17f CFG, phys 6.51 GB, slots 2×377 MiB — with umT5 and the VAE
    /// explicitly NOT loaded. `residentBytes` is a max-over-phase figure, and the fp32 umT5
    /// encode phase (~22 GB, §2.11) dominates the streamed DiT phase, so the receipt's
    /// number is not this package's envelope. Declaring it would claim admission on
    /// machines that would then die in text-encode. A whole-pipeline app-seam re-measure
    /// gates both a `FootprintConfigured` hint here and the §5.6 bandwidth field on
    /// `QuantFootprint` (additive only when a streamed variant actually ships).
    public static var int4Streamed: BerniniRConfiguration {
        BerniniRConfiguration(repo: "mlx-community/Bernini-R-int4", quant: .int4, streamedBlocks: true)
    }

    /// `streamedBlocks` is excluded alongside `granuleRootDirectory`: streaming needs BOTH,
    /// and the granule root is an environment-specific path (like `modelDirectory`) that
    /// never encodes. Encoding the flag alone would buy nothing and would break decoding of
    /// configs persisted before it existed — a non-optional `Bool` in `CodingKeys` makes the
    /// key mandatory. Streaming is a runtime opt-in, not part of the config's identity.
    private enum CodingKeys: String, CodingKey {
        case repo, revision, quant, lightning
    }
}

/// Cold-start weight prewarm (engine ≥0.7.0): page the resolved checkpoint into the OS file cache
/// before `load()` runs its GPU evals, so the cold load-time `eval` never faults weights off
/// slow/external storage inside a live Metal command buffer (the cold-load GPU watchdog,
/// `kIOGPUCommandBufferCallbackErrorTimeout`). The acute case is a cold Bernini **bf16** load
/// (~64 GB off the archive volume); small/int4 variants are unlikely to bite. Each published
/// variant lives in its OWN flat directory (`ckpt-{bf16,int4,lightning}`), so — unlike LTX's
/// co-located transformers — paging the whole resolved `modelDirectory` already loads only the
/// files this variant uses (no exclusion needed). Only the config knows the resolved path;
/// execution is the engine's (`WeightPrewarmer`, best-effort). Nil when the HF-download path is
/// used (nothing local to page yet) → prewarm is a no-op.
/// ⚠️ The granule root is deliberately NOT prewarmed when streaming is on. `BlockStreamer`
/// opens granules `F_NOCACHE` precisely so the page cache cannot absorb the model — paging
/// them in here would defeat the bounded-memory property that is the whole point. The
/// checkpoint dir stays prewarmed either way: the streamed path still loads the non-block
/// globals from `{high,low}_noise_model.safetensors` through the normal loader.
extension BerniniRConfiguration: WeightPrewarming {
    public var prewarmPaths: [URL] { [modelDirectory].compactMap { $0 } }
}
