import Foundation
import WanCore
import MLX
import MLXToolKit

import BerniniR

/// MLXEngine package: Bernini-R (provenance-audited byte-stock Wan2.2-T2V-A14B) exposing the
/// canonical `textToVideo` + `textToImage` surfaces from ONE loaded dual-expert renderer
/// (t2i = single-frame t2v — the "one model, N surfaces" premise actually holds here).
///
/// Engine-owned lifecycle (C13): the engine constructs from a `BerniniRConfiguration`, pages the
/// working set in with `load()`, drives `run(_:)`, and reclaims with `unload()`. Lifecycle is
/// isolated to `InferenceActor`; the non-`Sendable` `BerniniPipeline` is actor-isolated state and
/// never crosses the boundary. Cancellation (eviction's lever) is honored at every denoising-step
/// boundary via the core's `onStep` callback.
///
/// The renderer's editing surfaces (r2v / v2v / rv2v — already parity-locked in the core) arrive
/// as additive surfaces with `imageEdit`/`videoEdit` at contract 1.2.0; reference-conditioned
/// generation (r2v) is a `T2VRequest` canonical-field candidate per the promotion rule (second
/// package wanting the lever).
@InferenceActor
public final class BerniniRPackage: ModelPackage {
    public typealias Configuration = BerniniRConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // Bernini-R / Wan2.2 weights are Apache-2.0; this port code is Apache-2.0.
            license: LicenseDeclaration(weightLicense: .apache2, portCodeLicense: .apache2),
            provenance: Provenance(
                sourceRepo: "mlx-community/Bernini-R-bf16",
                revision: "main",
                tier: 1
            ),
            requirements: RequirementsManifest(
                // Measured app-seam peak process memory (phys_footprint), the governor's budget
                // basis. Declarations sit just above the worst measured peak per quant.
                //   bf16: 110.47 GB — multi-frame t2v (Lightning 4-step, 17f/832x480, 2026-06-13);
                //     supersedes the earlier 90.8 GB t2i-only figure. -> declare 112 GB. (bf16
                //     *editing* is unmeasured and would be higher; revisit if bf16 v2v ships.)
                //   int4: 66.22 GB — rv2v APG editing (heaviest path, 4 fwd/step), clean standalone
                //     app re-measure 2026-06-14 AFTER the §2.4 umT5 post-encode eviction. Supersedes
                //     the 96.04 GB pre-eviction figure: releasing umT5 before the heavy 2x-token concat
                //     forward (+ a contamination-free baseline) drops the peak ~30 GB. -> declare 67 GB.
                //     This lands int4 editing UNDER the 0.7x production budget (96.2 GB) — un-blocks §3.1
                //     (the package was un-admittable at 98 GB).
                footprints: [
                    QuantFootprint(quant: .bf16, residentBytes: 112_000_000_000),
                    QuantFootprint(quant: .int4, residentBytes: 67_000_000_000),
                ],
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: .max
            ),
            specialties: [
                SpecialtyWeight(.general, strength: 0.6),
            ],
            surfaces: [
                T2VContract.descriptor(
                    name: "bernini-r-t2v",
                    summary: "Wan2.2-A14B dual-expert text-to-video (Bernini-R renderer, MLX). "
                        + "High-quality short clips; 832x480 native, frames must be 4n+1. "
                        + "`.fast` mode (DPM++/16) is ~2.5× quicker at near-identical quality.",
                    modes: [.fast, .quality]
                ),
                T2IContract.descriptor(
                    name: "bernini-r-t2i",
                    summary: "Text-to-image via single-frame Wan2.2-A14B video diffusion "
                        + "(Bernini-R renderer, MLX). Photorealistic stills, 832x480 native. "
                        + "`.fast` mode (DPM++/16) is ~2.5× quicker at near-identical quality.",
                    modes: [.fast, .quality]
                ),
                VEditContract.descriptor(
                    name: "bernini-r-vedit",
                    summary: "Prompt-based video editing (Bernini-R v2v/rv2v, MLX). Re-render a "
                        + "source clip toward a prompt; optional reference images steer the "
                        + "subject's identity. (Reference-to-video *generation* — no source clip — "
                        + "is the textToVideo surface with referenceImages.)",
                    modes: []
                ),
            ]
        )
    }

    private let configuration: Configuration
    /// The resident pipeline (dual experts + VAE + tokenizer), paged in by `load()`.
    /// umT5 is NOT resident — paged in per request and evicted before denoise (§2.4).
    private var pipeline: BerniniPipeline?

    public nonisolated init(configuration: Configuration) {
        self.configuration = configuration
    }

    /// Page the working set in. Idempotent when already resident. Resolution order:
    /// explicit `modelDirectory` → `BERNINI_R_WEIGHTS_DIR` env → HF download to the local cache.
    public func load() async throws {
        guard pipeline == nil else { return }
        let directory: URL
        if let explicit = configuration.modelDirectory {
            directory = explicit
        } else {
            directory = try await WeightLoader.snapshotDownload(repoID: configuration.repo)
        }
        pipeline = try await BerniniPipeline.fromPretrained(modelDir: directory)
    }

    public func unload() async {
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run() — before the notLoaded
        // guard, capability validation, or dispatch (run-lifecycle program, engine 0.27.0).
        try Task.checkCancellation()
        guard let pipeline else { throw PackageError.notLoaded }
        switch request.capability {
        case .textToImage:
            guard let t2i = request as? T2IRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2IRequest", got: String(describing: type(of: request)))
            }
            return try runT2I(t2i, pipeline: pipeline)
        case .textToVideo:
            guard let t2v = request as? T2VRequest else {
                throw PackageError.configurationMismatch(
                    expected: "T2VRequest", got: String(describing: type(of: request)))
            }
            return try await runT2V(t2v, pipeline: pipeline)
        case .videoEdit:
            guard let vedit = request as? VEditRequest else {
                throw PackageError.configurationMismatch(
                    expected: "VEditRequest", got: String(describing: type(of: request)))
            }
            return try await runVideoEdit(vedit, pipeline: pipeline)
        default:
            throw PackageError.unsupportedCapability(request.capability)
        }
    }

    // MARK: - Surfaces

    /// W6 guard: Bernini **int4** on the 40-step UniPC "quality" path renders **all-black** (a numerical
    /// failure in the int4 UniPC trajectory — there is no int4-specific scheduler branching; `.fast`
    /// (DPM++/16) is the validated/published int4 path). The default mode resolves to UniPC/40, so an
    /// int4 user who does nothing would silently get a ~96-min black clip. Until a root cause lands,
    /// auto-fall-back int4 "quality"→"fast" with a stderr warning. `BERNINI_ALLOW_INT4_QUALITY=1` forces
    /// the 40-step path (for debugging, e.g. with `WAN_DEBUG_STATS`). bf16/Lightning are unaffected.
    private func guardInt4Quality(
        _ sampling: (scheduler: SchedulerKind, steps: Int?)
    ) -> (scheduler: SchedulerKind, steps: Int?) {
        guard configuration.quant == .int4, sampling.scheduler == .unipc,
              ProcessInfo.processInfo.environment["BERNINI_ALLOW_INT4_QUALITY"] == nil
        else { return sampling }
        FileHandle.standardError.write(Data(
            ("[Bernini] int4 + 40-step \"quality\" (UniPC) renders all-black (W6) — falling back to "
             + ".fast (DPM++/16). Set BERNINI_ALLOW_INT4_QUALITY=1 to force the quality path.\n").utf8))
        return resolveSampling(mode: .fast, steps: nil)
    }

    private func runT2I(_ request: T2IRequest, pipeline: BerniniPipeline) throws -> T2IResponse {
        try Task.checkCancellation()
        // Lightning config → the fixed 4-step CFG-free sampler (overrides mode/steps).
        // Otherwise `.fast` mode → DPM++/16, else the 40-step UniPC default.
        let lit = configuration.lightning
        let sampling = guardInt4Quality(resolveSampling(mode: request.mode, steps: request.steps))
        let frames = try pipeline.t2i(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            width: request.width ?? 832,
            height: request.height ?? 480,
            steps: lit ? nil : sampling.steps,
            // Canonical scalar guidance applies to both expert phases (mlx-video float
            // semantics); absent -> config defaults (low 3.0, high 4.0).
            guideScale: lit ? nil : request.guidanceScale.map { ($0, $0) },
            scheduler: lit ? nil : sampling.scheduler,
            lightning: lit,
            seed: request.seed
        ) { _, _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }
        // Post-core checkpoint: the streaming VAE decode inside the pipeline bails per
        // chunk (non-throwing) — discard a truncated result and rethrow here.
        try Task.checkCancellation()
        // The 16-ch Wan VAE decodes 4 output frames per latent frame (oracle behavior,
        // parity-matched); the still is frame 0.
        let (data, width, height) = try encodePNG(frame: frames[0, 0..., 0, 0..., 0...])
        return T2IResponse(image: Image(format: .png, data: data, width: width, height: height))
    }

    private func runT2V(_ request: T2VRequest, pipeline: BerniniPipeline) async throws -> T2VResponse {
        guard request.initImage == nil else {
            // Bernini-R v1 backs text-only generation; image conditioning arrives with the
            // editing surfaces (r2v rides reference images, not an init frame).
            throw PackageError.configurationMismatch(
                expected: "no initImage (text-only t2v in v1; editing surfaces land at contract 1.2.0)",
                got: "initImage")
        }
        try Task.checkCancellation()
        let numFrames = request.numFrames ?? 49
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480

        // r2v: reference-conditioned generation (subject-consistent). Uses the quality
        // APG sampler — editing is not distilled, so lightning configs do plain t2v only.
        if let refs = request.referenceImages, !refs.isEmpty {
            let refPixels = try refs.map {
                try decodeReferencePixels($0, width: width, height: height)
            }
            let frames = try pipeline.r2v(
                prompt: request.prompt, referencePixels: refPixels,
                negativePrompt: request.negativePrompt,
                width: width, height: height, numFrames: numFrames,
                steps: request.steps ?? 40, seed: request.seed ?? 42
            ) { _, _, _ in try Task.checkCancellation() }
            // Post-core checkpoint: streaming VAE decode bails per chunk (non-throwing).
            try Task.checkCancellation()
            return try await framesToVideoResponse(frames, fps: fps)
        }

        // Plain t2v. Lightning config → fixed 4-step CFG-free; else `.fast`→DPM++/16, else UniPC/40.
        // W6: int4 + UniPC/40 is all-black → guardInt4Quality auto-falls-back to .fast.
        let lit = configuration.lightning
        let sampling = guardInt4Quality(resolveSampling(mode: request.mode, steps: request.steps))
        let frames = try pipeline.t2v(
            prompt: request.prompt,
            negativePrompt: request.negativePrompt,
            width: width,
            height: height,
            numFrames: numFrames,
            steps: lit ? nil : sampling.steps,
            guideScale: lit ? nil : request.guidanceScale.map { ($0, $0) },
            scheduler: lit ? nil : sampling.scheduler,
            lightning: lit,
            seed: request.seed
        ) { _, _, _ in
            try Task.checkCancellation()  // C13: per-denoising-step cancellation
        }
        // Post-core checkpoint: streaming VAE decode bails per chunk (non-throwing).
        try Task.checkCancellation()
        return try await framesToVideoResponse(frames, fps: fps)
    }

    /// Prompt-based video editing (v2v / rv2v): decode the source clip + any reference
    /// images to pixels, re-render toward the prompt, re-encode to an mp4 `Video`.
    private func runVideoEdit(_ request: VEditRequest, pipeline: BerniniPipeline) async throws
        -> VEditResponse
    {
        try Task.checkCancellation()
        let numFrames = request.numFrames ?? 49
        let fps = request.fps ?? 16
        let width = request.width ?? 832
        let height = request.height ?? 480

        let sourcePixels = try await decodeVideoPixels(
            request.video, width: width, height: height, numFrames: numFrames)
        let refPixels = try (request.referenceImages ?? []).map {
            try decodeReferencePixels($0, width: width, height: height)
        }
        // rv2v when references steer the subject, plain v2v otherwise.
        let mode: EditGuidanceMode = refPixels.isEmpty ? .v2v : .rv2v
        let frames = try pipeline.videoEdit(
            prompt: request.prompt, sourceVideoPixels: sourcePixels, referencePixels: refPixels,
            mode: mode, negativePrompt: request.negativePrompt,
            width: width, height: height, numFrames: numFrames,
            steps: request.steps ?? 40, seed: request.seed ?? 42
        ) { _, _, _ in try Task.checkCancellation() }

        try Task.checkCancellation()
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return VEditResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }

    private func framesToVideoResponse(_ frames: MLXArray, fps: Double) async throws -> T2VResponse {
        let mp4 = try await encodeMP4(frames: frames, fps: fps)
        return T2VResponse(
            video: Video(format: .mp4, data: mp4,
                         durationSeconds: Double(frames.dim(2)) / fps, frameRate: fps))
    }
}

extension BerniniRPackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(BerniniRPackage.self)
    }
}
