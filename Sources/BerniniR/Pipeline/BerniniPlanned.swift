//
//  BerniniPlanned.swift
//  BerniniR — Bernini-v2 planner-conditioned generation (PORTING-SPEC-V2.md)
//
//  The production full-Bernini request path, composed from the parity-locked
//  pieces (VProc/V1/V2/V3 gates all green):
//
//    phase 1  PLAN    BerniniProcessor → backbone.embed → mask-token overwrite →
//                     25-step MaskGIT (planner bf16) → four context variants.
//                     Planner (mllm 15.4 GB + heads) loads per request and is
//                     RELEASED before phase 2 (§2.4-style sequential eviction —
//                     measured phase peak ≤19.2 GB, strictly under the render
//                     envelope).
//    phase 2  T5      unpadded umT5 encode of prompt + negative (load → evict).
//    phase 3  RENDER  wvitcfg vae_txt_vit_wapg on the resident dual-expert
//                     renderer (FlowMatch euler, shift 5.0 — the v2 reference;
//                     NOT the classic UniPC path).
//    phase 4  DECODE  streaming VAE decode (flat memory, bit-identical).
//
//  Requires a v2 checkpoint (mllm/ + vit_decoder + planner_glue beside the
//  classic renderer files); `hasPlannerPlane` probes for it. The planner drives
//  hand-written forwards → planned generation REFUSES while the DiT blocks are
//  streamed (same rule as the editing surfaces).
//

import Foundation
import MLX
import MLXRandom
import Tokenizers
import WanCore

public enum BerniniPlannedError: Error, CustomStringConvertible {
    case plannerPlaneMissing(URL)
    public var description: String {
        switch self {
        case .plannerPlaneMissing(let dir):
            return "checkpoint at \(dir.path) has no planner plane "
                + "(mllm/model.safetensors) — planned generation needs a "
                + "Bernini-v2 checkpoint (mlx-community/Bernini-v2-*)"
        }
    }
}

/// Does this checkpoint carry the Bernini-v2 planner plane?
public func hasPlannerPlane(modelDir: URL) -> Bool {
    FileManager.default.fileExists(
        atPath: modelDir.appending(path: "mllm/model.safetensors").path)
        && FileManager.default.fileExists(
            atPath: modelDir.appending(path: "vit_decoder.safetensors").path)
}

/// Planner-conditioned t2v/t2i. Returns decoded frames `[1, 3, T, H, W]` in
/// [-1, 1]. `renderer`/`vae` are the caller's RESIDENT models (the package's
/// pipeline); the planner and umT5 are paged in and evicted per phase.
public func plannedGenerate(
    renderer: BerniniRendererModel,
    vae: WanVAE,
    modelDir: URL,
    config: WanConfig,
    umt5Tokenizer: any Tokenizer,
    prompt: String,
    negativePrompt: String? = nil,
    width: Int = 832,
    height: Int = 480,
    numFrames: Int = 49,
    renderSteps: Int = 40,
    planningSteps: Int = 25,
    vitDenoisingSteps: Int = 3,
    seed: UInt64 = 42,
    // Inherit the caller's actor (SE-0420): the engine package calls this from
    // `InferenceActor`-isolated code with its non-Sendable pipeline pieces and
    // cancellation hooks — running on the caller's executor means nothing
    // crosses an isolation boundary.
    isolation: isolated (any Actor)? = #isolation,
    onPlanStep: ((Int, Int) throws -> Void)? = nil,
    onRenderStep: ((Int, Int) throws -> Void)? = nil
) async throws -> MLXArray {
    guard hasPlannerPlane(modelDir: modelDir) else {
        throw BerniniPlannedError.plannerPlaneMissing(modelDir)
    }
    // The planner's hand-driven forwards (and the multi-context render loop)
    // bypass `runBlocks` — refuse rather than read unrefilled streamer slots.
    try renderer.requireResidentBlocks("planned generation (Bernini-v2)")

    // ---- phase 1: PLAN (planner paged in, evicted at scope exit) ----
    MLXRandom.seed(seed)
    let task: BerniniPlannerTask = numFrames > 1 ? .t2v : .t2i
    var contexts: PlannerContexts!
    do {
        let planner = try BerniniPlanner.fromPretrained(
            modelDir: modelDir, dtype: .bfloat16)
        let processor = try await BerniniProcessor.fromPretrained(
            mllmDir: modelDir.appending(path: "mllm"))
        let inputs = processor.process(
            prompt: prompt, task: task,
            width: width, height: height, numFrames: numFrames)
        func toStream(_ s: BerniniProcessedStream) -> PlannerStream {
            let ids = MLXArray(s.inputIds.map(Int32.init))[.newAxis]
            let embeds = planner.backbone.embed(ids).asType(.bfloat16)
            let masked = planner.applyMaskTokens(
                embeds, outputIndices: s.visualOutputIndices)
            return PlannerStream(
                inputsEmbeds: masked, mask: s.attentionMask4D(),
                positionIds: s.positionIdsArray(),
                visualOutputIndices: s.visualOutputIndices)
        }
        var cond = toStream(inputs.cond)
        var uncond = toStream(inputs.uncond)
        var imgcond = toStream(inputs.imgcond)
        contexts = try planner.plan(
            cond: &cond, uncond: &uncond, imgcond: &imgcond,
            planningSteps: planningSteps, vitDenoisingSteps: vitDenoisingSteps,
            seed: seed, onStep: onPlanStep)
        eval(contexts.condEmbedsWtxtWvit, contexts.condEmbedsWtxtWovit,
             contexts.condEmbedsWotxtWvit, contexts.condEmbedsWotxtWovit)
    }
    MLX.Memory.clearCache()

    // ---- phase 2: T5 (unpadded, load → evict inside) ----
    let negative = negativePrompt ?? config.sampleNegPrompt
    let (t5, negT5) = try PlannerTextEncode.encodeUnpadded(
        modelDir: modelDir, config: config, tokenizer: umt5Tokenizer,
        prompt: prompt, negativePrompt: negative)

    func cat(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        concatenated([a, b.asType(.float32)], axis: 1)[0]
    }
    let wv = WvitcfgContexts(
        wtxtWvit: cat(t5, contexts.condEmbedsWtxtWvit),
        wtxtWovit: cat(t5, contexts.condEmbedsWtxtWovit),
        wotxtWvit: cat(negT5, contexts.condEmbedsWotxtWvit),
        wotxtWovit: cat(negT5, contexts.condEmbedsWotxtWovit))

    // ---- phase 3: RENDER (resident dual experts) ----
    let tLat = (numFrames - 1) / config.vaeStride[0] + 1
    let latent = try wvitcfgSample(
        high: renderer.highNoiseExpert, low: renderer.lowNoiseExpert,
        contexts: wv,
        targetShape: [config.vaeZDim, tLat,
                      height / config.vaeStride[1], width / config.vaeStride[2]],
        headDim: 128,
        boundaryTimestep: renderer.boundaryTimestep, steps: renderSteps,
        seed: seed
    ) { i, _ in
        try onRenderStep?(i, renderSteps)
    }

    // ---- phase 4: DECODE (streaming) ----
    let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
    eval(frames)
    return frames
}
