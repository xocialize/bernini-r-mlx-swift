// V3 e2e (RunBernini --v3-e2e): the full planner-conditioned pipeline on GPU —
// planner (bf16, production MaskGIT RNG) → T5-concat contexts → wvitcfg render
// (dual expert, production omegas) → streaming VAE decode → PNG frames.
//
// Planner INPUT streams come from the fixture pack until the Swift processor
// lands (the token/mask assembly is fixture-fed; everything downstream is the
// production path). Decoded output is the decisive V3 check — eyeball it.
//
//   swift run -c release RunBernini --v3-e2e \
//       --model-dir /Volumes/DEV_ARCHIVE/bernini-v2/ckpt-bf16 \
//       [--fixtures .../goldens-t2v-prod] [--frames 17] [--steps 16] [--out /tmp/bernini-v3]

import Foundation
import MLX
import MLXRandom
import MLXNN
import Tokenizers
import WanCore

import BerniniR

func runV3E2E(modelDir: URL) async throws {
    let fixturesPath = argValue("--fixtures")
        ?? "/Volumes/DEV_ARCHIVE/bernini-v2/measure/goldens-t2v-prod"
    let dir = URL(filePath: fixturesPath)
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: dir.appending(path: "\(name).npy"))
    }
    func fxInts(_ name: String) throws -> [Int] {
        try loadNumpyInts(url: dir.appending(path: "\(name).npy"))
    }
    let renderFrames = argValue("--frames").flatMap(Int.init) ?? 17
    let renderSteps = argValue("--steps").flatMap(Int.init) ?? 16
    let renderW = argValue("--width").flatMap(Int.init) ?? 480
    let renderH = argValue("--height").flatMap(Int.init) ?? 320
    let planningSteps = argValue("--planning-steps").flatMap(Int.init) ?? 25
    let e2eSeed = argValue("--seed").flatMap(UInt64.init) ?? 42
    let outDir = URL(filePath: argValue("--out") ?? "/tmp/bernini-v3")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    func loadStream(_ name: String) throws -> PlannerStream {
        let embeds = try fx("01_\(name)_embeds_postmask").asType(.bfloat16)
        let mask = try fx("00_\(name)_attention_mask_4d")
        let posHost = try fxInts("00_\(name)_position_ids")
        let l = posHost.count / 3
        let positionIds = MLXArray(posHost.map(Int32.init), [3, 1, l])
        let voutHost = try fxInts("00_\(name)_visual_output_token_mask")
        let voutIdx: [Int32] = voutHost.enumerated().compactMap {
            $0.element != 0 ? Int32($0.offset) : nil
        }
        return PlannerStream(
            inputsEmbeds: embeds, mask: mask, positionIds: positionIds,
            visualOutputIndices: voutIdx)
    }

    let promptArg = argValue("--prompt")

    let t0 = Date()
    print("[v3-e2e] PLANNER phase (bf16, GPU, \(planningSteps) MaskGIT steps)…")
    MLXRandom.seed(e2eSeed)
    var contexts: PlannerContexts!
    do {
        let planner = try BerniniPlanner.fromPretrained(modelDir: modelDir, dtype: .bfloat16)
        var cond: PlannerStream
        var uncond: PlannerStream
        var imgcond: PlannerStream
        if let promptArg {
            // PROMPT-DRIVEN: Swift processor -> embed -> mask-token overwrite.
            print("  processor: \"\(promptArg)\" (\(renderW)x\(renderH)x\(renderFrames)f)")
            let proc = try await BerniniProcessor.fromPretrained(
                mllmDir: modelDir.appending(path: "mllm"))
            let inputs = proc.process(
                prompt: promptArg, task: renderFrames > 1 ? .t2v : .t2i,
                width: renderW, height: renderH, numFrames: renderFrames)
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
            cond = toStream(inputs.cond)
            uncond = toStream(inputs.uncond)
            imgcond = toStream(inputs.imgcond)
        } else {
            cond = try loadStream("cond")
            uncond = try loadStream("uncond")
            imgcond = try loadStream("imgcond")
        }
        contexts = planner.plan(
            cond: &cond, uncond: &uncond, imgcond: &imgcond,
            planningSteps: planningSteps, vitDenoisingSteps: 3,
            seed: e2eSeed)
        eval(contexts.condEmbedsWtxtWvit, contexts.condEmbedsWotxtWovit)
    }  // planner + mllm released here — sequential eviction before the render phase
    MLX.GPU.clearCache()
    let tPlan = Date().timeIntervalSince(t0)
    print(String(format: "  planning done in %.1fs, peak %.1f GB",
                 tPlan, Double(GPU.peakMemory) / 1e9))

    // T5-concat exactly as pipeline.__call__ — prompt-driven via the unpadded
    // umT5 encode, else the fixture embeds.
    let t5: MLXArray
    let negT5: MLXArray
    if let promptArg {
        print("[v3-e2e] umT5 encode (unpadded, evicted after)…")
        let wanConfig = try WanConfig.load(
            from: modelDir.appendingPathComponent("config.json"))
        let umt5Tok = try await AutoTokenizer.from(pretrained: "google/umt5-xxl")
        (t5, negT5) = try PlannerTextEncode.encodeUnpadded(
            modelDir: modelDir, config: wanConfig, tokenizer: umt5Tok,
            prompt: promptArg,
            negativePrompt: wanConfig.sampleNegPrompt)
    } else {
        t5 = try fx("08_t5_embeds").asType(.float32)
        negT5 = try fx("08_neg_t5_embeds").asType(.float32)
    }
    func cat(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        concatenated([a, b.asType(.float32)], axis: 1)[0]
    }
    let wv = WvitcfgContexts(
        wtxtWvit: cat(t5, contexts.condEmbedsWtxtWvit),
        wtxtWovit: cat(t5, contexts.condEmbedsWtxtWovit),
        wotxtWvit: cat(negT5, contexts.condEmbedsWotxtWvit),
        wotxtWovit: cat(negT5, contexts.condEmbedsWotxtWovit))

    print("[v3-e2e] RENDER phase (\(renderW)x\(renderH)x\(renderFrames)f, \(renderSteps) steps, dual expert)…")
    let renderer = try BerniniRendererModel.fromPretrained(modelDir: modelDir)
    let tLat = (renderFrames - 1) / 4 + 1
    let latent = try wvitcfgSample(
        high: renderer.highNoiseExpert, low: renderer.lowNoiseExpert,
        contexts: wv,
        targetShape: [16, tLat, renderH / 8, renderW / 8], headDim: 128,
        boundaryTimestep: renderer.boundaryTimestep, steps: renderSteps,
        seed: e2eSeed
    ) { i, _ in
        print("  step \(i + 1)/\(renderSteps)")
    }
    let tRender = Date().timeIntervalSince(t0) - tPlan
    print(String(format: "  render done in %.1fs, peak %.1f GB",
                 tRender, Double(GPU.peakMemory) / 1e9))

    print("[v3-e2e] DECODE phase (streaming)…")
    let vae = WanVAE(zDim: 16, encoder: true)  // checkpoint carries both halves
    let vaeWeights = try Device.withDefaultDevice(.cpu) {
        let loaded = try MLX.loadArrays(
            url: modelDir.appendingPathComponent("vae.safetensors"))
        WeightLoader.materialize(loaded)
        return loaded
    }
    try vae.update(
        parameters: ModuleParameters.unflattened(vaeWeights), verify: [.noUnusedKeys])
    let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
    eval(frames)
    let f = frames[0]  // [3, T, H, W]
    for t in [0, f.dim(1) / 2, f.dim(1) - 1] where t < f.dim(1) {
        try writePNG(f[0..., t], to: outDir.appending(path: "v3_frame_\(t).png"))
    }
    print(String(format: "[v3-e2e] DONE total %.1fs, peak %.1f GB -> %@",
                 Date().timeIntervalSince(t0), Double(GPU.peakMemory) / 1e9, outDir.path))
}
