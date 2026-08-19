//
//  BerniniPlanner.swift
//  BerniniR — Bernini-v2 planner plane (PORTING-SPEC-V2.md)
//
//  The MaskGIT planning loop — upstream `bernini/pipeline.py sample_vit_embed`
//  ported verbatim over three streams (cond / uncond / imgcond):
//
//    per planning step:
//      3 × backbone forwards (penultimate-hidden tap)
//      connector.for_vit at the output-ViT positions of each stream
//      cosine reveal schedule picks this step's rows (order is INJECTED — the
//      oracle's np-seeded shuffle; MLX RNG ≠ numpy RNG)
//      DiffLossFM.sample on the revealed rows (3-stream CFG, SHARED injected noise)
//      sampled embeds written back into ALL THREE streams
//    then 2 final forwards (cond / uncond) → connector.for_gen → the four
//    renderer context variants (feature_type `masked_tgt_embed_with_qwen_txt_vit_tokens`).
//
//  Inference-only; `post_process_input_embeds(inference=True)` semantics: every
//  output-ViT position is overwritten with mask_tokens[0, 0] before step 1.
//

import Foundation
import MLX
import MLXRandom

/// One planner stream's prepared inputs (from the Bernini processor).
public struct PlannerStream {
    /// `[1, L, hidden]` — inputs_embeds AFTER the mask-token overwrite.
    public var inputsEmbeds: MLXArray
    /// `[1, L, L]` additive 0/−inf mask.
    public var mask: MLXArray
    /// `[3, 1, L]` M-RoPE position ids.
    public var positionIds: MLXArray
    /// Output-ViT positions (host-side indices into L).
    public var visualOutputIndices: [Int32]

    public init(
        inputsEmbeds: MLXArray, mask: MLXArray, positionIds: MLXArray,
        visualOutputIndices: [Int32]
    ) {
        self.inputsEmbeds = inputsEmbeds
        self.mask = mask
        self.positionIds = positionIds
        self.visualOutputIndices = visualOutputIndices
    }
}

/// The four renderer context variants + the sampled plan embeds.
public struct PlannerContexts {
    public let condEmbedsWtxtWvit: MLXArray    // [1, L_cond, 4096]
    public let condEmbedsWtxtWovit: MLXArray   // [1, n_txt_cond, 4096]
    public let condEmbedsWotxtWvit: MLXArray   // [1, n_vit, 4096]
    public let condEmbedsWotxtWovit: MLXArray  // [1, n_txt_uncond, 4096]
    public let predVitEmbed: MLXArray          // [1, n_vit, hidden]
}

public final class BerniniPlanner {
    public let backbone: QwenPlannerBackbone
    public let glue: PlannerGlue
    public let diffLoss: DiffLossFM

    public init(backbone: QwenPlannerBackbone, glue: PlannerGlue, diffLoss: DiffLossFM) {
        self.backbone = backbone
        self.glue = glue
        self.diffLoss = diffLoss
    }

    public static func fromPretrained(modelDir: URL, dtype: DType? = nil) throws
        -> BerniniPlanner
    {
        let backbone = try QwenPlannerBackbone.fromPretrained(
            mllmDir: modelDir.appending(path: "mllm"), dtype: dtype)
        let glue = try PlannerGlue.fromPretrained(
            file: modelDir.appending(path: "planner_glue.safetensors"), dtype: dtype)
        let diffLoss = try DiffLossFM.fromPretrained(
            file: modelDir.appending(path: "vit_decoder.safetensors"), dtype: dtype)
        return BerniniPlanner(backbone: backbone, glue: glue, diffLoss: diffLoss)
    }

    /// `post_process_input_embeds(inference=True)`: overwrite every output-ViT
    /// position with mask_tokens[0, 0].
    public func applyMaskTokens(_ embeds: MLXArray, outputIndices: [Int32]) -> MLXArray {
        var flat = embeds[0]  // [L, hidden]
        let idx = MLXArray(outputIndices)
        let token = glue.maskTokens[0, 0 ..< 1]  // [1, hidden]
        flat[idx] = broadcast(
            token.asType(flat.dtype), to: [outputIndices.count, flat.dim(1)])
        return flat[.newAxis]
    }

    /// The upstream cosine reveal schedule, host-side. Returns the per-step
    /// revealed positions (indices into the ViT-query axis), consuming the
    /// injected shuffle `order`.
    static func revealSchedule(order: [Int], planningSteps: Int) -> [[Int]] {
        let n = order.count
        var mask = [Bool](repeating: true, count: n)  // true = still masked
        var reveals: [[Int]] = []
        for step in 0 ..< planningSteps {
            let ratio = cos(Float.pi / 2 * Float(step + 1) / Float(planningSteps))
            var maskLen = Int((Float(n) * ratio).rounded(.down))
            let remaining = mask.filter { $0 }.count
            maskLen = max(1, min(remaining - 1, maskLen))
            var maskNext = [Bool](repeating: false, count: n)
            for i in 0 ..< maskLen { maskNext[order[i]] = true }
            let toPred: [Int]
            if step >= planningSteps - 1 {
                toPred = (0 ..< n).filter { mask[$0] }
            } else {
                toPred = (0 ..< n).filter { mask[$0] != maskNext[$0] }
            }
            mask = maskNext
            reveals.append(toPred)
        }
        return reveals
    }

    /// Seeded Fisher-Yates over 0..<n (SplitMix64) — the production stand-in for
    /// the oracle's np.random.shuffle (RNG streams differ by design; V1 parity
    /// injects the oracle's order instead).
    static func seededOrder(n: Int, seed: UInt64) -> [Int] {
        var state = seed &+ 0x9E37_79B9_7F4A_7C15
        func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        var order = Array(0 ..< n)
        for i in stride(from: n - 1, to: 0, by: -1) {
            let j = Int(next() % UInt64(i + 1))
            order.swapAt(i, j)
        }
        return order
    }

    /// `sample_vit_embed`, verbatim. Parity runs INJECT `revealOrder`/`fmNoises`
    /// (the oracle's np shuffle + torch noise); production leaves them nil and
    /// gets a seeded shuffle + MLXRandom noise (caller seeds MLXRandom).
    /// Streams are mutated in place (embeds written back each step).
    public func plan(
        cond: inout PlannerStream, uncond: inout PlannerStream,
        imgcond: inout PlannerStream,
        planningSteps: Int, vitDenoisingSteps: Int,
        vitTxtCfg: Float = 1.4, vitImgCfg: Float = 1.2,
        revealOrder: [Int]? = nil, fmNoises: [MLXArray]? = nil,
        seed: UInt64 = 42,
        tapLayers: Int? = nil, fmSigmas: MLXArray? = nil,
        fmTimesteps: MLXArray? = nil,
        onStep: ((Int, Int) throws -> Void)? = nil
    ) rethrows -> PlannerContexts {
        let n = cond.visualOutputIndices.count
        let order = revealOrder ?? Self.seededOrder(n: n, seed: seed)
        let reveals = Self.revealSchedule(order: order, planningSteps: planningSteps)
        if let fmNoises {
            precondition(fmNoises.count >= reveals.filter { !$0.isEmpty }.count,
                         "need one injected noise block per non-empty reveal step")
        }

        let condIdx = MLXArray(cond.visualOutputIndices)
        let uncondIdx = MLXArray(uncond.visualOutputIndices)
        let imgcondIdx = MLXArray(imgcond.visualOutputIndices)

        var noiseCursor = 0
        for step in 0 ..< planningSteps {
            // CAN seam: one checkpoint per planning step (3 backbone forwards each).
            try onStep?(step, planningSteps)
            let toPred = reveals[step]
            if toPred.isEmpty { continue }

            let hCond = backbone.penultimateHidden(
                inputsEmbeds: cond.inputsEmbeds, mask: cond.mask,
                positionIds: cond.positionIds, layersToRun: tapLayers)
            let hUncond = backbone.penultimateHidden(
                inputsEmbeds: uncond.inputsEmbeds, mask: uncond.mask,
                positionIds: uncond.positionIds, layersToRun: tapLayers)
            let hImgcond = backbone.penultimateHidden(
                inputsEmbeds: imgcond.inputsEmbeds, mask: imgcond.mask,
                positionIds: imgcond.positionIds, layersToRun: tapLayers)

            let vitCond = glue.connector.forVit(hCond[0][condIdx])          // [n_vit, C]
            let vitUncond = glue.connector.forVit(hUncond[0][uncondIdx])
            let vitImgcond = glue.connector.forVit(hImgcond[0][imgcondIdx])

            let predIdx = MLXArray(toPred.map { Int32($0) })
            // z rows ordered [cond | uncond | imgcond] — upstream concat order.
            let z = concatenated(
                [vitCond[predIdx], vitUncond[predIdx], vitImgcond[predIdx]], axis: 0)
            let noiseBlock = fmNoises?[noiseCursor]
                ?? MLXRandom.normal([toPred.count, cond.inputsEmbeds.dim(2)])
                    .asType(cond.inputsEmbeds.dtype)
            let sampled = diffLoss.sample(
                z: z, txtCfg: vitTxtCfg, imgCfg: vitImgCfg,
                numInferenceSteps: vitDenoisingSteps,
                injectedNoise: noiseBlock, injectedSigmas: fmSigmas,
                injectedTimesteps: fmTimesteps)
            noiseCursor += 1
            let newRows = sampled[0 ..< toPred.count]                        // [n_step, C]

            // Write back into ALL THREE streams at the revealed positions.
            func writeBack(_ stream: inout PlannerStream, _ streamIdx: [Int32]) {
                var flat = stream.inputsEmbeds[0]
                let absolute = MLXArray(toPred.map { streamIdx[$0] })
                flat[absolute] = newRows.asType(flat.dtype)
                stream.inputsEmbeds = flat[.newAxis]
            }
            writeBack(&cond, cond.visualOutputIndices)
            writeBack(&uncond, uncond.visualOutputIndices)
            writeBack(&imgcond, imgcond.visualOutputIndices)
            eval(cond.inputsEmbeds, uncond.inputsEmbeds, imgcond.inputsEmbeds)
        }

        // Final cond/uncond forwards → for_gen contexts (inference branch of
        // feat_from_planner_to_renderer: cond mask covers ALL positions).
        let hCond = backbone.penultimateHidden(
            inputsEmbeds: cond.inputsEmbeds, mask: cond.mask,
            positionIds: cond.positionIds, layersToRun: tapLayers)
        let hUncond = backbone.penultimateHidden(
            inputsEmbeds: uncond.inputsEmbeds, mask: uncond.mask,
            positionIds: uncond.positionIds, layersToRun: tapLayers)

        let condContexts = glue.connector.forGen(hCond)      // [1, L_cond, 4096]
        let uncondContexts = glue.connector.forGen(hUncond)  // [1, L_uncond, 4096]

        let lCond = cond.inputsEmbeds.dim(1)
        let condVit = Set(cond.visualOutputIndices.map(Int.init))
        let condTxt = (0 ..< lCond).filter { !condVit.contains($0) }.map { Int32($0) }
        let lUncond = uncond.inputsEmbeds.dim(1)
        let uncondVit = Set(uncond.visualOutputIndices.map(Int.init))
        let uncondTxt = (0 ..< lUncond).filter { !uncondVit.contains($0) }.map { Int32($0) }

        let contexts = PlannerContexts(
            condEmbedsWtxtWvit: condContexts,
            condEmbedsWtxtWovit: condContexts[0][MLXArray(condTxt)][.newAxis],
            condEmbedsWotxtWvit: condContexts[0][condIdx][.newAxis],
            condEmbedsWotxtWovit: uncondContexts[0][MLXArray(uncondTxt)][.newAxis],
            predVitEmbed: cond.inputsEmbeds[0][condIdx][.newAxis])
        eval(contexts.condEmbedsWtxtWvit, contexts.condEmbedsWotxtWovit,
             contexts.predVitEmbed)
        return contexts
    }
}
