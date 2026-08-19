//
//  PlannerGlue.swift
//  BerniniR — Bernini-v2 planner plane (PORTING-SPEC-V2.md)
//
//  The small glue modules around the backbone:
//    - `MLPConnector` (upstream `bernini/models/bernini.py`): `for_vit` projects
//      planner hidden states back into ViT-embed space inside the MaskGIT loop;
//      `for_gen` projects the final hidden states into the 4096-d renderer
//      cross-attention context.
//    - `mask_tokens`: the learned mask-token table; inference broadcasts row 0
//      over every target-ViT position.
//    - `buildCustomAttentionMask` (upstream `bernini/data/utils/attention_utils.py`):
//      the additive 0/−inf mask — every query sees prior text/image-input tokens
//      causally; planning (p) and output (o) tokens additionally see their OWN
//      segment bidirectionally.
//
//  Weight source: `planner_glue.safetensors` (keys verbatim upstream:
//  `connector.proj_gen.{0,2,3}.*`, `connector.pred_vit.{0,2,3,4}.*`, `mask_tokens`).
//

import Foundation
import MLX
import MLXNN
import WanCore

/// Upstream `MLPConnector` (both branches enabled in v2).
public final class MLPConnector: Module {
    // proj_gen: Linear → GELU → RMSNorm → Linear   (indices 0,1,2,3)
    @ModuleInfo(key: "proj_gen_0") var projGen0: Linear
    @ModuleInfo(key: "proj_gen_2") var projGen2: RMSNorm
    @ModuleInfo(key: "proj_gen_3") var projGen3: Linear
    // pred_vit: Linear → GELU → Linear → RMSNorm → Linear   (indices 0,1,2,3,4)
    @ModuleInfo(key: "pred_vit_0") var predVit0: Linear
    @ModuleInfo(key: "pred_vit_2") var predVit2: Linear
    @ModuleInfo(key: "pred_vit_3") var predVit3: RMSNorm
    @ModuleInfo(key: "pred_vit_4") var predVit4: Linear

    public init(inDim: Int = 3584, outDimForGen: Int = 4096, outDimForVit: Int = 3584) {
        self._projGen0.wrappedValue = Linear(inDim, outDimForGen)
        self._projGen2.wrappedValue = RMSNorm(dimensions: outDimForGen, eps: 1e-6)
        self._projGen3.wrappedValue = Linear(outDimForGen, outDimForGen)
        self._predVit0.wrappedValue = Linear(inDim, outDimForVit)
        self._predVit2.wrappedValue = Linear(outDimForVit, outDimForVit)
        self._predVit3.wrappedValue = RMSNorm(dimensions: outDimForVit, eps: 1e-6)
        self._predVit4.wrappedValue = Linear(outDimForVit, outDimForVit)
    }

    public func forGen(_ x: MLXArray) -> MLXArray {
        projGen3(projGen2(gelu(projGen0(x))))
    }

    public func forVit(_ x: MLXArray) -> MLXArray {
        predVit4(predVit3(predVit2(gelu(predVit0(x)))))
    }
}

/// `planner_glue.safetensors` bundle: connector + mask-token table.
public struct PlannerGlue {
    public let connector: MLPConnector
    /// `[1, num_mask_token, hidden]` — inference uses row 0 broadcast.
    public let maskTokens: MLXArray

    public static func fromPretrained(file: URL, dtype: DType? = nil) throws -> PlannerGlue {
        // CPU-pin (watchdog doctrine — see QwenPlannerBackbone.fromPretrained).
        let (params, maskTokens) = try Device.withDefaultDevice(.cpu) {
            () -> ([String: MLXArray], MLXArray) in
            let raw = try MLX.loadArrays(url: file)
            guard let maskTokens = raw["mask_tokens"] else {
                throw NSError(
                    domain: "BerniniPlanner", code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "planner_glue missing mask_tokens"])
            }
            var params: [String: MLXArray] = [:]
            for (key, value) in raw where key.hasPrefix("connector.") {
                var k = String(key.dropFirst("connector.".count))
                for idx in ["0", "2", "3", "4"] {
                    k = k.replacingOccurrences(of: "proj_gen.\(idx)", with: "proj_gen_\(idx)")
                    k = k.replacingOccurrences(of: "pred_vit.\(idx)", with: "pred_vit_\(idx)")
                }
                params[k] = dtype.map { value.asType($0) } ?? value
            }
            let cast = dtype.map { maskTokens.asType($0) } ?? maskTokens
            WeightLoader.materialize(params)
            eval(cast)
            return (params, cast)
        }
        let connector = MLPConnector()
        try connector.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])
        return PlannerGlue(connector: connector, maskTokens: maskTokens)
    }
}

/// Bernini token roles in the planner sequence (upstream `token_type` values).
public enum PlannerTokenType: Int32 {
    case text = 0       // t — causal
    case planning = 1   // p — causal over t/i + bidirectional within same segment
    case imageInput = 2 // i — causal
    case output = 3     // o — causal over t/i + bidirectional within same segment
}

/// Port of `build_custom_attention_mask`: `[B, L]` token types + segment ids →
/// additive `[B, L, L]` mask (0 visible, −inf hidden). Computed host-side in
/// fp32 — L is a few hundred; clarity over kernel-golf.
public func buildCustomAttentionMask(
    tokenType: [[Int32]], tokenSegmentIds: [[Int32]]
) -> MLXArray {
    let b = tokenType.count
    let l = tokenType[0].count
    var data = [Float](repeating: -.infinity, count: b * l * l)
    for bi in 0 ..< b {
        let types = tokenType[bi]
        let segs = tokenSegmentIds[bi]
        for q in 0 ..< l {
            let qType = types[q]
            let qSeg = segs[q]
            let base = bi * l * l + q * l
            for k in 0 ..< l {
                let kType = types[k]
                let kIsTI = kType == 0 || kType == 2
                var visible = kIsTI && k <= q  // causal over t/i, all query kinds
                if qType == 1 { visible = visible || (kType == 1 && segs[k] == qSeg) }
                if qType == 3 { visible = visible || (kType == 3 && segs[k] == qSeg) }
                if visible { data[base + k] = 0 }
            }
        }
    }
    return MLXArray(data, [b, l, l])
}
