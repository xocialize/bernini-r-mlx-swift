//
//  QwenPlannerBackbone.swift
//  BerniniR — Bernini-v2 planner plane (PORTING-SPEC-V2.md)
//
//  The Qwen2.5-VL TEXT backbone as Bernini drives it: forward-only, full-sequence,
//  inputs-embeds injection, an explicit additive 4D attention mask, 3-axis M-RoPE
//  position ids, and a PENULTIMATE-layer tap (upstream reads `hidden_states[-2]` —
//  the output of layer N-1, before the final layer and final norm).
//
//  Donor-lifted from mlx-swift-lm `Qwen25VL.swift` (Language.Attention mropeCosSin /
//  rotateHalf) and stripped to be self-contained on MLX/MLXNN/MLXFast: no KV cache
//  (never autoregressive here), no lm_head, no vision tower (phase 1: t2v/t2i — all
//  output-ViT positions are overwritten by mask_tokens, so only token counts matter;
//  the vision tower lands with the r2v/v2v phase).
//
//  Weight source: `mllm/model.safetensors` (HF Qwen2.5-VL layout, Bernini-trained).
//  Module keys match the HF names so loading is prefix-strip + update.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// Text-config subset of `mllm/config.json` (HF Qwen2_5_VLConfig).
public struct QwenPlannerConfig: Decodable, Sendable {
    public var hiddenSize: Int
    public var numHiddenLayers: Int
    public var numAttentionHeads: Int
    public var numKeyValueHeads: Int
    public var intermediateSize: Int
    public var rmsNormEps: Float
    public var ropeTheta: Float
    public var vocabSize: Int
    public var mropeSection: [Int]

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabSize = "vocab_size"
        case ropeScaling = "rope_scaling"
    }

    struct RopeScaling: Codable {
        var mropeSection: [Int]
        enum CodingKeys: String, CodingKey { case mropeSection = "mrope_section" }
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
        numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
        numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        vocabSize = try c.decode(Int.self, forKey: .vocabSize)
        mropeSection = try c.decode(RopeScaling.self, forKey: .ropeScaling).mropeSection
    }

    public static func load(mllmDir: URL) throws -> QwenPlannerConfig {
        let data = try Data(contentsOf: mllmDir.appending(path: "config.json"))
        return try JSONDecoder().decode(QwenPlannerConfig.self, from: data)
    }
}

/// `cat(-x[..., d/2:], x[..., :d/2])` — standard rotate-half.
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

final class PlannerAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let mropeSectionRaw: [Int]
    // Leading underscore: computed from theta+headDim, not a trained weight —
    // Module's loader skips it.
    private let _invFreq: MLXArray

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: QwenPlannerConfig) {
        let dim = config.hiddenSize
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = dim / heads
        self.scale = pow(Float(headDim), -0.5)
        self.mropeSectionRaw = config.mropeSection

        self._qProj.wrappedValue = Linear(dim, heads * headDim, bias: true)
        self._kProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._vProj.wrappedValue = Linear(dim, kvHeads * headDim, bias: true)
        self._oProj.wrappedValue = Linear(heads * headDim, dim, bias: false)

        let freqIndices = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
        self._invFreq = 1.0 / pow(MLXArray(config.ropeTheta), freqIndices / Float(headDim))
    }

    /// M-RoPE cos/sin from `[3, batch, seq]` position ids — donor logic verbatim:
    /// start from the temporal axis, overwrite the H/W frequency ranges in place.
    private func mropeCosSin(positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let invFreqExpanded = _invFreq.reshaped(1, 1, -1, 1)                       // [1,1,d/2,1]
        let posExpanded = positionIds[0..., 0..., .newAxis, 0...].asType(.float32) // [3,b,1,L]
        var freqs = matmul(invFreqExpanded, posExpanded)                           // [3,b,d/2,L]
        freqs = freqs.transposed(0, 1, 3, 2)                                       // [3,b,L,d/2]

        var freqsT = freqs[0]
        var offset = mropeSectionRaw[0]
        for axis in 1 ..< mropeSectionRaw.count {
            let length = mropeSectionRaw[axis]
            freqsT[0..., 0..., offset ..< (offset + length)] =
                freqs[axis][0..., 0..., offset ..< (offset + length)]
            offset += length
        }

        let emb = concatenated([freqsT, freqsT], axis: -1)                         // [b,L,d]
        // [1, batch, seq, dim] for head-broadcast over [b, heads, L, d]
        return (MLX.cos(emb)[.newAxis, 0..., 0..., 0...].transposed(1, 0, 2, 3),
                MLX.sin(emb)[.newAxis, 0..., 0..., 0...].transposed(1, 0, 2, 3))
    }

    /// `mask` is the ADDITIVE 0/−inf `[B, L, L]` Bernini mask (broadcast over heads).
    func callAsFunction(_ x: MLXArray, mask: MLXArray, positionIds: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let (cos, sin) = mropeCosSin(positionIds: positionIds)                     // [b,1,L,d]
        queries = (queries * cos) + (rotateHalf(queries) * sin)
        keys = (keys * cos) + (rotateHalf(keys) * sin)

        let additive = mask[0..., .newAxis, 0..., 0...].asType(queries.dtype)      // [B,1,L,L]
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: additive
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)
        return oProj(out)
    }
}

final class PlannerMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dim: Int, hidden: Int) {
        self._gateProj.wrappedValue = Linear(dim, hidden, bias: false)
        self._upProj.wrappedValue = Linear(dim, hidden, bias: false)
        self._downProj.wrappedValue = Linear(hidden, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

final class PlannerDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: PlannerAttention
    @ModuleInfo(key: "mlp") var mlp: PlannerMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: RMSNorm

    init(_ config: QwenPlannerConfig) {
        self._selfAttn.wrappedValue = PlannerAttention(config)
        self._mlp.wrappedValue = PlannerMLP(
            dim: config.hiddenSize, hidden: config.intermediateSize)
        self._inputLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayernorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positionIds: MLXArray) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), mask: mask, positionIds: positionIds)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

/// The planner backbone: embeddings + decoder layers, tapped at the penultimate layer.
public final class QwenPlannerBackbone: Module {
    public let config: QwenPlannerConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [PlannerDecoderLayer]
    // The final RMSNorm exists in the checkpoint; the penultimate tap never runs
    // it, but it is declared so the loader owns every `model.*` key.
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(config: QwenPlannerConfig) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self._layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            PlannerDecoderLayer(config)
        }
        self._norm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func embed(_ inputIds: MLXArray) -> MLXArray { embedTokens(inputIds) }

    /// `hidden_states[-2]`: run layers 0 ..< N-1, skip the final layer and final norm.
    /// `layersToRun` overrides for gate experiments; nil = numHiddenLayers - 1.
    public func penultimateHidden(
        inputsEmbeds: MLXArray, mask: MLXArray, positionIds: MLXArray,
        layersToRun: Int? = nil
    ) -> MLXArray {
        let n = layersToRun ?? (config.numHiddenLayers - 1)
        var h = inputsEmbeds
        for layer in layers.prefix(n) {
            h = layer(h, mask: mask, positionIds: positionIds)
        }
        return h
    }

    /// Load from `mllm/model.safetensors`: keep `model.*` (prefix-stripped),
    /// drop the vision tower and lm_head (phase 1).
    public static func fromPretrained(mllmDir: URL, dtype: DType? = nil) throws
        -> QwenPlannerBackbone
    {
        let config = try QwenPlannerConfig.load(mllmDir: mllmDir)
        let model = QwenPlannerBackbone(config: config)
        let raw = try MLX.loadArrays(url: mllmDir.appending(path: "model.safetensors"))
        var params: [String: MLXArray] = [:]
        for (key, value) in raw {
            guard key.hasPrefix("model.") else { continue }  // skips visual.*, lm_head
            let stripped = String(key.dropFirst("model.".count))
            // HF stores rotary_emb.inv_freq in some exports; never a trained weight.
            if stripped.contains("rotary_emb") { continue }
            params[stripped] = dtype.map { value.asType($0) } ?? value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])
        return model
    }
}
