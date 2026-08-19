//
//  PlannerTextEncode.swift
//  BerniniR — Bernini-v2 planner plane (PORTING-SPEC-V2.md)
//
//  Prompt-driven unpadded umT5 encoding for the wvitcfg context concat —
//  upstream `get_t5_text_embeddings_sample` semantics: encode with an attention
//  mask, return the ACTUAL-length embeddings (wan-core `encodeText` already
//  slices to seqLen; the classic pad only exists inside the masked encoder
//  call). Loads umT5 fp32, encodes both prompts, evicts (§2.4 discipline).
//

import Foundation
import MLX
import MLXNN
import Tokenizers
import WanCore

public enum PlannerTextEncode {
    /// Returns (promptEmbeds [1, Lp, 4096], negEmbeds [1, Ln, 4096]) at actual
    /// lengths. The encoder is dropped and its buffers reclaimed before return.
    public static func encodeUnpadded(
        modelDir: URL, config: WanConfig, tokenizer: any Tokenizer,
        prompt: String, negativePrompt: String
    ) throws -> (MLXArray, MLXArray) {
        var encoder: UMT5EncoderModel? = UMT5EncoderModel.fromConfig(config)
        let t5Weights = try WeightLoader.loadVerifiedSafetensors(
            url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
            expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
        ).mapValues { $0.asType(.float32) }
        WeightLoader.materialize(t5Weights)
        try encoder!.update(
            parameters: ModuleParameters.unflattened(t5Weights),
            verify: [.noUnusedKeys])

        let p = encodeText(
            encoder: encoder!, tokenizer: tokenizer, prompt: prompt,
            textLen: config.textLen)
        let n = encodeText(
            encoder: encoder!, tokenizer: tokenizer, prompt: negativePrompt,
            textLen: config.textLen)
        eval(p, n)
        encoder = nil
        MLX.Memory.clearCache()
        return (p[.newAxis], n[.newAxis])
    }
}
