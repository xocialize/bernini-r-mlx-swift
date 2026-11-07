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
        // The weight dict lives ONLY inside this closure — the module is the sole
        // strong ref by the time `encoder = nil` runs, so `clearCache()` below can
        // actually return the ~11 GB fp32 working set (the §2.4 discipline
        // `BerniniPipeline.withTextEncoder` keeps by scoping the load the same way).
        // CPU-pin the load+cast (watchdog doctrine — see
        // QwenPlannerBackbone.fromPretrained): in the package path this runs
        // after the plan phase has churned the page cache, so cold mmap reads
        // are the normal case and must not ride a Metal command buffer.
        var encoder: UMT5EncoderModel? = try {
            let model = UMT5EncoderModel.fromConfig(config)
            let t5Weights = try Device.withDefaultDevice(.cpu) { () -> [String: MLXArray] in
                let cast = try WeightLoader.loadVerifiedSafetensors(
                    url: modelDir.appendingPathComponent("t5_encoder.safetensors"),
                    expectedKeys: BerniniWeightKeys.t5Keys(layers: config.t5NumLayers)
                ).mapValues { $0.asType(.float32) }
                WeightLoader.materialize(cast)
                return cast
            }
            try model.update(
                parameters: ModuleParameters.unflattened(t5Weights),
                verify: [.noUnusedKeys])
            return model
        }()

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
