// S6 gate as a CLI (RunBernini --s6-gate): int4 vs bf16 per-pass cosine on
// the high-noise expert (the quantization-quality gate; the oracle's int4
// recipe measured 0.9992). Same DiT fixture input as the S1 forward gate.

import Foundation
import MLX
import MLXNN

import BerniniR

private func err(_ msg: String) {
    FileHandle.standardError.write(("[s6] " + msg + "\n").data(using: .utf8)!)
}

private func cosineSim(_ a: MLXArray, _ b: MLXArray) -> Float {
    let af = a.asType(.float32).flattened()
    let bf = b.asType(.float32).flattened()
    let dot = sum(af * bf)
    let norm = sqrt(sum(af * af)) * sqrt(sum(bf * bf))
    return (dot / (norm + 1e-12)).item(Float.self)
}

func runS6Gate(bf16Dir: URL, int4Dir: URL) throws {
    let fixtures = URL(filePath: FileManager.default.currentDirectoryPath)
        .appending(path: "Tests/BerniniRTests/Fixtures/parity")
    let x = try loadNumpy(url: fixtures.appending(path: "dit_x.npy"))
    let ctxRaw = try loadNumpy(url: fixtures.appending(path: "dit_ctx_raw.npy"))

    try Device.withDefaultDevice(.cpu) {
        let config = try WanConfig.load(from: bf16Dir.appending(path: "config.json"))

        func loadExpert(_ dir: URL, quantized: Bool) throws -> WanModel {
            err("construct WanModel (quantized=\(quantized))")
            let model = WanModel(config)
            if quantized {
                let q = try WanConfig.load(
                    from: dir.appending(path: "config.json")).quantization!
                err("applyQuantization begin")
                WeightLoader.applyQuantization(to: model, quantization: q)
                err("applyQuantization done")
            }
            err("loadVerifiedSafetensors begin")
            let weights = try WeightLoader.loadVerifiedSafetensors(
                url: dir.appending(path: "high_noise_model.safetensors"),
                expectedKeys: BerniniWeightKeys.ditKeys(quantized: quantized)
                    .subtracting(["freqs"]),
                toleratedExtras: quantized ? ["freqs"] : [])
            err("load done; update begin")
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])
            err("update done")
            return model
        }

        print("[s6-gate] loading bf16 high expert…")
        // Scope the bf16 expert so ARC frees its 28.6 GB before the int4
        // load — holding both plus init buffers drives memory pressure and
        // a Metal watchdog kill.
        let outBf16: MLXArray = try {
            let bf16 = try loadExpert(bf16Dir, quantized: false)
            let embedded = bf16.embedText([ctxRaw])
            let out = bf16(
                [x], t: MLXArray([Float(999)]), context: .embedded(embedded),
                seqLen: 16)[0]
            eval(out)
            return out
        }()
        MLX.GPU.clearCache()

        print("[s6-gate] loading int4 high expert…")
        let int4 = try loadExpert(int4Dir, quantized: true)
        // The int4 forward runs on the GPU stream: quantized matmuls route to
        // Metal even under a CPU pin, and a CPU-pinned 40-block graph then
        // becomes one Metal command buffer fenced on CPU ops at every block —
        // watchdog death (localized 2026-06-12). GPU float noise (~1e-3) is
        // negligible against int4 quantization error for a 0.999-cosine
        // quality gate, so the CPU-parity doctrine doesn't bind here.
        // text_embedding stays bf16 in the int4 recipe (only block linears
        // quantize); embed per-model for faithfulness.
        err("int4 forward (GPU stream)")
        let outInt4 = Device.withDefaultDevice(.gpu) {
            let embeddedQ = int4.embedText([ctxRaw])
            let out = int4(
                [x], t: MLXArray([Float(999)]), context: .embedded(embeddedQ),
                seqLen: 16)[0]
            eval(out)
            return out
        }
        err("int4 forward eval done")

        let cosine = cosineSim(outBf16, outInt4)
        // Gate = the oracle's own quantize gate (>= 0.99). On THIS fixture the
        // oracle (Python MLX) measures 0.9976654 and Swift 0.9977231 —
        // agreement to the 4th decimal (tools/int4_cosine_reference.py); the
        // published 0.9992 was a different input distribution.
        print("  per-pass cosine (full 40-block fwd, t=999) = \(cosine)  (gate 0.99; python ref 0.9976654)")
        let pass = cosine >= 0.99
        print(pass ? "[s6-gate] PASS" : "[s6-gate] FAIL")
        if !pass { exit(1) }
    }
}
