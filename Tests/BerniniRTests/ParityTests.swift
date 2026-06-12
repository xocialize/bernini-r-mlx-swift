import Foundation
import MLX
import MLXNN
import Testing
@testable import BerniniR

// S1e numeric gates: Swift vs the Python oracle (mlx-video backbone), fixture
// driven. Everything pinned to the CPU stream — GPU fp32 matmul accumulates
// ~8e-4 relative error per op, which both masks real bugs and gets mistaken
// for them (mlx-porting parity doctrine).
//
// Fixture generation: tools/dump_parity_fixtures.py (oracle venv). Tests skip
// when a fixture or the weights mirror isn't present.
//
// The DiT full-forward gate loads the 28.6 GB high-noise expert — opt in with
// BERNINI_R_PARITY_DIT=1.

private func fixture(_ name: String) -> MLXArray? {
    guard
        let url = Bundle.module.url(
            forResource: name, withExtension: "npy", subdirectory: "Fixtures/parity")
    else { return nil }
    return try? loadNumpy(url: url)
}

private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

private let weightsMirror = URL(
    filePath: "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")

private func weightsAvailable() -> Bool {
    let env = ProcessInfo.processInfo.environment["BERNINI_R_WEIGHTS_DIR"]
    let root = env.map { URL(filePath: $0).appending(path: "ckpt-bf16") } ?? weightsMirror
    return FileManager.default.fileExists(atPath: root.path)
}

private func ckptURL(_ file: String) -> URL {
    let env = ProcessInfo.processInfo.environment["BERNINI_R_WEIGHTS_DIR"]
    let root = env.map { URL(filePath: $0).appending(path: "ckpt-bf16") } ?? weightsMirror
    return root.appending(path: file)
}

private func a14b() throws -> WanConfig {
    try WanConfig.load(from: ckptURL("config.json"))
}

private func onCPU<T>(_ body: () throws -> T) rethrows -> T {
    try Device.withDefaultDevice(.cpu, body)
}

@Suite(.serialized) struct ParityTests {

    // — RoPE: float64-derived tables and the precomputed grid factorization —

    @Test func ropeFreqTableMatchesOracle() throws {
        guard let expected = fixture("rope_freqs_d128") else { return }
        try onCPU {
            let d = 128
            let freqs = concatenated(
                [
                    ropeParams(1024, d - 4 * (d / 6)),
                    ropeParams(1024, 2 * (d / 6)),
                    ropeParams(1024, 2 * (d / 6)),
                ], axis: 1)
            let diff = maxAbs(freqs, expected)
            #expect(diff <= 1e-6, "rope freqs max_abs=\(diff)")
        }
    }

    @Test func ropePrecomputeMatchesOracle() throws {
        guard let expCos = fixture("rope_cos_grid344"),
              let expSin = fixture("rope_sin_grid344"),
              let freqs = fixture("rope_freqs_d128") else { return }
        try onCPU {
            let (cosF, sinF) = ropePrecomputeCosSin(gridSizes: [(3, 4, 4)], freqs: freqs)
            let dCos = maxAbs(cosF, expCos)
            let dSin = maxAbs(sinF, expSin)
            #expect(dCos <= 1e-6, "cos max_abs=\(dCos)")
            #expect(dSin <= 1e-6, "sin max_abs=\(dSin)")
        }
    }

    // — UniPC: full 40-step trajectory incl. corrector linear solves —

    @Test func unipcTrajectoryMatchesOracle() throws {
        guard var sample = fixture("unipc_sample0"),
              let vs = fixture("unipc_vs"),
              let expFinal = fixture("unipc_sample_final") else { return }
        try onCPU {
            let scheduler = FlowUniPCScheduler()
            scheduler.setTimesteps(40, shift: 3.0)
            let checkpoints: Set<Int> = [0, 1, 2, 12]
            for i in 0..<40 {
                sample = scheduler.step(
                    modelOutput: vs[i], timestep: scheduler.timesteps[i], sample: sample)
                if checkpoints.contains(i), let exp = fixture("unipc_sample_after_\(i)") {
                    let diff = maxAbs(sample, exp)
                    #expect(diff <= 1e-5, "unipc step \(i) max_abs=\(diff)")
                }
            }
            let diff = maxAbs(sample, expFinal)
            #expect(diff <= 1e-5, "unipc final max_abs=\(diff)")
        }
    }

    // — VAE: encode + decode on real fp32 weights —

    @Test func vaeEncodeDecodeMatchesOracle() throws {
        guard weightsAvailable(),
              let input = fixture("vae_input"),
              let expLatent = fixture("vae_latent"),
              let expDecoded = fixture("vae_decoded") else { return }
        try onCPU {
            let vae = WanVAE(zDim: 16, encoder: true)
            let weights = try MLX.loadArrays(url: ckptURL("vae.safetensors"))
            try vae.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])

            let z = vae.encode(input)
            eval(z)
            let dEnc = maxAbs(z, expLatent)
            #expect(dEnc <= 1e-4, "vae encode max_abs=\(dEnc)")

            let y = vae.decode(expLatent)
            eval(y)
            let dDec = maxAbs(y, expDecoded)
            #expect(dDec <= 5e-3, "vae decode max_abs=\(dDec)")
        }
    }

    // — umT5: full 24-layer forward on real weights (fp32, like the oracle) —

    @Test func t5FeaturesMatchOracle() throws {
        guard weightsAvailable(),
              let ids = fixture("t5_ids"),
              let expected = fixture("t5_features") else { return }
        try onCPU {
            let config = try a14b()
            let model = UMT5EncoderModel.fromConfig(config)
            let weights = try WeightLoader.loadVerifiedSafetensors(
                url: ckptURL("t5_encoder.safetensors"),
                expectedKeys: BerniniWeightKeys.t5Keys()
            ).mapValues { $0.asType(.float32) }
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])

            let out = model(ids.asType(.int32), mask: MLXArray.ones([1, 16], type: Int32.self))
            eval(out)
            let diff = maxAbs(out, expected)
            #expect(diff <= 1e-2, "t5 features max_abs=\(diff)")
        }
    }

    // — DiT: embed_text + full 40-block forward on the real high-noise expert.
    //   Heavy (28.6 GB load); opt in with BERNINI_R_PARITY_DIT=1. —

    @Test func ditForwardMatchesOracle() throws {
        guard ProcessInfo.processInfo.environment["BERNINI_R_PARITY_DIT"] == "1",
              weightsAvailable(),
              let x = fixture("dit_x"),
              let ctxRaw = fixture("dit_ctx_raw"),
              let expEmbedded = fixture("dit_ctx_embedded"),
              let expT999 = fixture("dit_out_t999"),
              let expT400 = fixture("dit_out_t400") else { return }
        try onCPU {
            let config = try a14b()
            let model = WanModel(config)
            let weights = try WeightLoader.loadVerifiedSafetensors(
                url: ckptURL("high_noise_model.safetensors"),
                expectedKeys: BerniniWeightKeys.ditKeys())
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])

            let embedded = model.embedText([ctxRaw])
            eval(embedded)
            let dEmb = maxAbs(embedded, expEmbedded)
            #expect(dEmb <= 5e-3, "embed_text max_abs=\(dEmb)")

            let out999 = model(
                [x], t: MLXArray([Float(999)]), context: .embedded(embedded), seqLen: 16)
            eval(out999[0])
            let d999 = maxAbs(out999[0], expT999)
            #expect(d999 <= 1e-2, "dit fwd t=999 max_abs=\(d999)")

            let out400 = model(
                [x], t: MLXArray([Float(400)]), context: .embedded(embedded), seqLen: 16)
            eval(out400[0])
            let d400 = maxAbs(out400[0], expT400)
            #expect(d400 <= 1e-2, "dit fwd t=400 max_abs=\(d400)")
        }
    }
}
