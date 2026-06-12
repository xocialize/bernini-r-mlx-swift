import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import BerniniR

// S4 gates: APG / CFG-chain sampling vs oracle goldens (tools/
// dump_sampling_golden.py). The RNG cross-binding check is light; the three
// sampler trajectories load BOTH experts — opt in with BERNINI_R_PARITY_E2E=1.

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

@Suite(.serialized) struct SamplingTests {

    /// Python-MLX and Swift-MLX wrap the same mlx::core RNG — verify the
    /// seed streams actually match (then `seed:` parameters are
    /// cross-binding-reproducible and noise injection is belt-and-braces).
    @Test func rngSeedStreamMatchesPythonMLX() throws {
        guard let expected = fixture("rng_seed42_target") else { return }
        try Device.withDefaultDevice(.cpu) {
            MLXRandom.seed(42)
            let noise = MLXRandom.normal([16, 1, 16, 16])
            let diff = maxAbs(noise, expected)
            #expect(diff == 0, "seed-42 stream diverges from Python MLX: max_abs=\(diff)")
        }
    }

    @Test func samplersMatchOracleGoldens() throws {
        guard ProcessInfo.processInfo.environment["BERNINI_R_PARITY_E2E"] == "1",
              FileManager.default.fileExists(atPath: weightsMirror.path),
              let noise = fixture("rng_seed42_target"),
              let ref = fixture("sampling_ref"),
              let video = fixture("sampling_video"),
              let ctxCondRaw = fixture("sampling_ctx_cond_raw"),
              let ctxNullRaw = fixture("sampling_ctx_null_raw"),
              let expR2V = fixture("sampling_r2v_final"),
              let expRV2V = fixture("sampling_rv2v_final"),
              let expV2V = fixture("sampling_v2v_final") else { return }

        try Device.withDefaultDevice(.cpu) {
            let renderer = try BerniniRendererModel.fromPretrained(modelDir: weightsMirror)
            let high = renderer.highNoiseExpert
            let low = renderer.lowNoiseExpert
            let boundary = renderer.boundaryTimestep

            let condHigh = high.embedText([ctxCondRaw])
            let condLow = low.embedText([ctxCondRaw])
            let uncondHigh = high.embedText([ctxNullRaw])
            let uncondLow = low.embedText([ctxNullRaw])
            eval(condHigh, condLow, uncondHigh, uncondLow)

            let outR2V = r2vSample(
                high: high, low: low, refLatents: [ref],
                condCtxHigh: condHigh, condCtxLow: condLow,
                uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
                targetShape: [16, 1, 16, 16], headDim: 128,
                boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
            let dR2V = maxAbs(outR2V, expR2V)
            #expect(dR2V <= 0.05, "r2v final max_abs=\(dR2V)")

            let outRV2V = cfgEditSample(
                high: high, low: low, guidanceMode: .rv2v,
                videoLatents: [video], refLatents: [ref],
                condCtxHigh: condHigh, condCtxLow: condLow,
                uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
                targetShape: [16, 1, 16, 16], headDim: 128,
                boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
            let dRV2V = maxAbs(outRV2V, expRV2V)
            #expect(dRV2V <= 0.05, "rv2v final max_abs=\(dRV2V)")

            let outV2V = cfgEditSample(
                high: high, low: low, guidanceMode: .v2v,
                videoLatents: [video], refLatents: [],
                condCtxHigh: condHigh, condCtxLow: condLow,
                uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
                targetShape: [16, 1, 16, 16], headDim: 128,
                boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
            let dV2V = maxAbs(outV2V, expV2V)
            #expect(dV2V <= 0.05, "v2v final max_abs=\(dV2V)")
        }
    }
}
