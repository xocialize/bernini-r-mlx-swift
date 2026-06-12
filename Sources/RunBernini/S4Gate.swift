// S4 gate as a CLI (RunBernini --s4-gate): the sampler-parity check outside
// the swiftpm-testing-helper environment, where an identical workload hits a
// spurious Metal GPU-timeout (under investigation; the plain-CLI GPU smoke
// and all single-expert test gates pass). Reads the same oracle fixtures the
// test bundle uses, straight from the source tree.

import Foundation
import MLX

import BerniniR

private let fixturesDir = URL(
    filePath: FileManager.default.currentDirectoryPath)
    .appending(path: "Tests/BerniniRTests/Fixtures/parity")

private func fixture(_ name: String) throws -> MLXArray {
    try loadNumpy(url: fixturesDir.appending(path: "\(name).npy"))
}

private func maxAbs(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

func runS4Gate(modelDir: URL) throws {
    try Device.withDefaultDevice(.cpu) {
        print("[s4-gate] loading dual experts (CPU stream)…")
        let renderer = try BerniniRendererModel.fromPretrained(modelDir: modelDir)
        let high = renderer.highNoiseExpert
        let low = renderer.lowNoiseExpert
        let boundary = renderer.boundaryTimestep

        let noise = try fixture("rng_seed42_target")
        let ref = try fixture("sampling_ref")
        let video = try fixture("sampling_video")
        let ctxCondRaw = try fixture("sampling_ctx_cond_raw")
        let ctxNullRaw = try fixture("sampling_ctx_null_raw")

        let condHigh = high.embedText([ctxCondRaw])
        let condLow = low.embedText([ctxCondRaw])
        let uncondHigh = high.embedText([ctxNullRaw])
        let uncondLow = low.embedText([ctxNullRaw])
        eval(condHigh, condLow, uncondHigh, uncondLow)

        print("[s4-gate] r2v…")
        let outR2V = r2vSample(
            high: high, low: low, refLatents: [ref],
            condCtxHigh: condHigh, condCtxLow: condLow,
            uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
            targetShape: [16, 1, 16, 16], headDim: 128,
            boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
        let dR2V = try maxAbs(outR2V, fixture("sampling_r2v_final"))
        print("  r2v final max_abs = \(dR2V)  (gate 0.05)")

        print("[s4-gate] rv2v…")
        let outRV2V = cfgEditSample(
            high: high, low: low, guidanceMode: .rv2v,
            videoLatents: [video], refLatents: [ref],
            condCtxHigh: condHigh, condCtxLow: condLow,
            uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
            targetShape: [16, 1, 16, 16], headDim: 128,
            boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
        let dRV2V = try maxAbs(outRV2V, fixture("sampling_rv2v_final"))
        print("  rv2v final max_abs = \(dRV2V)  (gate 0.05)")

        print("[s4-gate] v2v…")
        let outV2V = cfgEditSample(
            high: high, low: low, guidanceMode: .v2v,
            videoLatents: [video], refLatents: [],
            condCtxHigh: condHigh, condCtxLow: condLow,
            uncondCtxHigh: uncondHigh, uncondCtxLow: uncondLow,
            targetShape: [16, 1, 16, 16], headDim: 128,
            boundaryTimestep: boundary, steps: 4, injectedNoise: noise)
        let dV2V = try maxAbs(outV2V, fixture("sampling_v2v_final"))
        print("  v2v final max_abs = \(dV2V)  (gate 0.05)")

        let pass = dR2V <= 0.05 && dRV2V <= 0.05 && dV2V <= 0.05
        print(pass ? "[s4-gate] PASS" : "[s4-gate] FAIL")
        if !pass { exit(1) }
    }
}
