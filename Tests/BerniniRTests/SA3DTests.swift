import Foundation
import MLX
import MLXNN
import Testing
@testable import BerniniR

// S3 gates: SA-3D RoPE phase math vs oracle fixtures (light), the
// target-only ≡ plain-forward identity on a random-init tiny config
// (structural, no weights), and the real-expert multiseg parity
// (heavy — shares the BERNINI_R_PARITY_DIT=1 opt-in).

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

/// Tiny structural config (mirrors the oracle's small-config smoke test):
/// dim 64 · 4 heads (head_dim 16) · 2 layers — paths and math, not weights.
private func tinyConfig() -> WanConfig {
    WanConfig(
        modelType: "t2v", modelVersion: "2.2", patchSize: [1, 2, 2], textLen: 16,
        inDim: 16, dim: 64, ffnDim: 128, freqDim: 32, textDim: 32, outDim: 16,
        numHeads: 4, numLayers: 2, windowSize: [-1, -1], qkNorm: true,
        crossAttnNorm: true, eps: 1e-6, vaeStride: [4, 8, 8], vaeZDim: 16,
        dualModel: true, boundary: 0.875, sampleShift: 3.0, sampleSteps: 40,
        sampleGuideScale: [3.0, 4.0], numTrainTimesteps: 1000, sampleFps: 16,
        frameNum: 81, sampleNegPrompt: "", maxArea: 0, t5VocabSize: 256384,
        t5Dim: 4096, t5DimAttn: 4096, t5DimFfn: 10240, t5NumHeads: 64,
        t5NumLayers: 24, t5NumBuckets: 32, quantization: nil)
}

@Suite(.serialized) struct SA3DTests {

    @Test func visualIDPhaseMatchesOracle() throws {
        try withCPU {
            for sid in [0, 1, 3] {
                guard let expCos = fixture("sa3d_phase_cos_sid\(sid)"),
                      let expSin = fixture("sa3d_phase_sin_sid\(sid)") else { return }
                let (c, s) = visualIDPhase(sourceID: sid, headDim: 128)
                let dC = maxAbs(c, expCos)
                let dS = maxAbs(s, expSin)
                #expect(dC <= 1e-6, "phase cos sid \(sid) max_abs=\(dC)")
                #expect(dS <= 1e-6, "phase sin sid \(sid) max_abs=\(dS)")
            }
            // source_id 0 is the identity phase — the t2v-unchanged guarantee
            let (c0, s0) = visualIDPhase(sourceID: 0, headDim: 128)
            #expect(maxAbs(c0, MLXArray.ones([64])) == 0)
            #expect(maxAbs(s0, MLXArray.zeros([64])) == 0)
        }
    }

    @Test func multiSegmentRopeMatchesOracle() throws {
        guard let expCos = fixture("sa3d_cos_ref144s1_tgt344s0"),
              let expSin = fixture("sa3d_sin_ref144s1_tgt344s0"),
              let freqs = fixture("rope_freqs_d128") else { return }
        try withCPU {
            let (cos, sin) = prepareSA3DRopeCosSin(
                segments: [((1, 4, 4), 1), ((3, 4, 4), 0)], freqs: freqs, headDim: 128)
            let dC = maxAbs(cos, expCos)
            let dS = maxAbs(sin, expSin)
            #expect(dC <= 1e-6, "sa3d cos max_abs=\(dC)")
            #expect(dS <= 1e-6, "sa3d sin max_abs=\(dS)")
        }
    }

    /// Oracle smoke-test equivalent: with NO conditioning segments, the
    /// multiseg path must reproduce the plain forward (same random-init
    /// weights, two different code paths).
    @Test func targetOnlyMultisegMatchesPlainForward() throws {
        try withCPU {
            MLXRandom.seed(3)
            let config = tinyConfig()
            let model = WanModel(config)

            let target = MLXRandom.normal([16, 1, 8, 8])
            let ctxRaw = MLXRandom.normal([8, config.textDim]) * 0.5
            let embedded = model.embedText([ctxRaw])

            let plain = model(
                [target], t: MLXArray([Float(999)]), context: .embedded(embedded),
                seqLen: 16)[0]
            let multi = forwardMultiseg(
                model: model, condSegments: [], targetLatent: target,
                t: MLXArray(Float(999)), context: embedded, headDim: config.headDim)
            eval(plain, multi)
            let diff = maxAbs(plain, multi)
            #expect(diff <= 1e-3, "target-only multiseg vs plain: max_abs=\(diff)")
        }
    }

    /// Real-expert multiseg parity (heavy — 28.6 GB load).
    @Test func multisegMatchesOracleOnRealExpert() throws {
        guard ProcessInfo.processInfo.environment["BERNINI_R_PARITY_DIT"] == "1",
              FileManager.default.fileExists(atPath: weightsMirror.path),
              let target = fixture("multiseg_target"),
              let ref = fixture("multiseg_ref"),
              let ctxRaw = fixture("multiseg_ctx_raw"),
              let expTargetOnly = fixture("multiseg_out_targetonly"),
              let exp1Ref = fixture("multiseg_out_1ref") else { return }
        try withCPU {
            let config = try WanConfig.load(
                from: weightsMirror.appending(path: "config.json"))
            let model = WanModel(config)
            let weights = try WeightLoader.loadVerifiedSafetensors(
                url: weightsMirror.appending(path: "high_noise_model.safetensors"),
                expectedKeys: BerniniWeightKeys.ditKeys())
            try model.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])

            let embedded = model.embedText([ctxRaw])
            eval(embedded)
            let t = MLXArray(Float(999))

            let outTargetOnly = forwardMultiseg(
                model: model, condSegments: [], targetLatent: target, t: t,
                context: embedded, headDim: config.headDim)
            eval(outTargetOnly)
            let dT = maxAbs(outTargetOnly, expTargetOnly)
            #expect(dT <= 1e-2, "multiseg target-only max_abs=\(dT)")

            let out1Ref = forwardMultiseg(
                model: model, condSegments: [(ref, 1)], targetLatent: target, t: t,
                context: embedded, headDim: config.headDim)
            eval(out1Ref)
            let dR = maxAbs(out1Ref, exp1Ref)
            #expect(dR <= 1e-2, "multiseg 1-ref max_abs=\(dR)")
        }
    }
}
