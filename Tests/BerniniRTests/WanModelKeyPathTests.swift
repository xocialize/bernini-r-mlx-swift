import Foundation
import MLX
import Testing
@testable import BerniniR

// S1c gate (structural half): the instantiated WanModel's flattened parameter
// paths must equal the 1095-key contract — proving the @ModuleInfo key plan,
// the optional norms (qk_norm / cross_attn_norm), the absence of norm1/norm2
// affine params, and that `freqs`/inv_freq buffers stay out of the tree.

private func a14bConfig() -> WanConfig {
    WanConfig(
        modelType: "t2v", modelVersion: "2.2", patchSize: [1, 2, 2], textLen: 512,
        inDim: 16, dim: 5120, ffnDim: 13824, freqDim: 256, textDim: 4096, outDim: 16,
        numHeads: 40, numLayers: 40, windowSize: [-1, -1], qkNorm: true,
        crossAttnNorm: true, eps: 1e-6, vaeStride: [4, 8, 8], vaeZDim: 16,
        dualModel: true, boundary: 0.875, sampleShift: 3.0, sampleSteps: 40,
        sampleGuideScale: [3.0, 4.0], numTrainTimesteps: 1000, sampleFps: 16,
        frameNum: 81, sampleNegPrompt: "", maxArea: 0, t5VocabSize: 256384,
        t5Dim: 4096, t5DimAttn: 4096, t5DimFfn: 10240, t5NumHeads: 64,
        t5NumLayers: 24, t5NumBuckets: 32, quantization: nil)
}

@Suite struct WanModelKeyPathTests {
    @Test func parameterPathsMatchContract() {
        let model = WanModel(a14bConfig())
        let paths = Set(model.parameters().flattened().map(\.0))
        let expected = BerniniWeightKeys.ditKeys()
        let missing = expected.subtracting(paths)
        let unexpected = paths.subtracting(expected)
        #expect(missing.isEmpty, "missing: \(missing.sorted().prefix(8))")
        #expect(unexpected.isEmpty, "unexpected: \(unexpected.sorted().prefix(8))")
    }

    @Test func ropeTableShapeAndHeadDimSplit() {
        let model = WanModel(a14bConfig())
        // head_dim 128 -> rope dims t44/h42/w42, half_d total 64
        #expect(model.freqs.shape == [1024, 64, 2])
    }
}

// S1d gate (schedule half): sigma schedule + integer timesteps must match the
// float64 numpy reference bit-for-bit at fp32. Goldens generated from the
// oracle venv's numpy (40 steps, shift 3.0 — the Bernini production config).
@Suite struct SchedulerScheduleTests {
    @Test func sigmasMatchNumpyGoldens() {
        let scheduler = FlowUniPCScheduler()
        scheduler.setTimesteps(40, shift: 3.0)
        #expect(scheduler.sigmas.count == 41)
        #expect(scheduler.sigmas[0] == 0.9996664524078369)
        #expect(scheduler.sigmas[1] == 0.9911890625953674)
        #expect(scheduler.sigmas[2] == 0.982419490814209)
        #expect(scheduler.sigmas[20] == 0.7496247887611389)
        #expect(scheduler.sigmas[39] == 0.07136054337024689)
        #expect(scheduler.sigmas[40] == 0.0)
    }

    @Test func integerTimestepsMatchReference() {
        let scheduler = FlowUniPCScheduler()
        scheduler.setTimesteps(40, shift: 3.0)
        #expect(scheduler.timesteps.count == 40)
        #expect(scheduler.timesteps[0] == 999.0)
        #expect(scheduler.timesteps[1] == 991.0)
        #expect(scheduler.timesteps[20] == 749.0)
        #expect(scheduler.timesteps[39] == 71.0)
        // Expert boundary (t >= 875 high-noise): switch happens at step 12,
        // whose integer timestep is 874 — the first low-noise step.
        let firstLow = scheduler.timesteps.firstIndex { $0 < 875 }
        #expect(firstLow == 12)
        #expect(scheduler.timesteps[12] == 874.0)
    }
}
