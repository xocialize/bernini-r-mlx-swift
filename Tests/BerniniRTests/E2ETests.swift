import Foundation
import MLX
import MLXNN
import Testing
@testable import BerniniR

// S2 gate: the full dual-expert CFG t2v denoise (4 steps — both experts +
// the UniPC corrector) plus VAE decode, vs the Python oracle's golden, with
// injected noise/contexts, on the CPU stream.
//
// Heavy (loads BOTH 28.6 GB experts ≈ 57 GB) — opt in with
// BERNINI_R_PARITY_E2E=1. Fixture generation: tools/dump_e2e_golden.py.

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

@Suite struct E2ETests {
    @Test func t2vDenoiseMatchesGolden() throws {
        guard ProcessInfo.processInfo.environment["BERNINI_R_PARITY_E2E"] == "1",
              FileManager.default.fileExists(atPath: weightsMirror.path),
              let noise = fixture("e2e_noise"),
              let ctxCond = fixture("e2e_ctx_cond"),
              let ctxNull = fixture("e2e_ctx_null"),
              let expFinal = fixture("e2e_latent_final"),
              let expFrames = fixture("e2e_frames") else { return }

        try withCPU {
            let renderer = try BerniniRendererModel.fromPretrained(modelDir: weightsMirror)

            // Per-step probes isolate which step diverges if the final gate fails.
            let latent = denoiseT2V(
                renderer: renderer,
                contextCond: ctxCond,
                contextNull: ctxNull,
                noise: noise,
                options: T2VOptions(steps: 4, shift: 3.0, guideScale: (3.0, 4.0))
            ) { step, _, current in
                if let exp = fixture("e2e_latent_after_\(step)") {
                    let d = maxAbs(current, exp)
                    #expect(d <= 0.05, "e2e latent after step \(step): max_abs=\(d)")
                }
            }
            let dFinal = maxAbs(latent, expFinal)
            #expect(dFinal <= 0.05, "e2e final latent max_abs=\(dFinal)")

            let vae = WanVAE(zDim: 16, encoder: true)
            let vaeWeights = try MLX.loadArrays(
                url: weightsMirror.appending(path: "vae.safetensors"))
            try vae.update(
                parameters: ModuleParameters.unflattened(vaeWeights),
                verify: [.noUnusedKeys])
            let frames = vae.decode(latent.expandedDimensions(axis: 0))
            eval(frames)
            let dFrames = maxAbs(frames, expFrames)
            #expect(dFrames <= 0.5, "e2e decoded frames max_abs=\(dFrames)")
        }
    }
}
