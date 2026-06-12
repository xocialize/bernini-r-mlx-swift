// S5 gate as a CLI (RunBernini --s5-gate): streaming decode must be
// BIT-IDENTICAL to whole-sequence decode at 1/2/3/5 latent frames (the
// chunk-boundary cases), real VAE weights, CPU stream. CLI form because the
// SPM test product's Cmlx bundle assembly is broken in this environment
// (metallib not found from the xctest context — same family as the S4
// testing-helper issue; executables are unaffected).

import Foundation
import MLX
import MLXNN
import MLXRandom

import BerniniR

func runS5Gate(modelDir: URL) throws {
    try Device.withDefaultDevice(.cpu) {
        print("[s5-gate] loading VAE (fp32, CPU stream)…")
        let vae = WanVAE(zDim: 16, encoder: true)
        let weights = try MLX.loadArrays(
            url: modelDir.appending(path: "vae.safetensors"))
        try vae.update(
            parameters: ModuleParameters.unflattened(weights),
            verify: [.noUnusedKeys])

        var allPass = true
        for tLat in [1, 2, 3, 5] {
            MLXRandom.seed(UInt64(100 + tLat))
            let z = MLXRandom.normal([1, 16, tLat, 8, 8])

            let whole = vae.decode(z)
            let streamed = decodeStreaming(vae: vae, z)
            eval(whole, streamed)

            let shapeOK = whole.shape == streamed.shape
            let diff = MLX.abs(whole - streamed).max().item(Float.self)
            let pass = shapeOK && diff == 0
            allPass = allPass && pass
            print(
                "  tLat=\(tLat): shape \(streamed.shape) "
                    + "max_abs=\(diff) \(pass ? "OK (bit-identical)" : "MISMATCH")")
        }
        print(allPass ? "[s5-gate] PASS" : "[s5-gate] FAIL")
        if !allPass { exit(1) }
    }
}
