import Foundation
import MLX
import MLXNN
import MLXRandom
import Testing
@testable import BerniniR

// S5 gate: streaming decode must be BIT-IDENTICAL to whole-sequence decode
// (the oracle's own gate, parametrized over chunk-boundary cases), on real
// VAE weights, CPU stream.

private let weightsMirror = URL(
    filePath: "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")

@Suite struct StreamingDecodeTests {
    @Test(arguments: [1, 2, 3, 5])
    func streamingBitIdentical(tLat: Int) throws {
        guard FileManager.default.fileExists(atPath: weightsMirror.path) else { return }
        try withCPU {
            let vae = WanVAE(zDim: 16, encoder: true)
            let weights = try MLX.loadArrays(
                url: weightsMirror.appending(path: "vae.safetensors"))
            try vae.update(
                parameters: ModuleParameters.unflattened(weights),
                verify: [.noUnusedKeys])

            MLXRandom.seed(UInt64(100 + tLat))
            let z = MLXRandom.normal([1, 16, tLat, 8, 8])

            let whole = vae.decode(z)
            let streamed = decodeStreaming(vae: vae, z)
            eval(whole, streamed)

            #expect(whole.shape == streamed.shape)
            let diff = MLX.abs(whole - streamed).max().item(Float.self)
            #expect(diff == 0, "tLat=\(tLat): streaming diverges, max_abs=\(diff)")
        }
    }
}
