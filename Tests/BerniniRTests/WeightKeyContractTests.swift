import Foundation
import Testing
@testable import BerniniR

// S0 gate: the generated key contract must match the actual checkpoint headers
// byte-for-byte (0 missing / 0 unexpected), for both variants. Pure-Foundation
// safetensors header reads — no MLX, runs on the offline CLI tier.
//
// Weights resolve via BERNINI_R_WEIGHTS_DIR (a dir containing ckpt-bf16/ and
// ckpt-int4/, or itself a ckpt dir), falling back to the workspace's archive
// mirror. Tests skip cleanly when neither is mounted.

private struct SafetensorsHeader {
    let entries: [String: (dtype: String, shape: [Int])]

    init(url: URL) throws {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        let lenData = try handle.read(upToCount: 8)!
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
        let json = try JSONSerialization.jsonObject(
            with: handle.read(upToCount: Int(len))!) as! [String: Any]
        var entries: [String: (String, [Int])] = [:]
        for (key, value) in json where key != "__metadata__" {
            let v = value as! [String: Any]
            entries[key] = (v["dtype"] as! String, v["shape"] as! [Int])
        }
        self.entries = entries
    }

    var keys: Set<String> { Set(entries.keys) }
}

private let archiveMirror = "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights"

private func weightsRoot() -> URL? {
    let env = ProcessInfo.processInfo.environment["BERNINI_R_WEIGHTS_DIR"]
    for candidate in [env, archiveMirror].compactMap({ $0 }) {
        if FileManager.default.fileExists(atPath: candidate) {
            return URL(filePath: candidate)
        }
    }
    return nil
}

private func ckpt(_ variant: String) -> URL? {
    guard let root = weightsRoot() else { return nil }
    let nested = root.appending(path: variant)
    if FileManager.default.fileExists(atPath: nested.path) { return nested }
    return nil
}

private func assertExactKeySet(
    _ header: SafetensorsHeader, _ expected: Set<String>, label: String
) {
    let missing = expected.subtracting(header.keys)
    let unexpected = header.keys.subtracting(expected)
    #expect(missing.isEmpty, "\(label): \(missing.count) missing, e.g. \(missing.sorted().prefix(5))")
    #expect(unexpected.isEmpty, "\(label): \(unexpected.count) unexpected, e.g. \(unexpected.sorted().prefix(5))")
}

@Suite struct WeightKeyContractTests {

    @Test func generatedCountsMatchPinnedInventory() {
        // Header-verified 2026-06-12; these hold with no weights mounted.
        #expect(BerniniWeightKeys.ditKeys().count == 1095)
        #expect(BerniniWeightKeys.ditKeys(quantized: true).count == 1896)
        #expect(BerniniWeightKeys.t5Keys().count == 242)
    }

    @Test func bf16ExpertsMatchContract() throws {
        guard let dir = ckpt("ckpt-bf16") else { return }
        for expert in ["high_noise_model", "low_noise_model"] {
            let header = try SafetensorsHeader(
                url: dir.appending(path: "\(expert).safetensors"))
            assertExactKeySet(header, BerniniWeightKeys.ditKeys(), label: "bf16 \(expert)")
            #expect(Set(header.entries.values.map(\.dtype)) == ["BF16"])
        }
    }

    @Test func int4ExpertsMatchQuantizedContract() throws {
        guard let dir = ckpt("ckpt-int4") else { return }
        for expert in ["high_noise_model", "low_noise_model"] {
            let header = try SafetensorsHeader(
                url: dir.appending(path: "\(expert).safetensors"))
            assertExactKeySet(
                header, BerniniWeightKeys.ditKeys(quantized: true),
                label: "int4 \(expert)")
            // Bit-packed weights are U32; scales/biases ride bf16; freqs is fp32.
            #expect(header.entries["blocks.0.self_attn.q.weight"]?.dtype == "U32")
            #expect(header.entries["blocks.0.self_attn.q.scales"]?.dtype == "BF16")
            #expect(header.entries["freqs"]?.dtype == "F32")
        }
    }

    @Test func t5EncoderMatchesContract() throws {
        guard let dir = ckpt("ckpt-bf16") else { return }
        let header = try SafetensorsHeader(url: dir.appending(path: "t5_encoder.safetensors"))
        assertExactKeySet(header, BerniniWeightKeys.t5Keys(), label: "t5_encoder")
    }

    @Test func vaeMatchesFixture() throws {
        guard let dir = ckpt("ckpt-bf16") else { return }
        let fixture = Bundle.module.url(forResource: "vae_keys", withExtension: "txt",
                                        subdirectory: "Fixtures")!
        let expected = Set(
            try String(contentsOf: fixture, encoding: .utf8)
                .split(separator: "\n").map(String.init))
        #expect(expected.count == 194)
        let header = try SafetensorsHeader(url: dir.appending(path: "vae.safetensors"))
        assertExactKeySet(header, expected, label: "vae")
        // The VAE ships fp32 — loading it as bf16 is a port bug, not a choice.
        #expect(Set(header.entries.values.map(\.dtype)) == ["F32"])
    }

    @Test func configShapeSpotChecks() throws {
        guard let dir = ckpt("ckpt-bf16") else { return }
        let config = try WanConfig.load(from: dir.appending(path: "config.json"))
        let header = try SafetensorsHeader(
            url: dir.appending(path: "high_noise_model.safetensors"))
        let patchVolume = config.patchSize.reduce(1, *) * config.inDim  // 1*2*2*16 = 64
        #expect(header.entries["patch_embedding_proj.weight"]?.shape == [config.dim, patchVolume])
        #expect(header.entries["blocks.0.self_attn.q.weight"]?.shape == [config.dim, config.dim])
        #expect(header.entries["blocks.0.ffn.fc1.weight"]?.shape == [config.ffnDim, config.dim])
        #expect(header.entries["text_embedding_0.weight"]?.shape == [config.dim, config.textDim])
        #expect(header.entries["time_embedding_0.weight"]?.shape == [config.dim, config.freqDim])
        #expect(header.entries["time_projection.weight"]?.shape == [6 * config.dim, config.dim])
        #expect(header.entries["blocks.0.modulation"]?.shape == [1, 6, config.dim])
        #expect(header.entries["head.modulation"]?.shape == [1, 2, config.dim])
        // head projects dim -> patch volume * out_dim (1*2*2 * 16 = 64)
        #expect(header.entries["head.head.weight"]?.shape
                == [config.patchSize.reduce(1, *) * config.outDim, config.dim])
    }

    @Test func resolvedConfigMatchesOracleTruths() throws {
        guard let dir = ckpt("ckpt-bf16") else { return }
        let config = try WanConfig.load(from: dir.appending(path: "config.json"))
        #expect(config.dim == 5120)
        #expect(config.ffnDim == 13824)
        #expect(config.numHeads == 40)
        #expect(config.numLayers == 40)
        #expect(config.headDim == 128)
        #expect(config.patchSize == [1, 2, 2])
        #expect(config.qkNorm)
        #expect(config.crossAttnNorm)
        #expect(config.eps == 1e-06)
        #expect(config.vaeZDim == 16)
        #expect(config.vaeStride == [4, 8, 8])
        #expect(config.dualModel)
        #expect(config.boundaryTimestep == 875.0)
        #expect(config.sampleShift == 3.0)
        #expect(config.sampleGuideScale == [3.0, 4.0])
        #expect(config.textLen == 512)
        #expect(config.quantization == nil)

        let int4 = try WanConfig.load(
            from: ckpt("ckpt-int4")!.appending(path: "config.json"))
        #expect(int4.quantization == WanQuantization(groupSize: 64, bits: 4))
    }
}
