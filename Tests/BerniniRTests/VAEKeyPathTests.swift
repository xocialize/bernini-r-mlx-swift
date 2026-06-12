import Foundation
import Testing
@testable import BerniniR

// S1 gate (structural half): the WanVAE module tree, instantiated with no
// weights, must flatten to EXACTLY the 194 parameter key paths pinned in
// Fixtures/vae_keys.txt (header-verified against the published checkpoint's
// vae.safetensors). This is what lets the loader refuse partial loads —
// any drift in the Sequential-index emulation (residual.0/.2/.3/.6,
// resample.1, head.0/.2, time_conv, shortcut, to_qkv, gamma) shows up here.
//
// Pure structural test: arrays are created lazily and never evaluated, so no
// Metal kernel runs and this passes under plain `xcrun swift test` (no
// metallib needed — that constraint only bites tests that eval on GPU).
@Suite struct VAEKeyPathTests {

    private func fixtureKeys() throws -> Set<String> {
        let url = Bundle.module.url(
            forResource: "vae_keys", withExtension: "txt", subdirectory: "Fixtures")!
        return Set(
            try String(contentsOf: url, encoding: .utf8)
                .split(separator: "\n").map(String.init))
    }

    @Test func flattenedParameterPathsMatchFixture() throws {
        let expected = try fixtureKeys()
        #expect(expected.count == 194)

        let vae = WanVAE(zDim: 16, encoder: true)
        let actual = Set(vae.parameters().flattened().map(\.0))

        let missing = expected.subtracting(actual)
        let unexpected = actual.subtracting(expected)
        #expect(
            missing.isEmpty,
            "\(missing.count) fixture keys absent from module, e.g. \(missing.sorted().prefix(5))")
        #expect(
            unexpected.isEmpty,
            "\(unexpected.count) module keys not in fixture, e.g. \(unexpected.sorted().prefix(5))")
        #expect(actual.count == 194)
    }

    @Test func decoderOnlyConstructionDropsEncoderSide() throws {
        let expected = try fixtureKeys().filter {
            !($0.hasPrefix("encoder.") || $0.hasPrefix("conv1."))
        }
        let vae = WanVAE(zDim: 16, encoder: false)
        let actual = Set(vae.parameters().flattened().map(\.0))
        #expect(actual == Set(expected))
    }

    @Test func parametersAreFP32() throws {
        // The checkpoint VAE ships fp32; the module must materialize fp32 slots.
        let vae = WanVAE(zDim: 16, encoder: true)
        let dtypes = Set(vae.parameters().flattened().map { $0.1.dtype })
        #expect(dtypes == [.float32])
    }
}
