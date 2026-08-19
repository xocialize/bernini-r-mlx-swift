import BerniniR
import Foundation
import MLXToolKit
import Testing
@testable import MLXBerniniR

// S7 offline conformance (no MLX kernels — the per-edit CLI gate tier):
// manifest declarations, the two-layer license gate, surface descriptors,
// configuration Codable, and registration construction.

@Suite struct ConformanceTests {

    @Test func licenseGateAdmits() {
        // C7 (weights) + C8 (port code): Apache-2.0 / Apache-2.0 must be admitted.
        let result = LicensePolicy.permissiveOnly.evaluate(BerniniRPackage.manifest.license)
        #expect(result.isAdmitted)
    }

    @Test func manifestDeclaresSurfaces() {
        let manifest = BerniniRPackage.manifest
        let capabilities = Set(manifest.surfaces.map(\.capability))
        // textToVideo + textToImage (generation) + videoEdit (v2v/rv2v, contract 1.3.0).
        #expect(capabilities == [.textToVideo, .textToImage, .videoEdit])
        // Descriptors carry hand-tuned, non-empty summaries with a required prompt (C11).
        for surface in manifest.surfaces {
            #expect(!surface.summary.isEmpty)
            #expect(surface.parameters.contains { $0.name == "prompt" && $0.required })
        }
    }

    @Test func footprintsCoverBothPublishedVariants() {
        let quants = Set(BerniniRPackage.manifest.requirements.footprints.map(\.quant))
        #expect(quants == [.bf16, .int4])
        // Measured values, not weight sizes: each must exceed its variant's disk size
        // (bf16 ~64 GB, int4 ~27 GB on disk).
        for footprint in BerniniRPackage.manifest.requirements.footprints {
            switch footprint.quant {
            case .bf16: #expect(footprint.residentBytes > 64_000_000_000)
            case .int4: #expect(footprint.residentBytes > 27_000_000_000)
            default: break
            }
        }
    }

    @Test func requirementsGateOnMetalAndOS() {
        let requirements = BerniniRPackage.manifest.requirements
        #expect(requirements.requiredBackends.contains(.metalGPU))
        #expect((requirements.os.minMacOS?.major ?? 0) >= 26)
        #expect(requirements.chipFloor == .max)
    }

    @Test func configurationCodableRoundTrip() throws {
        var config = BerniniRConfiguration.int4
        // Environment-specific paths are excluded from Codable.
        config.modelDirectory = URL(filePath: "/tmp/somewhere")
        let data = try JSONEncoder().encode(config)
        let decoded = try JSONDecoder().decode(BerniniRConfiguration.self, from: data)
        #expect(decoded.repo == "mlx-community/Bernini-R-int4")
        #expect(decoded.quant == .int4)
        #expect(decoded.modelDirectory == nil)
    }

    @Test func generationSurfacesDeclareFastAndQualityModes() {
        // C11: the `.fast` accelerated mode must be introspectable on the GENERATION
        // surfaces (t2v/t2i) so a planner can choose it. (videoEdit has no speed mode —
        // editing runs the quality APG sampler.)
        let gen = BerniniRPackage.manifest.surfaces.filter {
            $0.capability == .textToVideo || $0.capability == .textToImage
        }
        #expect(gen.count == 2)
        for surface in gen {
            #expect(surface.supportedModes.contains(.fast))
            #expect(surface.supportedModes.contains(.quality))
        }
    }

    @Test func fastModeResolvesToDpmpp16() {
        // `.fast` → DPM++/16 (the validated 2.5× path); default → 40-step UniPC.
        let fast = resolveSampling(mode: .fast, steps: nil)
        #expect(fast.scheduler == .dpmpp)
        #expect(fast.steps == 16)

        let quality = resolveSampling(mode: .quality, steps: nil)
        #expect(quality.scheduler == .unipc)
        #expect(quality.steps == nil)  // nil → core uses the config default (40)

        // An explicit step count always wins over the mode default.
        #expect(resolveSampling(mode: .fast, steps: 8).steps == 8)
        #expect(resolveSampling(mode: nil, steps: nil).scheduler == .unipc)
    }

    @Test func lightningConfigRoundTrips() throws {
        let config = BerniniRConfiguration.lightning
        #expect(config.lightning)
        #expect(config.quant == .bf16)
        let decoded = try JSONDecoder().decode(
            BerniniRConfiguration.self, from: JSONEncoder().encode(config))
        #expect(decoded.lightning)  // the flag survives Codable
        // The standard config is not Lightning.
        #expect(!BerniniRConfiguration.int4.lightning)
    }

    @Test func v2ConfigurationsRoundTrip() throws {
        // E7: the Bernini-v2 checkpoint family rides in as CONFIGURATIONS of this
        // package — same surfaces, planner-conditioned dispatch keyed off the
        // resolved checkpoint's planner plane, not new modes (C12).
        let v2 = BerniniRConfiguration.v2
        #expect(v2.repo == "mlx-community/Bernini-v2-bf16")
        #expect(v2.quant == .bf16)
        #expect(!v2.lightning)

        let v2Int4 = BerniniRConfiguration.v2Int4
        #expect(v2Int4.repo == "mlx-community/Bernini-v2-int4")
        #expect(v2Int4.quant == .int4)

        for config in [v2, v2Int4] {
            let decoded = try JSONDecoder().decode(
                BerniniRConfiguration.self, from: JSONEncoder().encode(config))
            #expect(decoded.repo == config.repo)
            #expect(decoded.quant == config.quant)
        }
    }

    @Test func plannerPlaneProbeKeysOnV2Files() throws {
        // `hasPlannerPlane` is the planned-dispatch key: it must require BOTH the
        // MLLM shard and the DiffLoss head — a classic renderer checkpoint (or a
        // partial copy) must stay on the classic samplers.
        let root = FileManager.default.temporaryDirectory
            .appending(path: "bernini-v2-probe-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(
            at: root.appending(path: "mllm"), withIntermediateDirectories: true)
        #expect(!hasPlannerPlane(modelDir: root))

        FileManager.default.createFile(
            atPath: root.appending(path: "mllm/model.safetensors").path, contents: Data())
        #expect(!hasPlannerPlane(modelDir: root))  // MLLM alone is not the plane

        FileManager.default.createFile(
            atPath: root.appending(path: "vit_decoder.safetensors").path, contents: Data())
        #expect(hasPlannerPlane(modelDir: root))
    }

    @Test func registrationConstructs() throws {
        // C13: the engine constructs the package via the registration factory.
        let registration = PackageRegistration.of(BerniniRPackage.self)
        #expect(registration.manifest.surfaces.count == 3)
        let package = try registration.makePackage(BerniniRConfiguration())
        #expect(package is BerniniRPackage)
    }
}
