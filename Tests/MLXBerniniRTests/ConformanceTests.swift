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

    @Test func manifestDeclaresBothSurfaces() {
        let manifest = BerniniRPackage.manifest
        let capabilities = Set(manifest.surfaces.map(\.capability))
        #expect(capabilities == [.textToVideo, .textToImage])
        // Descriptors carry hand-tuned, non-empty summaries (C11).
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

    @Test func surfacesDeclareFastAndQualityModes() {
        // C11: the `.fast` accelerated mode must be introspectable so a planner
        // can choose it.
        for surface in BerniniRPackage.manifest.surfaces {
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

    @Test func registrationConstructs() throws {
        // C13: the engine constructs the package via the registration factory.
        let registration = PackageRegistration.of(BerniniRPackage.self)
        #expect(registration.manifest.surfaces.count == 2)
        let package = try registration.makePackage(BerniniRConfiguration())
        #expect(package is BerniniRPackage)
    }
}
