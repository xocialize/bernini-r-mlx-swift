import Testing
@testable import BerniniR

// S0 gate: wrapper defaults must match the verified oracle config
// (bernini-r-mlx `tests/smoke/test_config.py` equivalent).
@Suite struct ConfigTests {
    @Test func defaultsMatchVerifiedOracleConfig() {
        let config = BerniniRendererConfig()
        #expect(config.switchDiTBoundary == 0.875)
        #expect(config.shift == 3.0)
        #expect(config.useSrcIdRotaryEmb)
        #expect(config.maxSequenceLength == 512)
        #expect(config.boundaryTimestep == 875.0)
    }
}
