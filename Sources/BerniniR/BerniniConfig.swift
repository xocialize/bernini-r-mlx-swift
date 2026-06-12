// BerniniConfig — wrapper config mirroring the Python oracle's `config.py`
// (bernini-r-mlx). Defaults are oracle truths verified against the released
// checkpoint; deviations are port bugs (PORTING-SPEC.md "Pinned config").

import Foundation

/// Mirror of `BerniniRendererConfig` (oracle `bernini_r_mlx/config.py`).
public struct BerniniRendererConfig: Codable, Sendable {
    /// Expert switch point as a fraction of training timesteps; high-noise expert
    /// runs at t >= boundary * numTrainTimesteps.
    public var switchDiTBoundary: Double
    /// Flow shift for FlowUniPC. Bernini-R uses 3.0 — NOT Wan2.2's 12.0.
    public var shift: Double
    /// Segment-Aware 3D RoPE (zero-parameter position-id scheme). source_id == 0
    /// makes the phase the identity, so plain t2v is numerically stock Wan2.2.
    public var useSrcIdRotaryEmb: Bool
    /// UMT5 text token limit.
    public var maxSequenceLength: Int
    /// Training schedule length.
    public var numTrainTimesteps: Int

    public init(
        switchDiTBoundary: Double = 0.875,
        shift: Double = 3.0,
        useSrcIdRotaryEmb: Bool = true,
        maxSequenceLength: Int = 512,
        numTrainTimesteps: Int = 1000
    ) {
        self.switchDiTBoundary = switchDiTBoundary
        self.shift = shift
        self.useSrcIdRotaryEmb = useSrcIdRotaryEmb
        self.maxSequenceLength = maxSequenceLength
        self.numTrainTimesteps = numTrainTimesteps
    }

    /// Absolute timestep of the high/low expert boundary (875.0 at defaults).
    public var boundaryTimestep: Double { switchDiTBoundary * Double(numTrainTimesteps) }
}
