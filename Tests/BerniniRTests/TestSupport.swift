import MLX

// All numeric gates run on the CPU stream (parity doctrine). Pinning the
// GLOBAL default device to CPU — once, before any MLX op — keeps mlx from
// constructing the Metal device at TaskLocal-default resolution. That
// construction loads the metallib, which is flaky inside SPM test bundles
// (the `missing creator for mutated node (mlx-swift_Cmlx.bundle)` build
// quirk): CPU-only tests otherwise die on "Failed to load the default
// metallib" without ever needing the GPU.
private enum CPUPin {
    static let once: Void = {
        Device.setDefault(device: .cpu)
    }()
}

/// CPU-pinned scope for parity tests: global pin (pre-Metal-init) + the
/// TaskLocal scope.
func withCPU<T>(_ body: () throws -> T) rethrows -> T {
    CPUPin.once
    return try Device.withDefaultDevice(.cpu, body)
}
