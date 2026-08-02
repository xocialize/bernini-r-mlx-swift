import Foundation
import MLX
import WanCore

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

/// Serializes every test body that touches MLX's PROCESS-GLOBAL state.
///
/// swift-testing runs suites in parallel with one another, and runs the cases
/// of a non-`.serialized` parameterized test in parallel too — `.serialized`
/// only orders tests *within* a single suite. This bundle shares two MLX
/// globals across all of them:
///
///  * **the RNG key sequence** — `MLXRandom.seed` resets it and EVERY draw
///    advances it. That includes the implicit draws MLXNN makes for random
///    weight init, so `WanModel(config)` / `WanVAE(...)` are RNG consumers
///    even in "pure structural" tests that never eval: the key split is
///    eager (the global key is reseated at call time), only the resulting
///    arrays are lazy.
///  * **the default device** — `Device.setDefault`, pinned above.
///
/// `SamplingTests.rngSeedStreamMatchesPythonMLX` asserts an ABSOLUTE stream
/// (seed 42 → `rng_seed42_target`), and production depends on exactly that:
/// the `seed:` parameters on `denoise*` / `generate` seed this same global
/// (Sampling.swift, BerniniPipeline.swift, BerniniI2V.swift). Any concurrent
/// draw landing between its `seed(42)` and its `normal(...)` reseats the
/// stream, and the gate fails with "seed-42 stream diverges from Python MLX".
/// For an absolute-stream assertion the only sound fix is that nothing else
/// in the process draws while it runs — hence one lock, taken by every
/// MLX-touching body rather than only the three that call `seed` explicitly.
///
/// Recursive because helpers nest it (e.g. `ParityTests.onCPU`).
private let mlxGlobalLock = NSRecursiveLock()

/// CPU-pinned, globally-serialized scope for MLX tests: the global device pin
/// (pre-Metal-init) + the TaskLocal device scope + the global-state lock.
///
/// EVERY test that constructs an MLX/MLXNN object or draws from `MLXRandom`
/// must go through this — including structural tests that never evaluate,
/// since module construction advances the RNG stream (see above).
func withCPU<T>(_ body: () throws -> T) rethrows -> T {
    CPUPin.once
    mlxGlobalLock.lock()
    defer { mlxGlobalLock.unlock() }
    return try Device.withDefaultDevice(.cpu, body)
}
