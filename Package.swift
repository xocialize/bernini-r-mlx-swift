// swift-tools-version: 6.2
// bernini-r-mlx-swift — Swift/MLX port of ByteDance Bernini-R (provenance-audited
// byte-stock Wan2.2-T2V-A14B) for MLXEngine's textToVideo/textToImage, with the
// renderer's r2v/v2v/rv2v editing surfaces. Python oracle: DEV_ARCHIVE/bernini-r-mlx;
// Swift component donor: DEV_ARCHIVE/longcat-avatar-mlx-swift. See PORTING-SPEC.md.

import PackageDescription

let package = Package(
    name: "BerniniR",
    platforms: [
        // v26 to match the MLXEngine contract (MLXToolKit) the wrapper target links.
        .macOS(.v26)
    ],
    products: [
        .library(name: "BerniniR", targets: ["BerniniR"]),
        // The MLXEngine wrapper: a conformant `ModelPackage` over the core pipeline.
        .library(name: "MLXBerniniR", targets: ["MLXBerniniR"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", from: "0.30.0"),
        // Tokenizers (umT5 sentencepiece) only; weight download is our own loader.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        // MLXEngine contract (MLXToolKit) for the wrapper target; ≥0.27.0 for the CAN
        // cancellation gate (MLXServeConformance.CancellationConformance). The core
        // `BerniniR` target stays engine-agnostic.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.27.0"),
        // The neutral Wan substrate (DiT + VAE + umT5 + RoPE + schedulers + loader),
        // extracted so Helios/Phantom/TI2V-5B share it. Local path during B0; tagged dep later.
        // The shared Wan substrate (B0 extraction). Versioned dep since 2026-08-14 —
        // this is the "swap bernini's path dep → tagged URL" formalization the B0 plan
        // left user-gated (WAN-CORE-EXTRACTION-PLAN.md:7). Pulls BlockStreamKit
        // transitively, itself now a URL dep on the public kit repo.
        .package(url: "https://github.com/xocialize/wan-core-mlx-swift", from: "0.2.0"),
    ],
    targets: [
        .target(
            name: "BerniniR",
            dependencies: [
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources/BerniniR"
        ),
        .target(
            name: "MLXBerniniR",
            dependencies: [
                "BerniniR",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXBerniniR"
        ),
        .executableTarget(
            name: "RunBernini",
            dependencies: [
                "BerniniR",
                "MLXBerniniR",  // --v4-package: the engine-seam smoke drives BerniniRPackage.run()
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/RunBernini"
        ),
        .testTarget(
            name: "MLXBerniniRTests",
            dependencies: [
                "MLXBerniniR",
                .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
            ],
            path: "Tests/MLXBerniniRTests"
        ),
        .testTarget(
            name: "BerniniRTests",
            dependencies: [
                "BerniniR",
                .product(name: "WanCore", package: "wan-core-mlx-swift"),
            ],
            path: "Tests/BerniniRTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
