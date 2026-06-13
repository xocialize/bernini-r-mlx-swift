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
        // MLXEngine contract (MLXToolKit) for the wrapper target. Local-path dep like the
        // other model wrappers; the core `BerniniR` target stays engine-agnostic.
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.4.0"),
    ],
    targets: [
        .target(
            name: "BerniniR",
            dependencies: [
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
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
            ],
            path: "Sources/MLXBerniniR"
        ),
        .executableTarget(
            name: "RunBernini",
            dependencies: ["BerniniR"],
            path: "Sources/RunBernini"
        ),
        .testTarget(
            name: "MLXBerniniRTests",
            dependencies: ["MLXBerniniR"],
            path: "Tests/MLXBerniniRTests"
        ),
        .testTarget(
            name: "BerniniRTests",
            dependencies: ["BerniniR"],
            path: "Tests/BerniniRTests",
            resources: [.copy("Fixtures")]
        ),
    ]
)
