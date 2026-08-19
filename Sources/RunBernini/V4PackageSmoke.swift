// V4 package smoke (RunBernini --v4-package): the ENGINE seam for Bernini-v2 —
// a real `BerniniRPackage` with a `.v2` configuration against a local checkpoint,
// driven through the canonical `run()` dispatch. What this proves beyond V3:
// `hasPlannerPlane` routes t2i/t2v onto `plannedGenerate` inside the package,
// the CAN hooks thread through, and canonical artifacts (PNG `Image`, MP4
// `Video`) come back out.
//
//   swift run -c release RunBernini --v4-package \
//       --model-dir /Volumes/DEV_ARCHIVE/bernini-v2/ckpt-bf16 \
//       [--width 480] [--height 320] [--frames 17] [--steps 16] [--seed 42] \
//       [--t2i-only | --t2v-only] [--out /tmp/bernini-v4]
//
// Defaults are the V3-calibrated smoke config (480x320, 17f, 16 steps). Prewarm
// the checkpoint from DEV_ARCHIVE first (cold-load Metal watchdog).

import Foundation
import MLX
import MLXBerniniR
import MLXToolKit

func runV4PackageSmoke(modelDir: URL) async throws {
    let width = argValue("--width").flatMap(Int.init) ?? 480
    let height = argValue("--height").flatMap(Int.init) ?? 320
    let frames = argValue("--frames").flatMap(Int.init) ?? 17
    let steps = argValue("--steps").flatMap(Int.init) ?? 16
    let smokeSeed = argValue("--seed").flatMap(UInt64.init) ?? 42
    let outDir = URL(filePath: argValue("--out") ?? "/tmp/bernini-v4")
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
    let t2iOnly = CommandLine.arguments.contains("--t2i-only")
    let t2vOnly = CommandLine.arguments.contains("--t2v-only")

    var config = BerniniRConfiguration.v2
    config.modelDirectory = modelDir
    let package = BerniniRPackage(configuration: config)

    print("[v4] load() — resident renderer + VAE from \(modelDir.path)")
    let t0 = Date()
    try await package.load()
    print(String(format: "  load: %.1fs", -t0.timeIntervalSinceNow))

    if !t2vOnly {
        print("[v4] run(T2IRequest) — planned dispatch, \(width)x\(height), \(steps) steps")
        let t = Date()
        let response = try await package.run(
            T2IRequest(
                prompt: "A golden retriever puppy sitting in a sunlit meadow, "
                    + "photorealistic, shallow depth of field",
                width: width, height: height, steps: steps, seed: smokeSeed))
        guard let t2i = response as? T2IResponse else {
            fatalError("[v4] t2i returned \(type(of: response))")
        }
        let png = outDir.appending(path: "v4_t2i.png")
        try t2i.image.data.write(to: png)
        print(String(
            format: "  t2i: %.1fs, %@ %dx%d, %d bytes -> %@, peak %.1f GB",
            -t.timeIntervalSinceNow, t2i.image.format.rawValue,
            t2i.image.width ?? -1, t2i.image.height ?? -1, t2i.image.data.count,
            png.path, Double(GPU.peakMemory) / 1e9))
    }

    if !t2iOnly {
        print("[v4] run(T2VRequest) — planned dispatch, \(width)x\(height)x\(frames)f, \(steps) steps")
        let t = Date()
        let response = try await package.run(
            T2VRequest(
                prompt: "A golden retriever puppy running through a sunlit meadow, "
                    + "slow motion, photorealistic",
                numFrames: frames, width: width, height: height,
                steps: steps, seed: smokeSeed))
        guard let t2v = response as? T2VResponse else {
            fatalError("[v4] t2v returned \(type(of: response))")
        }
        let mp4 = outDir.appending(path: "v4_t2v.mp4")
        try t2v.video.data.write(to: mp4)
        print(String(
            format: "  t2v: %.1fs, %@ %d bytes (%.2fs @ %.0f fps) -> %@, peak %.1f GB",
            -t.timeIntervalSinceNow, t2v.video.format.rawValue, t2v.video.data.count,
            t2v.video.durationSeconds ?? 0, t2v.video.frameRate ?? 0,
            mp4.path, Double(GPU.peakMemory) / 1e9))
    }

    await package.unload()
    print(String(format: "[v4] DONE total %.1fs, peak %.1f GB",
                 -t0.timeIntervalSinceNow, Double(GPU.peakMemory) / 1e9))
}
