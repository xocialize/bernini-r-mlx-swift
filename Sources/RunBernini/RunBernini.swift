// RunBernini — GPU smoke CLI for the S2b gate: one real prompt-level t2i/t2v
// generation, PNG frame dump, timing + peak-memory record.
//
//   swift run -c release RunBernini "a red fox in snow" \
//     [--frames 1] [--width 832] [--height 480] [--steps 40] [--seed 42] \
//     [--model-dir /Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16] \
//     [--out /tmp/bernini]
//
// If the SPM-CLI metallib boundary bites (MLX error: failed to load the
// default metallib), run it under Xcode / xcodebuild instead — the workspace
// convention for live GPU inference.

import CoreGraphics
import Foundation
import ImageIO
import MLX
import UniformTypeIdentifiers

import BerniniR

func argValue(_ name: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let positional = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") }
let flagValues = Set(
    CommandLine.arguments.enumerated().compactMap {
        (i, a) in a.hasPrefix("--") && i + 1 < CommandLine.arguments.count
            ? CommandLine.arguments[i + 1] : nil
    })
let prompt =
    positional.first { !flagValues.contains($0) }
    ?? "A red fox standing in fresh snow, golden hour, photorealistic"
let numFrames = argValue("--frames").flatMap(Int.init) ?? 1
let width = argValue("--width").flatMap(Int.init) ?? 832
let height = argValue("--height").flatMap(Int.init) ?? 480
let steps = argValue("--steps").flatMap(Int.init) ?? 40
let seed = argValue("--seed").flatMap(UInt64.init) ?? 42
let solver = argValue("--solver").flatMap(SchedulerKind.init(rawValue:)) ?? .unipc
let lightning = CommandLine.arguments.contains("--lightning")
let r2vRef = argValue("--r2v")  // reference image path → r2v (subject-consistent t2v)
let modelDir = URL(
    filePath: argValue("--model-dir")
        ?? "/Volumes/DEV_ARCHIVE/weights/bernini-r-mlx-weights/ckpt-bf16")
let outDir = URL(filePath: argValue("--out") ?? "/tmp/bernini")

func writePNG(_ frame: MLXArray, to url: URL) throws {
    // frame: [3, H, W] in [-1, 1]
    let h = frame.dim(1)
    let w = frame.dim(2)
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255)
        .asType(.uint8)
        .transposed(1, 2, 0)  // [H, W, 3]
    eval(rgb)
    let bytes: [UInt8] = rgb.asArray(UInt8.self)
    let data = CFDataCreate(nil, bytes, bytes.count)!
    let provider = CGDataProvider(data: data)!
    let image = CGImage(
        width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24,
        bytesPerRow: w * 3, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
        provider: provider, decode: nil, shouldInterpolate: false,
        intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(
        url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, image, nil)
    CGImageDestinationFinalize(dest)
}

/// Decode a reference image file → pixels [1,3,1,H,W] in [-1,1], top-down
/// (mirrors the wrapper's FrameDecode; CLI-side for the --r2v gate).
func decodeRefImage(_ path: String, width: Int, height: Int) throws -> MLXArray {
    let data = try Data(contentsOf: URL(filePath: path))
    guard let src = CGImageSourceCreateWithData(data as CFData, nil),
          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        throw NSError(domain: "RunBernini", code: 1)
    }
    var rgba = [UInt8](repeating: 0, count: width * height * 4)
    let ctx = CGContext(
        data: &rgba, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.interpolationQuality = .high
    ctx.translateBy(x: 0, y: CGFloat(height)); ctx.scaleBy(x: 1, y: -1)
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
    let plane = height * width
    var chw = [Float](repeating: 0, count: 3 * plane)
    for y in 0..<height {
        for x in 0..<width {
            let p = (y * width + x) * 4, i = y * width + x
            chw[i] = Float(rgba[p]) / 255 * 2 - 1
            chw[plane + i] = Float(rgba[p + 1]) / 255 * 2 - 1
            chw[2 * plane + i] = Float(rgba[p + 2]) / 255 * 2 - 1
        }
    }
    return MLXArray(chw, [1, 3, 1, height, width])
}

@main
struct RunBernini {
    static func main() async throws {
        if CommandLine.arguments.contains("--s4-gate") {
            try runS4Gate(modelDir: modelDir)
            return
        }
        if CommandLine.arguments.contains("--s5-gate") {
            try runS5Gate(modelDir: modelDir)
            return
        }
        if CommandLine.arguments.contains("--s6-gate") {
            try runS6Gate(
                bf16Dir: modelDir,
                int4Dir: modelDir.deletingLastPathComponent()
                    .appending(path: "ckpt-int4"))
            return
        }
        try FileManager.default.createDirectory(
            at: outDir, withIntermediateDirectories: true)

        print("Loading pipeline from \(modelDir.path) …")
        let tLoad = Date()
        let pipeline = try await BerniniPipeline.fromPretrained(modelDir: modelDir)
        print(String(format: "  load: %.1fs", -tLoad.timeIntervalSinceNow))

        print(
            "Generating \(numFrames) frame(s) @ \(width)x\(height), "
                + (lightning ? "4 steps, euler (lightning)" : "\(steps) steps, \(solver.rawValue)")
                + ", seed \(seed)")
        print("  prompt: \(prompt)")
        let tGen = Date()
        var lastStepEnd = Date()
        let progress: (Int, Int, MLXArray) -> Void = { step, total, _ in
            let dt = -lastStepEnd.timeIntervalSinceNow
            lastStepEnd = Date()
            print(
                String(
                    format: "  step %d/%d  %.1fs  active %.1f GB  peak %.1f GB",
                    step + 1, total, dt, Double(GPU.activeMemory) / 1e9,
                    Double(GPU.peakMemory) / 1e9))
        }
        let frames: MLXArray
        if let r2vRef {
            print("  r2v reference: \(r2vRef)")
            let refPixels = try decodeRefImage(r2vRef, width: width, height: height)
            frames = try pipeline.r2v(
                prompt: prompt, referencePixels: [refPixels],
                width: width, height: height, numFrames: numFrames,
                steps: steps, seed: seed, onStep: progress)
        } else {
            frames = try pipeline.t2v(
                prompt: prompt, width: width, height: height, numFrames: numFrames,
                steps: lightning ? nil : steps,
                scheduler: lightning ? nil : solver, lightning: lightning, seed: seed,
                onStep: progress)
        }
        print(String(format: "  generate: %.1fs total", -tGen.timeIntervalSinceNow))

        let t = frames.dim(2)
        for i in 0..<t {
            let url = outDir.appending(path: String(format: "frame_%03d.png", i))
            try writePNG(frames[0, 0..., i, 0..., 0...], to: url)
        }
        print("Wrote \(t) frame(s) -> \(outDir.path)")
        print(String(format: "Peak GPU memory: %.1f GB", Double(GPU.peakMemory) / 1e9))
    }
}
