// Canonical-artifact encoders for the BerniniR wrapper: decoded frames
// ([1, 3, T, H, W] in [-1, 1]) → PNG `Image` (frame 0, for textToImage) or
// H.264 MP4 `Video` (for textToVideo). Pure AVFoundation/CoreGraphics —
// no MLX beyond reading the frame tensor out.

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import MLX
import MLXToolKit
import UniformTypeIdentifiers

enum FrameEncodeError: Error {
    case pixelBufferAllocation
    case writerSetup(String)
    case pngEncode
}

/// Frame tensor [3, H, W] in [-1, 1] → interleaved RGB bytes [H, W, 3].
private func rgbBytes(_ frame: MLXArray) -> (bytes: [UInt8], width: Int, height: Int) {
    let h = frame.dim(1)
    let w = frame.dim(2)
    let scaled = (frame.asType(.float32) + 1) * Float(127.5)
    let rgb = clip(scaled, min: 0, max: 255).asType(.uint8).transposed(1, 2, 0)
    eval(rgb)
    return (rgb.asArray(UInt8.self), w, h)
}

/// Encode one frame [3, H, W] as PNG.
func encodePNG(frame: MLXArray) throws -> (data: Data, width: Int, height: Int) {
    let (bytes, w, h) = rgbBytes(frame)
    let cfData = CFDataCreate(nil, bytes, bytes.count)!
    guard
        let provider = CGDataProvider(data: cfData),
        let image = CGImage(
            width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24,
            bytesPerRow: w * 3, space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false,
            intent: .defaultIntent)
    else { throw FrameEncodeError.pngEncode }
    let out = NSMutableData()
    guard
        let dest = CGImageDestinationCreateWithData(
            out, UTType.png.identifier as CFString, 1, nil)
    else { throw FrameEncodeError.pngEncode }
    CGImageDestinationAddImage(dest, image, nil)
    guard CGImageDestinationFinalize(dest) else { throw FrameEncodeError.pngEncode }
    return (out as Data, w, h)
}

/// Fill a BGRA CVPixelBuffer from interleaved RGB bytes.
private func pixelBuffer(
    rgb: [UInt8], width: Int, height: Int, pool: CVPixelBufferPool
) throws -> CVPixelBuffer {
    var bufferOut: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &bufferOut)
    guard let buffer = bufferOut else { throw FrameEncodeError.pixelBufferAllocation }
    CVPixelBufferLockBaseAddress(buffer, [])
    defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
    let base = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
    let stride = CVPixelBufferGetBytesPerRow(buffer)
    for y in 0..<height {
        for x in 0..<width {
            let src = (y * width + x) * 3
            let dst = y * stride + x * 4
            base[dst + 0] = rgb[src + 2]  // B
            base[dst + 1] = rgb[src + 1]  // G
            base[dst + 2] = rgb[src + 0]  // R
            base[dst + 3] = 255  // A
        }
    }
    return buffer
}

/// Encode frames [1, 3, T, H, W] in [-1, 1] as an H.264 MP4 at `fps`.
/// Writes to a temp file (AVAssetWriter is file-based) and returns the bytes.
/// `@InferenceActor` so the non-`Sendable` frame tensor never crosses an
/// isolation boundary (Swift 6 region isolation).
@InferenceActor
func encodeMP4(frames: MLXArray, fps: Double) async throws -> Data {
    let t = frames.dim(2)
    let h = frames.dim(3)
    let w = frames.dim(4)

    let url = FileManager.default.temporaryDirectory
        .appending(path: "bernini-\(UUID().uuidString).mp4")
    defer { try? FileManager.default.removeItem(at: url) }

    let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
    let input = AVAssetWriterInput(
        mediaType: .video,
        outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: w,
            AVVideoHeightKey: h,
        ])
    input.expectsMediaDataInRealTime = false
    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: input,
        sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: w,
            kCVPixelBufferHeightKey as String: h,
        ])
    guard writer.canAdd(input) else {
        throw FrameEncodeError.writerSetup("cannot add video input")
    }
    writer.add(input)
    guard writer.startWriting() else {
        throw FrameEncodeError.writerSetup(writer.error?.localizedDescription ?? "startWriting")
    }
    writer.startSession(atSourceTime: .zero)

    let timescale = CMTimeScale(600)
    let frameDuration = CMTime(
        value: CMTimeValue((600.0 / fps).rounded()), timescale: timescale)

    for i in 0..<t {
        let (bytes, fw, fh) = rgbBytes(frames[0, 0..., i, 0..., 0...])
        guard let pool = adaptor.pixelBufferPool else {
            throw FrameEncodeError.writerSetup("no pixel buffer pool")
        }
        let buffer = try pixelBuffer(rgb: bytes, width: fw, height: fh, pool: pool)
        while !input.isReadyForMoreMediaData {
            try await Task.sleep(for: .milliseconds(5))
        }
        adaptor.append(buffer, withPresentationTime: CMTimeMultiply(frameDuration, multiplier: Int32(i)))
    }

    input.markAsFinished()
    await writer.finishWriting()
    if let error = writer.error {
        throw FrameEncodeError.writerSetup(error.localizedDescription)
    }
    return try Data(contentsOf: url)
}
