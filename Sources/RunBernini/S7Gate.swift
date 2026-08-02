// S7 gate as a CLI (RunBernini --s7-gate): the HV2 block-streaming CONSUMER wiring.
//
// wan-core's own gates already prove the streaming machinery (BlockStreamerTests on a
// synthetic model; RunWanStream → probes/hv2_wan_blockstreamer.out on real A14B at scale).
// What neither covers is the seam this repo owns: that
// `BerniniRendererModel.fromPretrained(modelDir:streaming:)` resolves the right granule
// directories, quantizes before binding, loads the non-block globals, and hands back a
// renderer whose forward is INDISTINGUISHABLE from the resident load.
//
//   swift run -c release RunBernini --s7-gate
//     [--model-dir …/ckpt-int4] [--granules-root /Volumes/Satechi/Models/wan-granules]
//
// Arms:
//   W1 streamed load — construct through the consumer entry point; report slots/sweep.
//   W2 bit-exactness — streamed forward ≡ resident forward, memcmp, same fixture as S1/S6.
//   W3 stale-granule guard — the int4 checkpoint against the bf16 granule tree must throw
//      BEFORE binding (a derived artifact pointed at the wrong checkpoint).
//   W4 editing guard — the multiseg surfaces must refuse while streamed (they bypass the
//      streamer's group window and would read unrefilled slots).
//
// Policy is `.forceStream` throughout: the fixture is 16 tokens, so the `.auto` runtime gate
// would correctly fall back to resident on the first forward and the streamed path would
// never be exercised. Scale behavior is the receipts' job, not this gate's.

import Foundation
import WanCore
import MLX
import MLXNN

import BerniniR

private func say(_ msg: String) {
    print("[s7] " + msg)
    fflush(stdout)
}

private func bitIdentical(_ a: MLXArray, _ b: MLXArray) -> Bool {
    eval(a, b)
    guard a.nbytes == b.nbytes, a.dtype == b.dtype else { return false }
    let da = a.asData(access: .noCopyIfContiguous)
    let db = b.asData(access: .noCopyIfContiguous)
    return da.data.withUnsafeBytes { pa in
        db.data.withUnsafeBytes { pb in
            memcmp(pa.baseAddress!, pb.baseAddress!, a.nbytes) == 0
        }
    }
}

/// One high-noise-expert forward on the S1 DiT fixture (seqLen 16, B=1).
private func ditForward(_ renderer: BerniniRendererModel, x: MLXArray, ctxRaw: MLXArray)
    -> MLXArray
{
    let model = renderer.highNoiseExpert
    let embedded = model.embedText([ctxRaw])
    let kv = model.prepareCrossKV(embedded)
    let out = model(
        [x], t: MLXArray([Float(999)]), context: .embedded(embedded), seqLen: 16,
        crossKVCaches: kv)[0]
    eval(out)
    return out
}

func runS7Gate(modelDir: URL, granulesRoot: URL) throws {
    let fixtures = URL(filePath: FileManager.default.currentDirectoryPath)
        .appending(path: "Tests/BerniniRTests/Fixtures/parity")
    let x = try loadNumpy(url: fixtures.appending(path: "dit_x.npy"))
    let ctxRaw = try loadNumpy(url: fixtures.appending(path: "dit_ctx_raw.npy"))

    let config = try WanConfig.load(from: modelDir.appending(path: "config.json"))
    let variant = config.quantization != nil ? "int4" : "bf16"
    let granuleRoot = granulesRoot.appending(path: variant)
    say("checkpoint \(modelDir.lastPathComponent) (\(variant), dualModel=\(config.dualModel))")
    say("granules   \(granuleRoot.path)")

    var failures: [String] = []

    // ---- W1 · streamed load through the consumer entry point ----
    let options = BlockStreamingOptions(gatePolicy: .forceStream, quiet: true)
    let t0 = Date()
    let streamed = try BerniniRendererModel.fromPretrained(
        modelDir: modelDir,
        streaming: BerniniStreamingConfiguration(granuleRoot: granuleRoot, options: options))
    let loadSeconds = -t0.timeIntervalSinceNow
    guard let streamer = streamed.blockStreamer else {
        say("W1 ❌ renderer loaded without a streamer")
        throw BerniniStreamingError.unsupportedSurface("streamed load produced no streamer")
    }
    say(String(
        format: "W1 ✅ streamed load %.1fs · %d blocks · group %d · slots %.2f GiB · sweep %.2f GiB",
        loadSeconds, streamer.blockCount, streamer.groupSize,
        Double(streamer.slotResidentBytes) / 1_073_741_824,
        Double(streamer.sweepBytes) / 1_073_741_824))
    say("   isStreaming=\(streamed.isStreaming) verdict=\(streamer.verdict.rawValue)")

    let streamedOut = ditForward(streamed, x: x, ctxRaw: ctxRaw)

    // ---- W4 · editing guard (while the streamer is still attached) ----
    do {
        try streamed.requireResidentBlocks("videoEdit (rv2v)")
        failures.append("W4: editing surfaces did NOT refuse while streaming")
        say("W4 ❌ editing surface allowed while streamed")
    } catch let error as BerniniStreamingError {
        say("W4 ✅ editing refused while streamed: \(error.localizedDescription)")
    }

    // ---- W3 · stale/mismatched granule guard ----
    // The sibling quant's tree: same block layout, same geometry, DIFFERENT weights — the
    // case `bind`'s shape check cannot catch for bf16-vs-bf16, so the manifest provenance
    // check has to. Skipped when the sibling tree isn't laid out on this machine.
    let sibling = granulesRoot.appending(path: variant == "int4" ? "bf16" : "int4")
    if FileManager.default.fileExists(
        atPath: sibling.appending(path: "high").appending(path: "manifest.json").path)
    {
        do {
            _ = try BerniniRendererModel.fromPretrained(
                modelDir: modelDir,
                streaming: BerniniStreamingConfiguration(granuleRoot: sibling, options: options))
            failures.append("W3: mismatched granule tree was accepted")
            say("W3 ❌ granules from the sibling quant were accepted")
        } catch let error as BerniniStreamingError {
            say("W3 ✅ mismatched granules rejected: \(error.localizedDescription)")
        }
    } else {
        say("W3 ⚠️ skipped — no sibling granule tree at \(sibling.path)")
    }

    // Release the streamed renderer BEFORE loading the resident one: its slots and globals
    // stay live otherwise, and the resident load is ~17 GB on its own (the phys-ratchet
    // lesson from the HV2 receipt harness, consumer edition).
    streamer.detach()
    MLX.Memory.clearCache()

    // ---- W2 · streamed ≡ resident, through the same public entry point ----
    let resident = try BerniniRendererModel.fromPretrained(modelDir: modelDir)
    let residentOut = ditForward(resident, x: x, ctxRaw: ctxRaw)
    let exact = bitIdentical(streamedOut, residentOut)
    if exact {
        say("W2 ✅ streamed forward ≡ resident forward (memcmp-identical)")
    } else {
        let diff = (streamedOut.asType(.float32) - residentOut.asType(.float32))
            .abs().max().item(Float.self)
        failures.append("W2: streamed forward diverged (max abs \(diff))")
        say("W2 ❌ streamed forward diverged — max abs \(diff)")
    }

    guard failures.isEmpty else {
        say("FAILED: " + failures.joined(separator: " · "))
        exit(1)
    }
    say("== S7 PASS · consumer streaming wiring is output-identical to the resident load ==")
}
