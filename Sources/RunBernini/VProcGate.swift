// VProc gate (RunBernini --vproc-gate): processor-plane parity vs the torch
// oracle fixtures (PROCESSOR-SPEC.md). Rebuilds the three planner streams
// (cond / uncond / imgcond) from meta.json's request with `BerniniProcessor`
// and compares bit-exact per stream:
//
//   input_ids · token_types · token_segment_ids · position_ids ·
//   visual_output indices · the rebuilt 4-D additive mask (max-abs 0)
//
// Tokenizer-only — NO model weights load (BerniniPlanner is never touched);
// host-side [Int] comparisons throughout. The uncond stream uses the
// production `DEFAULT_NEG_PROMPT` (the processor's built-in default).
//
//   swift run RunBernini --vproc-gate [--fixtures <dir>] [--model-dir <ckpt-bf16>]

import Foundation
import MLX
import WanCore

import BerniniR

/// `loadNumpyInts` (V1Gate) covers '<i8'/'|b1'; the processor fixtures add
/// '<i4' (`token_types` is int32). Minimal reader for that one descr.
private func loadNumpyInt32s(url: URL) throws -> [Int] {
    let data = try Data(contentsOf: url)
    guard data.count > 10, data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else {
        throw NSError(domain: "VProcGate", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "not a .npy: \(url.path)"])
    }
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let header = String(decoding: data[10 ..< 10 + headerLen], as: UTF8.self)
    guard header.contains("'<i4'") else {
        throw NSError(domain: "VProcGate", code: 2,
                      userInfo: [NSLocalizedDescriptionKey:
                        "expected '<i4' descr in \(url.lastPathComponent): \(header)"])
    }
    return data.dropFirst(10 + headerLen).withUnsafeBytes { raw in
        raw.bindMemory(to: Int32.self).map(Int.init)
    }
}

/// Exact-match check with first-divergence context on failure.
private func checkInts(_ label: String, ours: [Int], expected: [Int]) -> Bool {
    if ours == expected {
        print("  \(label): PASS (\(ours.count))")
        return true
    }
    var i = 0
    let n = min(ours.count, expected.count)
    while i < n, ours[i] == expected[i] { i += 1 }
    print("  \(label): FAIL — len \(ours.count) vs \(expected.count), first diff at \(i)")
    let lo = max(0, i - 3)
    if lo < n {
        let hiE = min(expected.count, i + 5)
        let hiO = min(ours.count, i + 5)
        print("    expected[\(lo)..<\(hiE)] = \(Array(expected[lo ..< hiE]))")
        print("    actual  [\(lo)..<\(hiO)] = \(Array(ours[lo ..< hiO]))")
    }
    return false
}

/// Mask check: exact float equality per cell (0.0 / −inf only; −inf == −inf).
private func checkMask(_ label: String, ours: MLXArray, expectedURL: URL) throws -> Bool {
    let expected = try loadNumpy(url: expectedURL)
    guard ours.shape == expected.shape else {
        print("  \(label): FAIL — shape \(ours.shape) vs \(expected.shape)")
        return false
    }
    let a: [Float] = ours.asArray(Float.self)
    let b: [Float] = expected.asArray(Float.self)
    var mismatches = 0
    var firstDiff = -1
    for j in 0 ..< a.count where a[j] != b[j] {
        if firstDiff < 0 { firstDiff = j }
        mismatches += 1
    }
    if mismatches == 0 {
        print("  \(label): PASS (max_abs 0, \(a.count) cells)")
        return true
    }
    print("  \(label): FAIL — \(mismatches) cells differ, first at flat \(firstDiff) "
        + "(expected \(b[firstDiff]), actual \(a[firstDiff]))")
    return false
}

func runVProcGate() async throws {
    let fixturesPath = argValue("--fixtures")
        ?? ProcessInfo.processInfo.environment["BERNINI_V2_FIXTURES"]
        ?? "/Volumes/DEV_ARCHIVE/bernini-v2/measure/goldens-t2v-small"
    let dir = URL(filePath: fixturesPath)
    // Tokenizer assets: --model-dir's mllm/ when given, else the archived
    // upstream checkpoint (the shipped Qwen2.5-VL tokenizer files).
    let mllmDir = argValue("--model-dir").map { URL(filePath: $0).appending(path: "mllm") }
        ?? URL(filePath: "/Volumes/DEV_ARCHIVE/bernini-v2/ckpt-bf16/mllm")

    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appending(path: "meta.json"))) as? [String: Any] ?? [:]
    let prompt = meta["prompt"] as? String ?? "a red fox sitting in snow at sunrise"
    let taskName = meta["task"] as? String ?? "t2v"
    guard let task = BerniniPlannerTask(rawValue: taskName) else {
        print("[vproc-gate] unsupported task '\(taskName)' (t2v/t2i only)")
        exit(1)
    }
    let height = meta["height"] as? Int ?? 256
    let width = meta["width"] as? Int ?? 256
    let numFrames = meta["num_frames"] as? Int ?? 17

    print("[vproc-gate] fixtures=\(fixturesPath)")
    print("[vproc-gate] tokenizer=\(mllmDir.path)")
    print("[vproc-gate] \(taskName) \(width)x\(height)x\(numFrames)f  prompt: \(prompt)")

    let processor = try await BerniniProcessor.fromPretrained(mllmDir: mllmDir)
    let outputs = processor.process(
        prompt: prompt, task: task, width: width, height: height, numFrames: numFrames)
    print("[vproc-gate] grid = [\(outputs.gridT), \(outputs.gridH), \(outputs.gridW)]  "
        + "N = \(outputs.vitTokenCount)  L(cond) = \(outputs.cond.length)  "
        + "L(uncond) = \(outputs.uncond.length)")

    var allPass = true
    let streams: [(String, BerniniProcessedStream)] = [
        ("cond", outputs.cond), ("uncond", outputs.uncond), ("imgcond", outputs.imgcond),
    ]
    for (name, stream) in streams {
        print("[vproc-gate] stream \(name):")
        func fixture(_ item: String) -> URL {
            dir.appending(path: "00_\(name)_\(item).npy")
        }
        allPass = checkInts(
            "input_ids", ours: stream.inputIds,
            expected: try loadNumpyInts(url: fixture("input_ids"))) && allPass
        // token_types / token_segment_ids exist only in packs regenerated with
        // the LOCAL PATCH dump (`bernini_template.py:372-375`) — the small pack
        // has them; older packs (goldens-t2v-prod) don't. Skip, don't crash.
        if FileManager.default.fileExists(atPath: fixture("token_types").path) {
            allPass = checkInts(
                "token_types", ours: stream.tokenTypes.map(Int.init),
                expected: try loadNumpyInt32s(url: fixture("token_types"))) && allPass
            allPass = checkInts(
                "token_segment_ids", ours: stream.tokenSegmentIds.map(Int.init),
                expected: try loadNumpyInts(url: fixture("token_segment_ids"))) && allPass
        } else {
            print("  token_types/token_segment_ids: SKIP (fixture absent in this pack)")
        }
        allPass = checkInts(
            "position_ids", ours: stream.positionIds,
            expected: try loadNumpyInts(url: fixture("position_ids"))) && allPass
        let voutExpected: [Int] = try loadNumpyInts(
            url: fixture("visual_output_token_mask")
        ).enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        allPass = checkInts(
            "visual_output_indices", ours: stream.visualOutputIndices.map(Int.init),
            expected: voutExpected) && allPass
        allPass = try checkMask(
            "attention_mask_4d", ours: stream.attentionMask4D(),
            expectedURL: fixture("attention_mask_4d")) && allPass
    }

    print(allPass ? "[vproc-gate] ALL PASS" : "[vproc-gate] FAIL")
    if !allPass { exit(1) }
}
