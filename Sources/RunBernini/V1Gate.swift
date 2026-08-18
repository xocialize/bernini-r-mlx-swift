// V1 gate (RunBernini --v1-gate): planner-plane parity vs the CPU-fp32 torch
// oracle fixtures (PORTING-SPEC-V2.md, gate V1). Stages:
//
//   V1a  backbone      01_embeds → penultimate hidden vs 02_call00/01/02
//   V1b  DiffLossFM    05_call00 z + injected noise → sample vs fm_out
//   V1c  full MaskGIT  streams + injected order/noises → 07_* contexts + pred_vit
//
// Fixture dir: --fixtures <dir> or BERNINI_V2_FIXTURES; defaults to the small
// t2v pack. CPU stream, fp32 — the canonical parity substrate. The 4D masks are
// consumed from fixtures here (the mask BUILDER gets its own gate with the
// processor port in phase #6, which dumps token_type/segment_ids).
//
//   swift run RunBernini --v1-gate --model-dir /Volumes/DEV_ARCHIVE/bernini-v2/ckpt-bf16

import Foundation
import MLX
import WanCore

import BerniniR

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// WanCore's loadNumpy is '<f4'-only; the planner fixtures also carry int64
/// (`position_ids`, `reveal_order`) and bool (`visual_output_token_mask`).
/// Minimal local reader for those two descrs → host Ints.
private func loadNumpyInts(url: URL) throws -> [Int] {
    let data = try Data(contentsOf: url)
    guard data.count > 10, data.prefix(6) == Data([0x93] + Array("NUMPY".utf8)) else {
        throw NSError(domain: "V1Gate", code: 1,
                      userInfo: [NSLocalizedDescriptionKey: "not a .npy: \(url.path)"])
    }
    let headerLen = Int(data[8]) | (Int(data[9]) << 8)
    let header = String(decoding: data[10 ..< 10 + headerLen], as: UTF8.self)
    let payload = data.dropFirst(10 + headerLen)
    if header.contains("'<i8'") {
        return payload.withUnsafeBytes { raw in
            raw.bindMemory(to: Int64.self).map(Int.init)
        }
    } else if header.contains("'|b1'") {
        return payload.map(Int.init)
    } else {
        throw NSError(domain: "V1Gate", code: 2,
                      userInfo: [NSLocalizedDescriptionKey:
                        "unsupported fixture descr in \(url.lastPathComponent): \(header)"])
    }
}

private struct V1Fixtures {
    let dir: URL
    func load(_ name: String) throws -> MLXArray {
        try loadNumpy(url: dir.appending(path: "\(name).npy"))
    }
    func meta() throws -> [String: Any] {
        let data = try Data(contentsOf: dir.appending(path: "meta.json"))
        return try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
    }
}

private func loadStream(_ f: V1Fixtures, _ name: String) throws -> PlannerStream {
    let embeds = try f.load("01_\(name)_embeds_postmask")            // [1, L, C] f32
    let mask = try f.load("00_\(name)_attention_mask_4d")            // [1, L, L] f32
    let posHost = try loadNumpyInts(
        url: f.dir.appending(path: "00_\(name)_position_ids.npy"))   // [3*L] i64
    let l = posHost.count / 3
    let positionIds = MLXArray(posHost.map(Int32.init), [3, 1, l])   // [3, 1, L]
    let voutHost = try loadNumpyInts(
        url: f.dir.appending(path: "00_\(name)_visual_output_token_mask.npy"))
    let voutIdx: [Int32] = voutHost.enumerated().compactMap {
        $0.element != 0 ? Int32($0.offset) : nil
    }
    return PlannerStream(
        inputsEmbeds: embeds, mask: mask, positionIds: positionIds,
        visualOutputIndices: voutIdx)
}

func runV1Gate(modelDir: URL) throws {
    let fixturesPath = argValue("--fixtures")
        ?? ProcessInfo.processInfo.environment["BERNINI_V2_FIXTURES"]
        ?? "/Volumes/DEV_ARCHIVE/bernini-v2/measure/goldens-t2v-small"
    let f = V1Fixtures(dir: URL(filePath: fixturesPath))
    let meta = try f.meta()
    let planningSteps = meta["planning_steps"] as? Int ?? 2
    let vitSteps = meta["vit_steps"] as? Int ?? 3

    try Device.withDefaultDevice(.cpu) {
        print("[v1-gate] loading planner fp32 (CPU stream)… fixtures=\(fixturesPath)")
        let planner = try BerniniPlanner.fromPretrained(modelDir: modelDir, dtype: .float32)

        var cond = try loadStream(f, "cond")
        var uncond = try loadStream(f, "uncond")
        var imgcond = try loadStream(f, "imgcond")

        // ---- V1a: tap-depth sweep on the cond stream, then all streams ----
        print("[v1-gate] V1a backbone tap-depth sweep (cond stream)…")
        let ref0 = try f.load("02_call00_hidden_penult")
        var bestTap = planner.backbone.config.numHiddenLayers - 1
        var bestDiff: Float = .greatestFiniteMagnitude
        for tap in [planner.backbone.config.numHiddenLayers - 2,
                    planner.backbone.config.numHiddenLayers - 1,
                    planner.backbone.config.numHiddenLayers] {
            let h = planner.backbone.penultimateHidden(
                inputsEmbeds: cond.inputsEmbeds, mask: cond.mask,
                positionIds: cond.positionIds, layersToRun: tap)
            let d = maxAbsDiff(h, ref0)
            print("  layersToRun=\(tap): max_abs = \(d)")
            if d < bestDiff { bestDiff = d; bestTap = tap }
        }
        print("  selected tap = \(bestTap)")
        var v1aWorst: Float = 0
        for (stream, callIdx, label) in [(uncond, 1, "uncond"), (imgcond, 2, "imgcond")] {
            let h = planner.backbone.penultimateHidden(
                inputsEmbeds: stream.inputsEmbeds, mask: stream.mask,
                positionIds: stream.positionIds, layersToRun: bestTap)
            let ref = try f.load(String(format: "02_call%02d_hidden_penult", callIdx))
            let d = maxAbsDiff(h, ref)
            v1aWorst = max(v1aWorst, d)
            print("  \(label): max_abs = \(d)")
        }
        v1aWorst = max(v1aWorst, bestDiff)
        let v1aGate: Float = 5e-3  // fp32 CPU, deep-stack budget
        print("  V1a \(v1aWorst <= v1aGate ? "PASS" : "FAIL") (gate \(v1aGate))")

        // ---- V1b: DiffLossFM sample on the step-0 fixture ----
        print("[v1-gate] V1b DiffLossFM sample…")
        let z0 = try f.load("05_call00_fm_z")
        let noise0 = try f.load("05_call00_fm_noise0")
        let sigmaFixture = f.dir.appending(path: "09_fm_sigmas.npy")
        let fmSigmas: MLXArray? = FileManager.default.fileExists(atPath: sigmaFixture.path)
            ? try f.load("09_fm_sigmas") : nil
        if fmSigmas == nil { print("  (no 09_fm_sigmas fixture — derived grid)") }
        let fmTimesteps: MLXArray? = FileManager.default.fileExists(
            atPath: f.dir.appending(path: "09_fm_timesteps.npy").path)
            ? try f.load("09_fm_timesteps") : nil
        // isolation probes: net math alone, then CFG combine, then the full sample
        if FileManager.default.fileExists(
            atPath: f.dir.appending(path: "05_probe_fwd_step0.npy").path),
            let sig = fmSigmas
        {
            let x3 = concatenated([noise0, noise0, noise0], axis: 0)
            let t0 = sig[0 ..< 1] * 1000
            let fwd = planner.diffLoss.netForward(x3, t: t0, c: z0)
            print("  probe net.forward: max_abs = \(maxAbsDiff(fwd, try f.load("05_probe_fwd_step0")))")
            let cfg = planner.diffLoss.netForwardWithTxtImgCfg(
                x3, t: t0, c: z0, txtCfg: 1.4, imgCfg: 1.2)
            print("  probe cfg combine: max_abs = \(maxAbsDiff(cfg, try f.load("05_probe_cfg_step0")))")
        }
        let out0 = planner.diffLoss.sample(
            z: z0, txtCfg: 1.4, imgCfg: 1.2, numInferenceSteps: vitSteps,
            injectedNoise: noise0, injectedSigmas: fmSigmas,
            injectedTimesteps: fmTimesteps)
        let dB = maxAbsDiff(out0, try f.load("05_call00_fm_out"))
        let v1bGate: Float = 1e-3
        print("  fm_out max_abs = \(dB)  \(dB <= v1bGate ? "PASS" : "FAIL") (gate \(v1bGate))")

        // ---- V1c: the full MaskGIT plan vs the final contexts ----
        print("[v1-gate] V1c full plan (\(planningSteps) steps × 3 forwards + 2 final)…")
        let order: [Int] = try loadNumpyInts(
            url: f.dir.appending(path: "04_reveal_order.npy"))
        var noises: [MLXArray] = []
        var i = 0
        while FileManager.default.fileExists(
            atPath: f.dir.appending(
                path: String(format: "05_call%02d_fm_noise0.npy", i)).path)
        {
            noises.append(try f.load(String(format: "05_call%02d_fm_noise0", i)))
            i += 1
        }
        let contexts = planner.plan(
            cond: &cond, uncond: &uncond, imgcond: &imgcond,
            planningSteps: planningSteps, vitDenoisingSteps: vitSteps,
            revealOrder: order, fmNoises: noises,
            tapLayers: bestTap, fmSigmas: fmSigmas, fmTimesteps: fmTimesteps)

        let checks: [(String, MLXArray)] = [
            ("07_cond_embeds_wtxt_wvit", contexts.condEmbedsWtxtWvit),
            ("07_cond_embeds_wtxt_wovit", contexts.condEmbedsWtxtWovit),
            ("07_cond_embeds_wotxt_wvit", contexts.condEmbedsWotxtWvit),
            ("07_cond_embeds_wotxt_wovit", contexts.condEmbedsWotxtWovit),
            ("07_pred_vit_embed", contexts.predVitEmbed),
        ]
        var v1cWorst: Float = 0
        for (name, ours) in checks {
            let d = maxAbsDiff(ours, try f.load(name))
            v1cWorst = max(v1cWorst, d)
            print("  \(name): max_abs = \(d)")
        }
        let v1cGate: Float = 2e-2  // two chained 27-layer stacks + FM head, fp32 CPU
        print("  V1c \(v1cWorst <= v1cGate ? "PASS" : "FAIL") (gate \(v1cGate))")

        let pass = v1aWorst <= v1aGate && dB <= v1bGate && v1cWorst <= v1cGate
        print(pass ? "[v1-gate] ALL PASS" : "[v1-gate] FAIL")
        if !pass { exit(1) }
    }
}
