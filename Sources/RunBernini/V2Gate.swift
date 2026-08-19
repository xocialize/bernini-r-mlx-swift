// V2 gate (RunBernini --v2-gate): planner-conditioned renderer parity — the
// wvitcfg sampler (vae_txt_vit_wapg, no-source t2i) vs the torch oracle fixtures
// from measure/gen_wvitcfg_goldens.py. CPU-fp32, high expert only (short run
// stays above the 0.875 boundary).
//
//   swift run RunBernini --v2-gate --model-dir /Volumes/DEV_ARCHIVE/bernini-v2/ckpt-bf16 \
//       [--fixtures /Volumes/DEV_ARCHIVE/bernini-v2/measure/goldens-wvitcfg]

import Foundation
import MLX
import WanCore

import BerniniR

private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    MLX.abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
}

/// `b c (t pt) (h ph) (w pw)` → `b (t h w) (pt ph pw c)` with pt=1, ph=pw=2 —
/// pack our unpacked [C, T, H, W] into the oracle's token layout.
private func packLatent(_ x: MLXArray) -> MLXArray {
    let (c, t, hp, wp) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let (h, w) = (hp / 2, wp / 2)
    return x.reshaped(c, t, h, 2, w, 2)          // [C,T,h,ph,w,pw]
        .transposed(1, 2, 4, 3, 5, 0)            // [T,h,w,ph,pw,C]
        .reshaped(1, t * h * w, 4 * c)
}

/// Inverse: oracle packed [1, L, 4C] → our [C, T, H, W].
private func unpackLatent(_ x: MLXArray, c: Int, t: Int, h: Int, w: Int) -> MLXArray {
    x.reshaped(t, h, w, 2, 2, c)                 // [T,h,w,ph,pw,C]
        .transposed(5, 0, 1, 3, 2, 4)            // [C,T,h,ph,w,pw]
        .reshaped(c, t, h * 2, w * 2)
}

func runV2Gate(modelDir: URL) throws {
    let fixturesPath = argValue("--fixtures")
        ?? "/Volumes/DEV_ARCHIVE/bernini-v2/measure/goldens-wvitcfg"
    let dir = URL(filePath: fixturesPath)
    func fx(_ name: String) throws -> MLXArray {
        try loadNumpy(url: dir.appending(path: "\(name).npy"))
    }
    let meta = try JSONSerialization.jsonObject(
        with: Data(contentsOf: dir.appending(path: "meta_wvitcfg.json"))) as? [String: Any] ?? [:]
    let steps = meta["steps"] as? Int ?? 2
    let height = meta["height"] as? Int ?? 128
    let width = meta["width"] as? Int ?? 128

    try Device.withDefaultDevice(.cpu) {
        print("[v2-gate] loading HIGH expert fp32 (CPU stream)… fixtures=\(fixturesPath)")
        let renderer = try BerniniRendererModel.fromPretrained(modelDir: modelDir)
        let high = renderer.highNoiseExpert

        // embedText consumes [L, D] contexts — squeeze the fixture batch dim.
        let contexts = WvitcfgContexts(
            wtxtWvit: try fx("10_ctx_wtxt_wvit")[0],
            wtxtWovit: try fx("10_ctx_wtxt_wovit")[0],
            wotxtWvit: try fx("10_ctx_wotxt_wvit")[0],
            wotxtWovit: try fx("10_ctx_wotxt_wovit")[0])

        let c = 16
        let (hLat, wLat) = (height / 8, width / 8)
        let packedNoise = try fx("11_init_noise")               // [1, L, 4C]
        let noise = unpackLatent(packedNoise, c: c, t: 1, h: hLat / 2, w: wLat / 2)
        let sigmas = try fx("14_sigmas")
        let timesteps = try fx("14_timesteps")

        // The oracle's eps-call order per step (no sources, all omegas > 0):
        //   call0 = wtxt_wvit (wvae)   call1 = wotxt_wovit (wovae, base)
        //   call2 = wotxt_wovit (wvae ≡ base)   call3 = wtxt_wovit (wvae)
        var worst: Float = 0          // step-0 per-forward rel (pre-feedback)
        var worstCos: Float = 1       // step-0 per-forward cosine
        var stepIdx = 0
        let final = try wvitcfgSample(
            high: high, low: nil, contexts: contexts,
            targetShape: [c, 1, hLat, wLat], headDim: 128,
            boundaryTimestep: renderer.boundaryTimestep, steps: steps,
            injectedNoise: noise, injectedSigmas: sigmas,
            injectedTimesteps: timesteps
        ) { i, trace in
            func check(_ name: String, _ ours: MLXArray, packedRef: String) throws {
                let ref = try fx(packedRef)
                let a = packLatent(ours).asType(.float32).reshaped(-1)
                let b = ref.asType(.float32).reshaped(-1)
                let d = MLX.abs(a - b).max().item(Float.self)
                let rel = d / (MLX.abs(b).max().item(Float.self) + 1e-9)
                let cosine = (a * b).sum().item(Float.self)
                    / (sqrt((a * a).sum().item(Float.self))
                        * sqrt((b * b).sum().item(Float.self)) + 1e-9)
                // Gate only STEP-0 per-forward eps: later steps inherit the
                // perturbed latent and the 11x omega sum amplifies — trajectory
                // health is judged by the final-latent cosine instead.
                if i == 0 && !name.contains("noise_pred") {
                    worst = max(worst, rel)
                    worstCos = min(worstCos, cosine)
                }
                print("  s\(i) \(name): max_abs = \(d)  rel = \(rel)  cos = \(cosine)")
            }
            try check("eps wtxt_wvit", trace.epsWtxtWvit, packedRef: "12_s\(i)_eps_call0")
            try check("eps base     ", trace.base, packedRef: "12_s\(i)_eps_call1")
            try check("eps wvae-base", trace.base, packedRef: "12_s\(i)_eps_call2")
            try check("eps wtxt_wovit", trace.epsWtxtWovit, packedRef: "12_s\(i)_eps_call3")
            try check("noise_pred   ", trace.noisePred, packedRef: "12_s\(i)_noise_pred")
            stepIdx = i
        }
        _ = stepIdx

        let refFinal = try fx("13_final_latent").reshaped(c, 1, hLat, wLat)
        let fa = final.asType(.float32).reshaped(-1)
        let fb = refFinal.asType(.float32).reshaped(-1)
        let dFinal = MLX.abs(fa - fb).max().item(Float.self)
        let relFinal = dFinal / (MLX.abs(fb).max().item(Float.self) + 1e-9)
        let cosFinal = (fa * fb).sum().item(Float.self)
            / (sqrt((fa * fa).sum().item(Float.self))
                * sqrt((fb * fb).sum().item(Float.self)) + 1e-9)
        print("  final latent: max_abs = \(dFinal)  rel = \(relFinal)  cos = \(cosFinal)")

        // Cross-framework fp32 (torch oracle vs MLX, 40-layer A14B): gate the
        // step-0 per-forward eps (accumulation budget: rel <= 1%, cos >= 0.9999 —
        // measured 0.3-0.7% / 0.99997+) and the TRAJECTORY on final-latent cosine.
        // Later-step max-abs grows by construction (feedback + 11x omega sum);
        // the decisive check is the V3 e2e decoded output.
        let pass = worst <= 1e-2 && worstCos >= 0.9999 && cosFinal >= 0.999
        print(pass
            ? "[v2-gate] ALL PASS (step0 eps worst rel \(worst), cos \(worstCos); final cos \(cosFinal))"
            : "[v2-gate] FAIL (step0 eps rel \(worst) [gate 1e-2], cos \(worstCos) [gate 0.9999], final cos \(cosFinal) [gate 0.999])")
        if !pass { exit(1) }
    }
}
