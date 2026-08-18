//
//  DiffLossFM.swift
//  BerniniR — Bernini-v2 planner plane (PORTING-SPEC-V2.md)
//
//  Port of upstream `bernini/models/diffloss_fm.py` (inference surface) +
//  `bernini/models/scheduler.py` FlowMatchScheduler: the per-token flow-matching
//  head (`SimpleMLPAdaLN`) that samples revealed ViT-plan tokens inside the
//  MaskGIT loop, with the 3-stream txt+img CFG combination.
//
//  Weight source: `vit_decoder.safetensors` (keys `net.*`, verbatim upstream;
//  Sequential indices are sanitized to named submodules at load).
//
//  v2 config (`clip_diff_cfg`): in/out 3584, width 4096, depth 16, z 3584,
//  scheduler shift 2.0, extra_one_step=true. Production sampling runs
//  `vit_denoising_step = 3` euler steps with SHARED noise across CFG streams —
//  noise is INJECTED here (MLX RNG ≠ torch RNG; parity fixtures carry it).
//

import Foundation
import MLX
import MLXNN

/// `x * (1 + scale) + shift` — the DiT-style modulation.
private func modulate(_ x: MLXArray, _ shift: MLXArray, _ scale: MLXArray) -> MLXArray {
    x * (1 + scale) + shift
}

/// Sinusoidal timestep embedding → 2-layer MLP (upstream `TimestepEmbedder`).
final class FMTimestepEmbedder: Module {
    let frequencyEmbeddingSize: Int
    @ModuleInfo(key: "mlp_0") var mlp0: Linear
    @ModuleInfo(key: "mlp_2") var mlp2: Linear

    init(hiddenSize: Int, frequencyEmbeddingSize: Int = 256) {
        self.frequencyEmbeddingSize = frequencyEmbeddingSize
        self._mlp0.wrappedValue = Linear(frequencyEmbeddingSize, hiddenSize, bias: true)
        self._mlp2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    static func timestepEmbedding(_ t: MLXArray, dim: Int, maxPeriod: Float = 10000)
        -> MLXArray
    {
        let half = dim / 2
        let freqs = MLX.exp(
            -log(maxPeriod) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let args = t[0..., .newAxis].asType(.float32) * freqs[.newAxis, 0...]
        return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let freq = Self.timestepEmbedding(t, dim: frequencyEmbeddingSize)
        return mlp2(silu(mlp0(freq.asType(mlp0.weight.dtype))))
    }
}

/// Upstream `ResBlock`: affine LayerNorm → modulate → Linear-SiLU-Linear, gated residual.
final class FMResBlock: Module {
    @ModuleInfo(key: "in_ln") var inLn: LayerNorm
    @ModuleInfo(key: "mlp_0") var mlp0: Linear
    @ModuleInfo(key: "mlp_2") var mlp2: Linear
    @ModuleInfo(key: "adaLN") var adaLN: Linear

    init(channels: Int) {
        self._inLn.wrappedValue = LayerNorm(dimensions: channels, eps: 1e-6)
        self._mlp0.wrappedValue = Linear(channels, channels, bias: true)
        self._mlp2.wrappedValue = Linear(channels, channels, bias: true)
        self._adaLN.wrappedValue = Linear(channels, 3 * channels, bias: true)
    }

    func callAsFunction(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let mod = adaLN(silu(y))
        let parts = split(mod, parts: 3, axis: -1)
        let (shiftMlp, scaleMlp, gateMlp) = (parts[0], parts[1], parts[2])
        var h = modulate(inLn(x), shiftMlp, scaleMlp)
        h = mlp2(silu(mlp0(h)))
        return x + gateMlp * h
    }
}

/// Upstream `FinalLayer`: affine-free LayerNorm → modulate → Linear.
final class FMFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var normFinal: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "adaLN") var adaLN: Linear

    init(modelChannels: Int, outChannels: Int) {
        self._normFinal.wrappedValue = LayerNorm(
            dimensions: modelChannels, eps: 1e-6, affine: false)
        self._linear.wrappedValue = Linear(modelChannels, outChannels, bias: true)
        self._adaLN.wrappedValue = Linear(modelChannels, 2 * modelChannels, bias: true)
    }

    func callAsFunction(_ x: MLXArray, _ c: MLXArray) -> MLXArray {
        let parts = split(adaLN(silu(c)), parts: 2, axis: -1)
        return linear(modulate(normFinal(x), parts[0], parts[1]))
    }
}

/// Upstream `SimpleMLPAdaLN`.
final class SimpleMLPAdaLN: Module {
    let inChannels: Int

    @ModuleInfo(key: "time_embed") var timeEmbed: FMTimestepEmbedder
    @ModuleInfo(key: "cond_embed") var condEmbed: Linear
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "res_blocks") var resBlocks: [FMResBlock]
    @ModuleInfo(key: "final_layer") var finalLayer: FMFinalLayer

    init(inChannels: Int, modelChannels: Int, outChannels: Int, zChannels: Int,
         numResBlocks: Int)
    {
        self.inChannels = inChannels
        self._timeEmbed.wrappedValue = FMTimestepEmbedder(hiddenSize: modelChannels)
        self._condEmbed.wrappedValue = Linear(zChannels, modelChannels)
        self._inputProj.wrappedValue = Linear(inChannels, modelChannels)
        self._resBlocks.wrappedValue = (0 ..< numResBlocks).map { _ in
            FMResBlock(channels: modelChannels)
        }
        self._finalLayer.wrappedValue = FMFinalLayer(
            modelChannels: modelChannels, outChannels: outChannels)
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, c: MLXArray) -> MLXArray {
        var h = inputProj(x)
        let y = timeEmbed(t) + condEmbed(c)
        for block in resBlocks { h = block(h, y) }
        return finalLayer(h, y)
    }

    /// Upstream `forward_with_txt_img_cfg`: rows are [cond | uncond | imgcond] thirds.
    func forwardWithTxtImgCfg(
        _ x: MLXArray, t: MLXArray, c: MLXArray, txtCfgScale: Float, imgCfgScale: Float
    ) -> MLXArray {
        let third = x.dim(0) / 3
        let part = x[0 ..< third]
        let combined = concatenated([part, part, part], axis: 0)
        let out = self(combined, t: t, c: c)
        let eps = out[0..., 0 ..< inChannels]
        let condEps = eps[0 ..< third]
        let uncondEps = eps[third ..< 2 * third]
        let imgcondEps = eps[(2 * third)...]
        let partEps = uncondEps
            + imgCfgScale * (imgcondEps - uncondEps)
            + txtCfgScale * (condEps - imgcondEps)
        return concatenated([partEps, partEps, partEps], axis: 0)
    }
}

/// Upstream `FlowMatchScheduler` (inference surface).
public struct FlowMatchScheduler {
    public let numTrainTimesteps: Int
    public private(set) var sigmas: MLXArray
    public private(set) var timesteps: MLXArray

    /// Sigma schedule exactly as upstream `set_timesteps` (torch computes the
    /// linspace in the target dtype; the oracle default is bf16 — match it, then
    /// carry fp32 for arithmetic).
    public init(
        numInferenceSteps: Int, numTrainTimesteps: Int = 1000, shift: Float,
        sigmaMax: Float = 1.0, sigmaMin: Float = 0.003 / 1.002,
        extraOneStep: Bool = true
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        let count = extraOneStep ? numInferenceSteps + 1 : numInferenceSteps
        var s = MLX.linspace(sigmaMax, sigmaMin, count: count).asType(.bfloat16)
            .asType(.float32)
        if extraOneStep { s = s[0 ..< numInferenceSteps] }
        s = shift * s / (1 + (shift - 1) * s)
        self.sigmas = s
        self.timesteps = s * Float(numTrainTimesteps)
    }

    /// Exact-grid injection (parity fixtures carry the oracle's torch-bf16
    /// sigma AND timestep grids; torch computes `timesteps = sigmas * 1000` IN
    /// BF16 — e.g. 800.78 rounds to 800.0 — and the sinusoidal time embedding
    /// amplifies that. Never re-derive timesteps from injected sigmas.)
    public init(sigmas: MLXArray, timesteps: MLXArray? = nil, numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
        self.sigmas = sigmas.asType(.float32)
        self.timesteps = timesteps?.asType(.float32)
            ?? (self.sigmas * Float(numTrainTimesteps)).asType(.bfloat16).asType(.float32)
    }

    /// Upstream `step`: euler — `sample + v * (sigma_next - sigma)`. Torch holds
    /// sigmas in bf16, so the DELTA is bf16-rounded — emulate that grid.
    public func step(modelOutput: MLXArray, stepIndex: Int, sample: MLXArray) -> MLXArray {
        let sigma = sigmas[stepIndex]
        let sigmaNext: MLXArray =
            stepIndex + 1 < sigmas.dim(0) ? sigmas[stepIndex + 1] : MLXArray(Float(0))
        let delta = (sigmaNext - sigma).asType(.bfloat16).asType(.float32)
        return sample + modelOutput * delta
    }
}

/// The `DiffLoss_FM` inference wrapper (`vit_decoder`).
public final class DiffLossFM: Module {
    public let shift: Float
    @ModuleInfo(key: "net") var net: SimpleMLPAdaLN

    public init(zChannels: Int = 3584, targetChannels: Int = 3584, depth: Int = 16,
                width: Int = 4096, shift: Float = 2.0)
    {
        self.shift = shift
        self._net.wrappedValue = SimpleMLPAdaLN(
            inChannels: targetChannels, modelChannels: width,
            outChannels: targetChannels, zChannels: zChannels, numResBlocks: depth)
    }

    /// Upstream `sample` with txt+img CFG, with the SHARED noise INJECTED:
    /// `noise` is the `[N/3, C]` block the oracle drew once and tiled ×3.
    /// `injectedSigmas`/`injectedTimesteps` override the schedule for exact-grid parity.
    public func sample(
        z: MLXArray, txtCfg: Float, imgCfg: Float, numInferenceSteps: Int,
        injectedNoise noise: MLXArray, injectedSigmas: MLXArray? = nil,
        injectedTimesteps: MLXArray? = nil
    ) -> MLXArray {
        let scheduler = injectedSigmas.map {
            FlowMatchScheduler(sigmas: $0, timesteps: injectedTimesteps)
        } ?? FlowMatchScheduler(numInferenceSteps: numInferenceSteps, shift: shift)
        var samples = concatenated([noise, noise, noise], axis: 0).asType(z.dtype)
        for i in 0 ..< numInferenceSteps {
            let t = scheduler.timesteps[i ..< i + 1].asType(z.dtype)
            let pred = net.forwardWithTxtImgCfg(
                samples, t: t, c: z, txtCfgScale: txtCfg, imgCfgScale: imgCfg)
            samples = scheduler.step(modelOutput: pred, stepIndex: i, sample: samples)
        }
        return samples
    }

    /// Gate probes: expose the inner net for isolation checks.
    public func netForward(_ x: MLXArray, t: MLXArray, c: MLXArray) -> MLXArray {
        net(x, t: t, c: c)
    }

    public func netForwardWithTxtImgCfg(
        _ x: MLXArray, t: MLXArray, c: MLXArray, txtCfg: Float, imgCfg: Float
    ) -> MLXArray {
        net.forwardWithTxtImgCfg(x, t: t, c: c, txtCfgScale: txtCfg, imgCfgScale: imgCfg)
    }

    /// Load `vit_decoder.safetensors` (keys `net.*` verbatim upstream). Sequential
    /// indices sanitize to the named submodules above.
    public static func fromPretrained(file: URL, dtype: DType? = nil) throws -> DiffLossFM {
        let model = DiffLossFM()
        let raw = try MLX.loadArrays(url: file)
        var params: [String: MLXArray] = [:]
        for (key, value) in raw {
            var k = key
            k = k.replacingOccurrences(of: "time_embed.mlp.0", with: "time_embed.mlp_0")
            k = k.replacingOccurrences(of: "time_embed.mlp.2", with: "time_embed.mlp_2")
            k = k.replacingOccurrences(of: ".mlp.0", with: ".mlp_0")
            k = k.replacingOccurrences(of: ".mlp.2", with: ".mlp_2")
            k = k.replacingOccurrences(of: "adaLN_modulation.1", with: "adaLN")
            params[k] = dtype.map { value.asType($0) } ?? value
        }
        try model.update(
            parameters: ModuleParameters.unflattened(params), verify: [.noUnusedKeys])
        return model
    }
}
