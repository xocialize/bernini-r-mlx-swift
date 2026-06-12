// 1:1 translation of bernini_r_mlx/model/multiseg.py — the multi-segment
// renderer forward, the core of Bernini's editing/reference engine. The
// token sequence is [cond segments | noisy target], each segment
// patch-embedded with its own SA-3D source_id; self-attention is full over
// the whole sequence (no mask), SA-3D RoPE keeps segments separated, and
// only the target tokens are kept from the output.
//
// Trick carried from the oracle: the block's precomputed-rope path derives
// its length solely from gridSizes[0]'s product, so gridSizes [(L_total,1,1)]
// plus a concatenated (cos, sin) of length L_total applies SA-3D RoPE across
// the entire multi-segment sequence.

import Foundation
import MLX
import MLXNN

/// One conditioning segment: VAE latent [C, T, H, W] + SA-3D source id.
public typealias MultisegSegment = (latent: MLXArray, sourceID: Int)

/// Patch-embed one VAE latent segment and build its SA-3D rope.
/// Returns (tokens [1, L, dim], (cos, sin) each [L, 1, headDim/2], grid).
func patchSegment(
    model: WanModel, latent: MLXArray, sourceID: Int, headDim: Int,
    theta: Float = 10000.0
) -> (MLXArray, (MLXArray, MLXArray), (Int, Int, Int)) {
    let (tokens, grid) = model.patchify(latent)
    let (cosB, sinB) = ropePrecomputeCosSin(
        gridSizes: [grid], freqs: model.freqs, dtype: .float32)
    let (cos, sin) = applySegmentPhase(
        cosB: cosB, sinB: sinB, sourceID: sourceID, headDim: headDim, theta: theta)
    return (tokens, (cos, sin), grid)
}

/// Reproduce WanModel's scalar-timestep modulation: (e0 [1,1,6,dim], e [1,dim]).
func timeEmbed(model: WanModel, t: MLXArray) -> (MLXArray, MLXArray) {
    var t = t
    if t.ndim == 0 {
        t = t.expandedDimensions(axis: 0)
    }
    let sinusoid = t.expandedDimensions(axis: -1).asType(.float32) * model.invFreq
    let sinEmb = concatenated([cos(sinusoid), sin(sinusoid)], axis: -1)
    let e = model.timeEmbedding1(silu(model.timeEmbedding0(sinEmb)))
    let e0 = model.timeProjection(silu(e))
    return (e0.reshaped(1, 1, 6, model.dim), e)
}

/// Run one expert over [cond segments | noisy target] and return the target
/// prediction [C, T, H, W] (float32).
/// - condSegments: ordered (latent [C,T,H,W], source_id) context (may be empty).
/// - targetLatent: noisy target latent [C, T, H, W] (source_id 0).
/// - context: pre-embedded UMT5 text [1, textLen, dim].
public func forwardMultiseg(
    model: WanModel,
    condSegments: [MultisegSegment],
    targetLatent: MLXArray,
    t: MLXArray,
    context: MLXArray,
    headDim: Int,
    crossKVCaches: [(MLXArray, MLXArray)]? = nil,
    theta: Float = 10000.0
) -> MLXArray {
    let wDtype = linearDtype(model.patchEmbeddingProj)

    var tokParts: [MLXArray] = []
    var cosParts: [MLXArray] = []
    var sinParts: [MLXArray] = []
    for (latent, sid) in condSegments {
        let (tok, (cos, sin), _) = patchSegment(
            model: model, latent: latent, sourceID: sid, headDim: headDim, theta: theta)
        tokParts.append(tok)
        cosParts.append(cos)
        sinParts.append(sin)
    }

    let (tgtTok, (tgtCos, tgtSin), tgtGrid) = patchSegment(
        model: model, latent: targetLatent, sourceID: 0, headDim: headDim, theta: theta)
    tokParts.append(tgtTok)
    cosParts.append(tgtCos)
    sinParts.append(tgtSin)

    var x = concatenated(tokParts, axis: 1)  // [1, L_total, dim]
    let cos = concatenated(cosParts, axis: 0)  // [L_total, 1, half_d]
    let sin = concatenated(sinParts, axis: 0)
    let lTotal = x.dim(1)
    let tgtLen = tgtTok.dim(1)

    let (e0, e) = timeEmbed(model: model, t: t)

    let ropeCosSin = (cos.asType(wDtype), sin.asType(wDtype))
    for (i, block) in model.blocks.enumerated() {
        let kv = crossKVCaches?[i]
        x = block(
            x, e: e0, seqLens: [lTotal],
            gridSizes: [(lTotal, 1, 1)],  // block reads only the product as rope length
            freqs: model.freqs, context: context, contextLens: nil,
            crossKVCache: kv, ropeCosSin: ropeCosSin,
            attnMask: nil)  // full attention across all segments
    }

    x = model.head(x, e)  // [1, L_total, out_dim * prod(patch)]
    let target = x[0..., (lTotal - tgtLen)..., 0...]  // target tokens (last segment)
    return model.unpatchify(target, gridSizes: [tgtGrid])[0].asType(.float32)
}
