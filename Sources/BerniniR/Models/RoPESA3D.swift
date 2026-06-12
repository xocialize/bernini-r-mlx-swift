// 1:1 translation of bernini_r_mlx/model/rope_sa3d.py — Segment-Aware 3D
// RoPE (`use_src_id_rotary_emb`), the one new "layer" in Bernini-R. Zero
// parameters: the standard Wan 3D rope is complex-multiplied by a constant
// per-segment "visual id" phase (a 1-D rope over the FULL head dim evaluated
// at position source_id). For a single segment with source_id == 0 the phase
// is identity (cos 1, sin 0) — plain t2v is numerically unchanged; SA-3D
// only affects multi-segment r2v / v2v / rv2v.

import Foundation
import MLX

/// One visual segment: latent grid (f, h, w) + SA-3D source id
/// (0 = target, 1.. = conditioning).
public typealias SA3DSegmentSpec = (grid: (Int, Int, Int), sourceID: Int)

/// Per-segment phase (cos, sin), each [headDim / 2] — matches
/// `get_1d_rotary_pos_embed(head_dim, ...)[source_id]`.
func visualIDPhase(
    sourceID: Int, headDim: Int, theta: Float = 10000.0
) -> (MLXArray, MLXArray) {
    let k = MLXArray(stride(from: 0, to: headDim, by: 2).map { Float($0) })
    let invFreq = exp(-(k / Float(headDim)) * log(MLXArray(theta)))
    let ang = Float(sourceID) * invFreq
    return (cos(ang), sin(ang))
}

/// Complex-multiply base (cos_b, sin_b) by the segment phase: (a+bi)(c+di).
func applySegmentPhase(
    cosB: MLXArray, sinB: MLXArray, sourceID: Int, headDim: Int, theta: Float
) -> (MLXArray, MLXArray) {
    let (c, d) = visualIDPhase(sourceID: sourceID, headDim: headDim, theta: theta)
    let cosN = cosB * c - sinB * d
    let sinN = cosB * d + sinB * c
    return (cosN, sinN)
}

/// Build SA-3D RoPE (cos, sin) for a multi-segment token sequence.
/// - segments: ordered ((f, h, w) latent grid, source_id) — one per visual
///   segment, concatenated along the sequence axis in this order.
/// - freqs: the model's base 3D rope table (`WanModel.freqs`).
/// Returns (cos, sin) each [totalSeq, 1, headDim / 2], a drop-in for
/// `ropeApply`'s `precomputedCosSin`.
public func prepareSA3DRopeCosSin(
    segments: [SA3DSegmentSpec],
    freqs: MLXArray,
    headDim: Int,
    theta: Float = 10000.0,
    dtype: DType = .float32
) -> (MLXArray, MLXArray) {
    var cosParts: [MLXArray] = []
    var sinParts: [MLXArray] = []
    for (grid, sourceID) in segments {
        let (cosB, sinB) = ropePrecomputeCosSin(
            gridSizes: [grid], freqs: freqs, dtype: dtype)
        let (cosN, sinN) = applySegmentPhase(
            cosB: cosB, sinB: sinB, sourceID: sourceID, headDim: headDim, theta: theta)
        cosParts.append(cosN.asType(dtype))
        sinParts.append(sinN.asType(dtype))
    }
    return (concatenated(cosParts, axis: 0), concatenated(sinParts, axis: 0))
}
