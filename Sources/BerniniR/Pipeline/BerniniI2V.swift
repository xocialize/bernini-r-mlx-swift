// AnimeGen-I2V conditioning path — Wan2.2-I2V-A14B channel-concat (in_dim=36).
//
// The DiT input is [noisy_latent(16), mask(4), image_latent(16)] = 36 channels. Mirrors
// mlx-video's wan_2 i2v (generate.py `is_i2v_channel_concat`): encode a video whose first frame
// is the conditioning image and the rest zeros through the 16-ch WanVAE → z_video(16); build a
// first-frame temporal mask(4); y = [mask, z_video](20) is CONSTANT across steps and concatenated
// onto the 16-ch noisy latent every denoise step (via `denoiseT2V(yCond:)`). No CLIP branch
// (the checkpoint's image_dim is null). Recipe: CFG-free euler, 4 steps, shift 3.0.

import Foundation
import WanCore
import MLX
import MLXRandom

extension BerniniPipeline {

    /// Build the constant i2v conditioning y = [mask(4), image_latent(16)] → [20, T_lat, H_lat, W_lat].
    /// `imagePixels`: [1, 3, 1, H, W] in [-1, 1] (the conditioning frame; wrapper-decoded).
    private func buildI2VCond(
        imagePixels: MLXArray, width: Int, height: Int, numFrames: Int
    ) -> MLXArray {
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]
        let tLat = (numFrames - 1) / config.vaeStride[0] + 1

        // Video: frame0 = image, rest = zeros → [1, 3, F, H, W]; encode → [16, T_lat, H_lat, W_lat].
        let video = concatenated(
            [imagePixels, zeros([1, 3, numFrames - 1, height, width])], axis: 2)
        let zVideo = vae.encode(video)[0]  // [16, T_lat, H_lat, W_lat]

        // Mask: 1 for the first pixel-frame, 0 for the rest; temporally reshaped to 4 channels
        // (VAE temporal stride 4). Matches mlx-video build: repeat first frame 4×, reshape, transpose.
        var msk = concatenated(
            [ones([1, 1, hLat, wLat]), zeros([1, numFrames - 1, hLat, wLat])], axis: 1)
        msk = concatenated(
            [repeated(msk[0..., ..<1], count: 4, axis: 1), msk[0..., 1...]], axis: 1)  // [1, F+3, h, w]
        msk = msk.reshaped([1, tLat, 4, hLat, wLat]).transposed(0, 2, 1, 3, 4)[0]  // [4, T_lat, h, w]

        let y = concatenated([msk, zVideo], axis: 0)  // [20, T_lat, H_lat, W_lat]
        eval(y)
        return y
    }

    /// Image-to-video (AnimeGen-I2V): condition on a single first frame. CFG-free few-step recipe.
    /// Returns frames [1, 3, T, H, W] in [-1, 1].
    public func i2v(
        prompt: String,
        imagePixels: MLXArray,
        negativePrompt: String? = nil,
        width: Int = 832,
        height: Int = 480,
        numFrames: Int = 81,
        steps: Int = 4,
        shift: Double = 3.0,
        seed: UInt64? = nil,
        onStep: ((Int, Int, MLXArray) throws -> Void)? = nil
    ) throws -> MLXArray {
        let negative = negativePrompt ?? config.sampleNegPrompt
        // §2.4: page umT5 in, encode, evict before denoise.
        let (contextCond, contextNull) = try withTextEncoder { enc -> (MLXArray, MLXArray) in
            let c = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: prompt, textLen: config.textLen)
            let n = encodeText(
                encoder: enc, tokenizer: tokenizer, prompt: negative, textLen: config.textLen)
            eval(c, n)
            return (c, n)
        }

        let y = buildI2VCond(
            imagePixels: imagePixels, width: width, height: height, numFrames: numFrames)

        let tLat = (numFrames - 1) / config.vaeStride[0] + 1
        let hLat = height / config.vaeStride[1]
        let wLat = width / config.vaeStride[2]
        if let seed { MLXRandom.seed(seed) }
        let noise = MLXRandom.normal([config.vaeZDim, tLat, hLat, wLat])

        // CFG-free euler few-step (AnimeGen-I2V recipe). yCond carries the 36-ch conditioning.
        let options = T2VOptions(steps: steps, shift: shift, scheduler: .euler, noCFG: true)
        let latent = try denoiseT2V(
            renderer: renderer, contextCond: contextCond, contextNull: contextNull,
            noise: noise, options: options, yCond: y, onStep: onStep)

        let frames = decodeStreaming(vae: vae, latent.expandedDimensions(axis: 0))
        eval(frames)
        return frames
    }
}
