//
//  BerniniProcessor.swift
//  BerniniR — Bernini-v2 planner plane (PROCESSOR-SPEC.md)
//
//  The planner PREPROCESSOR for the no-input t2v / t2i tasks: prompt →
//  the three planner streams (cond / uncond / imgcond) exactly as upstream
//  builds them:
//    - conversation template + per-task system sentence
//      (`bernini/data_utils.py generate_unified_inputs`,
//       `bernini/data/bernini_template.py encode_messages`)
//    - Qwen2 BPE tokenization — per chunk the `<|im_start|>role\n` header and
//      the stripped content are encoded SEPARATELY with
//      add_special_tokens=False; no `<|im_end|>`, no inter-chunk separator
//    - the target vision block emitted directly as
//      `[<|vision_start|>] + pad×N + [<|vision_end|>]` (equivalent to
//      upstream's `<|visual_output_token_pad_0|>` placeholder-then-replace;
//      the runtime ids 151729.. never escape `encode_messages`)
//    - token roles/segments (`token_types` ∈ {0, 3}, `token_segment_ids` =
//      arange with the target run = 1) and the 4-D additive mask via the
//      existing `buildCustomAttentionMask`
//    - 3-axis M-RoPE position ids (transformers 4.57.3
//      `Qwen2_5_VLModel.get_rope_index` as invoked: `second_per_grid_ts`
//      absent → video t-step = tokens_per_second·1.0 = 2, image t-step = 0)
//    - N (the ViT target token count) from the Qwen processor grid formulas —
//      the fake ViT/VAE runs are shape-only for t2v/t2i and are skipped.
//
//  Everything host-side except the mask build; no model weights are touched.
//

import Foundation
import MLX
import Tokenizers

/// Bernini task kinds covered by this processor (no input images/videos).
public enum BerniniPlannerTask: String, Sendable {
    case t2v
    case t2i
}

/// One stream's planner-ready arrays, in the fixture layout
/// (`00_<stream>_*.npy` of the goldens packs).
public struct BerniniProcessedStream {
    /// `[L]` — MLLM ids, target pads already 151655/151656.
    public let inputIds: [Int]
    /// `[L]` — 0 = text (t), 3 = visual output (o); 1/2 never occur for t2v/t2i.
    public let tokenTypes: [Int32]
    /// `[L]` — arange(L) with the target run overwritten to `visual_id + 1` = 1.
    public let tokenSegmentIds: [Int32]
    /// Indices of the N target pads (== `visual_output_token_mask.nonzero()`).
    public let visualOutputIndices: [Int32]
    /// `[3·L]` host layout, rows (t, h, w) — the fixtures' `(3, L)`.
    public let positionIds: [Int]

    public var length: Int { inputIds.count }

    /// Additive 0 / −inf mask, `[1, L, L]` f32 (upstream
    /// `build_custom_attention_mask` over the roles/segments above).
    public func attentionMask4D() -> MLXArray {
        buildCustomAttentionMask(
            tokenType: [tokenTypes], tokenSegmentIds: [tokenSegmentIds])
    }

    /// Planner-call form `[3, 1, L]` (the model site transposes the
    /// unsqueezed `(1, 3, L)` → `(3, 1, L)`).
    public func positionIdsArray() -> MLXArray {
        MLXArray(positionIds.map(Int32.init), [3, 1, length])
    }
}

/// The three streams + the shared target-grid facts.
public struct BerniniProcessedInputs {
    public let cond: BerniniProcessedStream
    public let uncond: BerniniProcessedStream
    /// Identical to `cond` for t2v/t2i (nothing to drop; fixture-verified).
    public let imgcond: BerniniProcessedStream
    /// N = gridT·gridH·gridW / merge_size² — the ViT target token count.
    public let vitTokenCount: Int
    public let gridT: Int
    public let gridH: Int
    public let gridW: Int
}

/// Upstream `bernini_process_sample` preprocessing (t2v/t2i scope).
public struct BerniniProcessor {
    public let tokenizer: any Tokenizer

    /// `DEFAULT_NEG_PROMPT` (`bernini/cli.py:28-33`) — the standard Wan2.2
    /// Chinese negative prompt, used verbatim (NOT `_prompt_clean`ed on the
    /// Qwen side) as the uncond stream's user text.
    public static let defaultNegativePrompt =
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
        + "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
        + "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
        + "杂乱的背景，三条腿，背景人很多，倒着走"

    // Fixed special ids (`mllm/tokenizer_config.json`, `mllm/config.json:6-12`).
    static let visionStartId = 151652
    static let visionEndId = 151653
    static let imagePadId = 151655
    static let videoPadId = 151656

    /// `SYSTEM_PROMPT[task]` (`bernini_template.py:108-116`; t2v/t2i rows only —
    /// i2i/v2v/r2v/rv2v are out of this processor's scope).
    static func systemPrompt(for task: BerniniPlannerTask) -> String {
        switch task {
        case .t2v:
            return "You are a helpful assistant specialized in text-to-video generation."
        case .t2i:
            return "You are a helpful assistant specialized in text-to-image generation."
        }
    }

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Load the Qwen2 BPE from a local `mllm/` folder (tokenizer.json +
    /// tokenizer_config.json — the shipped files; the goldens depend on the
    /// as-shipped pretokenizer regex, NOT transformers' `fix_mistral_regex`
    /// variant).
    public static func fromPretrained(mllmDir: URL) async throws -> BerniniProcessor {
        BerniniProcessor(tokenizer: try await AutoTokenizer.from(modelFolder: mllmDir))
    }

    // MARK: - Text cleaning

    /// Upstream `_prompt_clean` (`pipeline.py:164-167`): ftfy.fix_text +
    /// html.unescape ×2 + `\s+` → " " + strip. ftfy and HTML unescaping are
    /// the identity for ASCII prompts without entities and are deliberately
    /// skipped here (PROCESSOR-SPEC.md Ambiguities #1) — prompts with
    /// mojibake/entities need host-side cleaning before this call.
    static func promptClean(_ text: String) -> String {
        text.replacingOccurrences(
            of: "\\s+", with: " ", options: .regularExpression
        ).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Grid / N

    /// transformers `smart_resize` (`image_processing_qwen2_vl.py:54-81`) with
    /// the pipeline's pixel bounds: factor 28, min 3136 (56²), max 50176 (224²).
    /// First rounding is Python `round` = half-to-even.
    static func smartResize(
        height: Int, width: Int, factor: Int = 28,
        minPixels: Int = 3136, maxPixels: Int = 50176
    ) -> (h: Int, w: Int) {
        let maxSide = Double(max(height, width))
        let minSide = Double(min(height, width))
        precondition(
            maxSide / minSide <= 200,
            "absolute aspect ratio must be smaller than 200 (\(height)×\(width))")
        var hBar = Int((Double(height) / Double(factor)).rounded(.toNearestOrEven)) * factor
        var wBar = Int((Double(width) / Double(factor)).rounded(.toNearestOrEven)) * factor
        if hBar * wBar > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            hBar = max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            wBar = max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if hBar * wBar < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            hBar = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            wBar = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (hBar, wBar)
    }

    /// The target ViT grid (§2.4): t2v temporal axis from `smart_video_nframes`
    /// at the default fps pair (vae_fps 16, vit_fps 2) grouped by
    /// temporal_patch_size 2; t2i is a single temporally-duplicated image
    /// (grid_t = 1). Spatial axis from `smartResize` / patch_size 14.
    static func targetGrid(
        task: BerniniPlannerTask, height: Int, width: Int, numFrames: Int
    ) -> (t: Int, h: Int, w: Int) {
        let gridT: Int
        switch task {
        case .t2v:
            // F = clamp(2·⌊numFrames·(vit_fps/vae_fps)/2⌋, min 2, max 2·⌊numFrames/2⌋)
            //   = max(2, 2·⌊numFrames/16⌋) for the default fps pair.
            let f = min(max(2, 2 * (numFrames / 16)), 2 * (numFrames / 2))
            gridT = f / 2
        case .t2i:
            gridT = 1
        }
        let (hBar, wBar) = smartResize(height: height, width: width)
        return (gridT, hBar / 14, wBar / 14)
    }

    // MARK: - Processing

    /// Build the three planner streams for a t2v/t2i request. `negativePrompt`
    /// enters the uncond stream raw (upstream passes it uncleaned into
    /// `encode_messages`); pass `""` to reproduce the no-user-chunk uncond.
    public func process(
        prompt: String,
        task: BerniniPlannerTask,
        width: Int,
        height: Int,
        numFrames: Int,
        negativePrompt: String = BerniniProcessor.defaultNegativePrompt
    ) -> BerniniProcessedInputs {
        let grid = Self.targetGrid(
            task: task, height: height, width: width, numFrames: numFrames)
        let n = grid.t * grid.h * grid.w / 4  // merge_size² = 4
        let cleaned = Self.promptClean(prompt)
        let padId = task == .t2v ? Self.videoPadId : Self.imagePadId
        let system = Self.systemPrompt(for: task)

        // cond: system + user(cleaned prompt) + target block. imgcond is
        // byte-identical for t2v/t2i (img/video dropout has nothing to drop).
        let cond = buildStream(
            systemText: system, userText: cleaned, task: task,
            padId: padId, n: n, grid: grid)
        // uncond: text_dropout drops the prompt; a >1-char neg prompt REPLACES
        // it (`bernini_template.py:209-216`); with none the user chunk
        // vanishes entirely (empty-content chunks are dropped).
        let uncondUser = negativePrompt.unicodeScalars.count > 1 ? negativePrompt : nil
        let uncond = buildStream(
            systemText: system, userText: uncondUser, task: task,
            padId: padId, n: n, grid: grid)

        return BerniniProcessedInputs(
            cond: cond, uncond: uncond, imgcond: cond,
            vitTokenCount: n, gridT: grid.t, gridH: grid.h, gridW: grid.w)
    }

    /// Assemble + tokenize one stream: `<|im_start|>role\n` headers and
    /// stripped contents encoded separately (never merged across the
    /// boundary), empty-content chunks dropped, vision block emitted as ids.
    private func buildStream(
        systemText: String, userText: String?, task: BerniniPlannerTask,
        padId: Int, n: Int, grid: (t: Int, h: Int, w: Int)
    ) -> BerniniProcessedStream {
        var ids: [Int] = []
        appendTextChunk(role: "system", content: systemText, into: &ids)
        if let userText {
            appendTextChunk(role: "user", content: userText, into: &ids)
        }
        // Assistant chunk: header + the target vision block. `special_token`
        // items ([SOV]/[EOV]/[EOS]) are skipped by `encode_messages`; the
        // `video_gen`/`image_gen` item is has_loss==1 and NEVER dropped.
        ids += encodeHeader(role: "assistant")
        ids.append(Self.visionStartId)
        let runStart = ids.count
        ids.append(contentsOf: [Int](repeating: padId, count: n))
        ids.append(Self.visionEndId)

        let l = ids.count
        var tokenTypes = [Int32](repeating: PlannerTokenType.text.rawValue, count: l)
        var tokenSegmentIds = (0 ..< l).map(Int32.init)
        for i in runStart ..< runStart + n {
            tokenTypes[i] = PlannerTokenType.output.rawValue
            tokenSegmentIds[i] = 1  // visual_id 0 + 1
        }

        return BerniniProcessedStream(
            inputIds: ids,
            tokenTypes: tokenTypes,
            tokenSegmentIds: tokenSegmentIds,
            visualOutputIndices: (runStart ..< runStart + n).map(Int32.init),
            positionIds: Self.ropePositions(
                length: l, runStart: runStart, task: task, grid: grid))
    }

    private func encodeHeader(role: String) -> [Int] {
        tokenizer.encode(text: "<|im_start|>" + role + "\n", addSpecialTokens: false)
    }

    /// One text chunk (`bernini_template.py:274-283`): dropped when the
    /// stripped content is empty; otherwise header ids + content ids.
    private func appendTextChunk(role: String, content: String, into ids: inout [Int]) {
        let stripped = content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else { return }
        ids += encodeHeader(role: role)
        ids += tokenizer.encode(text: stripped, addSpecialTokens: false)
    }

    /// `get_rope_index` as invoked (§4): one text run 0..<runStart (all three
    /// rows sequential), the vision run at base b = runStart with
    /// t = b + k·(video 2 / image 0), h = b + row, w = b + col flattened
    /// (t, h, w) row-major over the LLM grid (h/2 × w/2), then sequential
    /// rows-equal from the overall max + 1. Returns `[3·L]`, rows (t, h, w).
    static func ropePositions(
        length: Int, runStart: Int, task: BerniniPlannerTask,
        grid: (t: Int, h: Int, w: Int)
    ) -> [Int] {
        let llmH = grid.h / 2  // spatial_merge_size 2
        let llmW = grid.w / 2
        let tStep = task == .t2v ? 2 : 0  // tokens_per_second·second_per_grid_t
        var tRow = [Int](); tRow.reserveCapacity(length)
        var hRow = [Int](); hRow.reserveCapacity(length)
        var wRow = [Int](); wRow.reserveCapacity(length)
        for i in 0 ..< runStart {
            tRow.append(i); hRow.append(i); wRow.append(i)
        }
        let b = runStart
        for k in 0 ..< grid.t {
            for row in 0 ..< llmH {
                for col in 0 ..< llmW {
                    tRow.append(b + tStep * k)
                    hRow.append(b + row)
                    wRow.append(b + col)
                }
            }
        }
        var next = b + max(tStep * (grid.t - 1), llmH - 1, llmW - 1) + 1
        for _ in (runStart + grid.t * llmH * llmW) ..< length {
            tRow.append(next); hRow.append(next); wRow.append(next)
            next += 1
        }
        return tRow + hRow + wRow
    }
}
