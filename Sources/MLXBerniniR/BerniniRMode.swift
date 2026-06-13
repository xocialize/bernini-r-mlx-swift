import BerniniR
import MLXToolKit

extension Mode {
    /// DPM++(2M) at 16 steps — validated **2.53× faster** than the 40-step
    /// UniPC default with no perceptible quality loss (Bernini int4, 17-frame
    /// t2v, 415.6 s vs 1049.7 s, same seed/peak, 2026-06-12). The
    /// different-but-equally-valid sampler trajectory means the image differs
    /// slightly from the quality path at the same seed.
    public static let fast: Mode = "fast"
    /// 40-step UniPC — the reference quality path (the package default).
    public static let quality: Mode = "quality"
}

/// Resolve a request `mode` (+ any explicit `steps` override) to the core's
/// scheduler and step count. An explicit `steps` on the request always wins;
/// otherwise the mode picks the count (`.fast` → 16, else config default 40).
func resolveSampling(mode: Mode?, steps: Int?) -> (scheduler: SchedulerKind, steps: Int?) {
    switch mode {
    case .fast:
        return (.dpmpp, steps ?? 16)
    default:  // nil / .quality / unknown → reference path (config-default steps)
        return (.unipc, steps)
    }
}
