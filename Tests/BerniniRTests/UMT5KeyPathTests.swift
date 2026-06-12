import Foundation
import MLX
import MLXNN
import Testing
@testable import BerniniR

// THE KEY CONTRACT for the umT5 lift: the flattened parameter paths of the
// instantiated encoder must equal `BerniniWeightKeys.t5Keys()` (242 keys,
// all-weight-no-bias, per-block pos bias). Instantiation only — no weights,
// no forward pass, no `mx.eval` — so the lazy graph never touches Metal and
// this runs on the plain `swift test` (no-metallib) tier.

@Suite struct UMT5KeyPathTests {

    @Test func parameterPathsMatchT5KeyContract() {
        // Widths shrunk: the path set depends only on the module structure and
        // the layer count (kept at the real 24), not on tensor dimensions —
        // this keeps the un-evaluated init graph trivially small.
        let model = UMT5EncoderModel(
            vocabSize: 8, dim: 8, dimAttn: 8, dimFFN: 16,
            numHeads: 2, numLayers: 24, numBuckets: 4,
            sharedPos: false
        )
        let paths = Set(model.parameters().flattened().map(\.0))
        let expected = BerniniWeightKeys.t5Keys()

        let missing = expected.subtracting(paths)
        let unexpected = paths.subtracting(expected)
        #expect(missing.isEmpty,
                "\(missing.count) contract keys absent from the model, e.g. \(missing.sorted().prefix(5))")
        #expect(unexpected.isEmpty,
                "\(unexpected.count) model paths outside the contract, e.g. \(unexpected.sorted().prefix(5))")
        #expect(paths.count == 242)
    }

    @Test func defaultInitMatchesBerniniCheckpointConfig() {
        // The donor defaults already encode the Bernini t5 hyperparameters
        // (vocab 256384 / dim 4096 / ffn 10240 / 64H / 24L / 32 buckets /
        // per-block bias). Structure-only checks; nothing is evaluated.
        let model = UMT5EncoderModel()
        #expect(model.blocks.count == 24)
        #expect(model.posEmbedding == nil)          // sharedPos=false: no shared bias
        #expect(model.blocks[0].posEmbedding != nil) // per-block bias instead
        #expect(model.blocks[0].attn.numHeads == 64)
        #expect(model.blocks[0].attn.headDim == 64)
        #expect(model.blocks[0].ffn.dimFFN == 10240)
        #expect(Set(model.parameters().flattened().map(\.0)) == BerniniWeightKeys.t5Keys())
    }
}
