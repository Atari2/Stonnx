use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType, BoxResult, OperationResult},
};
use rand::Rng;
const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 10, 12, 13];

#[derive(Debug)]
struct DropoutAttrs {
    is_test: bool,
    ratio: f32,
    seed: i64,
}

// attrs_1 | attrs_6   => is_test: int (default 0)
//                        ratio: float (default 0.5)
// attrs_7 | attrs_10  => ratio: float (default 0.5)
// attrs_12 | attrs_13 => seed: int (default 0)
impl DropoutAttrs {
    fn new(node: &NodeProto, version: i64) -> Self {
        if version < 6 {
            // is_test, ratio
            Self {
                is_test: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "is_test")
                    .map_or(false, |a| a.i.unwrap_or(0) != 0),
                ratio: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "ratio")
                    .map_or(0.5, |a| a.f.unwrap_or(0.5)),
                seed: 0,
            }
        } else if version > 6 && version < 12 {
            Self {
                is_test: false,
                ratio: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "ratio")
                    .map_or(0.5, |a| a.f.unwrap_or(0.5)),
                seed: 0,
            }
        } else {
            let seed: i64 = rand::thread_rng().gen();
            Self {
                is_test: false,
                ratio: 0.5,
                seed: node
                    .attribute
                    .iter()
                    .find(|a| a.name() == "seed")
                    .map_or(seed, |a| a.i.unwrap_or(seed)),
            }
        }
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_dropout.py
/// https://onnx.ai/onnx/operators/onnx__Dropout.html
pub fn dropout(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    let attrs = DropoutAttrs::new(node, target_version);
    if target_version < 12 {
        todo!("Dropout for opset < 12");
    } else {
        todo!("Dropout for opset >= 12");
    }
}
