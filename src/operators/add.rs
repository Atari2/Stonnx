use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult};

const _OPSET_VERSIONS: [i64; 5] = [1, 6, 7, 13, 14];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_add.py
/// https://onnx.ai/onnx/operators/onnx__Add.html
pub fn add(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
) -> BoxResult<ArrayType> {
    let array_1 = inputs[0];
    let array_2 = inputs[1];

    match (array_1, array_2) {
        (ArrayType::F32(x), ArrayType::F32(y)) => Ok(ArrayType::F32(x + y)),
        (x, y) => {
            todo!("Add for types {} and {}", x, y);
        }
    }
}
