use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 5] = [1, 6, 7, 13, 14];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_div.py
/// https://onnx.ai/onnx/operators/onnx__Div.html
pub fn div(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let array_1 = inputs[0];
    let array_2 = inputs[1];

    match (array_1, array_2) {
        (ArrayType::F32(x), ArrayType::F32(y)) => Ok(ArrayType::F32(x / y).into()),
        (ArrayType::I64(x), ArrayType::I64(y)) => Ok(ArrayType::I64(x / y).into()),
        (x, y) => {
            todo!("Div for types {} and {}", x, y);
        }
    }
}
