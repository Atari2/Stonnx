use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 5] = [1, 6, 7, 13, 14];

/// Performs element-wise binary subtraction (with limited broadcast support).
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_add.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Add.html)
pub fn sub(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let array_1 = inputs[0];
    let array_2 = inputs[1];

    match (array_1, array_2) {
        (TensorType::F32(x), TensorType::F32(y)) => Ok(TensorType::F32(x - y).into()),
        (TensorType::I64(x), TensorType::I64(y)) => Ok(TensorType::I64(x - y).into()),
        (x, y) => {
            todo!("Add for types {} and {}", x, y);
        }
    }
}
