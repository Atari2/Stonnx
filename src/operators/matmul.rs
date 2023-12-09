use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

use super::_commonmatmul::matmul_impl;

const _OPSET_VERSIONS: [i64; 3] = [1, 9, 13];

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_matmul.py>
/// <https://onnx.ai/onnx/operators/onnx__MatMul.html>
pub fn matmul(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let a = inputs[0];
    let b = inputs[1];

    match (a, b) {
        (TensorType::F32(a), TensorType::F32(b)) => {
            Ok(TensorType::F32(matmul_impl(a.view(), b.view())?).into())
        }
        _ => todo!("Matmul for types {:?} and {:?}", a, b),
    }
}
