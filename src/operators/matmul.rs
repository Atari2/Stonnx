use crate::common::{ArrayType, BoxResult, OperationResult};
use crate::onnx::NodeProto;

use super::_commonmatmul::matmul_impl;

const _OPSET_VERSIONS: [i64; 3] = [1, 9, 13];

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_matmul.py>
/// <https://onnx.ai/onnx/operators/onnx__MatMul.html>
pub fn matmul(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let a = inputs[0];
    let b = inputs[1];

    match (a, b) {
        (ArrayType::F32(a), ArrayType::F32(b)) => {
            Ok(ArrayType::F32(matmul_impl(a.view(), b.view())?).into())
        }
        _ => todo!("Matmul for types {:?} and {:?}", a, b),
    }
}
