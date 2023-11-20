use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 3] = [1, 9, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_matmul.py
/// https://onnx.ai/onnx/operators/onnx__MatMul.html
pub fn matmul(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("MatMul")
}
