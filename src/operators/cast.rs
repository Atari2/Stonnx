use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 5] = [1, 6, 9, 13, 19];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_cast.py
/// https://onnx.ai/onnx/operators/onnx__Cast.html
pub fn cast(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("Cast")
}
