use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 4] = [1, 10, 11, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_slice.py
/// https://onnx.ai/onnx/operators/onnx__Slice.html
pub fn slice(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("Slice")
}
