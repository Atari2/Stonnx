use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 5] = [1, 2, 11, 13, 18];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_split.py
/// https://onnx.ai/onnx/operators/onnx__Split.html
pub fn split(
    _inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("Split")
}
