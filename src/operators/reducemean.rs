use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 4] = [1, 11, 13, 18];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_reduce_mean.py
/// https://onnx.ai/onnx/operators/onnx__ReduceMean.html
pub fn reducemean(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("ReduceMean")
}
