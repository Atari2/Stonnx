use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 9, 14, 15];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_batch_normalization.py
/// https://onnx.ai/onnx/operators/onnx__BatchNormalization.html
pub fn batchnormalization(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    todo!("BatchNormalization")
}
