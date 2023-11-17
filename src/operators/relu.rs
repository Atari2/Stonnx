use crate::{
    onnx::NodeProto,
    utils::{ArrayType, BoxResult, OperationResult},
};

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 13, 14];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_relu.py
/// https://onnx.ai/onnx/operators/onnx__Relu.html
pub fn relu(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    if inputs.len() != 1 {
        Err("Relu must have 1 input".into())
    } else {
        match inputs[0] {
            ArrayType::F32(a) => Ok(ArrayType::F32(a.mapv(|v| v.max(0.0))).into()),
            ArrayType::I64(a) => Ok(ArrayType::I64(a.mapv(|v| v.max(0))).into()),
            _ => Err("Relu: invalid input".into()),
        }
    }
}
