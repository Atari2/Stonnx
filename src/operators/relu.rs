use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 13, 14];

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_relu.py>
/// <https://onnx.ai/onnx/operators/onnx__Relu.html>
pub fn relu(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    if inputs.len() != 1 {
        Err(anyhow!("Relu must have 1 input"))
    } else {
        match inputs[0] {
            TensorType::F32(a) => Ok(TensorType::F32(a.mapv(|v| v.max(0.0))).into()),
            TensorType::I64(a) => Ok(TensorType::I64(a.mapv(|v| v.max(0))).into()),
            _ => Err(anyhow!("Relu: invalid input")),
        }
    }
}
