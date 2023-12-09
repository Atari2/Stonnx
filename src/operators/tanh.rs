use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// <https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_tanh.py>
/// <https://onnx.ai/onnx/operators/onnx__Tanh.html>
pub fn tanh(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0].to_owned();

    match data {
        TensorType::F32(x) => Ok(TensorType::F32(x.mapv(|v| v.tanh())).into()),
        x => {
            todo!("Tanh for type {}", x);
        }
    }
}
