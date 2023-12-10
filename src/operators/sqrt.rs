use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// Performs element-wise square root.
///
/// [Python reference](<https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_sqrt.py>)
///
/// [ONNX Documentation](<https://onnx.ai/onnx/operators/onnx__Sqrt.html>)
pub fn sqrt(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0].to_owned();

    match data {
        TensorType::F32(x) => Ok(TensorType::F32(x.mapv(|v| v.sqrt())).into()),
        x => {
            todo!("Sqrt for type {}", x);
        }
    }
}
