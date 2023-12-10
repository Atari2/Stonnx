use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// Performs element-wise exponential (with limited broadcast support).
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_exp.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Exp.html)
pub fn exp(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0].to_owned();

    match data {
        TensorType::F32(x) => Ok(TensorType::F32(x.mapv(|v| v.exp())).into()),
        x => {
            todo!("Exp for type {}", x);
        }
    }
}
