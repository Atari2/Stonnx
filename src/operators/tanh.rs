use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// Calculates the hyperbolic tangent of the given input tensor element-wise.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_tanh.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Tanh.html)
pub fn tanh(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0].to_owned();

    match data {
        TensorType::F32(x) => {
            let mut x = x.clone();
            x.par_mapv_inplace(|v| v.tanh());
            Ok(TensorType::F32(x).into())
        }
        x => {
            todo!("Tanh for type {}", x);
        }
    }
}
