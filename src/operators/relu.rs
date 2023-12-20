use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 13, 14];

/// Relu takes one input data (Tensor) and produces one output data (Tensor) where the rectified linear function, y = max(0, x), is applied to the tensor elementwise.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_relu.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Relu.html)
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
            TensorType::F32(a) => {
                let mut a = a.clone();
                a.par_mapv_inplace(|v| v.max(0.0));
                Ok(TensorType::F32(a).into())
            }
            TensorType::I64(a) => {
                let mut a = a.clone();
                a.par_mapv_inplace(|v| v.max(0));
                Ok(TensorType::I64(a).into())
            }
            _ => Err(anyhow!("Relu: invalid input")),
        }
    }
}
