use crate::{
    common::{BoxResult, OperatorResult, TensorType},
    onnx::NodeProto,
};
use anyhow::anyhow;

const _OPSET_VERSIONS: [i64; 1] = [1];

/// GlobalAveragePool consumes an input tensor X and applies average pooling across the values in the same channel.
///
/// This is equivalent to AveragePool with kernel size equal to the spatial dimension of input tensor.
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_global_average_pool.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__GlobalAveragePool.html)
pub fn global_average_pool(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64, // defined but never used because even thought Conv has 2 versions they both do the same thing
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let input = inputs[0];
    match input {
        TensorType::F32(x) => {
            let axis = Vec::from_iter(2..x.ndim());
            let mut y = x
                .mean_axis(ndarray::Axis(axis[0]))
                .ok_or_else(|| anyhow!("Error in GlobalAveragePool"))?;
            for (i, ax) in axis.iter().skip(1).enumerate() {
                y = y
                    .mean_axis(ndarray::Axis(*ax - (i + 1)))
                    .ok_or_else(|| anyhow!("Error in GlobalAveragePool"))?;
            }
            for x in axis.iter() {
                y.insert_axis_inplace(ndarray::Axis(*x));
            }
            Ok(TensorType::F32(y).into())
        }
        x => {
            todo!("GlobalAveragePool for type {}", x);
        }
    }
}
