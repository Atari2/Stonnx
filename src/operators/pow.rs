use ndarray::Ix0;

use crate::common::{BoxResult, OperatorResult, TensorType};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// Performs element-wise binary power (with limited broadcast support).
///
/// [Python reference](https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_pow.py)
///
/// [ONNX Documentation](https://onnx.ai/onnx/operators/onnx__Pow.html)
pub fn pow(
    inputs: &[&TensorType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperatorResult> {
    let data = inputs[0].to_owned();
    let power = inputs[1].to_owned();

    let pow = match power {
        TensorType::F32(power) => power.into_dimensionality::<Ix0>()?.into_scalar(),
        TensorType::I64(power) => power.into_dimensionality::<Ix0>()?.into_scalar() as f32,
        x => {
            todo!("Pow for type {}", x);
        }
    };

    match data {
        TensorType::F32(x) => {
            let mut x = x.clone();
            x.par_mapv_inplace(|v| v.powf(pow));
            Ok(TensorType::F32(x).into())
        }
        TensorType::I64(x) => {
            let mut x = x.clone();
            x.par_mapv_inplace(|v| v.pow(pow as u32));
            Ok(TensorType::I64(x).into())
        }
        x => {
            todo!("Pow for type {:?}", x);
        }
    }
}
