use ndarray::Ix0;

use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_pow.py
/// https://onnx.ai/onnx/operators/onnx__Pow.html
pub fn pow(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let data = inputs[0].to_owned();
    let power = inputs[1].to_owned();
    
    let pow = match power {
        ArrayType::I64(power) => power.clone().into_dimensionality::<Ix0>()?.into_scalar() as f32,
        x => {
            todo!("Pow for type {}", x);
        }
    };

    match data {
        ArrayType::F32(x) => Ok(ArrayType::F32(x.mapv(|v| v.powf( pow ))).into()),
        ArrayType::I64(x) => Ok(ArrayType::F32(x.mapv(|v| v as f32).mapv(|v| v.powi(pow as i32))).into()),
        x => {
            todo!("Sqrt for type {}", x);
        }
        
    }

}