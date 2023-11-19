use crate::onnx::NodeProto;
use crate::utils::{ArrayType, BoxResult, OperationResult};

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_sqrt.py
/// https://onnx.ai/onnx/operators/onnx__Sqrt.html
pub fn sqrt(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let data = inputs[0].to_owned();
    
    match data {
        ArrayType::F32(x) => Ok(ArrayType::F32(x.mapv(|v| v.sqrt())).into()),
        ArrayType::I64(x) => Ok(ArrayType::F32(x.mapv(|v| v as f32).mapv(|v| v.sqrt())).into()),
        x => {
            todo!("Sqrt for type {}", x);
        }
        
    }

}
