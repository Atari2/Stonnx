use crate::common::{ArrayType, BoxResult, OperationResult};
use crate::onnx::NodeProto;

const _OPSET_VERSIONS: [i64; 3] = [1, 6, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_exp.py
/// https://onnx.ai/onnx/operators/onnx__Exp.html
pub fn exp(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
    _output_len: usize,
) -> BoxResult<OperationResult> {
    let data = inputs[0].to_owned();

    match data {
        ArrayType::F32(x) => Ok(ArrayType::F32(x.mapv(|v| v.exp())).into()),
        x => {
            todo!("Exp for type {}", x);
        }
    }
}
