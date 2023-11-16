use crate::{
    onnx::NodeProto,
    utils::ArrayType,
};

const _OPSET_VERSIONS: [i64; 4] = [1, 6, 13, 14];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_relu.py
/// https://onnx.ai/onnx/operators/onnx__Relu.html
pub fn relu(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    if inputs.len() != 1 {
        Err("Relu must have 1 input".into())
    } else {
        match inputs[0] {
            ArrayType::F32(a) => Ok(ArrayType::F32(a.mapv(|v| v.max(0.0)))),
            ArrayType::I64(a) => Ok(ArrayType::I64(a.mapv(|v| v.max(0)))),
            _ => Err("Relu: invalid input".into()),
        }
    }
}
