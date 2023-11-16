use crate::{
    onnx::NodeProto,
    utils::ArrayType,
};

const _OPSET_VERSIONS: [i64; 2] = [1, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_lrn.py
/// https://onnx.ai/onnx/operators/onnx__LRN.html
pub fn lrn(
    inputs: &[&ArrayType],
    _node: &NodeProto,
    _opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("LRN")
}
