#![allow(unused_variables, dead_code)] // TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::ArrayType,
};

const _OPSET_VERSIONS: [i64; 4] = [1, 4, 11, 13];

#[derive(Debug)]
struct ConcatAttrs {

}

impl ConcatAttrs {
    fn new(node: &NodeProto) -> Self {
        todo!("ConcatAttrs::new")
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_concat.py
/// https://onnx.ai/onnx/operators/onnx__Concat.html
pub fn concat(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Concat")
}
