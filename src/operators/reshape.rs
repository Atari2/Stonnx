#![allow(unused_variables, dead_code)] // TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{ArrayType, pick_opset_version},
};

const OPSET_VERSIONS: [i64; 5] = [1, 5, 13, 14, 19];

#[derive(Debug)]
struct ReshapeAttrs {

}

impl ReshapeAttrs {
    fn new(node: &NodeProto) -> Self {
        todo!("ReshapeAttrs::new")
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_reshape.py
/// https://onnx.ai/onnx/operators/onnx__Reshape.html
pub fn reshape(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version >= 14 {
        todo!("Reshape_14")
    } else {
        todo!("Reshape_5")
    }
}
