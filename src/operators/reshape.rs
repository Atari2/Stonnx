#![allow(unused_variables, dead_code)] // TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{ArrayType, pick_opset_version},
};

use super::shape;

const OPSET_VERSIONS: [i64; 5] = [1, 5, 13, 14, 19];

#[derive(Debug)]
struct ReshapeAttrs {
    allowzero: Option<i64>
}

impl ReshapeAttrs {
    fn new(node: &NodeProto) -> Self {
        Self { 
            allowzero:
                node
                .attribute
                .iter()
                .find(|a| a.name() == "allowzero")
                .and_then(|a| a.i), }
    }
}

fn reshape_5(
    inputs: &[&ArrayType],
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Reshape for type {}", inputs[0])
}

// NEED TO FIX THIS
fn reshape_14(
    inputs: &[&ArrayType],
    attrs: ReshapeAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Reshape for type {}", inputs[0])
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
        todo!("Reshape for type {}", inputs[0])
    } else {
        todo!("Reshape for type {}", inputs[0])
    }
}
