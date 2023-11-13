#![allow(unused_variables)]
/// TODO: delete this line after implement whole operators
use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType},
};

const OPSET_VERSION: [i64; 6] = [1, 9, 11, 12, 13, 19];

#[derive(Debug)]
struct ConstantAttrs {}
impl ConstantAttrs {
    fn new(node: &NodeProto) -> Self {
        todo!("ConstantAttrs::new")
    }
}

fn constant_1(
    inputs: &[&ArrayType],
    attrs: ConstantAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Constant_1")
}

fn constant_9(
    inputs: &[&ArrayType],
    attrs: ConstantAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Constant_9")
}

fn constant_11(
    inputs: &[&ArrayType],
    attrs: ConstantAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Constant_11")
}

fn constant_12(
    inputs: &[&ArrayType],
    attrs: ConstantAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Constant_12")
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_constant.py
/// https://onnx.ai/onnx/operators/onnx__Constant.html
pub fn constant(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSION);
    if target_version == 1 {
        constant_1(inputs, ConstantAttrs::new(node))
    } else if target_version == 9 {
        constant_9(inputs, ConstantAttrs::new(node))
    } else if target_version == 11 {
        constant_11(inputs, ConstantAttrs::new(node))
    } else {
        constant_12(inputs, ConstantAttrs::new(node))
    }
}
