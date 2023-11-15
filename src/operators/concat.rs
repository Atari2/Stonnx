#![allow(unused_variables, dead_code)] use ndarray::concatenate;

// TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{ArrayType, pick_opset_version},
};

const _OPSET_VERSIONS: [i64; 4] = [1, 4, 11, 13];

#[derive(Debug)]
struct ConcatAttrs {
    axis: i64,
}

impl ConcatAttrs {
    fn new(node: &NodeProto) -> Self {
        Self {
            axis: node
                .attribute
                .iter()
                .find(|a| a.name() == "axis")
                .map_or(0, |a| a.i.unwrap_or(1)),
        }
    }
}

fn concat_1(
    inputs: &[&ArrayType],
    attrs: ConcatAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Concat for type {}", inputs[0])
}

fn concat_11(
    inputs: &[&ArrayType],
    attrs: ConcatAttrs,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    todo!("Concat for type {}", inputs[0])
}


/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_concat.py
/// https://onnx.ai/onnx/operators/onnx__Concat.html
pub fn concat(
    inputs: &[&ArrayType],
    node: &NodeProto,
    _opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(_opset_version, &_OPSET_VERSIONS);
    if target_version < 11 {
        // 1, 2, 4
        concat_1(inputs, ConcatAttrs::new(node)) 
    }
    else {
        // 11, 13
        concat_11(inputs, ConcatAttrs::new(node))
    }
}
