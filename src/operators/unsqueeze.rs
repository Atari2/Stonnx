#![allow(unused_variables)]
use crate::{
    onnx::NodeProto,
    utils::{pick_opset_version, ArrayType},
};

const OPSET_VERSIONS: [i64; 3] = [1, 11, 13];

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_unsqueeze.py
/// https://onnx.ai/onnx/operators/onnx__Unsqueeze.html
pub fn unsqueeze(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    todo!("Unsqueeze for opset version {}", target_version)
}
