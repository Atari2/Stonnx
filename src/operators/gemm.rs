#![allow(unused_variables, dead_code)] // TODO: remove this when operator is implemented
use crate::{
    onnx::NodeProto,
    utils::{ArrayType, pick_opset_version},
};

const OPSET_VERSIONS: [i64; 6] = [1, 6, 7, 9, 11, 13];

#[derive(Debug)]
struct GemmAttrs {

}

impl GemmAttrs {
    fn new(node: &NodeProto) -> Self {
        todo!("GemmAttrs::new")
    }
}

/// https://github.com/onnx/onnx/blob/main/onnx/reference/ops/op_gemm.py
/// https://onnx.ai/onnx/operators/onnx__Gemm.html
pub fn gemm(
    inputs: &[&ArrayType],
    node: &NodeProto,
    opset_version: i64,
) -> Result<ArrayType, Box<dyn std::error::Error>> {
    let target_version = pick_opset_version(opset_version, &OPSET_VERSIONS);
    if target_version >= 7 {
        todo!("Gemm_7")
    } else {
        todo!("Gemm_6")
    }
}
